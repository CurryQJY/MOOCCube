import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from usim import Config, PAM_RL_Pure_USIM


class USIMPlusConfig(Config):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.plus_only_cold = os.environ.get("USIM_PLUS_ONLY_COLD", "1") == "1"
        self.plus_top_m = int(os.environ.get("USIM_PLUS_TOPM", "8"))
        self.plus_temp = float(os.environ.get("USIM_PLUS_TEMP", "0.20"))
        self.plus_scale = float(os.environ.get("USIM_PLUS_SCALE", "0.12"))
        self.plus_affinity_alpha = float(os.environ.get("USIM_PLUS_AFFINITY_ALPHA", "0.00"))
        self.plus_item_batch = int(os.environ.get("USIM_PLUS_ITEM_BATCH", "1024"))


def build_item_popularity(df, n_items):
    counts = torch.zeros(n_items, dtype=torch.float32)
    vc = df["i_idx"].value_counts()
    for item_idx, count in vc.items():
        idx = int(item_idx)
        if 0 <= idx < n_items:
            counts[idx] = float(count)
    return counts


class ColdResidualAdapterMixin:
    def _init_plus_adapter(self, config):
        self.plus_proto_delta = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
        )
        self.plus_proto_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.emb_dim),
            nn.GELU(),
            nn.Linear(config.emb_dim, 1),
            nn.Sigmoid(),
        )
        self.plus_proto_norm = nn.LayerNorm(config.emb_dim)
        self.plus_item_popularity = None
        self.plus_warm_item_mask = None
        self.plus_warm_item_idx = None
        self.plus_warm_pos_map = None
        self.plus_neighbor_affinity = None
        if not hasattr(self, "plus_global_llm_tensor"):
            self.register_buffer(
                "plus_global_llm_tensor",
                torch.full((config.n_items,), -1.0, dtype=torch.float32),
                persistent=False,
            )

    def set_global_llm_scores(self, llm_scores):
        scores = torch.full((self.cfg.n_items,), -1.0, dtype=torch.float32)
        for idx in range(self.cfg.n_items):
            scores[idx] = float(llm_scores.get(int(idx), -1.0))
        if hasattr(self, "global_llm_tensor"):
            self.global_llm_tensor = scores.to(self.device)
        else:
            self.plus_global_llm_tensor = scores.to(self.device)

    def _get_plus_llm_tensor(self):
        if hasattr(self, "global_llm_tensor"):
            return self.global_llm_tensor.to(self.device)
        return self.plus_global_llm_tensor.to(self.device)

    def set_plus_artifacts(self, item_popularity, neighbor_affinity=None):
        if item_popularity is None:
            self.plus_item_popularity = None
            self.plus_warm_item_mask = None
            self.plus_warm_item_idx = None
            self.plus_warm_pos_map = None
            self.plus_neighbor_affinity = None
            return

        pop = item_popularity.to(self.device).float()
        self.plus_item_popularity = pop
        self.plus_warm_item_mask = pop >= float(self.cfg.cold_threshold)
        self.plus_warm_item_idx = torch.nonzero(self.plus_warm_item_mask, as_tuple=False).view(-1)
        pos_map = torch.full((self.cfg.n_items,), -1, dtype=torch.long, device=self.device)
        if self.plus_warm_item_idx.numel() > 0:
            pos_map[self.plus_warm_item_idx] = torch.arange(self.plus_warm_item_idx.numel(), device=self.device)
        self.plus_warm_pos_map = pos_map

        if neighbor_affinity is None:
            self.plus_neighbor_affinity = None
        else:
            self.plus_neighbor_affinity = neighbor_affinity.to(self.device).float()

    def _plus_apply_mask(self, i_idx=None, target_pop=None, force_cold=False):
        if not self.cfg.plus_only_cold:
            if target_pop is not None:
                return torch.ones((target_pop.size(0), 1), dtype=torch.float32, device=self.device)
            if i_idx is not None:
                return torch.ones((i_idx.size(0), 1), dtype=torch.float32, device=self.device)

        if target_pop is not None:
            return (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()

        if i_idx is None or self.plus_item_popularity is None:
            fill = 1.0 if force_cold else 0.0
            size = 1 if i_idx is None else i_idx.size(0)
            return torch.full((size, 1), fill, dtype=torch.float32, device=self.device)

        return (self.plus_item_popularity[i_idx] < float(self.cfg.cold_threshold)).float().unsqueeze(1)

    def _compose_plus_item(self, id_e, content_e, proto_e, apply_mask):
        feat = torch.cat([content_e, proto_e, proto_e - content_e], dim=-1)
        gate = self.plus_proto_gate(feat)
        proto_delta = self.plus_proto_delta(feat)
        proto_content = self.plus_proto_norm(
            content_e + apply_mask * gate * float(self.cfg.plus_scale) * proto_delta
        )
        mixed_content = apply_mask * proto_content + (1.0 - apply_mask) * content_e
        alpha = self.gate_net(torch.cat([id_e, mixed_content], dim=-1))
        item_fused = alpha * id_e + (1.0 - alpha) * mixed_content
        return item_fused, mixed_content, gate

    def _build_plus_base_banks(self, force_cold=False, item_batch=1024):
        llm_tensor = self._get_plus_llm_tensor()
        base_items = []
        id_items = []
        content_items = []
        with torch.no_grad():
            for start in range(0, self.cfg.n_items, item_batch):
                end = min(start + item_batch, self.cfg.n_items)
                idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                llm_s = llm_tensor[idx]
                base_item, id_e, content_e = super().get_item_vector(idx, llm_s, force_cold=force_cold)
                base_items.append(base_item)
                id_items.append(id_e)
                content_items.append(content_e)
        return torch.cat(base_items, dim=0), torch.cat(id_items, dim=0), torch.cat(content_items, dim=0)

    def _compute_plus_proto_bank(self, content_bank, item_batch=1024):
        proto_bank = torch.zeros_like(content_bank)
        if self.plus_warm_item_idx is None or self.plus_warm_item_idx.numel() < 1:
            return proto_bank

        top_m = min(max(1, int(self.cfg.plus_top_m)), int(self.plus_warm_item_idx.numel()))
        temp = max(float(self.cfg.plus_temp), 1e-6)
        aff_alpha = float(self.cfg.plus_affinity_alpha)

        content_norm = F.normalize(content_bank, dim=1)
        warm_content = content_norm[self.plus_warm_item_idx]
        warm_id = F.normalize(self.item_id_emb(self.plus_warm_item_idx).detach(), dim=1)

        for start in range(0, self.cfg.n_items, item_batch):
            end = min(start + item_batch, self.cfg.n_items)
            idx = torch.arange(start, end, device=self.device, dtype=torch.long)
            scores = torch.matmul(content_norm[idx], warm_content.t())

            if aff_alpha > 0.0 and self.plus_neighbor_affinity is not None:
                scores = scores + aff_alpha * self.plus_neighbor_affinity[idx][:, self.plus_warm_item_idx]

            if self.plus_warm_pos_map is not None:
                self_pos = self.plus_warm_pos_map[idx]
                valid_rows = self_pos >= 0
                if valid_rows.any():
                    scores[valid_rows, self_pos[valid_rows]] = -1e9

            top_scores, top_pos = torch.topk(scores, k=top_m, dim=1)
            weights = F.softmax(top_scores / temp, dim=1)
            proto_chunk = (warm_id[top_pos] * weights.unsqueeze(-1)).sum(dim=1)
            proto_bank[idx] = F.normalize(proto_chunk, dim=1)

        return proto_bank

    def build_item_bank_plus(self, force_cold=False, item_batch=1024, deterministic=False):
        was_training = self.training
        if deterministic and was_training:
            self.eval()
        try:
            with torch.no_grad():
                _, id_bank, content_bank = self._build_plus_base_banks(force_cold=force_cold, item_batch=item_batch)
                proto_bank = self._compute_plus_proto_bank(content_bank, item_batch=item_batch)
                all_vecs = []
                for start in range(0, self.cfg.n_items, item_batch):
                    end = min(start + item_batch, self.cfg.n_items)
                    idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                    apply_mask = self._plus_apply_mask(idx, force_cold=force_cold)
                    fused_item, _, _ = self._compose_plus_item(
                        id_bank[idx],
                        content_bank[idx],
                        proto_bank[idx],
                        apply_mask,
                    )
                    all_vecs.append(F.normalize(fused_item, dim=1))
                return torch.cat(all_vecs, dim=0)
        finally:
            if deterministic and was_training:
                self.train()

    def _compute_plus_proto_for_items(self, i_idx, content_e):
        proto_e = torch.zeros_like(content_e)
        if self.plus_warm_item_idx is None or self.plus_warm_item_idx.numel() < 1:
            return proto_e

        llm_tensor = self._get_plus_llm_tensor()
        with torch.no_grad():
            warm_idx = self.plus_warm_item_idx
            warm_llm = llm_tensor[warm_idx]
            _, _, warm_content = super().get_item_vector(warm_idx, warm_llm, force_cold=False)
            warm_content = F.normalize(warm_content, dim=1)
            warm_id = F.normalize(self.item_id_emb(warm_idx).detach(), dim=1)

            query = F.normalize(content_e.detach(), dim=1)
            scores = torch.matmul(query, warm_content.t())
            aff_alpha = float(self.cfg.plus_affinity_alpha)
            if aff_alpha > 0.0 and self.plus_neighbor_affinity is not None:
                scores = scores + aff_alpha * self.plus_neighbor_affinity[i_idx][:, warm_idx]

            if self.plus_warm_pos_map is not None:
                self_pos = self.plus_warm_pos_map[i_idx]
                valid_rows = self_pos >= 0
                if valid_rows.any():
                    scores[valid_rows, self_pos[valid_rows]] = -1e9

            top_m = min(max(1, int(self.cfg.plus_top_m)), int(warm_idx.numel()))
            temp = max(float(self.cfg.plus_temp), 1e-6)
            top_scores, top_pos = torch.topk(scores, k=top_m, dim=1)
            weights = F.softmax(top_scores / temp, dim=1)
            proto_e = (warm_id[top_pos] * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(proto_e, dim=1)

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        item_fused, id_e, content_e = super().get_item_vector(i_idx, llm_s, force_cold=force_cold)
        if (
            self.plus_item_popularity is None or
            self.plus_warm_item_idx is None or
            self.plus_warm_item_idx.numel() < 1 or
            float(self.cfg.plus_scale) <= 0.0
        ):
            return item_fused, id_e, content_e

        proto_e = self._compute_plus_proto_for_items(i_idx, content_e)
        apply_mask = self._plus_apply_mask(i_idx=i_idx, force_cold=force_cold)
        plus_item, plus_content, _ = self._compose_plus_item(id_e, content_e, proto_e, apply_mask)
        return plus_item, id_e, plus_content


class USIMPlus(ColdResidualAdapterMixin, PAM_RL_Pure_USIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self._init_plus_adapter(config)
