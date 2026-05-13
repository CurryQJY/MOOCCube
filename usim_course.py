import copy
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from hhcor_static_hin import build_history_tensor, _update_histories_from_df
from usim_course_base import (
    PAM_RL_Pure_USIM,
    _add_user_seen_from_df,
    _build_llm_score_tensor,
    _clone_user_seen,
    _lookup_llm_score,
    build_course_artifacts,
    compute_ranking_metrics,
    setup_seed,
    split_dataframe_by_periods,
)


class CourseConfig:
    def __init__(self, n_users, n_items, content_dim=768):
        from usim_course_base import Config as BaseConfig

        base = BaseConfig(n_users, n_items, content_dim)
        self.__dict__.update(base.__dict__)
        self.n_epochs = int(os.environ.get("USIM_N_EPOCHS", "3"))

        self.use_course_rerank = False
        self.use_structured_hard_neg = os.environ.get("USIM_STRUCTURED_HARD_NEG", "0") == "1"

        self.course_min_seen = int(os.environ.get("USIM_COURSE_MIN_SEEN", "3"))
        self.course_hist_warm_seen = int(os.environ.get("USIM_COURSE_HIST_WARM_SEEN", "5"))
        self.course_score_warm_seen = int(os.environ.get("USIM_COURSE_SCORE_WARM_SEEN", "4"))
        self.course_score_alpha = float(os.environ.get("USIM_COURSE_SCORE_ALPHA", "0.05"))
        self.course_score_lambda = float(os.environ.get("USIM_COURSE_SCORE_LAMBDA", "0.02"))
        self.course_penalty_min_seen = int(os.environ.get("USIM_COURSE_PENALTY_MIN_SEEN", "5"))
        self.course_penalty_cap = float(os.environ.get("USIM_COURSE_PENALTY_CAP", "0.10"))
        self.course_score_only_cold = os.environ.get("USIM_COURSE_SCORE_ONLY_COLD", "1") == "1"
        self.course_score_train = os.environ.get("USIM_COURSE_SCORE_TRAIN", "0") == "1"
        self.course_score_top_l = int(os.environ.get("USIM_COURSE_SCORE_TOPL", "20"))
        self.course_bonus_cap = float(os.environ.get("USIM_COURSE_BONUS_CAP", "0.06"))
        self.course_concept_min = float(os.environ.get("USIM_COURSE_CONCEPT_MIN", "0.12"))
        self.course_prereq_soft_thr = float(os.environ.get("USIM_COURSE_PREREQ_SOFT_THR", "0.20"))
        self.course_prereq_strict_thr = float(os.environ.get("USIM_COURSE_PREREQ_STRICT_THR", "0.60"))
        self.course_score_conf_floor = float(os.environ.get("USIM_COURSE_SCORE_CONF_FLOOR", "0.10"))
        self.cold_loss_weight = float(os.environ.get("USIM_COLD_LOSS_WEIGHT", "1.0"))
        self.use_cold_balanced_sampler = os.environ.get("USIM_USE_COLD_SAMPLER", "0") == "1"
        self.train_cold_ratio = float(os.environ.get("USIM_TRAIN_COLD_RATIO", "0.25"))
        self.stream_train_window = int(os.environ.get("USIM_STREAM_TRAIN_WINDOW", "0"))
        self.use_stream_cold_replay = os.environ.get("USIM_USE_STREAM_COLD_REPLAY", "0") == "1"
        self.stream_cold_replay_ratio = float(os.environ.get("USIM_STREAM_COLD_REPLAY_RATIO", "0.10"))
        self.stream_cold_replay_cap = int(os.environ.get("USIM_STREAM_COLD_REPLAY_CAP", "8192"))
        self.course_hist_force_cold = os.environ.get("USIM_COURSE_HIST_FORCE_COLD", "0") == "1"
        self.course_hist_deterministic = os.environ.get("USIM_COURSE_HIST_DETERMINISTIC", "1") == "1"
        self.course_concept_recent_k = int(os.environ.get("USIM_COURSE_CONCEPT_RECENT_K", "5"))
        self.use_cold_item_adapter = os.environ.get("USIM_USE_COLD_ITEM_ADAPTER", "0") == "1"
        self.cold_item_adapter_beta = float(os.environ.get("USIM_COLD_ITEM_ADAPTER_BETA", "0.20"))
        self.cold_item_concept_topk = int(os.environ.get("USIM_COLD_ITEM_CONCEPT_TOPK", "10"))
        self.cold_item_very_cold_thr = int(os.environ.get("USIM_COLD_ITEM_VERY_COLD_THR", "2"))
        self.cold_item_beta_very = float(os.environ.get("USIM_COLD_ITEM_BETA_VERY", "0.35"))
        self.cold_item_beta_mild = float(os.environ.get("USIM_COLD_ITEM_BETA_MILD", "0.15"))
        self.course_hist_len = int(os.environ.get("USIM_COURSE_HIST_LEN", "20"))
        self.course_hist_heads = int(os.environ.get("USIM_COURSE_HIST_HEADS", "2"))
        self.course_hist_layers = int(os.environ.get("USIM_COURSE_HIST_LAYERS", "1"))
        self.course_hist_dropout = float(os.environ.get("USIM_COURSE_HIST_DROPOUT", "0.10"))


class CourseSeqDataset(Dataset):
    def __init__(self, df, llm_scores, history_tensor):
        user_ids = [int(x) for x in df["u_idx"].values]
        item_ids = [int(x) for x in df["i_idx"].values]
        self.u = torch.tensor(user_ids, dtype=torch.long)
        self.i = torch.tensor(item_ids, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)
        self.llm_s = _build_llm_score_tensor(llm_scores, user_ids, item_ids).to(torch.float32)
        self.hist = history_tensor

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            "u": self.u[idx],
            "i": self.i[idx],
            "hist": self.hist[idx],
            "pop": self.pop[idx],
            "llm": self.llm_s[idx],
        }


def collate_course(batch):
    u = torch.stack([item["u"] for item in batch])
    i = torch.stack([item["i"] for item in batch])
    hist = torch.stack([item["hist"] for item in batch])
    pop = torch.stack([item["pop"] for item in batch])
    llm = torch.stack([item["llm"] for item in batch])
    return {"u": u, "i": i, "hist": hist}, pop, llm


class CourseAwareUSIM(PAM_RL_Pure_USIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.pos_emb = nn.Embedding(config.course_hist_len, config.emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim,
            nhead=config.course_hist_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.course_hist_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.course_hist_layers)
        self.course_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid(),
        )
        self.course_fuse = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.GELU(),
            nn.LayerNorm(config.emb_dim),
        )
        self.item_graph_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid(),
        )
        self.item_graph_fuse = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.GELU(),
            nn.LayerNorm(config.emb_dim),
        )
        self.register_buffer(
            "global_llm_tensor",
            torch.full((config.n_items,), -1.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "item_is_cold",
            torch.zeros(config.n_items, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "item_popularity",
            torch.zeros(config.n_items, dtype=torch.long),
            persistent=False,
        )

    def set_global_llm_scores(self, llm_scores):
        scores = torch.full((self.cfg.n_items,), -1.0, dtype=torch.float32)
        for idx in range(self.cfg.n_items):
            scores[idx] = float(_lookup_llm_score(llm_scores, idx))
        self.global_llm_tensor = scores.to(self.device)

    def set_item_cold_mask(self, item_is_cold):
        mask = torch.as_tensor(item_is_cold, dtype=torch.bool)
        if mask.numel() != self.cfg.n_items:
            raise ValueError(f"item_is_cold size mismatch: expect {self.cfg.n_items}, got {mask.numel()}")
        self.item_is_cold = mask.to(self.device)

    def set_item_popularity(self, item_popularity):
        pop = torch.as_tensor(item_popularity, dtype=torch.long)
        if pop.numel() != self.cfg.n_items:
            raise ValueError(f"item_popularity size mismatch: expect {self.cfg.n_items}, got {pop.numel()}")
        self.item_popularity = pop.to(self.device)

    def build_course_item_bank(self, force_cold=True, item_batch=1024, deterministic=False):
        all_vecs = []
        was_training = self.training
        if deterministic and was_training:
            self.eval()
        try:
            with torch.no_grad():
                for start in range(0, self.cfg.n_items, item_batch):
                    end = min(start + item_batch, self.cfg.n_items)
                    idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                    llm_s = self.global_llm_tensor[idx]
                    item_vec, _, _ = self.get_item_vector(idx, llm_s, force_cold=force_cold)
                    all_vecs.append(F.normalize(item_vec, dim=1))
        finally:
            if deterministic and was_training:
                self.train()
        return torch.cat(all_vecs, dim=0)

    def build_concept_graph_bank(self, item_bank):
        if self.item_concept_overlap is None:
            zeros = torch.zeros_like(item_bank)
            valid = torch.zeros((item_bank.size(0), 1), dtype=torch.float32, device=item_bank.device)
            return zeros, valid

        adj = self.item_concept_overlap.clone()
        adj.fill_diagonal_(0.0)

        topk = int(self.cfg.cold_item_concept_topk)
        if topk > 0 and topk < adj.size(1):
            top_vals, top_idx = torch.topk(adj, k=topk, dim=1)
            top_mask = torch.zeros_like(adj, dtype=torch.bool)
            top_mask.scatter_(1, top_idx, top_vals > 0)
            adj = adj.masked_fill(~top_mask, 0.0)

        row_sum = adj.sum(dim=1, keepdim=True)
        norm_adj = torch.where(
            row_sum > 0,
            adj / row_sum.clamp_min(1e-12),
            torch.zeros_like(adj),
        )
        concept_bank = torch.matmul(norm_adj, item_bank)
        concept_bank = F.normalize(concept_bank, dim=1)
        valid = (row_sum > 0).float()
        confidence = (adj.max(dim=1).values).unsqueeze(1).clamp(0.0, 1.0)
        return concept_bank, valid, confidence

    def enhance_cold_item(self, item_vec, item_idx, cold_mask, concept_bank, concept_valid, concept_conf):
        if (
            not self.cfg.use_cold_item_adapter or
            concept_bank is None or
            concept_valid is None or
            concept_conf is None or
            item_vec.size(0) < 1
        ):
            return item_vec

        concept_vec = concept_bank[item_idx]
        concept_ok = concept_valid[item_idx]
        cold_active = cold_mask.view(-1, 1).float() * concept_ok
        if cold_active.max().item() <= 0:
            return item_vec

        item_norm = F.normalize(item_vec, dim=1)
        gate = self.item_graph_gate(torch.cat([item_norm, concept_vec], dim=1))
        mixed = gate * item_norm + (1.0 - gate) * concept_vec
        fused = self.item_graph_fuse(torch.cat([mixed, concept_vec], dim=1))
        item_pop = self.item_popularity[item_idx]
        very_mask = (item_pop <= int(self.cfg.cold_item_very_cold_thr)).float().view(-1, 1)
        mild_mask = 1.0 - very_mask
        beta_very = float(min(1.0, max(0.0, self.cfg.cold_item_beta_very)))
        beta_mild = float(min(1.0, max(0.0, self.cfg.cold_item_beta_mild)))
        beta_base = very_mask * beta_very + mild_mask * beta_mild
        beta_global = float(min(1.0, max(0.0, self.cfg.cold_item_adapter_beta)))
        beta_eff = beta_global * beta_base * concept_conf[item_idx]
        adapted = (1.0 - beta_eff) * item_vec + beta_eff * fused
        return cold_active * adapted + (1.0 - cold_active) * item_vec
        adapted = (1.0 - beta_eff) * item_vec + beta_eff * fused
        return cold_active * adapted + (1.0 - cold_active) * item_vec

    def _build_seen_mat(self, user_ids, user_seen_items, exclude_items=None):
        batch_size = len(user_ids)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if user_seen_items is None:
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)

        exclude_list = None
        if exclude_items is not None:
            exclude_list = [int(x) for x in exclude_items.detach().cpu().tolist()]

        for row, uid in enumerate(user_ids):
            seen_items = user_seen_items.get(int(uid))
            if not seen_items:
                continue
            tgt = exclude_list[row] if exclude_list is not None else None
            seen_idx = [it for it in seen_items if 0 <= it < self.cfg.n_items and it != tgt]
            if seen_idx:
                idx_t = torch.tensor(seen_idx, dtype=torch.long, device=self.device)
                seen_mat[row, idx_t] = 1.0
        return seen_mat, seen_mat.sum(dim=1, keepdim=True)

    def _hist_to_seen_mat(self, hist_idx):
        batch_size = hist_idx.size(0)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        for row in range(batch_size):
            valid = hist_idx[row][(hist_idx[row] >= 0) & (hist_idx[row] < self.cfg.n_items)]
            if valid.numel() > 0:
                seen_mat[row, valid.unique()] = 1.0
        return seen_mat, seen_mat.sum(dim=1, keepdim=True)

    def _hist_to_recent_seen_mat(self, hist_idx, recent_k):
        batch_size = hist_idx.size(0)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if recent_k <= 0:
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)

        for row in range(batch_size):
            valid = hist_idx[row][(hist_idx[row] >= 0) & (hist_idx[row] < self.cfg.n_items)]
            if valid.numel() > 0:
                tail = valid[-recent_k:]
                seen_mat[row, tail.unique()] = 1.0
        return seen_mat, seen_mat.sum(dim=1, keepdim=True)

    def _seen_weight(self, seen_cnt_raw, warm_seen):
        warm_seen = max(int(warm_seen), int(self.cfg.course_min_seen), 1)
        active = (seen_cnt_raw >= float(self.cfg.course_min_seen)).float()
        weight = (seen_cnt_raw / float(warm_seen)).clamp(0.0, 1.0)
        return active * weight

    def _encode_history_sequence(self, hist_idx, item_bank):
        device = item_bank.device
        batch_size, seq_len = hist_idx.shape
        emb_dim = item_bank.size(1)

        mask = hist_idx >= 0
        pad_vec = torch.zeros(1, emb_dim, device=device)
        ext_bank = torch.cat([item_bank, pad_vec], dim=0)
        safe_idx = hist_idx.clone()
        safe_idx[~mask] = self.cfg.n_items
        hist_emb = ext_bank[safe_idx]

        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hist_emb = hist_emb + self.pos_emb(pos_idx)

        key_padding_mask = ~mask
        fully_padded = key_padding_mask.all(dim=1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False
            hist_emb = hist_emb.clone()
            hist_emb[fully_padded, 0, :] = 0.0

        seq_out = self.seq_encoder(hist_emb, src_key_padding_mask=key_padding_mask)
        lengths = mask.sum(dim=1)
        last_pos = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, emb_dim)
        seq_vec = seq_out.gather(1, last_pos).squeeze(1)
        seq_vec = seq_vec.clone()
        seq_vec[lengths == 0] = 0.0
        return seq_vec, lengths.unsqueeze(1).float()

    def encode_course_user(self, u_idx, hist_idx, item_bank):
        u_base = self.user_proj(self.user_emb(u_idx))
        hist_vec, hist_cnt_raw = self._encode_history_sequence(hist_idx, item_bank)
        gate = self.course_gate(torch.cat([u_base, hist_vec], dim=1))
        mixed = gate * u_base + (1.0 - gate) * hist_vec
        fused = self.course_fuse(torch.cat([mixed, hist_vec], dim=1))

        hist_weight = self._seen_weight(hist_cnt_raw, self.cfg.course_hist_warm_seen)
        user_vec = hist_weight * fused + (1.0 - hist_weight) * u_base
        return F.normalize(user_vec, dim=1), hist_vec, hist_cnt_raw

    def compute_course_score_adjust(
        self,
        user_ids,
        user_seen_items,
        candidate_idx,
        target_pop=None,
        exclude_items=None,
        seen_mat=None,
        seen_cnt_raw=None,
        concept_seen_mat=None,
        concept_seen_cnt_raw=None,
    ):
        batch_size = len(user_ids)
        if (
            batch_size < 1 or
            self.item_prereq_item_mat is None or
            self.item_prereq_item_cnt is None or
            self.item_concept_overlap is None
        ):
            return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)

        if seen_mat is None or seen_cnt_raw is None:
            if user_seen_items is None:
                return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)
            seen_mat, seen_cnt_raw = self._build_seen_mat(user_ids, user_seen_items, exclude_items=exclude_items)
        if seen_cnt_raw.max().item() < 1:
            return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)
        if concept_seen_mat is None or concept_seen_cnt_raw is None:
            concept_seen_mat = seen_mat
            concept_seen_cnt_raw = seen_cnt_raw

        prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
        prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
        violation_full = torch.where(
            prereq_cnt > 0,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)

        concept_full = torch.matmul(concept_seen_mat, self.item_concept_overlap.t()) / concept_seen_cnt_raw.clamp_min(1.0)
        cand_violation = violation_full.gather(1, candidate_idx)
        cand_concept = concept_full.gather(1, candidate_idx)

        concept_active = self._seen_weight(concept_seen_cnt_raw, self.cfg.course_score_warm_seen)
        penalty_active = (seen_cnt_raw >= float(self.cfg.course_penalty_min_seen)).float()
        if self.cfg.course_score_only_cold and target_pop is not None:
            cold_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
            concept_active = concept_active * cold_mask
            penalty_active = penalty_active * cold_mask

        safe_thr = float(min(0.95, max(0.0, self.cfg.course_prereq_soft_thr)))
        strict_thr = float(min(1.0, max(safe_thr, self.cfg.course_prereq_strict_thr)))
        concept_min = float(min(0.95, max(0.0, self.cfg.course_concept_min)))
        conf_floor = float(min(1.0, max(0.0, self.cfg.course_score_conf_floor)))

        penalty_gap = (cand_violation - safe_thr).clamp(min=0.0)
        penalty_scale = 1.0 + (cand_violation >= strict_thr).float()
        penalty = (self.cfg.course_score_lambda * penalty_gap * penalty_scale * penalty_active).clamp(
            min=0.0,
            max=float(self.cfg.course_penalty_cap),
        )

        concept_conf = ((cand_concept - concept_min) / max(1e-6, 1.0 - concept_min)).clamp(0.0, 1.0)
        prereq_safe = (cand_violation <= safe_thr).float()
        bonus_conf = concept_conf * concept_active * prereq_safe
        bonus_conf = torch.where(
            bonus_conf >= conf_floor,
            bonus_conf,
            torch.zeros_like(bonus_conf),
        )
        bonus = (self.cfg.course_score_alpha * bonus_conf).clamp(
            min=0.0,
            max=float(self.cfg.course_bonus_cap),
        )
        return bonus - penalty

    def maybe_add_course_score(
        self,
        base_scores,
        user_ids,
        candidate_idx,
        user_seen_items=None,
        target_pop=None,
        exclude_items=None,
        seen_mat=None,
        seen_cnt_raw=None,
        concept_seen_mat=None,
        concept_seen_cnt_raw=None,
        training=False,
    ):
        if training and not self.cfg.course_score_train:
            return base_scores

        top_l = int(self.cfg.course_score_top_l)
        if top_l <= 0 or top_l >= candidate_idx.size(1):
            adjust = self.compute_course_score_adjust(
                user_ids,
                user_seen_items,
                candidate_idx,
                target_pop=target_pop,
                exclude_items=exclude_items,
                seen_mat=seen_mat,
                seen_cnt_raw=seen_cnt_raw,
                concept_seen_mat=concept_seen_mat,
                concept_seen_cnt_raw=concept_seen_cnt_raw,
            )
            return base_scores + adjust

        k = min(top_l, candidate_idx.size(1))
        top_pos = torch.topk(base_scores, k=k, dim=1).indices
        top_items = candidate_idx.gather(1, top_pos)
        top_adjust = self.compute_course_score_adjust(
            user_ids,
            user_seen_items,
            top_items,
            target_pop=target_pop,
            exclude_items=exclude_items,
            seen_mat=seen_mat,
            seen_cnt_raw=seen_cnt_raw,
            concept_seen_mat=concept_seen_mat,
            concept_seen_cnt_raw=concept_seen_cnt_raw,
        )
        delta = torch.zeros_like(base_scores)
        delta.scatter_(1, top_pos, top_adjust)
        return base_scores + delta

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        item_bank = self.build_course_item_bank(
            force_cold=self.cfg.course_hist_force_cold,
            deterministic=self.cfg.course_hist_deterministic,
        )
        z_u_base, _, _ = self.encode_course_user(
            u,
            hist,
            item_bank,
        )
        concept_bank, concept_valid, concept_conf = self.build_concept_graph_bank(item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        if self.cfg.course_concept_recent_k > 0:
            concept_seen_mat, concept_seen_cnt = self._hist_to_recent_seen_mat(hist, self.cfg.course_concept_recent_k)
        else:
            concept_seen_mat, concept_seen_cnt = hist_seen_mat, hist_seen_cnt
        z_i_base, id_e_raw, content_e = self.get_item_vector(i, llm_s, force_cold=is_cold)
        z_i_base = self.enhance_cold_item(z_i_base, i, is_cold, concept_bank, concept_valid, concept_conf)

        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp

        cand_idx_full = i.view(1, -1).expand(logits.size(0), -1)
        logits = self.maybe_add_course_score(
            logits,
            user_ids,
            cand_idx_full,
            user_seen_items=user_seen_items,
            target_pop=pop,
            exclude_items=i,
            seen_mat=hist_seen_mat,
            seen_cnt_raw=hist_seen_cnt,
            concept_seen_mat=concept_seen_mat,
            concept_seen_cnt_raw=concept_seen_cnt,
            training=self.training,
        )

        labels = torch.arange(logits.size(0), device=self.device)
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp
        row_weights = torch.where(
            is_cold,
            torch.full_like(pop, float(self.cfg.cold_loss_weight), dtype=torch.float32),
            torch.ones_like(pop, dtype=torch.float32),
        )

        if self.training and self.cfg.use_mixed_hard_neg and logits_margin.size(0) > 1:
            batch_size = logits_margin.size(0)
            max_neg = batch_size - 1
            n_total_neg = min(self.cfg.train_num_negs, max_neg)

            if n_total_neg > 0:
                n_hard = int(n_total_neg * self.cfg.hard_neg_ratio)
                n_hard = max(0, min(n_hard, n_total_neg))
                n_rand = n_total_neg - n_hard

                neg_logits = logits_margin.clone()
                neg_logits[pos_mask] = -1e9

                hard_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                rand_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

                if n_hard > 0:
                    if self.cfg.use_structured_hard_neg and self.item_hard_adj is not None:
                        hard_mask = self.item_hard_adj[i][:, i]
                        hard_mask = hard_mask & (~pos_mask)
                        hard_logits = neg_logits.masked_fill(~hard_mask, -1e9)
                        hard_scores, hard_idx = torch.topk(hard_logits, k=n_hard, dim=1)
                        valid_mask = hard_scores > -1e8
                        if (~valid_mask).any():
                            bad_rows = torch.nonzero((~valid_mask).any(dim=1), as_tuple=False).view(-1).tolist()
                            for row in bad_rows:
                                need = int((~valid_mask[row]).sum().item())
                                if need < 1:
                                    continue
                                fallback = neg_logits[row].clone()
                                if valid_mask[row].any():
                                    fallback[hard_idx[row, valid_mask[row]]] = -1e9
                                _, fill_idx = torch.topk(fallback, k=need, dim=0)
                                hard_idx[row, ~valid_mask[row]] = fill_idx
                    else:
                        _, hard_idx = torch.topk(neg_logits, k=n_hard, dim=1)

                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(rand_scores, k=n_rand, dim=1)

                cand_idx = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                per_row_loss = F.cross_entropy(cand_logits, main_targets, reduction="none")
                main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)
            else:
                per_row_loss = F.cross_entropy(logits_margin, labels, reduction="none")
                main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)
        else:
            per_row_loss = F.cross_entropy(logits_margin, labels, reduction="none")
            main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)

        z_id = F.normalize(id_e_raw, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            seen_mat = hist_seen_mat
            seen_cnt = hist_seen_cnt.squeeze(1)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(seen_mat, prereq_mat_batch.t())
            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0 - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch),
            ).clamp(0.0, 1.0)

            valid_rows = seen_cnt >= float(self.cfg.prereq_aux_min_seen)
            if self.cfg.prereq_aux_only_cold:
                valid_rows = valid_rows & is_cold

            unmet_mask = violation_batch > float(self.cfg.prereq_aux_violation_thr)
            unmet_mask = unmet_mask & (~pos_mask)
            candidate_mask = unmet_mask & valid_rows.unsqueeze(1)

            if candidate_mask.any():
                neg_scores = logits.masked_fill(~candidate_mask, -1e9)
                neg_vals, _ = neg_scores.max(dim=1)
                has_neg = neg_vals > -1e8
                if has_neg.any():
                    pos_vals = logits[torch.arange(logits.size(0), device=self.device), labels]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss
        )
        return total_loss, candidate_stats


def build_all_item_vecs_course(model, item_batch=1024, force_cold=True, cold_mask=None):
    item_bank = model.build_course_item_bank(force_cold=force_cold, item_batch=item_batch)
    concept_bank, concept_valid, concept_conf = model.build_concept_graph_bank(item_bank)
    all_idx = torch.arange(model.cfg.n_items, device=model.device, dtype=torch.long)
    if cold_mask is None:
        fill = bool(force_cold)
        cold_mask = torch.full((model.cfg.n_items,), fill, dtype=torch.bool, device=model.device)
    else:
        cold_mask = torch.as_tensor(cold_mask, dtype=torch.bool, device=model.device)
    all_item_vecs = model.enhance_cold_item(item_bank, all_idx, cold_mask, concept_bank, concept_valid, concept_conf)
    return F.normalize(all_item_vecs, dim=1)


def build_eval_item_vecs_course(model, item_batch=1024):
    hot_mask = torch.zeros(model.cfg.n_items, dtype=torch.bool, device=model.device)
    cold_mask = torch.ones(model.cfg.n_items, dtype=torch.bool, device=model.device)
    hot_bank = build_all_item_vecs_course(
        model, item_batch=item_batch, force_cold=False, cold_mask=hot_mask
    )
    cold_bank = build_all_item_vecs_course(
        model, item_batch=item_batch, force_cold=True, cold_mask=cold_mask
    )
    return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}


def _select_course_eval_item_bank(all_item_vecs, eval_type):
    if isinstance(all_item_vecs, dict):
        if eval_type in all_item_vecs:
            return all_item_vecs[eval_type]
        if eval_type == "all" and "hot" in all_item_vecs:
            return all_item_vecs["hot"]
        if "cold" in all_item_vecs:
            return all_item_vecs["cold"]
    return all_item_vecs


def _build_course_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type, item_bank):
    if item_idx.numel() < 1:
        return torch.empty((0, model.cfg.emb_dim), device=item_idx.device)

    concept_bank, concept_valid, concept_conf = model.build_concept_graph_bank(item_bank)

    def _encode(mask, force_cold):
        vec, _, _ = model.get_item_vector(item_idx[mask], llm_s[mask], force_cold=force_cold)
        cold_mask = torch.full((int(mask.sum().item()),), bool(force_cold), dtype=torch.bool, device=item_idx.device)
        vec = model.enhance_cold_item(
            vec,
            item_idx[mask],
            cold_mask,
            concept_bank,
            concept_valid,
            concept_conf,
        )
        return vec

    if eval_type == "cold":
        pos_vec = _encode(torch.ones_like(item_idx, dtype=torch.bool), True)
        return F.normalize(pos_vec, dim=1)

    if eval_type == "hot":
        pos_vec = _encode(torch.ones_like(item_idx, dtype=torch.bool), False)
        return F.normalize(pos_vec, dim=1)

    pos_vec = torch.empty((item_idx.size(0), model.cfg.emb_dim), device=item_idx.device)
    cold_mask = pop_sel < model.cfg.cold_threshold
    hot_mask = ~cold_mask
    if cold_mask.any():
        pos_vec[cold_mask] = _encode(cold_mask, True)
    if hot_mask.any():
        pos_vec[hot_mask] = _encode(hot_mask, False)
    return F.normalize(pos_vec, dim=1)


def evaluate_course_usim(
    model,
    loader,
    device,
    k_list=(5, 10, 20),
    n_neg=200,
    eval_type="cold",
    full_ranking=False,
    user_seen_items=None,
    all_item_vecs=None,
):
    model.eval()
    accum_metrics = {}
    total_samples = 0
    seen_tensor_cache = {}

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device, dtype=torch.long)
        if all_item_vecs is None:
            all_item_vecs = build_eval_item_vecs_course(model)
        item_bank = _select_course_eval_item_bank(all_item_vecs, eval_type)
        hist_item_vecs = model.build_course_item_bank(
            force_cold=model.cfg.course_hist_force_cold,
            deterministic=model.cfg.course_hist_deterministic,
        )

        for batch, pop, llm in loader:
            if eval_type == "cold":
                mask = pop < model.cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = int(mask.sum().item())
            if n_sel < 1:
                continue

            u = batch["u"][mask].to(device)
            i = batch["i"][mask].to(device)
            hist = batch["hist"][mask].to(device)
            pop_sel = pop[mask].to(device)
            llm_sel = llm[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

            for uid in user_ids:
                if uid in seen_tensor_cache:
                    continue
                seen_items = user_seen_items.get(uid) if user_seen_items else None
                if seen_items:
                    seen_list = [it for it in seen_items if 0 <= it < n_items]
                    seen_tensor_cache[uid] = (
                        torch.tensor(seen_list, dtype=torch.long, device=device)
                        if seen_list else None
                    )
                else:
                    seen_tensor_cache[uid] = None

            z_u, _, _ = model.encode_course_user(
                u,
                hist,
                hist_item_vecs,
            )
            hist_seen_mat, hist_seen_cnt = model._hist_to_seen_mat(hist)
            if model.cfg.course_concept_recent_k > 0:
                concept_seen_mat, concept_seen_cnt = model._hist_to_recent_seen_mat(
                    hist,
                    model.cfg.course_concept_recent_k,
                )
            else:
                concept_seen_mat, concept_seen_cnt = hist_seen_mat, hist_seen_cnt

            pos_vec = _build_course_eval_pos_item_vecs(model, i, llm_sel, pop_sel, eval_type, item_bank)
            pos_scores = (z_u * pos_vec).sum(dim=1)

            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                row_idx = torch.arange(n_sel, device=device)
                target_scores = pos_scores.clone()
                if user_seen_items:
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache.get(uid)
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                else:
                    scores[row_idx, i] = target_scores

                cand_idx = all_item_idx.view(1, -1).expand(n_sel, -1)
                scores = model.maybe_add_course_score(
                    scores,
                    user_ids,
                    cand_idx,
                    user_seen_items=user_seen_items,
                    target_pop=pop_sel,
                    exclude_items=i,
                    seen_mat=hist_seen_mat,
                    seen_cnt_raw=hist_seen_cnt,
                    concept_seen_mat=concept_seen_mat,
                    concept_seen_cnt_raw=concept_seen_cnt,
                )
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache.get(uid)
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        avail = n_items - 1 - int((seen_idx != i[row]).sum().item())
                    avail_counts.append(max(1, avail))

                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache.get(uid)
                    if seen_idx is not None and seen_idx.numel() > 0:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]

                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                perm = torch.argsort(torch.rand(n_sel, cand_idx.size(1), device=device), dim=1)
                cand_idx = cand_idx.gather(1, perm)
                cand_vecs = item_bank[cand_idx].clone()
                target_rows, target_cols = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)
                cand_vecs[target_rows, target_cols] = pos_vec[target_rows]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                scores = model.maybe_add_course_score(
                    scores,
                    user_ids,
                    cand_idx,
                    user_seen_items=user_seen_items,
                    target_pop=pop_sel,
                    exclude_items=i,
                    seen_mat=hist_seen_mat,
                    seen_cnt_raw=hist_seen_cnt,
                    concept_seen_mat=concept_seen_mat,
                    concept_seen_cnt_raw=concept_seen_cnt,
                )
                target_indices = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)
            for key, val in batch_res.items():
                accum_metrics[key] = accum_metrics.get(key, 0.0) + val * n_sel
            total_samples += n_sel

    if total_samples < 1:
        return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


def train_one_epoch(model, loader, optimizer, device, cfg, user_seen_items):
    model.train()
    total_loss = 0.0
    steps = 0
    cand_dup_sum = 0.0
    cand_cov_sum = 0.0
    cand_batches = 0

    optimizer.zero_grad()
    cached_user_bank = None
    if cfg.candidate_strategy == "retrieve_sample":
        cached_user_bank = model._build_user_bank_raw()

    for batch_idx, (batch, pop, llm) in enumerate(loader):
        if (
            cached_user_bank is not None and
            cfg.user_bank_refresh_steps > 0 and
            batch_idx > 0 and
            (batch_idx % cfg.user_bank_refresh_steps == 0)
        ):
            cached_user_bank = model._build_user_bank_raw()

        batch = {k: v.to(device) for k, v in batch.items()}
        loss, cand_info = model(
            batch,
            pop.to(device),
            llm.to(device),
            user_bank_raw=cached_user_bank,
            user_seen_items=user_seen_items,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss.item())
        steps += 1
        if cand_info and cand_info.get("steps", 0) > 0:
            cand_dup_sum += cand_info["dup_rate"]
            cand_cov_sum += cand_info["topm_coverage"]
            cand_batches += 1

    avg_loss = total_loss / max(1, steps)
    avg_dup = cand_dup_sum / max(1, cand_batches) if cand_batches > 0 else None
    avg_cov = cand_cov_sum / max(1, cand_batches) if cand_batches > 0 else None
    return avg_loss, avg_dup, avg_cov


def build_course_train_loader(dataset, cfg):
    cold_mask = dataset.pop < cfg.cold_threshold
    n_total = len(dataset)
    n_cold = int(cold_mask.sum().item())
    n_hot = max(0, n_total - n_cold)
    stats = {
        "n_total": n_total,
        "n_cold": n_cold,
        "n_hot": n_hot,
        "sampler": "shuffle",
        "target_cold_ratio": None,
    }

    if cfg.use_cold_balanced_sampler and n_cold > 0 and n_hot > 0:
        cold_ratio = float(min(0.95, max(0.05, cfg.train_cold_ratio)))
        cold_w = cold_ratio / max(1, n_cold)
        hot_w = (1.0 - cold_ratio) / max(1, n_hot)
        weights = torch.where(
            cold_mask,
            torch.full((n_total,), cold_w, dtype=torch.double),
            torch.full((n_total,), hot_w, dtype=torch.double),
        )
        sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_course,
        )
        stats["sampler"] = "cold_balanced"
        stats["target_cold_ratio"] = cold_ratio
        return loader, stats

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_course,
    )
    return loader, stats


def slice_history_tensor(full_history_tensor, row_ids):
    if len(row_ids) < 1:
        return torch.empty((0, full_history_tensor.size(1)), dtype=torch.long)
    idx = torch.tensor([int(x) for x in row_ids], dtype=torch.long)
    return full_history_tensor.index_select(0, idx)


def build_stream_train_dataframe(periods, current_t, cfg):
    window_cfg = int(cfg.stream_train_window)
    if window_cfg <= 0:
        recent_df = pd.concat(periods[: current_t + 1], ignore_index=True)
        stats = {
            "window": 0,
            "recent_periods": current_t + 1,
            "recent_rows": len(recent_df),
            "replay_rows": 0,
            "old_cold_pool": 0,
            "train_rows": len(recent_df),
            "mode": "cumulative",
        }
        return recent_df, stats

    window = max(1, window_cfg)
    recent_start = max(0, current_t - window + 1)
    recent_df = pd.concat(periods[recent_start: current_t + 1], ignore_index=True)

    replay_df = recent_df.iloc[0:0].copy()
    old_cold_pool = 0
    replay_n = 0
    if cfg.use_stream_cold_replay and recent_start > 0:
        old_df = pd.concat(periods[:recent_start], ignore_index=True)
        old_cold_df = old_df[old_df["popularity"] < cfg.cold_threshold].copy()
        old_cold_pool = len(old_cold_df)
        replay_target = int(len(recent_df) * max(0.0, float(cfg.stream_cold_replay_ratio)))
        replay_n = min(old_cold_pool, int(cfg.stream_cold_replay_cap), replay_target)
        if replay_n > 0:
            replay_df = old_cold_df.sample(n=replay_n, random_state=2025 + int(current_t)).reset_index(drop=True)

    if len(replay_df) > 0:
        train_df = pd.concat([recent_df, replay_df], ignore_index=True)
    else:
        train_df = recent_df.reset_index(drop=True)

    stats = {
        "window": window,
        "recent_periods": current_t - recent_start + 1,
        "recent_rows": len(recent_df),
        "replay_rows": len(replay_df),
        "old_cold_pool": old_cold_pool,
        "train_rows": len(train_df),
        "mode": "window_replay",
    }
    return train_df, stats


def run_static_experiment(df, cfg, device, model, optimizer, llm_scores):
    static_seed = int(os.environ.get("USIM_STATIC_SEED", "2025"))
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))

    df_static = df.sample(frac=1.0, random_state=static_seed).reset_index(drop=True)
    n_total = len(df_static)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_df = df_static.iloc[:n_train]
    val_df = df_static.iloc[n_train:n_train + n_val]
    test_df = df_static.iloc[n_train + n_val:]

    train_hist, train_histories = build_history_tensor(
        train_df, base_histories={}, max_len=cfg.course_hist_len, update_histories=True
    )
    val_hist, _ = build_history_tensor(
        val_df, base_histories=train_histories, max_len=cfg.course_hist_len, update_histories=False
    )
    train_val_histories = copy.deepcopy(train_histories)
    _update_histories_from_df(train_val_histories, val_df)
    test_hist, _ = build_history_tensor(
        test_df, base_histories=train_val_histories, max_len=cfg.course_hist_len, update_histories=False
    )

    train_ds = CourseSeqDataset(train_df, llm_scores, train_hist)
    train_loader, train_sampler_stats = build_course_train_loader(train_ds, cfg)
    val_loader = DataLoader(
        CourseSeqDataset(val_df, llm_scores, val_hist),
        batch_size=2048,
        shuffle=False,
        collate_fn=collate_course,
    )
    test_loader = DataLoader(
        CourseSeqDataset(test_df, llm_scores, test_hist),
        batch_size=2048,
        shuffle=False,
        collate_fn=collate_course,
    )

    train_seen = {}
    _add_user_seen_from_df(train_seen, train_df)
    test_seen = _clone_user_seen(train_seen)
    _add_user_seen_from_df(test_seen, val_df)

    print(
        f"\n>>> Start STATIC train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
        f"| train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(
        f"  [TRAIN-SAMPLER] mode={train_sampler_stats['sampler']} | "
        f"cold={train_sampler_stats['n_cold']} | hot={train_sampler_stats['n_hot']} | "
        f"target_cold_ratio={train_sampler_stats['target_cold_ratio'] if train_sampler_stats['target_cold_ratio'] is not None else 'n/a'}"
    )

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    best_val = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        avg_loss, avg_dup, avg_cov = train_one_epoch(model, train_loader, optimizer, device, cfg, train_seen)
        epoch_sec = time.time() - epoch_start

        all_item_vecs_val = build_eval_item_vecs_course(model)
        val_cold, _ = evaluate_course_usim(
            model,
            val_loader,
            device,
            k_list=k_list,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=all_item_vecs_val,
        )
        val_key = val_cold.get("N@10", 0.0) if val_cold else 0.0
        if val_key > best_val:
            best_val = val_key
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if avg_dup is not None:
            print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_sec:.1f}s | CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                f"Val Full Cold N@10: {val_key:.4f}"
            )
        else:
            print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_sec:.1f}s | Val Full Cold N@10: {val_key:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

    all_item_vecs_test = build_eval_item_vecs_course(model)
    met_cold, n_cold_t = evaluate_course_usim(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    met_hot, n_hot_t = evaluate_course_usim(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_cold, fn_c = evaluate_course_usim(
        model, test_loader, device, k_list, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_hot, fn_h = evaluate_course_usim(
        model, test_loader, device, k_list, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT (STATIC): 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for m in metrics_keys:
        print(
            f"{m:<10} | {met_cold.get(m, 0.0) if met_cold else 0.0:<12.4f} | "
            f"{met_hot.get(m, 0.0) if met_hot else 0.0:<12.4f} | "
            f"{fmet_cold.get(m, 0.0) if fmet_cold else 0.0:<12.4f} | "
            f"{fmet_hot.get(m, 0.0) if fmet_hot else 0.0:<12.4f}"
        )
    print("-" * 90)
    print(f"采样 Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"全库 Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Course-Aware USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl").reset_index(drop=True).copy()
    df["_row_id"] = list(range(len(df)))
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = CourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    if {"i_idx", "popularity"}.issubset(df.columns):
        pop_series = df.groupby("i_idx")["popularity"].max()
        for item_idx, pop_val in pop_series.items():
            idx = int(item_idx)
            if 0 <= idx < cfg.n_items:
                item_final_pop[idx] = int(pop_val)
    item_is_cold = item_final_pop < int(cfg.cold_threshold)
    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )

    model = CourseAwareUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_global_llm_scores(llm_scores)
    model.set_item_cold_mask(item_is_cold)
    model.set_item_popularity(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Course-Aware RL-USIM (Batch Size={cfg.batch_size})")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | TopM={cfg.retrieve_top_m} | "
        f"Temp={cfg.candidate_temp:.2f} | Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}, "
        f"prereq_edges={course_stats['prereq_edges_kept']}"
    )
    print(
        f">> Course Main Score: alpha={cfg.course_score_alpha:.2f} | "
        f"lambda={cfg.course_score_lambda:.2f} | min_seen={cfg.course_min_seen} | "
        f"hist_warm={cfg.course_hist_warm_seen} | score_warm={cfg.course_score_warm_seen} | "
        f"penalty_min_seen={cfg.course_penalty_min_seen} | "
        f"bonus_cap={cfg.course_bonus_cap:.2f} | concept_min={cfg.course_concept_min:.2f} | "
        f"prereq_soft={cfg.course_prereq_soft_thr:.2f} | "
        f"prereq_strict={cfg.course_prereq_strict_thr:.2f} | "
        f"conf_floor={cfg.course_score_conf_floor:.2f} | "
        f"topL={cfg.course_score_top_l} | score_train={cfg.course_score_train} | "
        f"cold_loss_w={cfg.cold_loss_weight:.2f} | "
        f"only_cold={cfg.course_score_only_cold} | structured_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> Train Sampler: cold_balanced={cfg.use_cold_balanced_sampler} | "
        f"target_cold_ratio={cfg.train_cold_ratio:.2f}"
    )
    print(
        f">> History Bank: force_cold={cfg.course_hist_force_cold} | "
        f"deterministic={cfg.course_hist_deterministic}"
    )
    concept_scope = str(cfg.course_concept_recent_k) if cfg.course_concept_recent_k > 0 else "full_history"
    print(f">> Concept Bonus: recent_k={concept_scope}")
    print(
        f">> Cold Item Adapter: enabled={cfg.use_cold_item_adapter} | "
        f"beta={cfg.cold_item_adapter_beta:.2f} | "
        f"topk={cfg.cold_item_concept_topk} | "
        f"very_thr={cfg.cold_item_very_cold_thr} | "
        f"beta_very={cfg.cold_item_beta_very:.2f} | "
        f"beta_mild={cfg.cold_item_beta_mild:.2f} | "
        f"cold_items={int(item_is_cold.sum().item())}/{cfg.n_items}"
    )
    print(
        f">> Stream Train: window={cfg.stream_train_window} | "
        f"cold_replay={cfg.use_stream_cold_replay} | "
        f"replay_ratio={cfg.stream_cold_replay_ratio:.2f} | "
        f"replay_cap={cfg.stream_cold_replay_cap}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment(df, cfg, device, model, optimizer, llm_scores)
        return

    full_train_hist, _ = build_history_tensor(
        df, base_histories={}, max_len=cfg.course_hist_len, update_histories=True
    )
    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {"Period": [], "Count_cold": [], "Count_hot": []}
    for prefix in ["cold_", "hot_"]:
        for k in metrics_keys:
            history[prefix + k] = []

    accum_cold = {k: 0.0 for k in metrics_keys}
    accum_hot = {k: 0.0 for k in metrics_keys}
    count_cold, count_hot = 0, 0
    full_cold = {k: 0.0 for k in metrics_keys}
    full_hot = {k: 0.0 for k in metrics_keys}
    fc_cold, fc_hot = 0, 0

    warmup_periods = 3
    accumulated_dfs = []
    user_seen_items = {}
    user_histories = {}

    for t, p_df in enumerate(periods):
        eval_hist, _ = build_history_tensor(
            p_df, base_histories=user_histories, max_len=cfg.course_hist_len, update_histories=False
        )
        eval_ds = CourseSeqDataset(p_df, llm_scores, eval_hist)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_course)
        print(f"\n>>> Period {t} (当前: {len(eval_ds)}, 累积: {sum(len(d) for d in accumulated_dfs) + len(eval_ds)}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= warmup_periods:
            all_item_vecs_eval = build_eval_item_vecs_course(model)
            met_cold, n_cold_t = evaluate_course_usim(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_course_usim(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_course_usim(
                model, eval_loader, device, k_list, eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_course_usim(
                model, eval_loader, device, k_list, eval_type="hot", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )

            if met_cold:
                cold_res = met_cold
                for k in metrics_keys:
                    accum_cold[k] += met_cold[k] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for k in metrics_keys:
                    accum_hot[k] += met_hot[k] * n_hot_t
                count_hot += n_hot_t
            if fmet_cold:
                for k in metrics_keys:
                    full_cold[k] += fmet_cold[k] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for k in metrics_keys:
                    full_hot[k] += fmet_hot[k] * fn_h
                fc_hot += fn_h

            print(
                f"  采样 Cold={met_cold.get('R@10', 0.0) if met_cold else 0.0:.4f} "
                f"Hot={met_hot.get('R@10', 0.0) if met_hot else 0.0:.4f} | "
                f"全库 Cold={fmet_cold.get('R@10', 0.0) if fmet_cold else 0.0:.4f} "
                f"Hot={fmet_hot.get('R@10', 0.0) if fmet_hot else 0.0:.4f}"
            )
        else:
            print("  [WARMUP] Training only...")

        history["Period"].append(t)
        history["Count_cold"].append(n_cold_t)
        history["Count_hot"].append(n_hot_t)
        for key in metrics_keys:
            history["cold_" + key].append(cold_res.get(key, 0.0))
            history["hot_" + key].append(hot_res.get(key, 0.0))

        accumulated_dfs.append(p_df)
        train_df, train_mix_stats = build_stream_train_dataframe(periods, t, cfg)
        train_hist = slice_history_tensor(full_train_hist, train_df["_row_id"].tolist())
        train_ds = CourseSeqDataset(train_df, llm_scores, train_hist)
        train_loader, train_sampler_stats = build_course_train_loader(train_ds, cfg)
        print(
            f"  [TRAIN-SET] mode={train_mix_stats['mode']} | window={train_mix_stats['window']} | "
            f"recent_periods={train_mix_stats['recent_periods']} | recent={train_mix_stats['recent_rows']} | "
            f"replay={train_mix_stats['replay_rows']} | old_cold_pool={train_mix_stats['old_cold_pool']} | "
            f"train={train_mix_stats['train_rows']}"
        )
        print(
            f"  [TRAIN-SAMPLER] mode={train_sampler_stats['sampler']} | "
            f"cold={train_sampler_stats['n_cold']} | hot={train_sampler_stats['n_hot']} | "
            f"target_cold_ratio={train_sampler_stats['target_cold_ratio'] if train_sampler_stats['target_cold_ratio'] is not None else 'n/a'}"
        )

        for epoch in range(cfg.n_epochs):
            epoch_start = time.time()
            avg_loss, avg_dup, avg_cov = train_one_epoch(
                model, train_loader, optimizer, device, cfg, user_seen_items
            )
            epoch_sec = time.time() - epoch_start
            if avg_dup is not None:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(train_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(train_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Course-Aware USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course-Aware USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim.png")
    print(">> Saved mooc_result_course_usim.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
