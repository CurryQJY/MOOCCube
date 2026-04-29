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
from torch.utils.data import DataLoader

from hhcor_static_hin import build_history_tensor, _update_histories_from_df
from usim import _add_user_seen_from_df, _clone_user_seen, build_course_artifacts, setup_seed, split_dataframe_by_periods
from usim_course import (
    CourseAwareUSIM,
    CourseConfig,
    CourseSeqDataset,
    build_all_item_vecs_course,
    collate_course,
    evaluate_course_usim,
    train_one_epoch,
)


class ColdProtoCourseConfig(CourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.coldproto_top_m = int(os.environ.get("USIM_COLDPROTO_TOPM", "12"))
        self.coldproto_temp = float(os.environ.get("USIM_COLDPROTO_TEMP", "0.20"))
        self.coldproto_scale = float(os.environ.get("USIM_COLDPROTO_SCALE", "0.18"))
        self.coldproto_only_cold = os.environ.get("USIM_COLDPROTO_ONLY_COLD", "1") == "1"
        self.coldproto_concept_alpha = float(os.environ.get("USIM_COLDPROTO_CONCEPT_ALPHA", "0.15"))
        self.coldproto_distill_weight = float(os.environ.get("USIM_COLDPROTO_DISTILL_WEIGHT", "0.03"))
        self.coldproto_margin_weight = float(os.environ.get("USIM_COLDPROTO_MARGIN_WEIGHT", "0.03"))
        self.coldproto_margin = float(os.environ.get("USIM_COLDPROTO_MARGIN", "0.03"))


class ColdProtoCourseUSIM(CourseAwareUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.proto_delta = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
        )
        self.proto_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.emb_dim),
            nn.GELU(),
            nn.Linear(config.emb_dim, 1),
            nn.Sigmoid(),
        )
        self.proto_norm = nn.LayerNorm(config.emb_dim)
        self.item_popularity = None
        self.warm_item_mask = None
        self.warm_item_idx = None
        self.warm_pos_map = None
        self._last_proto_bank = None
        self._last_content_bank = None
        self._last_item_bank = None

    def set_proto_artifacts(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.warm_item_mask = None
            self.warm_item_idx = None
            self.warm_pos_map = None
            return

        pop = item_popularity.to(self.device).float()
        self.item_popularity = pop
        self.warm_item_mask = pop >= float(self.cfg.cold_threshold)
        self.warm_item_idx = torch.nonzero(self.warm_item_mask, as_tuple=False).view(-1)
        warm_pos_map = torch.full((self.cfg.n_items,), -1, dtype=torch.long, device=self.device)
        if self.warm_item_idx.numel() > 0:
            warm_pos_map[self.warm_item_idx] = torch.arange(self.warm_item_idx.numel(), device=self.device)
        self.warm_pos_map = warm_pos_map

    def _proto_apply_mask(self, i_idx=None, target_pop=None, force_cold=False):
        if not self.cfg.coldproto_only_cold:
            if target_pop is not None:
                return torch.ones((target_pop.size(0), 1), dtype=torch.float32, device=self.device)
            if i_idx is not None:
                return torch.ones((i_idx.size(0), 1), dtype=torch.float32, device=self.device)

        if target_pop is not None:
            return (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()

        if i_idx is None or self.item_popularity is None:
            fill = 1.0 if force_cold else 0.0
            size = 1 if i_idx is None else i_idx.size(0)
            return torch.full((size, 1), fill, dtype=torch.float32, device=self.device)

        return (self.item_popularity[i_idx] < float(self.cfg.cold_threshold)).float().unsqueeze(1)

    def _compose_proto_item(self, id_e, content_e, proto_e, apply_mask):
        feat = torch.cat([content_e, proto_e, proto_e - content_e], dim=-1)
        gate = self.proto_gate(feat)
        proto_candidate = self.proto_norm(
            content_e + apply_mask * gate * float(self.cfg.coldproto_scale) * self.proto_delta(feat)
        )
        proto_content = apply_mask * proto_candidate + (1.0 - apply_mask) * content_e
        alpha = self.gate_net(torch.cat([id_e, proto_content], dim=-1))
        item_fused = alpha * id_e + (1.0 - alpha) * proto_content
        return item_fused, proto_content, gate

    def _build_base_banks(self, force_cold=True, item_batch=1024):
        base_items = []
        id_items = []
        content_items = []
        with torch.no_grad():
            for start in range(0, self.cfg.n_items, item_batch):
                end = min(start + item_batch, self.cfg.n_items)
                idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                llm_s = self.global_llm_tensor[idx]
                base_item, id_e, content_e = super().get_item_vector(idx, llm_s, force_cold=force_cold)
                base_items.append(base_item)
                id_items.append(id_e)
                content_items.append(content_e)
        return torch.cat(base_items, dim=0), torch.cat(id_items, dim=0), torch.cat(content_items, dim=0)

    def _compute_proto_bank(self, content_bank, item_batch=1024):
        proto_bank = torch.zeros_like(content_bank)
        if self.warm_item_idx is None or self.warm_item_idx.numel() < 1:
            return proto_bank

        top_m = min(max(1, int(self.cfg.coldproto_top_m)), int(self.warm_item_idx.numel()))
        temp = max(float(self.cfg.coldproto_temp), 1e-6)
        concept_alpha = float(self.cfg.coldproto_concept_alpha)

        content_norm = F.normalize(content_bank, dim=1)
        warm_content = content_norm[self.warm_item_idx]
        warm_id = F.normalize(self.item_id_emb(self.warm_item_idx).detach(), dim=1)

        for start in range(0, self.cfg.n_items, item_batch):
            end = min(start + item_batch, self.cfg.n_items)
            idx = torch.arange(start, end, device=self.device, dtype=torch.long)
            scores = torch.matmul(content_norm[idx], warm_content.t())

            if concept_alpha > 0.0 and self.item_concept_overlap is not None:
                scores = scores + concept_alpha * self.item_concept_overlap[idx][:, self.warm_item_idx]

            if self.warm_pos_map is not None:
                self_pos = self.warm_pos_map[idx]
                valid_rows = self_pos >= 0
                if valid_rows.any():
                    scores[valid_rows, self_pos[valid_rows]] = -1e9

            top_scores, top_pos = torch.topk(scores, k=top_m, dim=1)
            weights = F.softmax(top_scores / temp, dim=1)
            proto_chunk = (warm_id[top_pos] * weights.unsqueeze(-1)).sum(dim=1)
            proto_bank[idx] = F.normalize(proto_chunk, dim=1)

        return proto_bank

    def build_course_item_bank(self, force_cold=True, item_batch=1024, deterministic=False):
        was_training = self.training
        if deterministic and was_training:
            self.eval()
        try:
            with torch.no_grad():
                base_bank, id_bank, content_bank = self._build_base_banks(force_cold=force_cold, item_batch=item_batch)
                proto_bank = self._compute_proto_bank(content_bank, item_batch=item_batch)
                item_bank = []
                for start in range(0, self.cfg.n_items, item_batch):
                    end = min(start + item_batch, self.cfg.n_items)
                    idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                    apply_mask = self._proto_apply_mask(idx, force_cold=force_cold)
                    fused_item, _, _ = self._compose_proto_item(
                        id_bank[idx],
                        content_bank[idx],
                        proto_bank[idx],
                        apply_mask,
                    )
                    item_bank.append(F.normalize(fused_item, dim=1))

                full_item_bank = torch.cat(item_bank, dim=0)
                self._last_proto_bank = proto_bank.detach()
                self._last_content_bank = content_bank.detach()
                self._last_item_bank = full_item_bank.detach()
                return full_item_bank
        finally:
            if deterministic and was_training:
                self.train()

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        item_bank = self.build_course_item_bank(force_cold=True)
        z_u_base, _, _ = self.encode_course_user(u, hist, item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)

        z_i_raw, id_e_raw, content_e = super().get_item_vector(i, llm_s, force_cold=False)
        proto_e = torch.zeros_like(content_e)
        if self._last_proto_bank is not None:
            proto_e = self._last_proto_bank[i]
        apply_mask = self._proto_apply_mask(i, target_pop=pop.to(self.device))
        z_i_base, proto_content, proto_gate = self._compose_proto_item(id_e_raw, content_e, proto_e, apply_mask)

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

        proto_distill_loss = torch.tensor(0.0, device=self.device)
        proto_margin_loss = torch.tensor(0.0, device=self.device)
        valid_proto_rows = (apply_mask.squeeze(1) > 0) & (proto_e.norm(dim=1) > 1e-8)
        if valid_proto_rows.any():
            proto_distill_loss = (
                1.0 - (F.normalize(proto_content[valid_proto_rows], dim=1) * F.normalize(proto_e[valid_proto_rows], dim=1)).sum(dim=1)
            ).mean()
            pos_proto = (z_u[valid_proto_rows] * z_i[valid_proto_rows]).sum(dim=1)
            pos_raw = (z_u[valid_proto_rows] * F.normalize(content_e[valid_proto_rows], dim=1)).sum(dim=1)
            proto_margin_loss = F.relu(float(self.cfg.coldproto_margin) - pos_proto + pos_raw).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            float(self.cfg.coldproto_distill_weight) * proto_distill_loss +
            float(self.cfg.coldproto_margin_weight) * proto_margin_loss
        )
        candidate_stats["proto_gate"] = float(proto_gate.mean().item())
        return total_loss, candidate_stats


def build_item_popularity(df, n_items):
    counts = torch.zeros(n_items, dtype=torch.float32)
    vc = df["i_idx"].value_counts()
    for item_idx, count in vc.items():
        idx = int(item_idx)
        if 0 <= idx < n_items:
            counts[idx] = float(count)
    return counts


def run_static_experiment_coldproto(df, cfg, device, model, optimizer, llm_scores):
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

    train_loader = DataLoader(
        CourseSeqDataset(train_df, llm_scores, train_hist),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_course,
    )
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
        f"\n>>> Start STATIC coldproto train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
        f"| train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
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

        all_item_vecs_val = build_all_item_vecs_course(model)
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

        tag = (
            f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
            if avg_dup is not None else ""
        )
        print(
            f"  [STATIC-COLDPROTO] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_sec:.1f}s | {tag}Val Full Cold N@10: {val_key:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC-COLDPROTO] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

    all_item_vecs_test = build_all_item_vecs_course(model)
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
    print(f"         FINAL REPORT (STATIC COLDPROTO): sampled (1+{cfg.eval_n_neg}) vs full ranking")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for m in metrics_keys:
        print(
            f"{m:<10} | {met_cold.get(m, 0.0) if met_cold else 0.0:<12.4f} | "
            f"{met_hot.get(m, 0.0) if met_hot else 0.0:<12.4f} | "
            f"{fmet_cold.get(m, 0.0) if fmet_cold else 0.0:<12.4f} | "
            f"{fmet_hot.get(m, 0.0) if fmet_hot else 0.0:<12.4f}"
        )
    print("-" * 90)
    print(f"Sampled Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"Full Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)


def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for Course ColdProto USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = ColdProtoCourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir="MOOCCube/relations",
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    item_popularity = build_item_popularity(df, cfg.n_items)

    model = ColdProtoCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_proto_artifacts(item_popularity)
    model.set_global_llm_scores(llm_scores)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course ColdProto USIM (Batch Size={cfg.batch_size})")
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
        f">> ColdProto: topM={cfg.coldproto_top_m} | scale={cfg.coldproto_scale:.2f} | "
        f"temp={cfg.coldproto_temp:.2f} | concept_alpha={cfg.coldproto_concept_alpha:.2f} | "
        f"distill_w={cfg.coldproto_distill_weight:.2f} | margin_w={cfg.coldproto_margin_weight:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_coldproto(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative coldproto train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {"Period": [], "Count_cold": [], "Count_hot": []}
    for prefix in ["cold_", "hot_"]:
        for key in metrics_keys:
            history[prefix + key] = []

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
        print(f"\n>>> Period {t} (current: {len(eval_ds)}, cumulative: {sum(len(d) for d in accumulated_dfs) + len(eval_ds)}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= warmup_periods:
            all_item_vecs_eval = build_all_item_vecs_course(model)
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
                for key in metrics_keys:
                    accum_cold[key] += met_cold[key] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for key in metrics_keys:
                    accum_hot[key] += met_hot[key] * n_hot_t
                count_hot += n_hot_t
            if fmet_cold:
                for key in metrics_keys:
                    full_cold[key] += fmet_cold[key] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for key in metrics_keys:
                    full_hot[key] += fmet_hot[key] * fn_h
                fc_hot += fn_h

            print(
                f"  Sampled Cold={met_cold.get('R@10', 0.0) if met_cold else 0.0:.4f} "
                f"Hot={met_hot.get('R@10', 0.0) if met_hot else 0.0:.4f} | "
                f"Full Cold={fmet_cold.get('R@10', 0.0) if fmet_cold else 0.0:.4f} "
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
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_hist, _ = build_history_tensor(
            combined_df, base_histories={}, max_len=cfg.course_hist_len, update_histories=True
        )
        train_ds = CourseSeqDataset(combined_df, llm_scores, train_hist)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_course)

        for epoch in range(cfg.n_epochs):
            epoch_start = time.time()
            avg_loss, avg_dup, avg_cov = train_one_epoch(model, train_loader, optimizer, device, cfg, user_seen_items)
            epoch_sec = time.time() - epoch_start
            if avg_dup is not None:
                print(
                    f"  [TRAIN-COLDPROTO] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-COLDPROTO] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course ColdProto USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"Sampled Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"Full Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_coldproto.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course ColdProto USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim_coldproto.png")
    print(">> Saved mooc_result_course_usim_coldproto.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
