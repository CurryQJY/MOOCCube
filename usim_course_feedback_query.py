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
from usim import _add_user_seen_from_df, _clone_user_seen, build_course_artifacts, compute_ranking_metrics, setup_seed, split_dataframe_by_periods
from usim_course import CourseSeqDataset, build_all_item_vecs_course, collate_course, train_one_epoch
from usim_course_feedback_lite import (
    FeedbackLiteCourseConfig,
    FeedbackLiteCourseUSIM,
    build_item_popularity,
)


class FeedbackQueryCourseConfig(FeedbackLiteCourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_query_top_m = int(os.environ.get("USIM_FEEDBACK_QUERY_TOPM", "40"))
        self.feedback_query_scale = float(os.environ.get("USIM_FEEDBACK_QUERY_SCALE", "0.20"))
        self.feedback_query_temp = float(os.environ.get("USIM_FEEDBACK_QUERY_TEMP", "0.20"))
        self.feedback_query_only_cold = os.environ.get("USIM_FEEDBACK_QUERY_ONLY_COLD", "1") == "1"
        self.feedback_query_seen_mask = os.environ.get("USIM_FEEDBACK_QUERY_SEEN_MASK", "1") == "1"
        self.feedback_query_accept_alpha = float(os.environ.get("USIM_FEEDBACK_QUERY_ACCEPT_ALPHA", "0.50"))
        self.feedback_query_good_alpha = float(os.environ.get("USIM_FEEDBACK_QUERY_GOOD_ALPHA", "0.50"))
        self.feedback_query_bad_penalty = float(os.environ.get("USIM_FEEDBACK_QUERY_BAD_PENALTY", "0.25"))
        self.feedback_query_aux_weight = float(os.environ.get("USIM_FEEDBACK_QUERY_AUX_WEIGHT", "0.03"))
        self.feedback_query_margin = float(os.environ.get("USIM_FEEDBACK_QUERY_MARGIN", "0.02"))


class FeedbackQueryCourseUSIM(FeedbackLiteCourseUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.query_adapter = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
        )
        self.query_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.emb_dim),
            nn.GELU(),
            nn.Linear(config.emb_dim, 1),
            nn.Sigmoid(),
        )
        self.query_norm = nn.LayerNorm(config.emb_dim)

    def _compose_feedback_query_score(self, accept_prob, type_probs):
        return (
            float(self.cfg.feedback_query_accept_alpha) * accept_prob +
            float(self.cfg.feedback_query_good_alpha) * type_probs[..., 0] -
            float(self.cfg.feedback_query_bad_penalty) * type_probs[..., 1:].sum(dim=-1)
        )

    def adapt_course_query(self, user_vec, user_ids, seen_mat, seen_cnt_raw, target_pop=None, item_bank=None):
        if item_bank is None:
            item_bank = self._get_feedback_item_bank()

        context_vec = self._build_feedback_context(user_ids, seen_mat, seen_cnt_raw)
        coarse_scores = torch.matmul(user_vec, item_bank.t())
        if self.cfg.feedback_query_seen_mask:
            coarse_scores = coarse_scores.masked_fill(seen_mat > 0, -1e9)

        top_m = min(max(1, int(self.cfg.feedback_query_top_m)), item_bank.size(0))
        top_scores, top_idx = torch.topk(coarse_scores, k=top_m, dim=1)

        with torch.no_grad():
            _, _, accept_prob, type_probs = self._feedback_pair_probs(context_vec, top_idx)
            feedback_score = self._compose_feedback_query_score(accept_prob, type_probs)

        blend_scores = top_scores + feedback_score
        blend_temp = max(float(self.cfg.feedback_query_temp), 1e-6)
        blend_weights = F.softmax(blend_scores / blend_temp, dim=1)
        proto_vec = torch.sum(item_bank[top_idx] * blend_weights.unsqueeze(-1), dim=1)

        query_feat = torch.cat([user_vec, context_vec, proto_vec], dim=1)
        gate = self.query_gate(query_feat)
        delta = self.query_adapter(query_feat)
        if self.cfg.feedback_query_only_cold and target_pop is not None:
            apply_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        else:
            apply_mask = torch.ones_like(gate)

        adapted = self.query_norm(user_vec + apply_mask * gate * float(self.cfg.feedback_query_scale) * delta)
        adapted = F.normalize(adapted, dim=1)
        stats = {
            "query_gate": float(gate.mean().item()),
            "query_feedback": float(feedback_score.mean().item()),
        }
        return adapted, context_vec, stats

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        item_bank = self.build_course_item_bank(force_cold=True)
        z_u_base, _, _ = self.encode_course_user(u, hist, item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        z_u_query, context_vec, query_stats = self.adapt_course_query(
            z_u_base,
            user_ids,
            hist_seen_mat,
            hist_seen_cnt,
            target_pop=pop,
            item_bank=item_bank,
        )
        z_i_base, id_e_raw, content_e = self.get_item_vector(i, llm_s, force_cold=False)

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

        z_u = z_u_query
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
                    prereq_aux_loss = F.relu(float(self.cfg.prereq_aux_margin) - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        accept_target, feedback_label = self._compute_feedback_targets(i, hist_seen_mat, hist_seen_cnt)
        pos_accept_logits, pos_type_logits, _, _ = self._feedback_pair_probs(context_vec, i.view(-1, 1))
        pos_accept_logits = pos_accept_logits.squeeze(1)
        pos_type_logits = pos_type_logits.squeeze(1)

        feedback_mask = torch.ones_like(is_cold, dtype=torch.bool)
        if self.cfg.feedback_lite_aux_only_cold:
            feedback_mask = is_cold

        accept_loss = torch.tensor(0.0, device=self.device)
        type_loss = torch.tensor(0.0, device=self.device)
        if feedback_mask.any():
            accept_loss = F.binary_cross_entropy_with_logits(
                pos_accept_logits[feedback_mask],
                accept_target.squeeze(1)[feedback_mask],
            )
            type_loss = F.cross_entropy(
                pos_type_logits[feedback_mask],
                feedback_label[feedback_mask],
            )

        query_aux_loss = torch.tensor(0.0, device=self.device)
        query_rows = is_cold if self.cfg.feedback_query_only_cold else torch.ones_like(is_cold, dtype=torch.bool)
        if query_rows.any():
            pos_base = (z_u_base * z_i).sum(dim=1)
            pos_query = (z_u_query * z_i).sum(dim=1)
            query_aux_loss = F.relu(float(self.cfg.feedback_query_margin) - pos_query[query_rows] + pos_base[query_rows]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            float(self.cfg.feedback_lite_accept_weight) * accept_loss +
            float(self.cfg.feedback_lite_type_weight) * type_loss +
            float(self.cfg.feedback_query_aux_weight) * query_aux_loss
        )
        candidate_stats["query_gate"] = query_stats["query_gate"]
        candidate_stats["query_feedback"] = query_stats["query_feedback"]
        return total_loss, candidate_stats


def evaluate_course_feedback_query(
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
            all_item_vecs = build_all_item_vecs_course(model)

        for batch, pop, _ in loader:
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

            z_u_base, _, _ = model.encode_course_user(u, hist, all_item_vecs)
            hist_seen_mat, hist_seen_cnt = model._hist_to_seen_mat(hist)
            z_u, _, _ = model.adapt_course_query(
                z_u_base,
                user_ids,
                hist_seen_mat,
                hist_seen_cnt,
                target_pop=pop_sel,
                item_bank=all_item_vecs,
            )

            if full_ranking:
                scores = torch.mm(z_u, all_item_vecs.t())
                if user_seen_items:
                    row_idx = torch.arange(n_sel, device=device)
                    target_scores = scores[row_idx, i].clone()
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache.get(uid)
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
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
                cand_vecs = all_item_vecs[cand_idx]
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
                )
                target_indices = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)
            for key, val in batch_res.items():
                accum_metrics[key] = accum_metrics.get(key, 0.0) + val * n_sel
            total_samples += n_sel

    if total_samples < 1:
        return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


def run_static_experiment_feedback_query(df, cfg, device, model, optimizer, llm_scores):
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
        f"\n>>> Start STATIC feedback-query train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
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
        val_cold, _ = evaluate_course_feedback_query(
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
            f"  [STATIC-FEEDBACK-QUERY] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_sec:.1f}s | {tag}Val Full Cold N@10: {val_key:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC-FEEDBACK-QUERY] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

    all_item_vecs_test = build_all_item_vecs_course(model)
    met_cold, n_cold_t = evaluate_course_feedback_query(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    met_hot, n_hot_t = evaluate_course_feedback_query(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_cold, fn_c = evaluate_course_feedback_query(
        model, test_loader, device, k_list, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_hot, fn_h = evaluate_course_feedback_query(
        model, test_loader, device, k_list, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT (STATIC FEEDBACK-QUERY): sampled (1+{cfg.eval_n_neg}) vs full ranking")
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
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Course Feedback-Query USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FeedbackQueryCourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    item_popularity = build_item_popularity(df, cfg.n_items)

    model = FeedbackQueryCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_feedback_artifacts(item_popularity)
    model.set_global_llm_scores(llm_scores)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course Feedback-Query USIM (Batch Size={cfg.batch_size})")
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
        f">> Feedback-Query: topM={cfg.feedback_query_top_m} | scale={cfg.feedback_query_scale:.2f} | "
        f"temp={cfg.feedback_query_temp:.2f} | only_cold={cfg.feedback_query_only_cold} | "
        f"aux_w={cfg.feedback_query_aux_weight:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_feedback_query(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative feedback-query train/eval - total {len(periods)} periods <<<")

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
            met_cold, n_cold_t = evaluate_course_feedback_query(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_course_feedback_query(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_course_feedback_query(
                model, eval_loader, device, k_list, eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_course_feedback_query(
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
                    f"  [TRAIN-FEEDBACK-QUERY] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-FEEDBACK-QUERY] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course Feedback-Query USIM)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_feedback_query.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course Feedback-Query USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim_feedback_query.png")
    print(">> Saved mooc_result_course_usim_feedback_query.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
