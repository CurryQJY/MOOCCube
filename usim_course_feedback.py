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
from usim_feedback import FeedbackLoopMixin, apply_feedback_config


class FeedbackCourseConfig(CourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        apply_feedback_config(self)


class FeedbackCourseUSIM(FeedbackLoopMixin, CourseAwareUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.init_feedback_modules(config)
        self.feedback_context_net = nn.Sequential(
            nn.Linear(config.emb_dim * 2 + 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim),
        )
        self.item_popularity = None
        self.item_difficulty = None

    def set_feedback_artifacts(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_difficulty = None
            return

        pop = item_popularity.to(self.device).float()
        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    def build_feedback_context(self, user_vec, hist_vec, seen_mat, seen_cnt_raw, item_idx, pop):
        hist_norm = (seen_cnt_raw / max(1.0, float(self.cfg.course_hist_len))).clamp(0.0, 1.0)
        cold_flag = (pop < self.cfg.cold_threshold).float().unsqueeze(1)
        context_summary = self.feedback_context_net(
            torch.cat([user_vec, hist_vec, hist_norm, cold_flag], dim=1)
        )
        return {
            "user_vec": F.normalize(user_vec, dim=1),
            "hist_vec": hist_vec,
            "seen_mat": seen_mat,
            "seen_cnt_raw": seen_cnt_raw,
            "item_idx": item_idx,
            "pop": pop.float().unsqueeze(1),
            "context_summary": context_summary,
        }

    def summarize_feedback_context(self, current_h, feedback_ctx=None):
        if feedback_ctx is None:
            return super().summarize_feedback_context(current_h, feedback_ctx)
        return feedback_ctx["context_summary"]

    def compute_feedback_signals(self, current_h, feedback_ctx=None, target_emb=None):
        if feedback_ctx is None:
            return super().compute_feedback_signals(current_h, feedback_ctx, target_emb)

        seen_mat = feedback_ctx["seen_mat"]
        seen_cnt_raw = feedback_ctx["seen_cnt_raw"]
        item_idx = feedback_ctx["item_idx"]
        batch_idx = torch.arange(item_idx.size(0), device=self.device)
        zero = torch.zeros((item_idx.size(0), 1), dtype=torch.float32, device=self.device)

        prereq_gap = zero
        concept_match = zero
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            prereq_gap = violation_full[batch_idx, item_idx].unsqueeze(1)

        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0)

        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[item_idx].unsqueeze(1)
        else:
            item_difficulty = zero

        user_readiness = (
            seen_cnt_raw / max(1.0, float(self.cfg.course_hist_warm_seen))
        ).clamp(0.0, 1.0)
        difficulty_gap = F.relu(item_difficulty - user_readiness)

        redundant_mask = (
            (concept_match >= float(self.cfg.feedback_redundant_thr)) &
            (seen_cnt_raw >= float(self.cfg.course_min_seen))
        )
        topic_drift_mask = (
            (concept_match < float(self.cfg.feedback_concept_thr)) &
            (seen_cnt_raw > 0)
        )
        prereq_mask = prereq_gap >= float(self.cfg.feedback_prereq_thr)
        difficulty_mask = difficulty_gap >= float(self.cfg.feedback_diff_thr)

        feedback_label = torch.zeros(item_idx.size(0), dtype=torch.long, device=self.device)
        feedback_label = torch.where(redundant_mask.squeeze(1), torch.full_like(feedback_label, 4), feedback_label)
        feedback_label = torch.where(topic_drift_mask.squeeze(1), torch.full_like(feedback_label, 3), feedback_label)
        feedback_label = torch.where(difficulty_mask.squeeze(1), torch.full_like(feedback_label, 2), feedback_label)
        feedback_label = torch.where(prereq_mask.squeeze(1), torch.full_like(feedback_label, 1), feedback_label)

        accept_target = (feedback_label == 0).float().unsqueeze(1)
        return {
            "accept_target": accept_target,
            "feedback_label": feedback_label,
            "prereq_gap": prereq_gap,
            "difficulty_gap": difficulty_gap,
            "concept_match": concept_match,
            "user_vec": feedback_ctx["user_vec"],
            "item_idx": item_idx,
        }

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        item_bank = self.build_course_item_bank(force_cold=True)
        z_u_base, hist_vec, _ = self.encode_course_user(u, hist, item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        z_i_base, id_e_raw, content_e = self.get_item_vector(i, llm_s, force_cold=False)

        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

        feedback_ctx = self.build_feedback_context(
            z_u_base,
            hist_vec,
            hist_seen_mat,
            hist_seen_cnt,
            i,
            pop.to(self.device),
        )
        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
            feedback_ctx=feedback_ctx,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)
        accept_loss, feedback_type_loss = self.compute_feedback_aux_loss(trajectory)

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

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            float(self.cfg.feedback_accept_weight) * accept_loss +
            float(self.cfg.feedback_type_weight) * feedback_type_loss
        )
        return total_loss, candidate_stats


def build_item_popularity(df, n_items):
    counts = torch.zeros(n_items, dtype=torch.float32)
    vc = df["i_idx"].value_counts()
    for item_idx, count in vc.items():
        idx = int(item_idx)
        if 0 <= idx < n_items:
            counts[idx] = float(count)
    return counts


def run_static_experiment_feedback(df, cfg, device, model, optimizer, llm_scores):
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
        f"\n>>> Start STATIC feedback train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
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
            f"  [STATIC-FEEDBACK] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_sec:.1f}s | {tag}Val Full Cold N@10: {val_key:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC-FEEDBACK] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

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
    print(f"         FINAL REPORT (STATIC FEEDBACK): sampled (1+{cfg.eval_n_neg}) vs full ranking")
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
    print(f"Loading Data for Course Feedback USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FeedbackCourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
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

    model = FeedbackCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_feedback_artifacts(item_popularity)
    model.set_global_llm_scores(llm_scores)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course Feedback USIM (Batch Size={cfg.batch_size})")
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
        f">> Feedback Loop: steps={cfg.usim_steps} | accept_tau={cfg.feedback_accept_tau:.2f} | "
        f"memory_scale={cfg.feedback_memory_scale:.2f} | context_scale={cfg.feedback_context_scale:.2f} | "
        f"update_scale={cfg.feedback_update_scale:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_feedback(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative feedback train/eval - total {len(periods)} periods <<<")

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
                    f"  [TRAIN-FEEDBACK] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-FEEDBACK] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course Feedback USIM)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_feedback.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course Feedback USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim_feedback.png")
    print(">> Saved mooc_result_course_usim_feedback.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
