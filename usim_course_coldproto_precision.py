import copy
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hhcor_static_hin import build_history_tensor, _update_histories_from_df
from usim import _add_user_seen_from_df, _clone_user_seen, build_course_artifacts, setup_seed, split_dataframe_by_periods
from usim_course import (
    CourseSeqDataset,
    build_all_item_vecs_course,
    collate_course,
    evaluate_course_usim,
    train_one_epoch,
)
from usim_course_coldproto import ColdProtoCourseConfig, ColdProtoCourseUSIM, build_item_popularity
from usim_course_feedback_lite import FeedbackLiteCourseUSIM


class ColdProtoPrecisionConfig(ColdProtoCourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.coldproto_top_m = int(os.environ.get("USIM_COLDPROTO_PREC_TOPM", "6"))
        self.coldproto_temp = float(os.environ.get("USIM_COLDPROTO_PREC_TEMP", "0.15"))
        self.coldproto_scale = float(os.environ.get("USIM_COLDPROTO_PREC_SCALE", "0.10"))
        self.coldproto_concept_alpha = float(os.environ.get("USIM_COLDPROTO_PREC_CONCEPT_ALPHA", "0.25"))
        self.coldproto_prec_min_concept = float(os.environ.get("USIM_COLDPROTO_PREC_MIN_CONCEPT", "0.10"))
        self.coldproto_prec_hard_bonus = float(os.environ.get("USIM_COLDPROTO_PREC_HARD_BONUS", "0.20"))
        self.coldproto_prec_prereq_bonus = float(os.environ.get("USIM_COLDPROTO_PREC_PREREQ_BONUS", "0.10"))
        self.coldproto_prec_min_neighbors = int(os.environ.get("USIM_COLDPROTO_PREC_MIN_NEIGHBORS", "3"))

        self.feedback_lite_accept_weight = float(os.environ.get("USIM_COLDPROTO_PREC_ACCEPT_WEIGHT", "0.05"))
        self.feedback_lite_type_weight = float(os.environ.get("USIM_COLDPROTO_PREC_TYPE_WEIGHT", "0.03"))
        self.feedback_lite_top_l = int(os.environ.get("USIM_COLDPROTO_PREC_TOPL", "12"))
        self.feedback_lite_train = os.environ.get("USIM_COLDPROTO_PREC_TRAIN_RERANK", "0") == "1"
        self.feedback_lite_only_cold = os.environ.get("USIM_COLDPROTO_PREC_ONLY_COLD", "1") == "1"
        self.feedback_lite_aux_only_cold = os.environ.get("USIM_COLDPROTO_PREC_AUX_ONLY_COLD", "1") == "1"
        self.feedback_lite_warm_seen = int(os.environ.get("USIM_COLDPROTO_PREC_WARM_SEEN", str(self.course_score_warm_seen)))
        self.feedback_lite_good_alpha = float(os.environ.get("USIM_COLDPROTO_PREC_GOOD_ALPHA", "0.03"))
        self.feedback_lite_accept_alpha = float(os.environ.get("USIM_COLDPROTO_PREC_ACCEPT_ALPHA", "0.01"))
        self.feedback_lite_prereq_penalty = float(os.environ.get("USIM_COLDPROTO_PREC_PREREQ_PENALTY", "0.02"))
        self.feedback_lite_diff_penalty = float(os.environ.get("USIM_COLDPROTO_PREC_DIFF_PENALTY", "0.01"))
        self.feedback_lite_topic_penalty = float(os.environ.get("USIM_COLDPROTO_PREC_TOPIC_PENALTY", "0.01"))
        self.feedback_lite_redundant_penalty = float(os.environ.get("USIM_COLDPROTO_PREC_REDUNDANT_PENALTY", "0.01"))
        self.feedback_lite_prereq_thr = float(os.environ.get("USIM_COLDPROTO_PREC_PREREQ_THR", "0.55"))
        self.feedback_lite_diff_thr = float(os.environ.get("USIM_COLDPROTO_PREC_DIFF_THR", "0.25"))
        self.feedback_lite_concept_thr = float(os.environ.get("USIM_COLDPROTO_PREC_CONCEPT_THR", "0.10"))
        self.feedback_lite_redundant_thr = float(os.environ.get("USIM_COLDPROTO_PREC_REDUNDANT_THR", "0.75"))


class ColdProtoPrecisionCourseUSIM(FeedbackLiteCourseUSIM, ColdProtoCourseUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)

    def set_precision_artifacts(self, item_popularity):
        self.set_proto_artifacts(item_popularity)
        self.set_feedback_artifacts(item_popularity)

    def _compute_proto_bank(self, content_bank, item_batch=1024):
        proto_bank = torch.zeros_like(content_bank)
        if self.warm_item_idx is None or self.warm_item_idx.numel() < 1:
            return proto_bank

        top_m = min(max(1, int(self.cfg.coldproto_top_m)), int(self.warm_item_idx.numel()))
        temp = max(float(self.cfg.coldproto_temp), 1e-6)
        concept_alpha = float(self.cfg.coldproto_concept_alpha)
        concept_thr = float(self.cfg.coldproto_prec_min_concept)
        hard_bonus = float(self.cfg.coldproto_prec_hard_bonus)
        prereq_bonus = float(self.cfg.coldproto_prec_prereq_bonus)
        min_neighbors = max(1, int(self.cfg.coldproto_prec_min_neighbors))

        content_norm = F.normalize(content_bank, dim=1)
        warm_content = content_norm[self.warm_item_idx]
        warm_id = F.normalize(self.item_id_emb(self.warm_item_idx).detach(), dim=1)

        for start in range(0, self.cfg.n_items, item_batch):
            end = min(start + item_batch, self.cfg.n_items)
            idx = torch.arange(start, end, device=self.device, dtype=torch.long)
            scores = torch.matmul(content_norm[idx], warm_content.t())

            concept_overlap = None
            if self.item_concept_overlap is not None:
                concept_overlap = self.item_concept_overlap[idx][:, self.warm_item_idx]
                scores = scores + concept_alpha * concept_overlap

            hard_mask = None
            if self.item_hard_adj is not None:
                hard_mask = self.item_hard_adj[idx][:, self.warm_item_idx]
                scores = scores + hard_bonus * hard_mask.float()

            prereq_mask = None
            if self.item_prereq_item_mat is not None:
                prereq_mask = self.item_prereq_item_mat[idx][:, self.warm_item_idx] > 0
                scores = scores + prereq_bonus * prereq_mask.float()

            if self.warm_pos_map is not None:
                self_pos = self.warm_pos_map[idx]
                valid_rows = self_pos >= 0
                if valid_rows.any():
                    scores[valid_rows, self_pos[valid_rows]] = -1e9

            if concept_overlap is not None:
                strong_mask = concept_overlap >= concept_thr
                if hard_mask is not None:
                    strong_mask = strong_mask | hard_mask
                if prereq_mask is not None:
                    strong_mask = strong_mask | prereq_mask
                masked_scores = scores.masked_fill(~strong_mask, -1e9)
                use_mask = strong_mask.sum(dim=1) >= min_neighbors
                scores = torch.where(use_mask.unsqueeze(1), masked_scores, scores)

            top_scores, top_pos = torch.topk(scores, k=top_m, dim=1)
            weights = F.softmax(top_scores / temp, dim=1)
            proto_chunk = (warm_id[top_pos] * weights.unsqueeze(-1)).sum(dim=1)
            proto_bank[idx] = F.normalize(proto_chunk, dim=1)

        return proto_bank

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        base_loss, candidate_stats = ColdProtoCourseUSIM.forward(
            self,
            batch,
            pop,
            llm_s,
            user_bank_raw=user_bank_raw,
            user_seen_items=user_seen_items,
        )

        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        context_vec = self._build_feedback_context(user_ids, hist_seen_mat, hist_seen_cnt)
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

        total_loss = (
            base_loss +
            float(self.cfg.feedback_lite_accept_weight) * accept_loss +
            float(self.cfg.feedback_lite_type_weight) * type_loss
        )
        return total_loss, candidate_stats


def run_static_experiment_coldproto_precision(df, cfg, device, model, optimizer, llm_scores):
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
        f"\n>>> Start STATIC coldproto-precision train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
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
            f"  [STATIC-COLDPROTO-PREC] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_sec:.1f}s | {tag}Val Full Cold N@10: {val_key:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC-COLDPROTO-PREC] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

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
    print(f"         FINAL REPORT (STATIC COLDPROTO-PREC): sampled (1+{cfg.eval_n_neg}) vs full ranking")
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
    print(f"Loading Data for Course ColdProto-Precision USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = ColdProtoPrecisionConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
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

    model = ColdProtoPrecisionCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_precision_artifacts(item_popularity)
    model.set_global_llm_scores(llm_scores)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course ColdProto-Precision USIM (Batch Size={cfg.batch_size})")
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
        f">> ColdProto-PREC: topM={cfg.coldproto_top_m} | scale={cfg.coldproto_scale:.2f} | "
        f"concept_thr={cfg.coldproto_prec_min_concept:.2f} | fb_topL={cfg.feedback_lite_top_l} | "
        f"fb_aux=({cfg.feedback_lite_accept_weight:.2f},{cfg.feedback_lite_type_weight:.2f})"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_coldproto_precision(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative coldproto-precision train/eval - total {len(periods)} periods <<<")

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
                    f"  [TRAIN-COLDPROTO-PREC] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-COLDPROTO-PREC] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course ColdProto-Precision USIM)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_coldproto_precision.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course ColdProto-Precision USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim_coldproto_precision.png")
    print(">> Saved mooc_result_course_usim_coldproto_precision.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
