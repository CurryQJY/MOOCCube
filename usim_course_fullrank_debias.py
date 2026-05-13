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
    build_course_train_loader,
    collate_course,
    evaluate_course_usim,
    train_one_epoch,
)
from usim_course_fullrank import FullRankCourseConfig, FullRankCourseUSIM, build_item_popularity


class FullRankDebiasCourseConfig(FullRankCourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.fullrank_debias_train_tau = float(os.environ.get("USIM_FULLRANK_DEBIAS_TRAIN_TAU", "0.35"))
        self.fullrank_debias_eval_tau = float(os.environ.get("USIM_FULLRANK_DEBIAS_EVAL_TAU", "0.10"))
        self.fullrank_tail_weight_power = float(os.environ.get("USIM_FULLRANK_TAIL_POWER", "0.50"))
        self.fullrank_tail_weight_cap = float(os.environ.get("USIM_FULLRANK_TAIL_CAP", "4.00"))


class FullRankDebiasCourseUSIM(FullRankCourseUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.item_log_pop = None
        self.item_tail_weight = None

    def set_fullrank_artifacts(self, item_popularity):
        super().set_fullrank_artifacts(item_popularity)
        if item_popularity is None:
            self.item_log_pop = None
            self.item_tail_weight = None
            return

        pop = item_popularity.to(self.device).float()
        centered_log_pop = torch.log1p(pop)
        centered_log_pop = centered_log_pop - centered_log_pop.mean()
        self.item_log_pop = centered_log_pop

        power = max(0.0, float(self.cfg.fullrank_tail_weight_power))
        cap = max(1.0, float(self.cfg.fullrank_tail_weight_cap))
        tail = ((pop.mean() + 1.0) / (pop + 1.0)).pow(power)
        tail = tail / tail.mean().clamp_min(1e-12)
        self.item_tail_weight = tail.clamp(max=cap)

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
        scores = super().maybe_add_course_score(
            base_scores,
            user_ids,
            candidate_idx,
            user_seen_items=user_seen_items,
            target_pop=target_pop,
            exclude_items=exclude_items,
            seen_mat=seen_mat,
            seen_cnt_raw=seen_cnt_raw,
            concept_seen_mat=concept_seen_mat,
            concept_seen_cnt_raw=concept_seen_cnt_raw,
            training=training,
        )
        if self.item_log_pop is None:
            return scores

        tau = float(self.cfg.fullrank_debias_train_tau if training else self.cfg.fullrank_debias_eval_tau)
        if tau <= 0.0:
            return scores
        return scores - tau * self.item_log_pop[candidate_idx]

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        hist_item_bank = self.build_course_item_bank(
            force_cold=self.cfg.course_hist_force_cold,
            deterministic=self.cfg.course_hist_deterministic,
        )
        z_u_base, _, _ = self.encode_course_user(u, hist, hist_item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        if self.cfg.course_concept_recent_k > 0:
            concept_seen_mat, concept_seen_cnt = self._hist_to_recent_seen_mat(hist, self.cfg.course_concept_recent_k)
        else:
            concept_seen_mat, concept_seen_cnt = hist_seen_mat, hist_seen_cnt

        _, id_e_raw, content_e = self.get_item_vector(i, llm_s, force_cold=False)
        z_u = F.normalize(z_u_base, dim=1)
        full_item_bank = self.build_course_item_bank_grad(
            force_cold=self.cfg.fullrank_force_cold_items,
            item_batch=self.cfg.fullrank_item_batch,
        )

        scores = torch.matmul(z_u, full_item_bank.t()) / self.cfg.temp
        row_idx = torch.arange(scores.size(0), device=self.device)
        target_scores = scores[row_idx, i].clone()
        if hist_seen_mat.max().item() > 0:
            scores = scores.masked_fill(hist_seen_mat.bool(), -1e9)
            scores[row_idx, i] = target_scores

        cand_idx = torch.arange(self.cfg.n_items, device=self.device, dtype=torch.long).view(1, -1).expand(scores.size(0), -1)
        scores = self.maybe_add_course_score(
            scores,
            user_ids,
            cand_idx,
            user_seen_items=user_seen_items,
            target_pop=pop,
            exclude_items=i,
            seen_mat=hist_seen_mat,
            seen_cnt_raw=hist_seen_cnt,
            concept_seen_mat=concept_seen_mat,
            concept_seen_cnt_raw=concept_seen_cnt,
            training=self.training,
        )

        row_weights = torch.where(
            is_cold,
            torch.full_like(pop, float(self.cfg.cold_loss_weight), dtype=torch.float32),
            torch.ones_like(pop, dtype=torch.float32),
        )
        if self.item_tail_weight is not None:
            row_weights = row_weights * self.item_tail_weight[i]

        per_row_loss = F.cross_entropy(scores, i, reduction="none")
        main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)
        candidate_stats = {"dup_rate": 0.0, "topm_coverage": 0.0, "steps": 0}

        labels = torch.arange(i.size(0), device=self.device)
        z_id = F.normalize(id_e_raw, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None
        ):
            seen_mat = hist_seen_mat
            seen_cnt = hist_seen_cnt.squeeze(1)
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            valid_rows = seen_cnt >= float(self.cfg.prereq_aux_min_seen)
            if self.cfg.prereq_aux_only_cold:
                valid_rows = valid_rows & is_cold

            candidate_mask = violation_full > float(self.cfg.prereq_aux_violation_thr)
            candidate_mask[torch.arange(i.size(0), device=self.device), i] = False
            candidate_mask = candidate_mask & valid_rows.unsqueeze(1)

            if candidate_mask.any():
                neg_scores = scores.masked_fill(~candidate_mask, -1e9)
                neg_vals, _ = neg_scores.max(dim=1)
                has_neg = neg_vals > -1e8
                if has_neg.any():
                    pos_vals = scores[torch.arange(i.size(0), device=self.device), i]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss
        )
        return total_loss, candidate_stats


def run_static_experiment_fullrank_debias(df, cfg, device, model, optimizer, llm_scores):
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
        f"\n>>> Start STATIC fullrank-debias train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
        f"| train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(
        f"  [TRAIN-SAMPLER] mode={train_sampler_stats['sampler']} | "
        f"cold={train_sampler_stats['n_cold']} | hot={train_sampler_stats['n_hot']} | "
        f"target_cold_ratio={train_sampler_stats['target_cold_ratio'] if train_sampler_stats['target_cold_ratio'] is not None else 'n/a'}"
    )

    k_list = [5, 10, 20]
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

        if avg_dup is not None:
            print(
                f"  [STATIC-FULLRANK-DEBIAS] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_sec:.1f}s | CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                f"Val Full Cold N@10: {val_key:.4f}"
            )
        else:
            print(
                f"  [STATIC-FULLRANK-DEBIAS] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_sec:.1f}s | Val Full Cold N@10: {val_key:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

    all_item_vecs_test = build_all_item_vecs_course(model)
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
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
    print(f"         FINAL REPORT (STATIC FULLRANK-DEBIAS): sampled (1+{cfg.eval_n_neg}) vs full ranking")
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
    print(f"Loading Data for Course FullRank-Debias USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FullRankDebiasCourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
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

    model = FullRankDebiasCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_global_llm_scores(llm_scores)
    model.set_fullrank_artifacts(item_popularity)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course FullRank-Debias USIM (Batch Size={cfg.batch_size})")
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
        f">> FullRank Debias: train_tau={cfg.fullrank_debias_train_tau:.2f} | "
        f"eval_tau={cfg.fullrank_debias_eval_tau:.2f} | "
        f"tail_power={cfg.fullrank_tail_weight_power:.2f} | tail_cap={cfg.fullrank_tail_weight_cap:.2f}"
    )
    print(
        f">> FullRank Train: force_cold_items={cfg.fullrank_force_cold_items} | "
        f"teacher_w={cfg.fullrank_teacher_weight:.2f}"
    )
    print(
        f">> Train Sampler: cold_balanced={cfg.use_cold_balanced_sampler} | "
        f"target_cold_ratio={cfg.train_cold_ratio:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_fullrank_debias(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative fullrank-debias train/eval - total {len(periods)} periods <<<")

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
        train_loader, train_sampler_stats = build_course_train_loader(train_ds, cfg)
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
            print(
                f"  [TRAIN-FULLRANK-DEBIAS] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                + (
                    f" | CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                    if avg_dup is not None else ""
                )
            )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course FullRank-Debias USIM)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_fullrank_debias.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course FullRank-Debias USIM: Cumulative Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mooc_result_course_usim_fullrank_debias.png")


if __name__ == "__main__":
    setup_seed(2025)
    main()
