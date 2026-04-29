import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hhcor_static_hin import build_history_tensor, _update_histories_from_df
from usim import _add_user_seen_from_df, _clone_user_seen, build_course_artifacts, setup_seed, split_dataframe_by_periods
from usim_course import (
    CourseAwareUSIM,
    CourseConfig,
    CourseSeqDataset,
    build_all_item_vecs_course,
    build_course_train_loader,
    collate_course,
    evaluate_course_usim,
    run_static_experiment,
    train_one_epoch,
)
from usim_plus import ColdResidualAdapterMixin, build_item_popularity


class CoursePlusConfig(CourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.use_cold_item_adapter = False
        self.plus_only_cold = os.environ.get("USIM_PLUS_ONLY_COLD", "1") == "1"
        self.plus_top_m = int(os.environ.get("USIM_PLUS_TOPM", "8"))
        self.plus_temp = float(os.environ.get("USIM_PLUS_TEMP", "0.20"))
        self.plus_scale = float(os.environ.get("USIM_PLUS_SCALE", "0.12"))
        self.plus_affinity_alpha = float(os.environ.get("USIM_PLUS_AFFINITY_ALPHA", "0.15"))
        self.plus_item_batch = int(os.environ.get("USIM_PLUS_ITEM_BATCH", "1024"))


class CourseUSIMPlus(ColdResidualAdapterMixin, CourseAwareUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self._init_plus_adapter(config)

    def set_course_plus_artifacts(self, item_popularity):
        self.set_plus_artifacts(item_popularity, neighbor_affinity=self.item_concept_overlap)


def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for Course USIM+ from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = CoursePlusConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    if {"i_idx", "popularity"}.issubset(df.columns):
        pop_series = df.groupby("i_idx")["popularity"].max()
        for item_idx, pop_val in pop_series.items():
            idx = int(item_idx)
            if 0 <= idx < cfg.n_items:
                item_final_pop[idx] = int(pop_val)
    item_is_cold = item_final_pop < int(cfg.cold_threshold)
    item_popularity = build_item_popularity(df, cfg.n_items)

    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir="MOOCCube/relations",
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )

    model = CourseUSIMPlus(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_global_llm_scores(llm_scores)
    model.set_item_cold_mask(item_is_cold)
    model.set_course_plus_artifacts(item_popularity)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course USIM+ (Batch Size={cfg.batch_size})")
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
        f">> USIM+: plus_topM={cfg.plus_top_m} | plus_scale={cfg.plus_scale:.2f} | "
        f"plus_temp={cfg.plus_temp:.2f} | affinity_alpha={cfg.plus_affinity_alpha:.2f} | "
        f"only_cold={cfg.plus_only_cold}"
    )
    print(
        f">> Course Main Score: alpha={cfg.course_score_alpha:.2f} | "
        f"lambda={cfg.course_score_lambda:.2f} | min_seen={cfg.course_min_seen} | "
        f"hist_warm={cfg.course_hist_warm_seen} | score_warm={cfg.course_score_warm_seen} | "
        f"penalty_min_seen={cfg.course_penalty_min_seen} | topL={cfg.course_score_top_l} | "
        f"score_train={cfg.course_score_train} | cold_loss_w={cfg.cold_loss_weight:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative USIM+ train/eval - total {len(periods)} periods <<<")

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
            if avg_dup is not None:
                print(
                    f"  [TRAIN-USIM+] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-USIM+] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course USIM+)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_plus.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course USIM+: Cumulative Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mooc_result_course_usim_plus.png")


if __name__ == "__main__":
    setup_seed(2025)
    main()
