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

from usim import (
    Config as BaseConfig,
    PAM_RL_Pure_USIM,
    StreamDataset,
    _add_user_seen_from_df,
    build_eval_item_vecs,
    build_course_artifacts,
    collate_fn,
    evaluate_usim,
    run_static_experiment,
    setup_seed,
    split_dataframe_by_periods,
)


class FeedbackConfig(BaseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.reward_terminal_weight = float(os.environ.get("USIM_FB_REWARD_TERM_W", "10.0"))
        self.reward_gain_weight = float(os.environ.get("USIM_FB_REWARD_GAIN_W", "5.0"))
        self.reward_gain_clip = float(os.environ.get("USIM_FB_REWARD_GAIN_CLIP", "0.05"))
        self.reward_dup_penalty_weight = float(os.environ.get("USIM_FB_REWARD_DUP_W", "0.50"))
        self.reward_cov_bonus_weight = float(os.environ.get("USIM_FB_REWARD_COV_W", "0.00"))


class FeedbackAwareUSIM(PAM_RL_Pure_USIM):
    def run_usim_episode(self, init_item_emb, target_emb=None, user_bank_raw=None):
        current_h = init_item_emb.clone()
        trajectory = {
            "log_probs": [],
            "values": [],
            "rewards": [],
            "entropies": [],
            "states": [],
            "time_steps": [],
            "candidates": [],
            "actions": [],
        }
        candidate_stats = {
            "dup_rate": 0.0,
            "topm_coverage": 0.0,
            "steps": 0,
            "step_gain": 0.0,
            "collapse_penalty": 0.0,
        }

        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw = self._build_user_bank_raw()

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, _, cand_stats = self.get_candidates(current_h, user_bank_raw=user_bank_raw)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            if cand_stats is not None:
                candidate_stats["dup_rate"] += cand_stats["dup_rate"]
                candidate_stats["topm_coverage"] += cand_stats["topm_coverage"]
                candidate_stats["steps"] += 1

            trajectory["states"].append(current_h.detach().clone())
            trajectory["time_steps"].append(time_step.detach().clone())
            trajectory["candidates"].append(candidates.detach().clone())
            trajectory["actions"].append(action_idx.detach().clone())

            prev_h = current_h
            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            step_gain_mean = 0.0
            collapse_penalty = 0.0
            if target_emb is not None:
                prev_dist = F.mse_loss(prev_h, target_emb, reduction="none").mean(dim=1, keepdim=True)
                new_dist = F.mse_loss(current_h, target_emb, reduction="none").mean(dim=1, keepdim=True)

                terminal_reward = -new_dist * float(self.cfg.reward_terminal_weight)
                step_gain = (prev_dist - new_dist).clamp(
                    min=-float(self.cfg.reward_gain_clip),
                    max=float(self.cfg.reward_gain_clip),
                )
                reward = terminal_reward + float(self.cfg.reward_gain_weight) * step_gain
                step_gain_mean = float(step_gain.mean().item())

                if cand_stats is not None:
                    collapse_penalty = float(self.cfg.reward_dup_penalty_weight) * float(cand_stats["dup_rate"])
                    reward = reward - collapse_penalty
                    if float(self.cfg.reward_cov_bonus_weight) > 0.0:
                        reward = reward + float(self.cfg.reward_cov_bonus_weight) * float(cand_stats["topm_coverage"])

            candidate_stats["step_gain"] += step_gain_mean
            candidate_stats["collapse_penalty"] += collapse_penalty
            trajectory["log_probs"].append(log_prob.detach())
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

        if candidate_stats["steps"] > 0:
            candidate_stats["dup_rate"] /= candidate_stats["steps"]
            candidate_stats["topm_coverage"] /= candidate_stats["steps"]
            candidate_stats["step_gain"] /= candidate_stats["steps"]
            candidate_stats["collapse_penalty"] /= candidate_stats["steps"]
        return current_h, trajectory, candidate_stats


def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for Feedback-Aware USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FeedbackConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
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

    model = FeedbackAwareUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Feedback-Aware RL-USIM + InfoNCE (Batch Size={cfg.batch_size})")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> Reward Loop: term_w={cfg.reward_terminal_weight:.2f} | "
        f"gain_w={cfg.reward_gain_weight:.2f} | "
        f"gain_clip={cfg.reward_gain_clip:.3f} | "
        f"dup_w={cfg.reward_dup_penalty_weight:.2f} | "
        f"cov_w={cfg.reward_cov_bonus_weight:.2f}"
    )
    print(
        f">> Cold Train: force_cold={cfg.train_force_cold} | "
        f"id_dropout={cfg.dropout_prob:.2f}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}, "
        f"prereq_edges={course_stats['prereq_edges_kept']} "
        f"(raw={course_stats['prereq_edges_raw']}, users={course_stats['prereq_users']})"
    )
    print(
        f">> Course Mode: rerank={cfg.use_course_rerank} "
        f"(alpha={cfg.rerank_alpha:.2f}, lambda={cfg.rerank_lambda:.2f}) | "
        f"min_seen={cfg.rerank_min_seen} | topL={cfg.rerank_top_l} | "
        f"cap={cfg.rerank_penalty_cap:.2f} | only_cold={cfg.rerank_only_cold} | "
        f"prereq[min_sup={cfg.prereq_min_support}, max_per_item={cfg.prereq_max_per_item}] | "
        f"prereq_aux={cfg.use_prereq_aux_loss} (w={cfg.prereq_aux_weight:.2f}) | "
        f"structured_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"tie=Full Cold R@{cfg.early_stop_k} | hot_drop_tol={cfg.early_stop_hot_r10_drop_tol:.2%} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        print(">> Static 模式复用原版 static 训练流程，仅更换模型与 reward。")
        run_static_experiment(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {"Period": [], "Count_cold": [], "Count_hot": []}
    for prefix in ["cold_", "hot_"]:
        for key in metrics_keys:
            history[prefix + key] = []

    accum_cold = {key: 0.0 for key in metrics_keys}
    accum_hot = {key: 0.0 for key in metrics_keys}
    count_cold, count_hot = 0, 0
    full_cold = {key: 0.0 for key in metrics_keys}
    full_hot = {key: 0.0 for key in metrics_keys}
    fc_cold, fc_hot = 0, 0

    warmup_periods = 3
    accumulated_dfs = []
    user_seen_items = {}

    for t in range(len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= warmup_periods:
            all_item_vecs_eval = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, n_cold_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="hot", full_ranking=True,
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

            c_s = met_cold["R@10"] if met_cold else 0.0
            h_s = met_hot["R@10"] if met_hot else 0.0
            c_f = fmet_cold["R@10"] if fmet_cold else 0.0
            h_f = fmet_hot["R@10"] if fmet_hot else 0.0
            print(f"  采样 Cold={c_s:.4f} Hot={h_s:.4f} | 全库 Cold={c_f:.4f} Hot={h_f:.4f}")
        else:
            print("  [WARMUP] Training only...")

        history["Period"].append(t)
        history["Count_cold"].append(n_cold_t)
        history["Count_hot"].append(n_hot_t)
        for key in metrics_keys:
            history["cold_" + key].append(cold_res.get(key, 0.0))
            history["hot_" + key].append(hot_res.get(key, 0.0))

        _add_user_seen_from_df(user_seen_items, p_df)
        accumulated_dfs.append(p_df)
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = (t >= warmup_periods) and cfg.use_epoch_early_stop and cfg.n_epochs > 1
        es_best = None
        es_best_state = None
        es_best_opt_state = None
        es_no_improve = 0

        for epoch in range(cfg.n_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            steps = 0
            cand_dup_sum = 0.0
            cand_cov_sum = 0.0
            cand_gain_sum = 0.0
            cand_pen_sum = 0.0
            cand_batches = 0
            optimizer.zero_grad()
            cached_user_bank = None
            if cfg.candidate_strategy == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
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
                    cand_gain_sum += cand_info.get("step_gain", 0.0)
                    cand_pen_sum += cand_info.get("collapse_penalty", 0.0)
                    cand_batches += 1

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                avg_gain = cand_gain_sum / cand_batches
                avg_pen = cand_pen_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f}"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if do_early_stop:
                all_item_vecs_es = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="cold", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )
                es_hot, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="hot", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )

                key_n = f"N@{cfg.early_stop_k}"
                key_r = f"R@{cfg.early_stop_k}"
                cur_n = es_cold.get(key_n, 0.0) if es_cold else 0.0
                cur_cr = es_cold.get(key_r, 0.0) if es_cold else 0.0
                cur_hr = es_hot.get(key_r, 0.0) if es_hot else 0.0

                if es_best is None:
                    is_better = True
                else:
                    hot_floor = es_best["hot_r"] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
                    hot_ok = cur_hr >= hot_floor
                    n_improve = cur_n > es_best["cold_n"] + cfg.early_stop_min_delta
                    n_tie = abs(cur_n - es_best["cold_n"]) <= cfg.early_stop_min_delta
                    r_tie_break = cur_cr > es_best["cold_r"] + 1e-12
                    is_better = hot_ok and (n_improve or (n_tie and r_tie_break))

                if is_better:
                    es_best = {
                        "epoch": epoch + 1,
                        "cold_n": float(cur_n),
                        "cold_r": float(cur_cr),
                        "hot_r": float(cur_hr),
                    }
                    es_best_state = copy.deepcopy(model.state_dict())
                    es_best_opt_state = copy.deepcopy(optimizer.state_dict())
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                print(
                    f"  [EARLYSTOP] Epoch {epoch + 1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
                )

                if es_no_improve >= cfg.early_stop_patience:
                    print(f"  [EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    break

        if do_early_stop and es_best_state is not None:
            model.load_state_dict(es_best_state)
            if es_best_opt_state is not None:
                optimizer.load_state_dict(es_best_opt_state)
            print(
                f"  [EARLYSTOP] Restore best epoch={es_best['epoch']} "
                f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
                f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
                f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
            )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Feedback-Aware RL-USIM)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_usim_feedback.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Feedback-Aware RL-USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_usim_feedback.png")
    print(">> Saved mooc_result_usim_feedback.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
