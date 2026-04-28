"""
Standalone FAST3 variant for the ConceptReward-only experiment.

This file is intentionally copied from the FAST3 mainline so the experiment
definition does not drift when the original script changes later.
"""
import copy
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from processed_data_utils import load_llm_scores_for_stream

from usim import (
    StreamDataset,
    _add_user_seen_from_df,
    build_eval_item_vecs,
    build_course_artifacts,
    collate_fn,
    evaluate_usim,
    setup_seed,
    split_dataframe_by_periods,
)
from usim_feedback import (
    FeedbackConfig,
    _build_feedback_ckpt_state,
    _deserialize_user_seen_items,
    _empty_course_stats,
    _feedback_ckpt_auto_resume,
    _feedback_ckpt_enabled,
    _feedback_ckpt_force_fresh,
    _feedback_output_path,
    _format_eta,
    _load_feedback_checkpoint,
    _maybe_clear_cuda_cache,
    _move_state_to_cpu,
    _optimizer_state_to_device,
    _save_final_report_exports,
    _save_feedback_checkpoint,
    _should_log_train_progress,
)
from usim_feedback_fast import FastFeedbackUSIM


class Fast3Config(FeedbackConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)

        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", "2"))
        self.stream_train_window = int(os.environ.get("USIM_TRAIN_WINDOW", "24"))

        self.ppo_lambda = float(os.environ.get("USIM_PPO_LAMBDA", "0.95"))
        self.ppo_value_clip = float(os.environ.get("USIM_PPO_VALUE_CLIP", "0.15"))
        self.ppo_adv_norm = os.environ.get("USIM_PPO_ADV_NORM", "1") == "1"

        self.fast3_target_alpha_cold = float(os.environ.get("USIM_FAST3_TGT_ALPHA_COLD", "0.28"))
        self.fast3_target_alpha_hot = float(os.environ.get("USIM_FAST3_TGT_ALPHA_HOT", "0.58"))
        self.fast3_target_alpha_step = float(os.environ.get("USIM_FAST3_TGT_ALPHA_STEP", "0.15"))
        self.fast3_target_alpha_entropy = float(os.environ.get("USIM_FAST3_TGT_ALPHA_ENT", "0.12"))
        self.fast3_target_alpha_min = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MIN", "0.12"))
        self.fast3_target_alpha_max = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MAX", "0.78"))

        self.feedback_course_sample_soft = os.environ.get("USIM_FB_COURSE_SAMPLE_SOFT", "1") == "1"
        self.feedback_course_sample_top_l = int(
            os.environ.get(
                "USIM_FB_COURSE_SAMPLE_TOPL",
                str(getattr(self, "feedback_course_sample_topk", 32)),
            )
        )
        self.feedback_load_course_artifacts = os.environ.get("USIM_FB_LOAD_COURSE_ARTIFACTS", "1") == "1"
        self.use_prereq_aux_loss = os.environ.get("USIM_USE_PREREQ_AUX_LOSS", "0") == "1"
        self.use_course_rerank = os.environ.get("USIM_USE_COURSE_RERANK", "0") == "1"
        self.use_structured_hard_neg = os.environ.get("USIM_USE_STRUCTURED_HARD_NEG", "0") == "1"
        self.feedback_course_prereq_weight = float(os.environ.get("USIM_FB_COURSE_PREREQ_W", "0.0"))
        self.feedback_course_concept_weight = float(os.environ.get("USIM_FB_COURSE_CONCEPT_W", "0.04"))
        self.feedback_course_difficulty_weight = float(os.environ.get("USIM_FB_COURSE_DIFF_W", "0.0"))
        self.feedback_course_redundant_weight = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_W", "0.0"))
        self.feedback_course_sample_beta = float(os.environ.get("USIM_FB_COURSE_SAMPLE_BETA", "0.0"))
        self.feedback_course_only_cold = os.environ.get("USIM_FB_COURSE_ONLY_COLD", "1") == "1"
        self.feedback_course_sample_only_cold = os.environ.get("USIM_FB_COURSE_SAMPLE_ONLY_COLD", "1") == "1"
        self.feedback_course_sample_topk = int(os.environ.get("USIM_FB_COURSE_SAMPLE_TOPK", "32"))
        self.feedback_course_sample_top_l = int(
            os.environ.get("USIM_FB_COURSE_SAMPLE_TOPL", "16")
        )


def _feedback_ckpt_dir():
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "fast3_concept_reward"))


def _apply_default_run_env():
    os.environ.setdefault("USIM_FB_FORCE_FRESH", "1")
    os.environ.setdefault("USIM_FB_AUTO_RESUME", "0")
    os.environ.setdefault("USIM_FB_OUTPUT_TAG", "fast3_concept_reward")
    os.environ.setdefault("USIM_FB_OUTPUT_DIR", os.path.join("outputs", "fast3_concept_reward"))
    os.environ.setdefault("USIM_FB_CKPT_DIR", _feedback_ckpt_dir())


class Fast3FeedbackUSIM(FastFeedbackUSIM):
    def _compute_target_alpha(self, target_pop, step_idx, entropy, num_candidates, batch_size):
        if target_pop is not None:
            cold_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        else:
            cold_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        alpha = (
            cold_mask * float(self.cfg.fast3_target_alpha_cold)
            + (1.0 - cold_mask) * float(self.cfg.fast3_target_alpha_hot)
        )

        if self.cfg.usim_steps > 1:
            progress = float(step_idx) / float(max(1, self.cfg.usim_steps - 1))
            alpha = alpha + float(self.cfg.fast3_target_alpha_step) * progress

        if entropy is not None and num_candidates > 1:
            max_entropy = max(1e-6, math.log(float(num_candidates)))
            entropy_norm = (entropy.detach().unsqueeze(1) / max_entropy).clamp(0.0, 1.0)
            alpha = alpha - float(self.cfg.fast3_target_alpha_entropy) * entropy_norm

        return alpha.clamp(
            min=float(self.cfg.fast3_target_alpha_min),
            max=float(self.cfg.fast3_target_alpha_max),
        )

    def _apply_course_sampling_bias(
        self,
        state_emb,
        candidates,
        cand_user_idx,
        item_idx,
        target_pop=None,
        user_seen_items=None,
    ):
        if (
            candidates is None
            or cand_user_idx is None
            or float(self.cfg.feedback_course_sample_beta) <= 0.0
        ):
            return candidates, cand_user_idx, None

        fit_score = self._compute_candidate_course_fit(
            cand_user_idx,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        if not torch.isfinite(fit_score).all():
            fit_score = torch.nan_to_num(fit_score, nan=0.0, posinf=0.0, neginf=0.0)

        if not getattr(self.cfg, "feedback_course_sample_soft", True):
            order = torch.argsort(fit_score, dim=1, descending=True)
            candidates = candidates.gather(1, order.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
            cand_user_idx = cand_user_idx.gather(1, order)
            fit_score = fit_score.gather(1, order)
            return candidates, cand_user_idx, fit_score

        batch_size, n_cand = cand_user_idx.shape
        top_l_cfg = int(getattr(self.cfg, "feedback_course_sample_top_l", 0))
        if top_l_cfg <= 0 or top_l_cfg >= n_cand:
            top_l = n_cand
        else:
            top_l = max(1, top_l_cfg)

        retrieval_score = (F.normalize(state_emb, dim=1).unsqueeze(1) * F.normalize(candidates, dim=2)).sum(dim=2)
        fit_scale = fit_score.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        fit_norm = fit_score / fit_scale
        combined_score = retrieval_score + float(self.cfg.feedback_course_sample_beta) * fit_norm

        base_order = torch.argsort(retrieval_score, dim=1, descending=True)
        top_idx = base_order[:, :top_l]
        rest_idx = base_order[:, top_l:]
        top_combined = combined_score.gather(1, top_idx)
        top_reorder = torch.argsort(top_combined, dim=1, descending=True)
        top_idx = top_idx.gather(1, top_reorder)
        final_order = torch.cat([top_idx, rest_idx], dim=1)

        candidates = candidates.gather(1, final_order.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        cand_user_idx = cand_user_idx.gather(1, final_order)
        fit_score = fit_score.gather(1, final_order)
        return candidates, cand_user_idx, fit_score

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
    ):
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
            "course_sample_fit": 0.0,
            "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0,
            "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
            "target_alpha": 0.0,
        }

        user_bank_norm = None
        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        elif user_bank_raw is not None and user_bank_norm is None:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, cand_user_idx, cand_stats = self.get_candidates(
                current_h,
                user_bank_raw=user_bank_raw,
                user_bank_norm=user_bank_norm,
            )
            candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                current_h,
                candidates,
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            if cand_stats is not None:
                candidate_stats["dup_rate"] += cand_stats["dup_rate"]
                candidate_stats["topm_coverage"] += cand_stats["topm_coverage"]
                candidate_stats["steps"] += 1
            if fit_score is not None:
                candidate_stats["course_sample_fit"] += float(fit_score.mean().item())

            trajectory["states"].append(current_h.detach().clone())
            trajectory["time_steps"].append(time_step.detach().clone())
            trajectory["candidates"].append(candidates.detach().clone())
            trajectory["actions"].append(action_idx.detach().clone())

            prev_h = current_h
            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]
            selected_user_ids = None
            if cand_user_idx is not None:
                selected_user_ids = cand_user_idx[batch_indices, action_idx]

            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                user_align = (h_detached * selected_user.detach()).sum(dim=1, keepdim=True)
                if target_emb is not None:
                    target_align = (h_detached * target_emb.detach()).sum(dim=1, keepdim=True)
                    target_alpha = self._compute_target_alpha(
                        target_pop=target_pop,
                        step_idx=t,
                        entropy=entropy,
                        num_candidates=candidates.size(1),
                        batch_size=current_h.size(0),
                    )
                    candidate_stats["target_alpha"] += float(target_alpha.mean().item())
                    score = (((1.0 - target_alpha) * user_align) + (target_alpha * target_align)).mean()
                else:
                    score = user_align.mean()
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

            course_terms = self._compute_course_reward_terms(
                selected_user_ids,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            reward = (
                reward
                + float(self.cfg.feedback_course_concept_weight) * course_terms["concept_bonus"]
                - float(self.cfg.feedback_course_prereq_weight) * course_terms["prereq_gap"]
                - float(self.cfg.feedback_course_difficulty_weight) * course_terms["difficulty_gap"]
                - float(self.cfg.feedback_course_redundant_weight) * course_terms["redundant"]
            )

            candidate_stats["step_gain"] += step_gain_mean
            candidate_stats["collapse_penalty"] += collapse_penalty
            candidate_stats["course_prereq_gap"] += float(course_terms["prereq_gap"].mean().item())
            candidate_stats["course_concept_bonus"] += float(course_terms["concept_bonus"].mean().item())
            candidate_stats["course_difficulty_gap"] += float(course_terms["difficulty_gap"].mean().item())
            candidate_stats["course_redundant"] += float(course_terms["redundant"].mean().item())
            trajectory["log_probs"].append(log_prob.detach())
            trajectory["values"].append(value.detach())
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

        if candidate_stats["steps"] > 0:
            for key in [
                "dup_rate",
                "topm_coverage",
                "step_gain",
                "collapse_penalty",
                "course_sample_fit",
                "course_prereq_gap",
                "course_concept_bonus",
                "course_difficulty_gap",
                "course_redundant",
                "target_alpha",
            ]:
                candidate_stats[key] /= candidate_stats["steps"]

        return current_h, trajectory, candidate_stats

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory["rewards"]).squeeze(-1)
        old_log_probs = torch.stack(trajectory["log_probs"])
        old_values = torch.stack(trajectory["values"]).squeeze(-1)
        states = trajectory["states"]
        time_steps = trajectory["time_steps"]
        candidates = trajectory["candidates"]
        actions = trajectory["actions"]

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        next_value = torch.zeros_like(old_values[0])
        gamma = float(self.cfg.ppo_gamma)
        lam = float(getattr(self.cfg, "ppo_lambda", 0.95))

        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + gamma * next_value - old_values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            next_value = old_values[t]

        returns = advantages + old_values
        if getattr(self.cfg, "ppo_adv_norm", False):
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
            advantages = (advantages - adv_mean) / adv_std

        total_ppo_loss = 0.0
        value_clip = float(getattr(self.cfg, "ppo_value_clip", 0.0))

        for _ in range(self.cfg.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            new_entropies_list = []

            for t in range(len(states)):
                _, new_log_prob, new_value, new_entropy = self.agent.get_action_value(
                    states[t],
                    time_steps[t],
                    candidates[t],
                    action_idx=actions[t],
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                new_entropies_list.append(new_entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze(-1)
            new_entropies = torch.stack(new_entropies_list)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            if value_clip > 0.0:
                value_delta = (new_values - old_values).clamp(-value_clip, value_clip)
                value_pred_clipped = old_values + value_delta
                critic_unclipped = (new_values - returns.detach()).pow(2)
                critic_clipped = (value_pred_clipped - returns.detach()).pow(2)
                critic_loss = 0.5 * torch.max(critic_unclipped, critic_clipped).mean()
            else:
                critic_loss = 0.5 * (new_values - returns.detach()).pow(2).mean()

            entropy_loss = -new_entropies.mean()
            total_ppo_loss += (
                actor_loss
                + self.cfg.ppo_coeffs["value"] * critic_loss
                + self.cfg.ppo_coeffs["entropy"] * entropy_loss
            )

        return total_ppo_loss / self.cfg.ppo_epochs


def main():
    _apply_default_run_env()
    data_dir = "processed_data_hin"
    print(f"Loading Data for Feedback-Aware USIM (FAST3 ConceptReward) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    llm_scores, llm_score_path, _ = load_llm_scores_for_stream(
        data_dir,
        df,
        cold_threshold=5,
        n_users=meta.get("n_users"),
        n_items=meta.get("n_items"),
        fallback_data_dirs=["processed_data"],
    )
    content_emb = torch.load(f"{data_dir}/content_emb.pt")
    if llm_score_path:
        print(f"   LLM scores loaded from {llm_score_path}")

    cfg = Fast3Config(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = build_course_artifacts(
            df,
            cfg.n_items,
            relation_dir="MOOCCube/relations",
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, course_stats = None, _empty_course_stats(cfg.n_items)
    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    pop_stats = df.groupby("i_idx")["popularity"].max()
    for item_id, pop_value in pop_stats.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_final_pop[idx] = int(pop_value)

    model = Fast3FeedbackUSIM(cfg, content_emb).to(device)
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Feedback-Aware RL-USIM + InfoNCE [FAST3 ConceptReward] (Batch Size={cfg.batch_size})")
    print(
        f">> Window={cfg.stream_train_window} | PPO epochs={cfg.ppo_epochs} | "
        f"lambda={cfg.ppo_lambda:.2f} | value_clip={cfg.ppo_value_clip:.2f} | "
        f"adv_norm={cfg.ppo_adv_norm}"
    )
    print(
        f">> Adaptive Mix: cold={cfg.fast3_target_alpha_cold:.2f} | "
        f"hot={cfg.fast3_target_alpha_hot:.2f} | step_gain={cfg.fast3_target_alpha_step:.2f} | "
        f"entropy_pen={cfg.fast3_target_alpha_entropy:.2f}"
    )
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> LLM Injection: safe_mode={cfg.llm_safe_mode} | weight={cfg.llm_weight:.2f} | "
        f"cold_only={cfg.llm_cold_only} | bank_mode={cfg.llm_bank_mode}"
    )
    print(
        f">> Course Soft Rerank: enabled={cfg.feedback_course_sample_soft} | "
        f"beta={cfg.feedback_course_sample_beta:.2f} | topL={cfg.feedback_course_sample_top_l}"
    )
    print(
        f">> Course Artifacts: enabled={cfg.feedback_load_course_artifacts} | "
        f"prereq_aux={cfg.use_prereq_aux_loss} | "
        f"rerank={cfg.use_course_rerank} | "
        f"struct_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

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
    ckpt_dir = _feedback_ckpt_dir()
    ckpt_enabled = _feedback_ckpt_enabled()
    auto_resume = _feedback_ckpt_auto_resume()
    force_fresh = _feedback_ckpt_force_fresh()
    print(
        f">> Checkpoint: enabled={ckpt_enabled} | auto_resume={auto_resume} | "
        f"force_fresh={force_fresh} | dir={ckpt_dir}"
    )

    start_period = 0
    resume_current_period = None
    resume_next_epoch = 0
    resume_accumulated_periods = 0
    resume_es_best = None
    resume_es_best_state = None
    resume_es_best_opt_state = None
    resume_es_no_improve = 0

    if ckpt_enabled and auto_resume and not force_fresh:
        resume_state = _load_feedback_checkpoint(ckpt_dir)
        if resume_state is not None:
            status = resume_state.get("status", "between_periods")
            if status == "finished":
                print(">> Resume: found finished checkpoint. Set USIM_FB_FORCE_FRESH=1 to start over.")
                return
            total_periods_saved = int(resume_state.get("total_periods", len(periods)))
            if total_periods_saved != len(periods):
                print(f">> Resume skipped: checkpoint total_periods={total_periods_saved}, current={len(periods)}")
            else:
                model.load_state_dict(resume_state["model_state"])
                optimizer.load_state_dict(resume_state["optimizer_state"])
                _optimizer_state_to_device(optimizer, device)
                history = copy.deepcopy(resume_state.get("history", history))
                accum_cold = copy.deepcopy(resume_state.get("accum_cold", accum_cold))
                accum_hot = copy.deepcopy(resume_state.get("accum_hot", accum_hot))
                count_cold = int(resume_state.get("count_cold", count_cold))
                count_hot = int(resume_state.get("count_hot", count_hot))
                full_cold = copy.deepcopy(resume_state.get("full_cold", full_cold))
                full_hot = copy.deepcopy(resume_state.get("full_hot", full_hot))
                fc_cold = int(resume_state.get("fc_cold", fc_cold))
                fc_hot = int(resume_state.get("fc_hot", fc_hot))
                user_seen_items = _deserialize_user_seen_items(resume_state.get("user_seen_items"))
                warmup_periods = int(resume_state.get("warmup_periods", warmup_periods))
                resume_accumulated_periods = int(resume_state.get("accumulated_periods", 0))
                accumulated_dfs = periods[:resume_accumulated_periods]
                start_period = int(resume_state.get("next_period", 0))
                resume_current_period = resume_state.get("current_period")
                if resume_current_period is not None:
                    resume_current_period = int(resume_current_period)
                    start_period = resume_current_period
                resume_next_epoch = int(resume_state.get("next_epoch", 0))
                resume_es_best = copy.deepcopy(resume_state.get("es_best"))
                resume_es_best_state = _move_state_to_cpu(resume_state.get("es_best_state"))
                resume_es_best_opt_state = _move_state_to_cpu(resume_state.get("es_best_opt_state"))
                resume_es_no_improve = int(resume_state.get("es_no_improve", 0))
                print(
                    f">> Resume: status={status} | start_period={start_period} | "
                    f"resume_current_period={resume_current_period} | next_epoch={resume_next_epoch} | "
                    f"accumulated_periods={resume_accumulated_periods}"
                )

    for t in range(start_period, len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (current={n_total}, accumulated={sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0
        resume_this_period = resume_current_period is not None and t == resume_current_period

        if resume_this_period:
            print(f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs}")
        elif t >= warmup_periods:
            print("  [EVAL-START] Build eval item bank and run sampled/full ranking...")
            all_item_vecs_eval = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, n_cold_t = evaluate_usim(
                model,
                eval_loader,
                device,
                llm_scores,
                k_list,
                n_neg=cfg.eval_n_neg,
                eval_type="cold",
                user_seen_items=user_seen_items,
                all_item_vecs=all_item_vecs_eval,
            )
            met_hot, n_hot_t = evaluate_usim(
                model,
                eval_loader,
                device,
                llm_scores,
                k_list,
                n_neg=cfg.eval_n_neg,
                eval_type="hot",
                user_seen_items=user_seen_items,
                all_item_vecs=all_item_vecs_eval,
            )
            fmet_cold, fn_c = evaluate_usim(
                model,
                eval_loader,
                device,
                llm_scores,
                k_list,
                eval_type="cold",
                full_ranking=True,
                user_seen_items=user_seen_items,
                all_item_vecs=all_item_vecs_eval,
            )
            fmet_hot, fn_h = evaluate_usim(
                model,
                eval_loader,
                device,
                llm_scores,
                k_list,
                eval_type="hot",
                full_ranking=True,
                user_seen_items=user_seen_items,
                all_item_vecs=all_item_vecs_eval,
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
            print(f"  Sampled Cold={c_s:.4f} Hot={h_s:.4f} | Full Cold={c_f:.4f} Hot={h_f:.4f}")
            del all_item_vecs_eval
            _maybe_clear_cuda_cache()
        else:
            print("  [WARMUP] Training only...")

        if not resume_this_period:
            history["Period"].append(t)
            history["Count_cold"].append(n_cold_t)
            history["Count_hot"].append(n_hot_t)
            for key in metrics_keys:
                history["cold_" + key].append(cold_res.get(key, 0.0))
                history["hot_" + key].append(hot_res.get(key, 0.0))
            _add_user_seen_from_df(user_seen_items, p_df)
            accumulated_dfs.append(p_df)

        window = cfg.stream_train_window
        if window > 0 and len(accumulated_dfs) > window:
            train_dfs = accumulated_dfs[-window:]
            print(
                f"  [WINDOW] Use latest {window}/{len(accumulated_dfs)} periods for training "
                f"({sum(len(d) for d in train_dfs)} samples)"
            )
        else:
            train_dfs = accumulated_dfs

        combined_df = pd.concat(train_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = t >= warmup_periods and cfg.use_epoch_early_stop and cfg.n_epochs > 1
        es_best = copy.deepcopy(resume_es_best) if resume_this_period else None
        es_best_state = copy.deepcopy(resume_es_best_state) if resume_this_period else None
        es_best_opt_state = copy.deepcopy(resume_es_best_opt_state) if resume_this_period else None
        es_no_improve = int(resume_es_no_improve) if resume_this_period else 0
        epoch_start_idx = resume_next_epoch if resume_this_period else 0

        if ckpt_enabled and not resume_this_period:
            period_start_state = _build_feedback_ckpt_state(
                model,
                optimizer,
                history,
                accum_cold,
                accum_hot,
                count_cold,
                count_hot,
                full_cold,
                full_hot,
                fc_cold,
                fc_hot,
                user_seen_items,
                accumulated_periods=t + 1,
                warmup_periods=warmup_periods,
                total_periods=len(periods),
                status="in_period",
                next_period=t,
                current_period=t,
                next_epoch=0,
            )
            _save_feedback_checkpoint(ckpt_dir, period_start_state)

        for epoch in range(epoch_start_idx, cfg.n_epochs):
            epoch_start = time.time()
            num_batches = max(1, len(train_loader))
            total_loss = 0.0
            steps = 0
            cand_dup_sum = 0.0
            cand_cov_sum = 0.0
            cand_gain_sum = 0.0
            cand_pen_sum = 0.0
            cand_mix_sum = 0.0
            course_sample_fit_sum = 0.0
            course_prereq_sum = 0.0
            course_concept_sum = 0.0
            course_diff_sum = 0.0
            course_redundant_sum = 0.0
            cand_batches = 0
            optimizer.zero_grad()

            cached_user_bank = None
            if cfg.candidate_strategy == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()

            print(
                f"  [TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Period {t + 1}/{len(periods)} | samples={len(combined_df)} | batches={num_batches}"
            )
            last_progress_log = epoch_start

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
                if (
                    cached_user_bank is not None
                    and cfg.user_bank_refresh_steps > 0
                    and batch_idx > 0
                    and batch_idx % cfg.user_bank_refresh_steps == 0
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
                    cand_mix_sum += cand_info.get("target_alpha", 0.0)
                    course_sample_fit_sum += cand_info.get("course_sample_fit", 0.0)
                    course_prereq_sum += cand_info.get("course_prereq_gap", 0.0)
                    course_concept_sum += cand_info.get("course_concept_bonus", 0.0)
                    course_diff_sum += cand_info.get("course_difficulty_gap", 0.0)
                    course_redundant_sum += cand_info.get("course_redundant", 0.0)
                    cand_batches += 1

                now_ts = time.time()
                if _should_log_train_progress(batch_idx, num_batches, cfg, last_progress_log, now_ts):
                    done = batch_idx + 1
                    elapsed = now_ts - epoch_start
                    avg_batch_sec = elapsed / max(1, done)
                    eta = avg_batch_sec * max(0, num_batches - done)
                    pct = 100.0 * done / max(1, num_batches)
                    print(
                        f"    [TRAIN-PROGRESS] {done}/{num_batches} ({pct:.0f}%) | "
                        f"avg_loss={total_loss / max(1, steps):.4f} | "
                        f"elapsed={_format_eta(elapsed)} | eta={_format_eta(eta)}"
                    )
                    last_progress_log = now_ts

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                avg_gain = cand_gain_sum / cand_batches
                avg_pen = cand_pen_sum / cand_batches
                avg_mix = cand_mix_sum / cand_batches
                avg_csf = course_sample_fit_sum / cand_batches
                avg_cp = course_prereq_sum / cand_batches
                avg_cc = course_concept_sum / cand_batches
                avg_cd = course_diff_sum / cand_batches
                avg_cr = course_redundant_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"MixAlpha: {avg_mix:.4f} | SampleFit: {avg_csf:.4f} | "
                    f"Course[p={avg_cp:.4f}, c={avg_cc:.4f}, d={avg_cd:.4f}, r={avg_cr:.4f}]"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if ckpt_enabled:
                epoch_state = _build_feedback_ckpt_state(
                    model,
                    optimizer,
                    history,
                    accum_cold,
                    accum_hot,
                    count_cold,
                    count_hot,
                    full_cold,
                    full_hot,
                    fc_cold,
                    fc_hot,
                    user_seen_items,
                    accumulated_periods=t + 1,
                    warmup_periods=warmup_periods,
                    total_periods=len(periods),
                    status="in_period",
                    next_period=t,
                    current_period=t,
                    next_epoch=epoch + 1,
                    es_best=es_best,
                    es_best_state=es_best_state,
                    es_best_opt_state=es_best_opt_state,
                    es_no_improve=es_no_improve,
                )
                _save_feedback_checkpoint(ckpt_dir, epoch_state)

            if do_early_stop:
                print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
                all_item_vecs_es = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim(
                    model,
                    eval_loader,
                    device,
                    llm_scores,
                    k_list,
                    eval_type="cold",
                    full_ranking=True,
                    user_seen_items=user_seen_items,
                    all_item_vecs=all_item_vecs_es,
                )
                es_hot, _ = evaluate_usim(
                    model,
                    eval_loader,
                    device,
                    llm_scores,
                    k_list,
                    eval_type="hot",
                    full_ranking=True,
                    user_seen_items=user_seen_items,
                    all_item_vecs=all_item_vecs_es,
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
                    es_best_state = _move_state_to_cpu(model.state_dict())
                    es_best_opt_state = _move_state_to_cpu(optimizer.state_dict())
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                print(
                    f"  [EARLYSTOP] Epoch {epoch + 1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
                )
                del all_item_vecs_es, es_cold, es_hot
                _maybe_clear_cuda_cache()

                if ckpt_enabled:
                    _save_feedback_checkpoint(
                        ckpt_dir,
                        _build_feedback_ckpt_state(
                            model,
                            optimizer,
                            history,
                            accum_cold,
                            accum_hot,
                            count_cold,
                            count_hot,
                            full_cold,
                            full_hot,
                            fc_cold,
                            fc_hot,
                            user_seen_items,
                            accumulated_periods=t + 1,
                            warmup_periods=warmup_periods,
                            total_periods=len(periods),
                            status="in_period",
                            next_period=t,
                            current_period=t,
                            next_epoch=epoch + 1,
                            es_best=es_best,
                            es_best_state=es_best_state,
                            es_best_opt_state=es_best_opt_state,
                            es_no_improve=es_no_improve,
                        ),
                    )

                if es_no_improve >= cfg.early_stop_patience:
                    print(f"  [EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    break

        if do_early_stop and es_best_state is not None:
            model.load_state_dict(es_best_state)
            if es_best_opt_state is not None:
                optimizer.load_state_dict(es_best_opt_state)
                _optimizer_state_to_device(optimizer, device)
            print(
                f"  [EARLYSTOP] Restore best epoch={es_best['epoch']} "
                f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
                f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
                f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
            )
            _maybe_clear_cuda_cache()

        if ckpt_enabled:
            _save_feedback_checkpoint(
                ckpt_dir,
                _build_feedback_ckpt_state(
                    model,
                    optimizer,
                    history,
                    accum_cold,
                    accum_hot,
                    count_cold,
                    count_hot,
                    full_cold,
                    full_hot,
                    fc_cold,
                    fc_hot,
                    user_seen_items,
                    accumulated_periods=t + 1,
                    warmup_periods=warmup_periods,
                    total_periods=len(periods),
                    status="between_periods",
                    next_period=t + 1,
                ),
            )

        if resume_this_period:
            resume_current_period = None
            resume_next_epoch = 0
            resume_es_best = None
            resume_es_best_state = None
            resume_es_best_opt_state = None
            resume_es_no_improve = 0

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (RL-USIM FAST3 ConceptReward)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    summary_rows = []
    sampled_row = {"Model": "USIM-Feedback-FAST3-ConceptReward", "Eval": "sampled", "ColdSamples": count_cold, "HotSamples": count_hot}
    full_row = {"Model": "USIM-Feedback-FAST3-ConceptReward", "Eval": "full_rank", "ColdSamples": fc_cold, "HotSamples": fc_hot}
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
        sampled_row[f"Cold_{key}"] = sc
        sampled_row[f"Hot_{key}"] = sh
        full_row[f"Cold_{key}"] = fc
        full_row[f"Hot_{key}"] = fh
    print("-" * 90)
    print(f"Sampled Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"Full Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)
    summary_rows.extend([sampled_row, full_row])

    final_sampled_cold = {key: (accum_cold[key] / count_cold if count_cold > 0 else 0.0) for key in metrics_keys}
    final_sampled_hot = {key: (accum_hot[key] / count_hot if count_hot > 0 else 0.0) for key in metrics_keys}
    final_full_cold = {key: (full_cold[key] / fc_cold if fc_cold > 0 else 0.0) for key in metrics_keys}
    final_full_hot = {key: (full_hot[key] / fc_hot if fc_hot > 0 else 0.0) for key in metrics_keys}
    detail_path, fullrank_path = _save_final_report_exports(
        protocol="stream",
        metrics_keys=metrics_keys,
        sampled_cold=final_sampled_cold,
        sampled_hot=final_sampled_hot,
        full_cold=final_full_cold,
        full_hot=final_full_hot,
        sampled_cold_count=count_cold,
        sampled_hot_count=count_hot,
        full_cold_count=fc_cold,
        full_hot_count=fc_hot,
        model_name="USIM-Feedback-FAST3-ConceptReward",
    )

    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_concept_reward.csv")
    summary_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_concept_reward_summary.csv")
    plot_path = _feedback_output_path("mooc_result_usim_feedback_fast3_concept_reward.png")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("RL-USIM [FAST3]: ConceptReward only")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(plot_path)
    print(f">> Saved {plot_path}, {metrics_path}, {summary_path}, {detail_path}, and {fullrank_path}")

    if ckpt_enabled:
        _save_feedback_checkpoint(
            ckpt_dir,
            _build_feedback_ckpt_state(
                model,
                optimizer,
                history,
                accum_cold,
                accum_hot,
                count_cold,
                count_hot,
                full_cold,
                full_hot,
                fc_cold,
                fc_hot,
                user_seen_items,
                accumulated_periods=len(periods),
                warmup_periods=warmup_periods,
                total_periods=len(periods),
                status="finished",
                next_period=len(periods),
            ),
            snapshot_name="finished.pt",
        )


if __name__ == "__main__":
    setup_seed(2025)
    main()
