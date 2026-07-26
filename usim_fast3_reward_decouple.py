"""
usim_fast3_reward_decouple.py
改进 3: 冷热解耦课程奖励权重

核心改动:
  课程奖励的 4 个子项 (concept / prereq / difficulty / redundant)
  分别对冷启动物品和热门物品施加不同的缩放系数,
  解决消融实验中冷热指标冲突的问题.

设计思路:
  - 冷启动物品: 放大 concept_bonus (发现相关内容), 降低 prereq 惩罚 (历史少, prereq 意义弱)
  - 热门物品:   放大 prereq/difficulty 惩罚 (历史足够验证), 降低 concept_bonus (已被充分覆盖)

依据论文:
  - CMCLRec (SIGIR 2024): 跨模态冷启动表征解耦, 避免冷热共享导致冲突
  - SwAN (RecSys 2024): 场景自适应的冷启动策略切换

运行方式:
  python usim_fast3_reward_decouple.py

新增环境变量 (均可选, 有默认值):
  DC_COLD_CONCEPT_SCALE    冷启动 concept 权重缩放   (默认 2.0)
  DC_COLD_PREREQ_SCALE     冷启动 prereq 权重缩放    (默认 0.5)
  DC_COLD_DIFF_SCALE       冷启动 difficulty 权重缩放 (默认 0.5)
  DC_COLD_REDUNDANT_SCALE  冷启动 redundant 权重缩放  (默认 0.5)
  DC_HOT_CONCEPT_SCALE     热门 concept 权重缩放      (默认 0.5)
  DC_HOT_PREREQ_SCALE      热门 prereq 权重缩放       (默认 2.0)
  DC_HOT_DIFF_SCALE        热门 difficulty 权重缩放    (默认 1.5)
  DC_HOT_REDUNDANT_SCALE   热门 redundant 权重缩放     (默认 1.5)
"""

import os
import math
import torch
import torch.nn.functional as F

import usim_feedback_fast3_standalone as _base


# ---------------------------------------------------------------------------
# Config: 在 Fast3Config 基础上加入冷热缩放系数
# ---------------------------------------------------------------------------
class DecoupleRewardConfig(_base.Fast3Config):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.dc_cold_concept_scale = float(os.environ.get("DC_COLD_CONCEPT_SCALE", "2.0"))
        self.dc_cold_prereq_scale = float(os.environ.get("DC_COLD_PREREQ_SCALE", "0.5"))
        self.dc_cold_difficulty_scale = float(os.environ.get("DC_COLD_DIFF_SCALE", "0.5"))
        self.dc_cold_redundant_scale = float(os.environ.get("DC_COLD_REDUNDANT_SCALE", "0.5"))
        self.dc_hot_concept_scale = float(os.environ.get("DC_HOT_CONCEPT_SCALE", "0.5"))
        self.dc_hot_prereq_scale = float(os.environ.get("DC_HOT_PREREQ_SCALE", "2.0"))
        self.dc_hot_difficulty_scale = float(os.environ.get("DC_HOT_DIFF_SCALE", "1.5"))
        self.dc_hot_redundant_scale = float(os.environ.get("DC_HOT_REDUNDANT_SCALE", "1.5"))


# ---------------------------------------------------------------------------
# Model: 覆写 run_usim_episode 和 _compute_candidate_course_fit
# ---------------------------------------------------------------------------
class DecoupleRewardUSIM(_base.Fast3FeedbackUSIM):

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        print(
            f">> [Decouple Reward] Cold scales: "
            f"concept={config.dc_cold_concept_scale:.2f}, "
            f"prereq={config.dc_cold_prereq_scale:.2f}, "
            f"diff={config.dc_cold_difficulty_scale:.2f}, "
            f"redundant={config.dc_cold_redundant_scale:.2f}"
        )
        print(
            f">> [Decouple Reward] Hot  scales: "
            f"concept={config.dc_hot_concept_scale:.2f}, "
            f"prereq={config.dc_hot_prereq_scale:.2f}, "
            f"diff={config.dc_hot_difficulty_scale:.2f}, "
            f"redundant={config.dc_hot_redundant_scale:.2f}"
        )

    # ---- helper: 计算每个样本的自适应课程奖励权重 ----
    def _get_decouple_course_weights(self, target_pop, batch_size):
        """返回 (concept_w, prereq_w, diff_w, redundant_w), 每个 shape=(B,1)."""
        if target_pop is not None:
            cold_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        else:
            cold_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        base_cw = float(self.cfg.feedback_course_concept_weight)
        base_pw = float(self.cfg.feedback_course_prereq_weight)
        base_dw = float(self.cfg.feedback_course_difficulty_weight)
        base_rw = float(self.cfg.feedback_course_redundant_weight)

        concept_w = (
            cold_mask * self.cfg.dc_cold_concept_scale
            + (1 - cold_mask) * self.cfg.dc_hot_concept_scale
        ) * base_cw
        prereq_w = (
            cold_mask * self.cfg.dc_cold_prereq_scale
            + (1 - cold_mask) * self.cfg.dc_hot_prereq_scale
        ) * base_pw
        diff_w = (
            cold_mask * self.cfg.dc_cold_difficulty_scale
            + (1 - cold_mask) * self.cfg.dc_hot_difficulty_scale
        ) * base_dw
        redundant_w = (
            cold_mask * self.cfg.dc_cold_redundant_scale
            + (1 - cold_mask) * self.cfg.dc_hot_redundant_scale
        ) * base_rw

        return concept_w, prereq_w, diff_w, redundant_w

    # ---- override: run_usim_episode (唯一改动: 课程奖励权重冷热解耦) ----
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

        # >>> DECOUPLE: 预计算冷热自适应课程奖励权重 (整个 episode 不变)
        dc_cw, dc_pw, dc_dw, dc_rw = self._get_decouple_course_weights(
            target_pop, init_item_emb.size(0)
        )

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
            action_idx, log_prob, value, entropy = self.agent.get_action_value(
                current_h, time_step, candidates
            )

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
                    score = (
                        ((1.0 - target_alpha) * user_align) + (target_alpha * target_align)
                    ).mean()
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
                    collapse_penalty = float(self.cfg.reward_dup_penalty_weight) * float(
                        cand_stats["dup_rate"]
                    )
                    reward = reward - collapse_penalty
                    if float(self.cfg.reward_cov_bonus_weight) > 0.0:
                        reward = reward + float(self.cfg.reward_cov_bonus_weight) * float(
                            cand_stats["topm_coverage"]
                        )

            course_terms = self._compute_course_reward_terms(
                selected_user_ids,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            # >>> DECOUPLE: 用冷热自适应权重替代固定权重
            reward = (
                reward
                + dc_cw * course_terms["concept_bonus"]
                - dc_pw * course_terms["prereq_gap"]
                - dc_dw * course_terms["difficulty_gap"]
                - dc_rw * course_terms["redundant"]
            )
            # <<< DECOUPLE

            candidate_stats["step_gain"] += step_gain_mean
            candidate_stats["collapse_penalty"] += collapse_penalty
            candidate_stats["course_prereq_gap"] += float(course_terms["prereq_gap"].mean().item())
            candidate_stats["course_concept_bonus"] += float(
                course_terms["concept_bonus"].mean().item()
            )
            candidate_stats["course_difficulty_gap"] += float(
                course_terms["difficulty_gap"].mean().item()
            )
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

    # ---- override: _compute_candidate_course_fit (候选采样阶段同步解耦) ----
    def _compute_candidate_course_fit(
        self, candidate_user_idx, item_idx, target_pop=None, user_seen_items=None
    ):
        batch_size, n_cand = candidate_user_idx.shape
        zero = torch.zeros((batch_size, n_cand), dtype=torch.float32, device=self.device)
        if user_seen_items is None or candidate_user_idx is None:
            return zero

        if self.cfg.feedback_course_sample_only_cold and target_pop is not None:
            active = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        else:
            active = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        flat_user_idx = candidate_user_idx.reshape(-1)
        unique_uids, inverse_map = flat_user_idx.unique(return_inverse=True)
        seen_mat_u, seen_cnt_u = self._build_seen_mat(unique_uids, user_seen_items)
        if seen_cnt_u.max().item() < 1:
            return zero
        seen_mat = seen_mat_u[inverse_map]
        seen_cnt_raw = seen_cnt_u[inverse_map]

        flat_item_idx = item_idx.view(-1, 1).expand(-1, n_cand).reshape(-1)
        batch_idx = torch.arange(flat_user_idx.size(0), device=self.device)
        fit = torch.zeros((flat_user_idx.size(0), 1), dtype=torch.float32, device=self.device)

        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)
        prereq_gap, prereq_safe = self._compute_prereq_gap_and_safe(seen_mat, flat_item_idx)

        concept_bonus = torch.zeros_like(fit)
        redundant = torch.zeros_like(fit)
        seen_active = (seen_cnt_raw >= 1.0).float()
        redundant_mode = str(
            getattr(self.cfg, "feedback_course_redundant_mode", "concept")
        ).strip().lower()
        if redundant_mode == "video_family":
            structural_full = self._compute_structural_redundancy_profile(
                flat_user_idx, user_seen_items
            )
            redundant = structural_full[batch_idx, flat_item_idx].unsqueeze(1).clamp(0.0, 1.0)
        if self.item_concept_overlap is not None:
            concept_full = (
                torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            )
            concept_match = concept_full[batch_idx, flat_item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(
                min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min))
            )
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            if redundant_mode != "video_family":
                redundant = (
                    (concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)
                ).clamp(0.0, 1.0)
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant)

        difficulty_gap = torch.zeros_like(fit)
        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[flat_item_idx].unsqueeze(1)
            difficulty_gap = F.relu(item_difficulty - user_readiness)

        # >>> DECOUPLE: 冷热自适应权重 (替代原有固定权重)
        dc_cw, dc_pw, dc_dw, dc_rw = self._get_decouple_course_weights(target_pop, batch_size)
        dc_cw_f = dc_cw.repeat_interleave(n_cand, dim=0)
        dc_pw_f = dc_pw.repeat_interleave(n_cand, dim=0)
        dc_dw_f = dc_dw.repeat_interleave(n_cand, dim=0)
        dc_rw_f = dc_rw.repeat_interleave(n_cand, dim=0)

        fit = (
            dc_cw_f * concept_bonus
            - dc_pw_f * prereq_gap
            - dc_dw_f * difficulty_gap
            - dc_rw_f * redundant
        ) * active.repeat_interleave(n_cand, dim=0)
        # <<< DECOUPLE

        return fit.view(batch_size, n_cand)


# ---------------------------------------------------------------------------
# Entry point: monkey-patch 后调用 standalone 的 main()
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 输出和检查点放到独立目录, 不影响原始实验结果
    os.environ.setdefault("USIM_FB_OUTPUT_TAG", "decouple_reward")
    os.environ.setdefault(
        "USIM_FB_CKPT_DIR", os.path.join("checkpoints", "fast3_decouple_reward")
    )
    os.environ.setdefault("USIM_FB_FORCE_FRESH", "1")

    # Monkey-patch: 让 standalone 的 main() 使用新的 Config 和 Model
    _base.Fast3Config = DecoupleRewardConfig
    _base.Fast3FeedbackUSIM = DecoupleRewardUSIM

    _base.setup_seed(2025)
    _base.main()
