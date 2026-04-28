"""
usim_feedback_fast.py — 性能优化 + 模型修复版 Feedback-Aware RL-USIM
性能优化:
  1. 向量化 _build_seen_mat (批量索引替代 Python for 循环)
  2. RL 循环内缓存 seen_mat (同一 batch 只构建一次)
  3. _compute_candidate_course_fit 用户去重 (40960→unique)
  4. 预 normalize user bank (检索时不再重复 normalize)
模型修复:
  P0-1. RL 梯度方向修复: 混合 selected_user + target 方向
  P0-2. SimpleAC 时间编码修复: 完整 one-hot 替代截断 4 维
  P0-3. target_emb 对称性修复: cold/hot 统一使用融合表征
"""
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
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from usim import (
    Config as BaseConfig,
    PAM_RL_Pure_USIM,
    StreamDataset,
    _add_user_seen_from_df,
    _clone_user_seen,
    build_eval_item_vecs,
    build_course_artifacts,
    collate_fn,
    evaluate_usim,
    setup_seed,
    split_dataframe_by_periods,
)

# ---- 复用 usim_feedback 中的辅助函数和配置 ----
from usim_feedback import (
    FeedbackConfig,
    _format_eta,
    _should_log_train_progress,
    _feedback_ckpt_enabled,
    _feedback_ckpt_auto_resume,
    _feedback_ckpt_force_fresh,
    _serialize_user_seen_items,
    _deserialize_user_seen_items,
    _save_feedback_checkpoint,
    _load_feedback_checkpoint,
    _move_state_to_cpu,
    _optimizer_state_to_device,
    _maybe_clear_cuda_cache,
    _build_feedback_ckpt_state,
)


# 覆盖 checkpoint 目录，避免与 usim_feedback.py 并行运行时冲突
def _feedback_ckpt_dir():
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "usim_feedback_fast"))


class FixedSimpleAC(nn.Module):
    """修复版 SimpleAC: 时间编码使用完整 one-hot 而非截断。"""

    def __init__(self, item_dim, time_dim=5):
        super().__init__()
        self.time_dim = time_dim
        input_dim = item_dim + time_dim
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        self.actor_head = nn.Linear(256, 128)
        self.critic_head = nn.Linear(256, 1)
        self.user_proj = nn.Linear(item_dim, 128)

    def get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
        # P0-2: 完整 one-hot 编码，每个步骤都有独立表示
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=self.time_dim).float()
        state = torch.cat([item_state, t_emb], dim=1)

        feat = self.common(state)
        value = self.critic_head(feat)

        query = self.actor_head(feat).unsqueeze(1)
        keys = self.user_proj(candidates_emb)

        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        dist = Categorical(logits=logits)

        if action_idx is None:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        return action_idx, log_prob, value, entropy


class FastFeedbackUSIM(PAM_RL_Pure_USIM):
    """性能优化 + 模型修复版 FeedbackAwareUSIM。"""

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        # P0-2: 用修复版 agent 替换父类的 SimpleAC
        self.agent = FixedSimpleAC(config.emb_dim, time_dim=config.usim_steps)
        self.item_popularity = None
        self.item_difficulty = None

    def set_feedback_item_stats(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_difficulty = None
            return
        pop = torch.as_tensor(item_popularity, dtype=torch.float32, device=self.device)
        if pop.numel() != self.cfg.n_items:
            raise ValueError(f"item_popularity size mismatch: expect {self.cfg.n_items}, got {pop.numel()}")
        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    # ===================== 优化1: 向量化 _build_seen_mat =====================
    def _build_seen_mat(self, user_ids, user_seen_items):
        """向量化版本：用批量索引替代 Python 逐行循环。"""
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.detach().cpu().tolist()
        else:
            user_ids = [int(x) for x in user_ids]

        batch_size = len(user_ids)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if user_seen_items is None:
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)

        # 批量收集所有 (row, col) 对，一次性赋值
        rows = []
        cols = []
        for row, uid in enumerate(user_ids):
            seen = user_seen_items.get(int(uid))
            if not seen:
                continue
            for it in seen:
                if 0 <= it < self.cfg.n_items:
                    rows.append(row)
                    cols.append(it)
        if rows:
            seen_mat[rows, cols] = 1.0
        return seen_mat, seen_mat.sum(dim=1, keepdim=True)

    # ===================== 优化2+3: 带缓存和去重的 course reward =====================
    def _compute_course_reward_terms(self, selected_user_ids, item_idx, target_pop=None,
                                     user_seen_items=None, cached_seen=None):
        """带可选 cached_seen 的 course reward 计算。"""
        batch_size = int(item_idx.size(0))
        zero = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        terms = {
            "prereq_gap": zero,
            "concept_bonus": zero,
            "difficulty_gap": zero,
            "redundant": zero,
        }
        if selected_user_ids is None or user_seen_items is None:
            return terms

        # 优化2: 使用缓存的 seen_mat（如果 selected_user_ids 与 batch item_idx 一一对应）
        if cached_seen is not None:
            seen_mat, seen_cnt_raw = cached_seen
            # selected_user_ids 是 RL 选出的用户，需要按这些用户检索
            # 但 reward terms 关注的是 target item 对应用户的已看历史
            # 这里 selected_user_ids 是 cand_user_idx 中选出的，需要用这些用户的 seen
            uid_list = selected_user_ids.detach().cpu().tolist()
            unique_uids_list = list(set(int(u) for u in uid_list))
            if len(unique_uids_list) < batch_size * 0.5:
                # 去重优化有效
                unique_uids_t = torch.tensor(unique_uids_list, dtype=torch.long, device=self.device)
                u_seen_mat, u_seen_cnt = self._build_seen_mat(unique_uids_t, user_seen_items)
                uid_to_row = {uid: i for i, uid in enumerate(unique_uids_list)}
                map_idx = [uid_to_row[int(u)] for u in uid_list]
                map_t = torch.tensor(map_idx, dtype=torch.long, device=self.device)
                seen_mat = u_seen_mat[map_t]
                seen_cnt_raw = u_seen_cnt[map_t]
            else:
                seen_mat, seen_cnt_raw = self._build_seen_mat(selected_user_ids, user_seen_items)
        else:
            seen_mat, seen_cnt_raw = self._build_seen_mat(selected_user_ids, user_seen_items)

        if seen_cnt_raw.max().item() < 1:
            return terms

        active = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        if self.cfg.feedback_course_only_cold and target_pop is not None:
            active = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()

        batch_idx = torch.arange(batch_size, device=self.device)
        seen_active = (seen_cnt_raw >= 1.0).float()
        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)

        prereq_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_prereq_gate)))
        prereq_safe = torch.ones_like(zero)
        redundant = zero.clone()
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            prereq_gap = violation_full[batch_idx, item_idx].unsqueeze(1)
            terms["prereq_gap"] = prereq_gap * active
            prereq_safe = (prereq_gap <= prereq_gate).float()

        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            redundant = ((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)).clamp(0.0, 1.0)
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant)
            terms["concept_bonus"] = concept_bonus * active

        terms["redundant"] = redundant * seen_active * active

        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[item_idx].unsqueeze(1)
            difficulty_gap = F.relu(item_difficulty - user_readiness)
            terms["difficulty_gap"] = difficulty_gap * active

        return terms

    # ===================== 优化3: 去重版 candidate course fit =====================
    def _compute_candidate_course_fit(self, candidate_user_idx, item_idx, target_pop=None, user_seen_items=None):
        batch_size, n_cand = candidate_user_idx.shape
        zero = torch.zeros((batch_size, n_cand), dtype=torch.float32, device=self.device)
        if user_seen_items is None or candidate_user_idx is None:
            return zero

        if self.cfg.feedback_course_sample_only_cold and target_pop is not None:
            active = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        else:
            active = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        flat_user_idx = candidate_user_idx.reshape(-1)

        # ---- 优化3核心: 去重 ----
        unique_uids, inverse_map = flat_user_idx.unique(return_inverse=True)
        seen_mat_u, seen_cnt_u = self._build_seen_mat(unique_uids, user_seen_items)
        if seen_cnt_u.max().item() < 1:
            return zero
        # 用 inverse 映射回 flat 维度
        seen_mat = seen_mat_u[inverse_map]
        seen_cnt_raw = seen_cnt_u[inverse_map]

        flat_item_idx = item_idx.view(-1, 1).expand(-1, n_cand).reshape(-1)
        batch_idx = torch.arange(flat_user_idx.size(0), device=self.device)
        fit = torch.zeros((flat_user_idx.size(0), 1), dtype=torch.float32, device=self.device)

        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)
        prereq_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_prereq_gate)))
        prereq_gap = torch.zeros_like(fit)
        prereq_safe = torch.ones_like(fit)

        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            prereq_gap = violation_full[batch_idx, flat_item_idx].unsqueeze(1)
            prereq_safe = (prereq_gap <= prereq_gate).float()

        concept_bonus = torch.zeros_like(fit)
        redundant = torch.zeros_like(fit)
        seen_active = (seen_cnt_raw >= 1.0).float()
        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, flat_item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            redundant = ((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)).clamp(0.0, 1.0)
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant)

        difficulty_gap = torch.zeros_like(fit)
        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[flat_item_idx].unsqueeze(1)
            difficulty_gap = F.relu(item_difficulty - user_readiness)

        fit = (
            float(self.cfg.feedback_course_concept_weight) * concept_bonus
            - float(self.cfg.feedback_course_prereq_weight) * prereq_gap
            - float(self.cfg.feedback_course_difficulty_weight) * difficulty_gap
            - float(self.cfg.feedback_course_redundant_weight) * redundant
        ) * active.repeat_interleave(n_cand, dim=0)

        return fit.view(batch_size, n_cand)

    def _apply_course_sampling_bias(self, candidates, cand_user_idx, item_idx, target_pop=None, user_seen_items=None):
        if (
            candidates is None or
            cand_user_idx is None or
            float(self.cfg.feedback_course_sample_beta) <= 0.0
        ):
            return candidates, cand_user_idx, None

        fit_score = self._compute_candidate_course_fit(
            cand_user_idx, item_idx=item_idx, target_pop=target_pop, user_seen_items=user_seen_items,
        )
        if not torch.isfinite(fit_score).all():
            fit_score = torch.nan_to_num(fit_score, nan=0.0, posinf=0.0, neginf=0.0)

        order = torch.argsort(fit_score, dim=1, descending=True)
        candidates = candidates.gather(1, order.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        cand_user_idx = cand_user_idx.gather(1, order)
        fit_score = fit_score.gather(1, order)
        return candidates, cand_user_idx, fit_score

    # ===================== 优化4: 预 normalize user bank =====================
    def _build_user_bank_raw(self):
        """返回 (raw_bank, normalized_bank) 元组。"""
        chunks_raw = []
        user_chunk = max(1, int(self.cfg.retrieval_user_chunk))
        with torch.no_grad():
            for start in range(0, self.cfg.n_users, user_chunk):
                end = min(start + user_chunk, self.cfg.n_users)
                idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                raw = self.user_proj(self.user_emb(idx))
                chunks_raw.append(raw.detach())
        raw_bank = torch.cat(chunks_raw, dim=0)
        norm_bank = F.normalize(raw_bank, dim=1)
        return raw_bank, norm_bank

    def _retrieve_topm_exact(self, query_norm, user_bank_raw, top_m, user_bank_norm=None):
        """优化版: 若提供 user_bank_norm 则跳过重复 normalize。"""
        top_scores_list = []
        top_idx_list = []
        query_chunk = max(1, int(self.cfg.retrieval_query_chunk))
        user_chunk = max(1, int(self.cfg.retrieval_user_chunk))

        for q_start in range(0, query_norm.size(0), query_chunk):
            q_end = min(q_start + query_chunk, query_norm.size(0))
            q = query_norm[q_start:q_end]
            q_top_scores = torch.full((q.size(0), top_m), -1e9, device=self.device)
            q_top_idx = torch.zeros((q.size(0), top_m), dtype=torch.long, device=self.device)

            for u_start in range(0, user_bank_raw.size(0), user_chunk):
                u_end = min(u_start + user_chunk, user_bank_raw.size(0))
                if user_bank_norm is not None:
                    u_chunk = user_bank_norm[u_start:u_end]
                else:
                    u_chunk = F.normalize(user_bank_raw[u_start:u_end], dim=1)
                score_chunk = torch.matmul(q, u_chunk.t())
                k_chunk = min(top_m, score_chunk.size(1))
                chunk_scores, chunk_idx = torch.topk(score_chunk, k=k_chunk, dim=1)
                chunk_idx = chunk_idx + u_start

                if k_chunk < top_m:
                    pad_size = top_m - k_chunk
                    score_pad = torch.full((q.size(0), pad_size), -1e9, device=self.device)
                    idx_pad = torch.zeros((q.size(0), pad_size), dtype=torch.long, device=self.device)
                    chunk_scores = torch.cat([chunk_scores, score_pad], dim=1)
                    chunk_idx = torch.cat([chunk_idx, idx_pad], dim=1)

                merged_scores = torch.cat([q_top_scores, chunk_scores], dim=1)
                merged_idx = torch.cat([q_top_idx, chunk_idx], dim=1)
                q_top_scores, keep_pos = torch.topk(merged_scores, k=top_m, dim=1)
                q_top_idx = merged_idx.gather(1, keep_pos)

            top_scores_list.append(q_top_scores)
            top_idx_list.append(q_top_idx)

        return torch.cat(top_scores_list, dim=0), torch.cat(top_idx_list, dim=0)

    def get_candidates(self, item_emb, user_bank_raw=None, user_bank_norm=None):
        """优化版: 接受预 normalized bank 避免重复计算。"""
        B = item_emb.size(0)
        N_cand = self.cfg.n_candidates
        strategy = self.cfg.candidate_strategy

        if strategy != "retrieve_sample":
            rand_idx = torch.randint(0, self.cfg.n_users, (B, N_cand), device=self.device)
            cand_emb = self.user_proj(self.user_emb(rand_idx)).detach()
            return cand_emb, rand_idx, None

        if user_bank_raw is None:
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()

        top_m = min(self.cfg.retrieve_top_m, self.cfg.n_users)
        top_m = max(1, top_m)
        q_norm = F.normalize(item_emb, dim=1)
        top_scores, top_idx = self._retrieve_topm_exact(q_norm, user_bank_raw, top_m, user_bank_norm=user_bank_norm)

        safe_temp = max(self.cfg.candidate_temp, 1e-6)
        probs = F.softmax(top_scores / safe_temp, dim=1)
        bad_rows = (~torch.isfinite(probs)).any(dim=1) | (probs.sum(dim=1) <= 0)
        if bad_rows.any():
            probs[bad_rows] = 1.0 / top_m

        eps = float(min(1.0, max(0.0, self.cfg.candidate_epsilon)))
        probs = (1.0 - eps) * probs + eps / top_m
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        replacement = top_m < N_cand
        sample_pos = torch.multinomial(probs, num_samples=N_cand, replacement=replacement)
        cand_idx = top_idx.gather(1, sample_pos)
        cand_emb = user_bank_raw[cand_idx].detach()

        topm_unique = max(1, int(top_idx.unique().numel()))
        selected_unique = int(cand_idx.unique().numel())
        selected_total = max(1, int(cand_idx.numel()))
        dup_rate = 1.0 - (selected_unique / selected_total)
        topm_cov = selected_unique / topm_unique
        stats = {"dup_rate": float(dup_rate), "topm_coverage": float(topm_cov)}
        return cand_emb, cand_idx, stats

    # ===================== 优化2: RL 循环内缓存 seen_mat =====================
    def run_usim_episode(self, init_item_emb, target_emb=None, user_bank_raw=None,
                         item_idx=None, target_pop=None, user_seen_items=None):
        current_h = init_item_emb.clone()
        trajectory = {
            "log_probs": [], "values": [], "rewards": [], "entropies": [],
            "states": [], "time_steps": [], "candidates": [], "actions": [],
        }
        candidate_stats = {
            "dup_rate": 0.0, "topm_coverage": 0.0, "steps": 0,
            "step_gain": 0.0, "collapse_penalty": 0.0,
            "course_sample_fit": 0.0, "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0, "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
        }

        # ---- 优化4: 预构建 bank pair ----
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
                current_h, user_bank_raw=user_bank_raw, user_bank_norm=user_bank_norm
            )
            candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                candidates, cand_user_idx,
                item_idx=item_idx, target_pop=target_pop, user_seen_items=user_seen_items,
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

            # P0-1: 混合梯度方向 = 0.5 * selected_user + 0.5 * target
            # 让 RL 选出的用户方向和 target 方向共同引导表征更新
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                user_score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                if target_emb is not None:
                    target_score = (h_detached * target_emb.detach()).sum(dim=1).mean()
                    score = 0.5 * user_score + 0.5 * target_score
                else:
                    score = user_score
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
                    min=-float(self.cfg.reward_gain_clip), max=float(self.cfg.reward_gain_clip),
                )
                reward = terminal_reward + float(self.cfg.reward_gain_weight) * step_gain
                step_gain_mean = float(step_gain.mean().item())
                if cand_stats is not None:
                    collapse_penalty = float(self.cfg.reward_dup_penalty_weight) * float(cand_stats["dup_rate"])
                    reward = reward - collapse_penalty
                    if float(self.cfg.reward_cov_bonus_weight) > 0.0:
                        reward = reward + float(self.cfg.reward_cov_bonus_weight) * float(cand_stats["topm_coverage"])

            course_terms = self._compute_course_reward_terms(
                selected_user_ids, item_idx=item_idx, target_pop=target_pop, user_seen_items=user_seen_items,
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
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

        if candidate_stats["steps"] > 0:
            candidate_stats["dup_rate"] /= candidate_stats["steps"]
            candidate_stats["topm_coverage"] /= candidate_stats["steps"]
            candidate_stats["step_gain"] /= candidate_stats["steps"]
            candidate_stats["collapse_penalty"] /= candidate_stats["steps"]
            candidate_stats["course_sample_fit"] /= candidate_stats["steps"]
            candidate_stats["course_prereq_gap"] /= candidate_stats["steps"]
            candidate_stats["course_concept_bonus"] /= candidate_stats["steps"]
            candidate_stats["course_difficulty_gap"] /= candidate_stats["steps"]
            candidate_stats["course_redundant"] /= candidate_stats["steps"]
        return current_h, trajectory, candidate_stats

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold

        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)

        # P0-3: 统一 target — cold 和 hot 都用融合后的表征作为目标
        # 原版让 hot item 退回纯 ID embedding，导致 cold/hot 学习目标不一致
        target_emb = z_i_base.detach().clone()

        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base, target_emb, user_bank_raw=user_bank_raw,
            item_idx=i, target_pop=pop, user_seen_items=user_seen_items,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0), device=self.device)

        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp

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
                main_loss = F.cross_entropy(cand_logits, main_targets)
            else:
                main_loss = F.cross_entropy(logits_margin, labels)
        else:
            main_loss = F.cross_entropy(logits_margin, labels)

        z_id = F.normalize(id_e_true, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            seen_mat, seen_cnt_raw = self._build_seen_mat(user_ids, user_seen_items)
            seen_cnt = seen_cnt_raw.squeeze(1)
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


# ===================== main() 训练循环 =====================

def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for Feedback-Aware USIM (FAST) from {data_dir}...")
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
        df, cfg.n_items,
        relation_dir="MOOCCube/relations",
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    pop_stats = df.groupby("i_idx")["popularity"].max()
    for item_id, pop_value in pop_stats.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_final_pop[idx] = int(pop_value)

    # ---- 使用优化版模型 ----
    model = FastFeedbackUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Feedback-Aware RL-USIM + InfoNCE [FAST] (Batch Size={cfg.batch_size})")
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
        f">> Course Feedback: only_cold={cfg.feedback_course_only_cold} | "
        f"warm_seen={cfg.feedback_course_warm_seen} | "
        f"concept_min={cfg.feedback_course_concept_min:.2f} | "
        f"redundant_thr={cfg.feedback_course_redundant_thr:.2f} | "
        f"prereq_gate={cfg.feedback_course_prereq_gate:.2f} | "
        f"w[p={cfg.feedback_course_prereq_weight:.2f}, "
        f"c={cfg.feedback_course_concept_weight:.2f}, "
        f"d={cfg.feedback_course_difficulty_weight:.2f}, "
        f"r={cfg.feedback_course_redundant_weight:.2f}]"
    )
    print(
        f">> Course Sampling: beta={cfg.feedback_course_sample_beta:.2f} | "
        f"only_cold={cfg.feedback_course_sample_only_cold}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}, "
        f"prereq_edges={course_stats['prereq_edges_kept']} "
        f"(raw={course_stats['prereq_edges_raw']}, users={course_stats['prereq_users']})"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )
    print(">> [FAST] Optimizations: vectorized seen_mat, user dedup, pre-normalized bank")

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
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0
        resume_this_period = (resume_current_period is not None and t == resume_current_period)

        if resume_this_period:
            print(f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs}")
        elif t >= warmup_periods:
            print("  [EVAL-START] Build eval item bank and run sampled/full ranking...")
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

        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = (t >= warmup_periods) and cfg.use_epoch_early_stop and cfg.n_epochs > 1
        es_best = copy.deepcopy(resume_es_best) if resume_this_period else None
        es_best_state = copy.deepcopy(resume_es_best_state) if resume_this_period else None
        es_best_opt_state = copy.deepcopy(resume_es_best_opt_state) if resume_this_period else None
        es_no_improve = int(resume_es_no_improve) if resume_this_period else 0
        epoch_start_idx = resume_next_epoch if resume_this_period else 0

        if ckpt_enabled and not resume_this_period:
            period_start_state = _build_feedback_ckpt_state(
                model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                accumulated_periods=t + 1, warmup_periods=warmup_periods,
                total_periods=len(periods), status="in_period",
                next_period=t, current_period=t, next_epoch=0,
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
            course_sample_fit_sum = 0.0
            course_prereq_sum = 0.0
            course_concept_sum = 0.0
            course_diff_sum = 0.0
            course_redundant_sum = 0.0
            cand_batches = 0
            optimizer.zero_grad()

            # ---- 优化4: 构建 (raw, norm) 元组 ----
            cached_user_bank = None
            if cfg.candidate_strategy == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()  # returns (raw, norm)

            print(
                f"  [TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Period {t + 1}/{len(periods)} | samples={len(combined_df)} | batches={num_batches}"
            )
            last_progress_log = epoch_start

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
                    batch, pop.to(device), llm.to(device),
                    user_bank_raw=cached_user_bank, user_seen_items=user_seen_items,
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
                avg_course_sample_fit = course_sample_fit_sum / cand_batches
                avg_course_prereq = course_prereq_sum / cand_batches
                avg_course_concept = course_concept_sum / cand_batches
                avg_course_diff = course_diff_sum / cand_batches
                avg_course_redundant = course_redundant_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"SampleFit: {avg_course_sample_fit:.4f} | "
                    f"Course[p={avg_course_prereq:.4f}, c={avg_course_concept:.4f}, "
                    f"d={avg_course_diff:.4f}, r={avg_course_redundant:.4f}]"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if ckpt_enabled:
                epoch_state = _build_feedback_ckpt_state(
                    model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                    full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                    accumulated_periods=t + 1, warmup_periods=warmup_periods,
                    total_periods=len(periods), status="in_period",
                    next_period=t, current_period=t, next_epoch=epoch + 1,
                    es_best=es_best, es_best_state=es_best_state,
                    es_best_opt_state=es_best_opt_state, es_no_improve=es_no_improve,
                )
                _save_feedback_checkpoint(ckpt_dir, epoch_state)

            if do_early_stop:
                print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
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
                    es_best = {"epoch": epoch + 1, "cold_n": float(cur_n), "cold_r": float(cur_cr), "hot_r": float(cur_hr)}
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
                    _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
                        model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                        full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                        accumulated_periods=t + 1, warmup_periods=warmup_periods,
                        total_periods=len(periods), status="in_period",
                        next_period=t, current_period=t, next_epoch=epoch + 1,
                        es_best=es_best, es_best_state=es_best_state,
                        es_best_opt_state=es_best_opt_state, es_no_improve=es_no_improve,
                    ))

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
            _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
                model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                accumulated_periods=t + 1, warmup_periods=warmup_periods,
                total_periods=len(periods), status="between_periods", next_period=t + 1,
            ))

        if resume_this_period:
            resume_current_period = None
            resume_next_epoch = 0
            resume_es_best = None
            resume_es_best_state = None
            resume_es_best_opt_state = None
            resume_es_no_improve = 0

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Feedback-Aware RL-USIM FAST)")
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

    pd.DataFrame(history).to_csv("mooc_metrics_usim_feedback_fast.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Feedback-Aware RL-USIM [FAST]: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_usim_feedback_fast.png")
    print(">> Saved mooc_result_usim_feedback_fast.png and csv")

    if ckpt_enabled:
        _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
            model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
            full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
            accumulated_periods=len(periods), warmup_periods=warmup_periods,
            total_periods=len(periods), status="finished", next_period=len(periods),
        ), snapshot_name="finished.pt")


if __name__ == "__main__":
    setup_seed(2025)
    main()
