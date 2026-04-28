import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import json
import os
import random
import time
import re
import copy
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ================= 1. 基础设置 =================
def setup_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed fixed: {seed}")


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name, default=0):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def apply_fast_dev_config(cfg):
    fast_dev = env_flag('USIM_FAST_DEV', False)
    if not fast_dev:
        return []

    notes = []
    cfg.n_epochs = env_int('USIM_FAST_N_EPOCHS', 1)
    cfg.usim_steps = env_int('USIM_FAST_USIM_STEPS', 2)
    cfg.retrieve_top_m = min(cfg.retrieve_top_m, env_int('USIM_FAST_TOP_M', 64))
    cfg.n_candidates = min(cfg.n_candidates, env_int('USIM_FAST_N_CAND', 8))
    cfg.eval_n_neg = min(cfg.eval_n_neg, env_int('USIM_FAST_EVAL_NEG', 50))
    cfg.use_epoch_early_stop = False
    notes.append(
        f"FAST_DEV: epochs={cfg.n_epochs}, usim_steps={cfg.usim_steps}, "
        f"top_m={cfg.retrieve_top_m}, n_candidates={cfg.n_candidates}, "
        f"eval_n_neg={cfg.eval_n_neg}, early_stop={cfg.use_epoch_early_stop}"
    )
    return notes


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items

        self.emb_dim = 128            # 统一维度
        self.hidden_dim = 256
        self.content_dim = content_dim

        self.cold_threshold = 5

        # 优化参数
        self.lr = 0.0005              # unified learning rate
        self.temp = 0.07              # 对齐 PAM Enhanced
        self.margin = 0.15            # Additive Margin

        # Dropout + auxiliary loss
        self.dropout_prob = 0.35      # ID Dropout
        self.aux_weight = 0.3         # 辅助对比损失权重
        self.use_semantic_delta = True
        self.semantic_delta_scale = 0.10
        self.semantic_drift_weight = 0.02
        self.semantic_align_weight = 0.05
        self.semantic_distill_weight = 0.08
        self.semantic_distill_temp = 0.12

        # RL Hyperparams
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {'value': 0.5, 'entropy': 0.01}

        self.usim_steps = 5           # USIM 步数
        self.n_candidates = 20
        self.usim_lr = 0.3
        self.candidate_strategy = "retrieve_sample"
        self.retrieve_top_m = 256
        self.candidate_temp = 0.20
        self.candidate_epsilon = 0.10
        self.retrieval_user_chunk = 16384
        self.retrieval_query_chunk = 256
        self.user_bank_refresh_steps = 200

        self.n_epochs = int(os.environ.get('USIM_N_EPOCHS', '3'))
        # After removing MAML, GPU memory usage is lower
        self.batch_size = 2048        # 真实的大 Batch Size!
        self.accum_steps = 1          # no gradient accumulation needed
        self.eval_n_neg = 200

        # Mixed hard-negative training (in-batch)
        self.use_mixed_hard_neg = True
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        self.use_structured_hard_neg = False

        # Course-specific inference rerank
        self.use_course_rerank = False
        self.rerank_alpha = 0.00
        self.rerank_lambda = 0.01
        self.rerank_min_seen = 8
        self.rerank_top_l = 50
        self.rerank_penalty_cap = 0.10
        self.rerank_only_cold = True

        # Course-level prerequisite graph (behavior-supported)
        self.prereq_min_support = 30
        self.prereq_max_per_item = 5
        self.prereq_min_items = 1
        self.prereq_max_forward = 20

        # Training-time prerequisite auxiliary loss
        self.use_prereq_aux_loss = True
        self.prereq_aux_weight = 0.03
        self.prereq_aux_margin = 0.05
        self.prereq_aux_violation_thr = 0.60
        self.prereq_aux_min_seen = 5
        self.prereq_aux_only_cold = True

        # Epoch-level early stopping (multi-metric)
        self.use_epoch_early_stop = True
        self.early_stop_k = 10
        self.early_stop_patience = 1
        self.early_stop_min_delta = 1e-4
        self.early_stop_hot_r10_drop_tol = 0.03


# ================= 2. PPO Agent =================

class SimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=4):
        super(SimpleAC, self).__init__()
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
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=10)[:, :4].float()
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


# ================= 3. Pure RL-USIM + PAM Enhanced Main Model =================

class PAM_RL_Pure_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 统一维度 Embeddings
        self.user_emb = nn.Embedding(config.n_users, config.emb_dim)
        self.item_id_emb = nn.Embedding(config.n_items, config.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # 2. 3-layer GELU content encoder
        self.content_proj = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim)
        )

        # 3. User projection layer (aligned with PAM Enhanced)
        self.user_proj = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim)
        )

        # [New] LLM score projection layer
        self.llm_proj = nn.Linear(1, config.emb_dim)
        self.semantic_delta = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.Tanh()
        )
        self.semantic_mix = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid()
        )

        # 4. 门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid()
        )

        # 5. RL Agent
        self.agent = SimpleAC(config.emb_dim, time_dim=4)

        # Course-specific priors (loaded at runtime)
        self.item_hard_adj = None
        self.item_prereq_item_mat = None
        self.item_prereq_item_cnt = None
        self.item_concept_overlap = None

    def set_course_artifacts(self, artifacts):
        if not artifacts:
            return
        self.item_hard_adj = artifacts.get('item_hard_adj')
        self.item_prereq_item_mat = artifacts.get('item_prereq_item_mat')
        self.item_prereq_item_cnt = artifacts.get('item_prereq_item_cnt')
        self.item_concept_overlap = artifacts.get('item_concept_overlap')

        if self.item_hard_adj is not None:
            self.item_hard_adj = self.item_hard_adj.to(self.device)
        if self.item_prereq_item_mat is not None:
            self.item_prereq_item_mat = self.item_prereq_item_mat.to(self.device)
        if self.item_prereq_item_cnt is not None:
            self.item_prereq_item_cnt = self.item_prereq_item_cnt.to(self.device)
        if self.item_concept_overlap is not None:
            self.item_concept_overlap = self.item_concept_overlap.to(self.device)

    def apply_course_rerank(self, scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=None):
        if not self.cfg.use_course_rerank:
            return scores
        if self.item_prereq_item_mat is None or self.item_prereq_item_cnt is None or self.item_concept_overlap is None:
            return scores

        batch_size = scores.size(0)
        if batch_size < 1:
            return scores

        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float, device=self.device)
        for row, uid in enumerate(user_ids):
            seen_idx = seen_tensor_cache.get(int(uid))
            if seen_idx is not None and seen_idx.numel() > 0:
                seen_mat[row, seen_idx] = 1.0

        prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
        prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
        has_prereq = prereq_cnt > 0
        violation = torch.where(
            has_prereq,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
            torch.zeros_like(prereq_seen)
        ).clamp(0.0, 1.0)

        seen_cnt_raw = seen_mat.sum(dim=1, keepdim=True)
        seen_cnt = seen_cnt_raw.clamp_min(1.0)
        concept_match = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt

        user_mask = torch.ones_like(seen_cnt_raw)
        if getattr(self.cfg, 'rerank_min_seen', 0) > 0:
            user_mask = (seen_cnt_raw >= float(self.cfg.rerank_min_seen)).float()

        row_mask = torch.ones_like(seen_cnt_raw)
        if target_pop is not None and getattr(self.cfg, 'rerank_only_cold', False):
            row_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()

        active_mask = user_mask * row_mask
        if active_mask.sum().item() < 1:
            return scores

        penalty = (self.cfg.rerank_lambda * violation).clamp(
            min=0.0,
            max=float(getattr(self.cfg, 'rerank_penalty_cap', 1.0))
        )
        adjust_full = (self.cfg.rerank_alpha * concept_match - penalty) * active_mask
        top_l = int(getattr(self.cfg, 'rerank_top_l', 0))

        if cand_idx is None:
            if top_l > 0 and top_l < scores.size(1):
                _, top_idx = torch.topk(scores, k=top_l, dim=1)
                sparse_adjust = torch.zeros_like(scores)
                sparse_adjust.scatter_(1, top_idx, adjust_full.gather(1, top_idx))
                return scores + sparse_adjust
            return scores + adjust_full

        cand_adjust = adjust_full.gather(1, cand_idx)
        if top_l > 0 and top_l < scores.size(1):
            _, top_pos = torch.topk(scores, k=top_l, dim=1)
            top_mask = torch.zeros_like(scores, dtype=torch.bool)
            top_mask.scatter_(1, top_pos, True)
            cand_adjust = cand_adjust.masked_fill(~top_mask, 0.0)
        return scores + cand_adjust

    def _build_user_bank_raw(self):
        user_bank_chunks = []
        user_chunk = max(1, int(self.cfg.retrieval_user_chunk))
        with torch.no_grad():
            for start in range(0, self.cfg.n_users, user_chunk):
                end = min(start + user_chunk, self.cfg.n_users)
                idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                raw = self.user_proj(self.user_emb(idx))
                user_bank_chunks.append(raw.detach())
        return torch.cat(user_bank_chunks, dim=0)

    def _retrieve_topm_exact(self, query_norm, user_bank_raw, top_m):
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
                u_chunk_raw = user_bank_raw[u_start:u_end]
                u_chunk = F.normalize(u_chunk_raw, dim=1)
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

    def get_candidates(self, item_emb, user_bank_raw=None):
        B = item_emb.size(0)
        N_cand = self.cfg.n_candidates
        strategy = self.cfg.candidate_strategy

        if strategy != "retrieve_sample":
            rand_idx = torch.randint(0, self.cfg.n_users, (B, N_cand), device=self.device)
            cand_emb = self.user_proj(self.user_emb(rand_idx)).detach()
            return cand_emb, rand_idx, None

        if user_bank_raw is None:
            user_bank_raw = self._build_user_bank_raw()

        top_m = min(self.cfg.retrieve_top_m, self.cfg.n_users)
        top_m = max(1, top_m)
        q_norm = F.normalize(item_emb, dim=1)
        top_scores, top_idx = self._retrieve_topm_exact(q_norm, user_bank_raw, top_m)

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
        # Global duplicate rate over the batch, useful to monitor candidate collapse.
        dup_rate = 1.0 - (selected_unique / selected_total)
        topm_cov = selected_unique / topm_unique
        stats = {"dup_rate": float(dup_rate), "topm_coverage": float(topm_cov)}
        return cand_emb, cand_idx, stats

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        id_e = self.item_id_emb(i_idx)
        
        # ID Dropout
        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)
            
        semantic_base = self.content_proj(self.item_con_emb(i_idx))
        
        # Use llm score as a small semantic prior rather than a direct replacement.
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        if self.cfg.use_semantic_delta:
            delta = self.semantic_delta(torch.cat([semantic_base, llm_e], dim=-1))
            delta = delta * float(self.cfg.semantic_delta_scale)
            semantic_refined = semantic_base + delta
            mix = self.semantic_mix(torch.cat([semantic_base, semantic_refined], dim=-1))
            content_e = mix * semantic_refined + (1.0 - mix) * semantic_base
        else:
            delta = torch.zeros_like(semantic_base)
            content_e = semantic_base
        
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        
        item_fused = alpha * id_e + (1 - alpha) * content_e
        return item_fused, id_e, semantic_base, content_e, delta

    def run_usim_episode(self, init_item_emb, target_emb=None, user_bank_raw=None):
        current_h = init_item_emb.clone()
        trajectory = {
            'log_probs': [], 'values': [], 'rewards': [], 'entropies': [],
            'states': [], 'time_steps': [], 'candidates': [], 'actions': []
        }
        candidate_stats = {'dup_rate': 0.0, 'topm_coverage': 0.0, 'steps': 0}
        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw = self._build_user_bank_raw()

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, _, cand_stats = self.get_candidates(current_h, user_bank_raw=user_bank_raw)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)
            if cand_stats is not None:
                candidate_stats['dup_rate'] += cand_stats['dup_rate']
                candidate_stats['topm_coverage'] += cand_stats['topm_coverage']
                candidate_stats['steps'] += 1

            trajectory['states'].append(current_h.detach().clone())
            trajectory['time_steps'].append(time_step.detach().clone())
            trajectory['candidates'].append(candidates.detach().clone())
            trajectory['actions'].append(action_idx.detach().clone())

            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            # Simulate representation shift after user click
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            # Reward 信号
            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_emb is not None:
                dist = F.mse_loss(current_h, target_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0

            trajectory['log_probs'].append(log_prob.detach())
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['entropies'].append(entropy)

        if candidate_stats['steps'] > 0:
            candidate_stats['dup_rate'] /= candidate_stats['steps']
            candidate_stats['topm_coverage'] /= candidate_stats['steps']
        return current_h, trajectory, candidate_stats

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory['rewards']).squeeze(-1)
        old_log_probs = torch.stack(trajectory['log_probs'])
        states = trajectory['states']
        time_steps = trajectory['time_steps']
        candidates = trajectory['candidates']
        actions = trajectory['actions']

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.cfg.ppo_gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        total_ppo_loss = 0

        for _ in range(self.cfg.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            new_entropies_list = []

            for t in range(len(states)):
                _, new_log_prob, new_value, new_entropy = self.agent.get_action_value(
                    states[t], time_steps[t], candidates[t], action_idx=actions[t]
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                new_entropies_list.append(new_entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze(-1)
            new_entropies = torch.stack(new_entropies_list)

            advantage = (returns - new_values).detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - new_values).pow(2).mean()
            entropy_loss = -new_entropies.mean()

            total_ppo_loss += actor_loss + self.cfg.ppo_coeffs['value'] * critic_loss + \
                              self.cfg.ppo_coeffs['entropy'] * entropy_loss

        return total_ppo_loss / self.cfg.ppo_epochs

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold

        # 1. 基础表征 (PAM 优势 + LLM 增益)
        z_u_base = self.user_proj(self.user_emb(u))
        z_i_base, id_e_raw, semantic_base, content_e, semantic_delta = self.get_item_vector(
            i, llm_s, force_cold=False
        )

        # 2. RL USIM 序列模拟
        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

        # For cold items, use PPO in training to build imagined interaction trajectories
        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base, target_emb, user_bank_raw=user_bank_raw
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        # 3. Ranking loss: remove MAML, use large-batch InfoNCE directly
        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)  # final representation after RL refinement

        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0)).to(self.device)

        # Additive Margin for Hard Negatives
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp

        # Mixed hard+random negatives built from in-batch candidates.
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

        # 4. 辅助损失 (Content <-> ID)
        z_id = F.normalize(id_e_raw, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        semantic_drift_loss = semantic_delta.pow(2).mean()

        if is_cold.any():
            semantic_align_loss = (
                1.0 - F.cosine_similarity(final_h[is_cold], semantic_base.detach()[is_cold], dim=1)
            ).mean()
        else:
            semantic_align_loss = torch.tensor(0.0, device=self.device)

        semantic_distill_loss = torch.tensor(0.0, device=self.device)
        if logits.size(0) > 1 and self.cfg.semantic_distill_weight > 0:
            distill_temp = max(float(self.cfg.semantic_distill_temp), 1e-6)
            teacher_u = z_u.detach()
            teacher_i = F.normalize(semantic_base.detach(), dim=1)
            teacher_logits = torch.matmul(teacher_u, teacher_i.t()) / distill_temp
            teacher_probs = F.softmax(teacher_logits, dim=1)
            student_log_probs = F.log_softmax(
                torch.matmul(z_u, z_i.t()) / distill_temp,
                dim=1
            )
            kl_rows = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1)
            row_weights = torch.where(
                is_cold,
                torch.ones_like(kl_rows),
                torch.full_like(kl_rows, 0.25)
            )
            semantic_distill_loss = (
                (kl_rows * row_weights).sum() / row_weights.sum().clamp_min(1.0)
            ) * (distill_temp ** 2)

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            batch_size = logits.size(0)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float, device=self.device)

            for row, uid in enumerate(user_ids):
                seen_items = user_seen_items.get(uid)
                if not seen_items:
                    continue
                seen_list = [it for it in seen_items if 0 <= it < self.cfg.n_items]
                if seen_list:
                    seen_idx = torch.tensor(seen_list, dtype=torch.long, device=self.device)
                    seen_mat[row, seen_idx] = 1.0

            seen_cnt = seen_mat.sum(dim=1)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(seen_mat, prereq_mat_batch.t())

            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0 - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch)
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
                    pos_vals = logits[torch.arange(batch_size, device=self.device), labels]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            self.cfg.semantic_drift_weight * semantic_drift_loss +
            self.cfg.semantic_align_weight * semantic_align_loss +
            self.cfg.semantic_distill_weight * semantic_distill_loss
        )
        train_stats = dict(candidate_stats)
        train_stats['semantic_delta_norm'] = float(semantic_delta.detach().norm(dim=1).mean().item())
        train_stats['semantic_shift'] = float((content_e.detach() - semantic_base.detach()).norm(dim=1).mean().item())
        train_stats['semantic_distill'] = float(semantic_distill_loss.detach().item())
        train_stats['semantic_align'] = float(semantic_align_loss.detach().item())
        return total_loss, train_stats


# ================= 4. 评估工具 =================

def compute_ranking_metrics(scores, target_indices, k_list=[5, 10, 20]):
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        recall = hits.mean().item()
        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0
        results[f'R@{k}'] = recall
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results

def split_dataframe_by_periods(df, period_type='M'):
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = df['timestamp']
    df['period_id'] = df['datetime'].dt.to_period(period_type)
    periods = []
    for p_key in sorted(df['period_id'].unique()):
        periods.append(df[df['period_id'] == p_key].reset_index(drop=True))
    return periods


def _read_relation_pairs(filepath):
    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                a, b = line.split('\t', 1)
            elif ',' in line:
                a, b = line.split(',', 1)
                if a == 'start_id' and b == 'end_id':
                    continue
            else:
                continue
            pairs.append((a.strip(), b.strip()))
    return pairs


def _parse_subject_from_course_id(course_id):
    cid = str(course_id)
    m = re.search(r'\+([A-Za-z]+)\d+', cid)
    if m:
        return m.group(1).upper()
    m = re.search(r'course-v1:([^+]+)\+', cid)
    if m:
        return m.group(1).upper()
    return 'UNK'


def build_course_artifacts(
    df,
    n_items,
    relation_dir='MOOCCube/relations',
    prereq_min_support=30,
    prereq_max_per_item=5,
    prereq_min_items=1,
    prereq_max_forward=20
):
    idx_course = df[['i_idx', 'course_id']].drop_duplicates(subset=['i_idx'])
    idx_to_course = [None] * n_items
    for row in idx_course.itertuples(index=False):
        i_idx = int(row.i_idx)
        if 0 <= i_idx < n_items:
            idx_to_course[i_idx] = str(row.course_id)

    course_to_idx = {cid: idx for idx, cid in enumerate(idx_to_course) if cid is not None}
    concept_sets = [set() for _ in range(n_items)]

    course_concept_file = os.path.join(relation_dir, 'course-concept.json')
    for cid, concept in _read_relation_pairs(course_concept_file):
        idx = course_to_idx.get(cid)
        if idx is not None and concept:
            concept_sets[idx].add(concept)

    item_prereq_item_mat = torch.zeros((n_items, n_items), dtype=torch.float32)
    item_prereq_item_cnt = torch.zeros(n_items, dtype=torch.float32)
    edge_support = defaultdict(int)
    user_seq_count = 0

    if {'u_idx', 'i_idx', 'timestamp'}.issubset(df.columns):
        seq_df = df[['u_idx', 'i_idx', 'timestamp']].sort_values(['u_idx', 'timestamp'])
        max_forward = max(1, int(prereq_max_forward))

        for _, group in seq_df.groupby('u_idx', sort=False):
            seq_raw = [int(x) for x in group['i_idx'].tolist()]
            if len(seq_raw) < 2:
                continue

            # Keep the first occurrence in each user sequence to reduce repeated-course noise.
            seq = []
            seen_local = set()
            for item in seq_raw:
                if item in seen_local:
                    continue
                seen_local.add(item)
                seq.append(item)

            if len(seq) < 2:
                continue

            user_seq_count += 1
            for p, a in enumerate(seq):
                end = min(len(seq), p + 1 + max_forward)
                for q in range(p + 1, end):
                    b = seq[q]
                    if a != b:
                        edge_support[(a, b)] += 1

    incoming = defaultdict(list)
    for (a, b), sup in edge_support.items():
        if sup >= int(prereq_min_support):
            incoming[int(b)].append((int(a), int(sup)))

    kept_edge_count = 0
    for b, src_list in incoming.items():
        src_list.sort(key=lambda x: (-x[1], x[0]))
        kept = src_list[:max(1, int(prereq_max_per_item))]
        if len(kept) < int(prereq_min_items):
            continue
        idx_list = torch.tensor([src for src, _ in kept], dtype=torch.long)
        item_prereq_item_mat[b, idx_list] = 1.0
        item_prereq_item_cnt[b] = float(len(kept))
        kept_edge_count += len(kept)

    item_concept_overlap = torch.zeros((n_items, n_items), dtype=torch.float32)
    for i in range(n_items):
        c_i = concept_sets[i]
        denom = len(c_i)
        if denom < 1:
            continue
        for j in range(n_items):
            c_j = concept_sets[j]
            if not c_j:
                continue
            inter = len(c_i & c_j)
            if inter > 0:
                item_concept_overlap[i, j] = inter / float(denom)

    subjects = [_parse_subject_from_course_id(cid) if cid is not None else 'UNK' for cid in idx_to_course]
    item_hard_adj = torch.zeros((n_items, n_items), dtype=torch.bool)
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            same_subject = subjects[i] != 'UNK' and subjects[i] == subjects[j]
            same_concept = item_concept_overlap[i, j] > 0
            if same_subject or same_concept:
                item_hard_adj[i, j] = True

    items_with_concept = int(sum(1 for c in concept_sets if len(c) > 0))
    items_with_prereq = int((item_prereq_item_cnt > 0).sum().item())
    hard_density = float(item_hard_adj.float().mean().item())
    stats = {
        'items_with_concept': items_with_concept,
        'items_with_prereq': items_with_prereq,
        'hard_density': hard_density,
        'prereq_edges_kept': int(kept_edge_count),
        'prereq_edges_raw': int(len(edge_support)),
        'prereq_users': int(user_seq_count),
        'prereq_min_support': int(prereq_min_support),
        'prereq_max_per_item': int(prereq_max_per_item)
    }

    artifacts = {
        'item_hard_adj': item_hard_adj,
        'item_prereq_item_mat': item_prereq_item_mat,
        'item_prereq_item_cnt': item_prereq_item_cnt,
        'item_concept_overlap': item_concept_overlap
    }
    return artifacts, stats


def build_all_item_vecs(model, device, llm_scores, item_batch=1024):
    n_items = model.cfg.n_items
    all_item_idx = torch.arange(n_items, device=device)
    all_llm_s = torch.tensor(
        [llm_scores.get(int(idx), -1.0) for idx in all_item_idx],
        dtype=torch.float,
        device=device
    )

    all_item_vecs = []
    with torch.no_grad():
        for start in range(0, n_items, item_batch):
            end = min(start + item_batch, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]
            z_i, _, _, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=True)
            all_item_vecs.append(F.normalize(z_i, dim=1))
    return torch.cat(all_item_vecs, dim=0)


# ================= 5. 评估函数 =================

def evaluate_usim(model, loader, device, llm_scores, k_list=[5, 10, 20], n_neg=200,
                  eval_type='cold', full_ranking=False, user_seen_items=None, all_item_vecs=None,
                  max_batches=None):
    model.eval()
    accum_metrics = {}
    total_samples = 0
    seen_tensor_cache = {}
    fast_dev = env_flag('USIM_FAST_DEV', False)
    if max_batches is None:
        max_batches = env_int('USIM_DEV_MAX_EVAL_BATCHES', 2 if fast_dev else 0)
    max_batches = max(0, int(max_batches or 0))

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        if all_item_vecs is None:
            all_item_vecs = build_all_item_vecs(model, device, llm_scores, item_batch=1024)

        for batch_idx, (batch, pop, llm) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if eval_type == 'cold':
                mask = pop < model.cfg.cold_threshold
            elif eval_type == 'hot':
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

            for uid in user_ids:
                if uid in seen_tensor_cache:
                    continue
                seen_items = user_seen_items.get(uid) if user_seen_items else None
                if seen_items:
                    seen_list = [it for it in seen_items if 0 <= it < n_items]
                    seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=device) if seen_list else None
                else:
                    seen_tensor_cache[uid] = None

            z_u = F.normalize(model.user_proj(model.user_emb(u)), dim=1)

            if full_ranking:
                scores = torch.mm(z_u, all_item_vecs.t())
                if user_seen_items:
                    row_idx = torch.arange(n_sel, device=device)
                    target_scores = scores[row_idx, i].clone()
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache[uid]
                        if seen_idx is None:
                            continue
                        scores[row, seen_idx] = -1e9

                    # Keep target score valid for this row.
                    scores[row_idx, i] = target_scores
                scores = model.apply_course_rerank(
                    scores,
                    user_ids,
                    seen_tensor_cache,
                    cand_idx=None,
                    target_pop=pop_sel
                )
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        avail = n_items - 1 - int((seen_idx != i[row]).sum().item())
                    avail_counts.append(max(1, avail))

                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)

                for row, user_id in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache[int(user_id)]
                    if seen_idx is not None:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]
                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                cand_vecs = all_item_vecs[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                scores = model.apply_course_rerank(
                    scores,
                    user_ids,
                    seen_tensor_cache,
                    cand_idx=cand_idx,
                    target_pop=pop_sel
                )
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)

            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0

    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


# ================= 6. Dataset =================

class StreamDataset(Dataset):
    def __init__(self, df, llm_scores):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.tensor([llm_scores.get(int(idx), -1.0) for idx in self.i], dtype=torch.float)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm': self.llm_s[idx]}

def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm = torch.stack([item['llm'] for item in batch])
    return {'u': u, 'i': i}, pop, llm


def _add_user_seen_from_df(user_seen_items, src_df):
    for u_idx, i_idx in zip(src_df['u_idx'].values, src_df['i_idx'].values):
        uid = int(u_idx)
        if uid not in user_seen_items:
            user_seen_items[uid] = set()
        user_seen_items[uid].add(int(i_idx))
    return user_seen_items


def _clone_user_seen(user_seen_items):
    return {uid: set(items) for uid, items in user_seen_items.items()}


def run_static_experiment(df, cfg, device, model, optimizer, llm_scores):
    fast_dev = env_flag('USIM_FAST_DEV', False)
    max_train_batches = env_int('USIM_DEV_MAX_TRAIN_BATCHES', 10 if fast_dev else 0)
    skip_eval = env_flag('USIM_DEV_SKIP_EVAL', False)
    skip_full_eval = env_flag('USIM_DEV_SKIP_FULL_EVAL', fast_dev)
    static_seed = int(os.environ.get('USIM_STATIC_SEED', '2025'))
    train_ratio = float(os.environ.get('USIM_STATIC_TRAIN_RATIO', '0.8'))
    val_ratio = float(os.environ.get('USIM_STATIC_VAL_RATIO', '0.1'))
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        print("[STATIC] ratio 非法，回退为 0.8/0.1/0.1")
        train_ratio, val_ratio = 0.8, 0.1

    df_static = df.sample(frac=1.0, random_state=static_seed).reset_index(drop=True)
    n_total = len(df_static)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        print(f"[STATIC] 划分失败: total={n_total}, train={n_train}, val={n_val}, test={n_test}")
        return

    train_df = df_static.iloc[:n_train]
    val_df = df_static.iloc[n_train:n_train + n_val]
    test_df = df_static.iloc[n_train + n_val:]

    train_ds = StreamDataset(train_df, llm_scores)
    val_ds = StreamDataset(val_df, llm_scores)
    test_ds = StreamDataset(test_df, llm_scores)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

    train_seen = {}
    _add_user_seen_from_df(train_seen, train_df)
    test_seen = _clone_user_seen(train_seen)
    _add_user_seen_from_df(test_seen, val_df)

    print(
        f"\n>>> Start STATIC train/eval (seed={static_seed}) | "
        f"split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} | "
        f"train={n_train}, val={n_val}, test={n_test}"
    )
    if fast_dev or max_train_batches > 0 or skip_eval or skip_full_eval:
        print(
            f"[DEV] fast_dev={fast_dev} | max_train_batches={max_train_batches} | "
            f"skip_eval={skip_eval} | skip_full_eval={skip_full_eval}"
        )

    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]

    do_early_stop = cfg.use_epoch_early_stop and cfg.n_epochs > 1 and (not skip_eval) and (not skip_full_eval)
    es_best = None
    es_best_state = None
    es_best_opt_state = None
    es_no_improve = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        steps = 0
        cand_dup_sum = 0.0
        cand_cov_sum = 0.0
        cand_batches = 0
        sem_delta_sum = 0.0
        sem_shift_sum = 0.0
        sem_distill_sum = 0.0
        sem_align_sum = 0.0
        sem_batches = 0

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
                user_seen_items=train_seen
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            if cand_info and cand_info.get('steps', 0) > 0:
                cand_dup_sum += cand_info['dup_rate']
                cand_cov_sum += cand_info['topm_coverage']
                cand_batches += 1
            if cand_info and 'semantic_delta_norm' in cand_info:
                sem_delta_sum += cand_info['semantic_delta_norm']
                sem_shift_sum += cand_info.get('semantic_shift', 0.0)
                sem_distill_sum += cand_info.get('semantic_distill', 0.0)
                sem_align_sum += cand_info.get('semantic_align', 0.0)
                sem_batches += 1
            if max_train_batches > 0 and (batch_idx + 1) >= max_train_batches:
                break

        epoch_sec = time.time() - epoch_start
        avg_loss = total_loss / max(1, steps)
        if cand_batches > 0:
            avg_dup = cand_dup_sum / cand_batches
            avg_cov = cand_cov_sum / cand_batches
            sem_msg = ""
            if sem_batches > 0:
                sem_msg = (
                    f" | SemDelta: {sem_delta_sum / sem_batches:.4f}"
                    f" | SemShift: {sem_shift_sum / sem_batches:.4f}"
                    f" | Distill: {sem_distill_sum / sem_batches:.4f}"
                    f" | SemAlign: {sem_align_sum / sem_batches:.4f}"
                )
            print(
                f"  [STATIC-TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}{sem_msg}"
            )
        else:
            sem_msg = ""
            if sem_batches > 0:
                sem_msg = (
                    f" | SemDelta: {sem_delta_sum / sem_batches:.4f}"
                    f" | SemShift: {sem_shift_sum / sem_batches:.4f}"
                    f" | Distill: {sem_distill_sum / sem_batches:.4f}"
                    f" | SemAlign: {sem_align_sum / sem_batches:.4f}"
                )
            print(
                f"  [STATIC-TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s{sem_msg}"
            )

        if do_early_stop:
            all_item_vecs_val = build_all_item_vecs(model, device, llm_scores, item_batch=1024)
            val_cold, _ = evaluate_usim(
                model, val_loader, device, llm_scores, k_list,
                eval_type='cold', full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=all_item_vecs_val
            )
            val_hot, _ = evaluate_usim(
                model, val_loader, device, llm_scores, k_list,
                eval_type='hot', full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=all_item_vecs_val
            )

            key_n = f"N@{cfg.early_stop_k}"
            key_r = f"R@{cfg.early_stop_k}"
            cur_n = val_cold.get(key_n, 0.0) if val_cold else 0.0
            cur_cr = val_cold.get(key_r, 0.0) if val_cold else 0.0
            cur_hr = val_hot.get(key_r, 0.0) if val_hot else 0.0

            if es_best is None:
                is_better = True
            else:
                hot_floor = es_best['hot_r'] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
                hot_ok = cur_hr >= hot_floor
                n_improve = cur_n > es_best['cold_n'] + cfg.early_stop_min_delta
                n_tie = abs(cur_n - es_best['cold_n']) <= cfg.early_stop_min_delta
                r_tie_break = cur_cr > es_best['cold_r'] + 1e-12
                is_better = hot_ok and (n_improve or (n_tie and r_tie_break))

            if is_better:
                es_best = {
                    'epoch': epoch + 1,
                    'cold_n': float(cur_n),
                    'cold_r': float(cur_cr),
                    'hot_r': float(cur_hr)
                }
                es_best_state = copy.deepcopy(model.state_dict())
                es_best_opt_state = copy.deepcopy(optimizer.state_dict())
                es_no_improve = 0
                es_tag = "update"
            else:
                es_no_improve += 1
                es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

            print(
                f"  [STATIC-EARLYSTOP] Epoch {epoch+1}: "
                f"Full Cold {key_n}={cur_n:.4f}, Full Cold {key_r}={cur_cr:.4f}, "
                f"Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
            )

            if es_no_improve >= cfg.early_stop_patience:
                print(f"  [STATIC-EARLYSTOP] Triggered at epoch {epoch+1}.")
                break

    if do_early_stop and es_best_state is not None:
        model.load_state_dict(es_best_state)
        if es_best_opt_state is not None:
            optimizer.load_state_dict(es_best_opt_state)
        print(
            f"  [STATIC-EARLYSTOP] Restore best epoch={es_best['epoch']} "
            f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
            f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
            f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
        )

    if skip_eval:
        print("[DEV] STATIC evaluation skipped.")
        met_cold, n_cold_t = None, 0
        met_hot, n_hot_t = None, 0
        fmet_cold, fn_c = None, 0
        fmet_hot, fn_h = None, 0
    else:
        all_item_vecs_test = build_all_item_vecs(model, device, llm_scores, item_batch=1024)
        met_cold, n_cold_t = evaluate_usim(
            model, test_loader, device, llm_scores, k_list,
            n_neg=cfg.eval_n_neg, eval_type='cold',
            user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
        )
        met_hot, n_hot_t = evaluate_usim(
            model, test_loader, device, llm_scores, k_list,
            n_neg=cfg.eval_n_neg, eval_type='hot',
            user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
        )
        if skip_full_eval:
            fmet_cold, fn_c = None, 0
            fmet_hot, fn_h = None, 0
        else:
            fmet_cold, fn_c = evaluate_usim(
                model, test_loader, device, llm_scores, k_list,
                eval_type='cold', full_ranking=True,
                user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
            )
            fmet_hot, fn_h = evaluate_usim(
                model, test_loader, device, llm_scores, k_list,
                eval_type='hot', full_ranking=True,
                user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
            )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT (STATIC): 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)

    for m in metrics_keys:
        sc = met_cold.get(m, 0.0) if met_cold else 0.0
        sh = met_hot.get(m, 0.0) if met_hot else 0.0
        fc = fmet_cold.get(m, 0.0) if fmet_cold else 0.0
        fh = fmet_hot.get(m, 0.0) if fmet_hot else 0.0
        print(f"{m:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"全库 Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)




# ================= 7. Main Training Loop =================

def main():
    DATA_DIR = "processed_data_hin"
    run_tag = os.environ.get('USIM_RUN_TAG', 'pure_usim_semantic')
    metrics_path = f"mooc_metrics_{run_tag}.csv"
    plot_path = f"mooc_result_{run_tag}.png"
    fast_dev = env_flag('USIM_FAST_DEV', False)
    dev_rows = env_int('USIM_DEV_ROWS', 40000 if fast_dev else 0)
    max_periods = env_int('USIM_DEV_MAX_PERIODS', 2 if fast_dev else 0)
    max_train_batches = env_int('USIM_DEV_MAX_TRAIN_BATCHES', 10 if fast_dev else 0)
    skip_eval = env_flag('USIM_DEV_SKIP_EVAL', False)
    skip_full_eval = env_flag('USIM_DEV_SKIP_FULL_EVAL', fast_dev)

    print(f"Loading Data for Pure RL-USIM + SemanticDelta from {DATA_DIR}...")
    if not os.path.exists(f"{DATA_DIR}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{DATA_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{DATA_DIR}/stream_data.pkl")
    with open(f"{DATA_DIR}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{DATA_DIR}/content_emb.pt")

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    dev_notes = apply_fast_dev_config(cfg)
    if dev_rows > 0 and dev_rows < len(df):
        df = df.iloc[:dev_rows].copy().reset_index(drop=True)
        dev_notes.append(f"DEV_ROWS: using first {len(df)} rows")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir='MOOCCube/relations',
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward
    )

    model = PAM_RL_Pure_USIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Pure RL-USIM + SemanticDelta (Batch Size={cfg.batch_size}, tag={run_tag})")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> Semantic Delta: enabled={cfg.use_semantic_delta} | "
        f"scale={cfg.semantic_delta_scale:.2f} | drift_w={cfg.semantic_drift_weight:.2f} | "
        f"align_w={cfg.semantic_align_weight:.2f} | distill_w={cfg.semantic_distill_weight:.2f} | "
        f"distill_temp={cfg.semantic_distill_temp:.2f}"
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
    if dev_notes or max_periods > 0 or max_train_batches > 0 or skip_eval or skip_full_eval:
        print(
            f">> Dev Mode: fast_dev={fast_dev} | max_periods={max_periods} | "
            f"max_train_batches={max_train_batches} | skip_eval={skip_eval} | "
            f"skip_full_eval={skip_full_eval}"
        )
        for note in dev_notes:
            print(f"   - {note}")

    use_static = os.environ.get('USIM_STATIC', '0') == '1'
    if use_static:
        run_static_experiment(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type='M')
    if max_periods > 0 and max_periods < len(periods):
        periods = periods[:max_periods]
    print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]
    history = {'Period': [], 'Count_cold': [], 'Count_hot': []}
    for prefix in ['cold_', 'hot_']:
        for k in metrics_keys:
            history[prefix + k] = []

    # sampled-eval accumulators
    accum_cold = {k: 0.0 for k in metrics_keys}
    accum_hot = {k: 0.0 for k in metrics_keys}
    count_cold, count_hot = 0, 0

    # full-ranking accumulators
    full_cold = {k: 0.0 for k in metrics_keys}
    full_hot = {k: 0.0 for k in metrics_keys}
    fc_cold, fc_hot = 0, 0

    WARMUP_PERIODS = 1 if fast_dev else 3
    accumulated_dfs = []
    user_seen_items = {}

    for t in range(len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if (not skip_eval) and t >= WARMUP_PERIODS:
            all_item_vecs_eval = build_all_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, n_cold_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type='cold',
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type='hot',
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            if skip_full_eval:
                fmet_cold, fn_c = None, 0
                fmet_hot, fn_h = None, 0
            else:
                fmet_cold, fn_c = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type='cold', full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
                )
                fmet_hot, fn_h = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type='hot', full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
                )

            if met_cold:
                cold_res = met_cold
                for k in metrics_keys: accum_cold[k] += met_cold[k] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for k in metrics_keys: accum_hot[k] += met_hot[k] * n_hot_t
                count_hot += n_hot_t
            if fmet_cold:
                for k in metrics_keys: full_cold[k] += fmet_cold[k] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for k in metrics_keys: full_hot[k] += fmet_hot[k] * fn_h
                fc_hot += fn_h

            c_s = met_cold['R@10'] if met_cold else 0
            h_s = met_hot['R@10'] if met_hot else 0
            c_f = fmet_cold['R@10'] if fmet_cold else 0
            h_f = fmet_hot['R@10'] if fmet_hot else 0
            print(f"  采样 Cold={c_s:.4f} Hot={h_s:.4f} | 全库 Cold={c_f:.4f} Hot={h_f:.4f}")
        else:
            if skip_eval:
                print("  [DEV] Evaluation skipped, training only...")
            else:
                print("  [WARMUP] Training only...")

        history['Period'].append(t)
        history['Count_cold'].append(n_cold_t)
        history['Count_hot'].append(n_hot_t)
        for k in metrics_keys:
            history['cold_' + k].append(cold_res.get(k, 0.0))
            history['hot_' + k].append(hot_res.get(k, 0.0))

        # Update per-user seen interactions for next-period full ranking.
        for u_idx, i_idx in zip(p_df['u_idx'].values, p_df['i_idx'].values):
            uid = int(u_idx)
            if uid not in user_seen_items:
                user_seen_items[uid] = set()
            user_seen_items[uid].add(int(i_idx))
        # --- 累积训练 ---
        accumulated_dfs.append(p_df)
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = (
            (t >= WARMUP_PERIODS) and cfg.use_epoch_early_stop and cfg.n_epochs > 1 and
            (not skip_eval) and (not skip_full_eval)
        )
        es_best = None
        es_best_state = None
        es_best_opt_state = None
        es_no_improve = 0
        for epoch in range(cfg.n_epochs):
            epoch_start = time.time()
            total_loss = 0
            steps = 0
            cand_dup_sum = 0.0
            cand_cov_sum = 0.0
            cand_batches = 0
            sem_delta_sum = 0.0
            sem_shift_sum = 0.0
            sem_distill_sum = 0.0
            sem_align_sum = 0.0
            sem_batches = 0
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
                    user_seen_items=user_seen_items
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                steps += 1
                if cand_info and cand_info.get('steps', 0) > 0:
                    cand_dup_sum += cand_info['dup_rate']
                    cand_cov_sum += cand_info['topm_coverage']
                    cand_batches += 1
                if cand_info and 'semantic_delta_norm' in cand_info:
                    sem_delta_sum += cand_info['semantic_delta_norm']
                    sem_shift_sum += cand_info.get('semantic_shift', 0.0)
                    sem_distill_sum += cand_info.get('semantic_distill', 0.0)
                    sem_align_sum += cand_info.get('semantic_align', 0.0)
                    sem_batches += 1
                if max_train_batches > 0 and (batch_idx + 1) >= max_train_batches:
                    break

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                sem_msg = ""
                if sem_batches > 0:
                    sem_msg = (
                        f" | SemDelta: {sem_delta_sum / sem_batches:.4f}"
                        f" | SemShift: {sem_shift_sum / sem_batches:.4f}"
                        f" | Distill: {sem_distill_sum / sem_batches:.4f}"
                        f" | SemAlign: {sem_align_sum / sem_batches:.4f}"
                    )
                print(
                    f"  [TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}{sem_msg}"
                )
            else:
                sem_msg = ""
                if sem_batches > 0:
                    sem_msg = (
                        f" | SemDelta: {sem_delta_sum / sem_batches:.4f}"
                        f" | SemShift: {sem_shift_sum / sem_batches:.4f}"
                        f" | Distill: {sem_distill_sum / sem_batches:.4f}"
                        f" | SemAlign: {sem_align_sum / sem_batches:.4f}"
                    )
                print(
                    f"  [TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s{sem_msg}"
                )

            if do_early_stop:
                all_item_vecs_es = build_all_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type='cold', full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )
                es_hot, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type='hot', full_ranking=True,
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
                    hot_floor = es_best['hot_r'] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
                    hot_ok = cur_hr >= hot_floor
                    n_improve = cur_n > es_best['cold_n'] + cfg.early_stop_min_delta
                    n_tie = abs(cur_n - es_best['cold_n']) <= cfg.early_stop_min_delta
                    r_tie_break = cur_cr > es_best['cold_r'] + 1e-12
                    is_better = hot_ok and (n_improve or (n_tie and r_tie_break))

                if is_better:
                    es_best = {
                        'epoch': epoch + 1,
                        'cold_n': float(cur_n),
                        'cold_r': float(cur_cr),
                        'hot_r': float(cur_hr)
                    }
                    es_best_state = copy.deepcopy(model.state_dict())
                    es_best_opt_state = copy.deepcopy(optimizer.state_dict())
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                print(
                    f"  [EARLYSTOP] Epoch {epoch+1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
                )

                if es_no_improve >= cfg.early_stop_patience:
                    print(f"  [EARLYSTOP] Triggered at epoch {epoch+1}.")
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

    # ==============================
    # final report
    # ==============================
    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Pure RL-USIM + SemanticDelta)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)

    for m in metrics_keys:
        sc = accum_cold[m] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[m] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[m] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[m] / fc_hot if fc_hot > 0 else 0.0
        print(f"{m:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv(metrics_path, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history['Period'], history['cold_R@10'], marker='o', label='Cold R@10')
    plt.plot(history['Period'], history['hot_R@10'], marker='s', label='Hot R@10')
    plt.axvline(x=WARMUP_PERIODS - 0.5, color='r', linestyle='--', label='Warmup End')
    plt.title('Pure RL-USIM + SemanticDelta: Cumulative Training')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(plot_path)
    print(f">> Saved {plot_path} and {metrics_path}")


if __name__ == "__main__":
    setup_seed(2025)
    main()
