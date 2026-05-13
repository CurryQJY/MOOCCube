"""
Standalone reconstructed original USIM.

Goal:
- fully self-contained single-file entrypoint
- no imports from usim.py
- no course-side signals, no course priors, no course rerank
- keep the current 1+200 sampled + full-ranking evaluation protocol

This is a reconstruction based on the current codebase rather than an exact
historical snapshot.
"""

import copy
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset


def setup_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed fixed: {seed}")


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state):
    if not rng_state:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch_state = rng_state["torch"]
    if torch.is_tensor(torch_state):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_stream_checkpoint(path, **state):
    state["rng_state"] = _capture_rng_state()
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items

        self.emb_dim = 128
        self.hidden_dim = 256
        self.content_dim = content_dim

        self.cold_threshold = 5

        self.lr = 0.0005
        self.temp = 0.07
        self.margin = 0.15

        self.dropout_prob = 0.35
        self.aux_weight = 0.3
        self.train_force_cold = _env_flag("USIM_TRAIN_FORCE_COLD", True)

        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {"value": 0.5, "entropy": 0.01}

        self.usim_steps = 5
        self.n_candidates = 20
        self.usim_lr = 0.3
        self.candidate_strategy = os.environ.get("USIM_CANDIDATE_STRATEGY", "retrieve_sample")
        self.retrieve_top_m = _env_int("USIM_RETRIEVE_TOP_M", 256)
        self.candidate_temp = _env_float("USIM_CANDIDATE_TEMP", 0.20)
        self.candidate_epsilon = _env_float("USIM_CANDIDATE_EPSILON", 0.10)
        self.retrieval_user_chunk = _env_int("USIM_RETRIEVAL_USER_CHUNK", 16384)
        self.retrieval_query_chunk = _env_int("USIM_RETRIEVAL_QUERY_CHUNK", 256)
        self.user_bank_refresh_steps = _env_int("USIM_USER_BANK_REFRESH_STEPS", 200)

        self.n_epochs = _env_int("USIM_N_EPOCHS", 3)
        self.batch_size = _env_int("USIM_BATCH_SIZE", 2048)
        self.accum_steps = 1
        self.eval_n_neg = _env_int("USIM_EVAL_N_NEG", 200)
        self.legacy_eval_pos_from_bank = _env_flag("USIM_LEGACY_EVAL_POS_FROM_BANK", False)

        self.use_mixed_hard_neg = _env_flag("USIM_USE_MIXED_HARD_NEG", False)
        self.train_num_negs = _env_int("USIM_TRAIN_NUM_NEGS", 32)
        self.hard_neg_ratio = _env_float("USIM_HARD_NEG_RATIO", 0.25)
        self.use_structured_hard_neg = _env_flag("USIM_USE_STRUCTURED_HARD_NEG", False)

        # Explicitly disabled for the reconstructed no-course version.
        self.use_course_rerank = False
        self.rerank_alpha = 0.0
        self.rerank_lambda = 0.0
        self.rerank_min_seen = 0
        self.rerank_top_l = 0
        self.rerank_penalty_cap = 0.0
        self.rerank_only_cold = True

        self.use_prereq_aux_loss = False
        self.prereq_aux_weight = 0.0
        self.prereq_aux_margin = 0.0
        self.prereq_aux_violation_thr = 1.0
        self.prereq_aux_min_seen = 10 ** 9
        self.prereq_aux_only_cold = True

        self.use_epoch_early_stop = _env_flag("USIM_USE_EPOCH_EARLY_STOP", False)
        self.early_stop_k = 10
        self.early_stop_patience = 1
        self.early_stop_min_delta = 1e-4
        self.early_stop_hot_r10_drop_tol = 0.03


class SimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=4):
        super().__init__()
        input_dim = item_dim + time_dim
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
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


class PAM_RL_Pure_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_emb = nn.Embedding(config.n_users, config.emb_dim)
        self.item_id_emb = nn.Embedding(config.n_items, config.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        self.content_proj = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim),
        )

        self.user_proj = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim),
        )

        self.llm_proj = nn.Linear(1, config.emb_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid(),
        )
        self.agent = SimpleAC(config.emb_dim, time_dim=4)

        self.item_hard_adj = None

    def set_course_artifacts(self, artifacts):
        return

    def apply_course_rerank(self, scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=None):
        return scores

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
        batch_size = item_emb.size(0)
        n_cand = self.cfg.n_candidates
        strategy = self.cfg.candidate_strategy

        if strategy != "retrieve_sample":
            rand_idx = torch.randint(0, self.cfg.n_users, (batch_size, n_cand), device=self.device)
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

        replacement = top_m < n_cand
        sample_pos = torch.multinomial(probs, num_samples=n_cand, replacement=replacement)
        cand_idx = top_idx.gather(1, sample_pos)
        cand_emb = user_bank_raw[cand_idx].detach()

        topm_unique = max(1, int(top_idx.unique().numel()))
        selected_unique = int(cand_idx.unique().numel())
        selected_total = max(1, int(cand_idx.numel()))
        dup_rate = 1.0 - (selected_unique / selected_total)
        topm_cov = selected_unique / topm_unique
        stats = {"dup_rate": float(dup_rate), "topm_coverage": float(topm_cov)}
        return cand_emb, cand_idx, stats

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        id_e = self.item_id_emb(i_idx)
        if torch.is_tensor(force_cold):
            force_mask = force_cold.to(device=id_e.device)
            if force_mask.dtype != torch.bool:
                force_mask = force_mask > 0
            force_mask = force_mask.view(-1)
            if force_mask.any():
                id_e = id_e.clone()
                id_e[force_mask] = 0.0
        elif force_cold:
            id_e = torch.zeros_like(id_e)

        if self.training and random.random() < float(self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        content_e = self.content_proj(self.item_con_emb(i_idx))
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        content_e = content_e + llm_e

        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        item_fused = alpha * id_e + (1 - alpha) * content_e
        return item_fused, id_e, content_e

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
        candidate_stats = {"dup_rate": 0.0, "topm_coverage": 0.0, "steps": 0}

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

            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_emb is not None:
                dist = F.mse_loss(current_h, target_emb, reduction="none").mean(dim=1, keepdim=True)
                reward = -dist * 10.0

            trajectory["log_probs"].append(log_prob.detach())
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

        if candidate_stats["steps"] > 0:
            candidate_stats["dup_rate"] /= candidate_stats["steps"]
            candidate_stats["topm_coverage"] /= candidate_stats["steps"]
        return current_h, trajectory, candidate_stats

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory["rewards"]).squeeze(-1)
        old_log_probs = torch.stack(trajectory["log_probs"])
        states = trajectory["states"]
        time_steps = trajectory["time_steps"]
        candidates = trajectory["candidates"]
        actions = trajectory["actions"]

        returns = []
        ret = 0
        for r in reversed(rewards):
            ret = r + self.cfg.ppo_gamma * ret
            returns.insert(0, ret)
        returns = torch.stack(returns).detach()

        total_ppo_loss = 0.0
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

            advantage = (returns - new_values).detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values).pow(2).mean()
            entropy_loss = -new_entropies.mean()
            total_ppo_loss += (
                actor_loss
                + self.cfg.ppo_coeffs["value"] * critic_loss
                + self.cfg.ppo_coeffs["entropy"] * entropy_loss
            )

        return total_ppo_loss / self.cfg.ppo_epochs

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold

        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)

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

        total_loss = main_loss + self.cfg.aux_weight * aux_loss + ppo_loss
        return total_loss, candidate_stats


def compute_ranking_metrics(scores, target_indices, k_list=None):
    if k_list is None:
        k_list = [5, 10, 20]
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
        results[f"R@{k}"] = recall
        results[f"N@{k}"] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results


def split_dataframe_by_periods(df, period_type="M"):
    work_df = df.copy()
    if not np.issubdtype(work_df["timestamp"].dtype, np.datetime64):
        work_df["datetime"] = pd.to_datetime(work_df["timestamp"], unit="s")
    else:
        work_df["datetime"] = work_df["timestamp"]
    work_df["period_id"] = work_df["datetime"].dt.to_period(period_type)
    periods = []
    for period_key in sorted(work_df["period_id"].unique()):
        periods.append(work_df[work_df["period_id"] == period_key].reset_index(drop=True))
    return periods


def _lookup_llm_score(llm_scores, item_idx, user_idx=None):
    if llm_scores is None:
        return -1.0

    item_score = llm_scores.get(int(item_idx))
    if item_score is not None:
        return float(item_score)
    return -1.0


def _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=None):
    values = [
        _lookup_llm_score(llm_scores, item_idx)
        for user_idx, item_idx in zip(user_ids, item_ids)
    ]
    return torch.tensor(values, dtype=torch.float, device=device)


def build_all_item_vecs(model, device, llm_scores, item_batch=1024, force_cold=True):
    n_items = model.cfg.n_items
    all_item_idx = torch.arange(n_items, device=device)
    all_llm_s = torch.tensor(
        [_lookup_llm_score(llm_scores, int(idx)) for idx in all_item_idx],
        dtype=torch.float,
        device=device,
    )

    all_item_vecs = []
    with torch.no_grad():
        for start in range(0, n_items, item_batch):
            end = min(start + item_batch, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]
            z_i, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=force_cold)
            all_item_vecs.append(F.normalize(z_i, dim=1))
    return torch.cat(all_item_vecs, dim=0)


def build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    return build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=True)


def _select_eval_item_bank(all_item_vecs, eval_type):
    if isinstance(all_item_vecs, dict):
        if eval_type in all_item_vecs:
            return all_item_vecs[eval_type]
        if eval_type == "all" and "hot" in all_item_vecs:
            return all_item_vecs["hot"]
        if "cold" in all_item_vecs:
            return all_item_vecs["cold"]
    return all_item_vecs


def _build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    if item_idx.numel() < 1:
        return torch.empty((0, model.cfg.emb_dim), device=item_idx.device)

    if eval_type == "hot":
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=False)
    else:
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=True)
    return F.normalize(pos_vec, dim=1)


def evaluate_usim(
    model,
    loader,
    device,
    llm_scores,
    k_list=None,
    n_neg=200,
    eval_type="cold",
    full_ranking=False,
    user_seen_items=None,
    all_item_vecs=None,
):
    if k_list is None:
        k_list = [5, 10, 20]

    model.eval()
    accum_metrics = {}
    total_samples = 0
    seen_tensor_cache = {}

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        if all_item_vecs is None:
            all_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
        item_bank = _select_eval_item_bank(all_item_vecs, eval_type)

        for batch, pop, llm in loader:
            if eval_type == "cold":
                mask = pop < model.cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u = batch["u"][mask].to(device)
            i = batch["i"][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            item_ids = [int(x) for x in i.detach().cpu().tolist()]

            for uid in user_ids:
                if uid in seen_tensor_cache:
                    continue
                seen_items = user_seen_items.get(uid) if user_seen_items else None
                if seen_items:
                    seen_list = [it for it in seen_items if 0 <= it < n_items]
                    seen_tensor_cache[uid] = (
                        torch.tensor(seen_list, dtype=torch.long, device=device) if seen_list else None
                    )
                else:
                    seen_tensor_cache[uid] = None

            z_u = F.normalize(model.user_proj(model.user_emb(u)), dim=1)
            legacy_pos_from_bank = bool(getattr(model.cfg, "legacy_eval_pos_from_bank", False))
            if legacy_pos_from_bank:
                pos_vec = None
                pos_scores = None
            else:
                pos_llm = _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=device)
                pos_vec = _build_eval_pos_item_vecs(model, i, pos_llm, pop_sel, eval_type)
                pos_scores = (z_u * pos_vec).sum(dim=1)

            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                row_idx = torch.arange(n_sel, device=device)
                if legacy_pos_from_bank:
                    target_scores = scores[row_idx, i].clone()
                else:
                    target_scores = pos_scores.clone()
                if user_seen_items:
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache[uid]
                        if seen_idx is None:
                            continue
                        scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                else:
                    scores[row_idx, i] = target_scores
                target_indices = i
            else:
                neg_items = torch.randint(0, n_items, (n_sel, n_neg), device=device)
                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                cand_vecs = item_bank[cand_idx]
                if not legacy_pos_from_bank:
                    cand_vecs = cand_vecs.clone()
                    cand_vecs[:, 0, :] = pos_vec
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)
            for key, value in batch_res.items():
                accum_metrics[key] = accum_metrics.get(key, 0.0) + value * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0
    return {key: value / total_samples for key, value in accum_metrics.items()}, total_samples


class StreamDataset(Dataset):
    def __init__(self, df, llm_scores):
        user_ids = [int(x) for x in df["u_idx"].values]
        item_ids = [int(x) for x in df["i_idx"].values]
        self.u = torch.tensor(user_ids, dtype=torch.long)
        self.i = torch.tensor(item_ids, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)
        self.llm_s = _build_llm_score_tensor(llm_scores, user_ids, item_ids)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {"u": self.u[idx], "i": self.i[idx], "pop": self.pop[idx], "llm": self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item["u"] for item in batch])
    i = torch.stack([item["i"] for item in batch])
    pop = torch.stack([item["pop"] for item in batch])
    llm = torch.stack([item["llm"] for item in batch])
    return {"u": u, "i": i}, pop, llm


def _add_user_seen_from_df(user_seen_items, src_df):
    for u_idx, i_idx in zip(src_df["u_idx"].values, src_df["i_idx"].values):
        uid = int(u_idx)
        if uid not in user_seen_items:
            user_seen_items[uid] = set()
        user_seen_items[uid].add(int(i_idx))
    return user_seen_items


def _clone_user_seen(user_seen_items):
    return {uid: set(items) for uid, items in user_seen_items.items()}


def _save_stream_exports(
    metrics_keys,
    sampled_cold,
    sampled_hot,
    full_cold,
    full_hot,
    sampled_cold_count,
    sampled_hot_count,
    full_cold_count,
    full_hot_count,
):
    output_tag = os.environ.get("USIM_OUTPUT_TAG", "original_reconstructed_standalone")
    detail_path = f"final_report_usim_{output_tag}.csv"
    fullrank_path = f"final_fullrank_usim_{output_tag}.csv"

    detail_rows = []
    for key in metrics_keys:
        detail_rows.append(
            {
                "metric": key,
                "sampled_cold": float(sampled_cold.get(key, 0.0)),
                "sampled_hot": float(sampled_hot.get(key, 0.0)),
                "full_cold": float(full_cold.get(key, 0.0)),
                "full_hot": float(full_hot.get(key, 0.0)),
            }
        )
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    fullrank_row = {
        "model": f"USIM-{output_tag}",
        "protocol": "stream",
        "full_cold_r5": float(full_cold.get("R@5", 0.0)),
        "full_cold_r10": float(full_cold.get("R@10", 0.0)),
        "full_cold_r20": float(full_cold.get("R@20", 0.0)),
        "full_cold_n5": float(full_cold.get("N@5", 0.0)),
        "full_cold_n10": float(full_cold.get("N@10", 0.0)),
        "full_cold_n20": float(full_cold.get("N@20", 0.0)),
        "full_hot_r5": float(full_hot.get("R@5", 0.0)),
        "full_hot_r10": float(full_hot.get("R@10", 0.0)),
        "full_hot_r20": float(full_hot.get("R@20", 0.0)),
        "full_hot_n5": float(full_hot.get("N@5", 0.0)),
        "full_hot_n10": float(full_hot.get("N@10", 0.0)),
        "full_hot_n20": float(full_hot.get("N@20", 0.0)),
        "sampled_cold_count": int(sampled_cold_count),
        "sampled_hot_count": int(sampled_hot_count),
        "full_cold_count": int(full_cold_count),
        "full_hot_count": int(full_hot_count),
        "notes": "fully standalone reconstructed no-course USIM",
    }
    pd.DataFrame([fullrank_row]).to_csv(fullrank_path, index=False)
    return detail_path, fullrank_path


def run_static_experiment(df, cfg, device, model, optimizer, llm_scores):
    static_seed = int(os.environ.get("USIM_STATIC_SEED", "2025"))
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))
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

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]

    do_early_stop = cfg.use_epoch_early_stop and cfg.n_epochs > 1
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

        optimizer.zero_grad()
        cached_user_bank = None
        if cfg.candidate_strategy == "retrieve_sample":
            cached_user_bank = model._build_user_bank_raw()

        for batch_idx, (batch, pop, llm) in enumerate(train_loader):
            if (
                cached_user_bank is not None
                and cfg.user_bank_refresh_steps > 0
                and batch_idx > 0
                and (batch_idx % cfg.user_bank_refresh_steps == 0)
            ):
                cached_user_bank = model._build_user_bank_raw()

            batch = {k: v.to(device) for k, v in batch.items()}
            loss, cand_info = model(
                batch,
                pop.to(device),
                llm.to(device),
                user_bank_raw=cached_user_bank,
                user_seen_items=train_seen,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            if cand_info and cand_info.get("steps", 0) > 0:
                cand_dup_sum += cand_info["dup_rate"]
                cand_cov_sum += cand_info["topm_coverage"]
                cand_batches += 1

        epoch_sec = time.time() - epoch_start
        avg_loss = total_loss / max(1, steps)
        if cand_batches > 0:
            avg_dup = cand_dup_sum / cand_batches
            avg_cov = cand_cov_sum / cand_batches
            print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
            )
        else:
            print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
            )

        if do_early_stop:
            all_item_vecs_val = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            val_cold, _ = evaluate_usim(
                model,
                val_loader,
                device,
                llm_scores,
                k_list,
                eval_type="cold",
                full_ranking=True,
                user_seen_items=train_seen,
                all_item_vecs=all_item_vecs_val,
            )
            val_hot, _ = evaluate_usim(
                model,
                val_loader,
                device,
                llm_scores,
                k_list,
                eval_type="hot",
                full_ranking=True,
                user_seen_items=train_seen,
                all_item_vecs=all_item_vecs_val,
            )

            key_n = f"N@{cfg.early_stop_k}"
            key_r = f"R@{cfg.early_stop_k}"
            cur_n = val_cold.get(key_n, 0.0) if val_cold else 0.0
            cur_cr = val_cold.get(key_r, 0.0) if val_cold else 0.0
            cur_hr = val_hot.get(key_r, 0.0) if val_hot else 0.0

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
                f"  [STATIC-EARLYSTOP] Epoch {epoch + 1}: "
                f"Full Cold {key_n}={cur_n:.4f}, Full Cold {key_r}={cur_cr:.4f}, "
                f"Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
            )

            if es_no_improve >= cfg.early_stop_patience:
                print(f"  [STATIC-EARLYSTOP] Triggered at epoch {epoch + 1}.")
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

    all_item_vecs_test = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
    met_cold, n_cold_t = evaluate_usim(
        model,
        test_loader,
        device,
        llm_scores,
        k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        user_seen_items=test_seen,
        all_item_vecs=all_item_vecs_test,
    )
    met_hot, n_hot_t = evaluate_usim(
        model,
        test_loader,
        device,
        llm_scores,
        k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        user_seen_items=test_seen,
        all_item_vecs=all_item_vecs_test,
    )
    fmet_cold, fn_c = evaluate_usim(
        model,
        test_loader,
        device,
        llm_scores,
        k_list,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=test_seen,
        all_item_vecs=all_item_vecs_test,
    )
    fmet_hot, fn_h = evaluate_usim(
        model,
        test_loader,
        device,
        llm_scores,
        k_list,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=test_seen,
        all_item_vecs=all_item_vecs_test,
    )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT (STATIC): 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)

    for key in metrics_keys:
        sc = met_cold.get(key, 0.0) if met_cold else 0.0
        sh = met_hot.get(key, 0.0) if met_hot else 0.0
        fc = fmet_cold.get(key, 0.0) if fmet_cold else 0.0
        fh = fmet_hot.get(key, 0.0) if fmet_hot else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"全库 Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for standalone reconstructed original USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    llm_scores = pd.read_pickle(f"{data_dir}/llm_scores.pkl")
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = Config(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PAM_RL_Pure_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Standalone Reconstructed Original USIM | Batch Size={cfg.batch_size}")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> Course Flags Disabled: rerank={cfg.use_course_rerank} | "
        f"prereq_aux={cfg.use_prereq_aux_loss} | "
        f"structured_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> Legacy knobs: train_force_cold={cfg.train_force_cold} | "
        f"eval_n_neg={cfg.eval_n_neg} | "
        f"legacy_eval_pos_from_bank={cfg.legacy_eval_pos_from_bank} | "
        f"mixed_hard_neg={cfg.use_mixed_hard_neg}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"tie=Full Cold R@{cfg.early_stop_k} | hot_drop_tol={cfg.early_stop_hot_r10_drop_tol:.2%} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

    use_static = os.environ.get("USIM_STATIC", "0") == "1"
    if use_static:
        print(">> STATIC mode enabled.")
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
    start_period = 0

    checkpoint_path = os.environ.get(
        "USIM_STREAM_CKPT",
        "usim_original_reconstructed_standalone.stream_ckpt.pt",
    )
    resume_enabled = os.environ.get("USIM_RESUME", "0") == "1"
    checkpoint_enabled = resume_enabled or os.environ.get("USIM_CHECKPOINT", "0") == "1"
    if resume_enabled and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        accum_cold = ckpt["accum_cold"]
        accum_hot = ckpt["accum_hot"]
        count_cold = ckpt["count_cold"]
        count_hot = ckpt["count_hot"]
        full_cold = ckpt["full_cold"]
        full_hot = ckpt["full_hot"]
        fc_cold = ckpt["fc_cold"]
        fc_hot = ckpt["fc_hot"]
        start_period = int(ckpt.get("next_period", 0))
        start_period = max(0, min(start_period, len(periods)))
        accumulated_dfs = [periods[i] for i in range(start_period)]
        user_seen_items = {}
        for old_df in accumulated_dfs:
            _add_user_seen_from_df(user_seen_items, old_df)
        _restore_rng_state(ckpt.get("rng_state"))
        print(f">> Resumed stream checkpoint from period {start_period}: {checkpoint_path}")

    for t in range(start_period, len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        cum_size = sum(len(d) for d in accumulated_dfs) + n_total
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {cum_size}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= warmup_periods:
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
            cand_batches = 0
            optimizer.zero_grad()
            cached_user_bank = None
            if cfg.candidate_strategy == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
                if (
                    cached_user_bank is not None
                    and cfg.user_bank_refresh_steps > 0
                    and batch_idx > 0
                    and (batch_idx % cfg.user_bank_refresh_steps == 0)
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

                total_loss += loss.item()
                steps += 1
                if cand_info and cand_info.get("steps", 0) > 0:
                    cand_dup_sum += cand_info["dup_rate"]
                    cand_cov_sum += cand_info["topm_coverage"]
                    cand_batches += 1

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if do_early_stop:
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

        if checkpoint_enabled:
            _save_stream_checkpoint(
                checkpoint_path,
                next_period=t + 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                history=history,
                accum_cold=accum_cold,
                accum_hot=accum_hot,
                count_cold=count_cold,
                count_hot=count_hot,
                full_cold=full_cold,
                full_hot=full_hot,
                fc_cold=fc_cold,
                fc_hot=fc_hot,
            )
            print(f"  [CKPT] Saved checkpoint for next period {t + 1}: {checkpoint_path}")

    final_sampled_cold = {
        key: (accum_cold[key] / count_cold if count_cold > 0 else 0.0)
        for key in metrics_keys
    }
    final_sampled_hot = {
        key: (accum_hot[key] / count_hot if count_hot > 0 else 0.0)
        for key in metrics_keys
    }
    final_full_cold = {
        key: (full_cold[key] / fc_cold if fc_cold > 0 else 0.0)
        for key in metrics_keys
    }
    final_full_hot = {
        key: (full_hot[key] / fc_hot if fc_hot > 0 else 0.0)
        for key in metrics_keys
    }

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Standalone Reconstructed Original USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        print(
            f"{key:<10} | {final_sampled_cold[key]:<12.4f} | {final_sampled_hot[key]:<12.4f} | "
            f"{final_full_cold[key]:<12.4f} | {final_full_hot[key]:<12.4f}"
        )
    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    output_tag = os.environ.get("USIM_OUTPUT_TAG", "original_reconstructed_standalone")
    metrics_path = f"mooc_metrics_pure_usim_{output_tag}.csv"
    plot_path = f"mooc_result_pure_usim_{output_tag}.png"
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Standalone Reconstructed Original USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(plot_path)

    detail_path, fullrank_path = _save_stream_exports(
        metrics_keys=metrics_keys,
        sampled_cold=final_sampled_cold,
        sampled_hot=final_sampled_hot,
        full_cold=final_full_cold,
        full_hot=final_full_hot,
        sampled_cold_count=count_cold,
        sampled_hot_count=count_hot,
        full_cold_count=fc_cold,
        full_hot_count=fc_hot,
    )
    print(f">> Saved {plot_path}, {metrics_path}, {detail_path}, and {fullrank_path}")
    if checkpoint_enabled and os.path.exists(checkpoint_path) and os.environ.get("USIM_KEEP_CHECKPOINT", "0") != "1":
        os.remove(checkpoint_path)
        print(f">> Removed completed checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    setup_seed(2025)
    main()
