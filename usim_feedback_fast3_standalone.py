"""
usim_feedback_fast3_standalone.py - standalone FAST3 variant

This file no longer depends on usim.py, usim_feedback.py, or
usim_feedback_fast.py. The required model, data, evaluation, course-graph,
checkpoint, and reporting utilities are all inlined here.
"""
import copy
import json
import math
import os
import random
import re
import time
from collections import OrderedDict, defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _llm_score_path(candidate):
    if candidate.endswith(".pkl"):
        return candidate
    return os.path.join(candidate, "llm_scores.pkl")


def _summarize_llm_score_compatibility(llm_scores, df, cold_threshold=5, n_users=None, n_items=None):
    df_pairs = set(zip(df["u_idx"].astype(int), df["i_idx"].astype(int)))
    cold_df = df[df["popularity"] < int(cold_threshold)]
    cold_pairs = set(zip(cold_df["u_idx"].astype(int), cold_df["i_idx"].astype(int)))

    stats = {
        "pair_total": 0,
        "pair_in_range": 0,
        "pair_in_df": 0,
        "pair_cold_hits": 0,
        "item_total": 0,
        "item_in_range": 0,
    }

    for key in llm_scores.keys():
        if isinstance(key, tuple) and len(key) == 2:
            try:
                u_idx = int(key[0])
                i_idx = int(key[1])
            except (TypeError, ValueError):
                continue
            stats["pair_total"] += 1
            in_range = True
            if n_users is not None:
                in_range = in_range and 0 <= u_idx < int(n_users)
            if n_items is not None:
                in_range = in_range and 0 <= i_idx < int(n_items)
            if in_range:
                stats["pair_in_range"] += 1
            pair = (u_idx, i_idx)
            if pair in df_pairs:
                stats["pair_in_df"] += 1
                if pair in cold_pairs:
                    stats["pair_cold_hits"] += 1
        else:
            try:
                item_idx = int(key)
            except (TypeError, ValueError):
                continue
            stats["item_total"] += 1
            if n_items is None or 0 <= item_idx < int(n_items):
                stats["item_in_range"] += 1

    pair_total = max(1, stats["pair_total"])
    item_total = max(1, stats["item_total"])
    stats["pair_match_ratio"] = stats["pair_in_df"] / pair_total
    stats["pair_cold_ratio"] = stats["pair_cold_hits"] / pair_total
    stats["pair_in_range_ratio"] = stats["pair_in_range"] / pair_total
    stats["item_in_range_ratio"] = stats["item_in_range"] / item_total
    return stats


def load_llm_scores_for_stream(
    data_dir,
    df,
    cold_threshold=5,
    n_users=None,
    n_items=None,
    fallback_data_dirs=None,
    verbose=True,
):
    candidates = [_llm_score_path(data_dir)]
    for candidate in fallback_data_dirs or []:
        path = _llm_score_path(candidate)
        if path not in candidates:
            candidates.append(path)

    loaded = []
    for idx, path in enumerate(candidates):
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            llm_scores = pd.read_pickle(f)
        stats = _summarize_llm_score_compatibility(
            llm_scores,
            df,
            cold_threshold=cold_threshold,
            n_users=n_users,
            n_items=n_items,
        )
        loaded.append(
            {
                "path": path,
                "scores": llm_scores,
                "stats": stats,
                "is_primary": idx == 0,
            }
        )

    if not loaded:
        if verbose:
            print(f"   No llm_scores.pkl found for {data_dir}.")
        return {}, None, None

    primary = loaded[0]
    best = max(
        loaded,
        key=lambda x: (
            x["stats"]["pair_cold_hits"],
            x["stats"]["pair_in_df"],
            x["stats"]["pair_in_range"],
            -x["stats"]["pair_total"],
        ),
    )

    primary_stats = primary["stats"]
    primary_bad = (
        primary_stats["pair_total"] > 0
        and (
            primary_stats["pair_in_df"] == 0
            or (
                primary_stats["pair_match_ratio"] < 0.05
                and primary_stats["pair_cold_hits"] == 0
            )
        )
    )

    chosen = best if (best is not primary and primary_bad) else primary

    if verbose:
        for entry in loaded:
            stats = entry["stats"]
            label = "primary" if entry["is_primary"] else "fallback"
            print(
                "   LLM score candidate "
                f"[{label}] {entry['path']}: "
                f"pairs={stats['pair_total']}, "
                f"in_df={stats['pair_in_df']}, "
                f"cold_hits={stats['pair_cold_hits']}, "
                f"in_range={stats['pair_in_range']}"
            )
        if chosen is not primary:
            print(
                "   Warning: primary llm_scores.pkl looks incompatible with the current stream; "
                f"using {chosen['path']} instead."
            )

    strict_mode = os.environ.get("USIM_LLM_SCORE_STRICT", "0") == "1"
    if strict_mode and chosen is not primary:
        raise ValueError(
            "Primary llm_scores.pkl is incompatible with the current stream. "
            "Set USIM_LLM_SCORE_STRICT=0 to allow fallback, or regenerate matching llm_scores.pkl."
        )

    return chosen["scores"], chosen["path"], chosen["stats"]


def _resolve_torch_device():
    force_cpu = os.environ.get("USIM_FORCE_CPU", "0") == "1"
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseConfig:
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
        self.train_force_cold = os.environ.get("USIM_TRAIN_FORCE_COLD", "1") == "1"
        self.llm_safe_mode = os.environ.get("USIM_LLM_SAFE_MODE", "0") == "1"
        self.llm_weight = float(
            os.environ.get("USIM_LLM_WEIGHT", "0.20" if self.llm_safe_mode else "1.0")
        )
        self.llm_cold_only = os.environ.get(
            "USIM_LLM_COLD_ONLY",
            "1" if self.llm_safe_mode else "0",
        ) == "1"
        self.llm_bank_mode = os.environ.get(
            "USIM_LLM_BANK_MODE",
            "none" if self.llm_safe_mode else "item",
        ).strip().lower()
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {"value": 0.5, "entropy": 0.01}
        self.usim_steps = int(os.environ.get("USIM_STEPS", "5"))
        self.n_candidates = int(os.environ.get("USIM_N_CANDIDATES", "20"))
        self.usim_lr = 0.3
        self.candidate_strategy = "retrieve_sample"
        self.retrieve_top_m = int(os.environ.get("USIM_RETRIEVE_TOP_M", "256"))
        self.candidate_temp = 0.20
        self.candidate_epsilon = 0.10
        self.retrieval_user_chunk = 16384
        self.retrieval_query_chunk = 256
        self.user_bank_refresh_steps = 200
        self.n_epochs = int(os.environ.get("USIM_N_EPOCHS", "3"))
        self.batch_size = int(os.environ.get("USIM_BATCH_SIZE", "2048"))
        self.accum_steps = 1
        self.eval_n_neg = 200
        self.use_mixed_hard_neg = True
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        self.use_structured_hard_neg = False
        self.use_course_rerank = False
        self.rerank_alpha = 0.00
        self.rerank_lambda = 0.01
        self.rerank_min_seen = 8
        self.rerank_top_l = 50
        self.rerank_penalty_cap = 0.10
        self.rerank_only_cold = True
        self.prereq_min_support = 30
        self.prereq_max_per_item = 5
        self.prereq_min_items = 1
        self.prereq_max_forward = 20
        self.concept_overlap_mode = os.environ.get("USIM_CONCEPT_OVERLAP_MODE", "plain").strip().lower()
        self.prereq_graph_source = os.environ.get("USIM_PREREQ_GRAPH_SOURCE", "behavior").strip().lower()
        self.prereq_concept_score_thr = float(os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", "0.10"))
        self.prereq_concept_min_hits = int(os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", "1"))
        self.prereq_concept_file = os.environ.get("USIM_PREREQ_CONCEPT_FILE", "prerequisite-dependency.json")
        self.use_prereq_aux_loss = True
        self.prereq_aux_weight = 0.03
        self.prereq_aux_margin = 0.05
        self.prereq_aux_violation_thr = 0.60
        self.prereq_aux_min_seen = 5
        self.prereq_aux_only_cold = True
        self.use_epoch_early_stop = os.environ.get("USIM_USE_EPOCH_EARLY_STOP", "1") == "1"
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
        self.device = _resolve_torch_device()
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
        self.item_prereq_item_mat = None
        self.item_prereq_item_cnt = None
        self.item_concept_overlap = None
        self.item_video_contain = None
        self.item_same_family = None

    def set_course_artifacts(self, artifacts):
        if not artifacts:
            return
        self.item_hard_adj = artifacts.get("item_hard_adj")
        self.item_prereq_item_mat = artifacts.get("item_prereq_item_mat")
        self.item_prereq_item_cnt = artifacts.get("item_prereq_item_cnt")
        self.item_concept_overlap = artifacts.get("item_concept_overlap")
        self.item_video_contain = artifacts.get("item_video_contain")
        self.item_same_family = artifacts.get("item_same_family")
        if self.item_hard_adj is not None:
            self.item_hard_adj = self.item_hard_adj.to(self.device)
        if self.item_prereq_item_mat is not None:
            self.item_prereq_item_mat = self.item_prereq_item_mat.to(self.device)
        if self.item_prereq_item_cnt is not None:
            self.item_prereq_item_cnt = self.item_prereq_item_cnt.to(self.device)
        if self.item_concept_overlap is not None:
            self.item_concept_overlap = self.item_concept_overlap.to(self.device)
        if self.item_video_contain is not None:
            self.item_video_contain = self.item_video_contain.to(self.device)
        if self.item_same_family is not None:
            self.item_same_family = self.item_same_family.to(self.device)

    def _compute_structural_redundancy_profile(self, user_ids, user_seen_items):
        batch_size = int(len(user_ids)) if not torch.is_tensor(user_ids) else int(user_ids.numel())
        profile = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if user_seen_items is None:
            return profile
        if self.item_video_contain is None and self.item_same_family is None:
            return profile

        if torch.is_tensor(user_ids):
            uid_tensor = user_ids.to(self.device).view(-1).long()
        else:
            uid_tensor = torch.tensor([int(x) for x in user_ids], dtype=torch.long, device=self.device)

        if uid_tensor.numel() < 1:
            return profile

        unique_uids, inverse = torch.unique(uid_tensor, sorted=False, return_inverse=True)
        unique_profiles = torch.zeros((unique_uids.numel(), self.cfg.n_items), dtype=torch.float32, device=self.device)
        video_min = float(min(0.999, max(0.0, getattr(self.cfg, "feedback_course_struct_video_min", 0.60))))
        video_band = max(1e-6, 1.0 - video_min)

        for row, uid in enumerate(unique_uids.detach().cpu().tolist()):
            seen = user_seen_items.get(int(uid))
            if not seen:
                continue
            seen_idx = [int(it) for it in seen if 0 <= int(it) < self.cfg.n_items]
            if len(seen_idx) < 1:
                continue
            seen_idx = torch.tensor(seen_idx, dtype=torch.long, device=self.device)

            hard = torch.zeros(self.cfg.n_items, dtype=torch.float32, device=self.device)
            if self.item_same_family is not None:
                hard = torch.maximum(hard, self.item_same_family[:, seen_idx].amax(dim=1).float())

            soft = torch.zeros_like(hard)
            if self.item_video_contain is not None:
                video_cover = self.item_video_contain[:, seen_idx].amax(dim=1)
                soft = ((video_cover - video_min) / video_band).clamp(0.0, 1.0)

            unique_profiles[row] = torch.maximum(hard, soft)

        return unique_profiles[inverse]

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
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)
        seen_cnt_raw = seen_mat.sum(dim=1, keepdim=True)
        seen_cnt = seen_cnt_raw.clamp_min(1.0)
        concept_match = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt
        user_mask = torch.ones_like(seen_cnt_raw)
        if getattr(self.cfg, "rerank_min_seen", 0) > 0:
            user_mask = (seen_cnt_raw >= float(self.cfg.rerank_min_seen)).float()
        row_mask = torch.ones_like(seen_cnt_raw)
        if target_pop is not None and getattr(self.cfg, "rerank_only_cold", False):
            row_mask = (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()
        active_mask = user_mask * row_mask
        if active_mask.sum().item() < 1:
            return scores
        penalty = (self.cfg.rerank_lambda * violation).clamp(min=0.0, max=float(getattr(self.cfg, "rerank_penalty_cap", 1.0)))
        adjust_full = (self.cfg.rerank_alpha * concept_match - penalty) * active_mask
        top_l = int(getattr(self.cfg, "rerank_top_l", 0))
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

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        id_e_true = self.item_id_emb(i_idx)
        id_e = id_e_true
        batch_size = id_e.size(0)
        mask_id = torch.zeros((batch_size, 1), dtype=torch.bool, device=id_e.device)
        if isinstance(force_cold, torch.Tensor):
            force_mask = force_cold.to(device=id_e.device)
            if force_mask.dtype != torch.bool:
                force_mask = force_mask > 0
            mask_id = mask_id | force_mask.view(-1, 1)
        elif force_cold:
            mask_id = torch.ones((batch_size, 1), dtype=torch.bool, device=id_e.device)
        if self.training and self.cfg.dropout_prob > 0:
            dropout_mask = torch.rand((batch_size, 1), device=id_e.device) < float(self.cfg.dropout_prob)
            mask_id = mask_id | dropout_mask
        if mask_id.any():
            id_e = torch.where(mask_id, torch.zeros_like(id_e), id_e)
        content_e = self.content_proj(self.item_con_emb(i_idx))
        llm_weight = float(getattr(self.cfg, "llm_weight", 0.0))
        if llm_weight > 0.0:
            mask_llm = (llm_s > -0.5).float().unsqueeze(1)
            if getattr(self.cfg, "llm_cold_only", False):
                if isinstance(force_cold, torch.Tensor):
                    cold_mask = force_cold.to(device=id_e.device)
                    if cold_mask.dtype != torch.bool:
                        cold_mask = cold_mask > 0
                    cold_mask = cold_mask.float().view(-1, 1)
                elif force_cold:
                    cold_mask = torch.ones_like(mask_llm)
                else:
                    cold_mask = torch.zeros_like(mask_llm)
                mask_llm = mask_llm * cold_mask
            val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
            llm_e = self.llm_proj(val_llm) * mask_llm
            content_e = content_e + llm_weight * llm_e
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        item_fused = alpha * id_e + (1 - alpha) * content_e
        return item_fused, id_e_true, content_e


class FeedbackConfig(BaseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_load_course_artifacts = os.environ.get("USIM_FB_LOAD_COURSE_ARTIFACTS", "1") == "1"
        self.reward_terminal_weight = float(os.environ.get("USIM_FB_REWARD_TERM_W", "10.0"))
        self.reward_gain_weight = float(os.environ.get("USIM_FB_REWARD_GAIN_W", "5.0"))
        self.reward_gain_clip = float(os.environ.get("USIM_FB_REWARD_GAIN_CLIP", "0.05"))
        self.reward_dup_penalty_weight = float(os.environ.get("USIM_FB_REWARD_DUP_W", "0.50"))
        self.reward_cov_bonus_weight = float(os.environ.get("USIM_FB_REWARD_COV_W", "0.00"))
        self.feedback_course_only_cold = os.environ.get("USIM_FB_COURSE_ONLY_COLD", "1") == "1"
        self.feedback_course_warm_seen = int(os.environ.get("USIM_FB_COURSE_WARM_SEEN", "5"))
        self.feedback_course_concept_min = float(os.environ.get("USIM_FB_COURSE_CONCEPT_MIN", "0.12"))
        self.feedback_course_redundant_mode = os.environ.get(
            "USIM_FB_COURSE_REDUNDANT_MODE",
            "concept",
        ).strip().lower()
        self.feedback_course_redundant_thr = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_THR", "0.70"))
        self.feedback_course_struct_video_min = float(
            os.environ.get("USIM_FB_COURSE_STRUCT_VIDEO_MIN", "0.60")
        )
        self.feedback_course_prereq_gate = float(os.environ.get("USIM_FB_COURSE_PREREQ_GATE", "0.20"))
        self.feedback_course_prereq_weight = float(os.environ.get("USIM_FB_COURSE_PREREQ_W", "0.08"))
        self.feedback_prereq_weighted_edges = os.environ.get("USIM_FB_PREREQ_WEIGHTED_EDGES", "0") == "1"
        self.feedback_prereq_soft_penalty = os.environ.get("USIM_FB_PREREQ_SOFT_PENALTY", "0") == "1"
        self.feedback_course_concept_weight = float(os.environ.get("USIM_FB_COURSE_CONCEPT_W", "0.04"))
        self.feedback_course_difficulty_weight = float(os.environ.get("USIM_FB_COURSE_DIFF_W", "0.03"))
        self.feedback_course_redundant_weight = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_W", "0.02"))
        self.feedback_course_sample_beta = float(os.environ.get("USIM_FB_COURSE_SAMPLE_BETA", "0.20"))
        self.feedback_course_sample_only_cold = os.environ.get("USIM_FB_COURSE_SAMPLE_ONLY_COLD", "1") == "1"
        self.feedback_course_sample_topk = int(os.environ.get("USIM_FB_COURSE_SAMPLE_TOPK", "32"))
        self.feedback_course_sample_top_l = int(
            os.environ.get("USIM_FB_COURSE_SAMPLE_TOPL", str(self.feedback_course_sample_topk))
        )
        self.use_prereq_aux_loss = os.environ.get(
            "USIM_USE_PREREQ_AUX_LOSS",
            "1" if self.use_prereq_aux_loss else "0",
        ) == "1"
        self.prereq_aux_weight = float(os.environ.get("USIM_PREREQ_AUX_WEIGHT", str(self.prereq_aux_weight)))
        self.prereq_aux_margin = float(os.environ.get("USIM_PREREQ_AUX_MARGIN", str(self.prereq_aux_margin)))
        self.prereq_aux_violation_thr = float(
            os.environ.get("USIM_PREREQ_AUX_VIOLATION_THR", str(self.prereq_aux_violation_thr))
        )
        self.prereq_aux_min_seen = int(os.environ.get("USIM_PREREQ_AUX_MIN_SEEN", str(self.prereq_aux_min_seen)))
        self.prereq_aux_only_cold = os.environ.get(
            "USIM_PREREQ_AUX_ONLY_COLD",
            "1" if self.prereq_aux_only_cold else "0",
        ) == "1"
        self.use_course_rerank = os.environ.get(
            "USIM_USE_COURSE_RERANK",
            "1" if self.use_course_rerank else "0",
        ) == "1"
        self.rerank_alpha = float(os.environ.get("USIM_COURSE_RERANK_ALPHA", str(self.rerank_alpha)))
        self.rerank_lambda = float(os.environ.get("USIM_COURSE_RERANK_LAMBDA", str(self.rerank_lambda)))
        self.rerank_min_seen = int(os.environ.get("USIM_COURSE_RERANK_MIN_SEEN", str(self.rerank_min_seen)))
        self.rerank_top_l = int(os.environ.get("USIM_COURSE_RERANK_TOPL", str(self.rerank_top_l)))
        self.rerank_penalty_cap = float(
            os.environ.get("USIM_COURSE_RERANK_PENALTY_CAP", str(self.rerank_penalty_cap))
        )
        self.rerank_only_cold = os.environ.get(
            "USIM_COURSE_RERANK_ONLY_COLD",
            "1" if self.rerank_only_cold else "0",
        ) == "1"
        self.use_structured_hard_neg = os.environ.get(
            "USIM_USE_STRUCTURED_HARD_NEG",
            "1" if self.use_structured_hard_neg else "0",
        ) == "1"
        self.train_log_interval = int(os.environ.get("USIM_FB_TRAIN_LOG_INTERVAL", "25"))
        self.train_log_first = int(os.environ.get("USIM_FB_TRAIN_LOG_FIRST", "1"))
        self.train_log_time_sec = float(os.environ.get("USIM_FB_TRAIN_LOG_TIME_SEC", "60"))
        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", str(self.ppo_epochs)))
        self.ppo_lambda = float(os.environ.get("USIM_PPO_LAMBDA", "0.95"))
        self.ppo_value_clip = float(os.environ.get("USIM_PPO_VALUE_CLIP", "0.20"))
        self.ppo_adv_norm = os.environ.get("USIM_PPO_ADV_NORM", "1") == "1"
        self.prereq_graph_source = os.environ.get("USIM_PREREQ_GRAPH_SOURCE", self.prereq_graph_source).strip().lower()
        self.prereq_concept_score_thr = float(
            os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", str(self.prereq_concept_score_thr))
        )
        self.prereq_concept_min_hits = int(
            os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", str(self.prereq_concept_min_hits))
        )
        self.prereq_concept_file = os.environ.get("USIM_PREREQ_CONCEPT_FILE", self.prereq_concept_file)
        self.prereq_hybrid_alpha = float(os.environ.get("USIM_PREREQ_HYBRID_ALPHA", "0.70"))
        self.prereq_hybrid_strong_concept_thr = float(
            os.environ.get("USIM_PREREQ_HYBRID_STRONG_CONCEPT_THR", "0.35")
        )


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _should_log_train_progress(batch_idx, num_batches, cfg, last_log_ts, now_ts):
    step = batch_idx + 1
    if step <= max(0, int(cfg.train_log_first)):
        return True
    interval = max(1, int(cfg.train_log_interval))
    if step % interval == 0:
        return True
    if step >= num_batches:
        return True
    if (now_ts - last_log_ts) >= float(cfg.train_log_time_sec):
        return True
    return False


def _feedback_ckpt_enabled():
    return os.environ.get("USIM_FB_SAVE_CKPT", "1") == "1"


def _feedback_ckpt_auto_resume():
    return os.environ.get("USIM_FB_AUTO_RESUME", "1") == "1"


def _feedback_ckpt_force_fresh():
    return os.environ.get("USIM_FB_FORCE_FRESH", "0") == "1"


def _feedback_ckpt_save_optimizer_state():
    return os.environ.get("USIM_FB_SAVE_OPT_STATE", "0") == "1"


def _serialize_user_seen_items(user_seen_items):
    return {
        int(uid): sorted(int(it) for it in items)
        for uid, items in user_seen_items.items()
    }


def _deserialize_user_seen_items(payload):
    if not payload:
        return {}
    return {
        int(uid): set(int(it) for it in items)
        for uid, items in payload.items()
    }


def _latest_feedback_ckpt_path(ckpt_dir):
    return os.path.join(ckpt_dir, "latest.pt")


def _save_feedback_checkpoint(ckpt_dir, state, snapshot_name=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    tmp_path = latest_path + ".tmp"
    state = copy.deepcopy(state)
    state["saved_at"] = time.time()
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    if snapshot_name:
        snapshot_path = os.path.join(ckpt_dir, snapshot_name)
        torch.save(state, snapshot_path)
    return latest_path


def _load_feedback_checkpoint(ckpt_dir):
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    if not os.path.exists(latest_path):
        return None
    return torch.load(latest_path, map_location="cpu")


def _move_state_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _maybe_clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_feedback_ckpt_state(
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
    accumulated_periods,
    warmup_periods,
    total_periods,
    status,
    next_period,
    current_period=None,
    next_epoch=0,
    es_best=None,
    es_best_state=None,
    es_best_opt_state=None,
    es_no_improve=0,
):
    save_opt_state = _feedback_ckpt_save_optimizer_state()
    return {
        "version": 1,
        "status": status,
        "next_period": int(next_period),
        "current_period": None if current_period is None else int(current_period),
        "next_epoch": int(next_epoch),
        "accumulated_periods": int(accumulated_periods),
        "warmup_periods": int(warmup_periods),
        "total_periods": int(total_periods),
        "history": copy.deepcopy(history),
        "accum_cold": copy.deepcopy(accum_cold),
        "accum_hot": copy.deepcopy(accum_hot),
        "count_cold": int(count_cold),
        "count_hot": int(count_hot),
        "full_cold": copy.deepcopy(full_cold),
        "full_hot": copy.deepcopy(full_hot),
        "fc_cold": int(fc_cold),
        "fc_hot": int(fc_hot),
        "user_seen_items": _serialize_user_seen_items(user_seen_items),
        "model_state": _move_state_to_cpu(model.state_dict()),
        "optimizer_state": _move_state_to_cpu(optimizer.state_dict()) if save_opt_state else None,
        "es_best": copy.deepcopy(es_best),
        "es_best_state": _move_state_to_cpu(es_best_state),
        "es_best_opt_state": _move_state_to_cpu(es_best_opt_state) if save_opt_state else None,
        "es_no_improve": int(es_no_improve),
    }


def _feedback_output_dir():
    explicit = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    tag = os.environ.get("USIM_FB_OUTPUT_TAG", "").strip()
    if tag:
        path = os.path.join("outputs", "usim_feedback_fast3_standalone", tag)
        os.makedirs(path, exist_ok=True)
        return path
    return "."


def _feedback_output_path(filename):
    return os.path.join(_feedback_output_dir(), filename)


def _save_final_report_exports(
    protocol,
    metrics_keys,
    sampled_cold,
    sampled_hot,
    full_cold,
    full_hot,
    sampled_cold_count,
    sampled_hot_count,
    full_cold_count,
    full_hot_count,
    model_name="USIM-Feedback-FAST3-Standalone",
):
    suffix = "" if protocol == "stream" else f"_{protocol}"
    detail_path = _feedback_output_path(f"final_report_usim_feedback_fast3_standalone{suffix}.csv")
    fullrank_path = _feedback_output_path(f"final_fullrank_usim_feedback_fast3_standalone{suffix}.csv")

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
        "model": model_name,
        "protocol": protocol,
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
        "notes": f"auto-exported from {model_name} ({protocol})",
    }
    pd.DataFrame([fullrank_row]).to_csv(fullrank_path, index=False)
    return detail_path, fullrank_path


def _empty_course_stats(n_items):
    return {
        "items_with_concept": 0,
        "items_with_prereq": 0,
        "items_with_video": 0,
        "redundant_family_groups": 0,
        "hard_density": 0.0,
        "prereq_edges_kept": 0,
        "prereq_edges_raw": 0,
        "prereq_users": 0,
        "n_items": int(n_items),
    }


class FixedSimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=5):
        super().__init__()
        self.time_dim = time_dim
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
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
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

    def _build_seen_mat(self, user_ids, user_seen_items):
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.detach().cpu().tolist()
        else:
            user_ids = [int(x) for x in user_ids]
        batch_size = len(user_ids)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if user_seen_items is None:
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)
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

    def _compute_prereq_gap_and_safe(self, seen_mat, item_idx):
        batch_size = int(item_idx.size(0))
        zero = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        prereq_gap = zero.clone()
        prereq_safe = torch.ones_like(zero)
        if self.item_prereq_item_mat is None or self.item_prereq_item_cnt is None:
            return prereq_gap, prereq_safe

        batch_idx = torch.arange(batch_size, device=self.device)
        prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
        prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
        violation_full = torch.where(
            prereq_cnt > 0,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1e-6),
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)
        prereq_gap_raw = violation_full[batch_idx, item_idx].unsqueeze(1)
        gate = float(min(1.0, max(0.0, self.cfg.feedback_course_prereq_gate)))

        if getattr(self.cfg, "feedback_prereq_soft_penalty", False):
            scale = max(1e-6, 1.0 - gate)
            prereq_gap = F.relu(prereq_gap_raw - gate) / scale
            prereq_safe = (1.0 - prereq_gap).clamp(0.0, 1.0)
        else:
            prereq_gap = prereq_gap_raw
            prereq_safe = (prereq_gap_raw <= gate).float()

        return prereq_gap, prereq_safe

    def _compute_course_reward_terms(self, selected_user_ids, item_idx, target_pop=None, user_seen_items=None, cached_seen=None):
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

        if cached_seen is not None:
            seen_mat, seen_cnt_raw = cached_seen
            uid_list = selected_user_ids.detach().cpu().tolist()
            unique_uids_list = list(set(int(u) for u in uid_list))
            if len(unique_uids_list) < batch_size * 0.5:
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

        prereq_gap, prereq_safe = self._compute_prereq_gap_and_safe(seen_mat, item_idx)
        terms["prereq_gap"] = prereq_gap * active

        redundant_mode = str(getattr(self.cfg, "feedback_course_redundant_mode", "concept")).strip().lower()
        if redundant_mode == "video_family":
            structural_full = self._compute_structural_redundancy_profile(selected_user_ids, user_seen_items)
            terms["redundant"] = structural_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0) * seen_active * active

        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            if redundant_mode == "video_family":
                redundant = terms["redundant"]
            else:
                redundant = ((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)).clamp(0.0, 1.0)
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant)
            terms["concept_bonus"] = concept_bonus * active
            if redundant_mode != "video_family":
                terms["redundant"] = redundant * seen_active * active

        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[item_idx].unsqueeze(1)
            difficulty_gap = F.relu(item_difficulty - user_readiness)
            terms["difficulty_gap"] = difficulty_gap * active

        return terms

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
        redundant_mode = str(getattr(self.cfg, "feedback_course_redundant_mode", "concept")).strip().lower()
        if redundant_mode == "video_family":
            structural_full = self._compute_structural_redundancy_profile(flat_user_idx, user_seen_items)
            redundant = structural_full[batch_idx, flat_item_idx].unsqueeze(1).clamp(0.0, 1.0)
        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, flat_item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            if redundant_mode != "video_family":
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

    def _build_user_bank_raw(self):
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

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold
        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)
        target_emb = z_i_base.detach().clone()
        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
            item_idx=i,
            target_pop=pop,
            user_seen_items=user_seen_items,
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


class Fast3Config(FeedbackConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)

        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", "2"))
        self.stream_train_window = int(os.environ.get("USIM_TRAIN_WINDOW", "24"))

        self.ppo_lambda = float(os.environ.get("USIM_PPO_LAMBDA", "0.95"))
        self.ppo_value_clip = float(os.environ.get("USIM_PPO_VALUE_CLIP", "0.20"))
        self.ppo_adv_norm = os.environ.get("USIM_PPO_ADV_NORM", "1") == "1"

        self.fast3_target_alpha_cold = float(os.environ.get("USIM_FAST3_TGT_ALPHA_COLD", "0.35"))
        self.fast3_target_alpha_hot = float(os.environ.get("USIM_FAST3_TGT_ALPHA_HOT", "0.60"))
        self.fast3_target_alpha_step = float(os.environ.get("USIM_FAST3_TGT_ALPHA_STEP", "0.20"))
        self.fast3_target_alpha_entropy = float(os.environ.get("USIM_FAST3_TGT_ALPHA_ENT", "0.20"))
        self.fast3_target_alpha_min = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MIN", "0.15"))
        self.fast3_target_alpha_max = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MAX", "0.85"))

        self.feedback_course_sample_soft = os.environ.get("USIM_FB_COURSE_SAMPLE_SOFT", "1") == "1"
        self.feedback_course_sample_top_l = int(
            os.environ.get(
                "USIM_FB_COURSE_SAMPLE_TOPL",
                str(getattr(self, "feedback_course_sample_topk", 32)),
            )
        )


def _feedback_ckpt_dir():
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "usim_feedback_fast3_standalone"))


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
        results[f"R@{k}"] = recall
        results[f"N@{k}"] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results


def split_dataframe_by_periods(df, period_type="M"):
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["datetime"] = df["timestamp"]
    df["period_id"] = df["datetime"].dt.to_period(period_type)
    periods = []
    for p_key in sorted(df["period_id"].unique()):
        periods.append(df[df["period_id"] == p_key].reset_index(drop=True))
    return periods


def _read_relation_pairs(filepath):
    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                a, b = line.split("\t", 1)
            elif "," in line:
                a, b = line.split(",", 1)
                if a == "start_id" and b == "end_id":
                    continue
            else:
                continue
            pairs.append((a.strip(), b.strip()))
    return pairs


def _parse_subject_from_course_id(course_id):
    cid = str(course_id)
    m = re.search(r"\+([A-Za-z]+)\d+", cid)
    if m:
        return m.group(1).upper()
    m = re.search(r"course-v1:([^+]+)\+", cid)
    if m:
        return m.group(1).upper()
    return "UNK"


def _iter_entity_objects(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            return data if isinstance(data, list) else []
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def _extract_course_unit_ids(course_obj):
    unit_ids = []
    for unit_id in course_obj.get("video_order", []) or []:
        unit_id = str(unit_id).strip()
        if unit_id:
            unit_ids.append(unit_id)
    if unit_ids:
        return unit_ids

    for resource in course_obj.get("resource", []) or []:
        unit_id = str(resource.get("resource_id") or "").strip()
        if unit_id:
            unit_ids.append(unit_id)
    return unit_ids


def _normalize_course_family_key(course_id, core_id=None):
    for raw in [course_id, core_id]:
        if raw is None:
            continue
        cid = str(raw).strip()
        if not cid:
            continue
        if "+" in cid:
            prefix, suffix = cid.rsplit("+", 1)
            if re.fullmatch(r"(?i)(sp|20\d{2}_t\d+(?:_[a-z]+)?|_?20\d{2}_?|20\d{2})", suffix):
                return prefix
        return cid
    return None


def _build_behavior_prereq_candidates(df, prereq_min_support=30, prereq_max_forward=20):
    edge_support = defaultdict(int)
    user_seq_count = 0
    if {"u_idx", "i_idx", "timestamp"}.issubset(df.columns):
        seq_df = df[["u_idx", "i_idx", "timestamp"]].sort_values(["u_idx", "timestamp"])
        max_forward = max(1, int(prereq_max_forward))
        for _, group in seq_df.groupby("u_idx", sort=False):
            seq_raw = [int(x) for x in group["i_idx"].tolist()]
            if len(seq_raw) < 2:
                continue
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
            incoming[int(b)].append((int(a), float(sup), int(sup)))
    stats = {
        "prereq_source": "behavior",
        "prereq_edges_raw": int(len(edge_support)),
        "prereq_users": int(user_seq_count),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_forward": int(prereq_max_forward),
    }
    return incoming, stats


def _build_concept_prereq_candidates(
    concept_sets,
    relation_dir="MOOCCube/relations",
    prereq_concept_file="prerequisite-dependency.json",
    prereq_concept_score_thr=0.10,
    prereq_concept_min_hits=1,
):
    prereq_path = os.path.join(relation_dir, prereq_concept_file)
    prereq_pairs = _read_relation_pairs(prereq_path)
    incoming_concepts = defaultdict(set)
    for prereq_concept, target_concept in prereq_pairs:
        if prereq_concept and target_concept and prereq_concept != target_concept:
            incoming_concepts[target_concept].add(prereq_concept)
    concept_required_sets = []
    for cset in concept_sets:
        required = set()
        for concept in cset:
            required.update(incoming_concepts.get(concept, ()))
        required.difference_update(cset)
        concept_required_sets.append(required)
    incoming = defaultdict(list)
    raw_edge_count = 0
    courses_with_required_concepts = 0
    n_items = len(concept_sets)
    min_hits = max(1, int(prereq_concept_min_hits))
    score_thr = max(0.0, float(prereq_concept_score_thr))
    for b in range(n_items):
        required = concept_required_sets[b]
        if not required:
            continue
        courses_with_required_concepts += 1
        denom = float(len(required))
        for a in range(n_items):
            if a == b or not concept_sets[a]:
                continue
            hits = len(concept_sets[a] & required)
            if hits < min_hits:
                continue
            score = hits / denom
            if score >= score_thr:
                incoming[b].append((a, float(score), int(hits)))
                raw_edge_count += 1
    stats = {
        "prereq_source": "concept",
        "prereq_edges_raw": int(raw_edge_count),
        "prereq_users": 0,
        "prereq_min_support": 0,
        "prereq_max_forward": 0,
        "prereq_concept_pairs": int(len(prereq_pairs)),
        "prereq_concept_score_thr": float(score_thr),
        "prereq_concept_min_hits": int(min_hits),
        "courses_with_required_concepts": int(courses_with_required_concepts),
        "prereq_concept_file": str(prereq_concept_file),
    }
    return incoming, stats


def _build_hybrid_prereq_candidates(
    df,
    concept_sets,
    relation_dir="MOOCCube/relations",
    prereq_min_support=30,
    prereq_max_forward=20,
    prereq_concept_file="prerequisite-dependency.json",
    prereq_concept_score_thr=0.20,
    prereq_concept_min_hits=2,
    hybrid_alpha=0.70,
    hybrid_strong_concept_thr=0.35,
):
    behavior_incoming, behavior_stats = _build_behavior_prereq_candidates(
        df,
        prereq_min_support=prereq_min_support,
        prereq_max_forward=prereq_max_forward,
    )
    concept_incoming, concept_stats = _build_concept_prereq_candidates(
        concept_sets,
        relation_dir=relation_dir,
        prereq_concept_file=prereq_concept_file,
        prereq_concept_score_thr=prereq_concept_score_thr,
        prereq_concept_min_hits=prereq_concept_min_hits,
    )

    alpha = min(1.0, max(0.0, float(hybrid_alpha)))
    strong_thr = max(float(prereq_concept_score_thr), float(hybrid_strong_concept_thr))
    incoming = defaultdict(list)
    raw_edge_count = 0

    for b, concept_edges in concept_incoming.items():
        behavior_edges = {
            int(src): float(score)
            for src, score, _ in behavior_incoming.get(b, [])
        }
        max_behavior = max(behavior_edges.values()) if behavior_edges else 0.0
        for src, concept_score, concept_hits in concept_edges:
            src = int(src)
            concept_score = float(concept_score)
            behavior_score = behavior_edges.get(src, 0.0)
            behavior_norm = behavior_score / max_behavior if max_behavior > 0.0 else 0.0
            if behavior_score <= 0.0 and concept_score < strong_thr:
                continue
            hybrid_score = alpha * concept_score + (1.0 - alpha) * behavior_norm
            incoming[int(b)].append((src, float(hybrid_score), int(concept_hits)))
            raw_edge_count += 1

    stats = {
        "prereq_source": "hybrid",
        "prereq_edges_raw": int(raw_edge_count),
        "prereq_users": int(behavior_stats.get("prereq_users", 0)),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_forward": int(prereq_max_forward),
        "prereq_concept_pairs": int(concept_stats.get("prereq_concept_pairs", 0)),
        "prereq_concept_score_thr": float(prereq_concept_score_thr),
        "prereq_concept_min_hits": int(prereq_concept_min_hits),
        "courses_with_required_concepts": int(concept_stats.get("courses_with_required_concepts", 0)),
        "prereq_concept_file": str(prereq_concept_file),
        "prereq_hybrid_alpha": float(alpha),
        "prereq_hybrid_strong_concept_thr": float(strong_thr),
        "behavior_prereq_edges_raw": int(behavior_stats.get("prereq_edges_raw", 0)),
        "concept_prereq_edges_raw": int(concept_stats.get("prereq_edges_raw", 0)),
    }
    return incoming, stats


def _build_item_concept_overlap(concept_sets, mode="plain"):
    mode = (mode or "plain").strip().lower()
    n_items = len(concept_sets)
    item_concept_overlap = torch.zeros((n_items, n_items), dtype=torch.float32)

    if mode == "plain":
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
        return item_concept_overlap

    if mode == "idf":
        concept_df = defaultdict(int)
        for cset in concept_sets:
            for concept in cset:
                concept_df[concept] += 1
        n_courses = max(1, n_items)
        concept_idf = {
            concept: math.log((n_courses + 1.0) / (df + 1.0)) + 1.0
            for concept, df in concept_df.items()
        }
        for i in range(n_items):
            c_i = concept_sets[i]
            if not c_i:
                continue
            denom = sum(concept_idf.get(c, 1.0) for c in c_i)
            if denom <= 0.0:
                continue
            for j in range(n_items):
                c_j = concept_sets[j]
                if not c_j:
                    continue
                inter = c_i & c_j
                if inter:
                    numer = sum(concept_idf.get(c, 1.0) for c in inter)
                    item_concept_overlap[i, j] = numer / denom
        return item_concept_overlap

    raise ValueError(f"Unsupported concept overlap mode: {mode}")


def build_course_artifacts(
    df,
    n_items,
    relation_dir="MOOCCube/relations",
    prereq_min_support=30,
    prereq_max_per_item=5,
    prereq_min_items=1,
    prereq_max_forward=20,
    concept_overlap_mode=None,
    prereq_graph_source=None,
    prereq_concept_score_thr=None,
    prereq_concept_min_hits=None,
    prereq_concept_file=None,
    prereq_hybrid_alpha=None,
    prereq_hybrid_strong_concept_thr=None,
):
    weighted_prereq_edges = os.environ.get("USIM_FB_PREREQ_WEIGHTED_EDGES", "0") == "1"
    idx_course = df[["i_idx", "course_id"]].drop_duplicates(subset=["i_idx"])
    idx_to_course = [None] * n_items
    for row in idx_course.itertuples(index=False):
        i_idx = int(row.i_idx)
        if 0 <= i_idx < n_items:
            idx_to_course[i_idx] = str(row.course_id)
    course_to_idx = {cid: idx for idx, cid in enumerate(idx_to_course) if cid is not None}
    concept_sets = [set() for _ in range(n_items)]
    video_sets = [set() for _ in range(n_items)]
    family_keys = [None] * n_items
    course_concept_file = os.path.join(relation_dir, "course-concept.json")
    for cid, concept in _read_relation_pairs(course_concept_file):
        idx = course_to_idx.get(cid)
        if idx is not None and concept:
            concept_sets[idx].add(concept)
    entity_dir = os.path.join(os.path.dirname(relation_dir), "entities")
    course_entity_file = os.path.join(entity_dir, "course.json")
    for course_obj in _iter_entity_objects(course_entity_file):
        cid = str(course_obj.get("id") or "").strip()
        idx = course_to_idx.get(cid)
        if idx is None:
            continue
        family_keys[idx] = _normalize_course_family_key(cid, course_obj.get("core_id"))
        video_sets[idx] = set(_extract_course_unit_ids(course_obj))
    item_prereq_item_mat = torch.zeros((n_items, n_items), dtype=torch.float32)
    item_prereq_item_cnt = torch.zeros(n_items, dtype=torch.float32)
    concept_overlap_mode = (concept_overlap_mode or os.environ.get("USIM_CONCEPT_OVERLAP_MODE", "plain")).strip().lower()
    prereq_graph_source = (prereq_graph_source or os.environ.get("USIM_PREREQ_GRAPH_SOURCE", "behavior")).strip().lower()
    prereq_concept_score_thr = float(
        prereq_concept_score_thr if prereq_concept_score_thr is not None
        else os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", "0.10")
    )
    prereq_concept_min_hits = int(
        prereq_concept_min_hits if prereq_concept_min_hits is not None
        else os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", "1")
    )
    prereq_concept_file = prereq_concept_file or os.environ.get("USIM_PREREQ_CONCEPT_FILE", "prerequisite-dependency.json")
    prereq_hybrid_alpha = float(
        prereq_hybrid_alpha if prereq_hybrid_alpha is not None
        else os.environ.get("USIM_PREREQ_HYBRID_ALPHA", "0.70")
    )
    prereq_hybrid_strong_concept_thr = float(
        prereq_hybrid_strong_concept_thr if prereq_hybrid_strong_concept_thr is not None
        else os.environ.get("USIM_PREREQ_HYBRID_STRONG_CONCEPT_THR", "0.35")
    )
    if prereq_graph_source == "behavior":
        incoming, prereq_stats = _build_behavior_prereq_candidates(
            df,
            prereq_min_support=prereq_min_support,
            prereq_max_forward=prereq_max_forward,
        )
    elif prereq_graph_source == "concept":
        incoming, prereq_stats = _build_concept_prereq_candidates(
            concept_sets,
            relation_dir=relation_dir,
            prereq_concept_file=prereq_concept_file,
            prereq_concept_score_thr=prereq_concept_score_thr,
            prereq_concept_min_hits=prereq_concept_min_hits,
        )
    elif prereq_graph_source == "hybrid":
        incoming, prereq_stats = _build_hybrid_prereq_candidates(
            df,
            concept_sets,
            relation_dir=relation_dir,
            prereq_min_support=prereq_min_support,
            prereq_max_forward=prereq_max_forward,
            prereq_concept_file=prereq_concept_file,
            prereq_concept_score_thr=prereq_concept_score_thr,
            prereq_concept_min_hits=prereq_concept_min_hits,
            hybrid_alpha=prereq_hybrid_alpha,
            hybrid_strong_concept_thr=prereq_hybrid_strong_concept_thr,
        )
    else:
        raise ValueError(f"Unsupported prereq_graph_source: {prereq_graph_source}")
    kept_edge_count = 0
    for b, src_list in incoming.items():
        src_list.sort(key=lambda x: (-float(x[1]), -int(x[2]), int(x[0])))
        kept = src_list[:max(1, int(prereq_max_per_item))]
        if len(kept) < int(prereq_min_items):
            continue
        idx_list = torch.tensor([src for src, _, _ in kept], dtype=torch.long)
        if weighted_prereq_edges:
            weight_tensor = torch.tensor([float(score) for _, score, _ in kept], dtype=torch.float32)
            max_weight = float(weight_tensor.max().item()) if weight_tensor.numel() > 0 else 0.0
            if max_weight > 0.0:
                weight_tensor = weight_tensor / max_weight
            item_prereq_item_mat[b, idx_list] = weight_tensor
            item_prereq_item_cnt[b] = float(weight_tensor.sum().item())
        else:
            item_prereq_item_mat[b, idx_list] = 1.0
            item_prereq_item_cnt[b] = float(len(kept))
        kept_edge_count += len(kept)
    item_concept_overlap = _build_item_concept_overlap(concept_sets, mode=concept_overlap_mode)
    item_video_contain = torch.zeros((n_items, n_items), dtype=torch.float32)
    item_same_family = torch.zeros((n_items, n_items), dtype=torch.bool)
    for i in range(n_items):
        v_i = video_sets[i]
        for j in range(n_items):
            if v_i and video_sets[j]:
                inter_video = len(v_i & video_sets[j])
                if inter_video > 0:
                    item_video_contain[i, j] = inter_video / float(len(v_i))
            if family_keys[i] and family_keys[i] == family_keys[j]:
                item_same_family[i, j] = True
    subjects = [_parse_subject_from_course_id(cid) if cid is not None else "UNK" for cid in idx_to_course]
    item_hard_adj = torch.zeros((n_items, n_items), dtype=torch.bool)
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            same_subject = subjects[i] != "UNK" and subjects[i] == subjects[j]
            same_concept = item_concept_overlap[i, j] > 0
            if same_subject or same_concept:
                item_hard_adj[i, j] = True
    items_with_concept = int(sum(1 for c in concept_sets if len(c) > 0))
    items_with_prereq = int((item_prereq_item_cnt > 0).sum().item())
    items_with_video = int(sum(1 for vids in video_sets if len(vids) > 0))
    family_group_sizes = defaultdict(int)
    for key in family_keys:
        if key:
            family_group_sizes[key] += 1
    hard_density = float(item_hard_adj.float().mean().item())
    stats = {
        "prereq_source": prereq_graph_source,
        "items_with_concept": items_with_concept,
        "items_with_prereq": items_with_prereq,
        "items_with_video": items_with_video,
        "redundant_family_groups": int(sum(1 for v in family_group_sizes.values() if v > 1)),
        "hard_density": hard_density,
        "prereq_edges_kept": int(kept_edge_count),
        "prereq_edges_raw": int(prereq_stats.get("prereq_edges_raw", 0)),
        "prereq_users": int(prereq_stats.get("prereq_users", 0)),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_per_item": int(prereq_max_per_item),
        "prereq_max_forward": int(prereq_max_forward),
        "concept_overlap_mode": concept_overlap_mode,
        "prereq_concept_pairs": int(prereq_stats.get("prereq_concept_pairs", 0)),
        "prereq_concept_score_thr": float(prereq_stats.get("prereq_concept_score_thr", prereq_concept_score_thr)),
        "prereq_concept_min_hits": int(prereq_stats.get("prereq_concept_min_hits", prereq_concept_min_hits)),
        "courses_with_required_concepts": int(prereq_stats.get("courses_with_required_concepts", 0)),
        "prereq_concept_file": str(prereq_stats.get("prereq_concept_file", prereq_concept_file)),
        "prereq_hybrid_alpha": float(prereq_stats.get("prereq_hybrid_alpha", prereq_hybrid_alpha)),
        "prereq_hybrid_strong_concept_thr": float(
            prereq_stats.get("prereq_hybrid_strong_concept_thr", prereq_hybrid_strong_concept_thr)
        ),
        "behavior_prereq_edges_raw": int(prereq_stats.get("behavior_prereq_edges_raw", 0)),
        "concept_prereq_edges_raw": int(prereq_stats.get("concept_prereq_edges_raw", 0)),
        "prereq_weighted_edges": weighted_prereq_edges,
    }
    artifacts = {
        "item_hard_adj": item_hard_adj,
        "item_prereq_item_mat": item_prereq_item_mat,
        "item_prereq_item_cnt": item_prereq_item_cnt,
        "item_concept_overlap": item_concept_overlap,
        "item_video_contain": item_video_contain,
        "item_same_family": item_same_family,
    }
    return artifacts, stats


def _lookup_llm_score(llm_scores, item_idx, user_idx=None, allow_pair=True, allow_item=True):
    if llm_scores is None:
        return -1.0
    item_idx = int(item_idx)
    if allow_pair and user_idx is not None:
        pair_score = llm_scores.get((int(user_idx), item_idx))
        if pair_score is not None:
            return float(pair_score)
    if allow_item:
        item_score = llm_scores.get(item_idx)
        if item_score is not None:
            return float(item_score)
    return -1.0


def _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=None):
    values = [_lookup_llm_score(llm_scores, item_idx, user_idx) for user_idx, item_idx in zip(user_ids, item_ids)]
    return torch.tensor(values, dtype=torch.float, device=device)


def build_all_item_vecs(model, device, llm_scores, item_batch=1024, force_cold=True):
    n_items = model.cfg.n_items
    all_item_idx = torch.arange(n_items, device=device)
    bank_mode = getattr(model.cfg, "llm_bank_mode", "none")
    if bank_mode == "item":
        all_llm_s = torch.tensor(
            [_lookup_llm_score(llm_scores, int(idx), allow_pair=False, allow_item=True) for idx in all_item_idx],
            dtype=torch.float,
            device=device,
        )
    else:
        all_llm_s = torch.full((n_items,), -1.0, dtype=torch.float, device=device)
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
    hot_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=False)
    cold_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=True)
    return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}


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
    if eval_type == "cold":
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=True)
        return F.normalize(pos_vec, dim=1)
    if eval_type == "hot":
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=False)
        return F.normalize(pos_vec, dim=1)
    pos_vec = torch.empty((item_idx.size(0), model.cfg.emb_dim), device=item_idx.device)
    cold_mask = pop_sel < model.cfg.cold_threshold
    hot_mask = ~cold_mask
    if cold_mask.any():
        cold_vec, _, _ = model.get_item_vector(item_idx[cold_mask], llm_s[cold_mask], force_cold=True)
        pos_vec[cold_mask] = cold_vec
    if hot_mask.any():
        hot_vec, _, _ = model.get_item_vector(item_idx[hot_mask], llm_s[hot_mask], force_cold=False)
        pos_vec[hot_mask] = hot_vec
    return F.normalize(pos_vec, dim=1)


def evaluate_usim(model, loader, device, llm_scores, k_list=[5, 10, 20], n_neg=200,
                  eval_type="cold", full_ranking=False, user_seen_items=None, all_item_vecs=None):
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
                    seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=device) if seen_list else None
                else:
                    seen_tensor_cache[uid] = None
            z_u = F.normalize(model.user_proj(model.user_emb(u)), dim=1)
            pos_llm = _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=device)
            pos_vec = _build_eval_pos_item_vecs(model, i, pos_llm, pop_sel, eval_type)
            pos_scores = (z_u * pos_vec).sum(dim=1)
            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                row_idx = torch.arange(n_sel, device=device)
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
                scores = model.apply_course_rerank(scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=pop_sel)
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
                cand_vecs = item_bank[cand_idx].clone()
                cand_vecs[:, 0, :] = pos_vec
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                scores = model.apply_course_rerank(scores, user_ids, seen_tensor_cache, cand_idx=cand_idx, target_pop=pop_sel)
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)
            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)
            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel
    if total_samples == 0:
        return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


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


def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for Feedback-Aware USIM (FAST3 Standalone) from {data_dir}...")
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
    device = _resolve_torch_device()
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

    print(f">> Architecture: Feedback-Aware RL-USIM + InfoNCE [FAST3 Standalone] (Batch Size={cfg.batch_size})")
    print(f">> Device: {device}")
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
        f">> Course Feedback: redundant_mode={cfg.feedback_course_redundant_mode} | "
        f"video_min={cfg.feedback_course_struct_video_min:.2f} | "
        f"concept_min={cfg.feedback_course_concept_min:.2f} | "
        f"redundant_thr={cfg.feedback_course_redundant_thr:.2f}"
    )
    print(
        f">> Course Artifacts: enabled={cfg.feedback_load_course_artifacts} | "
        f"prereq_aux={cfg.use_prereq_aux_loss} | "
        f"rerank={cfg.use_course_rerank} | "
        f"struct_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> Prereq Penalty: weighted_edges={cfg.feedback_prereq_weighted_edges} | "
        f"soft_penalty={cfg.feedback_prereq_soft_penalty} | "
        f"gate={cfg.feedback_course_prereq_gate:.2f} | "
        f"w={cfg.feedback_course_prereq_weight:.3f}"
    )
    print(
        f">> Course Priors: source={course_stats.get('prereq_source', 'behavior')} | "
        f"concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"video={course_stats.get('items_with_video', 0)}/{cfg.n_items}, "
        f"family_groups={course_stats.get('redundant_family_groups', 0)} | "
        f"concept_overlap={course_stats.get('concept_overlap_mode', 'plain')} | "
        f"hard_density={course_stats['hard_density']:.3f} | "
        f"prereq_raw={course_stats.get('prereq_edges_raw', 0)} | "
        f"prereq_kept={course_stats.get('prereq_edges_kept', 0)} | "
        f"prereq_weighted={course_stats.get('prereq_weighted_edges', False)}"
    )
    if course_stats.get("prereq_source") == "hybrid":
        print(
            f">> Hybrid Prereq: alpha={course_stats.get('prereq_hybrid_alpha', 0.0):.2f} | "
            f"strong_thr={course_stats.get('prereq_hybrid_strong_concept_thr', 0.0):.2f} | "
            f"behavior_raw={course_stats.get('behavior_prereq_edges_raw', 0)} | "
            f"concept_raw={course_stats.get('concept_prereq_edges_raw', 0)}"
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
        f"force_fresh={force_fresh} | save_opt={_feedback_ckpt_save_optimizer_state()} | dir={ckpt_dir}"
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
                if resume_state.get("optimizer_state") is not None:
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
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (RL-USIM FAST3 Standalone)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    summary_rows = []
    sampled_row = {"Model": "USIM-Feedback-FAST3-Standalone", "Eval": "sampled", "ColdSamples": count_cold, "HotSamples": count_hot}
    full_row = {"Model": "USIM-Feedback-FAST3-Standalone", "Eval": "full_rank", "ColdSamples": fc_cold, "HotSamples": fc_hot}
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
        model_name="USIM-Feedback-FAST3-Standalone",
    )

    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_standalone.csv")
    summary_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_standalone_summary.csv")
    plot_path = _feedback_output_path("mooc_result_usim_feedback_fast3_standalone.png")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("RL-USIM [FAST3 Standalone]: adaptive mix + stable PPO + soft rerank")
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
