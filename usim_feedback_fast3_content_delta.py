"""
usim_feedback_fast3_content_delta.py - standalone FAST3 variant with bounded content delta

This file no longer depends on usim.py, usim_feedback.py, or
usim_feedback_fast.py. The required model, data, evaluation, course-graph,
checkpoint, and reporting utilities are all inlined here.
"""
import copy
import hashlib
import json
import math
import os
import platform
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
        self.use_content_delta = os.environ.get("USIM_USE_CONTENT_DELTA", "1") == "1"
        self.content_delta_paper_style = os.environ.get("USIM_CONTENT_DELTA_PAPER_STYLE", "0") == "1"
        self.content_delta_replace_item = os.environ.get(
            "USIM_CONTENT_DELTA_REPLACE_ITEM",
            "1" if self.content_delta_paper_style else "0",
        ) == "1"
        self.content_delta_max_norm = float(
            os.environ.get("USIM_CONTENT_DELTA_MAX_NORM", os.environ.get("USIM_CONTENT_DELTA_MAX", "0.5"))
        )
        self.content_delta_cold_only = os.environ.get("USIM_CONTENT_DELTA_COLD_ONLY", "0") == "1"
        self.content_delta_normalize_base = os.environ.get("USIM_CONTENT_DELTA_NORMALIZE_BASE", "1") == "1"
        self.content_delta_normalize_output = os.environ.get("USIM_CONTENT_DELTA_NORMALIZE_OUTPUT", "1") == "1"
        self.content_delta_mode = os.environ.get("USIM_CONTENT_DELTA_MODE", "embedding").strip().lower()
        if self.content_delta_mode in {"mlp", "content", "content_projector"}:
            self.content_delta_mode = "projector"
        if self.content_delta_mode not in {"embedding", "projector", "hybrid"}:
            raise ValueError(
                "USIM_CONTENT_DELTA_MODE must be one of: embedding, projector, hybrid"
            )
        self.content_delta_projector_hidden = int(
            os.environ.get("USIM_CONTENT_DELTA_PROJECTOR_HIDDEN", str(self.hidden_dim))
        )
        self.content_delta_train_on_id_dropout = (
            os.environ.get("USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT", "1") == "1"
        )
        self.content_delta_only_after_epoch = int(
            os.environ.get("USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH", "0")
        )
        self.content_delta_scale = float(os.environ.get("USIM_CONTENT_DELTA_SCALE", "0.25"))
        self.content_delta_aux_mode = os.environ.get("USIM_CONTENT_DELTA_AUX_MODE", "base").strip().lower()
        self.content_delta_l2_weight = float(os.environ.get("USIM_CONTENT_DELTA_L2_W", "0.02"))
        self.content_delta_cap_weight = float(os.environ.get("USIM_CONTENT_DELTA_CAP_W", "0.02"))
        self.content_delta_cap_margin = float(os.environ.get("USIM_CONTENT_DELTA_CAP_MARGIN", "0.70"))
        self.content_delta_lr_mult = float(os.environ.get("USIM_CONTENT_DELTA_LR_MULT", "0.10"))
        self.content_delta_eval_bank_mode = os.environ.get(
            "USIM_CONTENT_DELTA_EVAL_BANK_MODE",
            "auto",
        ).strip().lower()
        self.cold_threshold = int(os.environ.get("USIM_COLD_THRESHOLD", "5"))
        self.lr = 0.0005
        self.temp = 0.07
        self.margin = 0.15
        self.dropout_prob = 0.35
        self.aux_weight = float(os.environ.get("USIM_AUX_WEIGHT", "0.3"))
        # ROLLBACK FLAG (USIM_AUX_HOT_ONLY): when "1", restrict the id<->content
        # auxiliary InfoNCE to hot rows only. Cold rows have under-trained
        # `id_e_true` and inject noise into content_e gradient. Default "0"
        # preserves legacy behavior; set "1" together with the cold-start patch
        # rollout. See _compute_aux_loss for the actual branching.
        self.aux_hot_only = os.environ.get("USIM_AUX_HOT_ONLY", "0") == "1"
        self.train_force_cold = os.environ.get("USIM_TRAIN_FORCE_COLD", "1") == "1"
        self.use_pseudo_cold_train = os.environ.get("USIM_USE_PSEUDO_COLD_TRAIN", "0") == "1"
        self.pseudo_cold_ratio = float(os.environ.get("USIM_PSEUDO_COLD_RATIO", "0.30"))
        self.pseudo_cold_ratio = min(1.0, max(0.0, self.pseudo_cold_ratio))
        self.pseudo_cold_min_pop = int(os.environ.get("USIM_PSEUDO_COLD_MIN_POP", "5"))
        self.pseudo_cold_mode = os.environ.get("USIM_PSEUDO_COLD_MODE", "batch_random").strip().lower()
        if self.pseudo_cold_mode not in {"batch_random", "batch_tail", "all_eligible", "none", "off"}:
            raise ValueError(
                "USIM_PSEUDO_COLD_MODE must be one of: batch_random, batch_tail, all_eligible, none, off"
            )
        self.disable_llm_score = os.environ.get("USIM_DISABLE_LLM_SCORE", "0") == "1"
        self.llm_safe_mode = os.environ.get("USIM_LLM_SAFE_MODE", "0") == "1"
        self.llm_weight = float(
            os.environ.get("USIM_LLM_WEIGHT", "0.20" if self.llm_safe_mode else "1.0")
        )
        self.llm_cold_only = os.environ.get(
            "USIM_LLM_COLD_ONLY",
            "1" if self.llm_safe_mode else "0",
        ) == "1"
        self.llm_hot_only = os.environ.get("USIM_LLM_HOT_ONLY", "0") == "1"
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
        self.eval_n_neg = int(os.environ.get("USIM_EVAL_N_NEG", "200"))
        # Sampled (1+N_neg) eval is no longer the headline metric; final tables
        # report full ranking (item-macro). Default flipped to "0" to save eval
        # time. Set USIM_RUN_SAMPLED_EVAL=1 to restore legacy 1+200 sampled eval.
        self.run_sampled_eval = os.environ.get("USIM_RUN_SAMPLED_EVAL", "0") == "1"
        self.use_mixed_hard_neg = True
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        self.use_structured_hard_neg = False
        self.mask_known_pos_neg = os.environ.get("USIM_MASK_KNOWN_POS_NEG", "0") == "1"
        self.use_paac = os.environ.get("USIM_USE_PAAC", "1") == "1"
        self.paac_align_weight = float(os.environ.get("USIM_PAAC_ALIGN_W", "0.0"))
        self.paac_align_max_pairs = int(os.environ.get("USIM_PAAC_ALIGN_MAX_PAIRS", "512"))
        self.paac_align_detach_hot = os.environ.get("USIM_PAAC_ALIGN_DETACH_HOT", "1") == "1"
        self.paac_contrast_weight = float(os.environ.get("USIM_PAAC_CONTRAST_W", "0.02"))
        self.paac_contrast_beta = float(os.environ.get("USIM_PAAC_CONTRAST_BETA", "0.20"))
        self.paac_contrast_gamma = float(os.environ.get("USIM_PAAC_CONTRAST_GAMMA", "0.20"))
        self.paac_batch_pop_ratio = float(os.environ.get("USIM_PAAC_BATCH_POP_RATIO", "0.50"))
        self.paac_group_mode = os.environ.get("USIM_PAAC_GROUP_MODE", "batch_quantile").strip().lower()
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
        self.early_stop_patience = int(os.environ.get("USIM_EARLY_STOP_PATIENCE", "1"))
        self.early_stop_min_delta = float(os.environ.get("USIM_EARLY_STOP_MIN_DELTA", "1e-4"))
        self.early_stop_average_mode = os.environ.get("USIM_EARLY_STOP_AVG_MODE", "interaction").strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("USIM_EARLY_STOP_AVG_MODE must be 'interaction' or 'item_macro'")
        # ROLLBACK FLAG (USIM_EARLY_STOP_SCORE_MODE): how cold/hot N@k are
        # combined into the early-stop score. "cold_only" (default) is the
        # legacy behavior. "geometric" / "harmonic" / "sum" let hot pull the
        # selector back when cold gains stop translating into hot improvement.
        # See _compute_early_stop_score for the formulas.
        self.early_stop_score_mode = os.environ.get(
            "USIM_EARLY_STOP_SCORE_MODE", "cold_only"
        ).strip().lower()
        if self.early_stop_score_mode not in {"cold_only", "geometric", "harmonic", "sum"}:
            raise ValueError(
                "USIM_EARLY_STOP_SCORE_MODE must be one of: cold_only, geometric, harmonic, sum"
            )
        self.early_stop_hot_r10_drop_tol = 0.03
        self.legacy_train_protocol = os.environ.get("USIM_LEGACY_TRAIN_PROTOCOL", "0") == "1"


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
    @staticmethod
    def _build_paper_content_base(content_emb, emb_dim):
        """Frozen standardized-PCA content base used by the paper-style delta path."""
        with torch.no_grad():
            x = content_emb.detach().float().cpu()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
            x = x - x.mean(dim=0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(x, full_matrices=False)
            except RuntimeError:
                # SVD is deterministic and preferred here, but low-rank PCA is
                # a robust fallback on older BLAS/CUDA stacks.
                _, _, vh = torch.pca_lowrank(x, q=min(emb_dim, min(x.shape) - 1), center=False)
                vh = vh.t()
            rank = min(int(emb_dim), int(vh.size(0)))
            z = torch.matmul(x, vh[:rank].t())
            if rank < int(emb_dim):
                pad = torch.zeros((z.size(0), int(emb_dim) - rank), dtype=z.dtype)
                z = torch.cat([z, pad], dim=1)
            return F.normalize(z, dim=1)

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
        self.paper_item_con_emb = None
        if getattr(config, "content_delta_paper_style", False):
            paper_content = self._build_paper_content_base(content_emb, config.emb_dim)
            self.paper_item_con_emb = nn.Embedding.from_pretrained(paper_content, freeze=True)
        self.content_delta = nn.Embedding(config.n_items, config.emb_dim)
        nn.init.zeros_(self.content_delta.weight)
        delta_mode = str(getattr(config, "content_delta_mode", "embedding")).strip().lower()
        use_delta = bool(getattr(config, "use_content_delta", False))
        self.content_delta.weight.requires_grad = use_delta and delta_mode in {"embedding", "hybrid"}
        delta_hidden = int(getattr(config, "content_delta_projector_hidden", config.hidden_dim))
        self.content_delta_projector = nn.Sequential(
            nn.LayerNorm(config.emb_dim),
            nn.Linear(config.emb_dim, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, config.emb_dim),
        )
        for module in self.content_delta_projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        last_delta_layer = self.content_delta_projector[-1]
        if isinstance(last_delta_layer, nn.Linear):
            nn.init.zeros_(last_delta_layer.weight)
            nn.init.zeros_(last_delta_layer.bias)
        for param in self.content_delta_projector.parameters():
            param.requires_grad = use_delta and delta_mode in {"projector", "hybrid"}
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
        self.user_seen_index = None

    @torch.no_grad()
    def set_user_seen_index(self, user_seen_items):
        """Pre-build a dense (n_users, n_items) bool tensor on device.

        This eliminates per-batch Python loops in `_build_seen_mat`,
        `_build_known_positive_batch_mask`, `apply_course_rerank`, and the
        `evaluate_usim` full-ranking seen-mask, providing a major speedup.
        Memory: n_users * n_items bytes (e.g., 199199 * 698 ≈ 139 MB).
        Pass `None` to clear (falls back to legacy per-batch construction).
        """
        if user_seen_items is None:
            self.user_seen_index = None
            return
        n_users = self.cfg.n_users
        n_items = self.cfg.n_items
        rows = []
        cols = []
        for uid, items in user_seen_items.items():
            u = int(uid)
            if u < 0 or u >= n_users:
                continue
            for it in items:
                i = int(it)
                if 0 <= i < n_items:
                    rows.append(u)
                    cols.append(i)
        mat = torch.zeros((n_users, n_items), dtype=torch.bool, device=self.device)
        if rows:
            rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
            mat[rows_t, cols_t] = True
        self.user_seen_index = mat

    def _resolve_user_id_tensor(self, user_ids):
        if isinstance(user_ids, torch.Tensor):
            return user_ids.to(self.device).long().view(-1)
        return torch.as_tensor(
            [int(x) for x in user_ids], dtype=torch.long, device=self.device
        )

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
        seen_index = getattr(self, "user_seen_index", None)
        if seen_index is not None:
            uid_t = self._resolve_user_id_tensor(user_ids)
            seen_mat = seen_index.index_select(0, uid_t).float()
        else:
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
            row_mask = self._cold_mask_from_pop(target_pop).float().view(-1, 1)
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

    def _cold_mask_from_pop(self, pop):
        if pop is None:
            return None
        return pop.to(device=self.device).float().view(-1) < float(self.cfg.cold_threshold)

    def _effective_train_cold_mask(self, pop):
        true_cold = self._cold_mask_from_pop(pop)
        if true_cold is None:
            return None
        if (
            not self.training
            or not getattr(self.cfg, "use_pseudo_cold_train", False)
            or float(getattr(self.cfg, "pseudo_cold_ratio", 0.0)) <= 0.0
        ):
            return true_cold

        mode = str(getattr(self.cfg, "pseudo_cold_mode", "batch_random")).strip().lower()
        if mode in {"none", "off"}:
            return true_cold

        pop_f = pop.to(device=self.device).float().view(-1)
        min_pop = float(max(1, int(getattr(self.cfg, "pseudo_cold_min_pop", 1))))
        eligible = (~true_cold) & (pop_f >= min_pop)
        n_eligible = int(eligible.sum().detach().item())
        if n_eligible < 1:
            return true_cold

        ratio = float(getattr(self.cfg, "pseudo_cold_ratio", 0.0))
        target_n = n_eligible if mode == "all_eligible" else int(math.ceil(n_eligible * ratio))
        target_n = max(1, min(n_eligible, target_n))

        pseudo_cold = torch.zeros_like(true_cold)
        if mode == "batch_tail":
            scores = (-pop_f).masked_fill(~eligible, -float("inf"))
        else:
            scores = torch.rand_like(pop_f).masked_fill(~eligible, -1.0)
        _, chosen = torch.topk(scores, k=target_n, dim=0)
        pseudo_cold[chosen] = True
        return true_cold | pseudo_cold

    def _target_pop_with_effective_cold(self, pop, effective_cold):
        if pop is None or effective_cold is None:
            return pop
        return torch.where(effective_cold.to(device=pop.device).view(-1), torch.zeros_like(pop), pop)

    def _content_delta_active_mask(self, i_idx, force_cold):
        batch_size = int(i_idx.size(0))
        active = torch.ones((batch_size, 1), dtype=torch.float32, device=i_idx.device)
        if not getattr(self.cfg, "content_delta_cold_only", False):
            return active
        if isinstance(force_cold, torch.Tensor):
            cold_mask = force_cold.to(device=i_idx.device)
            if cold_mask.dtype != torch.bool:
                cold_mask = cold_mask > 0
            return cold_mask.float().view(-1, 1)
        if force_cold:
            return active
        return torch.zeros_like(active)

    def _raw_content_delta(self, base_e, i_idx):
        mode = str(getattr(self.cfg, "content_delta_mode", "embedding")).strip().lower()
        pieces = []
        if mode in {"embedding", "hybrid"}:
            pieces.append(self.content_delta(i_idx))
        if mode in {"projector", "hybrid"}:
            pieces.append(self.content_delta_projector(base_e))
        if not pieces:
            return torch.zeros_like(base_e)
        delta = pieces[0]
        for piece in pieces[1:]:
            delta = delta + piece
        return delta

    def _content_base_embedding(self, i_idx):
        if getattr(self.cfg, "content_delta_paper_style", False):
            if self.paper_item_con_emb is None:
                raise RuntimeError("Paper-style ContentDelta requested but frozen content base is missing.")
            return self.paper_item_con_emb(i_idx)
        return self.content_proj(self.item_con_emb(i_idx))

    def _all_content_delta_vectors(self, detach_base=False):
        item_idx = torch.arange(self.cfg.n_items, device=self.item_id_emb.weight.device)
        base_e = self._content_base_embedding(item_idx)
        if getattr(self.cfg, "content_delta_normalize_base", True):
            base_e = F.normalize(base_e, dim=1)
        if detach_base:
            base_e = base_e.detach()
        return self._raw_content_delta(base_e, item_idx)

    def _apply_content_delta(self, content_e, i_idx, force_cold=False):
        base_e = content_e
        if getattr(self.cfg, "content_delta_normalize_base", True):
            base_e = F.normalize(base_e, dim=1)
        if not getattr(self.cfg, "use_content_delta", False):
            if getattr(self.cfg, "content_delta_normalize_output", True):
                return F.normalize(base_e, dim=1)
            return base_e
        delta = self._raw_content_delta(base_e, i_idx)
        max_norm = float(getattr(self.cfg, "content_delta_max_norm", 0.5))
        if max_norm >= 0.0:
            delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (max_norm / delta_norm).clamp(max=1.0)
        delta = delta * float(getattr(self.cfg, "content_delta_scale", 1.0))
        delta = delta * self._content_delta_active_mask(i_idx, force_cold)
        adjusted = base_e + delta
        if getattr(self.cfg, "content_delta_normalize_output", True):
            adjusted = F.normalize(adjusted, dim=1)
        return adjusted

    @torch.no_grad()
    def clip_content_delta_(self):
        if not getattr(self.cfg, "use_content_delta", False):
            return
        mode = str(getattr(self.cfg, "content_delta_mode", "embedding")).strip().lower()
        if mode not in {"embedding", "hybrid"}:
            return
        max_norm = float(getattr(self.cfg, "content_delta_max_norm", 0.5))
        if max_norm < 0.0:
            return
        weight = self.content_delta.weight.data
        norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        weight.mul_((max_norm / norms).clamp(max=1.0))

    @torch.no_grad()
    def content_delta_stats(self):
        if not getattr(self.cfg, "use_content_delta", False):
            return None
        was_training = self.training
        self.eval()
        try:
            delta = self._all_content_delta_vectors(detach_base=True).detach()
        finally:
            self.train(was_training)
        norms = delta.norm(dim=1)
        max_norm = float(getattr(self.cfg, "content_delta_max_norm", 0.5))
        if max_norm > 0.0:
            clipped_ratio = (norms >= max_norm * 0.999).float().mean().item()
            effective_norms = norms.clamp(max=max_norm) * abs(float(getattr(self.cfg, "content_delta_scale", 1.0)))
        else:
            clipped_ratio = 0.0
            effective_norms = norms * abs(float(getattr(self.cfg, "content_delta_scale", 1.0)))
        return {
            "mean_norm": float(norms.mean().item()),
            "max_norm": float(norms.max().item()),
            "clipped_ratio": float(clipped_ratio),
            "effective_mean_norm": float(effective_norms.mean().item()),
            "effective_max_norm": float(effective_norms.max().item()),
        }

    def content_delta_regularization(self):
        zero = self.item_id_emb.weight.new_zeros(())
        if not getattr(self.cfg, "use_content_delta", False):
            return zero
        if not any(p.requires_grad for p in self.content_delta_trainable_parameters()):
            return zero
        max_norm = float(getattr(self.cfg, "content_delta_max_norm", 0.5))
        denom = max(max_norm, 1e-6) if max_norm > 0.0 else 1.0
        norm_ratio = self._all_content_delta_vectors(detach_base=True).norm(dim=1) / denom
        reg = zero
        l2_weight = float(getattr(self.cfg, "content_delta_l2_weight", 0.0))
        if l2_weight > 0.0:
            reg = reg + l2_weight * norm_ratio.pow(2).mean()
        cap_weight = float(getattr(self.cfg, "content_delta_cap_weight", 0.0))
        if cap_weight > 0.0 and max_norm > 0.0:
            cap_margin = float(getattr(self.cfg, "content_delta_cap_margin", 0.70))
            cap_margin = min(0.99, max(0.0, cap_margin))
            reg = reg + cap_weight * F.relu(norm_ratio - cap_margin).pow(2).mean()
        return reg

    def content_delta_trainable_parameters(self):
        params = []
        if getattr(self.content_delta.weight, "requires_grad", False):
            params.append(self.content_delta.weight)
        params.extend([p for p in self.content_delta_projector.parameters() if p.requires_grad])
        return params

    def enable_delta_only_training_(self):
        delta_params = self.content_delta_trainable_parameters()
        if not delta_params:
            raise RuntimeError("Content delta-only training requested, but no delta parameters are trainable.")
        delta_param_ids = {id(p) for p in delta_params}
        for param in self.parameters():
            param.requires_grad = id(param) in delta_param_ids
        return delta_params

    def get_item_vector(self, i_idx, llm_s, force_cold=False, disable_id_dropout=False):
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
        if self.training and self.cfg.dropout_prob > 0 and not disable_id_dropout:
            dropout_mask = torch.rand((batch_size, 1), device=id_e.device) < float(self.cfg.dropout_prob)
            mask_id = mask_id | dropout_mask
        if mask_id.any():
            id_e = torch.where(mask_id, torch.zeros_like(id_e), id_e)
        content_base_e = self._content_base_embedding(i_idx)
        delta_force_cold = force_cold
        if self.training and getattr(self.cfg, "content_delta_train_on_id_dropout", False):
            delta_force_cold = mask_id.view(-1)
        content_e = self._apply_content_delta(content_base_e, i_idx, force_cold=delta_force_cold)
        llm_weight = float(getattr(self.cfg, "llm_weight", 0.0))
        if not getattr(self.cfg, "disable_llm_score", False) and llm_weight > 0.0:
            mask_llm = (llm_s > -0.5).float().unsqueeze(1)
            if getattr(self.cfg, "llm_cold_only", False) or getattr(self.cfg, "llm_hot_only", False):
                if isinstance(force_cold, torch.Tensor):
                    cold_mask = force_cold.to(device=id_e.device)
                    if cold_mask.dtype != torch.bool:
                        cold_mask = cold_mask > 0
                    cold_mask = cold_mask.float().view(-1, 1)
                elif force_cold:
                    cold_mask = torch.ones_like(mask_llm)
                else:
                    cold_mask = torch.zeros_like(mask_llm)
                if getattr(self.cfg, "llm_hot_only", False):
                    mask_llm = mask_llm * (1.0 - cold_mask)
                else:
                    mask_llm = mask_llm * cold_mask
            val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
            if getattr(self.cfg, "llm_safe_mode", False):
                neutral_llm = torch.full_like(val_llm, 0.5)
                llm_e = self.llm_proj(val_llm) - self.llm_proj(neutral_llm)
            else:
                llm_e = self.llm_proj(val_llm)
            llm_e = llm_e * mask_llm
            content_e = content_e + llm_weight * llm_e
        if getattr(self.cfg, "content_delta_replace_item", False):
            item_fused = content_e
        else:
            alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
            item_fused = alpha * id_e + (1 - alpha) * content_e
        aux_mode = str(getattr(self.cfg, "content_delta_aux_mode", "base")).strip().lower()
        aux_content_e = content_base_e if aux_mode in {"base", "raw", "no_delta"} else content_e
        return item_fused, id_e_true, aux_content_e


class FeedbackConfig(BaseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_load_course_artifacts = os.environ.get("USIM_FB_LOAD_COURSE_ARTIFACTS", "1") == "1"
        self.reward_terminal_weight = float(os.environ.get("USIM_FB_REWARD_TERM_W", "10.0"))
        self.reward_gain_weight = float(os.environ.get("USIM_FB_REWARD_GAIN_W", "5.0"))
        self.reward_gain_clip = float(os.environ.get("USIM_FB_REWARD_GAIN_CLIP", "0.05"))
        self.reward_dup_penalty_weight = float(os.environ.get("USIM_FB_REWARD_DUP_W", "0.50"))
        self.reward_cov_bonus_weight = float(os.environ.get("USIM_FB_REWARD_COV_W", "0.00"))
        self.use_course_reward = os.environ.get("USIM_USE_COURSE_REWARD", "1") == "1"
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
        self.feedback_course_redundant_concept_gate = float(
            os.environ.get("USIM_FB_COURSE_REDUNDANT_CONCEPT_GATE", "1.0")
        )
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
        path = os.path.join("outputs", "usim_feedback_fast3_content_delta", tag)
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
    model_name="USIM-Feedback-FAST3-ContentDelta",
    full_cold_item_macro=None,
    full_hot_item_macro=None,
    full_cold_item_macro_count=0,
    full_hot_item_macro_count=0,
):
    suffix = "" if protocol == "stream" else f"_{protocol}"
    detail_path = _feedback_output_path(f"final_report_usim_feedback_fast3_content_delta{suffix}.csv")
    fullrank_path = _feedback_output_path(f"final_fullrank_usim_feedback_fast3_content_delta{suffix}.csv")

    detail_rows = []
    for key in metrics_keys:
        detail_rows.append(
            {
                "metric": key,
                "sampled_cold": float(sampled_cold.get(key, 0.0)) if sampled_cold_count > 0 else None,
                "sampled_hot": float(sampled_hot.get(key, 0.0)) if sampled_hot_count > 0 else None,
                "full_cold": float(full_cold.get(key, 0.0)),
                "full_hot": float(full_hot.get(key, 0.0)),
                "full_cold_item_macro": (
                    float((full_cold_item_macro or {}).get(key, 0.0))
                    if full_cold_item_macro_count > 0 else None
                ),
                "full_hot_item_macro": (
                    float((full_hot_item_macro or {}).get(key, 0.0))
                    if full_hot_item_macro_count > 0 else None
                ),
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
        "full_cold_item_macro_r5": float((full_cold_item_macro or {}).get("R@5", 0.0)),
        "full_cold_item_macro_r10": float((full_cold_item_macro or {}).get("R@10", 0.0)),
        "full_cold_item_macro_r20": float((full_cold_item_macro or {}).get("R@20", 0.0)),
        "full_cold_item_macro_n5": float((full_cold_item_macro or {}).get("N@5", 0.0)),
        "full_cold_item_macro_n10": float((full_cold_item_macro or {}).get("N@10", 0.0)),
        "full_cold_item_macro_n20": float((full_cold_item_macro or {}).get("N@20", 0.0)),
        "full_hot_item_macro_r5": float((full_hot_item_macro or {}).get("R@5", 0.0)),
        "full_hot_item_macro_r10": float((full_hot_item_macro or {}).get("R@10", 0.0)),
        "full_hot_item_macro_r20": float((full_hot_item_macro or {}).get("R@20", 0.0)),
        "full_hot_item_macro_n5": float((full_hot_item_macro or {}).get("N@5", 0.0)),
        "full_hot_item_macro_n10": float((full_hot_item_macro or {}).get("N@10", 0.0)),
        "full_hot_item_macro_n20": float((full_hot_item_macro or {}).get("N@20", 0.0)),
        "sampled_cold_count": int(sampled_cold_count),
        "sampled_hot_count": int(sampled_hot_count),
        "full_cold_count": int(full_cold_count),
        "full_hot_count": int(full_hot_count),
        "full_cold_item_macro_count": int(full_cold_item_macro_count),
        "full_hot_item_macro_count": int(full_hot_item_macro_count),
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
        self.item_popularity_cpu = None
        self.item_difficulty = None

    def set_feedback_item_stats(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_popularity_cpu = None
            self.item_difficulty = None
            return
        pop = torch.as_tensor(item_popularity, dtype=torch.float32, device=self.device)
        if pop.numel() != self.cfg.n_items:
            raise ValueError(f"item_popularity size mismatch: expect {self.cfg.n_items}, got {pop.numel()}")
        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_popularity_cpu = pop.detach().cpu()
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    def _compute_aux_loss(self, id_e_true, content_e, effective_cold):
        """Auxiliary InfoNCE between ID and content towers (id_e_true <-> content_e).

        Two paths controlled by `cfg.aux_hot_only` (env: `USIM_AUX_HOT_ONLY`):

        - aux_hot_only=False (default, legacy): InfoNCE over the full batch.
          Identical to the pre-refactor implementation.
        - aux_hot_only=True: restrict the InfoNCE to hot rows only. For cold
          items, `id_e_true` is rarely observed during training, so the diagonal
          alignment signal is mostly noise; this branch removes that gradient
          channel without affecting hot rows.

        Returns a scalar tensor on the same device as `id_e_true`. When the
        hot-only branch finds <2 hot rows in the batch, returns a 0 scalar
        (need at least 2 rows for cross-entropy).
        """
        aux_hot_only = bool(getattr(self.cfg, "aux_hot_only", False))
        if not aux_hot_only or effective_cold is None:
            # Legacy path: bit-identical to the original block in forward().
            z_id = F.normalize(id_e_true, dim=1)
            z_con = F.normalize(content_e, dim=1)
            labels = torch.arange(z_id.size(0), device=z_id.device)
            sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
            return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        hot_mask = ~effective_cold.to(device=id_e_true.device).view(-1)
        n_hot = int(hot_mask.sum().item())
        if n_hot < 2:
            # Connect the zero scalar to id_e_true so the returned tensor
            # carries a grad_fn even when this branch contributes nothing.
            # This prevents `loss.backward()` from raising when the aux term
            # is the only loss path (matters for unit tests; harmless in the
            # full forward where main_loss already drives the graph).
            return (id_e_true.sum() * 0.0) + (content_e.sum() * 0.0)
        hot_idx = hot_mask.nonzero(as_tuple=False).view(-1)
        z_id = F.normalize(id_e_true.index_select(0, hot_idx), dim=1)
        z_con = F.normalize(content_e.index_select(0, hot_idx), dim=1)
        labels = torch.arange(n_hot, device=z_id.device)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

    def _build_seen_mat(self, user_ids, user_seen_items):
        # Fast path: use precomputed (n_users, n_items) bool index if available.
        seen_index = getattr(self, "user_seen_index", None)
        if seen_index is not None:
            uid_t = self._resolve_user_id_tensor(user_ids)
            seen_mat = seen_index.index_select(0, uid_t).float()
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)
        # Fallback: legacy per-batch construction (kept for safety / online updates).
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

    def _build_known_positive_batch_mask(self, user_ids, item_idx, user_seen_items):
        seen_index = getattr(self, "user_seen_index", None)
        if seen_index is not None:
            uid_t = self._resolve_user_id_tensor(user_ids)
            return seen_index.index_select(0, uid_t)[:, item_idx]
        if user_seen_items is None:
            return None
        seen_mat, _ = self._build_seen_mat(user_ids, user_seen_items)
        if seen_mat.numel() < 1:
            return None
        return seen_mat[:, item_idx] > 0

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
            active = self._cold_mask_from_pop(target_pop).float().view(-1, 1)

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
            redundant_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_redundant_concept_gate)))
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant_gate * redundant)
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
            active = self._cold_mask_from_pop(target_pop).float().view(-1, 1)
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
            redundant_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_redundant_concept_gate)))
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant_gate * redundant)

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

    def _paac_batch_pop_mask(self, pop):
        batch_size = int(pop.numel())
        if batch_size < 2:
            return torch.zeros_like(pop, dtype=torch.bool)
        mode = str(getattr(self.cfg, "paac_group_mode", "batch_quantile")).strip().lower()
        if mode in {"threshold", "cold_hot", "absolute"}:
            return pop.float() >= float(self.cfg.cold_threshold)
        ratio = float(getattr(self.cfg, "paac_batch_pop_ratio", 0.50))
        ratio = min(0.99, max(0.01, ratio))
        top_k = int(math.ceil(batch_size * ratio))
        top_k = max(1, min(batch_size - 1, top_k))
        order = torch.argsort(pop.float(), descending=True)
        mask = torch.zeros_like(pop, dtype=torch.bool)
        mask[order[:top_k]] = True
        return mask

    def _paac_reweighted_contrast_loss(self, logits, labels, pop, known_positive_mask=None):
        zero = logits.new_zeros(())
        if not getattr(self.cfg, "use_paac", False):
            return zero
        if float(getattr(self.cfg, "paac_contrast_weight", 0.0)) <= 0.0:
            return zero
        if logits.size(0) < 2 or self.item_popularity is None:
            return zero

        pop_mask = self._paac_batch_pop_mask(pop)
        if pop_mask.sum().item() < 1 or (~pop_mask).sum().item() < 1:
            return zero

        beta = float(getattr(self.cfg, "paac_contrast_beta", 0.35))
        beta = min(1.0, max(0.0, beta))
        gamma = float(getattr(self.cfg, "paac_contrast_gamma", 0.35))
        gamma = min(1.0, max(0.0, gamma))

        shifted = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(shifted)
        same_group = pop_mask.view(-1, 1) == pop_mask.view(1, -1)
        weights = torch.where(
            same_group,
            torch.ones_like(exp_logits),
            exp_logits.new_full(exp_logits.shape, beta),
        )
        if known_positive_mask is not None:
            off_diag = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            weights = weights.masked_fill(known_positive_mask & off_diag, 0.0)
        denom = (exp_logits * weights).sum(dim=1).clamp_min(1e-12)
        pos = exp_logits.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(1e-12)
        row_loss = -torch.log(pos / denom)

        pop_loss = row_loss[pop_mask].mean()
        unpop_loss = row_loss[~pop_mask].mean()
        return gamma * pop_loss + (1.0 - gamma) * unpop_loss

    def _paac_choose_alignment_pairs(self, user_ids, item_idx, user_seen_items):
        if user_seen_items is None or self.item_popularity_cpu is None:
            return [], []
        max_pairs = max(0, int(getattr(self.cfg, "paac_align_max_pairs", 512)))
        if max_pairs < 1:
            return [], []

        cold_threshold = float(self.cfg.cold_threshold)
        pop_ids = []
        unpop_ids = []
        user_list = [int(x) for x in user_ids.detach().cpu().tolist()]
        item_list = [int(x) for x in item_idx.detach().cpu().tolist()]
        popularity = self.item_popularity_cpu

        for uid, cur_item in zip(user_list, item_list):
            if cur_item < 0 or cur_item >= self.cfg.n_items:
                continue
            seen = user_seen_items.get(uid)
            if not seen:
                continue

            hot_item = None
            hot_pop = -1.0
            cold_item = None
            cold_pop = float("inf")
            for seen_item in seen:
                seen_item = int(seen_item)
                if seen_item == cur_item or seen_item < 0 or seen_item >= self.cfg.n_items:
                    continue
                seen_pop = float(popularity[seen_item].item())
                if seen_pop >= cold_threshold and seen_pop > hot_pop:
                    hot_item = seen_item
                    hot_pop = seen_pop
                if seen_pop < cold_threshold and seen_pop < cold_pop:
                    cold_item = seen_item
                    cold_pop = seen_pop

            cur_pop = float(popularity[cur_item].item())
            if cur_pop < cold_threshold and hot_item is not None:
                pop_ids.append(hot_item)
                unpop_ids.append(cur_item)
            elif cur_pop >= cold_threshold and cold_item is not None:
                pop_ids.append(cur_item)
                unpop_ids.append(cold_item)

            if len(pop_ids) >= max_pairs:
                break

        return pop_ids, unpop_ids

    def _paac_supervised_alignment_loss(self, user_ids, item_idx, user_seen_items):
        zero = self.user_emb.weight.new_zeros(())
        if not getattr(self.cfg, "use_paac", False):
            return zero, 0
        if float(getattr(self.cfg, "paac_align_weight", 0.0)) <= 0.0:
            return zero, 0
        if self.item_popularity is None:
            return zero, 0

        pop_ids, unpop_ids = self._paac_choose_alignment_pairs(user_ids, item_idx, user_seen_items)
        if not pop_ids:
            return zero, 0

        pop_t = torch.tensor(pop_ids, dtype=torch.long, device=self.device)
        unpop_t = torch.tensor(unpop_ids, dtype=torch.long, device=self.device)
        neutral_llm = torch.full((pop_t.numel(),), -1.0, dtype=torch.float32, device=self.device)

        pop_vec, _, _ = self.get_item_vector(
            pop_t,
            neutral_llm,
            force_cold=False,
            disable_id_dropout=True,
        )
        unpop_vec, _, _ = self.get_item_vector(
            unpop_t,
            neutral_llm,
            force_cold=True,
            disable_id_dropout=True,
        )
        pop_vec = F.normalize(pop_vec, dim=1)
        unpop_vec = F.normalize(unpop_vec, dim=1)
        if getattr(self.cfg, "paac_align_detach_hot", True):
            pop_vec = pop_vec.detach()
        align_loss = (pop_vec - unpop_vec).pow(2).sum(dim=1).mean()
        return align_loss, int(pop_t.numel())

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
        is_cold = self._cold_mask_from_pop(pop)
        effective_cold = self._effective_train_cold_mask(pop)
        episode_pop = self._target_pop_with_effective_cold(pop, effective_cold)
        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = effective_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)
        target_emb = z_i_base.detach().clone()
        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
            item_idx=i,
            target_pop=episode_pop,
            user_seen_items=user_seen_items,
        )
        pseudo_cold_mask = effective_cold & (~is_cold)
        candidate_stats["pseudo_cold_count"] = int(pseudo_cold_mask.sum().detach().item())
        candidate_stats["pseudo_cold_ratio"] = (
            float(pseudo_cold_mask.float().mean().detach().item()) if pseudo_cold_mask.numel() > 0 else 0.0
        )
        candidate_stats["effective_cold_ratio"] = (
            float(effective_cold.float().mean().detach().item()) if effective_cold.numel() > 0 else 0.0
        )
        ppo_loss = self.compute_ppo_loss(trajectory)
        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0), device=self.device)
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp
        known_positive_mask = None
        false_neg_mask = None
        fn_mask_ratio = 0.0
        if (
            self.training and getattr(self.cfg, "mask_known_pos_neg", False) and
            user_seen_items is not None and logits_margin.size(0) > 1
        ):
            known_positive_mask = self._build_known_positive_batch_mask(u, i, user_seen_items)
            if known_positive_mask is not None:
                false_neg_mask = known_positive_mask & (~pos_mask)
                off_diag_count = max(1, logits_margin.numel() - logits_margin.size(0))
                fn_mask_ratio = float(false_neg_mask.sum().detach().item()) / float(off_diag_count)
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
                if false_neg_mask is not None:
                    neg_logits = neg_logits.masked_fill(false_neg_mask, -1e9)
                hard_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                rand_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                if n_hard > 0:
                    if self.cfg.use_structured_hard_neg and self.item_hard_adj is not None:
                        hard_mask = self.item_hard_adj[i][:, i]
                        hard_mask = hard_mask & (~pos_mask)
                        if false_neg_mask is not None:
                            hard_mask = hard_mask & (~false_neg_mask)
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
                        if false_neg_mask is not None:
                            neg_logits = neg_logits.masked_fill(false_neg_mask, -1e9)
                        _, hard_idx = torch.topk(neg_logits, k=n_hard, dim=1)
                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if false_neg_mask is not None:
                        rand_scores = rand_scores.masked_fill(false_neg_mask, -1e9)
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(rand_scores, k=n_rand, dim=1)
                cand_idx = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                main_loss = F.cross_entropy(cand_logits, main_targets)
            else:
                main_logits = logits_margin
                if false_neg_mask is not None:
                    main_logits = main_logits.masked_fill(false_neg_mask, -1e9)
                main_loss = F.cross_entropy(main_logits, labels)
        else:
            main_logits = logits_margin
            if false_neg_mask is not None:
                main_logits = main_logits.masked_fill(false_neg_mask, -1e9)
            main_loss = F.cross_entropy(main_logits, labels)
        paac_contrast_loss = logits.new_zeros(())
        paac_align_loss = logits.new_zeros(())
        paac_align_pairs = 0
        if self.training and getattr(self.cfg, "use_paac", False):
            paac_contrast_loss = self._paac_reweighted_contrast_loss(
                logits_margin,
                labels,
                pop,
                known_positive_mask=known_positive_mask,
            )
            paac_align_loss, paac_align_pairs = self._paac_supervised_alignment_loss(
                u,
                i,
                user_seen_items,
            )
        # Auxiliary InfoNCE between id_e_true and content_e. Routed through
        # `_compute_aux_loss` so the hot-only rollout (USIM_AUX_HOT_ONLY=1) can
        # be flipped without touching the rest of the forward pass. Default
        # behavior is bit-identical to the original full-batch InfoNCE.
        aux_loss = self._compute_aux_loss(id_e_true, content_e, effective_cold)
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
                valid_rows = valid_rows & effective_cold
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
        delta_reg_loss = self.content_delta_regularization()
        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            float(getattr(self.cfg, "paac_contrast_weight", 0.0)) * paac_contrast_loss +
            float(getattr(self.cfg, "paac_align_weight", 0.0)) * paac_align_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            delta_reg_loss
        )
        candidate_stats["main_loss"] = float(main_loss.detach().item())
        candidate_stats["aux_loss"] = float(aux_loss.detach().item())
        candidate_stats["ppo_loss"] = float(ppo_loss.detach().item())
        candidate_stats["prereq_aux_loss"] = float(prereq_aux_loss.detach().item())
        candidate_stats["delta_reg_loss"] = float(delta_reg_loss.detach().item())
        candidate_stats["total_loss"] = float(total_loss.detach().item())
        candidate_stats["paac_contrast_loss"] = float(paac_contrast_loss.detach().item())
        candidate_stats["paac_align_loss"] = float(paac_align_loss.detach().item())
        candidate_stats["paac_align_pairs"] = int(paac_align_pairs)
        candidate_stats["fn_mask_ratio"] = float(fn_mask_ratio)
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
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "usim_feedback_fast3_content_delta"))


class Fast3FeedbackUSIM(FastFeedbackUSIM):
    def _compute_target_alpha(self, target_pop, step_idx, entropy, num_candidates, batch_size):
        if target_pop is not None:
            cold_mask = self._cold_mask_from_pop(target_pop).float().view(-1, 1)
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
            if getattr(self.cfg, "use_course_reward", True):
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
    values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
    return {key: float(val.mean().item()) for key, val in values.items()}


def compute_ranking_metric_values(scores, target_indices, k_list=[5, 10, 20]):
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        hit_ranks = torch.where(preds == targets)
        ndcg_vals = torch.zeros(batch_size, device=scores.device)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            ndcg_vals[hit_ranks[0]] = 1.0 / torch.log2(ranks + 2.0)
        results[f"R@{k}"] = hits
        results[f"N@{k}"] = ndcg_vals
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


def _is_pair_llm_key(key):
    return isinstance(key, tuple) and len(key) == 2


def _is_item_llm_key(key):
    return isinstance(key, (int, np.integer))


def _count_llm_key_types(llm_scores):
    if not llm_scores:
        return 0, 0
    pair_count = 0
    item_count = 0
    for key in llm_scores:
        if _is_pair_llm_key(key):
            pair_count += 1
        elif _is_item_llm_key(key):
            item_count += 1
    return pair_count, item_count


def _item_mean_llm_scores(llm_scores, keep_pair=False):
    item_scores = {}
    pair_sums = defaultdict(float)
    pair_counts = defaultdict(int)
    if not llm_scores:
        return item_scores

    for key, value in llm_scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        if _is_pair_llm_key(key):
            item_idx = int(key[1])
            pair_sums[item_idx] += score
            pair_counts[item_idx] += 1
            if keep_pair:
                item_scores[(int(key[0]), item_idx)] = score
        elif _is_item_llm_key(key):
            item_scores[int(key)] = score

    for item_idx, total in pair_sums.items():
        if item_idx not in item_scores and pair_counts[item_idx] > 0:
            item_scores[item_idx] = total / pair_counts[item_idx]
    return item_scores


def prepare_llm_scores(llm_scores, cfg):
    raw_pair, raw_item = _count_llm_key_types(llm_scores)
    raw_total = len(llm_scores) if llm_scores else 0
    mode = str(getattr(cfg, "llm_bank_mode", "item") or "item").strip().lower()
    if mode in {"off", "disable", "disabled"}:
        mode = "none"
    elif mode in {"item", "item_avg", "item_mean", "mean_item"}:
        mode = "item_mean"
    elif mode in {"hybrid", "pair_item", "pair_item_mean"}:
        mode = "hybrid_mean"

    if getattr(cfg, "disable_llm_score", False) or mode == "none" or float(getattr(cfg, "llm_weight", 1.0)) <= 0.0:
        effective_scores = {}
        effective_mode = "none"
    elif mode == "item_mean":
        effective_scores = _item_mean_llm_scores(llm_scores, keep_pair=False)
        effective_mode = "item_mean"
    elif mode == "hybrid_mean":
        effective_scores = _item_mean_llm_scores(llm_scores, keep_pair=True)
        effective_mode = "hybrid_mean"
    elif mode == "item_only":
        effective_scores = {
            int(key): float(value)
            for key, value in (llm_scores or {}).items()
            if _is_item_llm_key(key)
        }
        effective_mode = "item_only"
    else:
        effective_scores = llm_scores or {}
        effective_mode = "pair"

    eff_pair, eff_item = _count_llm_key_types(effective_scores)
    summary = {
        "mode": effective_mode,
        "raw_total": raw_total,
        "raw_pair": raw_pair,
        "raw_item": raw_item,
        "effective_total": len(effective_scores),
        "effective_pair": eff_pair,
        "effective_item": eff_item,
    }
    return effective_scores, summary


def _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=None):
    values = [_lookup_llm_score(llm_scores, item_idx, user_idx) for user_idx, item_idx in zip(user_ids, item_ids)]
    return torch.tensor(values, dtype=torch.float, device=device)


def _resolve_eval_force_cold(model, idx_batch, force_cold):
    if isinstance(force_cold, str):
        mode = force_cold.strip().lower()
        if mode in {"auto", "item", "item_pop"} and getattr(model, "item_popularity", None) is not None:
            return model.item_popularity[idx_batch].to(device=idx_batch.device) < float(model.cfg.cold_threshold)
        if mode in {"true", "cold", "all"}:
            return True
        if mode in {"false", "hot", "none", "off"}:
            return False
    return force_cold


def build_all_item_vecs(model, device, llm_scores, item_batch=1024, force_cold=True):
    n_items = model.cfg.n_items
    all_item_idx = torch.arange(n_items, device=device)
    bank_mode = getattr(model.cfg, "llm_bank_mode", "none")
    if bank_mode in {"item", "item_mean", "hybrid_mean", "item_only"}:
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
            force_cold_batch = _resolve_eval_force_cold(model, idx_batch, force_cold)
            z_i, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=force_cold_batch)
            all_item_vecs.append(F.normalize(z_i, dim=1))
    return torch.cat(all_item_vecs, dim=0)


def build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    if getattr(model.cfg, "legacy_train_protocol", False):
        hot_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=False)
        cold_force_cold = True
        if getattr(model.cfg, "content_delta_cold_only", False):
            bank_mode = str(getattr(model.cfg, "content_delta_eval_bank_mode", "auto")).strip().lower()
            if bank_mode in {"auto", "item", "item_pop"}:
                cold_force_cold = "auto"
            elif bank_mode in {"hot", "none", "off"}:
                cold_force_cold = False
            else:
                cold_force_cold = True
        cold_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=cold_force_cold)
        return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}

    was_training = model.training
    model.eval()
    try:
        hot_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=False)
        cold_force_cold = True
        if getattr(model.cfg, "content_delta_cold_only", False):
            bank_mode = str(getattr(model.cfg, "content_delta_eval_bank_mode", "auto")).strip().lower()
            if bank_mode in {"auto", "item", "item_pop"}:
                cold_force_cold = "auto"
            elif bank_mode in {"hot", "none", "off"}:
                cold_force_cold = False
            else:
                cold_force_cold = True
        cold_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=cold_force_cold)
        return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}
    finally:
        model.train(was_training)


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
                  eval_type="cold", full_ranking=False, user_seen_items=None, all_item_vecs=None,
                  average_mode="interaction"):
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    model.eval()
    accum_metrics = {}
    total_samples = 0
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    item_counts = {}
    seen_tensor_cache = {}
    seen_index = getattr(model, "user_seen_index", None)
    use_seen_index = seen_index is not None and user_seen_items is not None
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
            # Only populate the per-uid Python cache when the fast index isn't available
            # (legacy path used by sampled-eval forbidden-mask construction & course rerank).
            need_legacy_cache = (not use_seen_index) and (
                (not full_ranking) or model.cfg.use_course_rerank
            )
            if need_legacy_cache:
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
                if use_seen_index:
                    # Vectorized seen-mask in one GPU op (replaces per-user Python loop)
                    seen_mask_full = seen_index.index_select(0, u)  # (n_sel, n_items) bool
                    scores = scores.masked_fill(seen_mask_full, -1e9)
                    scores[row_idx, i] = target_scores
                elif user_seen_items:
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
                if use_seen_index:
                    # Build the (n_sel, n_items) forbidden mask in one GPU op
                    forbidden_full = seen_index.index_select(0, u).clone()  # bool, copy to allow mutation
                    row_idx_eval = torch.arange(n_sel, device=device)
                    forbidden_full[row_idx_eval, i] = True
                    avail_per_row = (~forbidden_full).sum(dim=1)
                    avail_counts = avail_per_row.clamp_min(1).tolist()
                else:
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
                for row in range(n_sel):
                    if use_seen_index:
                        forbidden = forbidden_full[row]
                    else:
                        forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                        forbidden[i[row]] = True
                        seen_idx = seen_tensor_cache[int(user_ids[row])]
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
            batch_values = compute_ranking_metric_values(scores, target_indices=target_indices, k_list=k_list)
            if average_mode == "item_macro":
                for row, item_id in enumerate(item_ids):
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    for key, values in batch_values.items():
                        per_item = item_accum[key]
                        per_item[item_id] = per_item.get(item_id, 0.0) + float(values[row].detach().cpu().item())
            else:
                for k, values in batch_values.items():
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + float(values.sum().detach().cpu().item())
            total_samples += n_sel
    if total_samples == 0:
        return None, 0
    if average_mode == "item_macro":
        if not item_counts:
            return None, 0
        macro = {}
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        return macro, len(item_counts)
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


def _static_seed():
    return int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025")))


def _split_exact_warm_user(source_df, seed, train_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    base_train_idx = []
    remaining_idx = []
    for _, group in source_df.groupby("u_idx", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        if idx.size == 0:
            continue
        base_train_idx.append(int(idx[0]))
        if idx.size > 1:
            remaining_idx.extend(int(x) for x in idx[1:])

    n_total = len(source_df)
    n_train_target = int(round(n_total * train_ratio))
    n_train_target = min(n_total, max(len(base_train_idx), n_train_target))
    n_val_target = int(round(n_total * val_ratio))
    n_val_target = max(0, min(n_val_target, n_total - n_train_target))

    remaining_idx = np.array(remaining_idx, dtype=np.int64)
    rng.shuffle(remaining_idx)
    extra_train = max(0, n_train_target - len(base_train_idx))
    extra_train = min(extra_train, remaining_idx.size)

    train_idx = list(base_train_idx) + [int(x) for x in remaining_idx[:extra_train]]
    tail_idx = remaining_idx[extra_train:]
    val_idx = [int(x) for x in tail_idx[:n_val_target]]
    test_idx = [int(x) for x in tail_idx[n_val_target:]]
    return train_idx, val_idx, test_idx


def _split_user_leave_one_out(source_df, seed, train_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    for _, group in source_df.groupby("u_idx", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        if n >= 3:
            n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
            n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
            if n_val + n_test >= n:
                n_val = 1 if val_ratio > 0 else 0
                n_test = 1
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_idx.extend(int(x) for x in idx[:n_train])
        if n_val > 0:
            val_idx.extend(int(x) for x in idx[n_train:n_train + n_val])
        if n_test > 0:
            test_idx.extend(int(x) for x in idx[n_train + n_val:n_train + n_val + n_test])
    return train_idx, val_idx, test_idx


def _loc_split(source_df, idx, split_source, seed=None, shuffle=False):
    if len(idx) == 0:
        out = source_df.iloc[0:0].copy()
    else:
        out = source_df.loc[idx].copy()
    if shuffle and len(out) > 0:
        out = out.sample(frac=1.0, random_state=seed)
    out["_split_source"] = split_source
    return out.reset_index(drop=True)


def _ensure_train_item_coverage(train_df, val_df, test_df, required_items):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    required_items = set(int(x) for x in required_items)
    train_items = set(train_df["i_idx"].astype(int))
    missing_items = sorted(required_items - train_items)
    moved_rows = 0

    for item_id in missing_items:
        moved = None
        for split_name in ("val", "test"):
            src_df = val_df if split_name == "val" else test_df
            hit_idx = src_df.index[src_df["i_idx"].astype(int) == item_id]
            if len(hit_idx) < 1:
                continue
            row_idx = hit_idx[0]
            moved = src_df.loc[[row_idx]].copy()
            moved["_split_source"] = moved["_split_source"].astype(str) + "_coverage_train"
            if split_name == "val":
                val_df = val_df.drop(index=row_idx)
            else:
                test_df = test_df.drop(index=row_idx)
            break

        if moved is not None:
            train_df = pd.concat([train_df, moved], ignore_index=True)
            moved_rows += 1

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        moved_rows,
    )


def _make_balanced_item_folds(item_counts, eligible_items, n_folds):
    eligible_items = [int(x) for x in eligible_items]
    n_folds = int(n_folds)
    if n_folds < 3:
        raise ValueError(f"USIM_STATIC_COLD_ITEM_FOLDS must be >= 3, got {n_folds}")
    if len(eligible_items) < n_folds:
        raise ValueError(
            f"Not enough eligible items ({len(eligible_items)}) for {n_folds} balanced folds"
        )

    sorted_items = (
        item_counts.loc[eligible_items]
        .sort_values(ascending=False, kind="mergesort")
        .index.astype(int)
        .tolist()
    )
    base_size = len(sorted_items) // n_folds
    extra = len(sorted_items) % n_folds
    capacities = [base_size + (1 if fold_id < extra else 0) for fold_id in range(n_folds)]
    folds = [[] for _ in range(n_folds)]
    fold_sums = [0 for _ in range(n_folds)]

    for item_id in sorted_items:
        candidates = [idx for idx in range(n_folds) if len(folds[idx]) < capacities[idx]]
        fold_id = min(candidates, key=lambda idx: (fold_sums[idx], len(folds[idx]), idx))
        folds[fold_id].append(int(item_id))
        fold_sums[fold_id] += int(item_counts.loc[item_id])

    return folds, fold_sums


def _sample_strict_item_cold_items(item_counts, eligible_items, seed, split_mode):
    val_item_ratio = float(os.environ.get("USIM_STATIC_VAL_COLD_ITEM_RATIO", "0.05"))
    test_item_ratio = float(os.environ.get("USIM_STATIC_COLD_ITEM_RATIO", "0.10"))
    n_val_items = max(1, int(round(eligible_items.size * val_item_ratio)))
    n_test_items = max(1, int(round(eligible_items.size * test_item_ratio)))
    if n_val_items + n_test_items >= eligible_items.size:
        n_val_items = max(1, min(n_val_items, eligible_items.size // 4))
        n_test_items = max(1, min(n_test_items, eligible_items.size // 4))

    if split_mode in {"strict_item_cold_balanced", "item_cold_balanced", "balanced_item_cold"}:
        n_folds = int(os.environ.get("USIM_STATIC_COLD_ITEM_FOLDS", "20"))
        n_folds = min(n_folds, int(eligible_items.size))
        folds, fold_sums = _make_balanced_item_folds(item_counts, eligible_items, n_folds)
        n_val_folds = max(1, int(round(n_folds * val_item_ratio)))
        n_test_folds = max(1, int(round(n_folds * test_item_ratio)))
        if n_val_folds + n_test_folds >= n_folds:
            n_val_folds = max(1, min(n_val_folds, n_folds // 4))
            n_test_folds = max(1, min(n_test_folds, n_folds // 4))
        if n_val_folds + n_test_folds >= n_folds:
            raise ValueError(
                "Invalid balanced item-cold fold allocation: "
                f"folds={n_folds}, val_folds={n_val_folds}, test_folds={n_test_folds}"
            )

        rng_items = np.random.default_rng(seed)
        fold_order = np.arange(n_folds)
        rng_items.shuffle(fold_order)
        val_fold_ids = [int(x) for x in fold_order[:n_val_folds]]
        test_fold_ids = [int(x) for x in fold_order[n_val_folds:n_val_folds + n_test_folds]]
        val_cold_items = {int(item_id) for fold_id in val_fold_ids for item_id in folds[fold_id]}
        test_cold_items = {int(item_id) for fold_id in test_fold_ids for item_id in folds[fold_id]}
        fold_sums_arr = np.asarray(fold_sums, dtype=np.float64)
        return val_cold_items, test_cold_items, {
            "strict_item_cold_sampling": "balanced_item_folds",
            "strict_item_cold_folds": int(n_folds),
            "strict_item_cold_val_folds": int(n_val_folds),
            "strict_item_cold_test_folds": int(n_test_folds),
            "strict_item_cold_val_fold_ids": val_fold_ids,
            "strict_item_cold_test_fold_ids": test_fold_ids,
            "strict_item_cold_fold_item_count_min": int(min(len(fold) for fold in folds)),
            "strict_item_cold_fold_item_count_max": int(max(len(fold) for fold in folds)),
            "strict_item_cold_fold_pop_sum_min": int(fold_sums_arr.min()),
            "strict_item_cold_fold_pop_sum_mean": float(fold_sums_arr.mean()),
            "strict_item_cold_fold_pop_sum_max": int(fold_sums_arr.max()),
            "strict_item_cold_fold_pop_sum_std": float(fold_sums_arr.std(ddof=1)) if n_folds > 1 else 0.0,
        }

    rng_items = np.random.default_rng(seed)
    shuffled_items = eligible_items.copy()
    rng_items.shuffle(shuffled_items)
    val_cold_items = {int(x) for x in shuffled_items[:n_val_items]}
    test_cold_items = {int(x) for x in shuffled_items[n_val_items:n_val_items + n_test_items]}
    return val_cold_items, test_cold_items, {
        "strict_item_cold_sampling": "random_items",
        "strict_item_cold_target_val_items": int(n_val_items),
        "strict_item_cold_target_test_items": int(n_test_items),
    }


def _static_split_df(df):
    seed = _static_seed()
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))
    split_mode = os.environ.get("USIM_STATIC_SPLIT_MODE", "user_threshold_exact").strip().lower()
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    work_df = df.copy().reset_index(drop=True)
    work_df["_row_id"] = np.arange(len(work_df), dtype=np.int64)

    strict_item_cold = split_mode in {
        "item_cold",
        "cold_item",
        "strict_item_cold",
        "strict_item_cold_balanced",
        "item_cold_balanced",
        "balanced_item_cold",
    }
    coverage_moves = 0
    if strict_item_cold:
        item_counts = work_df["i_idx"].astype(int).value_counts()
        min_inter = int(os.environ.get("USIM_STATIC_COLD_ITEM_MIN_INTER", "5"))
        eligible_items = item_counts[item_counts >= min_inter].index.to_numpy(copy=True)
        if eligible_items.size < 3:
            raise ValueError(f"Not enough items for strict item-cold split: eligible_items={eligible_items.size}")
        val_cold_items, test_cold_items, cold_sampling_info = _sample_strict_item_cold_items(
            item_counts,
            eligible_items,
            seed,
            split_mode,
        )
        heldout_items = val_cold_items | test_cold_items
        source_df = work_df[~work_df["i_idx"].astype(int).isin(heldout_items)].copy()
        train_idx, val_idx, test_idx = _split_exact_warm_user(source_df, seed, train_ratio, val_ratio)
        train_df = _loc_split(source_df, train_idx, "strict_item_cold_train", seed=seed, shuffle=True)
        val_warm_df = _loc_split(source_df, val_idx, "strict_item_cold_warm_val")
        test_warm_df = _loc_split(source_df, test_idx, "strict_item_cold_warm_test")
        train_df, val_warm_df, test_warm_df, coverage_moves = _ensure_train_item_coverage(
            train_df,
            val_warm_df,
            test_warm_df,
            source_df["i_idx"].astype(int).unique(),
        )
        train_users = set(train_df["u_idx"].astype(int))
        val_cold_df = work_df[
            work_df["i_idx"].astype(int).isin(val_cold_items)
            & work_df["u_idx"].astype(int).isin(train_users)
        ].copy()
        test_cold_df = work_df[
            work_df["i_idx"].astype(int).isin(test_cold_items)
            & work_df["u_idx"].astype(int).isin(train_users)
        ].copy()
        val_cold_df["_split_source"] = "strict_item_cold_val"
        test_cold_df["_split_source"] = "strict_item_cold_test"
        val_df = pd.concat([val_warm_df, val_cold_df], ignore_index=True)
        test_df = pd.concat([test_warm_df, test_cold_df], ignore_index=True)
        split_family = "strict_item_cold"
    elif split_mode in {"user", "per_user", "user_history", "user-stratified", "user_leave_one_out"}:
        train_idx, val_idx, test_idx = _split_user_leave_one_out(work_df, seed, train_ratio, val_ratio)
        train_df = _loc_split(work_df, train_idx, "user_leave_one_out_train", seed=seed, shuffle=True)
        val_df = _loc_split(work_df, val_idx, "user_leave_one_out_val")
        test_df = _loc_split(work_df, test_idx, "user_leave_one_out_test")
        split_family = "user_leave_one_out"
    elif split_mode in {"global", "random"}:
        rng = np.random.default_rng(seed)
        idx = work_df.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_ratio))
        n_val = int(round(len(idx) * val_ratio))
        train_df = _loc_split(work_df, idx[:n_train], "global_train", seed=seed, shuffle=True)
        val_df = _loc_split(work_df, idx[n_train:n_train + n_val], "global_val")
        test_df = _loc_split(work_df, idx[n_train + n_val:], "global_test")
        split_family = "global"
    elif split_mode in {"threshold", "user_threshold", "user_threshold_exact", "user_exact"}:
        train_idx, val_idx, test_idx = _split_exact_warm_user(work_df, seed, train_ratio, val_ratio)
        train_df = _loc_split(work_df, train_idx, "user_threshold_exact_train", seed=seed, shuffle=True)
        val_df = _loc_split(work_df, val_idx, "user_threshold_exact_val")
        test_df = _loc_split(work_df, test_idx, "user_threshold_exact_test")
        train_df, val_df, test_df, coverage_moves = _ensure_train_item_coverage(
            train_df,
            val_df,
            test_df,
            work_df["i_idx"].astype(int).unique(),
        )
        split_family = "user_threshold_exact"
    else:
        raise ValueError(f"Unsupported USIM_STATIC_SPLIT_MODE={split_mode!r}")

    train_users = set(train_df["u_idx"].astype(int))
    val_users = set(val_df["u_idx"].astype(int))
    test_users = set(test_df["u_idx"].astype(int))
    split_info = {
        "seed": seed,
        "split_mode": split_mode,
        "split_family": split_family,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "actual_train_ratio": float(len(train_df) / max(1, len(work_df))),
        "actual_val_ratio": float(len(val_df) / max(1, len(work_df))),
        "actual_test_ratio": float(len(test_df) / max(1, len(work_df))),
        "val_user_seen_ratio": float(len(val_users & train_users) / max(1, len(val_users))),
        "test_user_seen_ratio": float(len(test_users & train_users) / max(1, len(test_users))),
        "train_item_coverage_moves": int(coverage_moves),
    }
    if strict_item_cold:
        val_item_pop = item_counts.loc[list(val_cold_items)].astype(int) if val_cold_items else pd.Series(dtype=int)
        test_item_pop = item_counts.loc[list(test_cold_items)].astype(int) if test_cold_items else pd.Series(dtype=int)
        split_info.update(
            {
                "val_cold_items": int(len(val_cold_items)),
                "test_cold_items": int(len(test_cold_items)),
                "strict_item_cold_min_inter": int(os.environ.get("USIM_STATIC_COLD_ITEM_MIN_INTER", "5")),
                "strict_item_cold_eligible_items": int(len(eligible_items)),
                "strict_item_cold_val_item_pop_sum": int(val_item_pop.sum()) if len(val_item_pop) else 0,
                "strict_item_cold_test_item_pop_sum": int(test_item_pop.sum()) if len(test_item_pop) else 0,
                "strict_item_cold_val_item_pop_mean": float(val_item_pop.mean()) if len(val_item_pop) else 0.0,
                "strict_item_cold_test_item_pop_mean": float(test_item_pop.mean()) if len(test_item_pop) else 0.0,
            }
        )
        split_info.update(cold_sampling_info)
    return train_df, val_df, test_df, split_info


def _apply_train_popularity(train_df, val_df, test_df, cfg):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_counts = train_df["i_idx"].astype(int).value_counts().astype(int)
    for split_df in (train_df, val_df, test_df):
        if "raw_popularity" not in split_df.columns and "popularity" in split_df.columns:
            split_df["raw_popularity"] = split_df["popularity"]
        split_df["popularity"] = (
            split_df["i_idx"].astype(int).map(train_counts).fillna(0).astype(int)
        )
    item_train_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    for item_id, pop_value in train_counts.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_train_pop[idx] = int(pop_value)
    return train_df, val_df, test_df, item_train_pop


def _static_split_counts(train_df, val_df, test_df, cfg):
    rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows.append(
            {
                "split": name,
                "rows": int(len(split_df)),
                "users": int(split_df["u_idx"].nunique()),
                "items": int(split_df["i_idx"].nunique()),
                "cold_rows": int((split_df["popularity"] < cfg.cold_threshold).sum()),
                "hot_rows": int((split_df["popularity"] >= cfg.cold_threshold).sum()),
                "zero_train_pop_rows": int((split_df["popularity"] == 0).sum()),
                "cold_threshold": int(cfg.cold_threshold),
            }
        )
    return rows


def _write_static_split_artifacts(train_df, val_df, test_df, split_info, cfg):
    split_info_path = _feedback_output_path("static_split_summary.json")
    split_counts_path = _feedback_output_path("static_split_counts.csv")
    split_sources_path = _feedback_output_path("static_split_sources.csv")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    pd.DataFrame(_static_split_counts(train_df, val_df, test_df, cfg)).to_csv(split_counts_path, index=False)

    source_rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for source, count in split_df["_split_source"].value_counts().sort_index().items():
            source_rows.append({"split": name, "split_source": source, "rows": int(count)})
    pd.DataFrame(source_rows).to_csv(split_sources_path, index=False)

    exports = {
        "split_summary": split_info_path,
        "split_counts": split_counts_path,
        "split_sources": split_sources_path,
    }
    if os.environ.get("USIM_STATIC_EXPORT_SPLIT", "1") == "1":
        train_path = _feedback_output_path("static_train.pkl")
        val_path = _feedback_output_path("static_val.pkl")
        test_path = _feedback_output_path("static_test.pkl")
        assignments_path = _feedback_output_path("static_split_assignments.csv")
        train_df.to_pickle(train_path)
        val_df.to_pickle(val_path)
        test_df.to_pickle(test_path)
        assign_parts = []
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            cols = ["_row_id", "u_idx", "i_idx", "_split_source"]
            part = split_df[cols].copy()
            part["split"] = split_name
            assign_parts.append(part)
        pd.concat(assign_parts, ignore_index=True).to_csv(assignments_path, index=False)
        exports.update(
            {
                "train_split": train_path,
                "val_split": val_path,
                "test_split": test_path,
                "split_assignments": assignments_path,
            }
        )
    return exports


def _file_digest(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df):
    script_path = os.path.abspath(__file__)
    manifest = {
        "protocol": "static_threshold",
        "created_at_unix": time.time(),
        "script": {
            "path": script_path,
            "exists": os.path.exists(script_path),
            "size_bytes": os.path.getsize(script_path) if os.path.exists(script_path) else None,
            "mtime": os.path.getmtime(script_path) if os.path.exists(script_path) else None,
            "sha256": _file_digest(script_path),
        },
        "data": {
            "data_dir": data_dir,
            "rows": int(len(df)),
            "users": int(df["u_idx"].nunique()),
            "items": int(df["i_idx"].nunique()),
        },
        "split": split_info,
        "model_config": {
            "n_epochs": int(cfg.n_epochs),
            "batch_size": int(cfg.batch_size),
            "cold_threshold": int(cfg.cold_threshold),
            "eval_n_neg": int(cfg.eval_n_neg),
            "run_sampled_eval": bool(cfg.run_sampled_eval),
            "use_content_delta": bool(cfg.use_content_delta),
            "content_delta_mode": str(cfg.content_delta_mode),
            "content_delta_paper_style": bool(cfg.content_delta_paper_style),
            "content_delta_replace_item": bool(cfg.content_delta_replace_item),
            "content_delta_cold_only": bool(cfg.content_delta_cold_only),
            "content_delta_train_on_id_dropout": bool(cfg.content_delta_train_on_id_dropout),
            "content_delta_max_norm": float(cfg.content_delta_max_norm),
            "content_delta_scale": float(cfg.content_delta_scale),
            "content_delta_lr_mult": float(cfg.content_delta_lr_mult),
            "content_delta_only_after_epoch": int(cfg.content_delta_only_after_epoch),
            "aux_weight": float(cfg.aux_weight),
            "aux_hot_only": bool(cfg.aux_hot_only),
            "early_stop_score_mode": str(cfg.early_stop_score_mode),
            "early_stop_average_mode": str(cfg.early_stop_average_mode),
            "early_stop_k": int(cfg.early_stop_k),
            "early_stop_patience": int(cfg.early_stop_patience),
            "use_pseudo_cold_train": bool(cfg.use_pseudo_cold_train),
            "pseudo_cold_ratio": float(cfg.pseudo_cold_ratio),
            "pseudo_cold_min_pop": int(cfg.pseudo_cold_min_pop),
            "pseudo_cold_mode": str(cfg.pseudo_cold_mode),
            "use_paac": bool(cfg.use_paac),
            "use_course_rerank": bool(cfg.use_course_rerank),
            "use_course_reward": bool(cfg.use_course_reward),
            "feedback_course_only_cold": bool(cfg.feedback_course_only_cold),
            "feedback_course_prereq_weight": float(cfg.feedback_course_prereq_weight),
            "feedback_course_concept_weight": float(cfg.feedback_course_concept_weight),
            "feedback_course_difficulty_weight": float(cfg.feedback_course_difficulty_weight),
            "feedback_course_redundant_weight": float(cfg.feedback_course_redundant_weight),
            "feedback_course_sample_beta": float(cfg.feedback_course_sample_beta),
            "feedback_course_sample_only_cold": bool(cfg.feedback_course_sample_only_cold),
            "use_prereq_aux_loss": bool(cfg.use_prereq_aux_loss),
            "prereq_aux_weight": float(cfg.prereq_aux_weight),
            "prereq_aux_only_cold": bool(cfg.prereq_aux_only_cold),
            "prereq_graph_source": str(cfg.prereq_graph_source),
            "disable_llm_score": bool(cfg.disable_llm_score),
        },
        "course_stats": course_stats,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("USIM_")},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "torch": torch.__version__,
        },
        "exports": exports,
    }
    manifest_path = _feedback_output_path("static_protocol_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    exports["manifest"] = manifest_path
    return manifest_path


def _make_fast3_optimizer(model, cfg):
    delta_params = model.content_delta_trainable_parameters()
    delta_param_ids = {id(p) for p in delta_params}
    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in delta_param_ids
    ]
    if delta_params:
        return torch.optim.Adam(
            [
                {"params": base_params, "lr": cfg.lr},
                {"params": delta_params, "lr": cfg.lr * float(getattr(cfg, "content_delta_lr_mult", 1.0))},
            ]
        )
    return torch.optim.Adam(base_params, lr=cfg.lr)


def _metric_or_zero(metrics, key):
    return float(metrics.get(key, 0.0)) if metrics else 0.0


def _compute_early_stop_score(cold_metrics, hot_metrics, k, mode="cold_only"):
    """Combine cold/hot validation metrics into a single early-stop score.

    Modes (selected via ``cfg.early_stop_score_mode`` / ``USIM_EARLY_STOP_SCORE_MODE``):

    - ``cold_only`` (legacy default): use Cold N@k only. Behavior unchanged
      from the pre-refactor code.
    - ``geometric``: ``sqrt(cold_n * hot_n)``. Either side collapsing to 0
      drives the score to 0; balanced gains compound.
    - ``harmonic``: ``2 * cold_n * hot_n / (cold_n + hot_n)``; stronger pull
      toward balance than geometric. Returns 0 if either side is non-positive.
    - ``sum``: ``cold_n + hot_n``; simplest joint signal but lets a much
      larger side dominate.

    Both inputs may be ``None`` (e.g., when the matching eval split is empty);
    missing keys are treated as 0 via ``_metric_or_zero``.
    """
    cold = _metric_or_zero(cold_metrics, f"N@{k}")
    hot = _metric_or_zero(hot_metrics, f"N@{k}")
    mode = (mode or "cold_only").strip().lower()
    if mode == "geometric":
        return float((max(0.0, cold) * max(0.0, hot)) ** 0.5)
    if mode == "harmonic":
        if cold <= 0.0 or hot <= 0.0:
            return 0.0
        return float(2.0 * cold * hot / (cold + hot))
    if mode == "sum":
        return float(cold + hot)
    # Default / cold_only: legacy behavior.
    return float(cold)


def run_static_experiment(df, cfg, device, content_emb, llm_scores):
    train_df, val_df, test_df, split_info = _static_split_df(df)
    train_df, val_df, test_df, item_train_pop = _apply_train_popularity(train_df, val_df, test_df, cfg)
    split_info["cold_threshold"] = int(cfg.cold_threshold)
    if split_info.get("split_family") == "strict_item_cold" and int(cfg.cold_threshold) == 1:
        split_info["cold_definition"] = "strict item cold iff train_popularity == 0"
        split_info["true_item_cold_start"] = True
    else:
        split_info["cold_definition"] = "cold iff train_popularity < cold_threshold"
        split_info["true_item_cold_start"] = False

    artifact_source = os.environ.get("USIM_STATIC_ARTIFACT_SOURCE", "").strip().lower()
    if not artifact_source:
        artifact_source = "all_metadata" if cfg.prereq_graph_source == "concept" else "train"
    if artifact_source == "all_metadata" and cfg.prereq_graph_source != "concept":
        print("[STATIC] all_metadata artifacts are only safe for concept prerequisites; falling back to train.")
        artifact_source = "train"
    artifact_df = df if artifact_source == "all_metadata" else train_df
    split_info["artifact_source"] = artifact_source

    test_history_policy = os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower()
    if test_history_policy not in {"train_only", "train_val"}:
        raise ValueError(f"Unsupported USIM_STATIC_TEST_HISTORY={test_history_policy!r}")
    split_info["test_history_policy"] = test_history_policy

    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = build_course_artifacts(
            artifact_df,
            cfg.n_items,
            relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, course_stats = None, _empty_course_stats(cfg.n_items)

    model = Fast3FeedbackUSIM(cfg, content_emb).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_train_pop)
    optimizer = _make_fast3_optimizer(model, cfg)

    exports = _write_static_split_artifacts(train_df, val_df, test_df, split_info, cfg)
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    manifest_path = _write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df)

    print(f">> Architecture: Feedback-Aware RL-USIM + InfoNCE [FAST3 ContentDelta] (Batch Size={cfg.batch_size})")
    print(f">> Device: {device}")
    print(">> Protocol: static")
    print(
        f">> Static Split: mode={split_info['split_mode']} | family={split_info['split_family']} | "
        f"target={split_info['train_ratio']:.2f}/{split_info['val_ratio']:.2f}/{split_info['test_ratio']:.2f} | "
        f"actual={split_info['actual_train_ratio']:.2f}/{split_info['actual_val_ratio']:.2f}/{split_info['actual_test_ratio']:.2f}"
    )
    print(
        f">> Static Users: val_seen={split_info['val_user_seen_ratio']:.2%} | "
        f"test_seen={split_info['test_user_seen_ratio']:.2%} | artifact_source={artifact_source} | "
        f"test_history={test_history_policy}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | "
        f"score_mode={cfg.early_stop_score_mode} | "
        f"avg_mode={cfg.early_stop_average_mode} | k={cfg.early_stop_k} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )
    print(
        f">> Content Delta: enabled={cfg.use_content_delta} | mode={cfg.content_delta_mode} | "
        f"train_on_id_dropout={cfg.content_delta_train_on_id_dropout} | "
        f"paper_style={cfg.content_delta_paper_style} | replace_item={cfg.content_delta_replace_item} | "
        f"cold_only={cfg.content_delta_cold_only} | max_norm={cfg.content_delta_max_norm:.3f} | "
        f"scale={cfg.content_delta_scale:.3f} | aux_w={cfg.aux_weight:.3f} | "
        f"aux_hot_only={cfg.aux_hot_only} | "
        f"lr_mult={cfg.content_delta_lr_mult:.3f} | "
        f"delta_only_after_epoch={cfg.content_delta_only_after_epoch}"
    )
    print(
        f">> Pseudo-Cold Train: enabled={cfg.use_pseudo_cold_train} | "
        f"mode={cfg.pseudo_cold_mode} | ratio={cfg.pseudo_cold_ratio:.2f} | "
        f"min_pop={cfg.pseudo_cold_min_pop}"
    )
    print(
        f">> Eval: sampled={'enabled' if cfg.run_sampled_eval else 'disabled'} "
        f"(1+{cfg.eval_n_neg}) | full_ranking=enabled"
    )
    print(f">> Static trace manifest: {manifest_path}")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {
        "Epoch": [],
        "Loss": [],
        "Val_full_cold_R@10": [],
        "Val_full_hot_R@10": [],
        "Val_full_cold_N@10": [],
        "Val_full_hot_N@10": [],
    }
    static_diag_keys = [
        "MainLoss",
        "AuxLoss",
        "PPOLoss",
        "PrereqAuxLoss",
        "DeltaRegLoss",
        "CourseSampleFit",
        "CoursePrereqGap",
        "CourseConceptBonus",
        "CourseDifficultyGap",
        "CourseRedundant",
        "PAACContrastLoss",
        "PAACAlignLoss",
        "PAACAlignPairs",
        "FNMaskRatio",
        "PseudoColdRatio",
        "EffectiveColdRatio",
        "PseudoColdCount",
    ]
    for key in static_diag_keys:
        history[key] = []

    def _append_static_history(epoch_num, avg_loss, val_cold_metrics, val_hot_metrics, diag):
        history["Epoch"].append(epoch_num)
        history["Loss"].append(avg_loss)
        history["Val_full_cold_R@10"].append(_metric_or_zero(val_cold_metrics, "R@10"))
        history["Val_full_hot_R@10"].append(_metric_or_zero(val_hot_metrics, "R@10"))
        history["Val_full_cold_N@10"].append(_metric_or_zero(val_cold_metrics, "N@10"))
        history["Val_full_hot_N@10"].append(_metric_or_zero(val_hot_metrics, "N@10"))
        for key in static_diag_keys:
            history[key].append(diag.get(key, 0.0))

    train_seen = _add_user_seen_from_df({}, train_df)
    # Build the (n_users, n_items) bool index ONCE on device for fast batch lookup.
    # Replaces per-batch Python loops in _build_seen_mat / evaluate full-ranking seen-mask.
    model.set_user_seen_index(train_seen)
    train_loader = DataLoader(StreamDataset(train_df, llm_scores), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(StreamDataset(val_df, llm_scores), batch_size=2048, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(StreamDataset(test_df, llm_scores), batch_size=2048, shuffle=False, collate_fn=collate_fn)

    best_state = None
    best_opt_state = None
    best_epoch = 0
    best_score = -1e9
    no_improve = 0
    do_early_stop = cfg.use_epoch_early_stop and cfg.n_epochs > 1 and len(val_df) > 0
    delta_only_applied = False

    print(
        f"\n>>> Start STATIC train/eval | target_split={split_info['train_ratio']:.2f}/"
        f"{split_info['val_ratio']:.2f}/{split_info['test_ratio']:.2f} | "
        f"actual_split={split_info['actual_train_ratio']:.2f}/"
        f"{split_info['actual_val_ratio']:.2f}/{split_info['actual_test_ratio']:.2f} | "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    for epoch in range(cfg.n_epochs):
        delta_only_epoch = int(getattr(cfg, "content_delta_only_after_epoch", 0))
        if (
            delta_only_epoch > 0
            and not delta_only_applied
            and (epoch + 1) >= delta_only_epoch
        ):
            delta_params = model.enable_delta_only_training_()
            optimizer = torch.optim.Adam(
                delta_params,
                lr=cfg.lr * float(getattr(cfg, "content_delta_lr_mult", 1.0)),
            )
            best_opt_state = None
            delta_only_applied = True
            print(
                f"  [STATIC-DELTA-ONLY] Freeze base parameters at epoch {epoch + 1}; "
                f"trainable_delta_params={sum(p.numel() for p in delta_params)}"
            )
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        steps = 0
        main_loss_sum = 0.0
        aux_loss_sum = 0.0
        ppo_loss_sum = 0.0
        prereq_aux_loss_sum = 0.0
        delta_reg_loss_sum = 0.0
        course_sample_fit_sum = 0.0
        course_prereq_sum = 0.0
        course_concept_sum = 0.0
        course_diff_sum = 0.0
        course_redundant_sum = 0.0
        paac_contrast_sum = 0.0
        paac_align_sum = 0.0
        paac_align_pairs_sum = 0
        fn_mask_ratio_sum = 0.0
        pseudo_cold_ratio_sum = 0.0
        effective_cold_ratio_sum = 0.0
        pseudo_cold_count_sum = 0
        pseudo_info_batches = 0
        num_batches = len(train_loader)
        cached_user_bank = model._build_user_bank_raw() if cfg.candidate_strategy == "retrieve_sample" else None
        print(f"  [STATIC-TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | samples={len(train_df)} | batches={num_batches}")
        last_progress_log = epoch_start
        optimizer.zero_grad()

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
                user_seen_items=train_seen,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.clip_content_delta_()
            optimizer.zero_grad()
            total_loss += float(loss.item())
            steps += 1
            if cand_info:
                main_loss_sum += float(cand_info.get("main_loss", 0.0))
                aux_loss_sum += float(cand_info.get("aux_loss", 0.0))
                ppo_loss_sum += float(cand_info.get("ppo_loss", 0.0))
                prereq_aux_loss_sum += float(cand_info.get("prereq_aux_loss", 0.0))
                delta_reg_loss_sum += float(cand_info.get("delta_reg_loss", 0.0))
                course_sample_fit_sum += float(cand_info.get("course_sample_fit", 0.0))
                course_prereq_sum += float(cand_info.get("course_prereq_gap", 0.0))
                course_concept_sum += float(cand_info.get("course_concept_bonus", 0.0))
                course_diff_sum += float(cand_info.get("course_difficulty_gap", 0.0))
                course_redundant_sum += float(cand_info.get("course_redundant", 0.0))
                paac_contrast_sum += float(cand_info.get("paac_contrast_loss", 0.0))
                paac_align_sum += float(cand_info.get("paac_align_loss", 0.0))
                paac_align_pairs_sum += int(cand_info.get("paac_align_pairs", 0))
                fn_mask_ratio_sum += float(cand_info.get("fn_mask_ratio", 0.0))
                pseudo_cold_ratio_sum += float(cand_info.get("pseudo_cold_ratio", 0.0))
                effective_cold_ratio_sum += float(cand_info.get("effective_cold_ratio", 0.0))
                pseudo_cold_count_sum += int(cand_info.get("pseudo_cold_count", 0))
                pseudo_info_batches += 1

            now_ts = time.time()
            if _should_log_train_progress(batch_idx, num_batches, cfg, last_progress_log, now_ts):
                done = batch_idx + 1
                elapsed = now_ts - epoch_start
                eta = elapsed / max(1, done) * max(0, num_batches - done)
                pct = 100.0 * done / max(1, num_batches)
                print(
                    f"    [STATIC-TRAIN-PROGRESS] {done}/{num_batches} ({pct:.0f}%) | "
                    f"avg_loss={total_loss / max(1, steps):.4f} | "
                    f"elapsed={_format_eta(elapsed)} | eta={_format_eta(eta)}"
                )
                last_progress_log = now_ts

        avg_loss = total_loss / max(1, steps)
        delta_stats = model.content_delta_stats()
        delta_suffix = ""
        if delta_stats is not None:
            delta_suffix = (
                f" | DeltaNorm[mean={delta_stats['mean_norm']:.4f}, max={delta_stats['max_norm']:.4f}, "
                f"eff_mean={delta_stats['effective_mean_norm']:.4f}, eff_max={delta_stats['effective_max_norm']:.4f}, "
                f"clip={delta_stats['clipped_ratio']:.2%}]"
            )
        pseudo_suffix = ""
        epoch_diag = {key: 0.0 for key in static_diag_keys}
        if pseudo_info_batches > 0:
            epoch_diag.update({
                "MainLoss": main_loss_sum / pseudo_info_batches,
                "AuxLoss": aux_loss_sum / pseudo_info_batches,
                "PPOLoss": ppo_loss_sum / pseudo_info_batches,
                "PrereqAuxLoss": prereq_aux_loss_sum / pseudo_info_batches,
                "DeltaRegLoss": delta_reg_loss_sum / pseudo_info_batches,
                "CourseSampleFit": course_sample_fit_sum / pseudo_info_batches,
                "CoursePrereqGap": course_prereq_sum / pseudo_info_batches,
                "CourseConceptBonus": course_concept_sum / pseudo_info_batches,
                "CourseDifficultyGap": course_diff_sum / pseudo_info_batches,
                "CourseRedundant": course_redundant_sum / pseudo_info_batches,
                "PAACContrastLoss": paac_contrast_sum / pseudo_info_batches,
                "PAACAlignLoss": paac_align_sum / pseudo_info_batches,
                "PAACAlignPairs": paac_align_pairs_sum / pseudo_info_batches,
                "FNMaskRatio": fn_mask_ratio_sum / pseudo_info_batches,
                "PseudoColdRatio": pseudo_cold_ratio_sum / pseudo_info_batches,
                "EffectiveColdRatio": effective_cold_ratio_sum / pseudo_info_batches,
                "PseudoColdCount": pseudo_cold_count_sum,
            })
            pseudo_suffix = (
                f" | PseudoCold[count={pseudo_cold_count_sum}, "
                f"ratio={epoch_diag['PseudoColdRatio']:.2%}, "
                f"effective={epoch_diag['EffectiveColdRatio']:.2%}]"
                f" | LossParts[main={epoch_diag['MainLoss']:.4f}, "
                f"aux={epoch_diag['AuxLoss']:.4f}, ppo={epoch_diag['PPOLoss']:.4f}, "
                f"prereq={epoch_diag['PrereqAuxLoss']:.4f}, "
                f"delta={epoch_diag['DeltaRegLoss']:.4f}]"
                f" | Course[fit={epoch_diag['CourseSampleFit']:.4f}, "
                f"p={epoch_diag['CoursePrereqGap']:.4f}, "
                f"c={epoch_diag['CourseConceptBonus']:.4f}, "
                f"d={epoch_diag['CourseDifficultyGap']:.4f}, "
                f"r={epoch_diag['CourseRedundant']:.4f}]"
            )
        print(
            f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s{pseudo_suffix}{delta_suffix}"
        )

        val_cold_metrics, val_cold_count = None, 0
        val_hot_metrics, val_hot_count = None, 0
        if do_early_stop:
            print("  [STATIC-EARLYSTOP-EVAL] Run validation full ranking...")
            val_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            val_cold_metrics, val_cold_count = evaluate_usim(
                model, val_loader, device, llm_scores, k_list=k_list,
                n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=val_item_vecs,
                average_mode=cfg.early_stop_average_mode,
            )
            val_hot_metrics, val_hot_count = evaluate_usim(
                model, val_loader, device, llm_scores, k_list=k_list,
                n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=val_item_vecs,
                average_mode=cfg.early_stop_average_mode,
            )
            cur_score = _compute_early_stop_score(
                val_cold_metrics,
                val_hot_metrics,
                cfg.early_stop_k,
                mode=cfg.early_stop_score_mode,
            )
            cold_n_now = _metric_or_zero(val_cold_metrics, f"N@{cfg.early_stop_k}")
            hot_n_now = _metric_or_zero(val_hot_metrics, f"N@{cfg.early_stop_k}")
            cold_r_now = _metric_or_zero(val_cold_metrics, f"R@{cfg.early_stop_k}")
            hot_r_now = _metric_or_zero(val_hot_metrics, f"R@{cfg.early_stop_k}")
            print(
                f"  [STATIC-EARLYSTOP] Epoch {epoch + 1}: "
                f"Cold N@{cfg.early_stop_k}={cold_n_now:.4f}, "
                f"Hot N@{cfg.early_stop_k}={hot_n_now:.4f}, "
                f"Cold R@{cfg.early_stop_k}={cold_r_now:.4f}, "
                f"Hot R@{cfg.early_stop_k}={hot_r_now:.4f} | "
                f"score[{cfg.early_stop_score_mode}]={cur_score:.4f}"
            )
            if cur_score > best_score + cfg.early_stop_min_delta:
                best_score = cur_score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                best_opt_state = copy.deepcopy(optimizer.state_dict())
                no_improve = 0
                print("  [STATIC-EARLYSTOP] update")
            else:
                no_improve += 1
                print(f"  [STATIC-EARLYSTOP] no_improve={no_improve}/{cfg.early_stop_patience}")
                if no_improve >= cfg.early_stop_patience:
                    print(f"  [STATIC-EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    _append_static_history(epoch + 1, avg_loss, val_cold_metrics, val_hot_metrics, epoch_diag)
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch + 1

        _append_static_history(epoch + 1, avg_loss, val_cold_metrics, val_hot_metrics, epoch_diag)

    if best_state is not None:
        model.load_state_dict(best_state)
        if best_opt_state is not None:
            try:
                optimizer.load_state_dict(best_opt_state)
                _optimizer_state_to_device(optimizer, device)
            except ValueError as exc:
                print(f"  [STATIC-EARLYSTOP] Skip optimizer restore due to parameter-group mismatch: {exc}")
        if best_score <= -1e8:
            print(f"  [STATIC-FINAL] Use final epoch={best_epoch} (validation not run)")
        else:
            print(
                f"  [STATIC-EARLYSTOP] Restore best epoch={best_epoch} "
                f"(score[{cfg.early_stop_score_mode}]={best_score:.4f})"
            )

    print("  [STATIC-TEST] Build eval item bank and run test ranking...")
    test_seen = _clone_user_seen(train_seen)
    if test_history_policy == "train_val":
        _add_user_seen_from_df(test_seen, val_df)
    # Refresh the seen-index with the test history (train_only or train+val) before final eval.
    model.set_user_seen_index(test_seen)
    test_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)

    if cfg.run_sampled_eval:
        sampled_cold, sampled_cold_count = evaluate_usim(
            model, test_loader, device, llm_scores, k_list=k_list,
            n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
            user_seen_items=test_seen, all_item_vecs=test_item_vecs,
        )
        sampled_hot, sampled_hot_count = evaluate_usim(
            model, test_loader, device, llm_scores, k_list=k_list,
            n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
            user_seen_items=test_seen, all_item_vecs=test_item_vecs,
        )
    else:
        sampled_cold, sampled_hot = None, None
        sampled_cold_count, sampled_hot_count = 0, 0

    full_cold, full_cold_count = evaluate_usim(
        model, test_loader, device, llm_scores, k_list=k_list,
        n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=test_item_vecs,
    )
    full_hot, full_hot_count = evaluate_usim(
        model, test_loader, device, llm_scores, k_list=k_list,
        n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=test_item_vecs,
    )
    full_cold_item_macro, full_cold_item_macro_count = evaluate_usim(
        model, test_loader, device, llm_scores, k_list=k_list,
        n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=test_item_vecs,
        average_mode="item_macro",
    )
    full_hot_item_macro, full_hot_item_macro_count = evaluate_usim(
        model, test_loader, device, llm_scores, k_list=k_list,
        n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=test_item_vecs,
        average_mode="item_macro",
    )
    sampled_cold = sampled_cold or {key: 0.0 for key in metrics_keys}
    sampled_hot = sampled_hot or {key: 0.0 for key in metrics_keys}
    full_cold = full_cold or {key: 0.0 for key in metrics_keys}
    full_hot = full_hot or {key: 0.0 for key in metrics_keys}
    full_cold_item_macro = full_cold_item_macro or {key: 0.0 for key in metrics_keys}
    full_hot_item_macro = full_hot_item_macro or {key: 0.0 for key in metrics_keys}

    print("\n" + "=" * 90)
    print("         FINAL REPORT (STATIC MAIN): item-macro full ranking (RL-USIM FAST3 ContentDelta)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Cold ItemMacro':<14} | {'Hot ItemMacro':<14}")
    print("-" * 90)
    for key in metrics_keys:
        print(
            f"{key:<10} | "
            f"{full_cold_item_macro.get(key, 0.0):<14.4f} | "
            f"{full_hot_item_macro.get(key, 0.0):<14.4f}"
        )
    print("-" * 90)
    print(f"Item-Macro Counts: ColdItems={full_cold_item_macro_count}, HotItems={full_hot_item_macro_count}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("         FINAL REPORT (STATIC SUPP): interaction-weighted ranking")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc_text = f"{sampled_cold.get(key, 0.0):.4f}" if sampled_cold_count > 0 else "NA"
        sh_text = f"{sampled_hot.get(key, 0.0):.4f}" if sampled_hot_count > 0 else "NA"
        print(
            f"{key:<10} | {sc_text:<12} | {sh_text:<12} | "
            f"{full_cold.get(key, 0.0):<12.4f} | {full_hot.get(key, 0.0):<12.4f}"
        )
    print("-" * 90)
    print(f"Sampled Samples: Cold={sampled_cold_count}, Hot={sampled_hot_count}")
    print(f"Full Interaction Samples: Cold={full_cold_count}, Hot={full_hot_count}")
    print("=" * 90)

    detail_path, fullrank_path = _save_final_report_exports(
        protocol="static",
        metrics_keys=metrics_keys,
        sampled_cold=sampled_cold,
        sampled_hot=sampled_hot,
        full_cold=full_cold,
        full_hot=full_hot,
        sampled_cold_count=sampled_cold_count,
        sampled_hot_count=sampled_hot_count,
        full_cold_count=full_cold_count,
        full_hot_count=full_hot_count,
        model_name="USIM-Feedback-FAST3-ContentDelta",
        full_cold_item_macro=full_cold_item_macro,
        full_hot_item_macro=full_hot_item_macro,
        full_cold_item_macro_count=full_cold_item_macro_count,
        full_hot_item_macro_count=full_hot_item_macro_count,
    )

    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_content_delta_static.csv")
    summary_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    summary_rows = []
    for eval_name, cold_metrics, hot_metrics, cold_count, hot_count in [
        ("sampled", sampled_cold, sampled_hot, sampled_cold_count, sampled_hot_count),
        ("full_rank", full_cold, full_hot, full_cold_count, full_hot_count),
    ]:
        row = {
            "Model": "USIM-Feedback-FAST3-ContentDelta",
            "Eval": eval_name,
            "Protocol": "static",
            "ColdSamples": int(cold_count),
            "HotSamples": int(hot_count),
        }
        for key in metrics_keys:
            row[f"Cold_{key}"] = float(cold_metrics.get(key, 0.0)) if cold_count > 0 else None
            row[f"Hot_{key}"] = float(hot_metrics.get(key, 0.0)) if hot_count > 0 else None
        summary_rows.append(row)
    macro_row = {
        "Model": "USIM-Feedback-FAST3-ContentDelta",
        "Eval": "full_rank_item_macro",
        "Protocol": "static",
        "ColdSamples": int(full_cold_item_macro_count),
        "HotSamples": int(full_hot_item_macro_count),
    }
    for key in metrics_keys:
        macro_row[f"Cold_{key}"] = (
            float(full_cold_item_macro.get(key, 0.0)) if full_cold_item_macro_count > 0 else None
        )
        macro_row[f"Hot_{key}"] = (
            float(full_hot_item_macro.get(key, 0.0)) if full_hot_item_macro_count > 0 else None
        )
    summary_rows.append(macro_row)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    exports.update(
        {
            "static_epoch_metrics": metrics_path,
            "static_summary": summary_path,
            "final_detail": detail_path,
            "final_fullrank": fullrank_path,
        }
    )
    _write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df)
    print(f">> Saved {summary_path}, {detail_path}, and {fullrank_path}")


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Feedback-Aware USIM (FAST3 ContentDelta) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    llm_scores, llm_score_path, _ = load_llm_scores_for_stream(
        data_dir,
        df,
        cold_threshold=int(os.environ.get("USIM_COLD_THRESHOLD", "5")),
        n_users=meta.get("n_users"),
        n_items=meta.get("n_items"),
        fallback_data_dirs=["processed_data"],
    )
    content_emb = torch.load(f"{data_dir}/content_emb.pt")
    if llm_score_path:
        print(f"   LLM scores loaded from {llm_score_path}")

    cfg = Fast3Config(meta["n_users"], meta["n_items"], content_emb.shape[1])
    llm_scores, llm_summary = prepare_llm_scores(llm_scores, cfg)
    cfg.llm_bank_mode = llm_summary["mode"]
    print(
        ">> LLMScore: "
        f"mode={llm_summary['mode']} | "
        f"weight={cfg.llm_weight:.2f} | "
        f"safe={cfg.llm_safe_mode} | "
        f"cold_only={cfg.llm_cold_only} | "
        f"hot_only={cfg.llm_hot_only} | "
        f"raw={llm_summary['raw_total']} "
        f"(pair={llm_summary['raw_pair']}, item={llm_summary['raw_item']}) | "
        f"effective={llm_summary['effective_total']} "
        f"(pair={llm_summary['effective_pair']}, item={llm_summary['effective_item']})"
    )
    device = _resolve_torch_device()
    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment(df, cfg, device, content_emb, llm_scores)
        return

    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = build_course_artifacts(
            df,
            cfg.n_items,
            relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
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
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    delta_params = model.content_delta_trainable_parameters()
    delta_param_ids = {id(p) for p in delta_params}
    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in delta_param_ids
    ]
    if delta_params:
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": cfg.lr},
                {"params": delta_params, "lr": cfg.lr * float(getattr(cfg, "content_delta_lr_mult", 1.0))},
            ]
        )
    else:
        optimizer = torch.optim.Adam(base_params, lr=cfg.lr)

    print(f">> Architecture: Feedback-Aware RL-USIM + InfoNCE [FAST3 ContentDelta] (Batch Size={cfg.batch_size})")
    print(f">> Device: {device}")
    print(
        f">> Window={cfg.stream_train_window} | PPO epochs={cfg.ppo_epochs} | "
        f"lambda={cfg.ppo_lambda:.2f} | value_clip={cfg.ppo_value_clip:.2f} | "
        f"adv_norm={cfg.ppo_adv_norm}"
    )
    print(f">> Train Protocol: legacy={cfg.legacy_train_protocol}")
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
        f"cold_only={cfg.llm_cold_only} | hot_only={cfg.llm_hot_only} | bank_mode={cfg.llm_bank_mode}"
    )
    print(
        f">> Content Delta: enabled={cfg.use_content_delta} | "
        f"mode={cfg.content_delta_mode} | train_on_id_dropout={cfg.content_delta_train_on_id_dropout} | "
        f"paper_style={cfg.content_delta_paper_style} | replace_item={cfg.content_delta_replace_item} | "
        f"max_norm={cfg.content_delta_max_norm:.3f} | cold_only={cfg.content_delta_cold_only} | "
        f"scale={cfg.content_delta_scale:.3f} | aux={cfg.content_delta_aux_mode} | aux_w={cfg.aux_weight:.3f} | "
        f"lr_mult={cfg.content_delta_lr_mult:.3f} | "
        f"reg(l2={cfg.content_delta_l2_weight:.3f}, cap={cfg.content_delta_cap_weight:.3f}) | "
        f"eval_bank={cfg.content_delta_eval_bank_mode} | "
        f"norm_base={cfg.content_delta_normalize_base} | norm_output={cfg.content_delta_normalize_output}"
    )
    print(
        f">> Pseudo-Cold Train: enabled={cfg.use_pseudo_cold_train} | "
        f"mode={cfg.pseudo_cold_mode} | ratio={cfg.pseudo_cold_ratio:.2f} | "
        f"min_pop={cfg.pseudo_cold_min_pop}"
    )
    print(
        f">> PAAC: enabled={cfg.use_paac} | "
        f"align_w={cfg.paac_align_weight:.3f} | contrast_w={cfg.paac_contrast_weight:.3f} | "
        f"beta={cfg.paac_contrast_beta:.2f} | gamma={cfg.paac_contrast_gamma:.2f} | "
        f"batch_pop_ratio={cfg.paac_batch_pop_ratio:.2f} | max_pairs={cfg.paac_align_max_pairs} | "
        f"detach_hot={cfg.paac_align_detach_hot} | group_mode={cfg.paac_group_mode}"
    )
    print(f">> False Negative Mask: enabled={cfg.mask_known_pos_neg}")
    print(
        f">> Course Soft Rerank: enabled={cfg.feedback_course_sample_soft} | "
        f"beta={cfg.feedback_course_sample_beta:.2f} | topL={cfg.feedback_course_sample_top_l}"
    )
    print(
        f">> Course Feedback: redundant_mode={cfg.feedback_course_redundant_mode} | "
        f"video_min={cfg.feedback_course_struct_video_min:.2f} | "
        f"concept_min={cfg.feedback_course_concept_min:.2f} | "
        f"redundant_thr={cfg.feedback_course_redundant_thr:.2f} | "
        f"redundant_concept_gate={cfg.feedback_course_redundant_concept_gate:.2f}"
    )
    print(
        f">> Course Artifacts: enabled={cfg.feedback_load_course_artifacts} | "
        f"course_reward={cfg.use_course_reward} | "
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
    print(
        f">> Eval: sampled={'enabled' if cfg.run_sampled_eval else 'disabled'} "
        f"(1+{cfg.eval_n_neg}) | full_ranking=enabled"
    )

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {
        "Period": [],
        "Count_cold": [],
        "Count_hot": [],
        "Full_count_cold": [],
        "Full_count_hot": [],
    }
    for prefix in ["cold_", "hot_", "full_cold_", "full_hot_"]:
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
                    try:
                        optimizer.load_state_dict(resume_state["optimizer_state"])
                        _optimizer_state_to_device(optimizer, device)
                    except ValueError as exc:
                        print(f">> Resume: skip optimizer state due to parameter-group mismatch: {exc}")
                history = copy.deepcopy(resume_state.get("history", history))
                history_len = len(history.get("Period", []))
                history.setdefault("Full_count_cold", [0] * history_len)
                history.setdefault("Full_count_hot", [0] * history_len)
                for prefix in ["cold_", "hot_", "full_cold_", "full_hot_"]:
                    for key in metrics_keys:
                        history.setdefault(prefix + key, [0.0] * history_len)
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
        fmet_cold, fmet_hot = None, None
        fn_c, fn_h = 0, 0
        resume_this_period = resume_current_period is not None and t == resume_current_period

        if resume_this_period:
            print(f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs}")
        elif t >= warmup_periods:
            eval_mode = "sampled/full ranking" if cfg.run_sampled_eval else "full ranking only"
            print(f"  [EVAL-START] Build eval item bank and run {eval_mode}...")
            all_item_vecs_eval = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, met_hot = None, None
            if cfg.run_sampled_eval:
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
            sampled_msg = f"Sampled Cold={c_s:.4f} Hot={h_s:.4f}" if cfg.run_sampled_eval else "Sampled=NA"
            print(f"  {sampled_msg} | Full Cold={c_f:.4f} Hot={h_f:.4f}")
            del all_item_vecs_eval
            _maybe_clear_cuda_cache()
        else:
            print("  [WARMUP] Training only...")

        if not resume_this_period:
            history["Period"].append(t)
            history["Count_cold"].append(n_cold_t)
            history["Count_hot"].append(n_hot_t)
            history["Full_count_cold"].append(fn_c)
            history["Full_count_hot"].append(fn_h)
            for key in metrics_keys:
                history["cold_" + key].append(cold_res.get(key, 0.0))
                history["hot_" + key].append(hot_res.get(key, 0.0))
                history["full_cold_" + key].append(fmet_cold.get(key, 0.0) if fmet_cold else 0.0)
                history["full_hot_" + key].append(fmet_hot.get(key, 0.0) if fmet_hot else 0.0)
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
            if not getattr(cfg, "legacy_train_protocol", False):
                model.train()
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
            paac_contrast_sum = 0.0
            paac_align_sum = 0.0
            paac_align_pairs_sum = 0
            fn_mask_ratio_sum = 0.0
            pseudo_cold_ratio_sum = 0.0
            effective_cold_ratio_sum = 0.0
            pseudo_cold_count_sum = 0
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
                model.clip_content_delta_()
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
                    paac_contrast_sum += cand_info.get("paac_contrast_loss", 0.0)
                    paac_align_sum += cand_info.get("paac_align_loss", 0.0)
                    paac_align_pairs_sum += int(cand_info.get("paac_align_pairs", 0))
                    fn_mask_ratio_sum += cand_info.get("fn_mask_ratio", 0.0)
                    pseudo_cold_ratio_sum += cand_info.get("pseudo_cold_ratio", 0.0)
                    effective_cold_ratio_sum += cand_info.get("effective_cold_ratio", 0.0)
                    pseudo_cold_count_sum += int(cand_info.get("pseudo_cold_count", 0))
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
            delta_stats = model.content_delta_stats()
            delta_suffix = ""
            if delta_stats is not None:
                delta_suffix = (
                    f" | DeltaNorm[mean={delta_stats['mean_norm']:.4f}, "
                    f"max={delta_stats['max_norm']:.4f}, "
                    f"eff_mean={delta_stats['effective_mean_norm']:.4f}, "
                    f"eff_max={delta_stats['effective_max_norm']:.4f}, "
                    f"clip={delta_stats['clipped_ratio']:.2%}]"
                )
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
                avg_paac_cl = paac_contrast_sum / cand_batches
                avg_paac_align = paac_align_sum / cand_batches
                avg_paac_pairs = paac_align_pairs_sum / cand_batches
                avg_fn_mask = fn_mask_ratio_sum / cand_batches
                avg_pseudo = pseudo_cold_ratio_sum / cand_batches
                avg_eff_cold = effective_cold_ratio_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"MixAlpha: {avg_mix:.4f} | SampleFit: {avg_csf:.4f} | "
                    f"Course[p={avg_cp:.4f}, c={avg_cc:.4f}, d={avg_cd:.4f}, r={avg_cr:.4f}]"
                    f" | PAAC[cl={avg_paac_cl:.4f}, align={avg_paac_align:.4f}, pairs={avg_paac_pairs:.1f}]"
                    f" | FNMask={avg_fn_mask:.2%}"
                    f" | PseudoCold[count={pseudo_cold_count_sum}, ratio={avg_pseudo:.2%}, effective={avg_eff_cold:.2%}]"
                    f"{delta_suffix}"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | train={len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                    f"{delta_suffix}"
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

            if do_early_stop and es_val_loader is not None:
                print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot on previous period (validation)...")
                all_item_vecs_es = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim(
                    model,
                    es_val_loader,
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
                    es_val_loader,
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
                try:
                    optimizer.load_state_dict(es_best_opt_state)
                    _optimizer_state_to_device(optimizer, device)
                except ValueError as exc:
                    print(f"  [EARLYSTOP] Skip optimizer restore due to parameter-group mismatch: {exc}")
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
    sampled_title = f"sampled (1+{cfg.eval_n_neg})" if cfg.run_sampled_eval else "sampled disabled"
    print(f"         FINAL REPORT: {sampled_title} vs full ranking (RL-USIM FAST3 ContentDelta)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    summary_rows = []
    sampled_row = {"Model": "USIM-Feedback-FAST3-ContentDelta", "Eval": "sampled", "ColdSamples": count_cold, "HotSamples": count_hot}
    full_row = {"Model": "USIM-Feedback-FAST3-ContentDelta", "Eval": "full_rank", "ColdSamples": fc_cold, "HotSamples": fc_hot}
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else None
        sh = accum_hot[key] / count_hot if count_hot > 0 else None
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        sc_text = f"{sc:.4f}" if sc is not None else "NA"
        sh_text = f"{sh:.4f}" if sh is not None else "NA"
        print(f"{key:<10} | {sc_text:<12} | {sh_text:<12} | {fc:<12.4f} | {fh:<12.4f}")
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
        model_name="USIM-Feedback-FAST3-ContentDelta",
    )

    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_content_delta.csv")
    summary_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_content_delta_summary.csv")
    plot_path = _feedback_output_path("mooc_result_usim_feedback_fast3_content_delta.png")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    plt.figure(figsize=(12, 6))
    plot_prefix = "" if cfg.run_sampled_eval else "Full "
    plot_cold_key = "cold_R@10" if cfg.run_sampled_eval else "full_cold_R@10"
    plot_hot_key = "hot_R@10" if cfg.run_sampled_eval else "full_hot_R@10"
    plt.plot(history["Period"], history[plot_cold_key], marker="o", label=f"{plot_prefix}Cold R@10")
    plt.plot(history["Period"], history[plot_hot_key], marker="s", label=f"{plot_prefix}Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("RL-USIM [FAST3 ContentDelta]: adaptive mix + stable PPO + soft rerank")
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
    setup_seed(int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025"))))
    main()
