"""
usim_feedback_fast3_content_delta.py - standalone FAST3 variant with bounded content delta

This file no longer depends on usim.py, usim_feedback.py, or
usim_feedback_fast.py. The model and training flow remain here; config,
evaluation, static protocol, course-graph, checkpoint, and reporting helpers
live in the fast3_delta package.
"""
import copy
import hashlib
import json
import math
import os
import platform
import random
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from fast3_delta.checkpoint import (
    _build_feedback_ckpt_state,
    _deserialize_user_seen_items,
    _feedback_ckpt_auto_resume,
    _feedback_ckpt_dir,
    _feedback_ckpt_enabled,
    _feedback_ckpt_force_fresh,
    _feedback_ckpt_save_optimizer_state,
    _feedback_ckpt_snapshot_epochs,
    _latest_feedback_ckpt_path,
    _load_feedback_checkpoint,
    _maybe_clear_cuda_cache,
    _move_state_to_cpu,
    _optimizer_state_to_device,
    _save_feedback_checkpoint,
    _serialize_user_seen_items,
)
from fast3_delta.config import BaseConfig, Fast3Config, FeedbackConfig
from fast3_delta.course_artifacts import (
    _build_behavior_prereq_candidates,
    _build_concept_prereq_candidates,
    _build_hybrid_prereq_candidates,
    _build_item_concept_overlap,
    _empty_course_stats,
    _extract_course_unit_ids,
    _iter_entity_objects,
    _normalize_course_family_key,
    _parse_subject_from_course_id,
    _read_relation_pairs,
    build_course_artifacts,
)
from fast3_delta.eval import (
    _build_eval_pos_item_vecs,
    _build_llm_score_tensor,
    _count_llm_key_types,
    _is_item_llm_key,
    _is_pair_llm_key,
    _item_mean_llm_scores,
    _lookup_llm_score,
    _resolve_eval_force_cold,
    _select_eval_item_bank,
    build_all_item_vecs,
    build_eval_item_vecs,
    compute_ranking_metric_values,
    compute_ranking_metrics,
    evaluate_usim,
    prepare_llm_scores,
)
from fast3_delta.reports import (
    _feedback_output_dir,
    _feedback_output_path,
    _save_final_report_exports,
)
from fast3_delta.sg_urinit import apply_sg_urinit_
from fast3_delta.static_protocol import (
    StreamDataset,
    _add_user_seen_from_df,
    _apply_train_popularity,
    _clone_user_seen,
    _static_seed,
    _static_split_df,
    collate_fn,
    write_static_split_artifacts as _write_static_split_artifacts_impl,
)


def _apply_refinement_only_to_effective_cold(z_i_base, refined_h, effective_cold):
    """Route iterative refinement only through rows trained as true/pseudo cold."""
    if effective_cold is None:
        return refined_h
    cold_mask = effective_cold.to(device=z_i_base.device, dtype=torch.bool).view(-1, 1)
    return torch.where(cold_mask, refined_h, z_i_base)


def _batch_invariant_alignment_grad(
    current_h,
    selected_user,
    target_emb=None,
    target_alpha=None,
    reference_batch_size=1,
):
    """Compute per-row alignment gradients with a fixed, batch-independent scale."""
    h_detached = current_h.detach().requires_grad_(True)
    user_align = (h_detached * selected_user.detach()).sum(dim=1, keepdim=True)
    if target_emb is None:
        score_per_row = user_align
    else:
        if target_alpha is None:
            raise ValueError("target_alpha is required when target_emb is provided")
        target_align = (h_detached * target_emb.detach()).sum(dim=1, keepdim=True)
        score_per_row = (1.0 - target_alpha) * user_align + target_alpha * target_align
    scale = float(max(1, int(reference_batch_size)))
    return torch.autograd.grad(score_per_row.sum() / scale, h_detached)[0]


def _training_episode_target(z_i_base, rollout_policy, ppo_loss_weight):
    """Match Course-fit/no-PPO training transitions to unanchored inference."""
    policy = str(rollout_policy or "ppo").strip().lower()
    if policy == "course_fit" and float(ppo_loss_weight) == 0.0:
        return None
    return z_i_base.detach().clone()


def _original_usim_v2_enabled():
    """Return whether the isolated content-to-behaviour repair route is active."""
    return os.environ.get("USIM_ORIGINAL_V2", "0") == "1"


def _original_usim_v2_step_size():
    step_size = float(os.environ.get("USIM_ORIGINAL_V2_STEP_SIZE", "0.05"))
    if step_size <= 0.0:
        raise ValueError("USIM_ORIGINAL_V2_STEP_SIZE must be positive")
    return step_size


def _build_fixed_tail_pseudo_item_mask(item_popularity, ratio=0.30, min_pop=1):
    """Select a deterministic tail-item set covering the requested popularity mass."""
    pop = torch.as_tensor(item_popularity).detach().float().view(-1)
    mask = torch.zeros(pop.numel(), dtype=torch.bool, device=pop.device)
    eligible_idx = torch.nonzero(pop >= float(max(1, int(min_pop))), as_tuple=False).view(-1)
    if eligible_idx.numel() < 1 or float(ratio) <= 0.0:
        return mask
    if float(ratio) >= 1.0:
        mask[eligible_idx] = True
        return mask

    rows = [
        (float(pop[idx].item()), int(idx))
        for idx in eligible_idx.detach().cpu().tolist()
    ]
    rows.sort(key=lambda pair: (pair[0], pair[1]))
    target_mass = float(ratio) * sum(value for value, _ in rows)
    cumulative = 0.0
    selected = []
    for value, idx in rows:
        selected.append(idx)
        cumulative += value
        if cumulative >= target_mass:
            break
    if selected:
        mask[torch.tensor(selected, dtype=torch.long, device=pop.device)] = True
    return mask


def _remove_target_from_seen_history(seen_mat, seen_cnt_raw, item_idx):
    """Remove each row's target item before any course-fit term is computed."""
    if seen_mat is None or seen_cnt_raw is None:
        return seen_mat, seen_cnt_raw
    cleaned = seen_mat.clone()
    target_col = item_idx.to(device=cleaned.device).view(-1, 1).long()
    target_seen = cleaned.gather(1, target_col)
    cleaned.scatter_(1, target_col, 0.0)
    cleaned_counts = (seen_cnt_raw.to(cleaned.device) - target_seen).clamp_min(0.0)
    return cleaned, cleaned_counts


def _remove_masked_items_from_seen_history(seen_mat, item_mask):
    if seen_mat is None or item_mask is None:
        counts = None if seen_mat is None else seen_mat.sum(dim=1, keepdim=True)
        return seen_mat, counts
    cleaned = seen_mat.clone()
    mask = item_mask.to(device=cleaned.device, dtype=torch.bool).view(-1)
    if mask.numel() != cleaned.size(1):
        raise ValueError("item history mask must match the item catalog size")
    cleaned[:, mask] = 0.0
    return cleaned, cleaned.sum(dim=1, keepdim=True)


def _exclude_previously_selected_users(scores, candidate_user_ids, selected_user_ids):
    if scores is None or candidate_user_ids is None or selected_user_ids is None:
        return scores
    if selected_user_ids.numel() == 0:
        return scores
    repeated = (candidate_user_ids.unsqueeze(2) == selected_user_ids.unsqueeze(1)).any(dim=2)
    masked = scores.masked_fill(repeated, float("-inf"))
    all_blocked = repeated.all(dim=1)
    if all_blocked.any():
        masked[all_blocked] = scores[all_blocked]
    return masked


def _coursefit_active_update_mask(fit_score, action_idx, threshold=0.0):
    if fit_score is None:
        return None
    chosen = fit_score.gather(1, action_idx.view(-1, 1)).view(-1)
    return torch.isfinite(chosen) & (chosen > float(threshold))


def _deterministic_candidate_positions(probs, num_samples):
    k = min(int(num_samples), int(probs.size(1)))
    return torch.argsort(probs, dim=1, descending=True, stable=True)[:, :k]


def _finite_tensor_mean(values):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.mean().item())


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


def _normalize_course_term_tensor(term, mode="none", clip=2.0, eps=1e-6, scale=None):
    """Normalize a non-negative course-feedback term while preserving its sign role."""
    norm_mode = str(mode or "none").strip().lower()
    if norm_mode in {"none", "off", "0", "false"}:
        return term
    clip = float(max(eps, clip))
    eps = float(max(1e-12, eps))

    if scale is None:
        with torch.no_grad():
            detached = term.detach().abs()
            active = detached > eps
            if not bool(active.any().item()):
                return term
            scale_t = detached[active].mean().clamp_min(eps)
    else:
        scale_t = torch.as_tensor(scale, dtype=term.dtype, device=term.device).clamp_min(eps)

    return (term / scale_t).clamp(0.0, clip)


def _sage_tail_gate_from_pop(pop, max_pop, gate_min=0.1, gate_max=0.6, eps=1e-6):
    """Popularity-aware gate: lower train popularity receives more course signal."""
    pop_t = torch.as_tensor(pop, dtype=torch.float32)
    lo = float(min(gate_min, gate_max))
    hi = float(max(gate_min, gate_max))
    max_pop_t = torch.as_tensor(max_pop, dtype=pop_t.dtype, device=pop_t.device).clamp_min(float(eps))
    pop_norm = torch.log1p(pop_t.clamp_min(0.0)) / torch.log1p(max_pop_t)
    pop_norm = pop_norm.clamp(0.0, 1.0)
    return (lo + (hi - lo) * (1.0 - pop_norm)).clamp(lo, hi)


def _sage_cold_or_tail_mask_from_pop(pop, max_pop, cold_threshold=1, tail_pop_ratio=0.10, eps=1e-6):
    """Return rows where SAGE-lite should affect strict-cold or train-tail targets."""
    pop_t = torch.as_tensor(pop, dtype=torch.float32)
    max_pop_t = torch.as_tensor(max_pop, dtype=pop_t.dtype, device=pop_t.device).clamp_min(float(eps))
    cold = pop_t.clamp_min(0.0) < float(cold_threshold)
    tail_ratio = float(min(1.0, max(0.0, tail_pop_ratio)))
    tail = pop_t.clamp_min(0.0) <= (max_pop_t * tail_ratio)
    return cold | tail


def _sampling_probs_from_scores(scores, temp=1.0, epsilon=0.0):
    n_cols = int(scores.size(1))
    safe_temp = max(float(temp), 1e-6)
    probs = F.softmax(scores / safe_temp, dim=1)
    bad_rows = (~torch.isfinite(probs)).any(dim=1) | (probs.sum(dim=1) <= 0)
    if bad_rows.any():
        probs[bad_rows] = 1.0 / max(1, n_cols)
    eps = float(min(1.0, max(0.0, epsilon)))
    probs = (1.0 - eps) * probs + eps / max(1, n_cols)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _course_probs_from_fit(course_fit, course_temp=0.20):
    n_cols = int(course_fit.size(1))
    temp = max(float(course_temp), 1e-6)
    probs = F.softmax(course_fit / temp, dim=1)
    bad_rows = (~torch.isfinite(probs)).any(dim=1) | (probs.sum(dim=1) <= 0)
    if bad_rows.any():
        probs[bad_rows] = 1.0 / max(1, n_cols)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _sage_only_cold_or_tail_candidate_probs(
    top_scores,
    course_fit_topk,
    sage_gate_topk,
    candidate_temp=1.0,
    candidate_epsilon=0.0,
    course_temp=0.20,
):
    """Build row-wise candidate probabilities: SAGE top-k for active rows, full retrieval for hot rows."""
    full_probs = _sampling_probs_from_scores(top_scores, temp=candidate_temp, epsilon=candidate_epsilon)
    pool_k = int(course_fit_topk.size(1))
    retrieval_topk = _sampling_probs_from_scores(
        top_scores[:, :pool_k],
        temp=candidate_temp,
        epsilon=candidate_epsilon,
    )
    course_probs = _course_probs_from_fit(course_fit_topk, course_temp=course_temp)
    mixed_topk = (1.0 - sage_gate_topk) * retrieval_topk + sage_gate_topk * course_probs
    mixed_topk = mixed_topk / mixed_topk.sum(dim=1, keepdim=True).clamp_min(1e-12)
    active_rows = (sage_gate_topk > 0).any(dim=1, keepdim=True)

    active_probs = torch.zeros_like(top_scores, dtype=top_scores.dtype, device=top_scores.device)
    active_probs[:, :pool_k] = mixed_topk
    probs = torch.where(active_rows, active_probs, full_probs)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _sage_two_expert_candidate_probs(
    top_scores,
    course_fit_topk,
    sage_gate_topk,
    candidate_temp=1.0,
    candidate_epsilon=0.0,
    course_temp=0.20,
):
    """Two-expert SAGE-lite probabilities: uniform pool expert + course-fit expert."""
    full_probs = _sampling_probs_from_scores(top_scores, temp=candidate_temp, epsilon=candidate_epsilon)
    pool_k = int(course_fit_topk.size(1))
    uniform_expert = torch.full_like(course_fit_topk, 1.0 / max(1, pool_k))
    course_expert = _course_probs_from_fit(course_fit_topk, course_temp=course_temp)
    mixed_topk = (1.0 - sage_gate_topk) * uniform_expert + sage_gate_topk * course_expert
    mixed_topk = mixed_topk / mixed_topk.sum(dim=1, keepdim=True).clamp_min(1e-12)
    active_rows = (sage_gate_topk > 0).any(dim=1, keepdim=True)

    active_probs = torch.zeros_like(top_scores, dtype=top_scores.dtype, device=top_scores.device)
    active_probs[:, :pool_k] = mixed_topk
    probs = torch.where(active_rows, active_probs, full_probs)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _sage_course_sampling_combined_score(
    retrieval_score,
    fit_norm,
    sage_gate=None,
    beta=0.20,
    use_sage_lite=False,
    sage_only_cold_or_tail=False,
):
    base_score = retrieval_score + float(beta) * fit_norm
    if not use_sage_lite or sage_gate is None:
        return base_score
    sage_score = (1.0 - sage_gate) * retrieval_score + sage_gate * fit_norm
    if not sage_only_cold_or_tail:
        return sage_score
    active_rows = (sage_gate > 0).any(dim=1, keepdim=True)
    return torch.where(active_rows, sage_score, base_score)


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
        self.cgrc_recon_mlp = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        for module in self.cgrc_recon_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        for param in self.cgrc_recon_mlp.parameters():
            param.requires_grad = bool(getattr(config, "use_cgrc_recon", False))
        sage_gate_bucket_count = max(2, int(getattr(config, "sage_gate_bucket_count", 20)))
        sage_gate_hidden_dim = max(1, int(getattr(config, "sage_gate_hidden_dim", 32)))
        self.sage_gate_bucket_emb = nn.Embedding(sage_gate_bucket_count, sage_gate_hidden_dim)
        self.sage_gate_mlp = nn.Sequential(
            nn.Linear(sage_gate_hidden_dim, sage_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(sage_gate_hidden_dim, 1),
        )
        self.sage_score_gate_mlp = nn.Sequential(
            nn.Linear(sage_gate_hidden_dim, sage_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(sage_gate_hidden_dim, 2),
        )
        nn.init.normal_(self.sage_gate_bucket_emb.weight, mean=0.0, std=0.02)
        for module in list(self.sage_gate_mlp) + list(self.sage_score_gate_mlp):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        sage_gate_trainable = (
            bool(getattr(config, "use_sage_lite", False))
            and str(getattr(config, "sage_gate_mode", "heuristic")).strip().lower() == "bucket_mlp"
        )
        for param in self.sage_gate_bucket_emb.parameters():
            param.requires_grad = sage_gate_trainable
        for param in self.sage_gate_mlp.parameters():
            param.requires_grad = sage_gate_trainable
        sage_score_gate_trainable = (
            bool(getattr(config, "use_sage_lite", False))
            and bool(getattr(config, "sage_two_expert_score_fusion", False))
        )
        for param in self.sage_score_gate_mlp.parameters():
            param.requires_grad = sage_score_gate_trainable
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

    def _compute_structural_redundancy_for_pairs(self, item_idx, seen_mat):
        n_pairs = int(item_idx.numel())
        redundant = torch.zeros((n_pairs, 1), dtype=torch.float32, device=self.device)
        if n_pairs < 1 or seen_mat is None:
            return redundant
        if self.item_video_contain is None and self.item_same_family is None:
            return redundant

        item_idx = item_idx.to(self.device).view(-1).long()
        seen_mat = seen_mat.to(self.device).float()
        if seen_mat.size(0) != n_pairs:
            raise ValueError(
                "Structural redundancy pair scorer expects one seen row per item."
            )

        video_min = float(min(0.999, max(0.0, getattr(self.cfg, "feedback_course_struct_video_min", 0.60))))
        video_band = max(1e-6, 1.0 - video_min)
        chunk_size = max(1, int(getattr(self.cfg, "feedback_course_struct_chunk", 8192)))

        for start in range(0, n_pairs, chunk_size):
            end = min(n_pairs, start + chunk_size)
            seen_chunk = seen_mat[start:end]
            item_chunk = item_idx[start:end]
            score = torch.zeros((end - start,), dtype=torch.float32, device=self.device)

            if self.item_same_family is not None:
                family_rows = self.item_same_family.index_select(0, item_chunk).float()
                hard = (family_rows * seen_chunk).amax(dim=1)
                score = torch.maximum(score, hard)

            if self.item_video_contain is not None:
                video_rows = self.item_video_contain.index_select(0, item_chunk).float()
                video_cover = (video_rows * seen_chunk).amax(dim=1)
                soft = ((video_cover - video_min) / video_band).clamp(0.0, 1.0)
                score = torch.maximum(score, soft)

            redundant[start:end, 0] = score.clamp(0.0, 1.0)

        return redundant

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

    def _effective_train_cold_mask(self, pop, item_idx=None):
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

        if mode == "item_tail":
            if item_idx is None or self.item_popularity is None:
                raise RuntimeError("item_tail pseudo-cold mode requires item indices and item popularity")
            cache = getattr(self, "_fixed_pseudo_cold_item_mask_cache", None)
            if cache is None:
                cache = _build_fixed_tail_pseudo_item_mask(
                    self.item_popularity,
                    ratio=float(getattr(self.cfg, "pseudo_cold_ratio", 0.0)),
                    min_pop=int(getattr(self.cfg, "pseudo_cold_min_pop", 1)),
                ).to(self.device)
                self._fixed_pseudo_cold_item_mask_cache = cache
            pseudo_cold = cache.index_select(0, item_idx.to(self.device).view(-1).long())
            return true_cold | pseudo_cold

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
        self.item_popularity_max = None
        self.item_difficulty = None
        self.course_term_ema_scales = {}
        self._original_v2_teacher_item_emb = None
        self._original_v2_teacher_user_emb = None

    @torch.no_grad()
    def initialize_original_v2_teacher_(self):
        """Snapshot and freeze the pretrained IV space used as the V2 oracle."""
        self._original_v2_teacher_item_emb = self.item_id_emb.weight.detach().clone()
        self._original_v2_teacher_user_emb = self.user_proj(
            self.user_emb.weight.detach()
        ).detach().clone()
        for parameter in self.item_id_emb.parameters():
            parameter.requires_grad_(False)
        for parameter in self.user_emb.parameters():
            parameter.requires_grad_(False)
        for parameter in self.user_proj.parameters():
            parameter.requires_grad_(False)

    def set_feedback_item_stats(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_popularity_cpu = None
            self.item_popularity_max = None
            self.item_difficulty = None
            return
        pop = torch.as_tensor(item_popularity, dtype=torch.float32, device=self.device)
        if pop.numel() != self.cfg.n_items:
            raise ValueError(f"item_popularity size mismatch: expect {self.cfg.n_items}, got {pop.numel()}")
        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_popularity_cpu = pop.detach().cpu()
        self.item_popularity_max = pop.max().detach().clamp_min(1.0)
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    def _sage_popularity_bucket_ids(self, pop):
        bucket_count = max(2, int(getattr(self.cfg, "sage_gate_bucket_count", 20)))
        gate_device = self.sage_gate_bucket_emb.weight.device
        pop_t = torch.as_tensor(pop, dtype=torch.float32, device=gate_device).view(-1).clamp_min(0.0)
        if self.item_popularity_max is not None:
            max_pop = self.item_popularity_max.to(device=gate_device).float().clamp_min(1.0)
        else:
            max_pop = pop_t.max().clamp_min(1.0)
        strategy = str(getattr(self.cfg, "sage_gate_bucket_strategy", "paper")).strip().lower()
        if strategy == "log":
            pop_norm = torch.log1p(pop_t) / torch.log1p(max_pop)
            bucket = torch.floor(pop_norm.clamp(0.0, 1.0) * float(bucket_count - 1)).long()
        else:
            bucket = torch.floor(pop_t * float(bucket_count) / (max_pop + 1.0)).long()
        return bucket.clamp(0, bucket_count - 1)

    def _sage_bucket_mlp_gate_from_pop(self, pop, batch_size, n_cols=1):
        gate_device = self.sage_gate_bucket_emb.weight.device
        if pop is None:
            pop = torch.zeros((batch_size,), dtype=torch.float32, device=gate_device)
        pop_t = torch.as_tensor(pop, dtype=torch.float32, device=gate_device).view(-1)
        if pop_t.numel() != int(batch_size):
            pop_t = pop_t.reshape(int(batch_size), -1)[:, 0]
        bucket = self._sage_popularity_bucket_ids(pop_t)
        gate_unit = torch.sigmoid(self.sage_gate_mlp(self.sage_gate_bucket_emb(bucket))).view(batch_size, 1)
        lo = float(min(getattr(self.cfg, "sage_gate_min", 0.10), getattr(self.cfg, "sage_gate_max", 0.60)))
        hi = float(max(getattr(self.cfg, "sage_gate_min", 0.10), getattr(self.cfg, "sage_gate_max", 0.60)))
        gate = lo + (hi - lo) * gate_unit
        if n_cols != 1:
            gate = gate.expand(-1, n_cols)
        return gate

    def _sage_two_expert_score_weights_from_pop(self, pop, score_shape):
        gate_device = self.sage_gate_bucket_emb.weight.device
        batch_size, n_cols = int(score_shape[0]), int(score_shape[1])
        if pop is None:
            pop_t = torch.zeros((batch_size, n_cols), dtype=torch.float32, device=gate_device)
        else:
            pop_t = torch.as_tensor(pop, dtype=torch.float32, device=gate_device)
            if pop_t.dim() == 1:
                if pop_t.numel() == n_cols:
                    pop_t = pop_t.view(1, n_cols).expand(batch_size, -1)
                elif pop_t.numel() == batch_size:
                    pop_t = pop_t.view(batch_size, 1).expand(-1, n_cols)
                else:
                    pop_t = pop_t.view(batch_size, n_cols)
            else:
                pop_t = pop_t.view(batch_size, n_cols)
        bucket = self._sage_popularity_bucket_ids(pop_t.reshape(-1))
        logits = self.sage_score_gate_mlp(self.sage_gate_bucket_emb(bucket))
        weights = torch.softmax(logits, dim=1).view(batch_size, n_cols, 2)
        return weights

    def _resolve_score_fusion_pop(self, scores, cand_idx=None, target_pop=None):
        batch_size, n_cols = int(scores.size(0)), int(scores.size(1))
        if cand_idx is not None and self.item_popularity is not None:
            return self.item_popularity.to(device=scores.device).index_select(0, cand_idx.reshape(-1)).view(batch_size, n_cols)
        if cand_idx is None and self.item_popularity is not None and n_cols == int(self.cfg.n_items):
            return self.item_popularity.to(device=scores.device).view(1, n_cols).expand(batch_size, -1)
        if target_pop is not None:
            pop_t = target_pop.to(device=scores.device).float()
            if pop_t.dim() == 1:
                return pop_t.view(batch_size, 1).expand(-1, n_cols)
            return pop_t.view(batch_size, n_cols)
        return None

    def _build_seen_mat_for_user_ids(self, user_ids, seen_tensor_cache=None):
        if user_ids is None:
            return None
        seen_index = getattr(self, "user_seen_index", None)
        if seen_index is not None:
            uid_t = self._resolve_user_id_tensor(user_ids)
            return seen_index.index_select(0, uid_t).float()
        batch_size = len(user_ids)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if seen_tensor_cache is None:
            return seen_mat
        for row, uid in enumerate(user_ids):
            seen_idx = seen_tensor_cache.get(int(uid))
            if seen_idx is not None and seen_idx.numel() > 0:
                seen_mat[row, seen_idx] = 1.0
        return seen_mat

    def _compute_user_item_course_expert_scores(self, user_ids, seen_tensor_cache=None, cand_idx=None):
        if (
            user_ids is None
            or self.item_prereq_item_mat is None
            or self.item_prereq_item_cnt is None
            or self.item_concept_overlap is None
        ):
            return None
        seen_mat = self._build_seen_mat_for_user_ids(user_ids, seen_tensor_cache=seen_tensor_cache)
        if seen_mat is None or seen_mat.numel() < 1:
            return None
        seen_cnt_raw = seen_mat.sum(dim=1, keepdim=True)
        seen_cnt = seen_cnt_raw.clamp_min(1.0)
        prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
        prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
        prereq_gap = torch.where(
            prereq_cnt > 0,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)
        concept_match = (torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt).clamp(0.0, 1.0)
        redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
        concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
        concept_band = max(1e-6, redundant_thr - concept_min)
        concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
        redundant = ((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)).clamp(0.0, 1.0)
        redundant_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_redundant_concept_gate)))
        prereq_safe = (prereq_gap <= float(min(1.0, max(0.0, self.cfg.feedback_course_prereq_gate)))).float()
        concept_bonus = concept_bonus * prereq_safe * (1.0 - redundant_gate * redundant)
        difficulty_gap = torch.zeros_like(concept_match)
        if self.item_difficulty is not None:
            warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
            user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)
            difficulty_gap = F.relu(self.item_difficulty.to(device=self.device).view(1, -1) - user_readiness)
        course_score_full = (
            float(self.cfg.feedback_course_concept_weight) * concept_bonus
            - float(self.cfg.feedback_course_prereq_weight) * prereq_gap
            - float(self.cfg.feedback_course_difficulty_weight) * difficulty_gap
            - float(self.cfg.feedback_course_redundant_weight) * redundant
        )
        if cand_idx is not None:
            return course_score_full.gather(1, cand_idx.to(device=self.device).long())
        return course_score_full

    def apply_sage_two_expert_score_fusion(
        self,
        scores,
        course_fit=None,
        target_pop=None,
        user_ids=None,
        seen_tensor_cache=None,
        cand_idx=None,
    ):
        if not (
            bool(getattr(self.cfg, "use_sage_lite", False))
            and bool(getattr(self.cfg, "sage_two_expert_score_fusion", False))
        ):
            return scores
        if course_fit is None:
            course_fit = self._compute_user_item_course_expert_scores(
                user_ids,
                seen_tensor_cache=seen_tensor_cache,
                cand_idx=cand_idx,
            )
        if course_fit is None:
            return scores
        course_fit = course_fit.to(device=scores.device, dtype=scores.dtype)
        if not torch.isfinite(course_fit).all():
            course_fit = torch.nan_to_num(course_fit, nan=0.0, posinf=0.0, neginf=0.0)
        course_scale = course_fit.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        course_expert = course_fit / course_scale
        pop_grid = self._resolve_score_fusion_pop(scores, cand_idx=cand_idx, target_pop=target_pop)
        weights = self._sage_two_expert_score_weights_from_pop(pop_grid, scores.shape).to(device=scores.device, dtype=scores.dtype)
        fused = weights[..., 0] * scores + weights[..., 1] * course_expert
        if bool(getattr(self.cfg, "sage_only_cold_or_tail", False)):
            if pop_grid is None:
                return scores
            max_pop = (
                self.item_popularity_max.to(device=scores.device)
                if self.item_popularity_max is not None
                else pop_grid.max().clamp_min(1.0)
            )
            active = _sage_cold_or_tail_mask_from_pop(
                pop_grid.reshape(-1),
                max_pop,
                cold_threshold=float(getattr(self.cfg, "cold_threshold", 1)),
                tail_pop_ratio=float(getattr(self.cfg, "sage_tail_pop_ratio", 0.10)),
            ).view_as(scores)
            fused = torch.where(active, fused, scores)
        return fused

    def _sage_tail_gate(self, target_pop, batch_size, n_cols=1):
        only_cold_or_tail = bool(getattr(self.cfg, "sage_only_cold_or_tail", False))
        gate_mode = str(getattr(self.cfg, "sage_gate_mode", "heuristic")).strip().lower()
        if gate_mode == "bucket_mlp":
            gate = self._sage_bucket_mlp_gate_from_pop(target_pop, batch_size, n_cols=n_cols)
            if only_cold_or_tail:
                gate_device = gate.device
                if target_pop is None:
                    active = torch.zeros((batch_size, 1), dtype=torch.bool, device=gate_device)
                else:
                    target_pop_t = target_pop.to(device=gate_device).float().view(-1)
                    max_pop = (
                        self.item_popularity_max.to(device=gate_device)
                        if self.item_popularity_max is not None
                        else target_pop_t.max().clamp_min(1.0)
                    )
                    active = _sage_cold_or_tail_mask_from_pop(
                        target_pop_t,
                        max_pop,
                        cold_threshold=float(getattr(self.cfg, "cold_threshold", 1)),
                        tail_pop_ratio=float(getattr(self.cfg, "sage_tail_pop_ratio", 0.10)),
                    ).view(batch_size, 1)
                if n_cols != 1:
                    active = active.expand(-1, n_cols)
                gate = gate * active.float()
            return gate
        if target_pop is None or self.item_popularity_max is None:
            if only_cold_or_tail:
                gate = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
                if n_cols != 1:
                    gate = gate.expand(-1, n_cols)
                return gate
            gate_val = float(getattr(self.cfg, "sage_gate_max", 0.60))
            gate = torch.full((batch_size, 1), gate_val, dtype=torch.float32, device=self.device)
        else:
            target_pop = target_pop.to(device=self.device).float().view(-1)
            gate = _sage_tail_gate_from_pop(
                target_pop,
                self.item_popularity_max.to(device=self.device),
                gate_min=float(getattr(self.cfg, "sage_gate_min", 0.10)),
                gate_max=float(getattr(self.cfg, "sage_gate_max", 0.60)),
            ).view(batch_size, 1)
            if only_cold_or_tail:
                active = _sage_cold_or_tail_mask_from_pop(
                    target_pop,
                    self.item_popularity_max.to(device=self.device),
                    cold_threshold=float(getattr(self.cfg, "cold_threshold", 1)),
                    tail_pop_ratio=float(getattr(self.cfg, "sage_tail_pop_ratio", 0.10)),
                ).view(batch_size, 1)
                gate = gate * active.float()
        if n_cols != 1:
            gate = gate.expand(-1, n_cols)
        return gate

    def _cgrc_recon_active_gate(self, target_pop, batch_size, n_cols=1):
        if not bool(getattr(self.cfg, "use_cgrc_recon", False)):
            return torch.zeros((batch_size, n_cols), dtype=torch.float32, device=self.device)
        if not bool(getattr(self.cfg, "cgrc_recon_only_cold_or_tail", True)):
            return torch.ones((batch_size, n_cols), dtype=torch.float32, device=self.device)
        if target_pop is None or self.item_popularity_max is None:
            return torch.zeros((batch_size, n_cols), dtype=torch.float32, device=self.device)
        target_pop = target_pop.to(device=self.device).float().view(-1)
        active = _sage_cold_or_tail_mask_from_pop(
            target_pop,
            self.item_popularity_max.to(device=self.device),
            cold_threshold=float(getattr(self.cfg, "cold_threshold", 1)),
            tail_pop_ratio=float(getattr(self.cfg, "cgrc_recon_tail_pop_ratio", 0.10)),
        ).float().view(batch_size, 1)
        if n_cols != 1:
            active = active.expand(-1, n_cols)
        return active

    def _sample_cgrc_recon_pseudo_mask(self, pop):
        pop = pop.to(device=self.device).float().view(-1)
        eligible = pop >= float(getattr(self.cfg, "cold_threshold", 1))
        n_eligible = int(eligible.sum().detach().item())
        active = torch.zeros_like(eligible)
        if n_eligible < 1:
            return active
        ratio = float(getattr(self.cfg, "cgrc_recon_pseudo_ratio", 0.30))
        if ratio <= 0.0:
            return active
        if ratio >= 1.0:
            return eligible
        n_pick = max(1, min(n_eligible, int(math.ceil(n_eligible * ratio))))
        draw = torch.rand_like(pop).masked_fill(~eligible, -1.0)
        _, chosen = torch.topk(draw, k=n_pick, dim=0)
        active[chosen] = True
        return active & eligible

    def _cgrc_recon_logits_from_vectors(self, item_vec, candidate_user_vec):
        batch_size, n_cand, dim = candidate_user_vec.shape
        item_part = item_vec.view(batch_size, 1, dim).expand(-1, n_cand, -1)
        flat = torch.cat([candidate_user_vec, item_part], dim=2).reshape(batch_size * n_cand, dim * 2)
        return self.cgrc_recon_mlp(flat).view(batch_size, n_cand)

    def _cgrc_recon_candidate_logits(self, item_vec, cand_user_idx, user_bank_raw):
        if user_bank_raw is None or cand_user_idx is None:
            return None
        cand_user_vec = user_bank_raw[cand_user_idx.long()]
        return self._cgrc_recon_logits_from_vectors(item_vec, cand_user_vec)

    def _compute_cgrc_recon_aux_loss(self, user_idx, item_vec, pop):
        zero = self.user_emb.weight.new_zeros(())
        info = {
            "cgrc_recon_loss": 0.0,
            "cgrc_recon_active_ratio": 0.0,
            "cgrc_recon_pos_count": 0,
        }
        if (
            not self.training
            or not bool(getattr(self.cfg, "use_cgrc_recon", False))
            or float(getattr(self.cfg, "cgrc_recon_aux_weight", 0.0)) <= 0.0
            or user_idx is None
            or item_vec is None
            or pop is None
        ):
            return zero, info

        batch_size = int(user_idx.numel())
        if batch_size < 2:
            return zero, info
        active = self._sample_cgrc_recon_pseudo_mask(pop)
        active_idx = active.nonzero(as_tuple=True)[0]
        if active_idx.numel() < 1:
            return zero, info

        pool_user_ids = torch.unique(user_idx.to(device=self.device).long().view(-1), sorted=False)
        if pool_user_ids.numel() < 2:
            return zero, info

        active_item_vec = item_vec.index_select(0, active_idx)
        pos_user_ids = user_idx.to(device=self.device).long().view(-1).index_select(0, active_idx)
        with torch.no_grad():
            pool_user_vec_for_select = self.user_proj(self.user_emb(pool_user_ids)).detach()
            select_scores = torch.matmul(active_item_vec.detach(), pool_user_vec_for_select.t())
            k = min(max(2, int(getattr(self.cfg, "cgrc_recon_topk", 64))), int(pool_user_ids.numel()))
            _, top_cols = torch.topk(select_scores, k=k, dim=1)
            cand_user_ids = pool_user_ids[top_cols].clone()
            cand_user_ids[:, 0] = pos_user_ids

        flat_cand_ids = cand_user_ids.reshape(-1)
        cand_user_vec = self.user_proj(self.user_emb(flat_cand_ids)).view(active_idx.numel(), -1, self.cfg.emb_dim)
        if bool(getattr(self.cfg, "cgrc_recon_detach_user", False)):
            cand_user_vec = cand_user_vec.detach()

        logits = self._cgrc_recon_logits_from_vectors(active_item_vec, cand_user_vec)
        dup_pos = cand_user_ids.eq(pos_user_ids.view(-1, 1))
        if dup_pos.size(1) > 1:
            dup_pos[:, 0] = False
            logits = logits.masked_fill(dup_pos, -1e9)
        logits = logits / float(getattr(self.cfg, "cgrc_recon_temperature", 0.50))
        labels = torch.zeros(active_idx.numel(), dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        info["cgrc_recon_loss"] = float(loss.detach().item())
        info["cgrc_recon_active_ratio"] = float(active.float().mean().detach().item())
        info["cgrc_recon_pos_count"] = int(active_idx.numel())
        return loss, info

    def _normalize_course_term(self, name, term):
        mode = str(getattr(self.cfg, "feedback_course_term_norm", "none")).strip().lower()
        if mode in {"none", "off", "0", "false"}:
            return term
        clip = float(getattr(self.cfg, "feedback_course_term_norm_clip", 2.0))
        eps = float(getattr(self.cfg, "feedback_course_term_norm_eps", 1e-6))
        if mode == "batch":
            return _normalize_course_term_tensor(term, mode="batch", clip=clip, eps=eps)
        if mode != "ema":
            return term

        with torch.no_grad():
            detached = term.detach().abs()
            active = detached > eps
            if not bool(active.any().item()):
                return term
            batch_scale = detached[active].mean().clamp_min(eps)
            prev = self.course_term_ema_scales.get(name)
            if prev is None:
                scale = batch_scale
            else:
                decay = float(getattr(self.cfg, "feedback_course_term_norm_ema_decay", 0.95))
                prev = prev.to(device=batch_scale.device, dtype=batch_scale.dtype)
                scale = decay * prev + (1.0 - decay) * batch_scale
            self.course_term_ema_scales[name] = scale.detach()
        return _normalize_course_term_tensor(term, mode="ema", clip=clip, eps=eps, scale=scale)

    def _normalize_course_terms(self, prefix, terms):
        mode = str(getattr(self.cfg, "feedback_course_term_norm", "none")).strip().lower()
        if mode in {"none", "off", "0", "false"}:
            return terms
        return {
            name: self._normalize_course_term(f"{prefix}.{name}", value)
            for name, value in terms.items()
        }

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

    def _build_batch_false_negative_mask(self, user_ids, item_idx, user_seen_items, pos_mask):
        masks = []
        if (
            self.training and getattr(self.cfg, "mask_known_pos_neg", False) and
            user_seen_items is not None and item_idx.numel() > 1
        ):
            known_pos = self._build_known_positive_batch_mask(user_ids, item_idx, user_seen_items)
            if known_pos is not None:
                masks.append(known_pos & (~pos_mask))

        if getattr(self.cfg, "mask_same_item_neg", True) and item_idx.numel() > 1:
            same_item = item_idx.view(-1, 1) == item_idx.view(1, -1)
            masks.append(same_item & (~pos_mask))

        if not masks:
            return None

        false_neg_mask = masks[0].bool()
        for mask in masks[1:]:
            false_neg_mask = false_neg_mask | mask.bool()
        false_neg_mask = false_neg_mask & (~pos_mask)
        return false_neg_mask

    def _mask_false_negative_candidate_logits(self, cand_logits, cand_idx, false_neg_mask):
        if false_neg_mask is None:
            return cand_logits
        invalid = false_neg_mask.gather(1, cand_idx)
        if invalid.size(1) > 0:
            invalid[:, 0] = False
        return cand_logits.masked_fill(invalid, -1e9)

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

    def _compute_course_concept_match(self, seen_mat, item_idx):
        batch_size = int(item_idx.size(0))
        zero = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        if self.item_concept_overlap is None or seen_mat is None:
            return zero
        if seen_mat.numel() < 1:
            return zero

        item_idx = item_idx.view(-1).long()
        overlap_rows = self.item_concept_overlap.index_select(0, item_idx).to(device=self.device)
        seen_scores = overlap_rows * seen_mat.float()
        seen_cnt = seen_mat.sum(dim=1, keepdim=True).float()
        effective_seen_cnt = seen_cnt
        if bool(getattr(self.cfg, "feedback_course_match_exclude_target", False)):
            target_col = item_idx.view(-1, 1)
            target_seen = seen_mat.gather(1, target_col).float()
            seen_scores = seen_scores.scatter(1, target_col, 0.0)
            effective_seen_cnt = (seen_cnt - target_seen).clamp_min(0.0)
        mode = str(getattr(self.cfg, "feedback_course_match_mode", "mean")).strip().lower()

        if mode == "max":
            concept_match = seen_scores.max(dim=1, keepdim=True).values
        elif mode == "topk":
            k = max(1, int(getattr(self.cfg, "feedback_course_match_topk", 5)))
            k = min(k, seen_scores.size(1))
            top_vals = torch.topk(seen_scores, k=k, dim=1).values
            denom = torch.minimum(effective_seen_cnt, torch.full_like(effective_seen_cnt, float(k))).clamp_min(1.0)
            concept_match = top_vals.sum(dim=1, keepdim=True) / denom
        else:
            concept_match = seen_scores.sum(dim=1, keepdim=True) / effective_seen_cnt.clamp_min(1.0)

        return concept_match.clamp(0.0, 1.0)

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

        pseudo_item_mask = getattr(self, "_fixed_pseudo_cold_item_mask_cache", None)
        if pseudo_item_mask is not None:
            seen_mat, seen_cnt_raw = _remove_masked_items_from_seen_history(
                seen_mat,
                pseudo_item_mask,
            )
        if seen_cnt_raw.max().item() < 1:
            return terms
        if bool(getattr(self.cfg, "feedback_course_match_exclude_target", False)):
            seen_mat, seen_cnt_raw = _remove_target_from_seen_history(
                seen_mat,
                seen_cnt_raw,
                item_idx,
            )

        active = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        if self.cfg.feedback_course_only_cold and target_pop is not None:
            active = self._cold_mask_from_pop(target_pop).float().view(-1, 1)

        seen_active = (seen_cnt_raw >= 1.0).float()
        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)

        prereq_gap, prereq_safe = self._compute_prereq_gap_and_safe(seen_mat, item_idx)
        terms["prereq_gap"] = prereq_gap * active

        redundant_mode = str(getattr(self.cfg, "feedback_course_redundant_mode", "concept")).strip().lower()
        if redundant_mode == "video_family":
            terms["redundant"] = (
                self._compute_structural_redundancy_for_pairs(item_idx, seen_mat).clamp(0.0, 1.0)
                * seen_active
                * active
            )

        if self.item_concept_overlap is not None:
            concept_match = self._compute_course_concept_match(seen_mat, item_idx)
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

        return self._normalize_course_terms("reward", terms)

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

        pseudo_item_mask = getattr(self, "_fixed_pseudo_cold_item_mask_cache", None)
        if pseudo_item_mask is not None:
            seen_mat, seen_cnt_raw = _remove_masked_items_from_seen_history(
                seen_mat,
                pseudo_item_mask,
            )
        if seen_cnt_raw.max().item() < 1:
            return zero

        flat_item_idx = item_idx.view(-1, 1).expand(-1, n_cand).reshape(-1)
        if bool(getattr(self.cfg, "feedback_course_match_exclude_target", False)):
            seen_mat, seen_cnt_raw = _remove_target_from_seen_history(
                seen_mat,
                seen_cnt_raw,
                flat_item_idx,
            )
        fit = torch.zeros((flat_user_idx.size(0), 1), dtype=torch.float32, device=self.device)

        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)
        prereq_gap, prereq_safe = self._compute_prereq_gap_and_safe(seen_mat, flat_item_idx)

        concept_bonus = torch.zeros_like(fit)
        redundant = torch.zeros_like(fit)
        seen_active = (seen_cnt_raw >= 1.0).float()
        redundant_mode = str(getattr(self.cfg, "feedback_course_redundant_mode", "concept")).strip().lower()
        if redundant_mode == "video_family":
            redundant = self._compute_structural_redundancy_for_pairs(flat_item_idx, seen_mat).clamp(0.0, 1.0)
        if self.item_concept_overlap is not None:
            concept_match = self._compute_course_concept_match(seen_mat, flat_item_idx)
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

        concept_bonus = self._normalize_course_term("sample.concept_bonus", concept_bonus)
        prereq_gap = self._normalize_course_term("sample.prereq_gap", prereq_gap)
        difficulty_gap = self._normalize_course_term("sample.difficulty_gap", difficulty_gap)
        redundant = self._normalize_course_term("sample.redundant", redundant)

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

    def _compute_sage_aux_loss(
        self,
        item_emb,
        item_idx,
        target_pop=None,
        user_seen_items=None,
        user_bank_raw=None,
        user_bank_norm=None,
    ):
        zero = item_emb.new_zeros(())
        info = {
            "sage_aux_loss": 0.0,
            "sage_aux_active_ratio": 0.0,
            "sage_aux_pool_fit": 0.0,
        }
        if (
            not self.training
            or not bool(getattr(self.cfg, "use_sage_aux_loss", False))
            or float(getattr(self.cfg, "sage_aux_weight", 0.0)) <= 0.0
            or item_idx is None
            or target_pop is None
            or user_seen_items is None
        ):
            return zero, info
        if user_bank_raw is None:
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        elif user_bank_norm is None:
            user_bank_norm = F.normalize(user_bank_raw.detach(), dim=1)

        detach_user = bool(getattr(self.cfg, "sage_aux_detach_user", True))
        bank_raw = user_bank_raw.detach() if detach_user else user_bank_raw
        bank_norm = user_bank_norm.detach() if (detach_user and user_bank_norm is not None) else user_bank_norm
        pool_k = int(getattr(self.cfg, "sage_aux_pool_topk", getattr(self.cfg, "sage_pool_topk", 64)))
        pool_k = max(1, min(pool_k, self.cfg.n_users))
        top_scores, top_idx = self._retrieve_topm_exact(
            F.normalize(item_emb, dim=1),
            bank_raw,
            pool_k,
            user_bank_norm=bank_norm,
        )
        course_fit = self._compute_candidate_course_fit(
            top_idx,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        if not torch.isfinite(course_fit).all():
            course_fit = torch.nan_to_num(course_fit, nan=0.0, posinf=0.0, neginf=0.0)

        retrieval_temp = max(float(getattr(self.cfg, "sage_aux_retrieval_temp", 1.0)), 1e-6)
        course_temp = max(
            float(getattr(self.cfg, "sage_aux_course_temp", getattr(self.cfg, "sage_course_temp", 0.20))),
            1e-6,
        )
        retrieval_log_probs = F.log_softmax(top_scores / retrieval_temp, dim=1)
        course_probs = _course_probs_from_fit(course_fit, course_temp=course_temp).detach()
        row_loss = F.kl_div(retrieval_log_probs, course_probs, reduction="none").sum(dim=1)

        target_pop = target_pop.to(device=self.device).float().view(-1)
        if bool(getattr(self.cfg, "sage_aux_only_strict_cold", True)):
            active = target_pop < float(getattr(self.cfg, "cold_threshold", 1))
        else:
            max_pop = (
                self.item_popularity_max.to(device=self.device)
                if self.item_popularity_max is not None
                else target_pop.max().clamp_min(1.0)
            )
            active = _sage_cold_or_tail_mask_from_pop(
                target_pop,
                max_pop,
                cold_threshold=float(getattr(self.cfg, "cold_threshold", 1)),
                tail_pop_ratio=float(getattr(self.cfg, "sage_tail_pop_ratio", 0.10)),
            )
        active = active.view(-1)
        active_count = int(active.sum().detach().item())
        if active_count < 1:
            return zero, info

        loss = row_loss[active].mean()
        info["sage_aux_loss"] = float(loss.detach().item())
        info["sage_aux_active_ratio"] = float(active.float().mean().detach().item())
        info["sage_aux_pool_fit"] = float(course_fit[active].mean().detach().item())
        return loss, info

    def get_candidates(
        self,
        item_emb,
        user_bank_raw=None,
        user_bank_norm=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
    ):
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
        pool_scores = top_scores
        pool_idx = top_idx
        probs_override = None
        sage_active = (
            bool(getattr(self.cfg, "use_sage_lite", False))
            and item_idx is not None
            and user_seen_items is not None
            and top_m > 1
        )
        sage_gate_mean = 0.0
        sage_tail_active_mean = 0.0
        sage_pool_fit_mean = 0.0
        sage_two_expert = bool(getattr(self.cfg, "sage_use_two_expert", False))
        cgrc_recon_sample_active_mean = 0.0
        cgrc_recon_sample_score_mean = 0.0
        if sage_active:
            sage_only_cold_or_tail = bool(getattr(self.cfg, "sage_only_cold_or_tail", False))
            pool_k = int(getattr(self.cfg, "sage_pool_topk", 64))
            pool_k = max(N_cand, min(top_m, max(1, pool_k)))
            sage_pool_scores = top_scores[:, :pool_k]
            sage_pool_idx = top_idx[:, :pool_k]
            with torch.no_grad():
                sage_gate = self._sage_tail_gate(target_pop, B, n_cols=pool_k)
                sage_gate_mean = float(sage_gate.mean().item())
                sage_tail_active_mean = float((sage_gate > 0).float().mean().item())
                active_rows = (sage_gate > 0).any(dim=1)
                if sage_only_cold_or_tail and not bool(active_rows.any().item()):
                    course_fit = None
                elif sage_only_cold_or_tail:
                    active_idx = active_rows.nonzero(as_tuple=True)[0]
                    course_fit = torch.zeros((B, pool_k), dtype=top_scores.dtype, device=self.device)
                    active_target_pop = target_pop[active_idx] if target_pop is not None else None
                    active_fit = self._compute_candidate_course_fit(
                        sage_pool_idx[active_idx],
                        item_idx=item_idx[active_idx],
                        target_pop=active_target_pop,
                        user_seen_items=user_seen_items,
                    )
                    course_fit[active_idx] = active_fit.to(dtype=course_fit.dtype, device=course_fit.device)
                else:
                    course_fit = self._compute_candidate_course_fit(
                        sage_pool_idx,
                        item_idx=item_idx,
                        target_pop=target_pop,
                        user_seen_items=user_seen_items,
                    )
                if course_fit is None:
                    course_probs = None
                    sage_pool_fit_mean = 0.0
                    pool_k = top_m
                    pool_scores = top_scores
                    pool_idx = top_idx
                    probs_override = None
                else:
                    if not torch.isfinite(course_fit).all():
                        course_fit = torch.nan_to_num(course_fit, nan=0.0, posinf=0.0, neginf=0.0)
                    course_temp = max(float(getattr(self.cfg, "sage_course_temp", 0.20)), 1e-6)
                    course_probs = _course_probs_from_fit(course_fit, course_temp=course_temp)
                    sage_pool_fit_mean = float(course_fit.mean().item())
                    if sage_only_cold_or_tail:
                        probs_fn = (
                            _sage_two_expert_candidate_probs
                            if sage_two_expert
                            else _sage_only_cold_or_tail_candidate_probs
                        )
                        probs_override = probs_fn(
                            top_scores,
                            course_fit,
                            sage_gate,
                            candidate_temp=self.cfg.candidate_temp,
                            candidate_epsilon=self.cfg.candidate_epsilon,
                            course_temp=course_temp,
                        )
                        pool_scores = top_scores
                        pool_idx = top_idx
                        pool_k = top_m
                    elif sage_two_expert:
                        probs_override = _sage_two_expert_candidate_probs(
                            sage_pool_scores,
                            course_fit,
                            sage_gate,
                            candidate_temp=self.cfg.candidate_temp,
                            candidate_epsilon=self.cfg.candidate_epsilon,
                            course_temp=course_temp,
                        )
                        pool_scores = sage_pool_scores
                        pool_idx = sage_pool_idx
                    else:
                        pool_scores = sage_pool_scores
                        pool_idx = sage_pool_idx
        else:
            pool_k = top_m
            course_probs = None
            sage_gate = None

        if probs_override is not None:
            probs = probs_override
        else:
            probs = _sampling_probs_from_scores(
                pool_scores,
                temp=self.cfg.candidate_temp,
                epsilon=self.cfg.candidate_epsilon,
            )
        if sage_active and course_probs is not None and sage_gate is not None and probs_override is None:
            probs = (1.0 - sage_gate) * probs + sage_gate * course_probs
        if (
            bool(getattr(self.cfg, "use_cgrc_recon", False))
            and float(getattr(self.cfg, "cgrc_recon_sample_weight", 0.0)) > 0.0
            and pool_idx is not None
            and user_bank_raw is not None
        ):
            with torch.no_grad():
                recon_gate = self._cgrc_recon_active_gate(target_pop, B, n_cols=pool_idx.size(1))
                if bool((recon_gate > 0).any().item()):
                    recon_logits = self._cgrc_recon_candidate_logits(item_emb, pool_idx, user_bank_raw)
                    if recon_logits is not None:
                        if not torch.isfinite(recon_logits).all():
                            recon_logits = torch.nan_to_num(recon_logits, nan=0.0, posinf=0.0, neginf=0.0)
                        recon_probs = F.softmax(
                            recon_logits / float(getattr(self.cfg, "cgrc_recon_temperature", 0.50)),
                            dim=1,
                        )
                        mix_w = recon_gate * float(getattr(self.cfg, "cgrc_recon_sample_weight", 0.0))
                        probs = (1.0 - mix_w) * probs + mix_w * recon_probs
                        cgrc_recon_sample_active_mean = float((recon_gate > 0).float().mean().item())
                        cgrc_recon_sample_score_mean = float((recon_logits * recon_gate).sum().item() / recon_gate.sum().clamp_min(1.0).item())
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        deterministic_coursefit = str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower() == "course_fit"
        if deterministic_coursefit:
            sample_pos = _deterministic_candidate_positions(probs, N_cand)
        else:
            nonzero_per_row = (probs > 0).sum(dim=1)
            replacement = int(nonzero_per_row.min().detach().item()) < N_cand
            sample_pos = torch.multinomial(probs, num_samples=N_cand, replacement=replacement)
        cand_idx = pool_idx.gather(1, sample_pos)
        cand_emb = user_bank_raw[cand_idx].detach()
        topm_unique = max(1, int(pool_idx.unique().numel()))
        selected_unique = int(cand_idx.unique().numel())
        selected_total = max(1, int(cand_idx.numel()))
        dup_rate = 1.0 - (selected_unique / selected_total)
        topm_cov = selected_unique / topm_unique
        stats = {
            "dup_rate": float(dup_rate),
            "topm_coverage": float(topm_cov),
            "sage_active": float(1.0 if sage_active else 0.0),
            "sage_gate": float(sage_gate_mean),
            "sage_tail_active": float(sage_tail_active_mean),
            "sage_pool_fit": float(sage_pool_fit_mean),
            "sage_two_expert": float(1.0 if sage_active and sage_two_expert else 0.0),
            "cgrc_recon_sample_active": float(cgrc_recon_sample_active_mean),
            "cgrc_recon_sample_score": float(cgrc_recon_sample_score_mean),
        }
        return cand_emb, cand_idx, stats

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = self._cold_mask_from_pop(pop)
        effective_cold = self._effective_train_cold_mask(pop, item_idx=i)
        episode_pop = self._target_pop_with_effective_cold(pop, effective_cold)
        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = effective_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)
        original_v2 = _original_usim_v2_enabled() and self.training
        if original_v2:
            episode_rows = effective_cold.nonzero(as_tuple=False).view(-1)
            final_h = z_i_base
            trajectory = {"rewards": []}
            candidate_stats = {
                "steps": 0,
                "v2_active": 1.0,
                "v2_initial_target_l2": 0.0,
                "v2_rollout_delta_l2": 0.0,
            }
            if episode_rows.numel() > 0:
                teacher_item_emb = self._original_v2_teacher_item_emb
                teacher_user_emb = self._original_v2_teacher_user_emb
                if teacher_item_emb is None or teacher_user_emb is None:
                    raise RuntimeError(
                        "USIM_ORIGINAL_V2 requires a pretrained frozen IV teacher; "
                        "set USIM_FB_INIT_CKPT_DIR through the V2 launcher."
                    )
                episode_base = z_i_base.index_select(0, episode_rows)
                episode_item_idx = i.index_select(0, episode_rows)
                episode_target = teacher_item_emb.to(device=self.device).index_select(
                    0, episode_item_idx
                )
                episode_target_pop = (
                    episode_pop.index_select(0, episode_rows)
                    if episode_pop is not None
                    else None
                )
                episode_final, trajectory, candidate_stats = self.run_usim_episode(
                    episode_base,
                    episode_target,
                    user_bank_raw=user_bank_raw,
                    item_idx=episode_item_idx,
                    target_pop=episode_target_pop,
                    user_seen_items=user_seen_items,
                    oracle_user_idx=u.index_select(0, episode_rows),
                    oracle_user_emb=teacher_user_emb.to(device=self.device).index_select(
                        0, u.index_select(0, episode_rows)
                    ),
                )
                refined_episode = self._blend_rl_episode_output(episode_base, episode_final)
                final_h = z_i_base.index_copy(0, episode_rows, refined_episode)
                candidate_stats["v2_active"] = 1.0
                candidate_stats["v2_initial_target_l2"] = float(
                    (episode_base.detach() - episode_target).norm(dim=1).mean().item()
                )
                candidate_stats["v2_rollout_delta_l2"] = float(
                    (refined_episode.detach() - episode_base.detach()).norm(dim=1).mean().item()
                )
        else:
            target_emb = _training_episode_target(
                z_i_base,
                getattr(self.cfg, "rollout_policy", "ppo"),
                getattr(self.cfg, "ppo_loss_weight", 1.0),
            )
            final_h, trajectory, candidate_stats = self.run_usim_episode(
                z_i_base,
                target_emb,
                user_bank_raw=user_bank_raw,
                item_idx=i,
                target_pop=episode_pop,
                user_seen_items=user_seen_items,
            )
            refined_h = self._blend_rl_episode_output(z_i_base, final_h)
            final_h = _apply_refinement_only_to_effective_cold(
                z_i_base,
                refined_h,
                effective_cold,
            )
        pseudo_cold_mask = effective_cold & (~is_cold)
        candidate_stats["pseudo_cold_count"] = int(pseudo_cold_mask.sum().detach().item())
        candidate_stats["pseudo_cold_ratio"] = (
            float(pseudo_cold_mask.float().mean().detach().item()) if pseudo_cold_mask.numel() > 0 else 0.0
        )
        candidate_stats["effective_cold_ratio"] = (
            float(effective_cold.float().mean().detach().item()) if effective_cold.numel() > 0 else 0.0
        )
        candidate_stats["rl_residual_scale"] = float(getattr(self.cfg, "rl_residual_scale", 1.0))
        ppo_loss_raw = self.compute_ppo_loss(trajectory)
        ppo_loss = float(getattr(self.cfg, "ppo_loss_weight", 1.0)) * ppo_loss_raw
        z_u_for_main = z_u_base
        if original_v2 and bool(pseudo_cold_mask.any().item()):
            z_u_for_main = torch.where(
                pseudo_cold_mask.view(-1, 1),
                z_u_base.detach(),
                z_u_base,
            )
        z_u = F.normalize(z_u_for_main, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0), device=self.device)
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        if bool(getattr(self.cfg, "sage_two_expert_score_fusion", False)) and logits.size(0) > 1:
            batch_cand_idx = i.view(1, -1).expand(logits.size(0), -1)
            logits = self.apply_sage_two_expert_score_fusion(
                logits,
                user_ids=u,
                cand_idx=batch_cand_idx,
            )
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp
        false_neg_mask = None
        fn_mask_ratio = 0.0
        if self.training and logits_margin.size(0) > 1:
            false_neg_mask = self._build_batch_false_negative_mask(u, i, user_seen_items, pos_mask)
            if false_neg_mask is not None:
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
                cand_logits = self._mask_false_negative_candidate_logits(cand_logits, cand_idx, false_neg_mask)
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
                known_positive_mask=false_neg_mask,
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
        sage_aux_loss, sage_aux_info = self._compute_sage_aux_loss(
            z_i_base,
            item_idx=i,
            target_pop=pop,
            user_seen_items=user_seen_items,
            user_bank_raw=user_bank_raw,
        )
        cgrc_recon_loss, cgrc_recon_info = self._compute_cgrc_recon_aux_loss(
            u,
            content_e,
            pop,
        )
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
            float(getattr(self.cfg, "sage_aux_weight", 0.0)) * sage_aux_loss +
            float(getattr(self.cfg, "cgrc_recon_aux_weight", 0.0)) * cgrc_recon_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            delta_reg_loss
        )
        candidate_stats["main_loss"] = float(main_loss.detach().item())
        candidate_stats["aux_loss"] = float(aux_loss.detach().item())
        candidate_stats["ppo_loss"] = float(ppo_loss.detach().item())
        candidate_stats["ppo_loss_raw"] = float(ppo_loss_raw.detach().item())
        candidate_stats["prereq_aux_loss"] = float(prereq_aux_loss.detach().item())
        candidate_stats["sage_aux_loss"] = float(sage_aux_loss.detach().item())
        candidate_stats["sage_aux_active_ratio"] = float(sage_aux_info.get("sage_aux_active_ratio", 0.0))
        candidate_stats["sage_aux_pool_fit"] = float(sage_aux_info.get("sage_aux_pool_fit", 0.0))
        candidate_stats["cgrc_recon_loss"] = float(cgrc_recon_loss.detach().item())
        candidate_stats["cgrc_recon_active_ratio"] = float(cgrc_recon_info.get("cgrc_recon_active_ratio", 0.0))
        candidate_stats["cgrc_recon_pos_count"] = int(cgrc_recon_info.get("cgrc_recon_pos_count", 0))
        candidate_stats["delta_reg_loss"] = float(delta_reg_loss.detach().item())
        candidate_stats["total_loss"] = float(total_loss.detach().item())
        candidate_stats["paac_contrast_loss"] = float(paac_contrast_loss.detach().item())
        candidate_stats["paac_align_loss"] = float(paac_align_loss.detach().item())
        candidate_stats["paac_align_pairs"] = int(paac_align_pairs)
        candidate_stats["fn_mask_ratio"] = float(fn_mask_ratio)
        return total_loss, candidate_stats

    def _blend_rl_episode_output(self, z_i_base, final_h):
        residual_scale = float(getattr(self.cfg, "rl_residual_scale", 1.0))
        if residual_scale >= 1.0:
            return final_h
        if residual_scale <= 0.0:
            return z_i_base
        return z_i_base + residual_scale * (final_h - z_i_base)


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
        if getattr(self.cfg, "use_sage_lite", False):
            sage_gate = self._sage_tail_gate(target_pop, batch_size, n_cols=n_cand)
            combined_score = _sage_course_sampling_combined_score(
                retrieval_score,
                fit_norm,
                sage_gate=sage_gate,
                beta=float(self.cfg.feedback_course_sample_beta),
                use_sage_lite=True,
                sage_only_cold_or_tail=bool(getattr(self.cfg, "sage_only_cold_or_tail", False)),
            )
        else:
            combined_score = _sage_course_sampling_combined_score(
                retrieval_score,
                fit_norm,
                beta=float(self.cfg.feedback_course_sample_beta),
                use_sage_lite=False,
            )

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

    def _zero_action_stats(self, action_idx, candidates):
        batch_size = int(action_idx.size(0))
        zero_log_prob = candidates.new_zeros((batch_size,))
        zero_value = candidates.new_zeros((batch_size, 1))
        zero_entropy = candidates.new_zeros((batch_size,))
        return zero_log_prob, zero_value, zero_entropy

    def _select_rollout_action(self, current_h, time_step, candidates, fit_score=None):
        policy = str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower()
        if policy == "ppo":
            return self.agent.get_action_value(current_h, time_step, candidates)

        batch_size = candidates.size(0)
        n_candidates = candidates.size(1)
        if policy == "random":
            action_idx = torch.randint(n_candidates, (batch_size,), device=candidates.device)
        elif policy == "greedy_similarity":
            scores = (
                F.normalize(current_h, dim=1).unsqueeze(1)
                * F.normalize(candidates, dim=2)
            ).sum(dim=2)
            action_idx = torch.argmax(scores, dim=1)
        elif policy == "course_fit" and fit_score is not None:
            action_idx = torch.argmax(fit_score, dim=1)
        elif policy == "course_fit":
            scores = (
                F.normalize(current_h, dim=1).unsqueeze(1)
                * F.normalize(candidates, dim=2)
            ).sum(dim=2)
            action_idx = torch.argmax(scores, dim=1)
        else:
            raise ValueError(f"Unsupported rollout_policy={policy!r}")

        log_prob, value, entropy = self._zero_action_stats(action_idx, candidates)
        return action_idx, log_prob, value, entropy

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
        oracle_user_idx=None,
        oracle_user_emb=None,
    ):
        current_h = init_item_emb.clone()
        original_v2_training = (
            _original_usim_v2_enabled() and self.training and target_emb is not None
        )
        if original_v2_training:
            if oracle_user_idx is None or oracle_user_emb is None:
                raise ValueError(
                    "USIM_ORIGINAL_V2 training requires oracle_user_idx and oracle_user_emb"
                )
            oracle_user_idx = torch.as_tensor(
                oracle_user_idx, dtype=torch.long, device=self.device
            ).view(-1)
            oracle_user_emb = torch.as_tensor(
                oracle_user_emb, dtype=current_h.dtype, device=self.device
            )
            if oracle_user_idx.numel() != current_h.size(0):
                raise ValueError("oracle_user_idx must contain one user per episode row")
            if tuple(oracle_user_emb.shape) != tuple(current_h.shape):
                raise ValueError("oracle_user_emb must match the item-state shape")
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
            "sage_active": 0.0,
            "sage_gate": 0.0,
            "sage_tail_active": 0.0,
            "sage_pool_fit": 0.0,
            "sage_two_expert": 0.0,
            "cgrc_recon_sample_active": 0.0,
            "cgrc_recon_sample_score": 0.0,
            "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0,
            "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
            "target_alpha": 0.0,
            "v2_embedding_reward": 0.0,
            "v2_recommendation_reward": 0.0,
        }

        user_bank_norm = None
        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        elif user_bank_raw is not None and user_bank_norm is None:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)

        selected_user_history = torch.empty(
            (current_h.size(0), 0),
            dtype=torch.long,
            device=self.device,
        )
        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidate_query = (
                target_emb.detach() - current_h
                if original_v2_training
                else current_h
            )
            candidates, cand_user_idx, cand_stats = self.get_candidates(
                candidate_query,
                user_bank_raw=user_bank_raw,
                user_bank_norm=user_bank_norm,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            if original_v2_training:
                # The isolated repair first evaluates the USIM mechanism by
                # itself; CKG sampling is neither a candidate source nor a
                # reward term on this route.
                fit_score = None
            else:
                candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                    current_h,
                    candidates,
                    cand_user_idx,
                    item_idx=item_idx,
                    target_pop=target_pop,
                    user_seen_items=user_seen_items,
                )
            if original_v2_training:
                candidates = candidates.clone()
                cand_user_idx = cand_user_idx.clone()
                candidates[:, 0] = user_bank_raw.index_select(0, oracle_user_idx).detach()
                cand_user_idx[:, 0] = oracle_user_idx
                if candidates.size(1) > 1:
                    random_user_idx = torch.randint(
                        0,
                        int(user_bank_raw.size(0)),
                        (current_h.size(0),),
                        device=self.device,
                    )
                    candidates[:, 1] = user_bank_raw.index_select(0, random_user_idx).detach()
                    cand_user_idx[:, 1] = random_user_idx
            if str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower() == "course_fit":
                fit_score = _exclude_previously_selected_users(
                    fit_score,
                    cand_user_idx,
                    selected_user_history,
                )
            action_idx, log_prob, value, entropy = self._select_rollout_action(
                current_h,
                time_step,
                candidates,
                fit_score=fit_score,
            )

            if cand_stats is not None:
                candidate_stats["dup_rate"] += cand_stats["dup_rate"]
                candidate_stats["topm_coverage"] += cand_stats["topm_coverage"]
                candidate_stats["sage_active"] += float(cand_stats.get("sage_active", 0.0))
                candidate_stats["sage_gate"] += float(cand_stats.get("sage_gate", 0.0))
                candidate_stats["sage_tail_active"] += float(cand_stats.get("sage_tail_active", 0.0))
                candidate_stats["sage_pool_fit"] += float(cand_stats.get("sage_pool_fit", 0.0))
                candidate_stats["sage_two_expert"] += float(cand_stats.get("sage_two_expert", 0.0))
                candidate_stats["cgrc_recon_sample_active"] += float(cand_stats.get("cgrc_recon_sample_active", 0.0))
                candidate_stats["cgrc_recon_sample_score"] += float(cand_stats.get("cgrc_recon_sample_score", 0.0))
                candidate_stats["steps"] += 1
            if fit_score is not None:
                candidate_stats["course_sample_fit"] += _finite_tensor_mean(fit_score)

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
                selected_user_history = torch.cat(
                    [selected_user_history, selected_user_ids.view(-1, 1)],
                    dim=1,
                )

            target_alpha = None
            if target_emb is not None and not original_v2_training:
                target_alpha = self._compute_target_alpha(
                    target_pop=target_pop,
                    step_idx=t,
                    entropy=entropy,
                    num_candidates=candidates.size(1),
                    batch_size=current_h.size(0),
                )
                candidate_stats["target_alpha"] += float(target_alpha.mean().item())
            with torch.enable_grad():
                if original_v2_training:
                    # USIM Eq. (6): an imagined user alone defines the update;
                    # the behavioural oracle is used only to score the update.
                    grad = selected_user.detach()
                else:
                    grad = _batch_invariant_alignment_grad(
                        current_h,
                        selected_user,
                        target_emb=target_emb,
                        target_alpha=target_alpha,
                        reference_batch_size=getattr(self.cfg, "batch_size", current_h.size(0)),
                    )

            if str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower() == "course_fit":
                active_update = _coursefit_active_update_mask(fit_score, action_idx)
                if active_update is not None:
                    grad = grad * active_update.to(dtype=grad.dtype).view(-1, 1)

            step_size = (
                _original_usim_v2_step_size()
                if original_v2_training
                else self.cfg.usim_lr
            )
            current_h = current_h + step_size * grad

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            step_gain_mean = 0.0
            collapse_penalty = 0.0
            if target_emb is not None:
                if original_v2_training:
                    previous_embedding_error = (prev_h - target_emb.detach()).norm(dim=1, keepdim=True)
                    current_embedding_error = (current_h - target_emb.detach()).norm(dim=1, keepdim=True)
                    embedding_reward = previous_embedding_error - current_embedding_error
                    previous_prediction_error = torch.abs(
                        (prev_h * oracle_user_emb.detach()).sum(dim=1, keepdim=True)
                        - (target_emb.detach() * oracle_user_emb.detach()).sum(dim=1, keepdim=True)
                    )
                    current_prediction_error = torch.abs(
                        (current_h * oracle_user_emb.detach()).sum(dim=1, keepdim=True)
                        - (target_emb.detach() * oracle_user_emb.detach()).sum(dim=1, keepdim=True)
                    )
                    recommendation_reward = previous_prediction_error - current_prediction_error
                    step_penalty = float(os.environ.get("USIM_ORIGINAL_V2_STEP_PENALTY", "0.01"))
                    reward = embedding_reward + recommendation_reward - step_penalty
                    step_gain_mean = float(embedding_reward.mean().item())
                    candidate_stats["v2_embedding_reward"] += float(embedding_reward.mean().item())
                    candidate_stats["v2_recommendation_reward"] += float(
                        recommendation_reward.mean().item()
                    )
                else:
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

            if original_v2_training:
                course_terms = {
                    "prereq_gap": torch.zeros_like(reward),
                    "concept_bonus": torch.zeros_like(reward),
                    "difficulty_gap": torch.zeros_like(reward),
                    "redundant": torch.zeros_like(reward),
                }
            else:
                course_terms = self._compute_course_reward_terms(
                    selected_user_ids,
                    item_idx=item_idx,
                    target_pop=target_pop,
                    user_seen_items=user_seen_items,
                )
            if (not original_v2_training) and getattr(self.cfg, "use_course_reward", True):
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
                "sage_active",
                "sage_gate",
                "sage_tail_active",
                "sage_pool_fit",
                "sage_two_expert",
                "cgrc_recon_sample_active",
                "cgrc_recon_sample_score",
                "course_prereq_gap",
                "course_concept_bonus",
                "course_difficulty_gap",
                "course_redundant",
                "target_alpha",
                "v2_embedding_reward",
                "v2_recommendation_reward",
            ]:
                candidate_stats[key] /= candidate_stats["steps"]

        return current_h, trajectory, candidate_stats

    @torch.no_grad()
    def infer_refined_item_vectors(
        self,
        item_idx,
        llm_s=None,
        item_batch=1024,
        force_cold=True,
        user_bank_raw=None,
        user_seen_items=None,
    ):
        """Build reproducible cold-item vectors with the configured rollout policy."""
        item_idx = torch.as_tensor(item_idx, dtype=torch.long, device=self.device).view(-1)
        if item_idx.numel() < 1:
            return torch.empty(
                (0, self.cfg.emb_dim),
                dtype=self.item_id_emb.weight.dtype,
                device=self.device,
            )
        if llm_s is None:
            llm_s = torch.full(
                (item_idx.numel(),), -1.0, dtype=torch.float32, device=self.device
            )
        else:
            llm_s = torch.as_tensor(llm_s, dtype=torch.float32, device=self.device).view(-1)
            if llm_s.numel() != item_idx.numel():
                raise ValueError("llm_s must have the same length as item_idx")

        was_training = self.training
        self.eval()
        outputs = []
        bank = user_bank_raw if user_bank_raw is not None else self._build_user_bank_raw()
        history_context = user_seen_items
        if history_context is None and self.user_seen_index is not None:
            history_context = {}
        eval_seed = int(os.environ.get("USIM_ACTOR_INFERENCE_SEED", "7001"))
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)
                batch_size = max(1, int(item_batch))
                for start in range(0, item_idx.numel(), batch_size):
                    end = min(start + batch_size, item_idx.numel())
                    idx = item_idx[start:end]
                    score = llm_s[start:end]
                    base, _, _ = self.get_item_vector(
                        idx,
                        score,
                        force_cold=force_cold,
                        disable_id_dropout=True,
                    )
                    pop = None
                    if self.item_popularity is not None:
                        pop = self.item_popularity.to(self.device).index_select(0, idx).float()
                    final, _, _ = self.run_usim_episode(
                        base,
                        target_emb=None,
                        user_bank_raw=bank,
                        item_idx=idx,
                        target_pop=pop,
                        user_seen_items=history_context,
                    )
                    outputs.append(self._blend_rl_episode_output(base, final).detach())
        finally:
            self.train(was_training)
        return torch.cat(outputs, dim=0)

    def compute_ppo_loss(self, trajectory):
        if len(trajectory["rewards"]) == 0:
            return next(self.parameters()).sum() * 0.0

        rewards = torch.stack(trajectory["rewards"]).squeeze(-1)
        # PPO compares the current policy against the fixed rollout policy.
        # Keeping either tensor attached makes the shared-policy gradients
        # cancel at the first update (and leaks the old critic into clipping).
        old_log_probs = torch.stack(trajectory["log_probs"]).detach()
        old_values = torch.stack(trajectory["values"]).squeeze(-1).detach()
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

def _write_static_split_artifacts(train_df, val_df, test_df, split_info, cfg):
    return _write_static_split_artifacts_impl(
        train_df, val_df, test_df, split_info, cfg, _feedback_output_path
    )


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
            "original_usim_v2": _original_usim_v2_enabled(),
            "original_usim_v2_step_size": float(
                os.environ.get("USIM_ORIGINAL_V2_STEP_SIZE", "0.05")
            ),
            "original_usim_v2_step_penalty": float(
                os.environ.get("USIM_ORIGINAL_V2_STEP_PENALTY", "0.01")
            ),
            "original_usim_v2_teacher_checkpoint": os.environ.get(
                "USIM_FB_INIT_CKPT_DIR", ""
            ),
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
            "use_sg_urinit": bool(getattr(cfg, "use_sg_urinit", False)),
            "sg_urinit_cluster_k": int(getattr(cfg, "sg_urinit_cluster_k", 32)),
            "sg_urinit_local_weight": float(getattr(cfg, "sg_urinit_local_weight", 0.70)),
            "sg_urinit_global_weight": float(getattr(cfg, "sg_urinit_global_weight", 0.30)),
            "sg_urinit_target_norm": float(getattr(cfg, "sg_urinit_target_norm", 0.0)),
            "sg_urinit_max_iter": int(getattr(cfg, "sg_urinit_max_iter", 20)),
            "sg_urinit_seed": int(getattr(cfg, "sg_urinit_seed", 2025)),
            "sg_urinit_stats": getattr(cfg, "sg_urinit_stats", {"enabled": False}),
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
            "mask_known_pos_neg": bool(getattr(cfg, "mask_known_pos_neg", False)),
            "mask_same_item_neg": bool(getattr(cfg, "mask_same_item_neg", True)),
            "use_paac": bool(cfg.use_paac),
            "use_course_rerank": bool(cfg.use_course_rerank),
            "use_course_reward": bool(cfg.use_course_reward),
            "rl_residual_scale": float(getattr(cfg, "rl_residual_scale", 1.0)),
            "feedback_course_only_cold": bool(cfg.feedback_course_only_cold),
            "feedback_course_prereq_weight": float(cfg.feedback_course_prereq_weight),
            "feedback_course_concept_weight": float(cfg.feedback_course_concept_weight),
            "feedback_course_match_mode": str(getattr(cfg, "feedback_course_match_mode", "mean")),
            "feedback_course_match_topk": int(getattr(cfg, "feedback_course_match_topk", 5)),
            "feedback_course_match_exclude_target": bool(
                getattr(cfg, "feedback_course_match_exclude_target", False)
            ),
            "feedback_course_difficulty_weight": float(cfg.feedback_course_difficulty_weight),
            "feedback_course_redundant_mode": str(cfg.feedback_course_redundant_mode),
            "feedback_course_redundant_weight": float(cfg.feedback_course_redundant_weight),
            "feedback_course_struct_video_min": float(cfg.feedback_course_struct_video_min),
            "feedback_course_struct_chunk": int(getattr(cfg, "feedback_course_struct_chunk", 8192)),
            "feedback_course_term_norm": str(getattr(cfg, "feedback_course_term_norm", "none")),
            "feedback_course_term_norm_clip": float(getattr(cfg, "feedback_course_term_norm_clip", 2.0)),
            "feedback_course_term_norm_eps": float(getattr(cfg, "feedback_course_term_norm_eps", 1e-6)),
            "feedback_course_term_norm_ema_decay": float(getattr(cfg, "feedback_course_term_norm_ema_decay", 0.95)),
            "feedback_course_sample_beta": float(cfg.feedback_course_sample_beta),
            "feedback_course_sample_only_cold": bool(cfg.feedback_course_sample_only_cold),
            "use_sage_lite": bool(getattr(cfg, "use_sage_lite", False)),
            "sage_gate_min": float(getattr(cfg, "sage_gate_min", 0.10)),
            "sage_gate_max": float(getattr(cfg, "sage_gate_max", 0.60)),
            "sage_gate_mode": str(getattr(cfg, "sage_gate_mode", "heuristic")),
            "sage_gate_bucket_count": int(getattr(cfg, "sage_gate_bucket_count", 20)),
            "sage_gate_hidden_dim": int(getattr(cfg, "sage_gate_hidden_dim", 32)),
            "sage_gate_bucket_strategy": str(getattr(cfg, "sage_gate_bucket_strategy", "paper")),
            "sage_pool_topk": int(getattr(cfg, "sage_pool_topk", 64)),
            "sage_course_temp": float(getattr(cfg, "sage_course_temp", 0.20)),
            "sage_only_cold_or_tail": bool(getattr(cfg, "sage_only_cold_or_tail", False)),
            "sage_tail_pop_ratio": float(getattr(cfg, "sage_tail_pop_ratio", 0.10)),
            "sage_use_two_expert": bool(getattr(cfg, "sage_use_two_expert", False)),
            "sage_two_expert_score_fusion": bool(getattr(cfg, "sage_two_expert_score_fusion", False)),
            "use_sage_aux_loss": bool(getattr(cfg, "use_sage_aux_loss", False)),
            "sage_aux_weight": float(getattr(cfg, "sage_aux_weight", 0.02)),
            "sage_aux_pool_topk": int(getattr(cfg, "sage_aux_pool_topk", 64)),
            "sage_aux_course_temp": float(getattr(cfg, "sage_aux_course_temp", 0.20)),
            "sage_aux_retrieval_temp": float(getattr(cfg, "sage_aux_retrieval_temp", 1.0)),
            "sage_aux_only_strict_cold": bool(getattr(cfg, "sage_aux_only_strict_cold", True)),
            "sage_aux_detach_user": bool(getattr(cfg, "sage_aux_detach_user", True)),
            "use_cgrc_recon": bool(getattr(cfg, "use_cgrc_recon", False)),
            "cgrc_recon_aux_weight": float(getattr(cfg, "cgrc_recon_aux_weight", 0.0)),
            "cgrc_recon_sample_weight": float(getattr(cfg, "cgrc_recon_sample_weight", 0.0)),
            "cgrc_recon_pseudo_ratio": float(getattr(cfg, "cgrc_recon_pseudo_ratio", 0.30)),
            "cgrc_recon_topk": int(getattr(cfg, "cgrc_recon_topk", 64)),
            "cgrc_recon_temperature": float(getattr(cfg, "cgrc_recon_temperature", 0.50)),
            "cgrc_recon_only_cold_or_tail": bool(getattr(cfg, "cgrc_recon_only_cold_or_tail", True)),
            "cgrc_recon_tail_pop_ratio": float(getattr(cfg, "cgrc_recon_tail_pop_ratio", 0.10)),
            "cgrc_recon_detach_user": bool(getattr(cfg, "cgrc_recon_detach_user", False)),
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


def _load_init_model_state_from_checkpoint_dir(init_dir):
    init_dir = str(init_dir or "").strip()
    if not init_dir:
        return None, None
    for name in ("finished.pt", "latest.pt"):
        path = os.path.join(init_dir, name)
        if not os.path.exists(path):
            continue
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise KeyError(f"Init checkpoint missing model_state: {path}")
        return path, payload["model_state"]
    return None, None


def _validate_original_v2_teacher_state(model, state_dict):
    """Fail fast when the frozen V2 IV teacher cannot be reconstructed exactly."""
    if not isinstance(state_dict, dict):
        raise RuntimeError("USIM_ORIGINAL_V2 teacher checkpoint model_state must be a dictionary")

    expected_state = model.state_dict()
    required_keys = ["item_id_emb.weight", "user_emb.weight"]
    required_keys.extend(
        key for key in expected_state if key.startswith("user_proj.")
    )
    for key in required_keys:
        if key not in state_dict:
            raise RuntimeError(
                f"USIM_ORIGINAL_V2 teacher checkpoint is missing required state: {key}"
            )
        checkpoint_value = state_dict[key]
        expected_value = expected_state[key]
        if not torch.is_tensor(checkpoint_value):
            raise RuntimeError(
                f"USIM_ORIGINAL_V2 teacher checkpoint has non-tensor state: {key}"
            )
        if tuple(checkpoint_value.shape) != tuple(expected_value.shape):
            raise RuntimeError(
                "USIM_ORIGINAL_V2 teacher checkpoint has incompatible shape for "
                f"{key}: checkpoint={tuple(checkpoint_value.shape)} "
                f"model={tuple(expected_value.shape)}"
            )


def _restore_original_v2_fresh_agent_state(model, fresh_agent_state):
    """Keep the V2 policy/critic independent from the legacy warm checkpoint."""
    model.agent.load_state_dict(fresh_agent_state, strict=True)


def _apply_static_sg_urinit(model, train_df, content_emb, cfg):
    stats = apply_sg_urinit_(model, train_df, content_emb, cfg)
    cfg.sg_urinit_stats = stats
    return stats


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
    original_v2_fresh_agent_state = None
    if _original_usim_v2_enabled():
        original_v2_fresh_agent_state = {
            key: value.detach().clone() for key, value in model.agent.state_dict().items()
        }
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_train_pop)
    sg_urinit_stats = _apply_static_sg_urinit(model, train_df, content_emb, cfg)
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
        f">> SG-URInit: enabled={bool(getattr(cfg, 'use_sg_urinit', False))} | "
        f"initialized={int(sg_urinit_stats.get('initialized_users', 0))}/{cfg.n_users} | "
        f"cluster_k={int(getattr(cfg, 'sg_urinit_cluster_k', 32))} | "
        f"local_w={float(getattr(cfg, 'sg_urinit_local_weight', 0.70)):.2f} | "
        f"global_w={float(getattr(cfg, 'sg_urinit_global_weight', 0.30)):.2f} | "
        f"target_norm={float(sg_urinit_stats.get('target_norm', 0.0)):.4f}"
    )
    print(
        f">> Pseudo-Cold Train: enabled={cfg.use_pseudo_cold_train} | "
        f"mode={cfg.pseudo_cold_mode} | ratio={cfg.pseudo_cold_ratio:.2f} | "
        f"min_pop={cfg.pseudo_cold_min_pop}"
    )
    print(
        f">> Course Match: mode={getattr(cfg, 'feedback_course_match_mode', 'mean')} | "
        f"topk={getattr(cfg, 'feedback_course_match_topk', 5)} | "
        f"exclude_target={getattr(cfg, 'feedback_course_match_exclude_target', False)}"
    )
    print(
        f">> SAGE-lite: enabled={getattr(cfg, 'use_sage_lite', False)} | "
        f"gate_mode={getattr(cfg, 'sage_gate_mode', 'heuristic')} | "
        f"gate=[{getattr(cfg, 'sage_gate_min', 0.10):.2f},{getattr(cfg, 'sage_gate_max', 0.60):.2f}] | "
        f"buckets={getattr(cfg, 'sage_gate_bucket_count', 20)} | "
        f"bucket_strategy={getattr(cfg, 'sage_gate_bucket_strategy', 'paper')} | "
        f"gate_hidden={getattr(cfg, 'sage_gate_hidden_dim', 32)} | "
        f"pool_topk={getattr(cfg, 'sage_pool_topk', 64)} | "
        f"course_temp={getattr(cfg, 'sage_course_temp', 0.20):.2f} | "
        f"only_cold_or_tail={getattr(cfg, 'sage_only_cold_or_tail', False)} | "
        f"tail_pop_ratio={getattr(cfg, 'sage_tail_pop_ratio', 0.10):.3f} | "
        f"two_expert={getattr(cfg, 'sage_use_two_expert', False)} | "
        f"score_fusion={getattr(cfg, 'sage_two_expert_score_fusion', False)}"
    )
    print(
        f">> CGRC Recon: enabled={getattr(cfg, 'use_cgrc_recon', False)} | "
        f"aux_w={getattr(cfg, 'cgrc_recon_aux_weight', 0.0):.4f} | "
        f"sample_w={getattr(cfg, 'cgrc_recon_sample_weight', 0.0):.4f} | "
        f"pseudo_ratio={getattr(cfg, 'cgrc_recon_pseudo_ratio', 0.30):.2f} | "
        f"topk={getattr(cfg, 'cgrc_recon_topk', 64)} | "
        f"temp={getattr(cfg, 'cgrc_recon_temperature', 0.50):.2f} | "
        f"only_cold_or_tail={getattr(cfg, 'cgrc_recon_only_cold_or_tail', True)} | "
        f"tail_pop_ratio={getattr(cfg, 'cgrc_recon_tail_pop_ratio', 0.10):.3f} | "
        f"detach_user={getattr(cfg, 'cgrc_recon_detach_user', False)}"
    )
    print(
        f">> SAGE Aux: enabled={getattr(cfg, 'use_sage_aux_loss', False)} | "
        f"weight={getattr(cfg, 'sage_aux_weight', 0.02):.3f} | "
        f"pool_topk={getattr(cfg, 'sage_aux_pool_topk', 64)} | "
        f"course_temp={getattr(cfg, 'sage_aux_course_temp', 0.20):.2f} | "
        f"retrieval_temp={getattr(cfg, 'sage_aux_retrieval_temp', 1.0):.2f} | "
        f"strict_cold_only={getattr(cfg, 'sage_aux_only_strict_cold', True)} | "
        f"detach_user={getattr(cfg, 'sage_aux_detach_user', True)}"
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
    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback_fast3_content_delta_static.csv")
    static_diag_keys = [
        "MainLoss",
        "AuxLoss",
        "PPOLoss",
        "PrereqAuxLoss",
        "DeltaRegLoss",
        "CourseSampleFit",
        "SageActive",
        "SageGate",
        "SageTailActive",
        "SagePoolFit",
        "SageTwoExpert",
        "SageAuxLoss",
        "SageAuxActive",
        "SageAuxPoolFit",
        "CGRCReconLoss",
        "CGRCReconActive",
        "CGRCReconPos",
        "CGRCReconSampleActive",
        "CGRCReconSampleScore",
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
        "V2InitialTargetL2",
        "V2RolloutDeltaL2",
        "V2EmbeddingReward",
        "V2RecommendationReward",
        "V3EndRate",
        "V3ActiveSteps",
        "V3EmbeddingReward",
        "V3RecommendationReward",
        "V3CourseReward",
        "V3RolloutDeltaL2",
        "V3CourseLogitBiasAbs",
        "V3TrainResidualShare",
        "V3TrainPositiveShare",
        "V3TrainStateShare",
        "V3TrainRandomShare",
        "LIRAFitP25",
        "LIRAFitP50",
        "LIRAFitP75",
        "LIRAUpdateActiveRatio",
        "LIRAStoppedRatio",
        "LIRAStepDisplacementMean",
        "LIRAStepDisplacementMax",
        "LIRATotalDisplacementMean",
        "LIRATotalDisplacementMax",
        "LIRARepeatedUserRate",
    ]
    for key in static_diag_keys:
        history[key] = []

    def _ensure_static_history_schema():
        # Backward-compatible resume: older checkpoints may not have newer
        # diagnostic columns such as SAGE-lite stats.
        n_rows = len(history.get("Epoch", []))
        required_keys = [
            "Epoch",
            "Loss",
            "Val_full_cold_R@10",
            "Val_full_hot_R@10",
            "Val_full_cold_N@10",
            "Val_full_hot_N@10",
        ] + static_diag_keys
        for key in required_keys:
            values = history.get(key)
            if values is None:
                history[key] = [0.0] * n_rows
            elif len(values) < n_rows:
                history[key] = list(values) + [0.0] * (n_rows - len(values))

    def _append_static_history(epoch_num, avg_loss, val_cold_metrics, val_hot_metrics, diag):
        _ensure_static_history_schema()
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
    resume_start_epoch = 0
    ckpt_dir = _feedback_ckpt_dir()
    ckpt_enabled = _feedback_ckpt_enabled()
    auto_resume = _feedback_ckpt_auto_resume()
    force_fresh = _feedback_ckpt_force_fresh()
    resumed_from_ckpt = False

    snapshot_epochs = _feedback_ckpt_snapshot_epochs()

    def _save_static_checkpoint(status, next_epoch, snapshot_name=None, write_latest=True):
        if not ckpt_enabled:
            return
        es_best = {
            "epoch": int(best_epoch),
            "score": float(best_score),
            "score_mode": str(cfg.early_stop_score_mode),
            "average_mode": str(cfg.early_stop_average_mode),
            "k": int(cfg.early_stop_k),
        }
        state = _build_feedback_ckpt_state(
            model,
            optimizer,
            history,
            {},
            {},
            0,
            0,
            {},
            {},
            0,
            0,
            train_seen,
            accumulated_periods=0,
            warmup_periods=0,
            total_periods=1,
            status=status,
            next_period=0,
            current_period=0,
            next_epoch=int(next_epoch),
            es_best=es_best,
            es_best_state=best_state,
            es_best_opt_state=best_opt_state,
            es_no_improve=no_improve,
        )
        state.update(
            {
                "mode": "static",
                "protocol": "static",
                "split_mode": str(split_info.get("split_mode", "")),
                "cold_threshold": int(cfg.cold_threshold),
                "n_epochs_requested": int(cfg.n_epochs),
                "delta_only_applied": bool(delta_only_applied),
            }
        )
        effective_snapshot_name = snapshot_name
        if write_latest and snapshot_name is None and int(next_epoch) in snapshot_epochs:
            effective_snapshot_name = f"epoch_{int(next_epoch):03d}.pt"
        if write_latest:
            _save_feedback_checkpoint(ckpt_dir, state, snapshot_name=effective_snapshot_name)
        elif snapshot_name:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, os.path.join(ckpt_dir, snapshot_name))
        if effective_snapshot_name and effective_snapshot_name != snapshot_name:
            print(f"  [STATIC-CHECKPOINT] Saved snapshot {effective_snapshot_name}")

    print(
        f">> Static Checkpoint: enabled={ckpt_enabled} | resume={auto_resume} | "
        f"force_fresh={force_fresh} | save_opt={_feedback_ckpt_save_optimizer_state()} | dir={ckpt_dir}"
    )

    def _load_static_model_state_compat(state_dict, label):
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            print(
                f">> Static Resume: {label} loaded with compatibility mode | "
                f"missing={missing} | unexpected={unexpected}"
            )

    if ckpt_enabled and auto_resume and not force_fresh:
        resume_state = _load_feedback_checkpoint(ckpt_dir)
        if resume_state is None:
            print(">> Static Resume: no checkpoint found; starting from epoch 1.")
        elif resume_state.get("mode") != "static":
            print(">> Static Resume: latest checkpoint is not static; ignoring it.")
        else:
            _load_static_model_state_compat(resume_state["model_state"], "model_state")
            opt_state = resume_state.get("optimizer_state")
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                    _optimizer_state_to_device(optimizer, device)
                except ValueError as exc:
                    print(f">> Static Resume: skip optimizer state due to parameter-group mismatch: {exc}")
            history = resume_state.get("history") or history
            _ensure_static_history_schema()
            es_best = resume_state.get("es_best") or {}
            best_epoch = int(es_best.get("epoch", 0) or 0)
            best_score = float(es_best.get("score", -1e9))
            if best_score != best_score:
                best_score = -1e9
            best_state = resume_state.get("es_best_state")
            best_opt_state = resume_state.get("es_best_opt_state")
            no_improve = int(resume_state.get("es_no_improve", 0) or 0)
            delta_only_applied = bool(resume_state.get("delta_only_applied", False))
            resume_start_epoch = int(resume_state.get("next_epoch", 0) or 0)
            resume_start_epoch = max(0, min(resume_start_epoch, int(cfg.n_epochs)))
            resumed_from_ckpt = True
            print(
                f">> Static Resume: status={resume_state.get('status')} | "
                f"next_epoch={resume_start_epoch + 1}/{cfg.n_epochs} | "
                f"best_epoch={best_epoch} | best_score={best_score:.4f}"
            )

    init_path = None
    init_model_state = None
    if not resumed_from_ckpt:
        init_ckpt_dir = os.environ.get("USIM_FB_INIT_CKPT_DIR", "").strip()
        init_path, init_model_state = _load_init_model_state_from_checkpoint_dir(init_ckpt_dir)
        if init_model_state is not None:
            if _original_usim_v2_enabled():
                _validate_original_v2_teacher_state(model, init_model_state)
            _load_static_model_state_compat(init_model_state, "init_model_state")
            if _original_usim_v2_enabled():
                _restore_original_v2_fresh_agent_state(model, original_v2_fresh_agent_state)
                print(">> USIM V2 Policy: restored fresh actor/critic after warm-teacher loading")
            print(f">> Static Init: loaded model_state from {init_path}")
        elif init_ckpt_dir:
            print(f">> Static Init: no finished/latest checkpoint found in {init_ckpt_dir}; starting from scratch.")

    if _original_usim_v2_enabled():
        if resumed_from_ckpt:
            raise RuntimeError(
                "USIM_ORIGINAL_V2 does not support resume because its frozen IV teacher "
                "must be reconstructed from the explicit initialization checkpoint."
            )
        if init_model_state is None:
            raise RuntimeError(
                "USIM_ORIGINAL_V2 requires USIM_FB_INIT_CKPT_DIR with a finished.pt or latest.pt "
                "from the matching warm-teacher run."
            )
        model.initialize_original_v2_teacher_()
        print(f">> USIM V2 Teacher: frozen IV user/item space loaded from {init_path}")

    print(
        f"\n>>> Start STATIC train/eval | target_split={split_info['train_ratio']:.2f}/"
        f"{split_info['val_ratio']:.2f}/{split_info['test_ratio']:.2f} | "
        f"actual_split={split_info['actual_train_ratio']:.2f}/"
        f"{split_info['actual_val_ratio']:.2f}/{split_info['actual_test_ratio']:.2f} | "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    for epoch in range(resume_start_epoch, cfg.n_epochs):
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
        sage_active_sum = 0.0
        sage_gate_sum = 0.0
        sage_tail_active_sum = 0.0
        sage_pool_fit_sum = 0.0
        sage_two_expert_sum = 0.0
        sage_aux_loss_sum = 0.0
        sage_aux_active_sum = 0.0
        sage_aux_pool_fit_sum = 0.0
        cgrc_recon_loss_sum = 0.0
        cgrc_recon_active_sum = 0.0
        cgrc_recon_pos_sum = 0
        cgrc_recon_sample_active_sum = 0.0
        cgrc_recon_sample_score_sum = 0.0
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
        v2_initial_target_l2_sum = 0.0
        v2_rollout_delta_l2_sum = 0.0
        v2_embedding_reward_sum = 0.0
        v2_recommendation_reward_sum = 0.0
        v3_end_rate_sum = 0.0
        v3_active_steps_sum = 0.0
        v3_embedding_reward_sum = 0.0
        v3_recommendation_reward_sum = 0.0
        v3_course_reward_sum = 0.0
        v3_rollout_delta_l2_sum = 0.0
        v3_course_logit_bias_abs_sum = 0.0
        v3_train_residual_share_sum = 0.0
        v3_train_positive_share_sum = 0.0
        v3_train_state_share_sum = 0.0
        v3_train_random_share_sum = 0.0
        lira_fit_p25_sum = 0.0
        lira_fit_p50_sum = 0.0
        lira_fit_p75_sum = 0.0
        lira_active_sum = 0.0
        lira_stopped_sum = 0.0
        lira_step_mean_sum = 0.0
        lira_step_max_sum = 0.0
        lira_total_mean_sum = 0.0
        lira_total_max_sum = 0.0
        lira_repeat_sum = 0.0
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
                sage_active_sum += float(cand_info.get("sage_active", 0.0))
                sage_gate_sum += float(cand_info.get("sage_gate", 0.0))
                sage_tail_active_sum += float(cand_info.get("sage_tail_active", 0.0))
                sage_pool_fit_sum += float(cand_info.get("sage_pool_fit", 0.0))
                sage_two_expert_sum += float(cand_info.get("sage_two_expert", 0.0))
                sage_aux_loss_sum += float(cand_info.get("sage_aux_loss", 0.0))
                sage_aux_active_sum += float(cand_info.get("sage_aux_active_ratio", 0.0))
                sage_aux_pool_fit_sum += float(cand_info.get("sage_aux_pool_fit", 0.0))
                cgrc_recon_loss_sum += float(cand_info.get("cgrc_recon_loss", 0.0))
                cgrc_recon_active_sum += float(cand_info.get("cgrc_recon_active_ratio", 0.0))
                cgrc_recon_pos_sum += int(cand_info.get("cgrc_recon_pos_count", 0))
                cgrc_recon_sample_active_sum += float(cand_info.get("cgrc_recon_sample_active", 0.0))
                cgrc_recon_sample_score_sum += float(cand_info.get("cgrc_recon_sample_score", 0.0))
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
                v2_initial_target_l2_sum += float(cand_info.get("v2_initial_target_l2", 0.0))
                v2_rollout_delta_l2_sum += float(cand_info.get("v2_rollout_delta_l2", 0.0))
                v2_embedding_reward_sum += float(cand_info.get("v2_embedding_reward", 0.0))
                v2_recommendation_reward_sum += float(
                    cand_info.get("v2_recommendation_reward", 0.0)
                )
                v3_end_rate_sum += float(cand_info.get("v3_end_rate", 0.0))
                v3_active_steps_sum += float(cand_info.get("v3_active_steps", 0.0))
                v3_embedding_reward_sum += float(cand_info.get("v3_embedding_reward", 0.0))
                v3_recommendation_reward_sum += float(
                    cand_info.get("v3_recommendation_reward", 0.0)
                )
                v3_course_reward_sum += float(cand_info.get("v3_course_reward", 0.0))
                v3_rollout_delta_l2_sum += float(cand_info.get("v3_rollout_delta_l2", 0.0))
                v3_course_logit_bias_abs_sum += float(
                    cand_info.get("v3_course_logit_bias_abs", 0.0)
                )
                v3_train_residual_share_sum += float(
                    cand_info.get("v3_train_residual_share", 0.0)
                )
                v3_train_positive_share_sum += float(
                    cand_info.get("v3_train_positive_share", 0.0)
                )
                v3_train_state_share_sum += float(
                    cand_info.get("v3_train_state_share", 0.0)
                )
                v3_train_random_share_sum += float(
                    cand_info.get("v3_train_random_share", 0.0)
                )
                lira_fit_p25_sum += float(cand_info.get("fit_p25", 0.0))
                lira_fit_p50_sum += float(cand_info.get("fit_p50", 0.0))
                lira_fit_p75_sum += float(cand_info.get("fit_p75", 0.0))
                lira_active_sum += float(cand_info.get("update_active_ratio", 0.0))
                lira_stopped_sum += float(cand_info.get("stopped_ratio", 0.0))
                lira_step_mean_sum += float(cand_info.get("step_displacement_mean", 0.0))
                lira_step_max_sum += float(cand_info.get("step_displacement_max", 0.0))
                lira_total_mean_sum += float(cand_info.get("total_displacement_mean", 0.0))
                lira_total_max_sum += float(cand_info.get("total_displacement_max", 0.0))
                lira_repeat_sum += float(cand_info.get("repeated_user_rate", 0.0))
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
                "SageActive": sage_active_sum / pseudo_info_batches,
                "SageGate": sage_gate_sum / pseudo_info_batches,
                "SageTailActive": sage_tail_active_sum / pseudo_info_batches,
                "SagePoolFit": sage_pool_fit_sum / pseudo_info_batches,
                "SageTwoExpert": sage_two_expert_sum / pseudo_info_batches,
                "SageAuxLoss": sage_aux_loss_sum / pseudo_info_batches,
                "SageAuxActive": sage_aux_active_sum / pseudo_info_batches,
                "SageAuxPoolFit": sage_aux_pool_fit_sum / pseudo_info_batches,
                "CGRCReconLoss": cgrc_recon_loss_sum / pseudo_info_batches,
                "CGRCReconActive": cgrc_recon_active_sum / pseudo_info_batches,
                "CGRCReconPos": cgrc_recon_pos_sum,
                "CGRCReconSampleActive": cgrc_recon_sample_active_sum / pseudo_info_batches,
                "CGRCReconSampleScore": cgrc_recon_sample_score_sum / pseudo_info_batches,
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
                "V2InitialTargetL2": v2_initial_target_l2_sum / pseudo_info_batches,
                "V2RolloutDeltaL2": v2_rollout_delta_l2_sum / pseudo_info_batches,
                "V2EmbeddingReward": v2_embedding_reward_sum / pseudo_info_batches,
                "V2RecommendationReward": v2_recommendation_reward_sum / pseudo_info_batches,
                "V3EndRate": v3_end_rate_sum / pseudo_info_batches,
                "V3ActiveSteps": v3_active_steps_sum / pseudo_info_batches,
                "V3EmbeddingReward": v3_embedding_reward_sum / pseudo_info_batches,
                "V3RecommendationReward": v3_recommendation_reward_sum / pseudo_info_batches,
                "V3CourseReward": v3_course_reward_sum / pseudo_info_batches,
                "V3RolloutDeltaL2": v3_rollout_delta_l2_sum / pseudo_info_batches,
                "V3CourseLogitBiasAbs": v3_course_logit_bias_abs_sum / pseudo_info_batches,
                "V3TrainResidualShare": v3_train_residual_share_sum / pseudo_info_batches,
                "V3TrainPositiveShare": v3_train_positive_share_sum / pseudo_info_batches,
                "V3TrainStateShare": v3_train_state_share_sum / pseudo_info_batches,
                "V3TrainRandomShare": v3_train_random_share_sum / pseudo_info_batches,
                "LIRAFitP25": lira_fit_p25_sum / pseudo_info_batches,
                "LIRAFitP50": lira_fit_p50_sum / pseudo_info_batches,
                "LIRAFitP75": lira_fit_p75_sum / pseudo_info_batches,
                "LIRAUpdateActiveRatio": lira_active_sum / pseudo_info_batches,
                "LIRAStoppedRatio": lira_stopped_sum / pseudo_info_batches,
                "LIRAStepDisplacementMean": lira_step_mean_sum / pseudo_info_batches,
                "LIRAStepDisplacementMax": lira_step_max_sum / pseudo_info_batches,
                "LIRATotalDisplacementMean": lira_total_mean_sum / pseudo_info_batches,
                "LIRATotalDisplacementMax": lira_total_max_sum / pseudo_info_batches,
                "LIRARepeatedUserRate": lira_repeat_sum / pseudo_info_batches,
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
                f" | V2[target_l2={epoch_diag['V2InitialTargetL2']:.4f}, "
                f"delta_l2={epoch_diag['V2RolloutDeltaL2']:.4f}, "
                f"emb_r={epoch_diag['V2EmbeddingReward']:.4f}, "
                f"rec_r={epoch_diag['V2RecommendationReward']:.4f}]"
                f" | V3[end_rate={epoch_diag['V3EndRate']:.2%}, "
                f"active_steps={epoch_diag['V3ActiveSteps']:.3f}, "
                f"delta_l2={epoch_diag['V3RolloutDeltaL2']:.4f}, "
                f"emb_r={epoch_diag['V3EmbeddingReward']:.4f}, "
                f"rec_r={epoch_diag['V3RecommendationReward']:.4f}, "
                f"ckg_r={epoch_diag['V3CourseReward']:.4f}, "
                f"bias={epoch_diag['V3CourseLogitBiasAbs']:.4f}, "
                f"mix={epoch_diag['V3TrainResidualShare']:.2%}/"
                f"{epoch_diag['V3TrainPositiveShare']:.2%}/"
                f"{epoch_diag['V3TrainStateShare']:.2%}/"
                f"{epoch_diag['V3TrainRandomShare']:.2%}]"
                f" | SAGE[active={epoch_diag['SageActive']:.2f}, "
                f"gate={epoch_diag['SageGate']:.3f}, "
                f"tail_active={epoch_diag['SageTailActive']:.2f}, "
                f"pool_fit={epoch_diag['SagePoolFit']:.4f}, "
                f"two_expert={epoch_diag['SageTwoExpert']:.2f}, "
                f"aux={epoch_diag['SageAuxLoss']:.4f}, "
                f"aux_active={epoch_diag['SageAuxActive']:.2f}]"
                f" | CGRCRecon[loss={epoch_diag['CGRCReconLoss']:.4f}, "
                f"active={epoch_diag['CGRCReconActive']:.2f}, "
                f"pos={int(epoch_diag['CGRCReconPos'])}, "
                f"sample_active={epoch_diag['CGRCReconSampleActive']:.2f}, "
                f"sample_score={epoch_diag['CGRCReconSampleScore']:.4f}]"
                f" | LIRA[active={epoch_diag['LIRAUpdateActiveRatio']:.2%}, "
                f"stopped={epoch_diag['LIRAStoppedRatio']:.2%}, "
                f"fit_p50={epoch_diag['LIRAFitP50']:.4f}, "
                f"step={epoch_diag['LIRAStepDisplacementMean']:.4f}/"
                f"{epoch_diag['LIRAStepDisplacementMax']:.4f}, "
                f"total={epoch_diag['LIRATotalDisplacementMean']:.4f}/"
                f"{epoch_diag['LIRATotalDisplacementMax']:.4f}, "
                f"repeat={epoch_diag['LIRARepeatedUserRate']:.2%}]"
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
                    _save_static_checkpoint("early_stopped", epoch + 1)
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch + 1

        _append_static_history(epoch + 1, avg_loss, val_cold_metrics, val_hot_metrics, epoch_diag)
        _save_static_checkpoint("running", epoch + 1)

    if best_state is not None:
        _load_static_model_state_compat(best_state, "best_state")
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

    if bool(getattr(cfg, "validation_only", False)):
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        last_completed_epoch = int(history["Epoch"][-1]) if history.get("Epoch") else int(resume_start_epoch)
        _save_static_checkpoint(
            "validation_finished",
            last_completed_epoch,
            snapshot_name="validation_finished.pt",
            write_latest=False,
        )
        print(f"  [STATIC-VALIDATION-ONLY] Saved {metrics_path}; final test was not evaluated.")
        return

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
        export_item_metrics_path=_feedback_output_path(
            "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
        ),
        export_topk_path=os.environ.get("P1_TOPK_EXPORT_PATH", "").strip() or None,
        export_topk_k=int(os.environ.get("P1_TOPK_EXPORT_K", "20")),
        export_topk_metadata={
            "model": os.environ.get("P1_TOPK_EXPORT_MODEL", "ckg_rl"),
            "seed": int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025"))),
        },
    )
    full_hot_item_macro, full_hot_item_macro_count = evaluate_usim(
        model, test_loader, device, llm_scores, k_list=k_list,
        n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=test_item_vecs,
        average_mode="item_macro",
        export_item_metrics_path=_feedback_output_path(
            "per_item_full_hot_usim_feedback_fast3_content_delta_static.csv"
        ),
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
            "per_item_full_cold": _feedback_output_path(
                "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
            ),
            "per_item_full_hot": _feedback_output_path(
                "per_item_full_hot_usim_feedback_fast3_content_delta_static.csv"
            ),
        }
    )
    _write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df)
    last_completed_epoch = int(history["Epoch"][-1]) if history.get("Epoch") else int(resume_start_epoch)
    _save_static_checkpoint("finished", last_completed_epoch, snapshot_name="finished.pt", write_latest=False)
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
        f"adv_norm={cfg.ppo_adv_norm} | loss_weight={getattr(cfg, 'ppo_loss_weight', 1.0):.2f} | "
        f"rollout_policy={getattr(cfg, 'rollout_policy', 'ppo')}"
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
    print(
        f">> False Negative Mask: known_pos={cfg.mask_known_pos_neg} | "
        f"same_item={getattr(cfg, 'mask_same_item_neg', True)}"
    )
    print(
        f">> Course Soft Rerank: enabled={cfg.feedback_course_sample_soft} | "
        f"beta={cfg.feedback_course_sample_beta:.2f} | topL={cfg.feedback_course_sample_top_l}"
    )
    print(
        f">> SAGE-lite: enabled={getattr(cfg, 'use_sage_lite', False)} | "
        f"gate_mode={getattr(cfg, 'sage_gate_mode', 'heuristic')} | "
        f"gate=[{getattr(cfg, 'sage_gate_min', 0.10):.2f},{getattr(cfg, 'sage_gate_max', 0.60):.2f}] | "
        f"buckets={getattr(cfg, 'sage_gate_bucket_count', 20)} | "
        f"bucket_strategy={getattr(cfg, 'sage_gate_bucket_strategy', 'paper')} | "
        f"gate_hidden={getattr(cfg, 'sage_gate_hidden_dim', 32)} | "
        f"pool_topk={getattr(cfg, 'sage_pool_topk', 64)} | "
        f"course_temp={getattr(cfg, 'sage_course_temp', 0.20):.2f} | "
        f"only_cold_or_tail={getattr(cfg, 'sage_only_cold_or_tail', False)} | "
        f"tail_pop_ratio={getattr(cfg, 'sage_tail_pop_ratio', 0.10):.3f} | "
        f"two_expert={getattr(cfg, 'sage_use_two_expert', False)} | "
        f"score_fusion={getattr(cfg, 'sage_two_expert_score_fusion', False)}"
    )
    print(
        f">> Course Feedback: redundant_mode={cfg.feedback_course_redundant_mode} | "
        f"match={getattr(cfg, 'feedback_course_match_mode', 'mean')}@{getattr(cfg, 'feedback_course_match_topk', 5)} "
        f"exclude_target={getattr(cfg, 'feedback_course_match_exclude_target', False)} | "
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
