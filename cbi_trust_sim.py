"""Isolated CBI-constrained simulator components."""

from __future__ import annotations

import math
import weakref

import torch
import torch.nn.functional as F

from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM
from evaluate_cbi_all_refined_seed2025 import (
    build_all_refined_item_bank,
    cached_bank_positive_vectors,
)


_EVAL_BANKS = weakref.WeakKeyDictionary()


def project_to_content_cone(
    state: torch.Tensor,
    content_anchor: torch.Tensor,
    cosine_floor: float,
    eps: float = 1e-8,
):
    """Project normalized states into a cosine cone around content anchors."""
    floor = float(cosine_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("cosine_floor must be in [0, 1]")

    anchor = F.normalize(content_anchor, dim=1)
    unit_state = F.normalize(state, dim=1)
    cosine = (unit_state * anchor).sum(dim=1, keepdim=True)
    outside = cosine < floor
    orthogonal = unit_state - cosine * anchor
    orth_norm = orthogonal.norm(dim=1, keepdim=True)
    orth_unit = orthogonal / orth_norm.clamp_min(eps)
    boundary = floor * anchor + math.sqrt(max(0.0, 1.0 - floor**2)) * orth_unit
    degenerate = outside & (orth_norm <= eps)
    projected = torch.where(outside, boundary, unit_state)
    projected = torch.where(degenerate, anchor, projected)
    projected = F.normalize(projected, dim=1)
    final_cosine = (projected * anchor).sum(dim=1)
    return projected, {
        "projected_count": int(outside.sum().item()),
        "projected_ratio": float(outside.float().mean().item()),
        "min_cosine": float(final_cosine.min().item()),
        "mean_cosine": float(final_cosine.mean().item()),
    }


class CBITrustFast3FeedbackUSIM(Fast3FeedbackUSIM):
    """FAST3 simulator whose complete trajectory stays near frozen content."""

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
        deterministic=False,
    ):
        del target_emb
        initial_cbi_anchor = F.normalize(init_item_emb.detach(), dim=1)
        effective_target = initial_cbi_anchor
        if item_idx is None:
            content_anchor = initial_cbi_anchor
        else:
            content_anchor = F.normalize(self._content_base_embedding(item_idx).detach(), dim=1)
        self.last_effective_target = effective_target

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
            "trust_steps": 0,
            "trust_projected_ratio": 0.0,
            "trust_min_cosine": 1.0,
            "trust_mean_cosine": 0.0,
        }

        user_bank_norm = None
        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        elif user_bank_raw is not None:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)

        cosine_floor = float(
            getattr(self.cfg, "cbi_trust_cosine_floor", math.sqrt(1.0 - 0.5**2))
        )
        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, cand_user_idx, cand_stats = self.get_candidates(
                current_h,
                user_bank_raw=user_bank_raw,
                user_bank_norm=user_bank_norm,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                current_h,
                candidates,
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            action_idx, log_prob, value, entropy = self._select_rollout_action(
                current_h,
                time_step,
                candidates,
                fit_score=fit_score,
                deterministic=deterministic,
            )

            if cand_stats is not None:
                candidate_stats["dup_rate"] += cand_stats["dup_rate"]
                candidate_stats["topm_coverage"] += cand_stats["topm_coverage"]
                candidate_stats["sage_active"] += float(cand_stats.get("sage_active", 0.0))
                candidate_stats["sage_gate"] += float(cand_stats.get("sage_gate", 0.0))
                candidate_stats["sage_tail_active"] += float(cand_stats.get("sage_tail_active", 0.0))
                candidate_stats["sage_pool_fit"] += float(cand_stats.get("sage_pool_fit", 0.0))
                candidate_stats["sage_two_expert"] += float(cand_stats.get("sage_two_expert", 0.0))
                candidate_stats["cgrc_recon_sample_active"] += float(
                    cand_stats.get("cgrc_recon_sample_active", 0.0)
                )
                candidate_stats["cgrc_recon_sample_score"] += float(
                    cand_stats.get("cgrc_recon_sample_score", 0.0)
                )
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
                target_align = (h_detached * effective_target).sum(dim=1, keepdim=True)
                target_alpha = self._compute_target_alpha(
                    target_pop=target_pop,
                    step_idx=t,
                    entropy=entropy,
                    num_candidates=candidates.size(1),
                    batch_size=current_h.size(0),
                )
                candidate_stats["target_alpha"] += float(target_alpha.mean().item())
                score = (((1.0 - target_alpha) * user_align) + (target_alpha * target_align)).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad
            current_h, trust_stats = project_to_content_cone(
                current_h,
                content_anchor,
                cosine_floor=cosine_floor,
            )
            candidate_stats["trust_steps"] += 1
            candidate_stats["trust_projected_ratio"] += trust_stats["projected_ratio"]
            candidate_stats["trust_min_cosine"] = min(
                candidate_stats["trust_min_cosine"], trust_stats["min_cosine"]
            )
            candidate_stats["trust_mean_cosine"] += trust_stats["mean_cosine"]

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            prev_dist = F.mse_loss(prev_h, effective_target, reduction="none").mean(dim=1, keepdim=True)
            new_dist = F.mse_loss(current_h, effective_target, reduction="none").mean(dim=1, keepdim=True)
            terminal_reward = -new_dist * float(self.cfg.reward_terminal_weight)
            step_gain = (prev_dist - new_dist).clamp(
                min=-float(self.cfg.reward_gain_clip),
                max=float(self.cfg.reward_gain_clip),
            )
            reward = terminal_reward + float(self.cfg.reward_gain_weight) * step_gain
            step_gain_mean = float(step_gain.mean().item())
            collapse_penalty = 0.0
            if cand_stats is not None:
                collapse_penalty = float(self.cfg.reward_dup_penalty_weight) * float(cand_stats["dup_rate"])
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
            ]:
                candidate_stats[key] /= candidate_stats["steps"]
        if candidate_stats["trust_steps"] > 0:
            candidate_stats["trust_projected_ratio"] /= candidate_stats["trust_steps"]
            candidate_stats["trust_mean_cosine"] /= candidate_stats["trust_steps"]

        return current_h, trajectory, candidate_stats


def trust_build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    """Build one constrained, all-refined bank for every evaluation split."""
    bank, stats = build_all_refined_item_bank(
        model,
        device,
        llm_scores=llm_scores,
        item_batch=item_batch,
    )
    _EVAL_BANKS[model] = bank
    model.last_trust_bank_stats = stats
    return {"cold": bank, "hot": bank, "all": bank}


def trust_build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    """Index positive vectors from the same bank used for full ranking."""
    del llm_s, pop_sel, eval_type
    if model not in _EVAL_BANKS:
        raise RuntimeError("all-refined evaluation bank must be built before positive lookup")
    return cached_bank_positive_vectors(_EVAL_BANKS[model], item_idx)


def install_trust_eval_adapter(protocol_module, eval_module):
    """Install all-refined evaluation hooks only in the current Python process."""
    eval_module.build_eval_item_vecs = trust_build_eval_item_vecs
    eval_module.build_eval_pos_item_vecs = trust_build_eval_pos_item_vecs
    protocol_module.build_eval_item_vecs = trust_build_eval_item_vecs
    protocol_module.build_eval_pos_item_vecs = trust_build_eval_pos_item_vecs
