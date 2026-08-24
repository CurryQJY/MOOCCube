"""LIRA: leakage-controlled learner-guided cold-item refinement.

This is an isolated successor model.  It reuses the recovered data loading,
base recommender, checkpointing, and evaluator, but does not inherit the
paper claim or experiment identity of the legacy PPO Full.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


METHOD_NAME = "learner_guided_cold_refinement"
USIM_STATIC_DELEGATE_ENTRYPOINT = True
LegacyFast3FeedbackUSIM = legacy.Fast3FeedbackUSIM


def learner_fit(
    similarity: torch.Tensor,
    concept: torch.Tensor,
    prereq_gap: torch.Tensor,
    difficulty_gap: torch.Tensor,
    redundancy: torch.Tensor,
    concept_weight: float = 0.25,
    prereq_beta: float = 1.0,
    difficulty_beta: float = 1.0,
) -> torch.Tensor:
    """Return a finite non-negative learner/course compatibility gate."""
    base = (
        0.5 * (similarity.clamp(-1.0, 1.0) + 1.0)
        + float(concept_weight) * concept.clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    fit = (
        base
        * torch.exp(-float(prereq_beta) * prereq_gap.clamp(0.0, 1.0))
        * torch.exp(-float(difficulty_beta) * difficulty_gap.clamp(0.0, 1.0))
        * (1.0 - redundancy.clamp(0.0, 1.0))
    )
    return torch.nan_to_num(fit, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0
    return float(torch.quantile(finite.float(), q).item())


def bounded_learner_refinement(
    initial_h: torch.Tensor,
    candidate_vectors: torch.Tensor,
    candidate_user_ids: torch.Tensor,
    candidate_fit: torch.Tensor,
    effective_cold: torch.Tensor,
    steps: int = 3,
    lr: float = 0.10,
    min_fit: float = 0.05,
    step_cap: float = 0.05,
    total_cap: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deterministically refine cold rows with unique learners and norm caps."""
    if candidate_vectors.ndim != 3:
        raise ValueError("candidate_vectors must have shape [batch, candidates, dim]")
    if candidate_user_ids.shape != candidate_fit.shape:
        raise ValueError("candidate_user_ids and candidate_fit must have identical shape")
    if candidate_vectors.shape[:2] != candidate_fit.shape:
        raise ValueError("candidate tensors must agree on batch and candidate dimensions")
    if not torch.isfinite(candidate_fit).all():
        raise RuntimeError("learner fit contains a non-finite value")

    initial = initial_h
    current = initial_h.clone()
    cold = effective_cold.to(device=current.device, dtype=torch.bool).view(-1)
    active = cold.clone()
    used = torch.zeros_like(candidate_fit, dtype=torch.bool)
    selected_per_row: list[list[int]] = [[] for _ in range(current.size(0))]
    selected_fit_values: list[torch.Tensor] = []
    step_norm_values: list[torch.Tensor] = []
    active_events = 0
    possible_events = max(1, int(cold.sum().item()) * max(0, int(steps)))

    for _ in range(max(0, int(steps))):
        available_fit = candidate_fit.masked_fill(used, float("-inf"))
        best_fit, action = available_fit.max(dim=1)
        step_active = active & torch.isfinite(best_fit) & (best_fit > float(min_fit))
        active = active & step_active
        if not bool(step_active.any().item()):
            break

        rows = torch.arange(current.size(0), device=current.device)
        selected = candidate_vectors[rows, action]
        direction = F.normalize(selected - current, dim=1, eps=1e-12)
        requested_norm = (float(lr) * best_fit.clamp_min(0.0)).clamp(max=float(step_cap))
        step_delta = direction * requested_norm.view(-1, 1)
        step_delta = step_delta * step_active.to(step_delta.dtype).view(-1, 1)

        proposed = current + step_delta
        total_delta = proposed - initial
        total_norm = torch.linalg.vector_norm(total_delta, dim=1, keepdim=True)
        total_scale = (float(total_cap) / total_norm.clamp_min(1e-12)).clamp(max=1.0)
        current = initial + total_delta * total_scale
        current[~cold] = initial[~cold]

        chosen_rows = step_active.nonzero(as_tuple=False).view(-1)
        used[chosen_rows, action[chosen_rows]] = True
        selected_fit_values.append(best_fit[step_active])
        step_norm_values.append(torch.linalg.vector_norm(step_delta[step_active], dim=1))
        active_events += int(step_active.sum().item())
        for row in chosen_rows.detach().cpu().tolist():
            selected_per_row[row].append(int(candidate_user_ids[row, action[row]].item()))

    total_displacement = torch.linalg.vector_norm(current - initial, dim=1)
    if total_displacement.max().item() > float(total_cap) + 1e-6:
        raise RuntimeError("total displacement cap was violated")
    if not torch.equal(current[~cold], initial[~cold]):
        raise RuntimeError("warm rows changed during cold-only refinement")

    flat_selected = [uid for row in selected_per_row for uid in row]
    repeats = sum(len(row) - len(set(row)) for row in selected_per_row)
    selected_count = max(1, len(flat_selected))
    fit_values = torch.cat(selected_fit_values) if selected_fit_values else current.new_empty((0,))
    step_values = torch.cat(step_norm_values) if step_norm_values else current.new_empty((0,))
    diagnostics = {
        "fit_mean": float(fit_values.mean().item()) if fit_values.numel() else 0.0,
        "fit_p25": _quantile(fit_values, 0.25),
        "fit_p50": _quantile(fit_values, 0.50),
        "fit_p75": _quantile(fit_values, 0.75),
        "update_active_ratio": float(active_events / possible_events),
        "stopped_ratio": float((cold & ~active).sum().item() / max(1, int(cold.sum().item()))),
        "step_displacement_mean": float(step_values.mean().item()) if step_values.numel() else 0.0,
        "step_displacement_max": float(step_values.max().item()) if step_values.numel() else 0.0,
        "total_displacement_mean": float(total_displacement[cold].mean().item()) if cold.any() else 0.0,
        "total_displacement_max": float(total_displacement.max().item()),
        "repeated_user_rate": float(repeats / selected_count),
        "selected_user_ids": selected_per_row,
    }
    if not all(torch.isfinite(torch.tensor(v)).item() for k, v in diagnostics.items() if isinstance(v, float)):
        raise RuntimeError("refinement diagnostics contain a non-finite value")
    return current, diagnostics


class LearnerGuidedColdModel(LegacyFast3FeedbackUSIM):
    """Legacy-compatible recommender with PPO-free LIRA refinement."""

    _UNUSED_LEGACY_MODULES = (
        "agent",
        "llm_proj",
        "content_delta",
        "content_delta_projector",
        "sage_gate_bucket_emb",
        "sage_gate_mlp",
        "sage_score_gate_mlp",
        "cgrc_recon_mlp",
    )

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        for name in self._UNUSED_LEGACY_MODULES:
            if hasattr(self, name):
                delattr(self, name)

    def get_item_vector(self, item_idx, llm_scores, force_cold=False, disable_id_dropout=False):
        id_true = self.item_id_emb(item_idx)
        mask = torch.zeros((item_idx.size(0), 1), dtype=torch.bool, device=item_idx.device)
        if isinstance(force_cold, torch.Tensor):
            mask |= force_cold.to(device=item_idx.device, dtype=torch.bool).view(-1, 1)
        elif force_cold:
            mask.fill_(True)
        if self.training and self.cfg.dropout_prob > 0 and not disable_id_dropout:
            mask |= torch.rand_like(mask, dtype=torch.float32) < float(self.cfg.dropout_prob)
        id_visible = torch.where(mask, torch.zeros_like(id_true), id_true)
        content = F.normalize(self.content_proj(self.item_con_emb(item_idx)), dim=1)
        gate = self.gate_net(torch.cat([id_visible, content], dim=1))
        fused = gate * id_visible + (1.0 - gate) * content
        return fused, id_true, content

    def _compute_aux_loss(self, id_e_true, content_e, effective_cold):
        return (id_e_true.sum() + content_e.sum()) * 0.0

    def content_delta_regularization(self):
        return next(self.parameters()).sum() * 0.0

    def content_delta_stats(self):
        return None

    def content_delta_trainable_parameters(self):
        return []

    def clip_content_delta_(self):
        return None

    def _compute_sage_aux_loss(self, *args, **kwargs):
        zero = next(self.parameters()).sum() * 0.0
        return zero, {"sage_aux_active_ratio": 0.0, "sage_aux_pool_fit": 0.0}

    def _compute_cgrc_recon_aux_loss(self, *args, **kwargs):
        zero = next(self.parameters()).sum() * 0.0
        return zero, {"cgrc_recon_active_ratio": 0.0, "cgrc_recon_pos_count": 0}

    def _lira_candidate_fit(
        self,
        current_h: torch.Tensor,
        candidates: torch.Tensor,
        candidate_user_ids: torch.Tensor,
        item_idx: torch.Tensor,
        target_pop: torch.Tensor | None,
        user_seen_items,
    ) -> torch.Tensor:
        batch_size, n_candidates = candidate_user_ids.shape
        flat_users = candidate_user_ids.reshape(-1)
        flat_items = item_idx.view(-1, 1).expand(-1, n_candidates).reshape(-1)
        flat_pop = None
        if target_pop is not None:
            flat_pop = target_pop.view(-1, 1).expand(-1, n_candidates).reshape(-1)
        terms = self._compute_course_reward_terms(
            flat_users,
            item_idx=flat_items,
            target_pop=flat_pop,
            user_seen_items=user_seen_items,
        )
        similarity = (
            F.normalize(current_h, dim=1).unsqueeze(1)
            * F.normalize(candidates, dim=2)
        ).sum(dim=2)
        shape = (batch_size, n_candidates)
        return learner_fit(
            similarity,
            terms["concept_bonus"].view(shape),
            terms["prereq_gap"].view(shape),
            terms["difficulty_gap"].view(shape),
            terms["redundant"].view(shape),
            concept_weight=float(os.environ.get("LIRA_CONCEPT_WEIGHT", "0.25")),
            prereq_beta=float(os.environ.get("LIRA_PREREQ_BETA", "1.0")),
            difficulty_beta=float(os.environ.get("LIRA_DIFFICULTY_BETA", "1.0")),
        ).detach()

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
    ):
        if item_idx is None:
            raise ValueError("LIRA refinement requires item indices")
        empty_trajectory = {
            "log_probs": [], "values": [], "rewards": [], "entropies": [],
            "states": [], "time_steps": [], "candidates": [], "actions": [],
        }
        if int(getattr(self.cfg, "usim_steps", 3)) <= 0:
            # LIRA zero-step baseline
            zero_stats = {
                "dup_rate": 0.0, "topm_coverage": 0.0, "steps": 0,
                "step_gain": 0.0, "collapse_penalty": 0.0,
                "course_sample_fit": 0.0, "course_prereq_gap": 0.0,
                "course_concept_bonus": 0.0, "course_difficulty_gap": 0.0,
                "course_redundant": 0.0, "target_alpha": 0.0,
                "sage_active": 0.0, "sage_gate": 0.0, "sage_tail_active": 0.0,
                "sage_pool_fit": 0.0, "sage_two_expert": 0.0,
                "cgrc_recon_sample_active": 0.0, "cgrc_recon_sample_score": 0.0,
                "fit_mean": 0.0, "fit_p25": 0.0, "fit_p50": 0.0, "fit_p75": 0.0,
                "update_active_ratio": 0.0, "stopped_ratio": 0.0,
                "step_displacement_mean": 0.0, "step_displacement_max": 0.0,
                "total_displacement_mean": 0.0, "total_displacement_max": 0.0,
                "repeated_user_rate": 0.0, "selected_user_ids": [],
            }
            return init_item_emb, empty_trajectory, zero_stats
        user_bank_norm = None
        if user_bank_raw is None:
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        else:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)

        candidates, candidate_user_ids, retrieval_stats = self.get_candidates(
            init_item_emb,
            user_bank_raw=user_bank_raw,
            user_bank_norm=user_bank_norm,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        fit = self._lira_candidate_fit(
            init_item_emb,
            candidates,
            candidate_user_ids,
            item_idx,
            target_pop,
            user_seen_items,
        )
        effective_cold = (
            self._cold_mask_from_pop(target_pop)
            if target_pop is not None
            else torch.ones(init_item_emb.size(0), dtype=torch.bool, device=init_item_emb.device)
        )
        refined, diagnostics = bounded_learner_refinement(
            init_item_emb,
            candidates,
            candidate_user_ids,
            fit,
            effective_cold,
            steps=int(getattr(self.cfg, "usim_steps", 3)),
            lr=float(os.environ.get("LIRA_UPDATE_LR", "0.10")),
            min_fit=float(os.environ.get("LIRA_MIN_FIT", "0.05")),
            step_cap=float(os.environ.get("LIRA_STEP_CAP", "0.05")),
            total_cap=float(os.environ.get("LIRA_TOTAL_CAP", "0.10")),
        )
        candidate_stats = {
            "dup_rate": float((retrieval_stats or {}).get("dup_rate", 0.0)),
            "topm_coverage": float((retrieval_stats or {}).get("topm_coverage", 0.0)),
            "steps": int(getattr(self.cfg, "usim_steps", 3)),
            "step_gain": 0.0,
            "collapse_penalty": 0.0,
            "course_sample_fit": diagnostics["fit_mean"],
            "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0,
            "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
            "target_alpha": 0.0,
            "sage_active": 0.0,
            "sage_gate": 0.0,
            "sage_tail_active": 0.0,
            "sage_pool_fit": 0.0,
            "sage_two_expert": 0.0,
            "cgrc_recon_sample_active": 0.0,
            "cgrc_recon_sample_score": 0.0,
            **diagnostics,
        }
        return refined, empty_trajectory, candidate_stats


def main() -> None:
    legacy.Fast3FeedbackUSIM = LearnerGuidedColdModel
    legacy.setup_seed(int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025"))))
    legacy.main()


if __name__ == "__main__":
    main()
