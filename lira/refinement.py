from typing import Any
from collections.abc import Callable

import torch
import torch.nn.functional as F


def learner_fit(
    similarity: torch.Tensor,
    concept: torch.Tensor,
    prerequisite_gap: torch.Tensor,
    difficulty_gap: torch.Tensor,
    redundancy: torch.Tensor,
    concept_weight: float = 0.25,
    prerequisite_beta: float = 1.0,
    difficulty_beta: float = 1.0,
) -> torch.Tensor:
    base = (
        0.5 * (similarity.clamp(-1.0, 1.0) + 1.0)
        + concept_weight * concept.clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    score = (
        base
        * torch.exp(-prerequisite_beta * prerequisite_gap.clamp(0.0, 1.0))
        * torch.exp(-difficulty_beta * difficulty_gap.clamp(0.0, 1.0))
        * (1.0 - redundancy.clamp(0.0, 1.0))
    )
    return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    return float(torch.quantile(finite.float(), q).item()) if finite.numel() else 0.0


def bounded_refinement(
    initial: torch.Tensor,
    candidate_vectors: torch.Tensor,
    candidate_user_ids: torch.Tensor,
    fit: torch.Tensor,
    cold_mask: torch.Tensor,
    *,
    steps: int,
    update_lr: float,
    min_fit: float,
    step_cap: float,
    total_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if candidate_vectors.shape[:2] != fit.shape or candidate_user_ids.shape != fit.shape:
        raise ValueError("candidate tensors have incompatible shapes")
    if not torch.isfinite(fit).all():
        raise RuntimeError("fit contains non-finite values")

    current = initial.clone()
    cold = cold_mask.to(device=initial.device, dtype=torch.bool).view(-1)
    active = cold.clone()
    used = torch.zeros_like(fit, dtype=torch.bool)
    selected_ids: list[list[int]] = [[] for _ in range(initial.size(0))]
    fit_values: list[torch.Tensor] = []
    step_values: list[torch.Tensor] = []
    active_events = 0

    for _ in range(steps):
        available = fit.masked_fill(used, float("-inf"))
        best_fit, action = available.max(dim=1)
        step_active = active & torch.isfinite(best_fit) & (best_fit > min_fit)
        active &= step_active
        if not step_active.any():
            break
        rows = torch.arange(initial.size(0), device=initial.device)
        selected = candidate_vectors[rows, action]
        direction = F.normalize(selected - current, dim=1, eps=1e-12)
        step_norm = (update_lr * best_fit.clamp_min(0.0)).clamp(max=step_cap)
        delta = direction * step_norm.view(-1, 1) * step_active.view(-1, 1)
        proposed_delta = current + delta - initial
        proposed_norm = torch.linalg.vector_norm(proposed_delta, dim=1, keepdim=True)
        scale = (total_cap / proposed_norm.clamp_min(1e-12)).clamp(max=1.0)
        current = initial + proposed_delta * scale
        current[~cold] = initial[~cold]
        chosen_rows = step_active.nonzero(as_tuple=False).view(-1)
        used[chosen_rows, action[chosen_rows]] = True
        fit_values.append(best_fit[step_active])
        step_values.append(torch.linalg.vector_norm(delta[step_active], dim=1))
        active_events += int(step_active.sum().item())
        for row in chosen_rows.cpu().tolist():
            selected_ids[row].append(int(candidate_user_ids[row, action[row]].item()))

    total_displacement = torch.linalg.vector_norm(current - initial, dim=1)
    selected_fit = torch.cat(fit_values) if fit_values else initial.new_empty(0)
    step_displacement = torch.cat(step_values) if step_values else initial.new_empty(0)
    repeats = sum(len(row) - len(set(row)) for row in selected_ids)
    selected_count = max(1, sum(len(row) for row in selected_ids))
    cold_count = max(1, int(cold.sum().item()))
    diagnostics = {
        "fit_mean": float(selected_fit.mean().item()) if selected_fit.numel() else 0.0,
        "fit_p25": _quantile(selected_fit, 0.25),
        "fit_p50": _quantile(selected_fit, 0.50),
        "fit_p75": _quantile(selected_fit, 0.75),
        "update_active_ratio": active_events / max(1, cold_count * steps),
        "stopped_ratio": float((cold & ~active).sum().item() / cold_count),
        "step_displacement_mean": float(step_displacement.mean().item()) if step_displacement.numel() else 0.0,
        "step_displacement_max": float(step_displacement.max().item()) if step_displacement.numel() else 0.0,
        "total_displacement_mean": float(total_displacement[cold].mean().item()) if cold.any() else 0.0,
        "total_displacement_max": float(total_displacement.max().item()),
        "repeated_user_rate": repeats / selected_count,
        "selected_user_ids": selected_ids,
    }
    if diagnostics["total_displacement_max"] > total_cap + 1e-6:
        raise RuntimeError("total displacement cap was violated")
    if not torch.equal(current[~cold], initial[~cold]):
        raise RuntimeError("warm rows changed")
    return current, diagnostics


def dynamic_bounded_refinement(
    initial: torch.Tensor,
    cold_mask: torch.Tensor,
    candidate_provider: Callable,
    *,
    steps: int,
    update_lr: float,
    min_fit: float,
    min_gain: float,
    step_cap: float,
    total_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    current = initial.clone()
    cold = cold_mask.to(device=initial.device, dtype=torch.bool).view(-1)
    active = cold.clone()
    selected_history: list[list[int]] = [[] for _ in range(initial.size(0))]
    fit_values, step_values, gain_values = [], [], []
    active_events = 0

    for _ in range(steps):
        candidates, candidate_ids, fit = candidate_provider(current, selected_history)
        if not torch.isfinite(fit).all():
            raise RuntimeError("fit contains non-finite values")
        used = torch.zeros_like(fit, dtype=torch.bool)
        for row, history in enumerate(selected_history):
            if history:
                history_tensor = torch.tensor(history, device=candidate_ids.device)
                used[row] = (candidate_ids[row].view(-1, 1) == history_tensor.view(1, -1)).any(1)
        available = fit.masked_fill(used, float("-inf"))
        best_fit, action = available.max(dim=1)
        rows = torch.arange(initial.size(0), device=initial.device)
        selected = candidates[rows, action]
        direction = F.normalize(selected - current, dim=1, eps=1e-12)
        step_norm = (update_lr * best_fit.clamp_min(0.0)).clamp(max=step_cap)
        proposed = current + direction * step_norm.view(-1, 1)
        before = F.cosine_similarity(current, selected, dim=1)
        after = F.cosine_similarity(proposed, selected, dim=1)
        gain = after - before
        step_active = (
            active
            & torch.isfinite(best_fit)
            & (best_fit > min_fit)
            & torch.isfinite(gain)
            & (gain > min_gain)
        )
        active &= step_active
        if not step_active.any():
            break
        delta = direction * step_norm.view(-1, 1) * step_active.view(-1, 1)
        proposed_delta = current + delta - initial
        proposed_norm = torch.linalg.vector_norm(proposed_delta, dim=1, keepdim=True)
        scale = (total_cap / proposed_norm.clamp_min(1e-12)).clamp(max=1.0)
        current = initial + proposed_delta * scale
        current[~cold] = initial[~cold]
        chosen_rows = step_active.nonzero(as_tuple=False).view(-1)
        fit_values.append(best_fit[step_active])
        gain_values.append(gain[step_active])
        step_values.append(torch.linalg.vector_norm(delta[step_active], dim=1))
        active_events += int(step_active.sum().item())
        for row in chosen_rows.cpu().tolist():
            selected_history[row].append(int(candidate_ids[row, action[row]].item()))

    total = torch.linalg.vector_norm(current - initial, dim=1)
    selected_fit = torch.cat(fit_values) if fit_values else initial.new_empty(0)
    selected_gain = torch.cat(gain_values) if gain_values else initial.new_empty(0)
    step_displacement = torch.cat(step_values) if step_values else initial.new_empty(0)
    count = max(1, int(cold.sum().item()))
    diagnostics = {
        "fit_mean": float(selected_fit.mean().item()) if selected_fit.numel() else 0.0,
        "fit_p25": _quantile(selected_fit, 0.25),
        "fit_p50": _quantile(selected_fit, 0.50),
        "fit_p75": _quantile(selected_fit, 0.75),
        "gain_mean": float(selected_gain.mean().item()) if selected_gain.numel() else 0.0,
        "update_active_ratio": active_events / max(1, count * steps),
        "stopped_ratio": float((cold & ~active).sum().item() / count),
        "step_displacement_mean": float(step_displacement.mean().item()) if step_displacement.numel() else 0.0,
        "step_displacement_max": float(step_displacement.max().item()) if step_displacement.numel() else 0.0,
        "total_displacement_mean": float(total[cold].mean().item()) if cold.any() else 0.0,
        "total_displacement_max": float(total.max().item()),
        "repeated_user_rate": 0.0,
        "selected_user_ids": selected_history,
    }
    return current, diagnostics
