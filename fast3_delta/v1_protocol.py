"""Small, testable semantic rules used by the isolated CKG-RL V1 route."""

from __future__ import annotations

from typing import Mapping, Optional

import torch


def exclude_row_targets_from_history(seen_mat: torch.Tensor, target_items: torch.Tensor):
    """Remove each row's target course from its history without mutating the input."""
    if seen_mat.dim() != 2:
        raise ValueError("seen_mat must have shape [batch, n_items]")
    target_items = torch.as_tensor(target_items, dtype=torch.long, device=seen_mat.device).view(-1)
    if target_items.numel() != seen_mat.size(0):
        raise ValueError("target_items must have one value per history row")
    if target_items.numel() and (
        int(target_items.min().item()) < 0 or int(target_items.max().item()) >= seen_mat.size(1)
    ):
        raise ValueError("target_items contains an out-of-range course index")

    cleaned = seen_mat.clone()
    if target_items.numel():
        cleaned.scatter_(1, target_items.view(-1, 1), 0.0)
    return cleaned, cleaned.sum(dim=1, keepdim=True)


def batch_invariant_alignment_score(
    item_state: torch.Tensor,
    selected_user: torch.Tensor,
    target_state: torch.Tensor,
    target_alpha: torch.Tensor,
    reference_batch_size: int,
) -> torch.Tensor:
    """Aggregate per-course simulator scores with a fixed, configured scale."""
    reference_batch_size = int(reference_batch_size)
    if reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if item_state.shape != selected_user.shape or item_state.shape != target_state.shape:
        raise ValueError("item_state, selected_user, and target_state must share a shape")
    if target_alpha.numel() != item_state.size(0):
        raise ValueError("target_alpha must have one value per item row")

    alpha = target_alpha.to(device=item_state.device, dtype=item_state.dtype).view(-1, 1)
    user_align = (item_state * selected_user).sum(dim=1, keepdim=True)
    target_align = (item_state * target_state).sum(dim=1, keepdim=True)
    row_scores = (1.0 - alpha) * user_align + alpha * target_align
    return row_scores.sum() / float(reference_batch_size)


def apply_course_fit_sampling_bias(
    base_probs: torch.Tensor,
    course_fit: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Fuse course fit into the candidate identity distribution before sampling."""
    if base_probs.shape != course_fit.shape:
        raise ValueError("base_probs and course_fit must share a shape")
    if base_probs.dim() != 2:
        raise ValueError("base_probs must have shape [batch, candidates]")
    if float(beta) <= 0.0:
        return base_probs

    probs = torch.nan_to_num(base_probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    fit = torch.nan_to_num(course_fit, nan=0.0, posinf=0.0, neginf=0.0)
    scale = fit.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    log_weight = (float(beta) * fit / scale).clamp(min=-30.0, max=30.0)
    biased = probs * torch.exp(log_weight)
    return biased / biased.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _metric(metrics: Mapping[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0)) if metrics else 0.0


def _retained(
    candidate: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
) -> bool:
    key_n = f"N@{int(k)}"
    key_r = f"R@{int(k)}"
    for metric_key in (key_n, key_r):
        if _metric(candidate.get("hot", {}), metric_key) < _metric(base.get("hot", {}), metric_key) - float(hot_tolerance):
            return False
        if _metric(candidate.get("overall", {}), metric_key) < _metric(base.get("overall", {}), metric_key) - float(overall_tolerance):
            return False
    return True


def select_retained_cold_checkpoint(
    previous: Optional[Mapping[str, Mapping[str, float]]],
    candidate: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
    min_delta: float = 1e-4,
):
    """Keep the best retained checkpoint: Cold NDCG first, Recall second."""
    if not _retained(candidate, base, k, hot_tolerance, overall_tolerance):
        return previous
    if previous is None:
        return candidate

    key_n = f"N@{int(k)}"
    key_r = f"R@{int(k)}"
    candidate_n = _metric(candidate.get("cold", {}), key_n)
    previous_n = _metric(previous.get("cold", {}), key_n)
    if candidate_n > previous_n + float(min_delta):
        return candidate
    if abs(candidate_n - previous_n) <= float(min_delta):
        if _metric(candidate.get("cold", {}), key_r) > _metric(previous.get("cold", {}), key_r) + 1e-12:
            return candidate
    return previous


def update_running_retention_n_peaks(
    previous: Optional[Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    k: int,
):
    """Track validation Hot/Overall NDCG peaks for cross-epoch retention."""
    key_n = f"N@{int(k)}"
    previous = previous or {}
    return {
        "hot_n": max(float(previous.get("hot_n", float("-inf"))), _metric(candidate.get("hot", {}), key_n)),
        "overall_n": max(
            float(previous.get("overall_n", float("-inf"))),
            _metric(candidate.get("overall", {}), key_n),
        ),
    }


def _retained_by_running_n_peaks(
    candidate: Mapping[str, Mapping[str, float]],
    running_n_peaks: Optional[Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
) -> bool:
    """Require current Hot/Overall NDCG to stay near earlier validation peaks."""
    if not running_n_peaks:
        return True
    key_n = f"N@{int(k)}"
    hot_peak = float(running_n_peaks.get("hot_n", float("-inf")))
    overall_peak = float(running_n_peaks.get("overall_n", float("-inf")))
    return (
        _metric(candidate.get("hot", {}), key_n) >= hot_peak - float(hot_tolerance)
        and _metric(candidate.get("overall", {}), key_n)
        >= overall_peak - float(overall_tolerance)
    )


def select_running_retained_cold_checkpoint(
    previous: Optional[Mapping[str, Mapping[str, float]]],
    candidate: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    running_n_peaks: Optional[Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
    min_delta: float = 1e-4,
):
    """Select Cold-NDCG checkpoints only while both retention contracts hold.

    The legacy same-epoch refined-vs-base contract retains both NDCG and
    Recall.  The new cross-epoch contract intentionally retains NDCG only,
    matching the configured selection metric while avoiding an unrelated
    recall fluctuation from rejecting an otherwise valid cold checkpoint.
    """
    selected, _, _, _ = advance_running_retention_selector(
        previous=previous,
        candidate=candidate,
        base=base,
        running_n_peaks=running_n_peaks,
        k=k,
        hot_tolerance=hot_tolerance,
        overall_tolerance=overall_tolerance,
        min_delta=min_delta,
    )
    return selected


def advance_running_retention_selector(
    previous: Optional[Mapping[str, Mapping[str, float]]],
    candidate: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    running_n_peaks: Optional[Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
    min_delta: float = 1e-4,
):
    """Advance the V1.1 selector without allowing rejected epochs to set floors.

    Return ``(selected, next_n_peaks, base_retained, running_retained)``.
    The floor is deliberately updated after deciding the current epoch so an
    epoch cannot make its own retention test vacuous.
    """
    base_retained = _retained(candidate, base, k, hot_tolerance, overall_tolerance)
    running_retained = base_retained and _retained_by_running_n_peaks(
        candidate,
        running_n_peaks,
        k,
        hot_tolerance,
        overall_tolerance,
    )
    selected = previous
    if running_retained:
        selected = select_retained_cold_checkpoint(
            previous=previous,
            candidate=candidate,
            base=base,
            k=k,
            hot_tolerance=hot_tolerance,
            overall_tolerance=overall_tolerance,
            min_delta=min_delta,
        )
    next_n_peaks = running_n_peaks
    if base_retained:
        next_n_peaks = update_running_retention_n_peaks(
            running_n_peaks,
            candidate,
            k,
        )
    return selected, next_n_peaks, base_retained, running_retained


def retained_by_base(
    candidate: Mapping[str, Mapping[str, float]],
    base: Mapping[str, Mapping[str, float]],
    k: int,
    hot_tolerance: float,
    overall_tolerance: float,
) -> bool:
    """Public predicate used by the training loop for transparent diagnostics."""
    return _retained(candidate, base, k, hot_tolerance, overall_tolerance)
