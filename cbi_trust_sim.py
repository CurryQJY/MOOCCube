"""Isolated CBI-constrained simulator components."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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

