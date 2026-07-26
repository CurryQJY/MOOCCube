"""Deterministic item-level pseudo-cold plans for the V1 training path.

This module is intentionally independent of the trainer.  A caller creates one
plan from train-only item popularity, records its audit payload, and reuses its
mask when constructing the pseudo-cold training view.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch


_SCHEMA_VERSION = "pseudocold-v1"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _as_popularity_tensor(popularity: Any) -> torch.Tensor:
    pop = torch.as_tensor(popularity)
    if pop.dim() != 1:
        raise ValueError(f"train_popularity must be one-dimensional, got shape={tuple(pop.shape)}")
    if pop.is_complex():
        raise ValueError("train_popularity must be real-valued")
    if not torch.isfinite(pop).all():
        raise ValueError("train_popularity must contain only finite values")
    if (pop < 0).any():
        raise ValueError("train_popularity must be non-negative")
    return pop


def _as_item_mask(selected_item_mask: Any, n_items: int, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(selected_item_mask, device=device)
    if mask.dim() != 1 or int(mask.numel()) != int(n_items):
        raise ValueError(
            "selected_item_mask must be one-dimensional with one entry per item: "
            f"expected {n_items}, got shape={tuple(mask.shape)}"
        )
    return mask.to(dtype=torch.bool)


def _popularity_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "sum": 0.0}
    value_sum = float(sum(values))
    return {
        "count": int(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(value_sum / len(values)),
        "sum": value_sum,
    }


def _stratified_quotas(stratum_sizes: Sequence[int], target_count: int) -> list[int]:
    total = int(sum(stratum_sizes))
    if total == 0 or target_count == 0:
        return [0 for _ in stratum_sizes]

    raw = [target_count * size / total for size in stratum_sizes]
    quotas = [min(size, int(math.floor(value))) for size, value in zip(stratum_sizes, raw)]
    remaining = target_count - sum(quotas)
    rank = sorted(
        range(len(stratum_sizes)),
        key=lambda idx: (-(raw[idx] - math.floor(raw[idx])), idx),
    )
    while remaining > 0:
        advanced = False
        for idx in rank:
            if quotas[idx] >= stratum_sizes[idx]:
                continue
            quotas[idx] += 1
            remaining -= 1
            advanced = True
            if remaining == 0:
                break
        if not advanced:
            raise RuntimeError("unable to allocate pseudo-cold selection quotas")
    return quotas


def _stable_item_order(item_ids: Sequence[int], seed: int, stratum_index: int) -> list[int]:
    def rank(item_id: int) -> tuple[str, int]:
        token = f"{_SCHEMA_VERSION}|{seed}|{stratum_index}|{int(item_id)}".encode("ascii")
        return hashlib.sha256(token).hexdigest(), int(item_id)

    return sorted((int(item_id) for item_id in item_ids), key=rank)


@dataclasses.dataclass(frozen=True, eq=False)
class PseudoColdPlan:
    """An immutable pseudo-cold catalog selection and its reproducibility audit."""

    n_items: int
    seed: int
    cold_threshold: float
    min_popularity: float
    target_count: int
    ratio: float | None
    n_strata: int
    eligible_item_ids: tuple[int, ...]
    selected_item_ids: tuple[int, ...]
    _selected_mask: tuple[bool, ...] = dataclasses.field(repr=False)
    plan_hash: str
    audit: Mapping[str, Any]

    @property
    def selected_mask(self) -> torch.Tensor:
        """Return a fresh CPU mask so callers cannot mutate the plan itself."""
        return torch.tensor(self._selected_mask, dtype=torch.bool)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "n_items": int(self.n_items),
            "seed": int(self.seed),
            "cold_threshold": float(self.cold_threshold),
            "min_popularity": float(self.min_popularity),
            "target_count": int(self.target_count),
            "ratio": None if self.ratio is None else float(self.ratio),
            "n_strata": int(self.n_strata),
            "eligible_item_ids": list(self.eligible_item_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "selected_mask": list(self._selected_mask),
            "plan_hash": self.plan_hash,
            "audit": _thaw(self.audit),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PseudoColdPlan):
            return NotImplemented
        return self.to_dict() == other.to_dict()


def build_pseudocold_plan(
    train_popularity: Any,
    *,
    target_count: int | None = None,
    ratio: float | None = None,
    min_popularity: float = 1.0,
    cold_threshold: float = 1.0,
    seed: int = 2025,
    n_strata: int = 4,
) -> PseudoColdPlan:
    """Select fixed warm items with train-popularity stratification.

    Exactly one of ``target_count`` and ``ratio`` determines the desired catalog
    size.  The effective size is capped by eligible train-warm items and recorded
    in the immutable audit.  Selection within a stratum is keyed by SHA-256 rather
    than process-global random state, making it stable across runs and workers.
    """
    if (target_count is None) == (ratio is None):
        raise ValueError("provide exactly one of target_count or ratio")
    if target_count is not None and int(target_count) != target_count:
        raise ValueError("target_count must be an integer")
    if target_count is not None and int(target_count) < 0:
        raise ValueError("target_count must be non-negative")
    if ratio is not None and not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    if float(min_popularity) < 0.0:
        raise ValueError("min_popularity must be non-negative")
    if float(cold_threshold) < 0.0:
        raise ValueError("cold_threshold must be non-negative")
    if int(n_strata) != n_strata or int(n_strata) < 1:
        raise ValueError("n_strata must be a positive integer")

    pop = _as_popularity_tensor(train_popularity).detach().to(device="cpu", dtype=torch.float64)
    n_items = int(pop.numel())
    pop_values = [float(value) for value in pop.tolist()]
    eligibility_floor = max(float(min_popularity), float(cold_threshold))
    eligible_item_ids = tuple(
        item_id for item_id, value in enumerate(pop_values) if value >= eligibility_floor
    )
    requested_count = (
        int(target_count)
        if target_count is not None
        else int(math.ceil(len(eligible_item_ids) * float(ratio)))
    )
    selected_count = min(requested_count, len(eligible_item_ids))
    actual_strata = min(int(n_strata), len(eligible_item_ids))

    sorted_eligible = sorted(eligible_item_ids, key=lambda item_id: (pop_values[item_id], item_id))
    strata: list[list[int]] = []
    if actual_strata > 0:
        start = 0
        base_size, extra = divmod(len(sorted_eligible), actual_strata)
        for stratum_index in range(actual_strata):
            size = base_size + (1 if stratum_index < extra else 0)
            strata.append(sorted_eligible[start:start + size])
            start += size

    quotas = _stratified_quotas([len(stratum) for stratum in strata], selected_count)
    selected_by_stratum = [
        _stable_item_order(stratum, int(seed), stratum_index)[:quota]
        for stratum_index, (stratum, quota) in enumerate(zip(strata, quotas))
    ]
    selected_item_ids = tuple(sorted(item_id for chosen in selected_by_stratum for item_id in chosen))
    selected_set = set(selected_item_ids)
    selected_mask = tuple(item_id in selected_set for item_id in range(n_items))

    popularity_hash = _sha256({"train_popularity": pop_values})
    plan_payload = {
        "schema_version": _SCHEMA_VERSION,
        "seed": int(seed),
        "cold_threshold": float(cold_threshold),
        "min_popularity": float(min_popularity),
        "target_count": int(requested_count),
        "ratio": None if ratio is None else float(ratio),
        "n_strata": int(n_strata),
        "train_popularity_hash": popularity_hash,
        "eligible_item_ids": list(eligible_item_ids),
        "selected_item_ids": list(selected_item_ids),
    }
    plan_hash = _sha256(plan_payload)
    audit_strata = []
    for stratum_index, (stratum, chosen) in enumerate(zip(strata, selected_by_stratum)):
        values = [pop_values[item_id] for item_id in stratum]
        chosen_values = [pop_values[item_id] for item_id in chosen]
        audit_strata.append(
            {
                "index": int(stratum_index),
                "eligible_count": int(len(stratum)),
                "selected_count": int(len(chosen)),
                "eligible_popularity": _popularity_stats(values),
                "selected_popularity": _popularity_stats(chosen_values),
            }
        )
    audit = _freeze(
        {
            "schema_version": _SCHEMA_VERSION,
            "train_popularity_hash": popularity_hash,
            "n_items": n_items,
            "seed": int(seed),
            "cold_threshold": float(cold_threshold),
            "min_popularity": float(min_popularity),
            "eligibility_floor": eligibility_floor,
            "requested_target_count": int(requested_count),
            "ratio": None if ratio is None else float(ratio),
            "eligible_item_count": int(len(eligible_item_ids)),
            "selected_item_count": int(len(selected_item_ids)),
            "eligible_popularity": _popularity_stats([pop_values[idx] for idx in eligible_item_ids]),
            "selected_popularity": _popularity_stats([pop_values[idx] for idx in selected_item_ids]),
            "strata": audit_strata,
        }
    )
    return PseudoColdPlan(
        n_items=n_items,
        seed=int(seed),
        cold_threshold=float(cold_threshold),
        min_popularity=float(min_popularity),
        target_count=int(requested_count),
        ratio=None if ratio is None else float(ratio),
        n_strata=int(n_strata),
        eligible_item_ids=eligible_item_ids,
        selected_item_ids=selected_item_ids,
        _selected_mask=selected_mask,
        plan_hash=plan_hash,
        audit=audit,
    )


def mask_user_item_history(history: Any, selected_item_mask: Any) -> Any:
    """Return a copy of a user-item history with selected item columns removed.

    Supports the trainer's sparse ``dict[user_id, set[item_id]]`` view as well as
    dense and Torch sparse matrices.  It never mutates the supplied history.
    """
    if isinstance(history, Mapping):
        mask = torch.as_tensor(selected_item_mask, dtype=torch.bool).view(-1)
        selected_ids = {int(item_id) for item_id in torch.nonzero(mask, as_tuple=False).view(-1).tolist()}
        return {
            int(user_id): {int(item_id) for item_id in items if int(item_id) not in selected_ids}
            for user_id, items in history.items()
        }
    if not torch.is_tensor(history):
        raise TypeError("history must be a mapping or a torch user-item matrix")
    if history.dim() != 2:
        raise ValueError(f"history must be two-dimensional, got shape={tuple(history.shape)}")

    n_items = int(history.shape[1])
    mask = _as_item_mask(selected_item_mask, n_items, history.device)
    if history.layout == torch.strided:
        masked = history.clone()
        masked[:, mask] = 0
        return masked

    coo_history = history.coalesce() if history.layout == torch.sparse_coo else history.to_sparse_coo().coalesce()
    indices = coo_history.indices()
    keep = ~mask.index_select(0, indices[1])
    return torch.sparse_coo_tensor(
        indices[:, keep],
        coo_history.values()[keep],
        size=coo_history.shape,
        dtype=coo_history.dtype,
        device=coo_history.device,
    ).coalesce()


def effective_item_popularity(train_popularity: Any, selected_item_mask: Any) -> torch.Tensor:
    """Return train popularity with pseudo-cold catalog entries set to zero."""
    pop = _as_popularity_tensor(train_popularity)
    mask = _as_item_mask(selected_item_mask, int(pop.numel()), pop.device)
    return pop.clone().masked_fill(mask, 0)


def effective_item_difficulty(train_popularity: Any, selected_item_mask: Any) -> torch.Tensor:
    """Compute difficulty from the masked train-popularity view used by V1."""
    effective_popularity = effective_item_popularity(train_popularity, selected_item_mask).float()
    max_log = torch.log1p(effective_popularity.max()).clamp_min(1.0)
    return (1.0 - torch.log1p(effective_popularity) / max_log).clamp(0.0, 1.0)


__all__ = [
    "PseudoColdPlan",
    "build_pseudocold_plan",
    "effective_item_difficulty",
    "effective_item_popularity",
    "mask_user_item_history",
]
