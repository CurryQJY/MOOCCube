"""Single-seed Ridge-initialized course-reward PPO pilot.

The runner is intentionally separate from the selected main model and the
historical clean RL route.  It uses a frozen graph-gated checkpoint, replaces
strict-cold item rows with a warm-only content-to-collaborative Ridge map, and
trains only a legal user-direction policy on pseudo-cold warm items.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ckg_rl_usim_v32_clean import (
    CleanRecPPO,
    CleanRunConfig,
    CleanUSIMEngine,
    build_clean_course_signal,
    project_displacement,
)
from fast3_delta.static_protocol import load_shared_static_split
from graph_gated_scorer_clean import (
    GraphContentScorer,
    GraphScorerConfig,
    build_norm_adj,
    build_user_seen,
    compute_train_popularity,
)
from graph_knp_pseudocold import mask_bipartite_item_edges
from knp_adaptive_delta_eval import course_fit_walk
from knp_refine_eval_graph import evaluate_with_banks


ROOT = Path(__file__).resolve().parent
RIDGE_LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
DELTA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_OUTPUT = ROOT / "outputs" / "ridge_course_reward_rl_pilot_seed2025"


@dataclass(frozen=True)
class PolicyArmSpec:
    name: str
    use_course_bias: bool
    use_course_reward: bool


@dataclass(frozen=True)
class SelectionProtocol:
    selection_mode: str
    delta_grid: tuple[float, ...]
    fixed_max_delta: float


_POLICY_ARM_REGISTRY = {
    "ridge_ppo_core": PolicyArmSpec("ridge_ppo_core", False, False),
    "ridge_ppo_course_bias": PolicyArmSpec("ridge_ppo_course_bias", True, False),
    "ridge_ppo_course_reward_only": PolicyArmSpec(
        "ridge_ppo_course_reward_only", False, True
    ),
    "ridge_ppo_full": PolicyArmSpec("ridge_ppo_full", True, True),
}
_POLICY_ARM_ALIASES = {
    "ridge_ppo_no_course_reward": "ridge_ppo_course_bias",
    "ridge_ppo_course_reward": "ridge_ppo_full",
}


def resolve_policy_arms(value: str | Sequence[str]) -> list[PolicyArmSpec]:
    """Resolve a deterministic list of independently configured PPO arms."""
    raw = value.split(",") if isinstance(value, str) else list(value)
    names = [str(name).strip() for name in raw if str(name).strip()]
    if not names:
        raise ValueError("at least one policy arm is required")
    if len(names) != len(set(names)):
        raise ValueError("duplicate policy arm names are not allowed")
    arms = []
    for requested_name in names:
        canonical = _POLICY_ARM_ALIASES.get(requested_name, requested_name)
        if canonical not in _POLICY_ARM_REGISTRY:
            raise ValueError(f"unknown policy arm: {requested_name}")
        base = _POLICY_ARM_REGISTRY[canonical]
        arms.append(
            PolicyArmSpec(
                name=requested_name,
                use_course_bias=base.use_course_bias,
                use_course_reward=base.use_course_reward,
            )
        )
    return arms


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _protocol_split_hashes(split_dir: Path, *, include_test: bool) -> dict[str, str]:
    split_names = ("train", "val", "test") if include_test else ("train", "val")
    return {
        name: _sha256(Path(split_dir) / f"static_{name}.pkl")
        for name in split_names
    }


def _save_selected_policy_checkpoint(
    output_dir: Path,
    *,
    arm: str,
    policy_state: Mapping[str, torch.Tensor],
    selected_epoch: int,
    selected_delta: float | None = None,
    selection_mode: str = "delta_grid",
    fixed_max_delta: float = 0.25,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{arm}_selected_policy.pt"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "arm": str(arm),
        "selected_epoch": int(selected_epoch),
        "selection_mode": str(selection_mode),
        "policy_state": {
            name: value.detach().cpu().clone()
            for name, value in policy_state.items()
        },
    }
    if selection_mode == "direct_rollout":
        payload["fixed_max_delta"] = float(fixed_max_delta)
    else:
        if selected_delta is None:
            raise ValueError("delta-grid checkpoint requires selected_delta")
        payload["selected_delta"] = float(selected_delta)
    torch.save(payload, temp_path)
    temp_path.replace(path)
    metadata = {
        "path": str(path),
        "sha256": _sha256(path),
        "selected_epoch": int(selected_epoch),
        "selection_mode": str(selection_mode),
    }
    if selection_mode == "direct_rollout":
        metadata["fixed_max_delta"] = float(fixed_max_delta)
    else:
        metadata["selected_delta"] = float(selected_delta)
    return metadata


def _save_selected_eval_bundle(
    output_dir: Path,
    *,
    arm: str,
    user_bank: torch.Tensor,
    item_bank: torch.Tensor,
    selected_epoch: int,
    selected_delta: float | None = None,
    selection_mode: str = "delta_grid",
    fixed_max_delta: float = 0.25,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{arm}_selected_eval.pt"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "arm": str(arm),
        "selected_epoch": int(selected_epoch),
        "selection_mode": str(selection_mode),
        "user_bank": user_bank.detach().cpu().clone(),
        "item_bank": item_bank.detach().cpu().clone(),
    }
    if selection_mode == "direct_rollout":
        payload["fixed_max_delta"] = float(fixed_max_delta)
    else:
        if selected_delta is None:
            raise ValueError("delta-grid evaluation bundle requires selected_delta")
        payload["selected_delta"] = float(selected_delta)
    torch.save(payload, temp_path)
    temp_path.replace(path)
    metadata = {
        "path": str(path),
        "sha256": _sha256(path),
        "selected_epoch": int(selected_epoch),
        "selection_mode": str(selection_mode),
        "user_bank_shape": list(payload["user_bank"].shape),
        "item_bank_shape": list(payload["item_bank"].shape),
    }
    if selection_mode == "direct_rollout":
        metadata["fixed_max_delta"] = float(fixed_max_delta)
    else:
        metadata["selected_delta"] = float(selected_delta)
    return metadata


def _copy_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def summarize_rollout_stats(
    batches: Sequence[tuple[int, Mapping[str, float]]],
    *,
    step_penalty: float,
    max_steps: int,
) -> dict[str, float]:
    """Aggregate rollout diagnostics on a per-example, per-step scale."""
    if float(step_penalty) < 0.0 or int(max_steps) < 1:
        raise ValueError("step_penalty must be non-negative and max_steps positive")
    total = sum(int(size) for size, _ in batches)
    keys = (
        "embedding_reward",
        "recommendation_reward",
        "course_reward",
        "active_steps",
        "end_rate",
        "rollout_delta_l2",
    )
    out = {
        key: (
            0.0 if total == 0 else
            float(sum(int(size) * float(stats.get(key, 0.0)) for size, stats in batches) / total)
        )
        for key in keys
    }
    out["step_penalty_reward"] = -float(step_penalty) * out["active_steps"] / int(max_steps)
    out["mean_shaped_reward"] = (
        out["embedding_reward"]
        + out["recommendation_reward"]
        + out["course_reward"]
        + out["step_penalty_reward"]
    )
    return out


def make_policy_partitions(
    warm_ids: Sequence[int] | np.ndarray,
    eligible_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    val_fraction: float = 0.20,
) -> tuple[list[int], list[int], list[int]]:
    """Split eligible warm items into policy train/validation and donors.

    The donor pool is all warm items not used as pseudo-cold targets.  Sorting
    outputs makes the partition independent of input ordering while the
    seeded permutation determines which eligible items become validation.
    """
    warm = sorted({int(x) for x in np.asarray(warm_ids).reshape(-1).tolist()})
    eligible = sorted({int(x) for x in np.asarray(eligible_ids).reshape(-1).tolist()})
    if len(eligible) < 4:
        raise ValueError("at least four eligible warm items are required")
    if not set(eligible).issubset(warm):
        raise ValueError("eligible_ids must be a subset of warm_ids")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be strictly between zero and one")
    order = np.random.default_rng(int(seed)).permutation(np.asarray(eligible, dtype=np.int64))
    n_val = min(len(eligible) - 1, max(1, int(round(len(eligible) * float(val_fraction)))))
    val = {int(x) for x in order[:n_val].tolist()}
    train = set(eligible) - val
    donors = set(warm) - train - val
    if not train or not val or not donors:
        raise ValueError("partition must leave nonempty train, validation, and donor pools")
    return sorted(train), sorted(val), sorted(donors)


def target_free_histories(
    histories: Mapping[int, set[int]],
    *,
    selected_user_ids: torch.Tensor,
    target_item_ids: torch.Tensor,
) -> list[set[int]]:
    """Copy histories while removing the row-specific target item."""
    users = torch.as_tensor(selected_user_ids, dtype=torch.long).view(-1).tolist()
    targets = torch.as_tensor(target_item_ids, dtype=torch.long).view(-1).tolist()
    if len(users) != len(targets):
        raise ValueError("selected_user_ids and target_item_ids must have equal length")
    return [
        {int(i) for i in histories.get(int(u), set()) if int(i) != int(t)}
        for u, t in zip(users, targets)
    ]


def pseudo_cold_adjacency(
    adj: torch.Tensor,
    *,
    n_users: int,
    target_item_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Remove and renormalize all train edges touching target items."""
    if not adj.is_sparse:
        raise ValueError("adj must be a sparse COO tensor")
    n_items = int(adj.size(0)) - int(n_users)
    ids = torch.as_tensor(target_item_ids, dtype=torch.long, device=adj.device).view(-1)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= n_items):
        raise ValueError("target_item_ids contains an out-of-range item")
    mask = torch.zeros(n_items, dtype=torch.bool, device=adj.device)
    if ids.numel():
        mask[torch.unique(ids)] = True
    return mask_bipartite_item_edges(adj, int(n_users), mask)


def _ridge_solve(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    eye = np.eye(x.shape[1], dtype=x.dtype)
    return np.linalg.solve(x.T @ x + float(lam) * eye, x.T @ y)


def fit_ridge_bank(
    content: torch.Tensor,
    factual_bank: torch.Tensor,
    *,
    warm_ids: Sequence[int] | np.ndarray,
    cold_ids: Sequence[int] | torch.Tensor,
    seed: int = 2025,
    lambdas: Sequence[float] = RIDGE_LAMBDAS,
) -> tuple[torch.Tensor, float]:
    """Fit a standardized content-to-factual-bank Ridge map on warm_ids."""
    content = torch.as_tensor(content, dtype=torch.float32)
    factual_bank = torch.as_tensor(factual_bank, dtype=torch.float32)
    if content.ndim != 2 or factual_bank.ndim != 2 or content.size(0) != factual_bank.size(0):
        raise ValueError("content and factual_bank must have matching item rows")
    warm = np.asarray(sorted({int(x) for x in warm_ids}), dtype=np.int64)
    cold = torch.as_tensor(cold_ids, dtype=torch.long, device=factual_bank.device).view(-1)
    if warm.size < 2:
        raise ValueError("Ridge requires at least two warm donor items")
    x_raw = content.cpu().numpy().astype(np.float64)
    mu = x_raw[warm].mean(axis=0)
    sd = x_raw[warm].std(axis=0) + 1e-8
    x_all = np.concatenate([(x_raw - mu) / sd, np.ones((x_raw.shape[0], 1))], axis=1)
    y_all = F.normalize(factual_bank, dim=1).cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(warm)
    n_fit = max(1, min(len(order) - 1, int(round(0.8 * len(order)))))
    train_ids, val_ids = order[:n_fit], order[n_fit:]
    scores: list[tuple[float, float]] = []
    for lam in lambdas:
        w = _ridge_solve(x_all[train_ids], y_all[train_ids], float(lam))
        pred = x_all[val_ids] @ w if val_ids.size else x_all[train_ids] @ w
        target = y_all[val_ids] if val_ids.size else y_all[train_ids]
        scores.append((float(np.mean((pred - target) ** 2)), float(lam)))
    lam = min(scores, key=lambda pair: pair[0])[1]
    w = _ridge_solve(x_all[warm], y_all[warm], lam)
    bank = factual_bank.clone()
    if cold.numel():
        pred = torch.as_tensor(x_all[cold.cpu().numpy()] @ w, dtype=bank.dtype, device=bank.device)
        bank.index_copy_(0, cold, F.normalize(pred, dim=1))
    return bank, float(lam)


def blend_ridge_rows(
    base_bank: torch.Tensor,
    ridge_bank: torch.Tensor,
    row_ids: Sequence[int] | torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Blend selected Ridge rows with their Backbone anchors."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("ridge alpha must be in [0, 1]")
    base = torch.as_tensor(base_bank)
    ridge = torch.as_tensor(ridge_bank, dtype=base.dtype, device=base.device)
    if base.ndim != 2 or ridge.shape != base.shape:
        raise ValueError("base and Ridge banks must have matching 2-D shapes")
    ids = torch.as_tensor(row_ids, dtype=torch.long, device=base.device).view(-1)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= base.size(0)):
        raise ValueError("row_ids contains an out-of-range item")
    out = base.clone()
    if not ids.numel() or float(alpha) == 0.0:
        return out
    base_rows = base.index_select(0, ids)
    ridge_rows = ridge.index_select(0, ids)
    moved = F.normalize(
        (1.0 - float(alpha)) * base_rows + float(alpha) * ridge_rows,
        dim=1,
    )
    out.index_copy_(0, ids, moved)
    return out


def resolve_simulation_ridge_alpha(
    ridge_alpha: float,
    simulation_ridge_alpha: float | None,
) -> float:
    """Resolve pseudo-cold simulation severity without changing old commands."""
    resolved = ridge_alpha if simulation_ridge_alpha is None else simulation_ridge_alpha
    if not 0.0 <= float(resolved) <= 1.0:
        raise ValueError("simulation ridge alpha must be in [0, 1]")
    return float(resolved)


def resolve_delta_grid(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and freeze the residual scale grid used by validation."""
    grid = tuple(float(value) for value in values)
    if (
        not grid
        or grid[0] != 0.0
        or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in grid)
        or any(right <= left for left, right in zip(grid, grid[1:]))
    ):
        raise ValueError(
            "delta grid must start at zero and contain unique increasing "
            "finite values in [0, 1]"
        )
    return grid


def resolve_selection_protocol(
    selection_mode: str,
    *,
    delta_grid: Sequence[float],
    fixed_max_delta: float,
) -> SelectionProtocol:
    """Resolve either historical grid selection or one fixed direct rollout."""
    mode = str(selection_mode).strip().lower()
    if mode not in {"delta_grid", "direct_rollout"}:
        raise ValueError("selection mode must be 'delta_grid' or 'direct_rollout'")
    cap = float(fixed_max_delta)
    if not np.isfinite(cap) or not 0.0 <= cap <= 1.0:
        raise ValueError("fixed max delta must be finite and in [0, 1]")
    grid = resolve_delta_grid(delta_grid) if mode == "delta_grid" else ()
    return SelectionProtocol(mode, grid, cap)


def passes_retention_gate(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    tolerance: float,
    prefix: str = "",
) -> bool:
    """Require hot and overall Recall/NDCG retention at rank 10."""
    if float(tolerance) < 0.0:
        raise ValueError("retention tolerance must be non-negative")
    keys = ("hot_R@10", "hot_N@10", "overall_R@10", "overall_N@10")
    return all(
        float(candidate[f"{prefix}{key}"])
        >= float(baseline[f"{prefix}{key}"]) - float(tolerance)
        for key in keys
    )


def select_policy_row(
    epoch_rows: Sequence[Mapping[str, Any]],
    retention_baseline: Mapping[str, float],
    *,
    tolerance: float,
) -> Mapping[str, Any]:
    """Select the cold-best policy row under one explicit retention budget."""
    rows = list(epoch_rows)
    eligible = [
        row
        for row in rows
        if passes_retention_gate(row, retention_baseline, tolerance=tolerance)
    ]
    if not eligible:
        eligible = [
            next(
                row
                for row in rows
                if int(row["epoch"]) == 0 and float(row.get("delta", 0.0)) == 0.0
            )
        ]
    return max(
        eligible,
        key=lambda row: (
            float(row["cold_N@10"]),
            float(row.get("pseudo_val_cos", -1.0)),
            float(row["overall_N@10"]),
            -float(row.get("delta", 0.0)),
            -int(row["epoch"]),
        ),
    )


def replace_cold_rows(
    base_bank: torch.Tensor,
    raw_bank: torch.Tensor,
    cold_ids: Sequence[int] | torch.Tensor,
    *,
    max_delta: float,
) -> torch.Tensor:
    """Apply a bounded residual only to cold rows and normalize them.

    Operators in this repository return either a full item bank or compact rows
    aligned with ``cold_ids``.  Supporting both avoids fabricating a full bank
    merely to insert a cold-only rollout result.
    """
    if float(max_delta) < 0.0:
        raise ValueError("max_delta must be non-negative")
    ids = torch.as_tensor(cold_ids, dtype=torch.long, device=base_bank.device).view(-1)
    raw_bank = torch.as_tensor(raw_bank, dtype=base_bank.dtype, device=base_bank.device)
    if raw_bank.shape == base_bank.shape:
        raw_rows = raw_bank.index_select(0, ids)
    elif raw_bank.ndim == 2 and raw_bank.shape == (ids.numel(), base_bank.size(1)):
        raw_rows = raw_bank
    else:
        raise ValueError(
            "raw_bank must be a full item bank or compact rows aligned with cold_ids"
        )
    out = base_bank.clone()
    if not ids.numel() or float(max_delta) == 0.0:
        return out
    init = base_bank.index_select(0, ids)
    final = F.normalize(raw_rows, dim=1)
    delta = final - init
    norm = delta.norm(dim=1, keepdim=True)
    scale = torch.clamp(float(max_delta) / norm.clamp_min(1e-12), max=1.0)
    moved = F.normalize(init + delta * scale, dim=1)
    out.index_copy_(0, ids, moved)
    return out


def insert_cold_rows(
    base_bank: torch.Tensor,
    rows: torch.Tensor,
    cold_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Insert normalized direct-rollout rows without a second displacement cap."""
    ids = torch.as_tensor(cold_ids, dtype=torch.long, device=base_bank.device).view(-1)
    rows = torch.as_tensor(rows, dtype=base_bank.dtype, device=base_bank.device)
    if rows.shape != (ids.numel(), base_bank.size(1)):
        raise ValueError("direct rollout rows must align with cold_ids")
    out = base_bank.clone()
    if ids.numel():
        out.index_copy_(0, ids, F.normalize(rows, dim=1))
    return out


def direct_projected_bank(
    base_bank: torch.Tensor,
    raw_rows: torch.Tensor,
    cold_ids: Sequence[int] | torch.Tensor,
    *,
    fixed_max_delta: float,
) -> torch.Tensor:
    """Project a control trajectory once at the fixed protocol cap and insert it."""
    ids = torch.as_tensor(cold_ids, dtype=torch.long, device=base_bank.device).view(-1)
    initial = base_bank.index_select(0, ids)
    raw_rows = torch.as_tensor(raw_rows, dtype=base_bank.dtype, device=base_bank.device)
    if raw_rows.shape != initial.shape:
        raise ValueError("control trajectory rows must align with cold_ids")
    projected = project_displacement(initial, raw_rows, max_delta=float(fixed_max_delta))
    return insert_cold_rows(base_bank, projected, ids)


def _load_train_val(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, val_path = split_dir / "static_train.pkl", split_dir / "static_val.pkl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"missing train/validation split under {split_dir}")
    return pd.read_pickle(train_path).copy(), pd.read_pickle(val_path).copy()


def _load_test(split_dir: Path) -> pd.DataFrame:
    path = split_dir / "static_test.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing test split: {path}")
    return pd.read_pickle(path).copy()


def _positive_users_by_item(train_df: pd.DataFrame) -> dict[int, torch.Tensor]:
    out: dict[int, set[int]] = {}
    for u, i in zip(train_df["u_idx"].astype(int), train_df["i_idx"].astype(int)):
        out.setdefault(int(i), set()).add(int(u))
    return {i: torch.tensor(sorted(users), dtype=torch.long) for i, users in out.items()}


def _build_backbone(
    data_dir: Path,
    split_dir: Path,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    content = torch.load(data_dir / "content_emb.pt", map_location="cpu", weights_only=False)
    content = content if isinstance(content, torch.Tensor) else torch.as_tensor(content)
    content = content.float()
    train_df, val_df = _load_train_val(split_dir)
    train_pop = np.asarray(compute_train_popularity(train_df, int(meta["n_items"])))
    cfg = GraphScorerConfig(int(meta["n_users"]), int(meta["n_items"]), int(content.shape[1]))
    cfg.prereq_aux_weight = 0.0
    model = GraphContentScorer(cfg, content).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.set_item_degree(train_pop)
    model.eval()
    user_seen = build_user_seen(train_df)
    cold_mask = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool, device=device)
    cold_ids = torch.nonzero(cold_mask, as_tuple=False).view(-1)
    adj = build_norm_adj(train_df, cfg.n_users, cfg.n_items, device)
    with torch.no_grad():
        ego, _, _ = model.item_ego(apply_id_dropout=False, cold_mask_all=cold_mask)
        z_u_all, z_i_all = model.propagate(adj, ego)
    base_user_bank = F.normalize(z_u_all, dim=1)
    base_item_bank = F.normalize(z_i_all, dim=1)
    return {
        "meta": meta,
        "content": content,
        "train_df": train_df,
        "val_df": val_df,
        "train_pop": train_pop,
        "cfg": cfg,
        "model": model,
        "user_seen": user_seen,
        "cold_mask": cold_mask,
        "cold_ids": cold_ids,
        "adj": adj,
        "z_u_all": z_u_all,
        "base_user_bank": base_user_bank,
        "base_item_bank": base_item_bank,
    }


def _shared_pseudo_user_bank(bundle: Mapping[str, Any], target_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    model = bundle["model"]
    target = torch.as_tensor(target_ids, dtype=torch.long, device=bundle["adj"].device)
    masked_adj = pseudo_cold_adjacency(
        bundle["adj"], n_users=bundle["cfg"].n_users, target_item_ids=target
    )
    idx = masked_adj.indices()
    node_ids = bundle["cfg"].n_users + target
    if node_ids.numel():
        if bool(torch.isin(idx[0], node_ids).any()) or bool(torch.isin(idx[1], node_ids).any()):
            raise AssertionError("pseudo-cold target still has a train edge")
    pseudo_mask = bundle["cold_mask"].clone()
    pseudo_mask[target] = True
    with torch.no_grad():
        ego, _, _ = model.item_ego(False, pseudo_mask)
        users, _ = model.propagate(masked_adj, ego)
    return F.normalize(users, dim=1), masked_adj


def _make_engine(
    *,
    cfg: GraphScorerConfig,
    run_cfg: CleanRunConfig,
    course_signal: Any,
    use_course_bias: bool,
    with_course_reward: bool,
    max_delta: float,
    course_bias_signal: Any = None,
    course_reward_scale: float = 1.0,
    center_course_reward: bool = False,
    reward_geometry: str | None = None,
    embedding_reward_weight: float = 1.0,
    recommendation_reward_weight: float = 1.0,
) -> CleanUSIMEngine:
    if float(course_reward_scale) < 0.0:
        raise ValueError("course_reward_scale must be non-negative")
    bias_signal = course_signal if course_bias_signal is None else course_bias_signal
    reward_fn = None
    baseline_fn = None
    if course_signal is not None and with_course_reward:
        if float(course_reward_scale) == 1.0:
            reward_fn = course_signal.reward
        else:
            def reward_fn(selected_user_ids, item_ids, histories):
                raw = course_signal.reward(selected_user_ids, item_ids, histories)
                return torch.as_tensor(raw) * float(course_reward_scale)
        if center_course_reward:
            def baseline_fn(candidate_ids, item_ids, histories):
                raw = course_signal.candidate_reward_baseline(
                    candidate_ids, item_ids, histories
                )
                return torch.as_tensor(raw) * float(course_reward_scale)
    return CleanUSIMEngine(
        emb_dim=int(cfg.emb_dim),
        hidden_dim=256,
        max_steps=run_cfg.max_steps,
        candidate_count=run_cfg.candidate_count,
        step_size=run_cfg.step_size,
        step_penalty=run_cfg.step_penalty,
        max_delta=float(max_delta),
        retrieval_chunk=run_cfg.retrieval_chunk,
        course_bias_fn=(
            None if bias_signal is None or not use_course_bias
            else bias_signal.candidate_bias
        ),
        course_reward_fn=reward_fn,
        course_reward_baseline_fn=baseline_fn,
        reward_geometry=reward_geometry,
        embedding_reward_weight=float(embedding_reward_weight),
        recommendation_reward_weight=float(recommendation_reward_weight),
        # Read off run_cfg rather than threaded through every call site, so the
        # three pre-existing null selectors keep their exact argument lists.
        allow_end_action=bool(getattr(run_cfg, "allow_end_action", True)),
    )


@torch.no_grad()
@torch.no_grad()
def random_policy_walk(
    engine: Any,
    base_bank: torch.Tensor,
    cold_ids: torch.Tensor,
    user_bank: torch.Tensor,
    *,
    generator: torch.Generator,
    item_batch: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform-random-direction null for the displacement mechanism.

    Identical to ``course_fit_walk`` except the per-step direction is drawn
    uniformly from the retrieved candidates instead of by course-fit argmax, and
    no course signal is consulted at all.  It isolates "does the learned policy
    matter" from "does displacing the cold row at all matter"; the trained-policy
    arms are only meaningful if they beat this.

    Returns ``(init, raw_final_state)``, both aligned with ``cold_ids``.
    """
    inits, finals = [], []
    for start in range(0, int(cold_ids.numel()), int(item_batch)):
        ids = cold_ids[start:start + int(item_batch)]
        init = base_bank.index_select(0, ids)
        state = init
        for _ in range(int(engine.max_steps)):
            cand_ids = engine.legal_candidate_ids(state, user_bank)
            cands = engine._candidate_vectors(user_bank, cand_ids)
            sel = torch.randint(
                cands.size(1), (state.size(0),), generator=generator
            ).to(state.device)
            rows = torch.arange(state.size(0), device=state.device)
            state = state + float(engine.step_size) * cands[rows, sel]
        inits.append(init)
        finals.append(state)
    if not inits:
        empty = base_bank.new_empty((0, base_bank.size(1)))
        return empty, empty
    return torch.cat(inits), torch.cat(finals)


@torch.no_grad()
def centroid_step_walk(
    engine: Any,
    base_bank: torch.Tensor,
    cold_ids: torch.Tensor,
    user_bank: torch.Tensor,
    *,
    item_batch: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic mean-shift null: step along the candidate centroid.

    Identical to ``random_policy_walk`` except the per-step direction is the
    normalised mean of the retrieved candidates instead of one sampled candidate.

    Measured on MOOCCube seed2025 (cosine geometry): the random arm's realised
    displacement sits at cosine 0.9646 to that centroid, a trained argmax policy
    at 0.8614 (15.3 deg vs 30.5 deg off-centroid). Uniform sampling is an unbiased
    estimator of the centroid and argmax is not, so this arm names the target
    directly and removes the estimator's sampling noise. It is the strongest null
    available: no training, no policy, no course signal, one direction per step.
    """
    inits, finals = [], []
    for start in range(0, int(cold_ids.numel()), int(item_batch)):
        ids = cold_ids[start:start + int(item_batch)]
        init = base_bank.index_select(0, ids)
        state = init
        for _ in range(int(engine.max_steps)):
            cand_ids = engine.legal_candidate_ids(state, user_bank)
            cands = engine._candidate_vectors(user_bank, cand_ids)
            direction = F.normalize(F.normalize(cands, dim=2).mean(dim=1), dim=1)
            state = state + float(engine.step_size) * direction
        inits.append(init)
        finals.append(state)
    if not inits:
        empty = base_bank.new_empty((0, base_bank.size(1)))
        return empty, empty
    return torch.cat(inits), torch.cat(finals)


@torch.no_grad()
def global_shift_walk(
    engine: Any,
    base_bank: torch.Tensor,
    cold_ids: torch.Tensor,
    user_bank: torch.Tensor,
    *,
    item_batch: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One shared direction for every cold row: the normalised user-bank mean.

    Strictly weaker than ``centroid_step_walk``, which still retrieves each row's
    own candidates. Here no cold row sees anything specific to itself, so the arm
    cannot express "move this course toward these users" -- only "move every cold
    course the same way". Spends the same budget as the other walks
    (``step_size x max_steps``) so the delta sweep is comparable.
    """
    direction = F.normalize(
        F.normalize(user_bank, dim=1).mean(dim=0, keepdim=True), dim=1
    )
    budget = float(engine.step_size) * int(engine.max_steps)
    inits, finals = [], []
    for start in range(0, int(cold_ids.numel()), int(item_batch)):
        ids = cold_ids[start:start + int(item_batch)]
        init = base_bank.index_select(0, ids)
        inits.append(init)
        finals.append(init + budget * direction.to(init.device).expand_as(init))
    if not inits:
        empty = base_bank.new_empty((0, base_bank.size(1)))
        return empty, empty
    return torch.cat(inits), torch.cat(finals)


@torch.no_grad()
def norm_only_walk(
    engine: Any,
    base_bank: torch.Tensor,
    cold_ids: torch.Tensor,
    user_bank: torch.Tensor,
    *,
    item_batch: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-hubness null: scale the cold row radially, direction unchanged.

    The displacement is ``budget * init/|init|``, so after the delta projection the
    row is exactly ``init * (1 + delta/|init|)`` -- same direction, larger norm,
    hence a uniform lift of its inner product with every user. ``user_bank`` is
    accepted only to match the other walks' signature and is deliberately unused:
    this arm consults nothing at all. If this recovers the displacement gain, the
    mechanism is a norm/hubness mean shift and no direction, learned or otherwise,
    is doing work.
    """
    budget = float(engine.step_size) * int(engine.max_steps)
    inits, finals = [], []
    for start in range(0, int(cold_ids.numel()), int(item_batch)):
        ids = cold_ids[start:start + int(item_batch)]
        init = base_bank.index_select(0, ids)
        inits.append(init)
        finals.append(init + budget * F.normalize(init, dim=1))
    if not inits:
        empty = base_bank.new_empty((0, base_bank.size(1)))
        return empty, empty
    return torch.cat(inits), torch.cat(finals)


def _rollout_bank(
    engine: CleanUSIMEngine,
    initial_bank: torch.Tensor,
    item_ids: torch.Tensor,
    user_bank: torch.Tensor,
    user_history: Mapping[int, set[int]],
    *,
    batch_size: int = 16,
    training: bool = False,
    state_source: str = "raw",
) -> tuple[torch.Tensor, dict[str, float]]:
    if state_source not in {"raw", "projected"}:
        raise ValueError("state_source must be 'raw' or 'projected'")
    states, stats = [], []
    for start in range(0, item_ids.numel(), int(batch_size)):
        ids = item_ids[start:start + int(batch_size)]
        initial = initial_bank.index_select(0, ids)
        result = engine.rollout(
            initial,
            user_bank=user_bank,
            training=training,
            item_ids=ids,
            user_history=user_history,
        )
        state = result.raw_final_state if state_source == "raw" else result.final_state
        states.append(state.detach())
        stats.append(result.stats)
    if not states:
        return initial_bank.new_empty((0, initial_bank.size(1))), {}
    keys = stats[0].keys()
    return torch.cat(states), {k: float(np.mean([s[k] for s in stats])) for k in keys}


@torch.no_grad()
def selected_cold_bank(
    *,
    engine: CleanUSIMEngine | None,
    ridge_bank: torch.Tensor,
    cold_ids: torch.Tensor,
    user_bank: torch.Tensor,
    user_history: Mapping[int, set[int]],
    selected_epoch: int,
    selected_delta: float | None = None,
    batch_size: int = 16,
    selection_mode: str = "delta_grid",
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Return the identity Ridge bank at epoch zero, otherwise PPO refinement."""
    if int(selected_epoch) == 0:
        return ridge_bank.clone(), {
            "policy_mode": "identity_generator",
            "rollout_delta_l2": 0.0,
            "end_rate": 1.0,
            "active_steps": 0.0,
        }
    if engine is None:
        raise ValueError("a trained engine is required for a positive policy epoch")
    direct = selection_mode == "direct_rollout"
    rows, stats = _rollout_bank(
        engine, ridge_bank, cold_ids, user_bank, user_history,
        batch_size=batch_size, training=False,
        state_source="projected" if direct else "raw",
    )
    if direct:
        bank = insert_cold_rows(ridge_bank, rows, cold_ids)
        return bank, {"policy_mode": "ppo_direct_rollout", **stats}
    if selected_delta is None:
        raise ValueError("delta-grid selection requires selected_delta")
    bank = replace_cold_rows(ridge_bank, rows, cold_ids, max_delta=float(selected_delta))
    return bank, {"policy_mode": "ppo_rollout", **stats}


def _policy_epoch_state(
    engine: CleanUSIMEngine,
    state: dict[str, torch.Tensor],
) -> None:
    engine.policy.load_state_dict(state)
    engine.policy.eval()


def _train_policy(
    *,
    arm: PolicyArmSpec,
    bundle: Mapping[str, Any],
    sim_bank: torch.Tensor,
    pseudo_user_bank: torch.Tensor,
    policy_train_ids: Sequence[int],
    policy_val_ids: Sequence[int],
    positives: Mapping[int, torch.Tensor],
    course_signal: Any,
    course_bias_signal: Any,
    course_reward_scale: float,
    center_course_reward: bool,
    reward_geometry: str | None = None,
    embedding_reward_weight: float = 1.0,
    recommendation_reward_weight: float = 1.0,
    no_epoch_selection: bool = False,
    run_cfg: CleanRunConfig,
    delta_grid: Sequence[float],
    selection_mode: str,
    fixed_max_delta: float,
    max_epochs: int,
    batch_size: int,
    val_row_limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    device = bundle["base_item_bank"].device
    direct = selection_mode == "direct_rollout"
    rollout_cap = float(fixed_max_delta) if direct else max(delta_grid)
    engine = _make_engine(
        cfg=bundle["cfg"], run_cfg=run_cfg, course_signal=course_signal,
        use_course_bias=arm.use_course_bias,
        with_course_reward=arm.use_course_reward,
        max_delta=rollout_cap,
        course_bias_signal=course_bias_signal,
        course_reward_scale=course_reward_scale,
        center_course_reward=center_course_reward,
        reward_geometry=reward_geometry,
        embedding_reward_weight=embedding_reward_weight,
        recommendation_reward_weight=recommendation_reward_weight,
    )
    engine.policy.to(device)
    optimizer = torch.optim.Adam(engine.policy.parameters(), lr=float(run_cfg.policy_lr))
    ppo = CleanRecPPO(
        engine.policy,
        replay_capacity=run_cfg.replay_capacity,
        replay_batch_size=run_cfg.replay_batch_size,
        gamma=run_cfg.ppo_gamma,
        clip_ratio=run_cfg.ppo_clip_ratio,
        value_weight=run_cfg.ppo_value_weight,
        terminal_value_weight=run_cfg.ppo_terminal_value_weight,
        entropy_weight=run_cfg.ppo_entropy_weight,
    )
    user_history = bundle["user_seen"]
    factual = bundle["base_item_bank"]
    all_ids = torch.as_tensor(policy_val_ids, dtype=torch.long, device=device)
    train_ids = [int(x) for x in policy_train_ids]
    state_by_epoch = {0: _copy_state(engine.policy)}
    train_stats_by_epoch = {
        0: summarize_rollout_stats(
            [], step_penalty=run_cfg.step_penalty, max_steps=run_cfg.max_steps
        )
    }
    epoch_rows: list[dict[str, Any]] = []

    def evaluate_epoch(epoch: int) -> None:
        _policy_epoch_state(engine, state_by_epoch[epoch])
        if epoch == 0:
            pseudo_final = sim_bank.index_select(0, all_ids)
            rollout_stats = {
                "policy_mode": "identity_generator",
                "rollout_delta_l2": 0.0,
                "end_rate": 1.0,
                "active_steps": 0.0,
            }
            cold_raw = bundle["ridge_bank"].index_select(0, bundle["cold_ids"])
            cold_stats = dict(rollout_stats)
        else:
            pseudo_final, rollout_stats = _rollout_bank(
                engine, sim_bank, all_ids, pseudo_user_bank, user_history,
                batch_size=batch_size, training=False,
                state_source="projected" if direct else "raw",
            )
            cold_raw, cold_stats = _rollout_bank(
                engine, bundle["ridge_bank"], bundle["cold_ids"], bundle["base_user_bank"],
                user_history, batch_size=batch_size, training=False,
                state_source="projected" if direct else "raw",
            )
        target = factual.index_select(0, all_ids)
        pseudo_cos = float((F.normalize(pseudo_final, dim=1) * target).sum(dim=1).mean().item())
        cold_ids = bundle["cold_ids"]
        rows = []
        val_df = bundle["val_df"]
        if val_row_limit is not None:
            val_df = val_df.iloc[:int(val_row_limit)].copy()
        candidate_deltas: Sequence[float | None] = (None,) if direct else delta_grid
        for delta in candidate_deltas:
            bank = (
                insert_cold_rows(bundle["ridge_bank"], cold_raw, cold_ids)
                if direct
                else replace_cold_rows(
                    bundle["ridge_bank"], cold_raw, cold_ids, max_delta=float(delta)
                )
            )
            metrics = evaluate_with_banks(
                bundle["z_u_all"], bank, val_df, device,
                bundle["train_pop"], user_history,
            )
            row = {"epoch": int(epoch), **metrics}
            if delta is not None:
                row["delta"] = float(delta)
            rows.append(row)
        epoch_rows.extend(rows)
        print(
            f"[{arm.name}] epoch={epoch} pseudo_cos={pseudo_cos:.5f} "
            f"rollout_delta={rollout_stats.get('rollout_delta_l2', 0.0):.5f} "
            f"cold_raw_delta={cold_stats.get('rollout_delta_l2', 0.0):.5f}",
            flush=True,
        )
        for row in rows:
            point = "direct" if direct else f"d{row['delta']:g}"
            print(
                f"[{arm.name}] e{epoch} {point} "
                f"val c={row['cold_N@10']:.5f} h={row['hot_N@10']:.5f} "
                f"o={row['overall_R@10']:.5f}", flush=True,
            )
        epoch_rows[-len(rows):] = [dict(r, pseudo_val_cos=pseudo_cos,
                                        pseudo_rollout_stats=rollout_stats,
                                        cold_rollout_stats=cold_stats,
                                        train_rollout_stats=train_stats_by_epoch[epoch])
                                   for r in epoch_rows[-len(rows):]]

    evaluate_epoch(0)
    for epoch in range(1, int(max_epochs) + 1):
        engine.policy.train()
        losses = []
        rollout_batches: list[tuple[int, Mapping[str, float]]] = []
        order = np.random.default_rng(run_cfg.seed + epoch).permutation(train_ids)
        for start in range(0, len(order), int(batch_size)):
            ids_np = order[start:start + int(batch_size)]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            initial = sim_bank.index_select(0, ids)
            target = factual.index_select(0, ids).detach()
            positive_users = [positives[int(i)] for i in ids_np.tolist()]
            rollout = engine.rollout(
                initial,
                user_bank=pseudo_user_bank,
                training=True,
                target_emb=target,
                positive_user_ids=positive_users,
                item_ids=ids,
                user_history=user_history,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = ppo.loss(rollout.trajectory, pseudo_user_bank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.policy.parameters(), max_norm=5.0)
            optimizer.step()
            ppo.sync_target()
            losses.append(float(loss.detach().item()))
            rollout_batches.append((int(ids.numel()), rollout.stats))
        state_by_epoch[epoch] = _copy_state(engine.policy)
        train_stats_by_epoch[epoch] = summarize_rollout_stats(
            rollout_batches,
            step_penalty=run_cfg.step_penalty,
            max_steps=run_cfg.max_steps,
        )
        diag = train_stats_by_epoch[epoch]
        print(
            f"[{arm.name}] epoch={epoch} train_loss={np.mean(losses):.6f} "
            f"reward(embed={diag['embedding_reward']:.6f},"
            f"rec={diag['recommendation_reward']:.6f},"
            f"course={diag['course_reward']:.6f},"
            f"step={diag['step_penalty_reward']:.6f})",
            flush=True,
        )
        evaluate_epoch(epoch)

    # The PPO arm otherwise selects over policy_epochs x len(delta_grid) = 90
    # combinations on 34 validation cold courses while every zero-training null
    # selects over 15. That asymmetry is the measured source of the arm's
    # val->test drop (PPO -0.0006..-0.0041 vs the nulls' +0.0002/-0.0003).
    # Restricting to the final epoch equalises the budget. The epoch-0/delta-0
    # row is kept because select_policy_row falls back to it when the retention
    # gate rejects every candidate.
    rows_for_selection = list(epoch_rows)
    if bool(no_epoch_selection):
        last_epoch = max(int(row["epoch"]) for row in rows_for_selection)
        rows_for_selection = [
            row
            for row in rows_for_selection
            if int(row["epoch"]) == last_epoch
            or (int(row["epoch"]) == 0 and float(row.get("delta", 0.0)) == 0.0)
        ]
    selected = select_policy_row(
        rows_for_selection,
        bundle["retention_val_metrics"],
        tolerance=run_cfg.hot_retention_tolerance,
    )
    _policy_epoch_state(engine, state_by_epoch[int(selected["epoch"])])
    selected_epoch = int(selected["epoch"])
    result = {
        "arm": arm.name,
        "arm_config": {
            "use_course_bias": arm.use_course_bias,
            "use_course_reward": arm.use_course_reward,
        },
        "engine": engine,
        "selected": selected,
        "selected_epoch": selected_epoch,
        "selection_mode": selection_mode,
        "fixed_max_delta": rollout_cap,
        "selected_train_rollout_stats": train_stats_by_epoch[selected_epoch],
        "train_rollout_stats_by_epoch": train_stats_by_epoch,
        "epoch_rows": epoch_rows,
        "policy_state": _copy_state(engine.policy),
    }
    if not direct:
        result["selected_delta"] = float(selected["delta"])
    return result


def _select_greedy(
    *,
    bundle: Mapping[str, Any],
    course_signal: Any,
    run_cfg: CleanRunConfig,
    delta_grid: Sequence[float],
    selection_mode: str,
    fixed_max_delta: float,
    val_row_limit: int | None,
) -> dict[str, Any]:
    device = bundle["base_item_bank"].device
    direct = selection_mode == "direct_rollout"
    rollout_cap = float(fixed_max_delta) if direct else max(delta_grid)
    engine = _make_engine(
        cfg=bundle["cfg"], run_cfg=run_cfg, course_signal=course_signal,
        use_course_bias=True, with_course_reward=False, max_delta=rollout_cap,
    )
    engine.policy.to(device)
    engine.policy.eval()
    cold_ids = bundle["cold_ids"]
    init, raw, _, _ = course_fit_walk(
        engine, bundle["ridge_bank"], cold_ids, bundle["base_user_bank"], bundle["user_seen"]
    )
    val_df = bundle["val_df"] if val_row_limit is None else bundle["val_df"].iloc[:int(val_row_limit)]
    rows = []
    banks = []
    candidate_deltas: Sequence[float | None] = (None,) if direct else delta_grid
    for delta in candidate_deltas:
        bank = (
            direct_projected_bank(
                bundle["ridge_bank"], raw, cold_ids, fixed_max_delta=rollout_cap
            )
            if direct
            else replace_cold_rows(
                bundle["ridge_bank"], raw, cold_ids, max_delta=float(delta)
            )
        )
        metrics = evaluate_with_banks(
            bundle["z_u_all"], bank, val_df, device,
            bundle["train_pop"], bundle["user_seen"],
        )
        row = dict(metrics)
        if delta is not None:
            row["delta"] = float(delta)
        rows.append(row)
        banks.append(bank)
    retention_val = bundle["retention_val_metrics"]
    feasible = [
        r for r in rows
        if passes_retention_gate(
            r, retention_val, tolerance=run_cfg.hot_retention_tolerance
        )
    ]
    selected = max(
        feasible or [rows[0]],
        key=lambda r: (r["cold_N@10"], -float(r.get("delta", 0.0))),
    )
    selected_index = rows.index(selected)
    result = {
        "arm": "ridge_greedy_course_fit", "engine": engine, "raw": raw,
        "bank": banks[selected_index], "selected": selected,
        "selection_mode": selection_mode, "fixed_max_delta": rollout_cap,
        "val_rows": rows,
    }
    if not direct:
        result["selected_delta"] = float(selected["delta"])
    return result


def _select_random_policy(
    *,
    bundle: Mapping[str, Any],
    run_cfg: CleanRunConfig,
    delta_grid: Sequence[float],
    selection_mode: str,
    fixed_max_delta: float,
    val_row_limit: int | None,
    seed: int,
) -> dict[str, Any]:
    """Zero-information displacement null: random direction, no course signal."""
    device = bundle["base_item_bank"].device
    direct = selection_mode == "direct_rollout"
    rollout_cap = float(fixed_max_delta) if direct else max(delta_grid)
    engine = _make_engine(
        cfg=bundle["cfg"], run_cfg=run_cfg, course_signal=None,
        use_course_bias=False, with_course_reward=False, max_delta=rollout_cap,
    )
    engine.policy.to(device)
    engine.policy.eval()
    cold_ids = bundle["cold_ids"]
    _, raw = random_policy_walk(
        engine, bundle["ridge_bank"], cold_ids, bundle["base_user_bank"],
        generator=torch.Generator().manual_seed(int(seed)),
    )
    val_df = bundle["val_df"] if val_row_limit is None else bundle["val_df"].iloc[:int(val_row_limit)]
    rows = []
    banks = []
    candidate_deltas: Sequence[float | None] = (None,) if direct else delta_grid
    for delta in candidate_deltas:
        bank = (
            direct_projected_bank(
                bundle["ridge_bank"], raw, cold_ids, fixed_max_delta=rollout_cap
            )
            if direct
            else replace_cold_rows(
                bundle["ridge_bank"], raw, cold_ids, max_delta=float(delta)
            )
        )
        metrics = evaluate_with_banks(
            bundle["z_u_all"], bank, val_df, device,
            bundle["train_pop"], bundle["user_seen"],
        )
        row = dict(metrics)
        if delta is not None:
            row["delta"] = float(delta)
        rows.append(row)
        banks.append(bank)
    retention_val = bundle["retention_val_metrics"]
    feasible = [
        r for r in rows
        if passes_retention_gate(
            r, retention_val, tolerance=run_cfg.hot_retention_tolerance
        )
    ]
    selected = max(
        feasible or [rows[0]],
        key=lambda r: (r["cold_N@10"], -float(r.get("delta", 0.0))),
    )
    selected_index = rows.index(selected)
    result = {
        "arm": "ridge_random_policy", "engine": engine, "raw": raw,
        "bank": banks[selected_index], "selected": selected,
        "selection_mode": selection_mode, "fixed_max_delta": rollout_cap,
        "val_rows": rows,
    }
    if not direct:
        result["selected_delta"] = float(selected["delta"])
    return result


def _select_centroid_step(
    *,
    bundle: Mapping[str, Any],
    run_cfg: CleanRunConfig,
    delta_grid: Sequence[float],
    selection_mode: str,
    fixed_max_delta: float,
    val_row_limit: int | None,
) -> dict[str, Any]:
    """Deterministic mean-shift null; see centroid_step_walk for the measurement."""
    device = bundle["base_item_bank"].device
    direct = selection_mode == "direct_rollout"
    rollout_cap = float(fixed_max_delta) if direct else max(delta_grid)
    engine = _make_engine(
        cfg=bundle["cfg"], run_cfg=run_cfg, course_signal=None,
        use_course_bias=False, with_course_reward=False, max_delta=rollout_cap,
    )
    engine.policy.to(device)
    engine.policy.eval()
    cold_ids = bundle["cold_ids"]
    _, raw = centroid_step_walk(
        engine, bundle["ridge_bank"], cold_ids, bundle["base_user_bank"]
    )
    val_df = bundle["val_df"] if val_row_limit is None else bundle["val_df"].iloc[:int(val_row_limit)]
    rows = []
    banks = []
    candidate_deltas: Sequence[float | None] = (None,) if direct else delta_grid
    for delta in candidate_deltas:
        bank = (
            direct_projected_bank(
                bundle["ridge_bank"], raw, cold_ids, fixed_max_delta=rollout_cap
            )
            if direct
            else replace_cold_rows(
                bundle["ridge_bank"], raw, cold_ids, max_delta=float(delta)
            )
        )
        metrics = evaluate_with_banks(
            bundle["z_u_all"], bank, val_df, device,
            bundle["train_pop"], bundle["user_seen"],
        )
        row = dict(metrics)
        if delta is not None:
            row["delta"] = float(delta)
        rows.append(row)
        banks.append(bank)
    retention_val = bundle["retention_val_metrics"]
    feasible = [
        r for r in rows
        if passes_retention_gate(
            r, retention_val, tolerance=run_cfg.hot_retention_tolerance
        )
    ]
    selected = max(
        feasible or [rows[0]],
        key=lambda r: (r["cold_N@10"], -float(r.get("delta", 0.0))),
    )
    selected_index = rows.index(selected)
    result = {
        "arm": "ridge_centroid_step", "engine": engine, "raw": raw,
        "bank": banks[selected_index], "selected": selected,
        "selection_mode": selection_mode, "fixed_max_delta": rollout_cap,
        "val_rows": rows,
    }
    if not direct:
        result["selected_delta"] = float(selected["delta"])
    return result


def _select_displacement_null(
    *,
    arm: str,
    walk: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    bundle: Mapping[str, Any],
    run_cfg: CleanRunConfig,
    delta_grid: Sequence[float],
    selection_mode: str,
    fixed_max_delta: float,
    val_row_limit: int | None,
) -> dict[str, Any]:
    """Shared selector for the direction-free nulls (`global_shift`, `norm_only`).

    Same protocol as ``_select_random_policy``: course-free engine, one walk, then
    the delta grid swept under the hot-retention gate. Written as one function
    instead of two more copies, and deliberately NOT applied to the three existing
    null selectors -- those keep their exact code so their numbers stay comparable
    with earlier batches. Neither walk consumes randomness, so no generator is
    threaded in.
    """
    device = bundle["base_item_bank"].device
    direct = selection_mode == "direct_rollout"
    rollout_cap = float(fixed_max_delta) if direct else max(delta_grid)
    engine = _make_engine(
        cfg=bundle["cfg"], run_cfg=run_cfg, course_signal=None,
        use_course_bias=False, with_course_reward=False, max_delta=rollout_cap,
    )
    engine.policy.to(device)
    engine.policy.eval()
    cold_ids = bundle["cold_ids"]
    _, raw = walk(engine, bundle["ridge_bank"], cold_ids, bundle["base_user_bank"])
    val_df = (
        bundle["val_df"] if val_row_limit is None
        else bundle["val_df"].iloc[:int(val_row_limit)]
    )
    rows: list[dict[str, Any]] = []
    banks: list[torch.Tensor] = []
    candidate_deltas: Sequence[float | None] = (None,) if direct else delta_grid
    for delta in candidate_deltas:
        bank = (
            direct_projected_bank(
                bundle["ridge_bank"], raw, cold_ids, fixed_max_delta=rollout_cap
            )
            if direct
            else replace_cold_rows(
                bundle["ridge_bank"], raw, cold_ids, max_delta=float(delta)
            )
        )
        metrics = evaluate_with_banks(
            bundle["z_u_all"], bank, val_df, device,
            bundle["train_pop"], bundle["user_seen"],
        )
        row = dict(metrics)
        if delta is not None:
            row["delta"] = float(delta)
        rows.append(row)
        banks.append(bank)
    feasible = [
        r for r in rows
        if passes_retention_gate(
            r, bundle["retention_val_metrics"],
            tolerance=run_cfg.hot_retention_tolerance,
        )
    ]
    selected = max(
        feasible or [rows[0]],
        key=lambda r: (r["cold_N@10"], -float(r.get("delta", 0.0))),
    )
    result = {
        "arm": arm, "engine": engine, "raw": raw,
        "bank": banks[rows.index(selected)], "selected": selected,
        "selection_mode": selection_mode, "fixed_max_delta": rollout_cap,
        "val_rows": rows,
    }
    if not direct:
        result["selected_delta"] = float(selected["delta"])
    return result


def _uniform_bias_frontier(
    z_u: torch.Tensor,
    bank: torch.Tensor,
    cold_mask: torch.Tensor,
    eval_df: pd.DataFrame,
    device: torch.device,
    train_pop: np.ndarray,
    user_seen: Mapping[int, set[int]],
    biases: Sequence[float] = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16),
) -> list[dict[str, float]]:
    if z_u.ndim != 2 or bank.ndim != 2 or z_u.size(1) != bank.size(1):
        raise ValueError("z_u and bank must be 2-D with a shared embedding width")
    if cold_mask.view(-1).numel() != bank.size(0):
        raise ValueError("cold_mask must have exactly one entry per bank row")
    col = cold_mask.to(bank.dtype).view(-1, 1)
    bank_aug = torch.cat([bank, col], dim=1)
    out = []
    for bias in biases:
        users_aug = torch.cat([
            z_u,
            torch.full((z_u.size(0), 1), float(bias), dtype=z_u.dtype, device=device),
        ], dim=1)
        m = evaluate_with_banks(users_aug, bank_aug, eval_df, device, train_pop, user_seen)
        out.append({"bias": float(bias), "cold_N@10": m["cold_N@10"],
                    "hot_N@10": m["hot_N@10"], "overall_N@10": m["overall_N@10"]})
    return out


def _cold_at_hot(frontier: Sequence[Mapping[str, float]], hot: float) -> tuple[float, bool]:
    points = sorted((float(x["hot_N@10"]), float(x["cold_N@10"])) for x in frontier)
    if hot <= points[0][0]:
        return points[0][1], True
    if hot >= points[-1][0]:
        return points[-1][1], True
    for (h0, c0), (h1, c1) in zip(points, points[1:]):
        if h0 <= hot <= h1:
            t = 0.0 if h1 == h0 else (hot - h0) / (h1 - h0)
            return c0 + t * (c1 - c0), False
    return points[-1][1], True


def pilot_verdict(
    *,
    selected_epoch: int,
    cold_delta: float,
    matched_hot_delta: float,
    matched_hot_clamped: bool,
) -> str:
    """Classify a pilot without promoting a single-seed result to a main model."""
    if int(selected_epoch) == 0:
        return "INCONCLUSIVE_IDENTITY"
    if float(cold_delta) <= 0.0:
        return "FAIL_NO_COLD_GAIN"
    if bool(matched_hot_clamped):
        return "INCONCLUSIVE_CLAMPED"
    if float(matched_hot_delta) <= 0.0:
        return "FAIL_TRADEOFF"
    return "PASS_SINGLE_SEED"


def _evaluate_arm(
    name: str,
    bank: torch.Tensor,
    bundle: Mapping[str, Any],
    test_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    metrics, per_item = evaluate_with_banks(
        bundle["z_u_all"], bank, test_df, bundle["base_item_bank"].device,
        bundle["train_pop"], bundle["user_seen"], return_per_item=True,
    )
    pd.DataFrame(per_item).to_csv(out_dir / f"per_item_{name}.csv", index=False)
    return metrics


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    _seed_everything(args.seed)
    policy_arm_specs = resolve_policy_arms(args.ppo_arms)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = _path(args.data_dir)
    split_dir = _path(args.split_root) / f"strict_item_cold_balanced_thr1_seed_{int(args.seed)}"
    checkpoint = _path(args.ckpt_root) / f"seed{int(args.seed)}" / "best.pt"
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (data_dir / "meta.json", data_dir / "content_emb.pt", checkpoint):
        if not p.is_file():
            raise FileNotFoundError(p)

    bundle = _build_backbone(data_dir, split_dir, checkpoint, device)
    warm_ids = np.flatnonzero(bundle["train_pop"] >= bundle["cfg"].cold_threshold)
    eligible = np.flatnonzero(
        (bundle["train_pop"] >= bundle["cfg"].cold_threshold)
        & (bundle["train_pop"] <= float(args.max_pseudo_pop))
    )
    policy_train, policy_val, donors = make_policy_partitions(
        warm_ids, eligible, seed=args.seed, val_fraction=args.pseudo_val_fraction
    )
    if args.max_train_items is not None:
        policy_train = policy_train[:int(args.max_train_items)]
    if args.max_val_items is not None:
        policy_val = policy_val[:int(args.max_val_items)]
    if not policy_train or not policy_val:
        raise ValueError("policy train and validation partitions must remain nonempty")
    donors = sorted(set(int(x) for x in warm_ids) - set(policy_train) - set(policy_val))

    strict_cold = bundle["cold_ids"]
    ridge_alpha = float(getattr(args, "ridge_alpha", 1.0))
    simulation_ridge_alpha = resolve_simulation_ridge_alpha(
        ridge_alpha,
        getattr(args, "simulation_ridge_alpha", None),
    )
    selection_protocol = resolve_selection_protocol(
        getattr(args, "selection_mode", "delta_grid"),
        delta_grid=getattr(args, "delta_grid", DELTA_GRID),
        fixed_max_delta=getattr(args, "fixed_max_delta", 0.25),
    )
    delta_grid = selection_protocol.delta_grid
    selection_mode = selection_protocol.selection_mode
    fixed_max_delta = selection_protocol.fixed_max_delta
    rollout_cap = fixed_max_delta if selection_mode == "direct_rollout" else max(delta_grid)
    retention_reference = str(
        getattr(args, "retention_reference", "ridge")
    ).strip().lower()
    if retention_reference not in {"ridge", "backbone"}:
        raise ValueError("retention reference must be 'ridge' or 'backbone'")
    full_ridge_bank, final_lam = fit_ridge_bank(
        bundle["content"], bundle["base_item_bank"], warm_ids=warm_ids,
        cold_ids=strict_cold, seed=args.seed,
    )
    ridge_bank = blend_ridge_rows(
        bundle["base_item_bank"], full_ridge_bank, strict_cold, alpha=ridge_alpha
    )
    sim_target_ids = sorted(set(policy_train) | set(policy_val))
    full_sim_bank, sim_lam = fit_ridge_bank(
        bundle["content"], bundle["base_item_bank"], warm_ids=donors,
        cold_ids=sim_target_ids, seed=args.seed,
    )
    sim_bank = blend_ridge_rows(
        bundle["base_item_bank"], full_sim_bank, sim_target_ids,
        alpha=simulation_ridge_alpha,
    )
    pseudo_user_bank, masked_adj = _shared_pseudo_user_bank(bundle, sim_target_ids)
    positives = _positive_users_by_item(bundle["train_df"])
    if any(int(i) not in positives for i in policy_train):
        raise ValueError("every pseudo-cold train item needs train positives")

    run_cfg = CleanRunConfig(
        seed=int(args.seed), data_dir=str(data_dir), split_dir=str(split_dir),
        output_dir=str(out_dir), checkpoint_dir=str(out_dir),
        emb_dim=int(bundle["cfg"].emb_dim), hidden_dim=256,
        policy_epochs=int(args.policy_epochs), batch_size=int(args.policy_batch_size),
        eval_batch_size=int(args.eval_batch_users), candidate_count=int(args.candidate_count),
        retrieval_chunk=int(args.retrieval_chunk), max_steps=int(args.max_steps),
        step_size=float(args.step_size), step_penalty=float(args.step_penalty),
        max_delta=rollout_cap, replay_capacity=int(args.replay_capacity),
        allow_end_action=not bool(getattr(args, "no_end_action", False)),
        replay_batch_size=int(args.replay_batch_size), policy_lr=float(args.policy_lr),
        hot_retention_tolerance=float(args.hot_tolerance), use_course_signal=True,
        course_relation_dir=str(_path(args.course_relation_dir)),
        course_bias_scale=float(args.course_bias_scale),
        course_concept_weight=float(args.course_concept_weight),
        course_prereq_weight=float(args.course_prereq_weight),
        course_difficulty_weight=float(args.course_difficulty_weight),
        course_redundant_weight=float(args.course_redundant_weight),
        ppo_entropy_weight=float(args.ppo_entropy_weight),
    )
    course_signal, course_stats = build_clean_course_signal(
        bundle["train_df"], n_items=bundle["cfg"].n_items, config=run_cfg
    )
    # Keep the observable candidate bias fixed at the reference signal while
    # leave-one-out runs change only the scalar training reward.
    bias_cfg = replace(
        run_cfg,
        course_concept_weight=0.04,
        course_prereq_weight=0.08,
        course_difficulty_weight=0.03,
        course_redundant_weight=0.02,
    )
    course_bias_signal, course_bias_stats = build_clean_course_signal(
        bundle["train_df"], n_items=bundle["cfg"].n_items, config=bias_cfg
    )
    with torch.no_grad():
        ridge_val_bank = ridge_bank.clone()
    val_df = bundle["val_df"]
    if args.max_val_rows is not None:
        val_df = val_df.iloc[:int(args.max_val_rows)].copy()
    ridge_val_metrics = evaluate_with_banks(
        bundle["z_u_all"], ridge_val_bank, val_df, device,
        bundle["train_pop"], bundle["user_seen"],
    )
    backbone_val_metrics = evaluate_with_banks(
        bundle["z_u_all"], bundle["base_item_bank"], val_df, device,
        bundle["train_pop"], bundle["user_seen"],
    )
    retention_val_metrics = (
        backbone_val_metrics
        if retention_reference == "backbone"
        else ridge_val_metrics
    )
    if retention_reference == "backbone" and not passes_retention_gate(
        ridge_val_metrics,
        backbone_val_metrics,
        tolerance=float(args.hot_tolerance),
    ):
        raise ValueError("anchored Ridge already exceeds the Backbone retention budget")
    bundle = dict(bundle)
    bundle["ridge_bank"] = ridge_bank
    bundle["ridge_val_metrics"] = ridge_val_metrics
    bundle["backbone_val_metrics"] = backbone_val_metrics
    bundle["retention_val_metrics"] = retention_val_metrics

    frozen_config = dict(vars(args))
    frozen_config["delta_grid"] = list(delta_grid)
    frozen_config["selection_mode"] = selection_mode
    frozen_config["fixed_max_delta"] = fixed_max_delta
    manifest: dict[str, Any] = {
        "method": "ridge_initialized_course_reward_ppo_pilot",
        "seed": int(args.seed), "device": str(device),
        "data_dir": str(data_dir), "split_dir": str(split_dir),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "split_hashes": _protocol_split_hashes(
            split_dir,
            include_test=not bool(args.skip_test),
        ),
        "test_loaded_after_policy_selection": True,
        "policy_train_ids": policy_train, "policy_val_ids": policy_val,
        "ridge_donor_ids": donors, "strict_cold_count": int(strict_cold.numel()),
        "pseudo_target_count": len(sim_target_ids),
        "ridge_lambda_final": final_lam, "ridge_lambda_simulation": sim_lam,
        "ridge_initializer": {
            "name": "backbone_anchored_ridge",
            "alpha": ridge_alpha,
        },
        "pseudo_simulator": {
            "name": "backbone_anchored_ridge",
            "alpha": simulation_ridge_alpha,
        },
        "masked_adj_nnz": int(masked_adj._nnz()),
        "course_stats": course_stats,
        "course_bias_stats": course_bias_stats,
        "policy_arms": [
            {
                "name": arm.name,
                "use_course_bias": arm.use_course_bias,
                "use_course_reward": arm.use_course_reward,
            }
            for arm in policy_arm_specs
        ],
        "course_signal_weights": {
            "bias": {
                "bias_scale": bias_cfg.course_bias_scale,
                "concept": bias_cfg.course_concept_weight,
                "prerequisite": bias_cfg.course_prereq_weight,
                "difficulty": bias_cfg.course_difficulty_weight,
                "redundancy": bias_cfg.course_redundant_weight,
            },
            "reward": {
                "scale": float(args.course_reward_scale),
                "mode": args.course_reward_mode,
                "concept": run_cfg.course_concept_weight,
                "prerequisite": run_cfg.course_prereq_weight,
                "difficulty": run_cfg.course_difficulty_weight,
                "redundancy": run_cfg.course_redundant_weight,
            },
        },
        "selection": {
            "selection_mode": selection_mode,
            "delta_grid": list(delta_grid),
            "fixed_max_delta": fixed_max_delta,
            "step_size": float(args.step_size),
            "max_steps": int(args.max_steps),
            "hot_tolerance": float(args.hot_tolerance),
            "retention_reference": retention_reference,
            "validation_rows": int(len(val_df)),
            "backbone_val_metrics": backbone_val_metrics,
            "ridge_val_metrics": ridge_val_metrics,
        },
        "policy_checkpoints": {},
        "evaluation_bundles": {},
        "code_provenance": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "config": frozen_config,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    arms: dict[str, Any] = {"ridge_base": {"bank": ridge_bank}}
    greedy = _select_greedy(
        bundle=bundle, course_signal=course_bias_signal, run_cfg=run_cfg,
        delta_grid=delta_grid, selection_mode=selection_mode,
        fixed_max_delta=fixed_max_delta, val_row_limit=args.max_val_rows,
    )
    arms["ridge_greedy_course_fit"] = greedy
    if bool(getattr(args, "with_random_policy_arm", False)):
        rnd = _select_random_policy(
            bundle=bundle, run_cfg=run_cfg, delta_grid=delta_grid,
            selection_mode=selection_mode, fixed_max_delta=fixed_max_delta,
            val_row_limit=args.max_val_rows, seed=int(args.seed),
        )
        arms["ridge_random_policy"] = rnd
    if bool(getattr(args, "with_centroid_step_arm", False)):
        arms["ridge_centroid_step"] = _select_centroid_step(
            bundle=bundle, run_cfg=run_cfg, delta_grid=delta_grid,
            selection_mode=selection_mode, fixed_max_delta=fixed_max_delta,
            val_row_limit=args.max_val_rows,
        )
    for null_flag, null_arm, null_walk in (
        ("with_global_shift_arm", "ridge_global_shift", global_shift_walk),
        ("with_norm_only_arm", "ridge_norm_only", norm_only_walk),
    ):
        if bool(getattr(args, null_flag, False)):
            arms[null_arm] = _select_displacement_null(
                arm=null_arm, walk=null_walk,
                bundle=bundle, run_cfg=run_cfg, delta_grid=delta_grid,
                selection_mode=selection_mode, fixed_max_delta=fixed_max_delta,
                val_row_limit=args.max_val_rows,
            )
    for arm_spec in policy_arm_specs:
        _seed_everything(args.seed)
        result = _train_policy(
            arm=arm_spec, bundle=bundle, sim_bank=sim_bank,
            pseudo_user_bank=pseudo_user_bank, policy_train_ids=policy_train,
            policy_val_ids=policy_val, positives=positives,
            course_signal=course_signal, run_cfg=run_cfg,
            course_bias_signal=course_bias_signal,
            course_reward_scale=float(args.course_reward_scale),
            center_course_reward=args.course_reward_mode == "centered",
            reward_geometry=getattr(args, "reward_geometry", "euclidean"),
            embedding_reward_weight=float(
                getattr(args, "embedding_reward_weight", 1.0)
            ),
            recommendation_reward_weight=float(
                getattr(args, "recommendation_reward_weight", 1.0)
            ),
            no_epoch_selection=bool(getattr(args, "no_epoch_selection", False)),
            delta_grid=delta_grid, selection_mode=selection_mode,
            fixed_max_delta=fixed_max_delta, max_epochs=args.policy_epochs,
            batch_size=args.policy_batch_size, val_row_limit=args.max_val_rows,
            output_dir=out_dir,
        )
        result["engine"].policy.eval()
        bank, rollout_stats = selected_cold_bank(
            engine=result["engine"], ridge_bank=ridge_bank, cold_ids=strict_cold,
            user_bank=bundle["base_user_bank"], user_history=bundle["user_seen"],
            selected_epoch=result["selected_epoch"],
            selected_delta=result.get("selected_delta"),
            batch_size=args.policy_batch_size,
            selection_mode=selection_mode,
        )
        result["rollout_stats"] = rollout_stats
        result["bank"] = bank
        manifest["policy_checkpoints"][arm_spec.name] = (
            _save_selected_policy_checkpoint(
                out_dir,
                arm=arm_spec.name,
                policy_state=result["policy_state"],
                selected_epoch=result["selected_epoch"],
                selected_delta=result.get("selected_delta"),
                selection_mode=selection_mode,
                fixed_max_delta=fixed_max_delta,
            )
        )
        manifest["evaluation_bundles"][arm_spec.name] = (
            _save_selected_eval_bundle(
                out_dir,
                arm=arm_spec.name,
                user_bank=bundle["z_u_all"],
                item_bank=bank,
                selected_epoch=result["selected_epoch"],
                selected_delta=result.get("selected_delta"),
                selection_mode=selection_mode,
                fixed_max_delta=fixed_max_delta,
            )
        )
        arms[arm_spec.name] = result

    def _arm_diagnostics() -> dict[str, Any]:
        """Write-only per-arm diagnostics: selected point and rollout behaviour.

        The selected delta of the non-PPO arms and the rollout statistics of the
        PPO arms were previously computed and then dropped on the floor -- only
        the PPO arms' delta reached the manifest, via policy_checkpoints. A
        displacement table cannot be read without the delta and the realised
        displacement of every arm, so both are persisted here. This block adds
        JSON keys only; it cannot change any metric.
        """
        diag: dict[str, Any] = {}
        for name, value in arms.items():
            if not isinstance(value, dict):
                continue
            entry: dict[str, Any] = {}
            if "selected_delta" in value:
                entry["selected_delta"] = float(value["selected_delta"])
            if "selected_epoch" in value:
                entry["selected_epoch"] = int(value["selected_epoch"])
            if isinstance(value.get("rollout_stats"), Mapping):
                entry["rollout_stats"] = {
                    k: float(v) for k, v in value["rollout_stats"].items()
                    if isinstance(v, (int, float))
                }
            if entry:
                diag[name] = entry
        return diag

    if args.skip_test:
        manifest["test_loaded_after_policy_selection"] = False
        manifest["test_skipped"] = True
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        result = {
            "manifest": manifest,
            "validation": {name: (ridge_val_metrics if name == "ridge_base"
                                   else value.get("selected"))
                           for name, value in arms.items()},
            "arm_diagnostics": _arm_diagnostics(),
        }
        (out_dir / "pilot_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    # Delayed outer-test read: no test frame is loaded before all policy choices above.
    test_df = _load_test(split_dir)
    test_metrics: dict[str, Any] = {}
    for name, value in arms.items():
        test_metrics[name] = _evaluate_arm(name, value["bank"], bundle, test_df, out_dir)
    frontier = _uniform_bias_frontier(
        bundle["z_u_all"], ridge_bank, bundle["cold_mask"], test_df, device,
        bundle["train_pop"], bundle["user_seen"],
    )
    for name, metrics in test_metrics.items():
        ref, clamped = _cold_at_hot(frontier, metrics["hot_N@10"])
        metrics["matched_hot_cold_vs_ridge_bias"] = float(metrics["cold_N@10"] - ref)
        metrics["matched_hot_clamped"] = bool(clamped)
    gates = {}
    ridge_test = test_metrics["ridge_base"]
    for name in (arm.name for arm in policy_arm_specs):
        metrics = test_metrics[name]
        gates[name] = {
            "cold_N@10_delta_vs_ridge": float(metrics["cold_N@10"] - ridge_test["cold_N@10"]),
            "matched_hot_cold_delta_vs_ridge_bias": float(metrics["matched_hot_cold_vs_ridge_bias"]),
            "matched_hot_clamped": bool(metrics["matched_hot_clamped"]),
            "selected_epoch": int(arms[name]["selected_epoch"]),
            "selection_mode": selection_mode,
        }
        if selection_mode == "direct_rollout":
            gates[name]["fixed_max_delta"] = fixed_max_delta
        else:
            gates[name]["selected_delta"] = float(arms[name]["selected_delta"])
        gates[name]["verdict"] = pilot_verdict(
            selected_epoch=gates[name]["selected_epoch"],
            cold_delta=gates[name]["cold_N@10_delta_vs_ridge"],
            matched_hot_delta=gates[name]["matched_hot_cold_delta_vs_ridge_bias"],
            matched_hot_clamped=gates[name]["matched_hot_clamped"],
        )
    manifest["test_loaded_after_policy_selection"] = True
    manifest["test_loaded_at_unix"] = time.time()
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    result = {
        "manifest": manifest,
        "validation": {name: (ridge_val_metrics if name == "ridge_base"
                               else value.get("selected"))
                       for name, value in arms.items()},
        "test": test_metrics,
        "uniform_bias_frontier": frontier,
        "gate": gates,
        "arm_diagnostics": _arm_diagnostics(),
    }
    (out_dir / "pilot_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    p.add_argument("--split-root", default="outputs/content_delta_pop5/static_item_cold_balanced")
    p.add_argument("--ckpt-root", default="outputs/graph_knp_final")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", default="")
    p.add_argument("--course-relation-dir", default="MOOCCube/relations")
    p.add_argument(
        "--ppo-arms",
        default="ridge_ppo_no_course_reward,ridge_ppo_course_reward",
        help="comma-separated PPO arm names",
    )
    p.add_argument("--course-bias-scale", type=float, default=0.20)
    p.add_argument("--course-concept-weight", type=float, default=0.04)
    p.add_argument("--course-prereq-weight", type=float, default=0.08)
    p.add_argument("--course-difficulty-weight", type=float, default=0.03)
    p.add_argument("--course-redundant-weight", type=float, default=0.02)
    p.add_argument("--course-reward-scale", type=float, default=1.0)
    p.add_argument(
        "--embedding-reward-weight",
        type=float,
        default=1.0,
        help="weight on the target-progress reward term; 0 ablates it (default 1.0 "
             "reproduces earlier runs bit-for-bit)",
    )
    p.add_argument(
        "--recommendation-reward-weight",
        type=float,
        default=1.0,
        help="weight on the positive-user score-gain reward term; 0 ablates it",
    )
    p.add_argument(
        "--reward-geometry", choices=("euclidean", "cosine"), default="euclidean",
        help="cosine makes both reward terms agree with the normalised scorer; "
             "euclidean is the historical form and pays for radial motion",
    )
    p.add_argument(
        "--course-reward-mode", choices=("absolute", "centered"), default="absolute"
    )
    p.add_argument(
        "--ppo-entropy-weight", type=float, default=0.01,
        help="coefficient on -entropy in the PPO loss. The 0.01 default was "
             "never varied in the 2026-08 batches, and measured training reward "
             "there is ~1.1e-3 (embed 9.0e-4 + rec 2.0e-4) while the entropy "
             "term reaches 0.01*ln(20)=0.030 -- a ~27x mismatch whose optimum is "
             "the uniform policy, i.e. exactly the ridge_random_policy null. "
             "Lower this to let the action-discriminating signal drive updates.",
    )
    p.add_argument(
        "--no-epoch-selection",
        action="store_true",
        help="restrict policy selection to the final epoch, so the PPO arm and "
             "the zero-training nulls both choose among len(delta_grid) "
             "candidates instead of policy_epochs x len(delta_grid). Measured "
             "asymmetry without it: PPO val->test -0.0006..-0.0041 vs the "
             "nulls' +0.0002/-0.0003.",
    )
    p.add_argument("--max-pseudo-pop", type=float, default=25.0)
    p.add_argument(
        "--with-random-policy-arm",
        action="store_true",
        help="add the ridge_random_policy zero-information displacement null",
    )
    p.add_argument(
        "--with-centroid-step-arm",
        action="store_true",
        help="add ridge_centroid_step: deterministic mean shift, no policy at all",
    )
    p.add_argument(
        "--with-global-shift-arm",
        action="store_true",
        help=(
            "add ridge_global_shift: one shared direction (user-bank mean) for "
            "every cold row, no per-item retrieval at all"
        ),
    )
    p.add_argument(
        "--with-norm-only-arm",
        action="store_true",
        help=(
            "add ridge_norm_only: radial scaling only, direction unchanged -- the "
            "pure hubness null for the displacement mechanism"
        ),
    )
    p.add_argument(
        "--no-end-action",
        action="store_true",
        help=(
            "mask the END action so every episode runs the full step budget; "
            "removes the no-op escape solution that step_penalty creates"
        ),
    )
    p.add_argument("--pseudo-val-fraction", type=float, default=0.20)
    p.add_argument("--policy-epochs", type=int, default=5)
    p.add_argument("--policy-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-users", type=int, default=512)
    p.add_argument("--candidate-count", type=int, default=20)
    p.add_argument("--retrieval-chunk", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--step-size", type=float, default=0.05)
    p.add_argument("--step-penalty", type=float, default=0.01)
    p.add_argument("--replay-capacity", type=int, default=8192)
    p.add_argument("--replay-batch-size", type=int, default=512)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--hot-tolerance", type=float, default=0.003)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--simulation-ridge-alpha", type=float, default=None)
    p.add_argument(
        "--retention-reference",
        choices=("ridge", "backbone"),
        default="ridge",
    )
    p.add_argument(
        "--selection-mode",
        choices=("delta_grid", "direct_rollout"),
        default="delta_grid",
    )
    p.add_argument("--fixed-max-delta", type=float, default=0.25)
    p.add_argument("--delta-grid", type=float, nargs="+", default=list(DELTA_GRID))
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-val-items", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--log-file", default=None)
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.log_file:
        log_path = _path(args.log_file)
    else:
        log_path = _path(args.out) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
            result = run_pilot(args)
            print(json.dumps({"status": "complete", "has_test": "test" in result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
