"""Validation-only frozen-Hot masked pseudo-cold adapter experiment.

The entrypoint deliberately contains no CBI, ID embedding, simulator, PPO,
course-reward, edge write-back, or test-evaluation path.  It trains one shared
content adapter from edges removed from a frozen CGRC Hot expert.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cgrc_paper_static_hin as cgrc
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions, setup_seed
from hin_eval_common import evaluate_embedding_ranker


_REPO_ROOT = Path(__file__).resolve().parent
HOT_CHECKPOINT_SHA256 = "A41C466D8244FA08E043CFD8DC0289E3F99F5DD5AF351F4B891D62780A2C258F"
_METRIC_KEYS = ("cold_r10", "cold_n10", "hot_r10", "hot_n10", "overall_r10", "overall_n10")


def _canonical_split_path(seed: int) -> Path:
    return (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    ).resolve()


@dataclass(frozen=True)
class AdapterConfig:
    """All non-path experiment choices are locked for the one-seed screen."""

    seed: int
    data_dir: str = "processed_data_hin_clean_pop5"
    split_dir: str = "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025"
    output_dir: str = "outputs/ckg_frozen_hot_pseudocold_adapter_seed2025"
    checkpoint_dir: str = "checkpoints/ckg_frozen_hot_pseudocold_adapter_seed2025"
    hot_output_dir: str = "outputs/ckg_hot_graph_preflight_seed2025"
    hot_checkpoint_dir: str = "checkpoints/ckg_hot_graph_preflight_seed2025"
    n_items: int = 698
    warm_item_count: int = 596
    train_zero_item_count: int = 102
    pseudo_cold_item_count: int = 102
    trust_tau: float = 0.24929234
    epochs: int = 15
    emb_dim: int = 64
    hidden_dim: int = 64
    layers_full: int = 2
    batch_size: int = 4096
    negatives_per_positive: int = 32
    ranking_temperature: float = 0.50
    lr: float = 1e-3
    weight_decay: float = 0.0
    delta_reg_weight: float = 0.0
    parity_atol: float = 1e-5
    retention_tolerance: float = 0.003
    cold_gain_minimum: float = 0.003
    cold_threshold: int = 1
    device: str = ""
    test_evaluation: bool = False
    use_cbi: bool = False
    use_simulator: bool = False
    use_ppo: bool = False
    use_course_rewards: bool = False

    @classmethod
    def for_seed(cls, seed: int) -> "AdapterConfig":
        seed = int(seed)
        return cls(
            seed=seed,
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                f"strict_item_cold_balanced_thr1_seed_{seed}"
            ),
            output_dir=f"outputs/ckg_frozen_hot_pseudocold_adapter_seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_frozen_hot_pseudocold_adapter_seed{seed}",
        )


class SharedColdAdapter(nn.Module):
    """Content-only residual map projected in the final unit-sphere space."""

    def __init__(self, emb_dim: int, hidden_dim: int, trust_tau: float):
        super().__init__()
        if int(emb_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError("emb_dim and hidden_dim must be positive")
        if not 0.0 <= float(trust_tau) <= 2.0:
            raise ValueError("trust_tau must be in [0, 2]")
        self.trust_tau = float(trust_tau)
        self.net = nn.Sequential(
            nn.LayerNorm(int(emb_dim)),
            nn.Linear(int(emb_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(emb_dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, content_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = F.normalize(content_base, dim=-1)
        raw_delta = self.net(base)
        tangent = raw_delta - (raw_delta * base).sum(dim=-1, keepdim=True) * base
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        # Leave a few ULPs inside the registered chordal radius before the
        # final normalization, so float32 rounding cannot cross the cap.
        safe_tau = max(0.0, self.trust_tau - 8.0 * torch.finfo(base.dtype).eps)
        max_angle = 2.0 * math.asin(safe_tau / 2.0)
        angle = tangent_norm.clamp(max=max_angle)
        uncapped_scale = torch.sinc(tangent_norm / math.pi)
        capped_scale = math.sin(max_angle) / tangent_norm.clamp_min(1e-12)
        tangent_scale = torch.where(tangent_norm <= max_angle, uncapped_scale, capped_scale)
        output = torch.cos(angle) * base + tangent_scale * tangent
        output = F.normalize(output, dim=-1)
        final_delta = output - base
        return output, final_delta


def mask_item_edges(graph: sp.csr_matrix, selected_items: Sequence[int] | np.ndarray) -> sp.csr_matrix:
    """Copy an interaction graph while deleting every selected-item edge."""
    selected = np.asarray(selected_items, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return graph.tocsr(copy=True)
    coo = graph.tocoo()
    keep = ~np.isin(coo.col, selected)
    masked = sp.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape, dtype=graph.dtype
    )
    masked.eliminate_zeros()
    return masked


def derive_train_item_partitions(
    train_zero_mask: np.ndarray | torch.Tensor, cfg: AdapterConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Derive and fail closed on the fixed strict-split catalog composition."""
    mask = np.asarray(train_zero_mask, dtype=bool).reshape(-1)
    if mask.size != int(cfg.n_items):
        raise ValueError(f"expected {cfg.n_items} catalog items, got {mask.size}")
    warm_ids = np.flatnonzero(~mask).astype(np.int64, copy=False)
    zero_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if warm_ids.size != int(cfg.warm_item_count):
        raise ValueError(f"expected {cfg.warm_item_count} warm items, got {warm_ids.size}")
    if zero_ids.size != int(cfg.train_zero_item_count):
        raise ValueError(f"expected {cfg.train_zero_item_count} train-zero items, got {zero_ids.size}")
    return warm_ids, zero_ids


def select_epoch_pseudocold_items(
    warm_item_ids: np.ndarray, *, epoch: int, cfg: AdapterConfig
) -> np.ndarray:
    """Choose the same pseudo-cold course set for a fixed seed and epoch."""
    warm = np.sort(np.asarray(warm_item_ids, dtype=np.int64).reshape(-1))
    if warm.size != int(cfg.warm_item_count):
        raise ValueError("pseudo-cold selection requires the locked warm catalog")
    if int(cfg.pseudo_cold_item_count) > warm.size:
        raise ValueError("pseudo-cold item count exceeds the warm catalog")
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(epoch)]))
    return np.sort(
        rng.choice(warm, size=int(cfg.pseudo_cold_item_count), replace=False).astype(np.int64)
    )


def pseudo_cold_selection_audit(selected_items: Sequence[int] | np.ndarray, cfg: AdapterConfig) -> dict[str, Any]:
    """Record the exact selected set without storing item IDs in the row CSV."""
    selected = np.asarray(selected_items, dtype=np.int64).reshape(-1)
    if selected.size != int(cfg.pseudo_cold_item_count) or np.unique(selected).size != selected.size:
        raise ValueError("pseudo-cold audit requires exactly the registered distinct item count")
    if not np.array_equal(selected, np.sort(selected)):
        selected = np.sort(selected)
    return {
        "pseudo_cold_item_count": int(selected.size),
        "warm_item_count": int(cfg.warm_item_count),
        "train_zero_item_count": int(cfg.train_zero_item_count),
        "pseudo_cold_warm_ratio": float(selected.size / cfg.warm_item_count),
        "pseudo_cold_ids_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }


def negative_candidates(
    user_ids: Sequence[int],
    original_user_rated: Sequence[set[int]],
    item_pool: np.ndarray,
    per_user: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample items absent from each user's complete original train history."""
    if int(per_user) < 0:
        raise ValueError("per_user must be non-negative")
    pool = np.unique(np.asarray(item_pool, dtype=np.int64).reshape(-1))
    result: list[np.ndarray] = []
    for uid in user_ids:
        rated = original_user_rated[int(uid)]
        rated_array = np.fromiter(rated, dtype=np.int64, count=len(rated))
        allowed = pool[~np.isin(pool, rated_array)]
        if allowed.size == 0 or int(per_user) == 0:
            result.append(np.empty(0, dtype=np.int64))
        else:
            result.append(
                np.asarray(
                    rng.choice(allowed, size=int(per_user), replace=allowed.size < int(per_user)),
                    dtype=np.int64,
                ).reshape(-1)
            )
    return result


def training_negative_candidates(
    user_ids: Sequence[int],
    original_user_rated: Sequence[set[int]],
    train_zero_mask: np.ndarray | torch.Tensor,
    per_user: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw training negatives only from courses observed in the train catalog."""
    mask = np.asarray(train_zero_mask, dtype=bool).reshape(-1)
    return negative_candidates(user_ids, original_user_rated, np.flatnonzero(~mask), per_user, rng)


def require_complete_negative_candidates(
    candidates: Sequence[np.ndarray], *, expected_count: int
) -> list[np.ndarray]:
    """Reject a batch if any removed positive lacks the registered negatives."""
    if any(np.asarray(values).size != int(expected_count) for values in candidates):
        raise RuntimeError("every removed positive requires the configured warm-only negatives")
    return [np.asarray(values, dtype=np.int64).reshape(-1) for values in candidates]


def item_balanced_edge_objective(
    edge_loss: torch.Tensor,
    item_ids: torch.Tensor,
    degree_by_item: torch.Tensor,
    *,
    selected_item_count: int,
) -> torch.Tensor:
    """Give every selected course total loss weight 1 / |S| across its edges."""
    if int(selected_item_count) < 1:
        raise ValueError("selected_item_count must be positive")
    if edge_loss.ndim != 1 or item_ids.shape != edge_loss.shape:
        raise ValueError("edge losses and item IDs must be aligned one-dimensional tensors")
    degree = degree_by_item[item_ids].to(dtype=edge_loss.dtype)
    if torch.any(degree <= 0):
        raise ValueError("selected item degrees must be positive")
    return (edge_loss / degree).sum() / float(selected_item_count)


def build_adapter_optimizer(adapter: SharedColdAdapter, cfg: AdapterConfig) -> torch.optim.Adam:
    """Construct the only optimizer in Stage B: Adam over shared adapter parameters."""
    return torch.optim.Adam(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_true_eval_item_bank(
    hot_bank: torch.Tensor,
    content_bank: torch.Tensor,
    train_zero_mask: np.ndarray | torch.Tensor,
    adapter: nn.Module,
) -> torch.Tensor:
    """Route every train-zero catalog item through the shared content adapter."""
    if hot_bank.shape != content_bank.shape:
        raise ValueError("hot_bank and content_bank must have identical shape")
    mask = torch.as_tensor(train_zero_mask, device=hot_bank.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != hot_bank.shape[0]:
        raise ValueError("train_zero_mask must have one entry per catalog item")
    cold_bank, _ = adapter(content_bank)
    if cold_bank.shape != hot_bank.shape:
        raise ValueError("adapter output must match item-bank shape")
    return F.normalize(torch.where(mask.view(-1, 1), cold_bank, hot_bank), dim=-1)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Hot preflight record: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Hot preflight record must be a JSON object: {path}")
    return payload


def require_completed_hot_preflight(
    manifest_path: str | Path, result_path: str | Path, expected_epoch: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only a completed, passed Hot preflight at the requested epoch."""
    manifest = _read_json_object(Path(manifest_path))
    result = _read_json_object(Path(result_path))
    if manifest.get("status") != "completed":
        raise ValueError("completed Hot preflight manifest is required")
    if manifest.get("gate_status") != "completed" or result.get("gate_status") != "completed":
        raise ValueError("completed Hot preflight gate status is required")
    if result.get("passed_hot_preflight") is not True:
        raise ValueError("passed Hot preflight result is required")
    selected = result.get("selected_validation_epoch")
    if not isinstance(selected, Mapping):
        raise ValueError("selected validation epoch is required")
    try:
        selected_epoch = int(selected["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected validation epoch is required") from exc
    if selected_epoch != int(expected_epoch):
        raise ValueError("selected validation epoch does not match the required Hot checkpoint")
    return manifest, result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require_hot_checkpoint(path: str | Path, expected_epoch: int) -> dict[str, Any]:
    """Bind the adapter run to the exact selected Hot checkpoint and architecture."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ValueError(f"Hot checkpoint is missing: {checkpoint_path}")
    if _sha256(checkpoint_path) != HOT_CHECKPOINT_SHA256.upper():
        raise ValueError("Hot checkpoint SHA256 does not match the locked Hot expert")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("Hot checkpoint must contain a model_state mapping")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise ValueError("Hot checkpoint epoch does not match the selected epoch")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Hot checkpoint must contain its training config")
    for key, value in (("emb_dim", 64), ("mlp_hidden", 64), ("layers_full", 2)):
        if int(config.get(key, -1)) != value:
            raise ValueError(f"Hot checkpoint {key} does not match the locked architecture")
    return payload


def require_preflight_input_hashes(
    manifest: Mapping[str, Any],
    *,
    data_files: Mapping[str, Path],
    split_files: Mapping[str, Path],
    source_files: Mapping[str, Path],
) -> None:
    """Require current inputs to equal both before and after Hot-preflight records."""
    for field, files in (("data_sha256", data_files), ("split_sha256", split_files), ("source_sha256", source_files)):
        before = manifest.get(field)
        after = manifest.get(f"{field}_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"Hot preflight manifest is missing {field} integrity records")
        for key, path in files.items():
            digest = _sha256(Path(path))
            if str(before.get(key, "")).upper() != digest or str(after.get(key, "")).upper() != digest:
                raise ValueError(f"Hot preflight {field} mismatch for {key}")


def initialize_validation_rows(
    epoch_zero: Mapping[str, Any], reference_result: Mapping[str, Any], parity_atol: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify epoch-0 parity before it becomes the retention baseline."""
    reference = reference_result.get("selected_validation_epoch")
    if not isinstance(reference, Mapping):
        raise ValueError("Hot preflight result is missing selected_validation_epoch")
    row = dict(epoch_zero)
    if int(row.get("epoch", -1)) != 0:
        raise ValueError("epoch-0 parity requires an epoch=0 validation row")
    for key in _METRIC_KEYS:
        try:
            difference = abs(float(row[key]) - float(reference[key]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("epoch-0 parity metrics are incomplete") from exc
        if difference > float(parity_atol):
            raise ValueError(f"epoch-0 parity failed for {key}: difference={difference:.8g}")
    return [row], dict(row)


def select_adapter_epoch(
    rows: Iterable[Mapping[str, Any]], baseline: Mapping[str, Any], tolerance: float = 0.003
) -> dict[str, Any]:
    """Maximize Cold metrics only among checkpoints retaining Hot and Overall."""
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    required = ("hot_r10", "hot_n10", "overall_r10", "overall_n10")
    try:
        floors = {key: float(baseline[key]) - tolerance for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("baseline must provide Hot and Overall R@10/N@10") from exc
    eligible: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        try:
            passes = all(float(row[key]) >= floors[key] for key in required)
            selection_key = (float(row["cold_n10"]), float(row["cold_r10"]), int(row["epoch"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("validation rows must include epoch and Cold/Hot/Overall R@10/N@10") from exc
        if passes:
            row["passes_retention_guards"] = True
            row["_selection_key"] = selection_key
            eligible.append(row)
    if not eligible:
        raise ValueError("no validation epoch satisfies Hot and Overall guards")
    best = max(eligible, key=lambda item: item["_selection_key"])
    best.pop("_selection_key", None)
    return best


def load_validation_only_inputs(
    data_dir: str | Path, split_dir: str | Path
) -> tuple[dict[str, Any], torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Read the four validation-only inputs without using a split convenience loader."""
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    meta_path = data_path / "meta.json"
    content_path = data_path / "content_emb.pt"
    train_path = split_path / "static_train.pkl"
    validation_path = split_path / "static_val.pkl"
    for path in (meta_path, content_path, train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"validation input is missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        content = torch.load(content_path, map_location="cpu", weights_only=False)
    except TypeError:
        content = torch.load(content_path, map_location="cpu")
    if not isinstance(content, torch.Tensor):
        raise ValueError("content_emb.pt must contain a tensor")
    train = pd.read_pickle(train_path).copy()
    validation = pd.read_pickle(validation_path).copy()
    required = {"u_idx", "i_idx"}
    if not required.issubset(train.columns) or not required.issubset(validation.columns):
        raise ValueError("train and validation data must include u_idx and i_idx")
    counts = train["i_idx"].astype(int).value_counts().astype(int)
    for frame in (train, validation):
        frame["popularity"] = frame["i_idx"].astype(int).map(counts).fillna(0).astype(int)
    return meta, content.float(), train, validation


def _resolve_device(requested: str) -> torch.device:
    raw = str(requested).strip().lower()
    if raw:
        if raw == "cpu":
            return torch.device("cpu")
        if raw.startswith("cuda") and torch.cuda.is_available():
            return torch.device(raw)
        raise RuntimeError(f"requested unavailable device: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_sparse_graph(R: sp.csr_matrix, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    return cgrc._sparse_adj_tensor(
        cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(R, n_users, n_items)), device
    )


def _evaluate_validation(
    *,
    adapter: SharedColdAdapter,
    content_bank: torch.Tensor,
    full_hot_items: torch.Tensor,
    full_hot_users: torch.Tensor,
    train_zero_mask: np.ndarray,
    val_loader: DataLoader,
    train_seen: Mapping[int, set],
    cfg: AdapterConfig,
    output_dir: Path,
    epoch: int,
) -> dict[str, Any]:
    adapter.eval()
    with torch.no_grad():
        item_bank = build_true_eval_item_bank(full_hot_items, content_bank, train_zero_mask, adapter)
        get_user = lambda batch: full_hot_users[batch["u"]]
        cold, cold_count = evaluate_embedding_ranker(
            val_loader,
            device=item_bank.device,
            n_items=item_bank.shape[0],
            cold_threshold=cfg.cold_threshold,
            get_user_vectors_fn=get_user,
            all_item_vectors=item_bank,
            k_list=(5, 10, 20),
            eval_type="cold",
            full_ranking=True,
            user_seen_items=dict(train_seen),
            average_mode="item_macro",
            export_item_metrics_path=str(output_dir / f"epoch_{epoch:03d}_per_item_cold.csv"),
        )
        hot, hot_count = evaluate_embedding_ranker(
            val_loader,
            device=item_bank.device,
            n_items=item_bank.shape[0],
            cold_threshold=cfg.cold_threshold,
            get_user_vectors_fn=get_user,
            all_item_vectors=item_bank,
            k_list=(5, 10, 20),
            eval_type="hot",
            full_ranking=True,
            user_seen_items=dict(train_seen),
            average_mode="item_macro",
            export_item_metrics_path=str(output_dir / f"epoch_{epoch:03d}_per_item_hot.csv"),
        )
    if cold is None or hot is None:
        raise RuntimeError("validation split did not yield both Cold and Hot item-macro metrics")
    return {
        "epoch": int(epoch),
        "cold_r10": float(cold["R@10"]),
        "cold_n10": float(cold["N@10"]),
        "hot_r10": float(hot["R@10"]),
        "hot_n10": float(hot["N@10"]),
        "overall_r10": _count_weighted_overall(cold["R@10"], cold_count, hot["R@10"], hot_count),
        "overall_n10": _count_weighted_overall(cold["N@10"], cold_count, hot["N@10"], hot_count),
        "cold_item_count": int(cold_count),
        "hot_item_count": int(hot_count),
    }


def _count_weighted_overall(cold: float, cold_count: int, hot: float, hot_count: int) -> float:
    total = int(cold_count) + int(hot_count)
    if total < 1:
        raise ValueError("validation must contain at least one item")
    return (float(cold) * int(cold_count) + float(hot) * int(hot_count)) / total


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "epoch", "cold_r10", "cold_n10", "hot_r10", "hot_n10", "overall_r10", "overall_n10",
        "cold_item_count", "hot_item_count", "train_loss", "ranking_loss", "mean_final_delta",
        "masked_item_count", "masked_edge_count", "pseudo_cold_item_count", "warm_item_count",
        "train_zero_item_count", "pseudo_cold_warm_ratio", "pseudo_cold_ids_sha256",
        "passes_retention_guards",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(cfg: AdapterConfig) -> None:
    if int(cfg.seed) != 2025:
        raise ValueError("this isolated feasibility run is locked to seed 2025")
    if Path(cfg.split_dir).resolve() != _canonical_split_path(cfg.seed):
        raise ValueError("adapter run must use the canonical shared split")
    if any((cfg.test_evaluation, cfg.use_cbi, cfg.use_simulator, cfg.use_ppo, cfg.use_course_rewards)):
        raise ValueError("test evaluation, CBI, simulation, PPO, and course rewards are forbidden")
    locked_ints = {
        "n_items": 698,
        "warm_item_count": 596,
        "train_zero_item_count": 102,
        "pseudo_cold_item_count": 102,
        "epochs": 15,
        "emb_dim": 64,
        "hidden_dim": 64,
        "layers_full": 2,
        "batch_size": 4096,
        "negatives_per_positive": 32,
        "cold_threshold": 1,
    }
    for field, expected in locked_ints.items():
        if int(getattr(cfg, field)) != expected:
            raise ValueError(f"Stage B requires {field}={expected}")
    locked_floats = {
        "trust_tau": 0.24929234,
        "ranking_temperature": 0.5,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "delta_reg_weight": 0.0,
        "parity_atol": 1e-5,
        "retention_tolerance": 0.003,
        "cold_gain_minimum": 0.003,
    }
    for field, expected in locked_floats.items():
        if not math.isclose(float(getattr(cfg, field)), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Stage B requires {field}={expected}")


def resolve_run_input_roots(cfg: AdapterConfig) -> tuple[Path, Path]:
    """Anchor every validated/loaded input under the repository, not process CWD."""
    data_root = Path(cfg.data_dir)
    split_root = Path(cfg.split_dir)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root
    if not split_root.is_absolute():
        split_root = _REPO_ROOT / split_root
    return data_root.resolve(), split_root.resolve()


def _preflight_files(cfg: AdapterConfig) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    data_root, split_root = resolve_run_input_roots(cfg)
    data = {
        "processed_data_hin_clean_pop5\\meta.json": data_root / "meta.json",
        "processed_data_hin_clean_pop5\\content_emb.pt": data_root / "content_emb.pt",
    }
    split_prefix = "outputs\\content_delta_pop5\\static_item_cold_balanced\\strict_item_cold_balanced_thr1_seed_2025\\"
    split = {
        split_prefix + "static_train.pkl": split_root / "static_train.pkl",
        split_prefix + "static_val.pkl": split_root / "static_val.pkl",
    }
    sources = {
        "ckg_hot_graph_preflight.py": _REPO_ROOT / "ckg_hot_graph_preflight.py",
        "cgrc_paper_static_hin.py": _REPO_ROOT / "cgrc_paper_static_hin.py",
        "hin_data_common.py": _REPO_ROOT / "hin_data_common.py",
        "hin_eval_common.py": _REPO_ROOT / "hin_eval_common.py",
        "lightgcn_static_hin.py": _REPO_ROOT / "lightgcn_static_hin.py",
    }
    return data, split, sources


def _train_one_epoch(
    *,
    adapter: SharedColdAdapter,
    optimizer: torch.optim.Optimizer,
    content_bank: torch.Tensor,
    masked_users: torch.Tensor,
    masked_hot_items: torch.Tensor,
    selected_items: np.ndarray,
    removed_users: np.ndarray,
    removed_items: np.ndarray,
    original_user_rated: Sequence[set[int]],
    train_zero_mask: np.ndarray,
    cfg: AdapterConfig,
    epoch: int,
) -> dict[str, float]:
    if removed_users.size == 0:
        raise RuntimeError("selected pseudo-cold courses have no removed training edges")
    selected_mask = torch.zeros(content_bank.shape[0], dtype=torch.bool, device=content_bank.device)
    selected_mask[torch.as_tensor(selected_items, device=content_bank.device)] = True
    degree = np.bincount(removed_items, minlength=content_bank.shape[0]).astype(np.float32)
    if np.any(degree[selected_items] < 1):
        raise RuntimeError("every selected pseudo-cold course must contribute a removed edge")
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(epoch), 211]))
    order = rng.permutation(removed_users.size)
    loss_sum = 0.0
    ranking_sum = 0.0
    delta_sum = 0.0
    batches = 0
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, order.size, int(cfg.batch_size)):
        indices = order[start:start + int(cfg.batch_size)]
        users_np = removed_users[indices]
        positive_np = removed_items[indices]
        negatives = training_negative_candidates(
            users_np.tolist(), original_user_rated, train_zero_mask, cfg.negatives_per_positive, rng
        )
        neg_np = np.stack(
            require_complete_negative_candidates(negatives, expected_count=cfg.negatives_per_positive), axis=0
        )
        users = torch.as_tensor(users_np, device=content_bank.device, dtype=torch.long)
        positives = torch.as_tensor(positive_np, device=content_bank.device, dtype=torch.long)
        negatives_t = torch.as_tensor(neg_np, device=content_bank.device, dtype=torch.long)
        adapted, final_delta = adapter(content_bank)
        item_bank = torch.where(selected_mask.view(-1, 1), adapted, masked_hot_items)
        user_vec = masked_users[users]
        positive_score = (user_vec * item_bank[positives]).sum(dim=1, keepdim=True)
        negative_score = (user_vec.unsqueeze(1) * item_bank[negatives_t]).sum(dim=2)
        logits = torch.cat([positive_score, negative_score], dim=1) / float(cfg.ranking_temperature)
        per_edge = F.cross_entropy(logits, torch.zeros(users.numel(), dtype=torch.long, device=logits.device), reduction="none")
        degree_t = torch.as_tensor(degree, device=logits.device, dtype=per_edge.dtype)
        ranking = item_balanced_edge_objective(
            per_edge,
            positives,
            degree_t,
            selected_item_count=int(selected_items.size),
        )
        selected_delta = final_delta[selected_mask].norm(dim=1)
        ranking.backward()
        loss_sum += float(ranking.detach().item())
        ranking_sum += float(ranking.detach().item())
        delta_sum += float(selected_delta.detach().mean().item())
        batches += 1
    if batches == 0:
        raise RuntimeError("pseudo-cold epoch produced no valid ranking batches")
    optimizer.step()
    return {
        "train_loss": loss_sum / batches,
        "ranking_loss": ranking_sum / batches,
        "mean_final_delta": delta_sum / batches,
    }


def run_adapter_preflight(cfg: AdapterConfig) -> dict[str, Any]:
    """Run the isolated validation-only masked pseudo-cold feasibility screen."""
    _validate_config(cfg)
    setup_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    if (output_dir / "adapter_preflight_result.json").exists():
        raise FileExistsError("adapter output already contains a formal result")
    device = _resolve_device(cfg.device)
    hot_output = Path(cfg.hot_output_dir)
    manifest, reference = require_completed_hot_preflight(
        hot_output / "run_manifest.json", hot_output / "preflight_result.json", expected_epoch=15
    )
    data_root, split_root = resolve_run_input_roots(cfg)
    data_files, split_files, source_files = _preflight_files(cfg)
    require_preflight_input_hashes(manifest, data_files=data_files, split_files=split_files, source_files=source_files)
    checkpoint = require_hot_checkpoint(Path(cfg.hot_checkpoint_dir) / "epoch_015.pt", expected_epoch=15)
    meta, content, train_df, val_df = load_validation_only_inputs(data_root, split_root)
    if int(meta.get("n_items", -1)) != cfg.n_items or int(content.shape[0]) != cfg.n_items:
        raise ValueError("catalog shape does not match the locked strict split")
    counts = train_df["i_idx"].astype(int).value_counts()
    train_zero_mask = np.array([int(counts.get(index, 0)) == 0 for index in range(cfg.n_items)], dtype=bool)
    warm_ids, _ = derive_train_item_partitions(train_zero_mask, cfg)
    model = cgrc.CGRCNet(
        int(meta["n_users"]), int(meta["n_items"]), int(content.shape[1]), cfg.emb_dim, cfg.hidden_dim, content
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    r_train = cgrc._build_interaction_csr(train_df, model.n_users, model.n_items)
    original_user_rated = cgrc._build_user_rated(train_df, model.n_users)
    train_seen = build_user_seen(train_df)
    val_loader = DataLoader(
        InteractionDataset(val_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_interactions
    )
    with torch.no_grad():
        full_adj = _build_sparse_graph(r_train, model.n_users, model.n_items, device)
        full_users, full_items = cgrc._lightgcn_mean_all_layers(
            full_adj, model.user_emb, model.item_x(), model.n_users, cfg.layers_full
        )
        full_users = F.normalize(full_users, dim=1).detach()
        full_items = F.normalize(full_items, dim=1).detach()
        content_bank = F.normalize(model.item_x(), dim=1).detach()
    warm_distance_q75 = float(torch.quantile((full_items[warm_ids] - content_bank[warm_ids]).norm(dim=1), 0.75).item())
    if abs(warm_distance_q75 - float(cfg.trust_tau)) > 1e-4:
        raise ValueError("runtime Hot/content q75 trust calibration does not match the locked tau")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter = SharedColdAdapter(cfg.emb_dim, cfg.hidden_dim, cfg.trust_tau).to(device)
    epoch_zero = _evaluate_validation(
        adapter=adapter, content_bank=content_bank, full_hot_items=full_items, full_hot_users=full_users,
        train_zero_mask=train_zero_mask, val_loader=val_loader, train_seen=train_seen, cfg=cfg,
        output_dir=output_dir, epoch=0,
    )
    epoch_zero.update(
        {
            "train_loss": 0.0,
            "ranking_loss": 0.0,
            "mean_final_delta": 0.0,
            "masked_item_count": 0,
            "masked_edge_count": 0,
            "pseudo_cold_item_count": 0,
            "warm_item_count": int(cfg.warm_item_count),
            "train_zero_item_count": int(cfg.train_zero_item_count),
            "pseudo_cold_warm_ratio": 0.0,
            "pseudo_cold_ids_sha256": "",
        }
    )
    rows, baseline = initialize_validation_rows(epoch_zero, reference, cfg.parity_atol)
    _write_rows(output_dir / "validation_epochs.csv", rows)
    optimizer = build_adapter_optimizer(adapter, cfg)
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_parameter_ids != {id(parameter) for parameter in adapter.parameters()}:
        raise RuntimeError("Stage B optimizer must contain only shared adapter parameters")
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        selected = select_epoch_pseudocold_items(warm_ids, epoch=epoch, cfg=cfg)
        masked_R = mask_item_edges(r_train, selected)
        if masked_R[:, selected].nnz != 0:
            raise RuntimeError("selected pseudo-cold courses retained a student-side graph edge")
        removed = r_train.tocoo()
        removed_mask = np.isin(removed.col, selected)
        removed_users = removed.row[removed_mask].astype(np.int64, copy=False)
        removed_items = removed.col[removed_mask].astype(np.int64, copy=False)
        with torch.no_grad():
            masked_adj = _build_sparse_graph(masked_R, model.n_users, model.n_items, device)
            masked_users, masked_items = cgrc._lightgcn_mean_all_layers(
                masked_adj, model.user_emb, model.item_x(), model.n_users, cfg.layers_full
            )
            masked_users = F.normalize(masked_users, dim=1).detach()
            masked_items = F.normalize(masked_items, dim=1).detach()
        stats = _train_one_epoch(
            adapter=adapter, optimizer=optimizer, content_bank=content_bank, masked_users=masked_users,
            masked_hot_items=masked_items, selected_items=selected, removed_users=removed_users,
            removed_items=removed_items, original_user_rated=original_user_rated,
            train_zero_mask=train_zero_mask, cfg=cfg, epoch=epoch,
        )
        row = _evaluate_validation(
            adapter=adapter, content_bank=content_bank, full_hot_items=full_items, full_hot_users=full_users,
            train_zero_mask=train_zero_mask, val_loader=val_loader, train_seen=train_seen, cfg=cfg,
            output_dir=output_dir, epoch=epoch,
        )
        row.update(stats)
        row["masked_item_count"] = int(selected.size)
        row["masked_edge_count"] = int(removed_users.size)
        audit = pseudo_cold_selection_audit(selected, cfg)
        row.update(audit)
        row["passes_retention_guards"] = bool(
            row["hot_r10"] >= baseline["hot_r10"] - cfg.retention_tolerance
            and row["hot_n10"] >= baseline["hot_n10"] - cfg.retention_tolerance
            and row["overall_r10"] >= baseline["overall_r10"] - cfg.retention_tolerance
            and row["overall_n10"] >= baseline["overall_n10"] - cfg.retention_tolerance
        )
        rows.append(row)
        _write_rows(output_dir / "validation_epochs.csv", rows)
        np.save(output_dir / f"epoch_{epoch:03d}_pseudocold_items.npy", selected)
        torch.save(
            {"epoch": epoch, "adapter_state": adapter.state_dict(), "config": asdict(cfg), "pseudo_cold_audit": audit},
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
        )
        print(
            f"[PSEUDO-COLD] epoch={epoch}/{cfg.epochs} loss={row['train_loss']:.4f} "
            f"cold_R10={row['cold_r10']:.4f} cold_N10={row['cold_n10']:.4f} "
            f"overall_R10={row['overall_r10']:.4f} overall_N10={row['overall_n10']:.4f}",
            flush=True,
        )
    try:
        selected_row = select_adapter_epoch(rows, baseline, cfg.retention_tolerance)
    except ValueError as exc:
        selected_row = None
        passed = False
        selection_error = str(exc)
    else:
        passed = bool(
            selected_row["cold_n10"] >= baseline["cold_n10"] + cfg.cold_gain_minimum
            and selected_row["cold_r10"] >= baseline["cold_r10"]
        )
        selection_error = None
    status = "completed" if passed else "completed_gate_failed"
    result = {
        "experiment": "ckg_frozen_hot_masked_pseudocold_adapter",
        "status": status,
        "config": asdict(cfg),
        "test_evaluation": False,
        "epoch_zero_baseline": baseline,
        "selected_validation_epoch": selected_row,
        "passed_stage_b_screen": passed,
        "gate_status": status,
        "selection_error": selection_error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(output_dir / "adapter_preflight_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-Hot masked pseudo-cold adapter validation screen")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hot-output-dir", default="outputs/ckg_hot_graph_preflight_seed2025")
    parser.add_argument("--hot-checkpoint-dir", default="checkpoints/ckg_hot_graph_preflight_seed2025")
    parser.add_argument("--device", default="")
    return parser


def _config_from_args(args: argparse.Namespace) -> AdapterConfig:
    return replace(
        AdapterConfig.for_seed(args.seed),
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        hot_output_dir=args.hot_output_dir,
        hot_checkpoint_dir=args.hot_checkpoint_dir,
        device=args.device,
    )


def main() -> None:
    args = _parser().parse_args()
    result = run_adapter_preflight(_config_from_args(args))
    if result["gate_status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
