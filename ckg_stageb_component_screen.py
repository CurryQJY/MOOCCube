"""Validation-only component screens built on the frozen Stage-B protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cgrc_paper_static_hin as cgrc
from ckg_frozen_hot_pseudocold_adapter import (
    SharedColdAdapter,
    _build_sparse_graph,
    _evaluate_validation,
    _resolve_device,
    derive_train_item_partitions,
    initialize_validation_rows,
    item_balanced_edge_objective,
    load_validation_only_inputs,
    mask_item_edges,
    pseudo_cold_selection_audit,
    require_complete_negative_candidates,
    require_preflight_input_hashes,
    select_adapter_epoch,
    select_epoch_pseudocold_items,
    training_negative_candidates,
)
from ckg_frozen_hot_pseudocold_adapter_replication import (
    ReplicationAdapterConfig,
    _preflight_files,
    _read_json,
    _resolve_roots,
    require_replication_hot_contract,
)
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions, setup_seed


_STRICT_SEEDS = (2026, 2027)
_FIXED_TAU = 0.24929234
_ALLOWED_COMPONENTS = ("soft_anchor_l2",)
_REPO_ROOT = Path(__file__).resolve().parent
_PARENT_EXPERIMENT = "ckg_frozen_hot_masked_pseudocold_adapter_replication"
_PARENT_MANIFEST_EXPERIMENT = "ckg_frozen_hot_pseudocold_adapter_replication"


@dataclass(frozen=True)
class ComponentScreenConfig:
    """One isolated strict Stage-B component screen."""

    seed: int
    component_name: str
    phase: str = "screen"
    data_dir: str = "processed_data_hin_clean_pop5"
    split_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    hot_output_dir: str = ""
    hot_checkpoint_dir: str = ""
    parent_result_path: str = ""
    n_items: int = 698
    warm_item_count: int = 596
    train_zero_item_count: int = 102
    pseudo_cold_item_count: int = 102
    trust_tau: float = _FIXED_TAU
    epochs: int = 15
    emb_dim: int = 64
    hidden_dim: int = 64
    layers_full: int = 2
    batch_size: int = 4096
    negatives_per_positive: int = 32
    ranking_temperature: float = 0.50
    lr: float = 1e-3
    weight_decay: float = 0.0
    parity_atol: float = 1e-5
    retention_tolerance: float = 0.003
    incumbent_cold_n10_gain: float = 0.003
    soft_anchor_weight: float = 0.10
    cold_threshold: int = 1
    device: str = ""
    test_evaluation: bool = False

    @classmethod
    def for_seed(
        cls,
        seed: int,
        *,
        component_name: str,
        phase: str | None = None,
    ) -> "ComponentScreenConfig":
        seed = int(seed)
        if phase is None:
            phase = "screen" if seed == 2027 else "replication"
        return cls(
            seed=seed,
            component_name=str(component_name),
            phase=str(phase),
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                f"strict_item_cold_balanced_thr1_seed_{seed}"
            ),
            output_dir=f"outputs/ckg_stageb_component_{component_name}_seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_stageb_component_{component_name}_seed{seed}",
            hot_output_dir=f"outputs/ckg_hot_graph_preflight_replication_seed{seed}",
            hot_checkpoint_dir=f"checkpoints/ckg_hot_graph_preflight_replication_seed{seed}",
            parent_result_path=(
                "outputs/ckg_frozen_hot_pseudocold_adapter_replication_"
                f"seed{seed}/adapter_preflight_result.json"
            ),
        )


def validate_component_config(cfg: ComponentScreenConfig) -> None:
    """Fail closed on all knobs outside the registered strict component screen."""
    if int(cfg.seed) not in _STRICT_SEEDS:
        raise ValueError("component screen is restricted to strict seeds 2026 and 2027")
    if str(cfg.component_name) not in _ALLOWED_COMPONENTS:
        raise ValueError("component is not registered for the strict Stage-B screen")
    if str(cfg.phase) not in {"screen", "replication"}:
        raise ValueError("component screen phase is invalid")
    expected_phase = "screen" if int(cfg.seed) == 2027 else "replication"
    if str(cfg.phase) != expected_phase:
        raise ValueError("component screen phase does not match the strict seed")
    if bool(cfg.test_evaluation):
        raise ValueError("test evaluation is forbidden")
    expected = ComponentScreenConfig.for_seed(
        cfg.seed,
        component_name=cfg.component_name,
        phase=cfg.phase,
    )
    if str(cfg.data_dir) != str(expected.data_dir):
        raise ValueError("component screen requires the registered data directory")
    for field in (
        "n_items", "warm_item_count", "train_zero_item_count", "pseudo_cold_item_count",
        "trust_tau", "epochs", "emb_dim", "hidden_dim", "layers_full", "batch_size",
        "negatives_per_positive", "ranking_temperature", "lr", "weight_decay", "parity_atol",
        "retention_tolerance", "incumbent_cold_n10_gain", "soft_anchor_weight",
        "cold_threshold",
    ):
        actual = getattr(cfg, field)
        required = getattr(expected, field)
        if isinstance(required, float):
            if not math.isclose(float(actual), float(required), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"component screen knob is locked: {field}")
        elif actual != required:
            raise ValueError(f"component screen knob is locked: {field}")
    split_dir = Path(cfg.split_dir)
    if not split_dir.is_absolute():
        split_dir = _REPO_ROOT / split_dir
    expected_split = (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(cfg.seed)}"
    ).resolve()
    if split_dir.resolve() != expected_split:
        raise ValueError("component screen requires the canonical strict split")
    for field in ("output_dir", "checkpoint_dir", "hot_output_dir", "hot_checkpoint_dir", "parent_result_path"):
        if not str(getattr(cfg, field)).strip():
            raise ValueError(f"component screen requires {field}")
    canonical_artifacts = {
        "hot_output_dir": _REPO_ROOT / "outputs" / f"ckg_hot_graph_preflight_replication_seed{int(cfg.seed)}",
        "hot_checkpoint_dir": _REPO_ROOT / "checkpoints" / f"ckg_hot_graph_preflight_replication_seed{int(cfg.seed)}",
        "parent_result_path": (
            _REPO_ROOT
            / "outputs"
            / f"ckg_frozen_hot_pseudocold_adapter_replication_seed{int(cfg.seed)}"
            / "adapter_preflight_result.json"
        ),
    }
    for field, expected_path in canonical_artifacts.items():
        if _repo_path(str(getattr(cfg, field))) != expected_path.resolve():
            raise ValueError(f"component screen requires canonical {field}")


def decide_single_seed_screen(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    immutable_baseline: dict[str, float],
    *,
    retention_tolerance: float = 0.003,
    cold_n10_gain: float = 0.003,
) -> str:
    """Classify a single-seed candidate against fixed quality and gain rules."""
    guard_metrics = ("hot_r10", "hot_n10", "overall_r10", "overall_n10")
    if any(
        float(candidate[metric]) < float(immutable_baseline[metric]) - float(retention_tolerance)
        for metric in guard_metrics
    ):
        return "rejected_retention_guard"
    if float(candidate["cold_r10"]) < float(incumbent["cold_r10"]):
        return "rejected_cold_r10_regression"
    if float(candidate["cold_n10"]) < float(incumbent["cold_n10"]) + float(cold_n10_gain):
        return "rejected_insufficient_cold_gain"
    return "provisionally_accepted"


def component_selection_decision(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    immutable_baseline: dict[str, float],
    cfg: ComponentScreenConfig,
) -> str:
    """Use a gain gate only for the first strict-seed screen."""
    guard_metrics = ("hot_r10", "hot_n10", "overall_r10", "overall_n10")
    if any(
        float(candidate[metric]) < float(immutable_baseline[metric]) - float(cfg.retention_tolerance)
        for metric in guard_metrics
    ):
        return "rejected_retention_guard"
    if cfg.phase == "replication":
        return "replication_completed"
    return decide_single_seed_screen(
        candidate,
        incumbent,
        immutable_baseline,
        retention_tolerance=cfg.retention_tolerance,
        cold_n10_gain=cfg.incumbent_cold_n10_gain,
    )


def soft_anchor_loss(
    adapted: torch.Tensor,
    content_anchor: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    trust_tau: float,
) -> torch.Tensor:
    """Penalize selected pseudo-cold updates relative to the fixed content base."""
    if adapted.shape != content_anchor.shape:
        raise ValueError("adapted and content_anchor must have identical shapes")
    if selected_mask.ndim != 1 or selected_mask.numel() != adapted.shape[0]:
        raise ValueError("selected_mask must contain one entry per item")
    if float(trust_tau) <= 0.0:
        raise ValueError("trust_tau must be positive")
    selected = selected_mask.to(device=adapted.device, dtype=torch.bool)
    if not bool(selected.any()):
        return adapted.sum() * 0.0
    delta = adapted[selected] - content_anchor[selected]
    return (delta.norm(dim=1) / float(trust_tau)).pow(2).mean()


def component_training_loss(
    ranking_loss: torch.Tensor,
    adapted: torch.Tensor,
    content_anchor: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    trust_tau: float,
    soft_anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the registered soft content anchor to the masked ranking loss."""
    anchor = soft_anchor_loss(
        adapted,
        content_anchor,
        selected_mask,
        trust_tau=trust_tau,
    )
    return ranking_loss + float(soft_anchor_weight) * anchor, anchor


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def component_source_files() -> dict[str, Path]:
    """List the code whose exact version defines this component screen."""
    return {
        "component_runner": Path(__file__).resolve(),
        "adapter_helper": _REPO_ROOT / "ckg_frozen_hot_pseudocold_adapter.py",
        "replication_helper": _REPO_ROOT / "ckg_frozen_hot_pseudocold_adapter_replication.py",
        "cgrc_model": _REPO_ROOT / "cgrc_paper_static_hin.py",
        "data_common": _REPO_ROOT / "hin_data_common.py",
        "eval_common": _REPO_ROOT / "hin_eval_common.py",
        "lightgcn": _REPO_ROOT / "lightgcn_static_hin.py",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_source_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in component_source_files().items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "epoch", "cold_r10", "cold_n10", "hot_r10", "hot_n10", "overall_r10", "overall_n10",
        "cold_item_count", "hot_item_count", "train_loss", "ranking_loss", "soft_anchor_loss",
        "mean_final_delta", "masked_item_count", "masked_edge_count", "pseudo_cold_item_count",
        "warm_item_count", "train_zero_item_count", "pseudo_cold_warm_ratio",
        "pseudo_cold_ids_sha256", "passes_retention_guards",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _require_parent_incumbent(
    path: Path,
    *,
    seed: int,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = _read_json(path)
    if parent.get("experiment") != _PARENT_EXPERIMENT:
        raise ValueError("parent incumbent experiment is not registered")
    if parent.get("status") != "completed" or parent.get("gate_status") != "completed":
        raise ValueError("a completed parent incumbent is required")
    if parent.get("test_evaluation") is not False:
        raise ValueError("parent incumbent must be validation-only")
    if parent.get("passed_stage_b_screen") is not True:
        raise ValueError("a completed passed parent incumbent is required")
    selected = parent.get("selected_validation_epoch")
    if not isinstance(selected, Mapping):
        raise ValueError("parent incumbent lacks selected validation metrics")
    if selected.get("passes_retention_guards") is not True:
        raise ValueError("parent incumbent selected row did not pass retention guards")
    config = parent.get("config")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(seed):
        raise ValueError("parent incumbent seed mismatch")
    if config.get("test_evaluation") is not False:
        raise ValueError("parent incumbent config must be validation-only")
    expected_data_root = (_REPO_ROOT / "processed_data_hin_clean_pop5").resolve()
    expected_split_root = (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    ).resolve()
    if _repo_path(str(config.get("data_dir", ""))) != expected_data_root:
        raise ValueError("parent incumbent config mismatch: data_dir")
    if _repo_path(str(config.get("split_dir", ""))) != expected_split_root:
        raise ValueError("parent incumbent config mismatch: split_dir")
    for field in ("use_cbi", "use_simulator", "use_ppo", "use_course_rewards"):
        if config.get(field) is not False:
            raise ValueError(f"parent incumbent config mismatch: {field}")
    expected = ComponentScreenConfig.for_seed(seed, component_name="soft_anchor_l2")
    required_config = {
        "n_items": expected.n_items,
        "warm_item_count": expected.warm_item_count,
        "train_zero_item_count": expected.train_zero_item_count,
        "pseudo_cold_item_count": expected.pseudo_cold_item_count,
        "trust_tau": expected.trust_tau,
        "epochs": expected.epochs,
        "emb_dim": expected.emb_dim,
        "hidden_dim": expected.hidden_dim,
        "layers_full": expected.layers_full,
        "batch_size": expected.batch_size,
        "negatives_per_positive": expected.negatives_per_positive,
        "ranking_temperature": expected.ranking_temperature,
        "lr": expected.lr,
        "weight_decay": expected.weight_decay,
        "delta_reg_weight": 0.0,
        "parity_atol": expected.parity_atol,
        "retention_tolerance": expected.retention_tolerance,
        "cold_gain_minimum": expected.incumbent_cold_n10_gain,
        "cold_threshold": expected.cold_threshold,
    }
    for field, required in required_config.items():
        actual = config.get(field)
        if isinstance(required, float):
            try:
                matches = math.isclose(float(actual), required, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"parent incumbent config mismatch: {field}")
        elif actual != required:
            raise ValueError(f"parent incumbent config mismatch: {field}")
    parent_contract = parent.get("hot_checkpoint_contract")
    if not isinstance(parent_contract, Mapping):
        raise ValueError("parent incumbent lacks the Hot checkpoint contract")
    for field in ("seed", "epoch", "relative_path", "sha256"):
        if parent_contract.get(field) != contract.get(field):
            raise ValueError("parent incumbent Hot checkpoint contract mismatch")
    return parent, dict(selected)


def _is_sha256(value: object) -> bool:
    digest = str(value)
    return (
        len(digest) == 64
        and digest == digest.lower()
        and all(character in "0123456789abcdef" for character in digest)
    )


def _require_unchanged_manifest_hashes(
    manifest: Mapping[str, Any],
    *,
    before_field: str,
    after_field: str,
    label: str,
) -> dict[str, str]:
    before = manifest.get(before_field)
    after = manifest.get(after_field)
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or not before:
        raise ValueError(f"parent run manifest lacks {label} hashes")
    normalized_before = {str(key): str(value) for key, value in before.items()}
    normalized_after = {str(key): str(value) for key, value in after.items()}
    if normalized_before != normalized_after:
        raise ValueError(f"parent run manifest {label} changed during the parent run")
    if any(not key or not _is_sha256(value) for key, value in normalized_before.items()):
        raise ValueError(f"parent run manifest {label} hash is invalid")
    return normalized_before


def _require_manifest_file_hash(
    recorded_hashes: Mapping[str, str],
    path: Path,
    *,
    label: str,
) -> None:
    resolved = path.resolve()
    matches = [
        value
        for recorded_path, value in recorded_hashes.items()
        if _repo_path(recorded_path) == resolved
    ]
    if len(matches) != 1:
        raise ValueError(f"parent run manifest lacks a unique {label} hash")
    if not resolved.is_file() or matches[0] != _sha256(resolved):
        raise ValueError(f"parent run manifest {label} hash does not match the current artifact")


def _require_parent_run_manifest(
    path: Path,
    *,
    cfg: ComponentScreenConfig,
    parent_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the parent result to its stable, seed-specific replication manifest."""
    manifest = _read_json(path)
    if manifest.get("experiment") != _PARENT_MANIFEST_EXPERIMENT:
        raise ValueError("parent run manifest experiment is not registered")
    if manifest.get("status") != "completed" or manifest.get("gate_status") != "completed":
        raise ValueError("parent run manifest is not completed")
    try:
        manifest_seed = int(manifest.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("parent run manifest seed is invalid") from exc
    if manifest_seed != int(cfg.seed):
        raise ValueError("parent run manifest seed mismatch")
    locked_config = manifest.get("locked_config")
    if not isinstance(locked_config, Mapping) or locked_config.get("test_evaluation") is not False:
        raise ValueError("parent run manifest must be validation-only")
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("parent run manifest lacks paths")
    expected_parent = parent_path.resolve()
    if _repo_path(str(paths.get("result_path", ""))) != expected_parent:
        raise ValueError("parent run manifest result path mismatch")
    if _repo_path(str(paths.get("output_root", ""))) != expected_parent.parent:
        raise ValueError("parent run manifest output path mismatch")

    source_hashes = _require_unchanged_manifest_hashes(
        manifest,
        before_field="source_sha256",
        after_field="source_sha256_after",
        label="source",
    )
    _require_unchanged_manifest_hashes(
        manifest,
        before_field="launcher_sha256",
        after_field="launcher_sha256_after",
        label="launcher",
    )
    data_hashes = _require_unchanged_manifest_hashes(
        manifest,
        before_field="data_sha256",
        after_field="data_sha256_after",
        label="data",
    )
    split_hashes = _require_unchanged_manifest_hashes(
        manifest,
        before_field="split_sha256",
        after_field="split_sha256_after",
        label="split",
    )
    hot_hashes = _require_unchanged_manifest_hashes(
        manifest,
        before_field="hot_artifacts_sha256",
        after_field="hot_artifacts_sha256_after",
        label="Hot artifacts",
    )
    _require_unchanged_manifest_hashes(
        manifest,
        before_field="protected_files_before",
        after_field="protected_files_after",
        label="protected files",
    )
    if not set(_preflight_files(ReplicationAdapterConfig.for_seed(cfg.seed))[2]).issubset(source_hashes):
        raise ValueError("parent run manifest source inventory is incomplete")

    replication_cfg = ReplicationAdapterConfig.for_seed(cfg.seed)
    data_files, split_files, _ = _preflight_files(replication_cfg)
    for artifact in data_files.values():
        _require_manifest_file_hash(data_hashes, artifact, label="data artifact")
    for artifact in split_files.values():
        _require_manifest_file_hash(split_hashes, artifact, label="split artifact")
    hot_output_dir = _repo_path(cfg.hot_output_dir)
    hot_checkpoint_dir = _repo_path(cfg.hot_checkpoint_dir)
    hot_artifacts = (
        hot_output_dir / "run_manifest.json",
        hot_output_dir / "preflight_result.json",
        hot_output_dir / "validation_epochs.csv",
        hot_checkpoint_dir / str(contract["relative_path"]),
    )
    for artifact in hot_artifacts:
        _require_manifest_file_hash(hot_hashes, artifact, label="Hot artifact")
    return manifest


def _train_one_epoch_with_soft_anchor(
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
    cfg: ComponentScreenConfig,
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
    num_batches = max(1, math.ceil(order.size / int(cfg.batch_size)))
    train_sum = 0.0
    ranking_sum = 0.0
    anchor_sum = 0.0
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
        per_edge = F.cross_entropy(
            logits,
            torch.zeros(users.numel(), dtype=torch.long, device=logits.device),
            reduction="none",
        )
        degree_t = torch.as_tensor(degree, device=logits.device, dtype=per_edge.dtype)
        ranking = item_balanced_edge_objective(
            per_edge,
            positives,
            degree_t,
            selected_item_count=int(selected_items.size),
        )
        total, anchor = component_training_loss(
            ranking,
            adapted,
            content_bank,
            selected_mask,
            trust_tau=cfg.trust_tau,
            soft_anchor_weight=float(cfg.soft_anchor_weight) / float(num_batches),
        )
        total.backward()
        train_sum += float(total.detach().item())
        ranking_sum += float(ranking.detach().item())
        anchor_sum += float(anchor.detach().item())
        delta_sum += float(final_delta[selected_mask].norm(dim=1).mean().detach().item())
        batches += 1
    if batches == 0:
        raise RuntimeError("pseudo-cold epoch produced no valid ranking batches")
    optimizer.step()
    return {
        "train_loss": train_sum,
        "ranking_loss": ranking_sum,
        "soft_anchor_loss": anchor_sum / batches,
        "mean_final_delta": delta_sum / batches,
    }


def run_component_screen(cfg: ComponentScreenConfig) -> dict[str, Any]:
    """Run one fresh validation-only strict soft-anchor component screen."""
    validate_component_config(cfg)
    setup_seed(cfg.seed)
    source_hashes_before = _component_source_hashes()
    output_dir = _repo_path(cfg.output_dir)
    checkpoint_dir = _repo_path(cfg.checkpoint_dir)
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("component screen requires fresh output and checkpoint roots")
    device = _resolve_device(cfg.device)
    hot_output_dir = _repo_path(cfg.hot_output_dir)
    hot_checkpoint_dir = _repo_path(cfg.hot_checkpoint_dir)
    hot_manifest_path = hot_output_dir / "run_manifest.json"
    hot_result_path = hot_output_dir / "preflight_result.json"
    hot_manifest_sha256_before = _sha256(hot_manifest_path)
    hot_result_sha256_before = _sha256(hot_result_path)
    hot_manifest = _read_json(hot_manifest_path)
    if hot_manifest.get("status") != "completed" or hot_manifest.get("gate_status") != "completed":
        raise ValueError("completed Hot replication manifest is required")
    hot_result = _read_json(hot_result_path)
    contract = require_replication_hot_contract(
        hot_result,
        seed=cfg.seed,
        checkpoint_dir=hot_checkpoint_dir,
    )
    parent_path = _repo_path(cfg.parent_result_path)
    parent_result_sha256_before = _sha256(parent_path)
    parent_result, incumbent = _require_parent_incumbent(parent_path, seed=cfg.seed, contract=contract)
    parent_manifest_path = parent_path.parent / "run_manifest.json"
    parent_manifest_sha256_before = _sha256(parent_manifest_path)
    _require_parent_run_manifest(
        parent_manifest_path,
        cfg=cfg,
        parent_path=parent_path,
        contract=contract,
    )
    replication_cfg = ReplicationAdapterConfig.for_seed(cfg.seed)
    data_files, split_files, source_files = _preflight_files(replication_cfg)
    require_preflight_input_hashes(
        hot_manifest,
        data_files=data_files,
        split_files=split_files,
        source_files=source_files,
    )
    data_root, split_root = _resolve_roots(replication_cfg)
    meta, content, train_df, val_df = load_validation_only_inputs(data_root, split_root)
    if int(meta.get("n_items", -1)) != cfg.n_items or int(content.shape[0]) != cfg.n_items:
        raise ValueError("catalog shape does not match the locked strict split")
    counts = train_df["i_idx"].astype(int).value_counts()
    train_zero_mask = np.array([int(counts.get(index, 0)) == 0 for index in range(cfg.n_items)], dtype=bool)
    warm_ids, _ = derive_train_item_partitions(train_zero_mask, cfg)
    checkpoint_path = hot_checkpoint_dir / str(contract["relative_path"])
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = cgrc.CGRCNet(
        int(meta["n_users"]),
        int(meta["n_items"]),
        int(content.shape[1]),
        cfg.emb_dim,
        cfg.hidden_dim,
        content,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    r_train = cgrc._build_interaction_csr(train_df, model.n_users, model.n_items)
    original_user_rated = cgrc._build_user_rated(train_df, model.n_users)
    train_seen = build_user_seen(train_df)
    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    with torch.no_grad():
        full_adj = _build_sparse_graph(r_train, model.n_users, model.n_items, device)
        full_users, full_items = cgrc._lightgcn_mean_all_layers(
            full_adj,
            model.user_emb,
            model.item_x(),
            model.n_users,
            cfg.layers_full,
        )
        full_users = F.normalize(full_users, dim=1).detach()
        full_items = F.normalize(full_items, dim=1).detach()
        content_bank = F.normalize(model.item_x(), dim=1).detach()
    q75_audit = float(torch.quantile((full_items[warm_ids] - content_bank[warm_ids]).norm(dim=1), 0.75).item())
    if abs(q75_audit - float(contract["warm_q75_audit"])) > 1e-4:
        raise ValueError("Hot q75 audit does not match the registered contract")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    adapter = SharedColdAdapter(cfg.emb_dim, cfg.hidden_dim, cfg.trust_tau).to(device)
    epoch_zero = _evaluate_validation(
        adapter=adapter,
        content_bank=content_bank,
        full_hot_items=full_items,
        full_hot_users=full_users,
        train_zero_mask=train_zero_mask,
        val_loader=val_loader,
        train_seen=train_seen,
        cfg=cfg,
        output_dir=output_dir,
        epoch=0,
    )
    epoch_zero.update(
        {
            "train_loss": 0.0,
            "ranking_loss": 0.0,
            "soft_anchor_loss": 0.0,
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
    rows, immutable_baseline = initialize_validation_rows(epoch_zero, hot_result, cfg.parity_atol)
    _write_rows(output_dir / "validation_epochs.csv", rows)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {
        id(parameter) for parameter in adapter.parameters()
    }:
        raise RuntimeError("component optimizer must contain only shared adapter parameters")
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        selected = select_epoch_pseudocold_items(warm_ids, epoch=epoch, cfg=cfg)
        masked_R = mask_item_edges(r_train, selected)
        if masked_R[:, selected].nnz != 0:
            raise RuntimeError("selected pseudo-cold courses retained a graph edge")
        removed = r_train.tocoo()
        removed_mask = np.isin(removed.col, selected)
        removed_users = removed.row[removed_mask].astype(np.int64, copy=False)
        removed_items = removed.col[removed_mask].astype(np.int64, copy=False)
        with torch.no_grad():
            masked_adj = _build_sparse_graph(masked_R, model.n_users, model.n_items, device)
            masked_users, masked_items = cgrc._lightgcn_mean_all_layers(
                masked_adj,
                model.user_emb,
                model.item_x(),
                model.n_users,
                cfg.layers_full,
            )
            masked_users = F.normalize(masked_users, dim=1).detach()
            masked_items = F.normalize(masked_items, dim=1).detach()
        stats = _train_one_epoch_with_soft_anchor(
            adapter=adapter,
            optimizer=optimizer,
            content_bank=content_bank,
            masked_users=masked_users,
            masked_hot_items=masked_items,
            selected_items=selected,
            removed_users=removed_users,
            removed_items=removed_items,
            original_user_rated=original_user_rated,
            train_zero_mask=train_zero_mask,
            cfg=cfg,
            epoch=epoch,
        )
        row = _evaluate_validation(
            adapter=adapter,
            content_bank=content_bank,
            full_hot_items=full_items,
            full_hot_users=full_users,
            train_zero_mask=train_zero_mask,
            val_loader=val_loader,
            train_seen=train_seen,
            cfg=cfg,
            output_dir=output_dir,
            epoch=epoch,
        )
        row.update(stats)
        row["masked_item_count"] = int(selected.size)
        row["masked_edge_count"] = int(removed_users.size)
        audit = pseudo_cold_selection_audit(selected, cfg)
        row.update(audit)
        row["passes_retention_guards"] = bool(
            row["hot_r10"] >= immutable_baseline["hot_r10"] - cfg.retention_tolerance
            and row["hot_n10"] >= immutable_baseline["hot_n10"] - cfg.retention_tolerance
            and row["overall_r10"] >= immutable_baseline["overall_r10"] - cfg.retention_tolerance
            and row["overall_n10"] >= immutable_baseline["overall_n10"] - cfg.retention_tolerance
        )
        rows.append(row)
        _write_rows(output_dir / "validation_epochs.csv", rows)
        np.save(output_dir / f"epoch_{epoch:03d}_pseudocold_items.npy", selected)
        torch.save(
            {
                "epoch": epoch,
                "adapter_state": adapter.state_dict(),
                "config": asdict(cfg),
                "pseudo_cold_audit": audit,
            },
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
        )
        print(
            f"[COMPONENT] name={cfg.component_name} seed={cfg.seed} epoch={epoch}/{cfg.epochs} "
            f"loss={row['train_loss']:.4f} anchor={row['soft_anchor_loss']:.4f} "
            f"cold_N10={row['cold_n10']:.4f} overall_N10={row['overall_n10']:.4f}",
            flush=True,
        )
    try:
        selected_row = select_adapter_epoch(rows, immutable_baseline, cfg.retention_tolerance)
    except ValueError as exc:
        selected_row = None
        component_decision = "rejected_no_eligible_epoch"
        selection_error = str(exc)
    else:
        component_decision = component_selection_decision(
            selected_row,
            incumbent,
            immutable_baseline,
            cfg,
        )
        selection_error = None
    source_hashes_after = _component_source_hashes()
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("component source changed during the validation run")
    if _sha256(parent_path) != parent_result_sha256_before:
        raise RuntimeError("parent incumbent changed during the validation run")
    if _sha256(parent_manifest_path) != parent_manifest_sha256_before:
        raise RuntimeError("parent run manifest changed during the validation run")
    if _sha256(hot_manifest_path) != hot_manifest_sha256_before:
        raise RuntimeError("Hot manifest changed during the validation run")
    if _sha256(hot_result_path) != hot_result_sha256_before:
        raise RuntimeError("Hot result changed during the validation run")
    result = {
        "experiment": "ckg_stageb_component_screen",
        "status": "completed",
        "component_name": cfg.component_name,
        "phase": cfg.phase,
        "component_decision": component_decision,
        "passed_stage_b_screen": selected_row is not None,
        "config": asdict(cfg),
        "test_evaluation": False,
        "parent_result_path": str(parent_path),
        "parent_result_sha256": parent_result_sha256_before,
        "parent_manifest_path": str(parent_manifest_path),
        "parent_manifest_sha256": parent_manifest_sha256_before,
        "parent_selected_validation_epoch": incumbent,
        "hot_checkpoint_contract": contract,
        "hot_input_manifest_sha256": hot_manifest_sha256_before,
        "hot_result_sha256": hot_result_sha256_before,
        "stageb_source_sha256": source_hashes_before,
        "warm_q75_audit_recomputed": q75_audit,
        "immutable_hot_baseline": immutable_baseline,
        "selected_validation_epoch": selected_row,
        "selection_error": selection_error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parent_status": parent_result.get("status"),
    }
    _write_json(output_dir / "component_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict validation-only Stage-B component screen")
    parser.add_argument("--seed", required=True, type=int, choices=_STRICT_SEEDS)
    parser.add_argument("--component", required=True, choices=_ALLOWED_COMPONENTS)
    parser.add_argument("--phase", choices=("screen", "replication"), default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--parent-result", required=True)
    parser.add_argument("--hot-output-dir", required=True)
    parser.add_argument("--hot-checkpoint-dir", required=True)
    parser.add_argument("--device", default="")
    return parser


def _config_from_args(args: argparse.Namespace) -> ComponentScreenConfig:
    cfg = ComponentScreenConfig.for_seed(
        args.seed,
        component_name=args.component,
        phase=args.phase,
    )
    return replace(
        cfg,
        output_dir=str(args.output_dir),
        checkpoint_dir=str(args.checkpoint_dir),
        parent_result_path=str(args.parent_result),
        hot_output_dir=str(args.hot_output_dir),
        hot_checkpoint_dir=str(args.hot_checkpoint_dir),
        device=str(args.device),
    )


def main() -> None:
    result = run_component_screen(_config_from_args(_parser().parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
