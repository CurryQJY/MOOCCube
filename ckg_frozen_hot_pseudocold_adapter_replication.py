"""Validation-only Stage B replication for strict split seeds 2026 and 2027.

It consumes a completed seed-specific Hot checkpoint contract and preserves all
registered seed-2025 adapter hyperparameters, including the fixed trust tau.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

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
    _train_one_epoch,
    _write_json,
    _write_rows,
    build_adapter_optimizer,
    derive_train_item_partitions,
    initialize_validation_rows,
    load_validation_only_inputs,
    require_preflight_input_hashes,
    select_adapter_epoch,
)
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions, setup_seed


_REPO_ROOT = Path(__file__).resolve().parent
_REPLICATION_SEEDS = (2026, 2027)
_FIXED_TAU = 0.24929234
_ARCHITECTURE = {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 2}


def _canonical_split_path(seed: int) -> Path:
    return (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    ).resolve()


@dataclass(frozen=True)
class ReplicationAdapterConfig:
    seed: int
    data_dir: str = "processed_data_hin_clean_pop5"
    split_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    hot_output_dir: str = ""
    hot_checkpoint_dir: str = ""
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
    def for_seed(cls, seed: int) -> "ReplicationAdapterConfig":
        seed = int(seed)
        return cls(
            seed=seed,
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                f"strict_item_cold_balanced_thr1_seed_{seed}"
            ),
            output_dir=f"outputs/ckg_frozen_hot_pseudocold_adapter_replication_seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_frozen_hot_pseudocold_adapter_replication_seed{seed}",
            hot_output_dir=f"outputs/ckg_hot_graph_preflight_replication_seed{seed}",
            hot_checkpoint_dir=f"checkpoints/ckg_hot_graph_preflight_replication_seed{seed}",
        )


def validate_replication_adapter_config(cfg: ReplicationAdapterConfig) -> None:
    if int(cfg.seed) not in _REPLICATION_SEEDS:
        raise ValueError("replication adapter is restricted to seeds 2026 and 2027")
    if Path(cfg.split_dir).resolve() != _canonical_split_path(cfg.seed):
        raise ValueError("replication adapter requires the canonical seed-specific split")
    if any((cfg.test_evaluation, cfg.use_cbi, cfg.use_simulator, cfg.use_ppo, cfg.use_course_rewards)):
        raise ValueError("test evaluation, CBI, simulation, PPO, and course rewards are forbidden")
    expected = ReplicationAdapterConfig.for_seed(cfg.seed)
    for field in (
        "n_items", "warm_item_count", "train_zero_item_count", "pseudo_cold_item_count", "trust_tau",
        "epochs", "emb_dim", "hidden_dim", "layers_full", "batch_size", "negatives_per_positive",
        "ranking_temperature", "lr", "weight_decay", "delta_reg_weight", "parity_atol",
        "retention_tolerance", "cold_gain_minimum", "cold_threshold",
    ):
        if getattr(cfg, field) != getattr(expected, field):
            raise ValueError(f"replication adapter knob is locked: {field}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Hot replication record: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError("Hot replication result must be a JSON object")
    return result


def require_replication_hot_contract(
    result: Mapping[str, Any], *, seed: int, checkpoint_dir: str | Path
) -> dict[str, Any]:
    """Validate the Hot result's actual selected checkpoint contract."""
    if result.get("passed_hot_preflight") is not True or result.get("gate_status") != "completed":
        raise ValueError("completed passed Hot replication result is required")
    config = result.get("config")
    if config is not None and (
        not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(seed)
    ):
        raise ValueError("Hot replication result seed mismatch")
    selected = result.get("selected_validation_epoch")
    contract = result.get("selected_checkpoint_contract")
    if not isinstance(selected, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("Hot replication result lacks its selected checkpoint contract")
    required = {"schema_version", "seed", "epoch", "relative_path", "sha256", "architecture", "fixed_trust_tau", "warm_q75_audit"}
    if not required.issubset(contract):
        raise ValueError("Hot replication checkpoint contract is incomplete")
    if int(contract["schema_version"]) != 1 or int(contract["seed"]) != int(seed):
        raise ValueError("Hot replication checkpoint contract seed mismatch")
    if int(contract["epoch"]) != int(selected.get("epoch", -1)):
        raise ValueError("Hot replication checkpoint contract epoch mismatch")
    if not math.isclose(float(contract["fixed_trust_tau"]), _FIXED_TAU, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Hot replication checkpoint contract tau mismatch")
    architecture = contract["architecture"]
    if not isinstance(architecture, Mapping) or any(int(architecture.get(key, -1)) != value for key, value in _ARCHITECTURE.items()):
        raise ValueError("Hot replication checkpoint contract architecture mismatch")
    name = str(contract["relative_path"])
    expected_name = f"epoch_{int(contract['epoch']):03d}.pt"
    if Path(name).name != name or name != expected_name:
        raise ValueError("Hot replication checkpoint contract relative path is invalid")
    checkpoint = Path(checkpoint_dir) / name
    if not checkpoint.is_file() or _sha256(checkpoint) != str(contract["sha256"]).lower():
        raise ValueError("Hot replication selected checkpoint SHA256 mismatch")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("epoch", -1)) != int(contract["epoch"]):
        raise ValueError("Hot replication selected checkpoint payload epoch mismatch")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Hot replication selected checkpoint payload architecture mismatch")
    if int(checkpoint_config.get("seed", -1)) != int(seed):
        raise ValueError("Hot replication selected checkpoint payload seed mismatch")
    if any(int(checkpoint_config.get(key, -1)) != value for key, value in _ARCHITECTURE.items()):
        raise ValueError("Hot replication selected checkpoint payload architecture mismatch")
    return dict(contract)


def _resolve_roots(cfg: ReplicationAdapterConfig) -> tuple[Path, Path]:
    data = Path(cfg.data_dir)
    split = Path(cfg.split_dir)
    if not data.is_absolute():
        data = _REPO_ROOT / data
    if not split.is_absolute():
        split = _REPO_ROOT / split
    return data.resolve(), split.resolve()


def _preflight_files(cfg: ReplicationAdapterConfig) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    data_root, split_root = _resolve_roots(cfg)
    prefix = "outputs\\content_delta_pop5\\static_item_cold_balanced\\" f"strict_item_cold_balanced_thr1_seed_{cfg.seed}\\"
    return (
        {
            "processed_data_hin_clean_pop5\\meta.json": data_root / "meta.json",
            "processed_data_hin_clean_pop5\\content_emb.pt": data_root / "content_emb.pt",
        },
        {prefix + "static_train.pkl": split_root / "static_train.pkl", prefix + "static_val.pkl": split_root / "static_val.pkl"},
        {
            "ckg_hot_graph_preflight_replication.py": _REPO_ROOT / "ckg_hot_graph_preflight_replication.py",
            "ckg_hot_replication_contract.py": _REPO_ROOT / "ckg_hot_replication_contract.py",
            "ckg_hot_graph_preflight.py": _REPO_ROOT / "ckg_hot_graph_preflight.py",
            "cgrc_paper_static_hin.py": _REPO_ROOT / "cgrc_paper_static_hin.py",
            "hin_data_common.py": _REPO_ROOT / "hin_data_common.py",
            "hin_eval_common.py": _REPO_ROOT / "hin_eval_common.py",
            "lightgcn_static_hin.py": _REPO_ROOT / "lightgcn_static_hin.py",
        },
    )


def run_replication_adapter(cfg: ReplicationAdapterConfig) -> dict[str, Any]:
    """Run one seed-specific validation-only Stage B replication."""
    validate_replication_adapter_config(cfg)
    setup_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    if (output_dir / "adapter_preflight_result.json").exists():
        raise FileExistsError("replication adapter result already exists")
    hot_output = Path(cfg.hot_output_dir)
    hot_manifest = _read_json(hot_output / "run_manifest.json")
    if hot_manifest.get("status") != "completed" or hot_manifest.get("gate_status") != "completed":
        raise ValueError("completed Hot replication manifest is required")
    hot_result = _read_json(hot_output / "preflight_result.json")
    contract = require_replication_hot_contract(hot_result, seed=cfg.seed, checkpoint_dir=cfg.hot_checkpoint_dir)
    data_files, split_files, source_files = _preflight_files(cfg)
    require_preflight_input_hashes(hot_manifest, data_files=data_files, split_files=split_files, source_files=source_files)
    data_root, split_root = _resolve_roots(cfg)
    meta, content, train_df, val_df = load_validation_only_inputs(data_root, split_root)
    if int(meta.get("n_items", -1)) != cfg.n_items or int(content.shape[0]) != cfg.n_items:
        raise ValueError("replication catalog shape does not match the registered protocol")
    device = _resolve_device(cfg.device)
    checkpoint_path = Path(cfg.hot_checkpoint_dir) / str(contract["relative_path"])
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    model = cgrc.CGRCNet(
        int(meta["n_users"]), int(meta["n_items"]), int(content.shape[1]), cfg.emb_dim, cfg.hidden_dim, content
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    counts = train_df["i_idx"].astype(int).value_counts()
    train_zero_mask = np.array([int(counts.get(index, 0)) == 0 for index in range(cfg.n_items)], dtype=bool)
    warm_ids, _ = derive_train_item_partitions(train_zero_mask, cfg)
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
    q75_audit = float(torch.quantile((full_items[warm_ids] - content_bank[warm_ids]).norm(dim=1), 0.75).item())
    if abs(q75_audit - float(contract["warm_q75_audit"])) > 1e-4:
        raise ValueError("replication Hot q75 audit does not match the registered contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    epoch_zero.update({
        "train_loss": 0.0, "ranking_loss": 0.0, "mean_final_delta": 0.0,
        "masked_item_count": 0, "masked_edge_count": 0, "pseudo_cold_item_count": 0,
        "warm_item_count": int(cfg.warm_item_count), "train_zero_item_count": int(cfg.train_zero_item_count),
        "pseudo_cold_warm_ratio": 0.0, "pseudo_cold_ids_sha256": "",
    })
    rows, baseline = initialize_validation_rows(epoch_zero, hot_result, cfg.parity_atol)
    _write_rows(output_dir / "validation_epochs.csv", rows)
    optimizer = build_adapter_optimizer(adapter, cfg)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in adapter.parameters()}:
        raise RuntimeError("replication optimizer must contain only adapter parameters")
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        from ckg_frozen_hot_pseudocold_adapter import mask_item_edges, pseudo_cold_selection_audit, select_epoch_pseudocold_items

        selected = select_epoch_pseudocold_items(warm_ids, epoch=epoch, cfg=cfg)
        masked_R = mask_item_edges(r_train, selected)
        if masked_R[:, selected].nnz != 0:
            raise RuntimeError("selected pseudo-cold courses retained student graph edges")
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
        row.update(pseudo_cold_selection_audit(selected, cfg))
        row["passes_retention_guards"] = bool(
            row["hot_r10"] >= baseline["hot_r10"] - cfg.retention_tolerance
            and row["hot_n10"] >= baseline["hot_n10"] - cfg.retention_tolerance
            and row["overall_r10"] >= baseline["overall_r10"] - cfg.retention_tolerance
            and row["overall_n10"] >= baseline["overall_n10"] - cfg.retention_tolerance
        )
        rows.append(row)
        _write_rows(output_dir / "validation_epochs.csv", rows)
        torch.save(
            {"epoch": epoch, "adapter_state": adapter.state_dict(), "config": asdict(cfg), "hot_contract": contract},
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
        )
    try:
        selected_row = select_adapter_epoch(rows, baseline, cfg.retention_tolerance)
    except ValueError as exc:
        selected_row, passed, selection_error = None, False, str(exc)
    else:
        passed = bool(
            selected_row["cold_n10"] >= baseline["cold_n10"] + cfg.cold_gain_minimum
            and selected_row["cold_r10"] >= baseline["cold_r10"]
        )
        selection_error = None
    status = "completed" if passed else "completed_gate_failed"
    result = {
        "experiment": "ckg_frozen_hot_masked_pseudocold_adapter_replication",
        "status": status,
        "config": asdict(cfg),
        "test_evaluation": False,
        "hot_checkpoint_contract": contract,
        "warm_q75_audit_recomputed": q75_audit,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hot-output-dir", required=True)
    parser.add_argument("--hot-checkpoint-dir", required=True)
    parser.add_argument("--device", default="")
    return parser


def _config_from_args(args: argparse.Namespace) -> ReplicationAdapterConfig:
    base = ReplicationAdapterConfig.for_seed(args.seed)
    return ReplicationAdapterConfig(**{
        **asdict(base),
        "data_dir": args.data_dir,
        "split_dir": args.split_dir,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "hot_output_dir": args.hot_output_dir,
        "hot_checkpoint_dir": args.hot_checkpoint_dir,
        "device": args.device,
    })


def main() -> None:
    args = _parser().parse_args()
    result = run_replication_adapter(_config_from_args(args))
    if result["gate_status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
