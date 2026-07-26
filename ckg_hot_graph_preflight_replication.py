"""Strict train/validation-only Hot-expert preflight for seeds 2026 and 2027.

This is intentionally separate from the completed seed-2025 entrypoint so its
recorded source hash remains reproducible. It uses the same CGRC training
knobs and Hot validation floors while manually loading only the inputs needed
for this replication phase.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cgrc_paper_static_hin as cgrc
from ckg_hot_graph_preflight import (
    _epoch_train,
    _evaluate_validation,
    _resolve_device,
    _write_json,
    _write_validation_rows,
    preflight_gate_status,
    prepare_run_dirs,
    select_best_epoch,
)
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions, setup_seed
from lightgcn_static_hin import prepare_train_cache


_REPO_ROOT = Path(__file__).resolve().parent
_REPLICATION_SEEDS = (2026, 2027)


def _canonical_split_path(seed: int) -> Path:
    return (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    ).resolve()


@dataclass(frozen=True)
class ReplicationHotConfig:
    """Fixed Hot preflight protocol with only the split seed varying."""

    seed: int
    data_dir: str = "processed_data_hin_clean_pop5"
    split_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    epochs: int = 15
    batch_size: int = 4096
    emb_dim: int = 64
    mlp_hidden: int = 64
    layers_gprime: int = 2
    layers_full: int = 2
    mask_rho: float = 0.30
    lambda_e: float = 1.0
    tau: float = 0.50
    ranking_neg_per_user: int = 32
    le_max_edges: int = 4096
    recon_user_chunk: int = 4096
    lr: float = 1e-3
    reg_weight: float = 1e-4
    cold_threshold: int = 1
    hot_r10_floor: float = 0.2219
    hot_n10_floor: float = 0.1442
    device: str = ""
    test_evaluation: bool = False
    use_cbi: bool = False
    use_simulator: bool = False
    use_ppo: bool = False
    use_course_rewards: bool = False

    @classmethod
    def for_seed(cls, seed: int) -> "ReplicationHotConfig":
        seed = int(seed)
        return cls(
            seed=seed,
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                f"strict_item_cold_balanced_thr1_seed_{seed}"
            ),
            output_dir=f"outputs/ckg_hot_graph_preflight_replication_seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_hot_graph_preflight_replication_seed{seed}",
        )


def validate_replication_hot_config(cfg: ReplicationHotConfig) -> None:
    """Reject any seed or training knob outside the registered replication."""
    if int(cfg.seed) not in _REPLICATION_SEEDS:
        raise ValueError("replication Hot preflight is restricted to seeds 2026 and 2027")
    if Path(cfg.split_dir).resolve() != _canonical_split_path(cfg.seed):
        raise ValueError("replication Hot preflight requires the canonical seed-specific split")
    if any((cfg.test_evaluation, cfg.use_cbi, cfg.use_simulator, cfg.use_ppo, cfg.use_course_rewards)):
        raise ValueError("test evaluation, CBI, simulation, PPO, and course rewards are forbidden")
    expected = ReplicationHotConfig.for_seed(cfg.seed)
    for field in (
        "epochs", "batch_size", "emb_dim", "mlp_hidden", "layers_gprime", "layers_full",
        "mask_rho", "lambda_e", "tau", "ranking_neg_per_user", "le_max_edges",
        "recon_user_chunk", "lr", "reg_weight", "cold_threshold", "hot_r10_floor",
        "hot_n10_floor",
    ):
        if getattr(cfg, field) != getattr(expected, field):
            raise ValueError(f"replication Hot knob is locked: {field}")


def load_hot_replication_inputs(
    data_dir: str | Path, split_dir: str | Path
) -> tuple[dict[str, Any], torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Load exactly meta, content, static train, and static validation."""
    data_root = Path(data_dir)
    split_root = Path(split_dir)
    paths = {
        "meta": data_root / "meta.json",
        "content": data_root / "content_emb.pt",
        "train": split_root / "static_train.pkl",
        "validation": split_root / "static_val.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing replication Hot input: " + ", ".join(missing))
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    try:
        content = torch.load(paths["content"], map_location="cpu", weights_only=False)
    except TypeError:
        content = torch.load(paths["content"], map_location="cpu")
    if not isinstance(content, torch.Tensor):
        raise ValueError("content embedding must be a tensor")
    train = pd.read_pickle(paths["train"]).copy()
    validation = pd.read_pickle(paths["validation"]).copy()
    required = {"u_idx", "i_idx"}
    if not required.issubset(train.columns) or not required.issubset(validation.columns):
        raise ValueError("train and validation frames must contain u_idx and i_idx")
    counts = train["i_idx"].astype(int).value_counts().astype(int)
    for frame in (train, validation):
        frame["popularity"] = frame["i_idx"].astype(int).map(counts).fillna(0).astype(int)
    return meta, content.float(), train, validation


def run_replication_hot_preflight(cfg: ReplicationHotConfig) -> dict[str, Any]:
    """Train and validate one seed-specific frozen Hot expert."""
    validate_replication_hot_config(cfg)
    setup_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    prepare_run_dirs(output_dir, checkpoint_dir)
    data_root = Path(cfg.data_dir)
    split_root = Path(cfg.split_dir)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root
    if not split_root.is_absolute():
        split_root = _REPO_ROOT / split_root
    meta, content, train_df, val_df = load_hot_replication_inputs(data_root.resolve(), split_root.resolve())
    train_seen = build_user_seen(train_df)
    val_loader = DataLoader(
        InteractionDataset(val_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_interactions
    )
    device = _resolve_device(cfg.device)
    model = cgrc.CGRCNet(
        int(meta["n_users"]), int(meta["n_items"]), int(content.shape[1]), cfg.emb_dim, cfg.mlp_hidden, content
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_users, train_pos, _, user_neg_pool = prepare_train_cache(train_df, int(meta["n_items"]))
    train_item_pool = np.unique(train_pos).astype(np.int64, copy=False)
    user_rated = cgrc._build_user_rated(train_df, int(meta["n_users"]))
    r_base = cgrc._build_interaction_csr(train_df, int(meta["n_users"]), int(meta["n_items"]))
    r_coo = r_base.tocoo()
    sparse_full = cgrc._sparse_adj_tensor(
        cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(r_base, model.n_users, model.n_items)), device
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        stats = _epoch_train(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            r_base=r_base,
            sparse_full=sparse_full,
            r_coo_row=r_coo.row.astype(np.int64, copy=False),
            r_coo_col=r_coo.col.astype(np.int64, copy=False),
            train_users=train_users,
            train_pos=train_pos,
            user_rated=user_rated,
            user_neg_pool=user_neg_pool,
            train_item_pool=train_item_pool,
        )
        row = _evaluate_validation(
            model=model,
            cfg=cfg,
            device=device,
            sparse_full=sparse_full,
            val_loader=val_loader,
            train_seen=train_seen,
            epoch=epoch,
            output_dir=output_dir,
        )
        row.update(stats)
        row["passes_hot_floor"] = bool(
            row["hot_r10"] >= cfg.hot_r10_floor and row["hot_n10"] >= cfg.hot_n10_floor
        )
        row["hot_capacity_score"] = row["hot_r10"] + row["hot_n10"]
        rows.append(row)
        _write_validation_rows(output_dir / "validation_epochs.csv", rows)
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "config": asdict(cfg)},
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
        )
        print(
            f"[HOT-REPLICATION] seed={cfg.seed} epoch={epoch}/{cfg.epochs} "
            f"loss={row['train_loss']:.4f} hot_R10={row['hot_r10']:.4f} "
            f"hot_N10={row['hot_n10']:.4f} overall_R10={row['overall_r10']:.4f} "
            f"overall_N10={row['overall_n10']:.4f} pass={row['passes_hot_floor']}",
            flush=True,
        )
    best = select_best_epoch(rows, hot_r10_floor=cfg.hot_r10_floor, hot_n10_floor=cfg.hot_n10_floor)
    result = {
        "experiment": "ckg_hot_graph_preflight_replication",
        "input_protocol": "manual_meta_content_static_train_static_val",
        "config": asdict(cfg),
        "test_evaluation": False,
        "selected_validation_epoch": best,
        "passed_hot_preflight": bool(best["passes_hot_floor"]),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    result["gate_status"] = preflight_gate_status(result)
    _write_json(output_dir / "preflight_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="")
    return parser


def _config_from_args(args: argparse.Namespace) -> ReplicationHotConfig:
    base = ReplicationHotConfig.for_seed(args.seed)
    return dataclass_replace(
        base,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


def dataclass_replace(cfg: ReplicationHotConfig, **changes: Any) -> ReplicationHotConfig:
    return ReplicationHotConfig(**{**asdict(cfg), **changes})


def main() -> None:
    args = _parser().parse_args()
    result = run_replication_hot_preflight(_config_from_args(args))
    if result["gate_status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
