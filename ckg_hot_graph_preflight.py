"""Validation-only warm graph-expert preflight for the CKG-RL dual-route design.

This entrypoint deliberately contains no CBI, simulator, PPO, course-reward, or
test-evaluation path.  It trains a fresh graph expert on the existing strict
split and asks one question: can the Hot route reach the pre-registered
validation capacity floor before a Cold adapter is introduced?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cgrc_paper_static_hin as cgrc
from hin_data_common import (
    InteractionDataset,
    build_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    static_split_df,
)
from hin_eval_common import evaluate_embedding_ranker
from lightgcn_static_hin import prepare_train_cache


_REPO_ROOT = Path(__file__).resolve().parent


def _canonical_split_path(seed: int) -> Path:
    return (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    ).resolve()


@dataclass(frozen=True)
class PreflightConfig:
    seed: int
    data_dir: str = "processed_data_hin_clean_pop5"
    split_dir: str = "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025"
    output_dir: str = "outputs/ckg_hot_graph_preflight_seed2025"
    checkpoint_dir: str = "checkpoints/ckg_hot_graph_preflight_seed2025"
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
    use_cbi: bool = False
    use_simulator: bool = False
    use_ppo: bool = False
    use_course_rewards: bool = False
    test_evaluation: bool = False

    @classmethod
    def for_seed(cls, seed: int) -> "PreflightConfig":
        seed = int(seed)
        return cls(
            seed=seed,
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                f"strict_item_cold_balanced_thr1_seed_{seed}"
            ),
            output_dir=f"outputs/ckg_hot_graph_preflight_seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_hot_graph_preflight_seed{seed}",
        )


def drop_item_edges(graph: sp.csr_matrix, item_ids: Sequence[int]) -> sp.csr_matrix:
    """Return a copy of a user-item graph without every edge to ``item_ids``."""
    coo = graph.tocoo()
    ids = np.asarray(item_ids, dtype=np.int64)
    keep = ~np.isin(coo.col, ids)
    out = sp.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=graph.shape,
        dtype=graph.dtype,
    )
    out.eliminate_zeros()
    return out


def count_weighted_overall(
    cold_value: float,
    cold_count: int,
    hot_value: float,
    hot_count: int,
) -> float:
    total = int(cold_count) + int(hot_count)
    if total <= 0:
        raise ValueError("cold_count + hot_count must be positive")
    return (float(cold_value) * int(cold_count) + float(hot_value) * int(hot_count)) / total


def select_best_epoch(
    rows: Iterable[Dict[str, Any]],
    *,
    hot_r10_floor: float,
    hot_n10_floor: float,
) -> Dict[str, Any]:
    """Choose the strongest Hot checkpoint, prioritizing checkpoints that pass both floors."""
    candidates = []
    for source in rows:
        row = dict(source)
        row["passes_hot_floor"] = bool(
            float(row["hot_r10"]) >= float(hot_r10_floor)
            and float(row["hot_n10"]) >= float(hot_n10_floor)
        )
        row["hot_capacity_score"] = float(row["hot_r10"]) + float(row["hot_n10"])
        candidates.append(row)
    if not candidates:
        raise ValueError("cannot select a checkpoint from no validation rows")
    return max(
        candidates,
        key=lambda row: (
            int(row["passes_hot_floor"]),
            float(row["hot_capacity_score"]),
            int(row["epoch"]),
        ),
    )


def preflight_gate_status(result: Dict[str, Any]) -> str:
    return "completed" if bool(result.get("passed_hot_preflight")) else "completed_gate_failed"


def align_train_popularity(
    train_df: pd.DataFrame, validation_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign popularity exclusively from train rows; no test rows are consulted."""
    counts = train_df["i_idx"].astype(int).value_counts().astype(int)

    def align(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "raw_popularity" not in out.columns and "popularity" in out.columns:
            out["raw_popularity"] = out["popularity"]
        out["popularity"] = out["i_idx"].astype(int).map(counts).fillna(0).astype(int)
        return out

    return align(train_df), align(validation_df)


def _resolve_device(requested: str) -> torch.device:
    raw = str(requested).strip().lower()
    if raw:
        if raw == "cpu":
            return torch.device("cpu")
        if raw.startswith("cuda") and torch.cuda.is_available():
            return torch.device(raw)
        raise RuntimeError(f"requested unavailable device: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_validation_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "cold_r10",
        "cold_n10",
        "hot_r10",
        "hot_n10",
        "overall_r10",
        "overall_n10",
        "cold_item_count",
        "hot_item_count",
        "train_loss",
        "edge_recon_loss",
        "ranking_loss",
        "passes_hot_floor",
        "hot_capacity_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_run_dirs(output_dir: Path, checkpoint_dir: Path) -> None:
    """Create fresh run directories while allowing the launcher's manifest file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "preflight_result.json").exists():
        raise FileExistsError(f"preflight result already exists: {output_dir / 'preflight_result.json'}")
    if any(checkpoint_dir.glob("epoch_*.pt")):
        raise FileExistsError(f"preflight checkpoints already exist: {checkpoint_dir}")


def _validate_config(cfg: PreflightConfig) -> None:
    if cfg.seed != 2025:
        raise ValueError("this preflight is locked to seed 2025")
    if Path(cfg.split_dir).resolve() != _canonical_split_path(cfg.seed):
        raise ValueError("preflight must use the canonical shared split for seed 2025")
    if cfg.test_evaluation:
        raise ValueError("test evaluation is forbidden in the Hot-expert preflight")
    if any((cfg.use_cbi, cfg.use_simulator, cfg.use_ppo, cfg.use_course_rewards)):
        raise ValueError("CBI, simulation, PPO, and course rewards must all be disabled")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if cfg.hot_r10_floor <= 0.0 or cfg.hot_n10_floor <= 0.0:
        raise ValueError("Hot validation floors must be positive")


def _epoch_train(
    *,
    model: cgrc.CGRCNet,
    optimizer: torch.optim.Optimizer,
    cfg: PreflightConfig,
    device: torch.device,
    r_base: sp.csr_matrix,
    sparse_full: torch.Tensor,
    r_coo_row: np.ndarray,
    r_coo_col: np.ndarray,
    train_users: np.ndarray,
    train_pos: np.ndarray,
    user_rated: Sequence[set],
    user_neg_pool: Dict[int, np.ndarray],
    train_item_pool: np.ndarray,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    loss_e_sum = 0.0
    loss_r_sum = 0.0
    batches = 0
    for u_idx, i_idx, candidate_items in cgrc._iter_cgrc_batches(
        train_users,
        train_pos,
        user_rated,
        user_neg_pool,
        train_item_pool,
        cfg.batch_size,
        cfg.ranking_neg_per_user,
    ):
        x_all = model.item_x()
        masked_item_ids = cgrc._sample_cold_items(train_item_pool, cfg.mask_rho, device)
        loss_e = torch.zeros((), device=device, dtype=x_all.dtype)
        if masked_item_ids.numel() > 0:
            masked_cpu = masked_item_ids.detach().cpu().numpy()
            masked_edges = cgrc._masked_edges(r_coo_row, r_coo_col, masked_cpu)
            if len(masked_edges) > cfg.le_max_edges:
                chosen = np.random.choice(len(masked_edges), size=cfg.le_max_edges, replace=False)
                masked_edges = [masked_edges[int(index)] for index in chosen]
            if masked_edges:
                r_masked = drop_item_edges(r_base, masked_cpu)
                sparse_masked = cgrc._sparse_adj_tensor(
                    cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(r_masked, model.n_users, model.n_items)),
                    device,
                )
                layers = cgrc._propagate_gprime_frozen_cold(
                    sparse_masked,
                    model.user_emb,
                    x_all,
                    model.n_users,
                    cfg.layers_gprime,
                    masked_item_ids,
                )
                h_u_bar = cgrc._user_mean_layers_1_to_L(
                    layers, model.n_users, cfg.layers_gprime
                )
                users = sorted({int(uid) for uid, _ in masked_edges})
                user_ids = torch.tensor(users, dtype=torch.long, device=device)
                logits = cgrc._edge_logits_broadcast_chunked(
                    model,
                    h_u_bar[user_ids],
                    x_all,
                    masked_item_ids,
                    cfg.recon_user_chunk,
                )
                loss_e = cgrc._reconstruction_loss(
                    logits, masked_item_ids, masked_edges, user_ids, user_rated
                )

        z_u, z_i = cgrc._lightgcn_mean_all_layers(
            sparse_full, model.user_emb, x_all, model.n_users, cfg.layers_full
        )
        loss_r = cgrc._ranking_loss(
            z_u, z_i, u_idx, i_idx, candidate_items, user_rated, cfg.tau
        )
        u_tensor = torch.tensor(u_idx, dtype=torch.long, device=device)
        i_tensor = torch.tensor(i_idx, dtype=torch.long, device=device)
        reg = cgrc._l2_reg_loss(cfg.reg_weight, model.user_emb[u_tensor], x_all[i_tensor])
        loss = loss_r + cfg.lambda_e * loss_e + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().item())
        loss_e_sum += float(loss_e.detach().item())
        loss_r_sum += float(loss_r.detach().item())
        batches += 1
    denom = max(1, batches)
    return {
        "train_loss": loss_sum / denom,
        "edge_recon_loss": loss_e_sum / denom,
        "ranking_loss": loss_r_sum / denom,
    }


def _evaluate_validation(
    *,
    model: cgrc.CGRCNet,
    cfg: PreflightConfig,
    device: torch.device,
    sparse_full: torch.Tensor,
    val_loader: DataLoader,
    train_seen: Dict[int, set],
    epoch: int,
    output_dir: Path,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        all_u, all_i = cgrc._lightgcn_mean_all_layers(
            sparse_full,
            model.user_emb,
            model.item_x(),
            model.n_users,
            cfg.layers_full,
        )
        all_u = F.normalize(all_u, dim=1)
        all_i = F.normalize(all_i, dim=1)
        get_user = lambda batch: all_u[batch["u"]]
        cold, cold_count = evaluate_embedding_ranker(
            val_loader,
            device=device,
            n_items=model.n_items,
            cold_threshold=cfg.cold_threshold,
            get_user_vectors_fn=get_user,
            all_item_vectors=all_i,
            k_list=(5, 10, 20),
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            average_mode="item_macro",
            export_item_metrics_path=str(output_dir / f"epoch_{epoch:03d}_per_item_cold.csv"),
        )
        hot, hot_count = evaluate_embedding_ranker(
            val_loader,
            device=device,
            n_items=model.n_items,
            cold_threshold=cfg.cold_threshold,
            get_user_vectors_fn=get_user,
            all_item_vectors=all_i,
            k_list=(5, 10, 20),
            eval_type="hot",
            full_ranking=True,
            user_seen_items=train_seen,
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
        "overall_r10": count_weighted_overall(cold["R@10"], cold_count, hot["R@10"], hot_count),
        "overall_n10": count_weighted_overall(cold["N@10"], cold_count, hot["N@10"], hot_count),
        "cold_item_count": int(cold_count),
        "hot_item_count": int(hot_count),
    }


def run_preflight(cfg: PreflightConfig) -> Dict[str, Any]:
    _validate_config(cfg)
    setup_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    prepare_run_dirs(output_dir, checkpoint_dir)
    os.environ["USIM_STATIC_SPLIT_DIR"] = str(Path(cfg.split_dir).resolve())
    meta, frame, content = load_hin_processed(cfg.data_dir)
    train_df, val_df, _ = static_split_df(frame, seed=cfg.seed)
    train_df, val_df = align_train_popularity(train_df, val_df)
    train_seen = build_user_seen(train_df)
    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    device = _resolve_device(cfg.device)
    model = cgrc.CGRCNet(
        meta["n_users"],
        meta["n_items"],
        int(content.shape[1]),
        cfg.emb_dim,
        cfg.mlp_hidden,
        content,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_users, train_pos, _, user_neg_pool = prepare_train_cache(train_df, meta["n_items"])
    train_item_pool = np.unique(train_pos).astype(np.int64, copy=False)
    user_rated = cgrc._build_user_rated(train_df, meta["n_users"])
    r_base = cgrc._build_interaction_csr(train_df, meta["n_users"], meta["n_items"])
    r_coo = r_base.tocoo()
    sparse_full = cgrc._sparse_adj_tensor(
        cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(r_base, meta["n_users"], meta["n_items"])),
        device,
    )
    rows: list[Dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_stats = _epoch_train(
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
        row.update(train_stats)
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
            f"[HOT-PREFLIGHT] epoch={epoch}/{cfg.epochs} "
            f"loss={row['train_loss']:.4f} hot_R10={row['hot_r10']:.4f} "
            f"hot_N10={row['hot_n10']:.4f} overall_R10={row['overall_r10']:.4f} "
            f"overall_N10={row['overall_n10']:.4f} pass={row['passes_hot_floor']}",
            flush=True,
        )
    best = select_best_epoch(
        rows, hot_r10_floor=cfg.hot_r10_floor, hot_n10_floor=cfg.hot_n10_floor
    )
    result = {
        "experiment": "ckg_hot_graph_preflight",
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
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--layers-gprime", type=int, default=2)
    parser.add_argument("--layers-full", type=int, default=2)
    parser.add_argument("--mask-rho", type=float, default=0.30)
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--ranking-neg-per-user", type=int, default=32)
    parser.add_argument("--le-max-edges", type=int, default=4096)
    parser.add_argument("--recon-user-chunk", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--hot-r10-floor", type=float, default=0.2219)
    parser.add_argument("--hot-n10-floor", type=float, default=0.1442)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> PreflightConfig:
    return PreflightConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        mlp_hidden=args.mlp_hidden,
        layers_gprime=args.layers_gprime,
        layers_full=args.layers_full,
        mask_rho=args.mask_rho,
        lambda_e=args.lambda_e,
        tau=args.tau,
        ranking_neg_per_user=args.ranking_neg_per_user,
        le_max_edges=args.le_max_edges,
        recon_user_chunk=args.recon_user_chunk,
        lr=args.lr,
        reg_weight=args.reg_weight,
        cold_threshold=args.cold_threshold,
        device=args.device,
        hot_r10_floor=args.hot_r10_floor,
        hot_n10_floor=args.hot_n10_floor,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = _config_from_args(args)
    _validate_config(cfg)
    if args.dry_run:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        return
    print(json.dumps(run_preflight(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
