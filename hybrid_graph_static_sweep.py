"""Validation-guarded hybrid sweep for the static and graph scorers.

The static checkpoint is the cold-start reference.  A graph residual is
interpolated into hot-item scores only, while evaluation still ranks the full
catalog so hot-score changes that displace cold positives are measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fast3_delta.static_protocol import load_shared_static_split
from static_content_scorer_clean import (
    ScorerConfig,
    StaticContentScorer,
    compute_train_popularity as static_train_popularity,
)


def _import_graph_backend(backend: str):
    """Load graph scorer symbols. 'gated' uses degree-gated propagate + item_beta."""
    if backend == "gated":
        from graph_gated_scorer_clean import (  # type: ignore
            GraphContentScorer,
            GraphScorerConfig,
            build_norm_adj,
        )
    elif backend == "content":
        from graph_content_scorer_clean import (  # type: ignore
            GraphContentScorer,
            GraphScorerConfig,
            build_norm_adj,
        )
    else:
        raise ValueError(f"unsupported graph backend: {backend}")
    return GraphContentScorer, GraphScorerConfig, build_norm_adj


def blend_scores(
    static_scores: np.ndarray,
    graph_scores: np.ndarray,
    hot_mask: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Interpolate graph residuals for hot columns and preserve cold columns."""
    static_scores = np.asarray(static_scores)
    graph_scores = np.asarray(graph_scores)
    hot_mask = np.asarray(hot_mask, dtype=bool)
    if static_scores.shape != graph_scores.shape:
        raise ValueError("static_scores and graph_scores must have the same shape")
    if static_scores.ndim < 1 or hot_mask.shape != (static_scores.shape[-1],):
        raise ValueError("hot_mask must match the last score dimension")
    shape = (1,) * (static_scores.ndim - 1) + (hot_mask.size,)
    return static_scores + float(lam) * (graph_scores - static_scores) * hot_mask.reshape(shape)


def rank_calibrated_hot_scores(
    static_scores: torch.Tensor,
    graph_scores: torch.Tensor,
    hot_mask: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Reassign the static hot score multiset using the hybrid hot ordering."""
    if static_scores.shape != graph_scores.shape or static_scores.ndim != 2:
        raise ValueError("static_scores and graph_scores must be matching 2D tensors")
    if hot_mask.shape != (static_scores.shape[1],):
        raise ValueError("hot_mask must match the score columns")
    if float(lam) == 0.0:
        return static_scores.clone()
    hot_indices = hot_mask.nonzero(as_tuple=False).view(-1)
    if hot_indices.numel() == 0:
        return static_scores.clone()
    static_hot = static_scores.index_select(1, hot_indices)
    graph_hot = graph_scores.index_select(1, hot_indices)
    priority = static_hot + float(lam) * (graph_hot - static_hot)
    static_values = torch.sort(static_hot, dim=1, descending=True).values
    priority_order = torch.argsort(priority, dim=1, descending=True)
    calibrated_hot = torch.empty_like(static_hot)
    calibrated_hot.scatter_(1, priority_order, static_values)
    out = static_scores.clone()
    out[:, hot_indices] = calibrated_hot
    return out


def select_guarded_lambda(
    rows: Iterable[dict],
    baseline_cold_r10: float,
    baseline_cold_n10: float,
    tolerance: float = 0.0,
) -> dict:
    """Maximize validation hot N@10 among candidates preserving cold metrics."""
    rows = [dict(row) for row in rows]
    eligible = [
        row
        for row in rows
        if row["cold_R@10"] >= baseline_cold_r10 - tolerance
        and row["cold_N@10"] >= baseline_cold_n10 - tolerance
    ]
    if not eligible:
        raise ValueError("no lambda satisfies the cold guardrail")
    return max(
        eligible,
        key=lambda row: (float(row["hot_N@10"]), -float(row["lambda"])),
    )


def build_user_seen(train_df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user, item in zip(train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy()):
        seen.setdefault(int(user), set()).add(int(item))
    return seen


def _load_state(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)


@torch.inference_mode()
def build_static_banks(
    meta: dict,
    content_emb: torch.Tensor,
    checkpoint: Path,
    train_pop: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = ScorerConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    model = StaticContentScorer(cfg, content_emb).to(device)
    _load_state(model, checkpoint, device)
    model.eval()
    all_users = torch.arange(cfg.n_users, dtype=torch.long, device=device)
    all_items = torch.arange(cfg.n_items, dtype=torch.long, device=device)
    cold = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool, device=device)
    user_bank = F.normalize(model.user_vector(all_users), dim=1)
    item_bank = F.normalize(model.item_vector(all_items, force_cold=cold), dim=1)
    user_bank = user_bank.detach()
    item_bank = item_bank.detach()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return user_bank, item_bank


@torch.inference_mode()
def build_graph_banks(
    meta: dict,
    content_emb: torch.Tensor,
    checkpoint: Path,
    train_df: pd.DataFrame,
    train_pop: np.ndarray,
    device: torch.device,
    graph_backend: str = "content",
) -> tuple[torch.Tensor, torch.Tensor]:
    GraphContentScorer, GraphScorerConfig, build_norm_adj = _import_graph_backend(graph_backend)
    cfg = GraphScorerConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    model = GraphContentScorer(cfg, content_emb).to(device)
    _load_state(model, checkpoint, device)
    if graph_backend == "gated" and hasattr(model, "set_item_degree"):
        # Recompute degree gate from this split's train popularity (leak-free).
        model.set_item_degree(train_pop)
    model.eval()
    adj = build_norm_adj(train_df, cfg.n_users, cfg.n_items, device)
    cold = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool, device=device)
    item_ego, _, _ = model.item_ego(apply_id_dropout=False, cold_mask_all=cold)
    user_bank, item_bank = model.propagate(adj, item_ego)
    user_bank = F.normalize(user_bank, dim=1).detach()
    item_bank = F.normalize(item_bank, dim=1).detach()
    del model, adj, item_ego
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return user_bank, item_bank


def _dcg_hit(rank_pos: int, k: int) -> float:
    return 1.0 / np.log2(rank_pos + 2.0) if rank_pos < k else 0.0


@torch.inference_mode()
def evaluate_banks(
    static_user: torch.Tensor,
    static_item: torch.Tensor,
    graph_user: torch.Tensor,
    graph_item: torch.Tensor,
    lam: float,
    eval_df: pd.DataFrame,
    train_pop: np.ndarray,
    user_seen: dict[int, set[int]],
    eval_batch_users: int = 512,
    k_list: tuple[int, ...] = (5, 10, 20),
    fusion_mode: str = "linear",
) -> dict:
    """Evaluate a hybrid score bank under the existing full-catalog protocol."""
    hot = torch.as_tensor(train_pop > 0, dtype=torch.bool, device=static_item.device)
    users = eval_df["u_idx"].to_numpy(dtype=np.int64)
    items = eval_df["i_idx"].to_numpy(dtype=np.int64)
    max_k = max(k_list)
    acc = {f"{metric}@{k}": {} for metric in ("R", "N") for k in k_list}
    counts: dict[int, int] = {}

    for start in range(0, len(users), eval_batch_users):
        u_batch = users[start:start + eval_batch_users]
        i_batch = items[start:start + eval_batch_users]
        u_t = torch.as_tensor(u_batch, dtype=torch.long, device=static_user.device)
        static_scores = static_user.index_select(0, u_t) @ static_item.t()
        graph_scores = graph_user.index_select(0, u_t) @ graph_item.t()
        for row_idx, user in enumerate(u_batch):
            seen = user_seen.get(int(user))
            if seen:
                positive = int(i_batch[row_idx])
                for seen_item in seen:
                    if seen_item != positive:
                        static_scores[row_idx, seen_item] = -1e30
                        graph_scores[row_idx, seen_item] = -1e30
        if fusion_mode == "linear":
            scores = static_scores + float(lam) * (graph_scores - static_scores) * hot.view(1, -1)
        elif fusion_mode == "rank_calibrated":
            scores = rank_calibrated_hot_scores(static_scores, graph_scores, hot, lam)
        else:
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        topk = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
        for row_idx in range(len(u_batch)):
            target = int(i_batch[row_idx])
            row = topk[row_idx]
            rank_pos = int(np.where(row == target)[0][0]) if (row == target).any() else max_k + 1
            counts[target] = counts.get(target, 0) + 1
            for k in k_list:
                acc[f"R@{k}"][target] = acc[f"R@{k}"].get(target, 0.0) + (rank_pos < k)
                acc[f"N@{k}"][target] = acc[f"N@{k}"].get(target, 0.0) + _dcg_hit(rank_pos, k)

    cold_np = train_pop < 1
    out: dict[str, float | int] = {}
    for split_name, filt in (("cold", lambda item: cold_np[item]), ("hot", lambda item: not cold_np[item])):
        for k in k_list:
            for metric in ("R", "N"):
                values = [
                    acc[f"{metric}@{k}"].get(item, 0.0) / max(1, count)
                    for item, count in counts.items()
                    if filt(item)
                ]
                out[f"{split_name}_{metric}@{k}"] = float(np.mean(values)) if values else 0.0
        out[f"{split_name}_count"] = int(sum(1 for item in counts if filt(item)))
    for k in k_list:
        for metric in ("R", "N"):
            cold_count = int(out["cold_count"])
            hot_count = int(out["hot_count"])
            out[f"overall_{metric}@{k}"] = (
                out[f"cold_{metric}@{k}"] * cold_count
                + out[f"hot_{metric}@{k}"] * hot_count
            ) / max(1, cold_count + hot_count)
    return out


def _parse_lambdas(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("lambda values must be in [0, 1]")
    return sorted(set(values))


def run_seed(
    seed: int,
    data_dir: Path,
    split_dir: Path,
    static_checkpoint: Path,
    graph_checkpoint: Path,
    output_dir: Path,
    lambdas: list[float],
    device: torch.device,
    cold_tolerance: float,
    fusion_mode: str,
    graph_backend: str = "content",
) -> dict:
    with (data_dir / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    content_emb = torch.load(data_dir / "content_emb.pt", map_location="cpu", weights_only=True).float()
    train_df, val_df, test_df, split_info = load_shared_static_split(str(split_dir))
    n_items = int(meta["n_items"])
    train_pop = static_train_popularity(train_df, n_items)
    user_seen = build_user_seen(train_df)

    static_user, static_item = build_static_banks(meta, content_emb, static_checkpoint, train_pop, device)
    graph_user, graph_item = build_graph_banks(
        meta, content_emb, graph_checkpoint, train_df, train_pop, device,
        graph_backend=graph_backend,
    )

    val_rows = []
    for lam in lambdas:
        metrics = evaluate_banks(
            static_user, static_item, graph_user, graph_item, lam,
            val_df, train_pop, user_seen,
            fusion_mode=fusion_mode,
        )
        val_rows.append({"lambda": lam, **metrics})
    baseline_val = next(row for row in val_rows if row["lambda"] == 0.0)
    selected = select_guarded_lambda(
        val_rows,
        baseline_cold_r10=float(baseline_val["cold_R@10"]),
        baseline_cold_n10=float(baseline_val["cold_N@10"]),
        tolerance=cold_tolerance,
    )
    test_baseline = evaluate_banks(
        static_user, static_item, graph_user, graph_item, 0.0,
        test_df, train_pop, user_seen,
        fusion_mode=fusion_mode,
    )
    test_selected = evaluate_banks(
        static_user, static_item, graph_user, graph_item, float(selected["lambda"]),
        test_df, train_pop, user_seen,
        fusion_mode=fusion_mode,
    )
    result = {
        "seed": seed,
        "split_dir": str(split_dir),
        "split_info": split_info,
        "static_checkpoint": str(static_checkpoint),
        "graph_checkpoint": str(graph_checkpoint),
        "graph_backend": graph_backend,
        "lambda_grid": lambdas,
        "cold_tolerance": cold_tolerance,
        "fusion_mode": fusion_mode,
        "selected": selected,
        "validation": val_rows,
        "test_baseline_static": test_baseline,
        "test_selected_hybrid": test_selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    parser.add_argument("--data-dir", type=Path, default=Path("processed_data_hin_clean_pop5"))
    parser.add_argument("--split-root", type=Path, default=Path("outputs/content_delta_pop5/static_item_cold_balanced"))
    parser.add_argument("--static-root", type=Path, default=Path("outputs/static_content_scorer_clean"))
    parser.add_argument("--graph-root", type=Path, default=Path("outputs/graph_content_scorer_clean"))
    parser.add_argument(
        "--graph-backend",
        choices=("content", "gated"),
        default="content",
        help="content=graph_content_scorer_clean; gated=graph_gated_scorer_clean (degree-gated banks)",
    )
    parser.add_argument("--static-ckpt-template", default="seed{seed}_prereq",
                        help="subdir under static-root; {seed} replaced")
    parser.add_argument("--graph-ckpt-template", default="seed{seed}",
                        help="subdir under graph-root; {seed} replaced")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hybrid_graph_static_sweep"))
    parser.add_argument("--lambdas", type=_parse_lambdas, default=_parse_lambdas("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"))
    parser.add_argument("--cold-tolerance", type=float, default=0.0)
    parser.add_argument("--fusion-mode", choices=("linear", "rank_calibrated"), default="linear")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    all_results = []
    for seed in args.seeds:
        split_dir = args.split_root / f"strict_item_cold_balanced_thr1_seed_{seed}"
        static_ckpt = args.static_root / args.static_ckpt_template.format(seed=seed) / "best.pt"
        graph_ckpt = args.graph_root / args.graph_ckpt_template.format(seed=seed) / "best.pt"
        if not static_ckpt.exists():
            raise FileNotFoundError(f"static checkpoint missing: {static_ckpt}")
        if not graph_ckpt.exists():
            raise FileNotFoundError(f"graph checkpoint missing: {graph_ckpt}")
        result = run_seed(
            seed=seed,
            data_dir=args.data_dir,
            split_dir=split_dir,
            static_checkpoint=static_ckpt,
            graph_checkpoint=graph_ckpt,
            output_dir=args.output_root / f"seed{seed}",
            lambdas=args.lambdas,
            device=device,
            cold_tolerance=args.cold_tolerance,
            fusion_mode=args.fusion_mode,
            graph_backend=args.graph_backend,
        )
        all_results.append(result)
        selected = result["selected"]
        test = result["test_selected_hybrid"]
        print(
            f"seed={seed} selected_lambda={selected['lambda']:.2f} "
            f"val cold N@10={selected['cold_N@10']:.4f} "
            f"test cold N@10={test['cold_N@10']:.4f} "
            f"test hot N@10={test['hot_N@10']:.4f}",
            flush=True,
        )
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


if __name__ == "__main__":
    main()
