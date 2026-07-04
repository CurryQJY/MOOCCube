"""COCO paired seed-item significance tests for item-macro metrics.

The main comparison is CKG-RL (ours) against CCFCRec, the strongest COCO
baseline in the current main table. CGRC-paper is included as a secondary
comparison when its per-item exports are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
SCOPES = ["cold", "hot"]
DEFAULT_SEEDS = ["2025", "2026", "2027"]


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    max_elems: int = 8_000_000,
) -> tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    chunk = max(1, min(n_boot, max_elems // max(1, diffs.size)))
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        end = min(n_boot, start + chunk)
        idx = rng.integers(0, diffs.size, size=(end - start, diffs.size))
        means[start:end] = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_randomization_pvalue(
    diffs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    max_elems: int = 8_000_000,
) -> float:
    if diffs.size == 0:
        return float("nan")
    observed = abs(float(diffs.mean()))
    if observed <= 1e-15:
        return 1.0
    chunk = max(1, min(n_perm, max_elems // max(1, diffs.size)))
    extreme = 0
    done = 0
    for start in range(0, n_perm, chunk):
        end = min(n_perm, start + chunk)
        signs = rng.integers(0, 2, size=(end - start, diffs.size), dtype=np.int8)
        signed = np.where(signs == 1, diffs, -diffs)
        perm_means = np.abs(signed.mean(axis=1))
        extreme += int(np.count_nonzero(perm_means >= observed - 1e-15))
        done += end - start
    return float((extreme + 1) / (done + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def sig_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def split_dir(root: Path, seed: str, cold_threshold: int) -> Path:
    return root / f"strict_item_cold_balanced_thr{cold_threshold}_seed_{seed}"


def ours_path(split: Path, scope: str) -> Path:
    return split / f"per_item_full_{scope}_usim_feedback_fast3_content_delta_static.csv"


def baseline_path(split: Path, scope: str, baseline: str) -> Path:
    names = {
        "ccfcrec": f"per_item_full_{scope}_ccfcrec_static.csv",
        "cgrc": f"per_item_full_{scope}_cgrc_paper_static.csv",
    }
    return split / "main_table_compare" / names[baseline]


def load_pair(seed: str, scope: str, ours_file: Path, base_file: Path, baseline: str) -> pd.DataFrame:
    if not ours_file.exists():
        raise FileNotFoundError(f"Missing ours per-item file: {ours_file}")
    if not base_file.exists():
        raise FileNotFoundError(f"Missing {baseline} per-item file: {base_file}")

    left = pd.read_csv(ours_file)
    right = pd.read_csv(base_file)
    keep_cols = ["item_id", "count", *METRICS]
    missing_left = sorted(set(keep_cols) - set(left.columns))
    missing_right = sorted(set(keep_cols) - set(right.columns))
    if missing_left or missing_right:
        raise ValueError(
            f"Missing columns seed={seed} scope={scope} baseline={baseline}: "
            f"ours={missing_left} baseline={missing_right}"
        )

    left = left[keep_cols].copy()
    right = right[keep_cols].copy()
    left["item_id"] = left["item_id"].astype(int)
    right["item_id"] = right["item_id"].astype(int)

    left_ids = set(left["item_id"])
    right_ids = set(right["item_id"])
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)[:10]
        only_right = sorted(right_ids - left_ids)[:10]
        raise ValueError(
            f"Unpaired item ids seed={seed} scope={scope} baseline={baseline}: "
            f"ours_only={only_left} baseline_only={only_right}"
        )

    merged = left.merge(right, on="item_id", suffixes=("_ours", f"_{baseline}"), how="inner")
    if not np.array_equal(merged["count_ours"].to_numpy(), merged[f"count_{baseline}"].to_numpy()):
        mismatch = merged.loc[merged["count_ours"] != merged[f"count_{baseline}"], "item_id"].head(10).tolist()
        raise ValueError(
            f"Count mismatch seed={seed} scope={scope} baseline={baseline}: {mismatch}"
        )

    merged.insert(0, "scope", scope)
    merged.insert(0, "baseline", baseline)
    merged.insert(0, "seed", seed)
    merged["count"] = merged["count_ours"]
    for metric in METRICS:
        merged[f"{metric}_diff"] = merged[f"{metric}_ours"] - merged[f"{metric}_{baseline}"]
    return merged


def summarize_frame(
    frame: pd.DataFrame,
    label: str,
    baseline: str,
    scope: str,
    bootstrap: int,
    randomization: int,
    rng: np.random.Generator,
    tie_tol: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        diffs = frame[f"{metric}_diff"].to_numpy(dtype=np.float64)
        ci_low, ci_high = paired_bootstrap_ci(diffs, bootstrap, rng)
        p_value = sign_randomization_pvalue(diffs, randomization, rng)
        rows.append(
            {
                "comparison": f"ours_vs_{baseline}",
                "scope": scope,
                "analysis_unit": label,
                "metric": metric,
                "paired_units": int(diffs.size),
                "ours_mean": float(frame[f"{metric}_ours"].mean()),
                "baseline_mean": float(frame[f"{metric}_{baseline}"].mean()),
                "mean_diff": float(diffs.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "randomization_p": p_value,
                "wins": int(np.count_nonzero(diffs > tie_tol)),
                "ties": int(np.count_nonzero(np.abs(diffs) <= tie_tol)),
                "losses": int(np.count_nonzero(diffs < -tie_tol)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/coco/single_seed_triage/ours_full")
    parser.add_argument("--out-dir", default="outputs/coco/single_seed_triage/significance")
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--baselines", nargs="+", default=["ccfcrec", "cgrc"])
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=SCOPES)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--randomization", type=int, default=50000)
    parser.add_argument("--rng-seed", type=int, default=20260616)
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--include-seed-summaries", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_frames = []
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.rng_seed)

    for baseline in args.baselines:
        if baseline not in {"ccfcrec", "cgrc"}:
            raise ValueError(f"Unsupported baseline: {baseline}")
        for scope in args.scopes:
            scope_frames = []
            for seed in args.seeds:
                split = split_dir(root, str(seed), args.cold_threshold)
                frame = load_pair(
                    str(seed),
                    scope,
                    ours_path(split, scope),
                    baseline_path(split, scope, baseline),
                    baseline,
                )
                detail_frames.append(frame)
                scope_frames.append(frame)
                if args.include_seed_summaries:
                    summary_rows.extend(
                        summarize_frame(
                            frame,
                            f"seed_{seed}",
                            baseline,
                            scope,
                            args.bootstrap,
                            args.randomization,
                            rng,
                            args.tie_tol,
                        )
                    )

            pooled = pd.concat(scope_frames, ignore_index=True)
            summary_rows.extend(
                summarize_frame(
                    pooled,
                    "pooled_seed_item",
                    baseline,
                    scope,
                    args.bootstrap,
                    args.randomization,
                    rng,
                    args.tie_tol,
                )
            )

    summary = pd.DataFrame(summary_rows)
    summary["randomization_p_holm_by_comparison_scope_unit"] = np.nan
    for _, idx in summary.groupby(["comparison", "scope", "analysis_unit"]).groups.items():
        idx_list = list(idx)
        adjusted = holm_adjust(summary.loc[idx_list, "randomization_p"].astype(float).tolist())
        summary.loc[idx_list, "randomization_p_holm_by_comparison_scope_unit"] = adjusted
    summary["sig"] = summary["randomization_p"].map(sig_marker)
    summary["sig_holm"] = summary["randomization_p_holm_by_comparison_scope_unit"].map(sig_marker)

    detail = pd.concat(detail_frames, ignore_index=True)
    detail_path = out_dir / "coco_paired_seed_item_detail.csv"
    summary_path = out_dir / "coco_paired_seed_item_summary.csv"
    pooled_path = out_dir / "coco_paired_seed_item_pooled_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pooled = summary[summary["analysis_unit"] == "pooled_seed_item"].copy()
    pooled.to_csv(pooled_path, index=False)

    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved pooled summary: {pooled_path}")
    cols = [
        "comparison",
        "scope",
        "metric",
        "paired_units",
        "ours_mean",
        "baseline_mean",
        "mean_diff",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "randomization_p",
        "sig",
        "sig_holm",
        "wins",
        "ties",
        "losses",
    ]
    print(pooled[cols].to_string(index=False))


if __name__ == "__main__":
    main()
