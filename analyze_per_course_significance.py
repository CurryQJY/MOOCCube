"""Per-course paired significance tests for strict item-cold evaluation.

The script expects per-item item-macro exports produced by the final evaluators:

  - CKG-RL: per_item_full_cold_usim_feedback_fast3_content_delta_static.csv
  - CGRC:   per_item_full_cold_cgrc_paper_static.csv

It reports paired bootstrap confidence intervals, sign-randomization p-values,
and win/tie/loss counts over matched cold courses.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def seed_from_path(path: str) -> str:
    match = re.search(r"seed[_-](\d+)", path)
    if not match:
        raise ValueError(f"Cannot infer seed from path: {path}")
    return match.group(1)


def collect_by_seed(root: str, pattern: str) -> dict[str, str]:
    files = glob.glob(os.path.join(root, pattern), recursive=True)
    by_seed: dict[str, str] = {}
    for path in files:
        seed = seed_from_path(path)
        if seed in by_seed:
            raise ValueError(f"Multiple files for seed {seed}: {by_seed[seed]} and {path}")
        by_seed[seed] = path
    return by_seed


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if diffs.size < 1:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 4096):
        end = min(n_boot, start + 4096)
        sample = rng.choice(diffs, size=(end - start, diffs.size), replace=True)
        means[start:end] = sample.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def randomization_pvalue(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    if diffs.size < 1:
        return float("nan")
    observed = abs(float(diffs.mean()))
    extreme = 0
    done = 0
    for start in range(0, n_perm, 4096):
        end = min(n_perm, start + 4096)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(end - start, diffs.size), replace=True)
        perm_means = np.abs((signs * diffs).mean(axis=1))
        extreme += int(np.sum(perm_means >= observed - 1e-15))
        done += end - start
    return float((extreme + 1) / (done + 1))


def summarize_scope(scope: str, merged: pd.DataFrame, metrics: list[str], args) -> list[dict]:
    rows = []
    rng = np.random.default_rng(args.seed)
    for metric in metrics:
        diff_col = f"{metric}_diff"
        diffs = merged[diff_col].to_numpy(dtype=np.float64)
        ci_low, ci_high = paired_bootstrap_ci(diffs, args.bootstrap, rng)
        p_value = randomization_pvalue(diffs, args.randomization, rng)
        wins = int((diffs > args.tie_tol).sum())
        ties = int((np.abs(diffs) <= args.tie_tol).sum())
        losses = int((diffs < -args.tie_tol).sum())
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "paired_units": int(diffs.size),
                "mean_diff": float(diffs.mean()) if diffs.size else float("nan"),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "randomization_p": p_value,
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-root", default="outputs/content_delta_pop5/course_ablation_e60_3seed/full")
    parser.add_argument(
        "--baseline-root",
        default="outputs/content_delta_pop5/static_item_cold_balanced",
    )
    parser.add_argument(
        "--ours-pattern",
        default="**/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
    )
    parser.add_argument(
        "--baseline-pattern",
        default="**/main_table_balanced_itemmacro_cgrc_paper_v1/per_item_full_cold_cgrc_paper_static.csv",
    )
    parser.add_argument("--metrics", nargs="+", default=["R@10", "N@10"])
    parser.add_argument("--out-dir", default="outputs/content_delta_pop5/per_course_significance")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--randomization", type=int, default=200000)
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    ours = collect_by_seed(args.ours_root, args.ours_pattern)
    baseline = collect_by_seed(args.baseline_root, args.baseline_pattern)
    seeds = sorted(set(ours) & set(baseline))
    if not seeds:
        raise SystemExit(
            "No matched per-item files found. Re-run CKG-RL and CGRC after the per-item export patch."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_frames = []
    summary_rows = []
    for seed in seeds:
        left = pd.read_csv(ours[seed])
        right = pd.read_csv(baseline[seed])
        keep_cols = ["item_id", "count"] + args.metrics
        missing_left = set(keep_cols) - set(left.columns)
        missing_right = set(keep_cols) - set(right.columns)
        if missing_left or missing_right:
            raise ValueError(
                f"Missing columns for seed {seed}: ours={sorted(missing_left)}, baseline={sorted(missing_right)}"
            )
        merged = left[keep_cols].merge(
            right[keep_cols],
            on="item_id",
            suffixes=("_ours", "_baseline"),
            how="inner",
        )
        for metric in args.metrics:
            merged[f"{metric}_diff"] = merged[f"{metric}_ours"] - merged[f"{metric}_baseline"]
        merged.insert(0, "seed", seed)
        merged_frames.append(merged)
        summary_rows.extend(summarize_scope(f"seed_{seed}", merged, args.metrics, args))

    pooled = pd.concat(merged_frames, ignore_index=True)
    summary_rows.extend(summarize_scope("pooled_seed_course", pooled, args.metrics, args))

    pooled.to_csv(out_dir / "per_course_ours_vs_cgrc_detail.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "per_course_ours_vs_cgrc_summary.csv", index=False)
    print(f"Saved {out_dir / 'per_course_ours_vs_cgrc_summary.csv'}")
    print(f"Saved {out_dir / 'per_course_ours_vs_cgrc_detail.csv'}")


if __name__ == "__main__":
    main()
