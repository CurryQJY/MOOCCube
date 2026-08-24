"""Per-user paired significance tests for cold-item recommendation.

This script loads saved per-user metric arrays from each seed/variant,
then runs paired t-test and Wilcoxon signed-rank test.

Usage:
  1. First run `export_peruser_metrics.py` to dump per-user scores.
  2. Then run this script to compute p-values.

Alternatively, if you already have trained checkpoints, this script can
evaluate on-the-fly (see --live mode).

Output: A CSV table of p-values suitable for the paper appendix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd


METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]

SEEDS = [2025, 2026, 2027]

# Directories where per-user metric .npz files are stored
# Expected filename pattern: peruser_metrics_seed{seed}.npz
# Each .npz contains arrays keyed by metric name, shape=(n_test_users,)
DEFAULT_OURS_DIR = Path(
    "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/full"
)

COMPARISONS = {
    # name -> directory
    "CGRC": Path(
        "outputs/content_delta_pop5/static_item_cold_balanced/"
        "main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/"
        "CGRC-paper"
    ),
    "w/o Course-aware Reward": Path(
        "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/wo_course_reward"
    ),
    "w/o Course-aware User Selection": Path(
        "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/wo_course_candidate"
    ),
    "w/o Prereq. Auxiliary Loss": Path(
        "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/wo_prereq_aux"
    ),
    "w/o All Course Signals": Path(
        "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/wo_all_course_signals"
    ),
}


def load_peruser(directory: Path, seed: int) -> dict[str, np.ndarray] | None:
    """Load per-user metric arrays for a given seed."""
    path = directory / f"peruser_metrics_seed{seed}.npz"
    if not path.exists():
        # Try alternative naming
        path = directory / f"peruser_cold_seed{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test. Returns (t-statistic, p-value)."""
    diff = a - b
    if np.std(diff) < 1e-12:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    diff = a - b
    # Remove zeros (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return 0.0, 1.0
    try:
        stat, p_val = stats.wilcoxon(nonzero)
        return float(stat), float(p_val)
    except ValueError:
        return 0.0, 1.0


def run_tests(ours_dir: Path, comparisons: dict[str, Path]) -> pd.DataFrame:
    """Run significance tests across all seeds and comparisons."""
    rows = []

    for comp_name, comp_dir in comparisons.items():
        for metric in METRICS:
            t_pvals = []
            w_pvals = []
            mean_diffs = []

            for seed in SEEDS:
                ours_data = load_peruser(ours_dir, seed)
                comp_data = load_peruser(comp_dir, seed)

                if ours_data is None or comp_data is None:
                    continue
                if metric not in ours_data or metric not in comp_data:
                    continue

                a = ours_data[metric]
                b = comp_data[metric]

                if len(a) != len(b):
                    print(f"WARNING: length mismatch for {comp_name}/{metric}/seed{seed}: "
                          f"{len(a)} vs {len(b)}")
                    min_len = min(len(a), len(b))
                    a, b = a[:min_len], b[:min_len]

                _, t_p = paired_ttest(a, b)
                _, w_p = wilcoxon_test(a, b)
                t_pvals.append(t_p)
                w_pvals.append(w_p)
                mean_diffs.append(float(a.mean() - b.mean()))

            if not t_pvals:
                rows.append({
                    "Comparison": comp_name,
                    "Metric": metric,
                    "Mean Diff": None,
                    "Paired t p-value (avg)": None,
                    "Wilcoxon p-value (avg)": None,
                    "Paired t p-value (max)": None,
                    "Significant (p<0.05)": "N/A (no data)",
                })
                continue

            rows.append({
                "Comparison": comp_name,
                "Metric": metric,
                "Mean Diff": f"{np.mean(mean_diffs):.4f}",
                "Paired t p-value (avg)": f"{np.mean(t_pvals):.4f}",
                "Wilcoxon p-value (avg)": f"{np.mean(w_pvals):.4f}",
                "Paired t p-value (max)": f"{np.max(t_pvals):.4f}",
                "Significant (p<0.05)": "Yes" if max(t_pvals) < 0.05 else "No",
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Significance tests for cold-item recommendation")
    parser.add_argument("--ours-dir", type=str, default=str(DEFAULT_OURS_DIR),
                        help="Directory containing Ours per-user metric files")
    parser.add_argument("--output", type=str, default="output/doc/significance_tests.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    ours_dir = Path(args.ours_dir)

    # Check if per-user files exist
    has_files = any((ours_dir / f"peruser_metrics_seed{s}.npz").exists() for s in SEEDS)
    if not has_files:
        has_files = any((ours_dir / f"peruser_cold_seed{s}.npz").exists() for s in SEEDS)

    if not has_files:
        print("=" * 70)
        print("ERROR: No per-user metric files found!")
        print(f"  Searched in: {ours_dir}")
        print()
        print("You need to first export per-user metrics by modifying the")
        print("evaluation loop. See export_peruser_metrics.py for instructions.")
        print("=" * 70)
        return

    results = run_tests(ours_dir, COMPARISONS)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved significance test results to: {out_path}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
