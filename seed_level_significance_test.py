"""Seed-level paired significance tests using existing per-seed detail CSVs.

Two modes:
  1. Seed-level (df=2): Quick check using 3 seed aggregate values.
     Low power but requires no re-running.
  2. (Future) Per-user: After exporting per-user .npz files.

Usage:
    python seed_level_significance_test.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────

ABLATION_ROOT = Path("outputs/content_delta_pop5/course_ablation_e60_3seed")

# Ours (full model) detail CSV
OURS_DETAIL = ABLATION_ROOT / "full" / "fast3_static_runs_detail.csv"

# Ablation variants
ABLATION_DETAILS = {
    "w/o Course-aware Reward":      ABLATION_ROOT / "wo_course_reward"      / "fast3_static_runs_detail.csv",
    "w/o Course-aware User Selection": ABLATION_ROOT / "wo_course_candidate" / "fast3_static_runs_detail.csv",
    "w/o Prereq. Auxiliary Loss":   ABLATION_ROOT / "wo_prereq_aux"         / "fast3_static_runs_detail.csv",
    "w/o All Course Signals":       ABLATION_ROOT / "wo_all_course_signals" / "fast3_static_runs_detail.csv",
}

# Baseline detail CSV (CGRC)
BASELINE_DETAIL = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_balanced_itemmacro_cgrc_paper_v1/main_table_item_macro_detail.csv"
)

# Metrics to test (column names in the detail CSVs)
# Ours uses: full_cold_item_macro_{r5,r10,r20,n5,n10,n20}
OURS_METRIC_COLS = {
    "R@5":  "full_cold_item_macro_r5",
    "R@10": "full_cold_item_macro_r10",
    "R@20": "full_cold_item_macro_r20",
    "N@5":  "full_cold_item_macro_n5",
    "N@10": "full_cold_item_macro_n10",
    "N@20": "full_cold_item_macro_n20",
}

SEEDS = [2025, 2026, 2027]


def load_seed_values(detail_csv: Path, metric_cols: dict[str, str],
                     seeds: list[int] = SEEDS,
                     seed_col: str = "seed",
                     model_filter: str | None = None) -> dict[str, np.ndarray]:
    """Load per-seed metric values from a detail CSV.
    
    Returns: dict mapping metric_name -> np.array of shape (n_seeds,)
    """
    if not detail_csv.exists():
        print(f"  WARNING: {detail_csv} not found")
        return {}

    df = pd.read_csv(detail_csv)

    # Try to identify the seed column
    if seed_col not in df.columns:
        # Check if seeds are in a 'run_tag' column
        if "run_tag" in df.columns:
            # Extract seed from run_tag like "strict_item_cold_balanced_thr1_seed_2025"
            df["seed"] = df["run_tag"].str.extract(r"seed_(\d+)").astype(float)
        else:
            print(f"  WARNING: No seed column found in {detail_csv}")
            print(f"  Columns: {list(df.columns)}")
            return {}

    if model_filter:
        df = df[df["model"] == model_filter]

    result = {}
    for metric_name, col_name in metric_cols.items():
        if col_name not in df.columns:
            # Try alternative naming: cold_{R5} -> cold_R5_mean etc.
            alt = col_name.replace("full_cold_item_macro_", "cold_").upper()
            alt_col = f"{alt}"
            if alt_col in df.columns:
                col_name = alt_col
            else:
                print(f"  WARNING: column {col_name} not in {detail_csv}")
                continue

        vals = []
        for seed in seeds:
            row = df[df["seed"] == seed]
            if row.empty:
                print(f"  WARNING: seed {seed} not found in {detail_csv}")
                vals.append(np.nan)
            else:
                vals.append(float(row.iloc[0][col_name]))
        result[metric_name] = np.array(vals)

    return result


def load_baseline_seed_values(detail_csv: Path, model_name: str,
                              seeds: list[int] = SEEDS) -> dict[str, np.ndarray]:
    """Load baseline per-seed values. Baseline CSVs may have different column naming."""
    if not detail_csv.exists():
        print(f"  WARNING: {detail_csv} not found")
        return {}

    df = pd.read_csv(detail_csv)
    print(f"  Baseline columns sample: {list(df.columns)[:10]}...")
    print(f"  Baseline models: {df['model'].unique() if 'model' in df.columns else 'N/A'}")

    if "model" in df.columns:
        df = df[df["model"] == model_name]

    # Try to find seed column
    seed_col = None
    for candidate in ["seed", "random_seed", "run_seed"]:
        if candidate in df.columns:
            seed_col = candidate
            break

    if seed_col is None and "run_tag" in df.columns:
        df["seed"] = df["run_tag"].str.extract(r"seed_?(\d+)").astype(float)
        seed_col = "seed"
    elif seed_col is None and "notes" in df.columns:
        df["seed"] = df["notes"].str.extract(r"seed_?(\d+)").astype(float)
        seed_col = "seed"

    if seed_col is None:
        print(f"  WARNING: cannot find seed column in baseline CSV")
        return {}

    # Try to map metric columns
    result = {}
    for metric_name, ours_col in OURS_METRIC_COLS.items():
        # Baseline might use: cold_R5_mean, cold_r5, full_cold_item_macro_r5, etc.
        candidates = [
            ours_col,
            ours_col.replace("full_cold_item_macro_", "cold_"),
            ours_col.replace("full_cold_item_macro_", "cold_").upper(),
        ]
        col_name = None
        for c in candidates:
            if c in df.columns:
                col_name = c
                break

        # Also try with different formats
        if col_name is None:
            # Check all columns for partial match
            metric_suffix = ours_col.split("_")[-1]  # e.g., "r5"
            for c in df.columns:
                if metric_suffix in c.lower() and "cold" in c.lower() and "item_macro" in c.lower():
                    col_name = c
                    break

        if col_name is None:
            print(f"  WARNING: cannot find {metric_name} column for baseline")
            continue

        vals = []
        for seed in seeds:
            row = df[df[seed_col] == seed]
            if row.empty:
                vals.append(np.nan)
            else:
                vals.append(float(row.iloc[0][col_name]))
        result[metric_name] = np.array(vals)

    return result


def paired_ttest_seed(a: np.ndarray, b: np.ndarray):
    """Paired t-test on seed-level values (df=n_seeds-1)."""
    diff = a - b
    n = len(diff)
    if n < 2 or np.std(diff, ddof=1) < 1e-15:
        return 0.0, 1.0, 0.0
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val), float(np.mean(diff))


def main():
    print("=" * 80)
    print("SEED-LEVEL PAIRED SIGNIFICANCE TESTS")
    print(f"Seeds: {SEEDS}  (df={len(SEEDS)-1})")
    print("=" * 80)
    print()

    # Load Ours
    print("Loading Ours (full) per-seed values...")
    ours = load_seed_values(OURS_DETAIL, OURS_METRIC_COLS)
    if not ours:
        print("ERROR: Cannot load Ours values. Exiting.")
        return

    print("  Per-seed values:")
    for m, v in ours.items():
        print(f"    {m}: {v}")
    print()

    # ── Ablation comparisons ──
    print("─" * 80)
    print("ABLATION COMPARISONS (Ours vs each variant)")
    print("─" * 80)

    ablation_rows = []
    for variant_name, detail_path in ABLATION_DETAILS.items():
        print(f"\n  Loading {variant_name}...")
        comp = load_seed_values(detail_path, OURS_METRIC_COLS)
        if not comp:
            continue
        for m, v in comp.items():
            print(f"    {m}: {v}")

        for metric_name in OURS_METRIC_COLS:
            if metric_name not in ours or metric_name not in comp:
                continue
            t_stat, p_val, mean_diff = paired_ttest_seed(ours[metric_name], comp[metric_name])
            ablation_rows.append({
                "Comparison": variant_name,
                "Metric": metric_name,
                "Ours mean": f"{ours[metric_name].mean():.4f}",
                "Variant mean": f"{comp[metric_name].mean():.4f}",
                "Diff": f"{mean_diff:+.4f}",
                "t-stat": f"{t_stat:.3f}",
                "p-value": f"{p_val:.4f}",
                "Sig.(p<0.05)": "Yes*" if p_val < 0.05 else "No",
                "Sig.(p<0.10)": "Yes†" if p_val < 0.10 else "No",
            })

    if ablation_rows:
        print("\n")
        abl_df = pd.DataFrame(ablation_rows)
        print(abl_df.to_string(index=False))

    # ── Baseline comparison (CGRC) ──
    print("\n")
    print("─" * 80)
    print("BASELINE COMPARISON (Ours vs CGRC)")
    print("─" * 80)

    baseline_rows = []
    if BASELINE_DETAIL.exists():
        print(f"\n  Loading CGRC baseline from {BASELINE_DETAIL}...")
        cgrc = load_baseline_seed_values(BASELINE_DETAIL, "CGRC-paper")
        if cgrc:
            for m, v in cgrc.items():
                print(f"    {m}: {v}")

            for metric_name in OURS_METRIC_COLS:
                if metric_name not in ours or metric_name not in cgrc:
                    continue
                t_stat, p_val, mean_diff = paired_ttest_seed(ours[metric_name], cgrc[metric_name])
                baseline_rows.append({
                    "Comparison": "vs CGRC",
                    "Metric": metric_name,
                    "Ours mean": f"{ours[metric_name].mean():.4f}",
                    "CGRC mean": f"{cgrc[metric_name].mean():.4f}",
                    "Diff": f"{mean_diff:+.4f}",
                    "t-stat": f"{t_stat:.3f}",
                    "p-value": f"{p_val:.4f}",
                    "Sig.(p<0.05)": "Yes*" if p_val < 0.05 else "No",
                    "Sig.(p<0.10)": "Yes†" if p_val < 0.10 else "No",
                })
    else:
        print(f"  CGRC baseline detail CSV not found: {BASELINE_DETAIL}")
        print("  Trying alternative path...")
        alt = Path(
            "outputs/content_delta_pop5/static_item_cold_balanced/"
            "main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/"
            "main_table_item_macro_detail.csv"
        )
        if alt.exists():
            print(f"  Found: {alt}")
            cgrc = load_baseline_seed_values(alt, "CGRC-paper")
            if cgrc:
                for metric_name in OURS_METRIC_COLS:
                    if metric_name not in ours or metric_name not in cgrc:
                        continue
                    t_stat, p_val, mean_diff = paired_ttest_seed(ours[metric_name], cgrc[metric_name])
                    baseline_rows.append({
                        "Comparison": "vs CGRC",
                        "Metric": metric_name,
                        "Ours mean": f"{ours[metric_name].mean():.4f}",
                        "CGRC mean": f"{cgrc[metric_name].mean():.4f}",
                        "Diff": f"{mean_diff:+.4f}",
                        "t-stat": f"{t_stat:.3f}",
                        "p-value": f"{p_val:.4f}",
                        "Sig.(p<0.05)": "Yes*" if p_val < 0.05 else "No",
                        "Sig.(p<0.10)": "Yes†" if p_val < 0.10 else "No",
                    })

    if baseline_rows:
        print("\n")
        base_df = pd.DataFrame(baseline_rows)
        print(base_df.to_string(index=False))

    # ── Save combined results ──
    all_rows = ablation_rows + baseline_rows
    if all_rows:
        out_path = Path("output/doc/final_narrow_topconf/seed_level_significance.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")

    # ── Statistical power warning ──
    print("\n")
    print("=" * 80)
    print("⚠ NOTE: With only 3 seeds (df=2), the paired t-test has very low")
    print("  statistical power. A p-value > 0.05 does NOT mean the difference")
    print("  is not real — it means the test lacks power to detect it.")
    print()
    print("  For publication-strength evidence, run the per-user significance")
    print("  test (68 cold test users per seed → much higher power).")
    print("  See: export_peruser_metrics.py + run_significance_test.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
