"""Quick CGRC baseline comparison."""
import pandas as pd
import numpy as np
from scipy import stats

# Ours per-seed
ours_df = pd.read_csv(
    "outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_runs_detail.csv"
)
ours_df = ours_df.sort_values("seed").reset_index(drop=True)

# CGRC per-seed
cgrc_df = pd.read_csv(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_balanced_itemmacro_cgrc_paper_v1/main_table_item_macro_detail.csv"
)
cgrc_df = cgrc_df[cgrc_df["model"] == "CGRC-paper"].sort_values("seed").reset_index(drop=True)

metrics = [
    ("R@5",  "full_cold_item_macro_r5",  "cold_R5"),
    ("R@10", "full_cold_item_macro_r10", "cold_R10"),
    ("R@20", "full_cold_item_macro_r20", "cold_R20"),
    ("N@5",  "full_cold_item_macro_n5",  "cold_N5"),
    ("N@10", "full_cold_item_macro_n10", "cold_N10"),
    ("N@20", "full_cold_item_macro_n20", "cold_N20"),
]

print("=" * 90)
print("SEED-LEVEL PAIRED T-TEST: Ours vs CGRC (df=2)")
print("=" * 90)

rows = []
for name, ours_col, cgrc_col in metrics:
    a = ours_df[ours_col].values
    b = cgrc_df[cgrc_col].values
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = np.mean(a) - np.mean(b)
    sig = "p<0.05*" if p_val < 0.05 else ("p<0.10+" if p_val < 0.10 else "n.s.")
    rows.append({
        "Metric": name,
        "Ours (seeds)": f"[{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}]",
        "CGRC (seeds)": f"[{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}]",
        "Mean Diff": f"{diff:+.4f}",
        "t-stat": f"{t_stat:.3f}",
        "p-value": f"{p_val:.4f}",
        "Sig": sig,
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

print("\n")
print("=" * 90)
print("SEED-LEVEL PAIRED T-TEST: Ablation variants (df=2)")
print("=" * 90)

ablation_variants = {
    "w/o Course-aware Reward": "outputs/content_delta_pop5/course_ablation_e60_3seed/wo_course_reward/fast3_static_runs_detail.csv",
    "w/o Course-aware User Sel.": "outputs/content_delta_pop5/course_ablation_e60_3seed/wo_course_candidate/fast3_static_runs_detail.csv",
    "w/o Prereq. Auxiliary Loss": "outputs/content_delta_pop5/course_ablation_e60_3seed/wo_prereq_aux/fast3_static_runs_detail.csv",
    "w/o All Course Signals": "outputs/content_delta_pop5/course_ablation_e60_3seed/wo_all_course_signals/fast3_static_runs_detail.csv",
}

abl_rows = []
for variant_name, path in ablation_variants.items():
    vdf = pd.read_csv(path).sort_values("seed").reset_index(drop=True)
    for name, ours_col, _ in metrics:
        a = ours_df[ours_col].values
        b = vdf[ours_col].values
        t_stat, p_val = stats.ttest_rel(a, b)
        diff = np.mean(a) - np.mean(b)
        sig = "p<0.05*" if p_val < 0.05 else ("p<0.10+" if p_val < 0.10 else "n.s.")
        abl_rows.append({
            "Variant": variant_name,
            "Metric": name,
            "Diff (Ours-Var)": f"{diff:+.4f}",
            "t-stat": f"{t_stat:.3f}",
            "p-value": f"{p_val:.4f}",
            "Sig": sig,
        })

abl_df = pd.DataFrame(abl_rows)
print(abl_df.to_string(index=False))

# Count significant results
n_sig_05 = sum(1 for r in abl_rows if "0.05" in r["Sig"])
n_sig_10 = sum(1 for r in abl_rows if "0.05" in r["Sig"] or "0.10" in r["Sig"])
n_total = len(abl_rows)
print(f"\nSummary: {n_sig_05}/{n_total} significant at p<0.05, {n_sig_10}/{n_total} at p<0.10")
print("\nNOTE: df=2 has very low power. Per-user test (n=68) recommended for publication.")
