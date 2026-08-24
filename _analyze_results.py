"""Comprehensive analysis of USIM vs baselines on MOOCCubeX."""
import json, csv
import pandas as pd
import numpy as np
from pathlib import Path

RES = Path("results_mooccubex")

# ── 1. Load USIM final report ──
usim_report = pd.read_csv(RES / "final_report_usim_feedback_fast3_content_delta.csv")
usim_periods = pd.read_csv(RES / "mooc_metrics_usim_feedback_fast3_content_delta.csv")

# ── 2. Load Popularity Static ──
with open(RES / "popularity_static_result.json") as f:
    pop_static = json.load(f)
    if isinstance(pop_static, list):
        pop_static = pop_static[0]

# ── 3. Build comparison table ──
def fmt(v): return f"{v:.4f}" if v else "—"

metrics_list = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]

print("=" * 100)
print("  MOOCCubeX 实验结果分析")
print("=" * 100)

# ── Evaluation protocol note ──
print("\n⚠️  评价协议说明:")
print("  • USIM: Streaming evaluation — 每个 Period 在当期新数据上评估，加权平均")
print(f"    Cold users: 16,175 | Hot users: 5,245,307 (across all periods)")
print("  • Popularity Static: 静态 80/10/10 划分，在固定 test set 上评估")
print(f"    Cold users: {pop_static.get('count_full_cold', 'N/A')} | Hot users: {pop_static.get('count_full_hot', 'N/A')}")
print("  • BPR Static / LightGCN: OOM 未完成")

# ── Full ranking comparison ──
print("\n" + "=" * 100)
print("  Full Ranking Results (全库排序)")
print("=" * 100)

header = f"{'Method':<32}{'Group':<8}" + "".join(f"{m:<10}" for m in metrics_list)
print(header)
print("-" * 100)

for group, ukey in [("Cold", "full_cold"), ("Hot", "full_hot")]:
    # USIM
    usim_vals = {}
    for _, row in usim_report.iterrows():
        usim_vals[row["metric"]] = row[ukey]
    line = f"{'USIM (Ours)':<32}{group:<8}"
    for m in metrics_list:
        line += f"{usim_vals.get(m, 0):<10.4f}"
    print(line)

    # Popularity Static
    pop_vals = pop_static.get(ukey, {})
    line = f"{'Popularity Static':<32}{group:<8}"
    for m in metrics_list:
        line += f"{pop_vals.get(m, 0):<10.4f}"
    print(line)

    # Improvement
    line = f"{'  Δ (USIM vs Pop)':<32}{group:<8}"
    for m in metrics_list:
        u = usim_vals.get(m, 0)
        p = pop_vals.get(m, 0)
        if p > 0:
            pct = (u - p) / p * 100
            line += f"{pct:+.1f}%{'':<4}"
        else:
            line += f"{'N/A':<10}"
    print(line)
    print()

# ── Sampled (1+200) comparison ──
print("=" * 100)
print("  Sampled Ranking Results (1+200 负采样)")
print("=" * 100)
print(header)
print("-" * 100)

for group, csv_key, json_key in [("Cold", "sampled_cold", "sample_cold"), ("Hot", "sampled_hot", "sample_hot")]:
    usim_vals = {}
    for _, row in usim_report.iterrows():
        usim_vals[row["metric"]] = row[csv_key]
    line = f"{'USIM (Ours)':<32}{group:<8}"
    for m in metrics_list:
        line += f"{usim_vals.get(m, 0):<10.4f}"
    print(line)

    pop_vals = pop_static.get(json_key, {})
    line = f"{'Popularity Static':<32}{group:<8}"
    for m in metrics_list:
        line += f"{pop_vals.get(m, 0):<10.4f}"
    print(line)

    line = f"{'  Δ (USIM vs Pop)':<32}{group:<8}"
    for m in metrics_list:
        u = usim_vals.get(m, 0)
        p = pop_vals.get(m, 0)
        if p > 0:
            pct = (u - p) / p * 100
            line += f"{pct:+.1f}%{'':<4}"
        else:
            line += f"{'N/A':<10}"
    print(line)
    print()

# ── USIM per-period trend analysis ──
print("=" * 100)
print("  USIM Per-Period Trend (Full Ranking)")
print("=" * 100)
print(f"{'Period':<8}{'ColdUsers':<12}{'HotUsers':<12}{'Cold R@10':<12}{'Hot R@10':<12}{'Cold N@10':<12}{'Hot N@10':<12}")
print("-" * 80)

eval_periods = usim_periods[usim_periods["Full_count_cold"] > 0].copy()
for _, row in eval_periods.iterrows():
    p = int(row["Period"])
    print(f"{p:<8}{int(row['Full_count_cold']):<12}{int(row['Full_count_hot']):<12}"
          f"{row['full_cold_R@10']:<12.4f}{row['full_hot_R@10']:<12.4f}"
          f"{row['full_cold_N@10']:<12.4f}{row['full_hot_N@10']:<12.4f}")

# Stats
cold_r10 = eval_periods["full_cold_R@10"].values
hot_r10 = eval_periods["full_hot_R@10"].values
print(f"\n  Cold R@10 — min: {cold_r10.min():.4f}, max: {cold_r10.max():.4f}, "
      f"last5 avg: {cold_r10[-5:].mean():.4f}, overall avg: {cold_r10.mean():.4f}")
print(f"  Hot  R@10 — min: {hot_r10.min():.4f}, max: {hot_r10.max():.4f}, "
      f"last5 avg: {hot_r10[-5:].mean():.4f}, overall avg: {hot_r10.mean():.4f}")

# ── Key findings ──
print("\n" + "=" * 100)
print("  Key Findings")
print("=" * 100)

# Get R@10 values
u_fc = float(usim_report[usim_report["metric"] == "R@10"]["full_cold"].values[0])
p_fc = pop_static["full_cold"]["R@10"]
u_fh = float(usim_report[usim_report["metric"] == "R@10"]["full_hot"].values[0])
p_fh = pop_static["full_hot"]["R@10"]
u_sc = float(usim_report[usim_report["metric"] == "R@10"]["sampled_cold"].values[0])
p_sc = pop_static["sample_cold"]["R@10"]
u_sh = float(usim_report[usim_report["metric"] == "R@10"]["sampled_hot"].values[0])
p_sh = pop_static["sample_hot"]["R@10"]

print(f"\n  1. Cold-Start (核心目标):")
print(f"     Full R@10:    USIM {u_fc:.4f} vs Pop {p_fc:.4f} → USIM +{(u_fc/p_fc-1)*100:.1f}%")
print(f"     Sampled R@10: USIM {u_sc:.4f} vs Pop {p_sc:.4f} → USIM +{(u_sc/p_sc-1)*100:.1f}%")

print(f"\n  2. Hot Users:")
print(f"     Full R@10:    USIM {u_fh:.4f} vs Pop {p_fh:.4f} → Pop +{(p_fh/u_fh-1)*100:.1f}%")
print(f"     Sampled R@10: USIM {u_sh:.4f} vs Pop {p_sh:.4f} → Pop +{(p_sh/u_sh-1)*100:.1f}%")

print(f"\n  3. USIM 性能趋势:")
print(f"     Cold R@10 后5期均值: {cold_r10[-5:].mean():.4f} (vs 前5期: {cold_r10[:5].mean():.4f})")
print(f"     → 随数据累积，Cold-Start 性能持续提升 {(cold_r10[-5:].mean()/cold_r10[:5].mean()-1)*100:.0f}%")

print(f"\n  4. 基线缺失:")
print(f"     BPR Static / BPR Full / LightGCN Static / LightGCN Full — OOM 未完成")
print(f"     Popularity Full — OOM at Period 8/20")
print(f"     只有 Popularity Static 完成")

print(f"\n  5. 规模统计:")
print(f"     总交互: 5,261,548 | 评估周期: P3-P19 (17个) | 累积训练")
