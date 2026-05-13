"""
aggregate_pam_results.py — 聚合 pam_cube/pam_full_ranking_results.csv
为统一格式的 pam_full_result.json, 与 lightgcn_full_result.json 等基线对齐.

注意: train_pam.py 仅评估 cold 用户的 full ranking, 没有 hot / sampled 协议.
本脚本将:
  - 按 Count 加权平均 cold full ranking 指标 (跳过 Period < WARMUP 或 Count == 0)
  - 把结果同时写入 sample_cold 和 full_cold 字段 (sampled 字段标注为 None / 缺失)
  - hot 字段填 0 并在 notes 中明确说明
"""

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd


def aggregate(csv_path: str, warmup: int = 3) -> Dict[str, float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing PAM csv: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"R@5", "R@10", "R@20", "N@5", "N@10", "N@20", "Period", "Count"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"PAM csv missing columns: {sorted(miss)}")

    counts = df["Count"].to_numpy(np.int64)
    periods = df["Period"].to_numpy(np.int64)
    valid_mask = (periods >= warmup) & (counts > 0)
    if valid_mask.sum() < 1:
        raise RuntimeError(f"No valid PAM periods (warmup={warmup})")

    valid_counts = counts[valid_mask]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in (5, 10, 20)]
    weighted = {}
    for k in metrics_keys:
        vals = df[k].to_numpy(np.float64)[valid_mask]
        weighted[k] = float(np.average(vals, weights=valid_counts))

    return {
        "metrics": weighted,
        "total_count": int(valid_counts.sum()),
        "valid_periods": int(valid_mask.sum()),
        "warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join("pam_cube", "pam_full_ranking_results.csv"),
        help="Path to PAM raw csv",
    )
    parser.add_argument(
        "--out",
        default="pam_full_result.json",
        help="Output json path (aligned with other *_full_result.json baselines)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    agg = aggregate(args.csv, warmup=args.warmup)
    metrics = agg["metrics"]
    total = agg["total_count"]

    # 与其他基线对齐的字段; PAM 缺少 sampled 与 hot, 用 0 填充并在 notes 中说明
    zero_metrics = {k: 0.0 for k in metrics.keys()}
    out = {
        "sample_cold": dict(metrics),  # 复用 cold full ranking 作为占位
        "sample_hot": zero_metrics,
        "full_cold": dict(metrics),
        "full_hot": zero_metrics,
        "count_sample_cold": total,
        "count_sample_hot": 0,
        "count_full_cold": total,
        "count_full_hot": 0,
        "periods": agg["valid_periods"],
        "warmup_periods": agg["warmup"],
        "use_cumulative": True,
        "notes": (
            "PAM raw csv only contains cold full-ranking metrics. "
            "sample_cold is filled with the same cold full values for compatibility; "
            "sample_hot / full_hot are unavailable and reported as 0."
        ),
    }
    pd.DataFrame([out]).to_json(args.out, orient="records", force_ascii=False)
    print(f"Aggregated PAM cold full-ranking metrics (count={total}, periods={agg['valid_periods']}):")
    for k, v in metrics.items():
        print(f"  {k:<6} = {v:.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
