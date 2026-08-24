#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指标拼接上界 (Metric-Stitch Upper Bound)
==========================================

用途
----
在主表的 item-macro / full-ranking / test 评估口径下，构造一个"能力上界"：
    - Cold 桶指标  取自 CBI+模拟器模型 (USIM-Feedback-FAST3-ContentDelta)
    - Hot  桶指标  取自 Frozen Hot Graph Expert 的 test-only replay
    - Overall      按去重项数加权:
                   overall = (cold * n_cold + hot * n_hot) / (n_cold + n_hot)

这是一个**乐观上界**：每个桶都取自各自模型在自己满池 / 自己选择器下的最优结果，
不需要跨模型分数校准（因为主表 Overall 只是两个独立算出的标量的加权平均，
见 export_overall_baseline_comparison / evaluate_cbi_hybrid_refined 的加权口径）。
真正可部署的候选项级路由结果会低于此上界。

数据来源（均为已产出、SHA/来源可追溯的 test 结果）
    Cold: outputs/cbi_anchor_sim_3seed_serial/strict_item_cold_balanced_thr1_seed_<S>/
              per_item_full_cold_usim_feedback_fast3_content_delta_static.csv
    Hot : outputs/ckg_hot_graph_test_replay_3seed/seed<S>/per_item_test_hot.csv

逐种子拼接后跨 3 个种子 (2025/2026/2027) 汇总 mean±std。
本脚本只读取已有 per-item CSV，不训练、不重新评估，完全确定性、可复现。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

SEEDS = [2025, 2026, 2027]
METRIC_COLS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]

BASE = Path(__file__).resolve().parent

COLD_CSV_TMPL = (
    BASE
    / "outputs"
    / "cbi_anchor_sim_3seed_serial"
    / "strict_item_cold_balanced_thr1_seed_{seed}"
    / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
)
HOT_CSV_TMPL = (
    BASE
    / "outputs"
    / "ckg_hot_graph_test_replay_3seed"
    / "seed{seed}"
    / "per_item_test_hot.csv"
)


def read_per_item(path: Path) -> Tuple[Dict[str, float], int]:
    """读取 per-item CSV，返回 {metric: item-macro 均值} 和去重项数。

    item-macro = 对每个 item 一行的指标做简单算术平均（与主表口径一致）。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少 per-item 文件: {path}")
    sums = {m: 0.0 for m in METRIC_COLS}
    n = 0
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [m for m in METRIC_COLS if m not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} 缺少列 {missing}；实际列={reader.fieldnames}")
        for row in reader:
            for m in METRIC_COLS:
                sums[m] += float(row[m])
            n += 1
    if n == 0:
        raise ValueError(f"{path} 没有任何 item 行")
    macro = {m: sums[m] / n for m in METRIC_COLS}
    return macro, n


def stitch_seed(seed: int) -> Dict[str, object]:
    """对单个种子做拼接，返回 cold/hot/overall 各指标 + 项数。"""
    cold_macro, n_cold = read_per_item(Path(str(COLD_CSV_TMPL).format(seed=seed)))
    hot_macro, n_hot = read_per_item(Path(str(HOT_CSV_TMPL).format(seed=seed)))
    total = n_cold + n_hot
    overall = {
        m: (cold_macro[m] * n_cold + hot_macro[m] * n_hot) / total
        for m in METRIC_COLS
    }
    return {
        "seed": seed,
        "n_cold": n_cold,
        "n_hot": n_hot,
        "n_overall": total,
        "cold": cold_macro,
        "hot": hot_macro,
        "overall": overall,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)  # 样本标准差，与已有汇总口径一致
    return mean, math.sqrt(var)


def aggregate(per_seed: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """跨种子汇总 cold/hot/overall 各指标的 mean/std。"""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bucket in ("cold", "hot", "overall"):
        out[bucket] = {}
        for m in METRIC_COLS:
            vals = [float(s[bucket][m]) for s in per_seed]  # type: ignore[index]
            mean, std = mean_std(vals)
            out[bucket][m] = {"mean": mean, "std": std}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 CBI 冷 + Hot 专家热 的指标拼接上界")
    ap.add_argument(
        "--out-dir",
        default=str(BASE / "outputs" / "cbi_hot_stitch_upper_bound"),
        help="输出目录",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = [stitch_seed(s) for s in args.seeds]
    agg = aggregate(per_seed)

    # 逐种子明细 CSV
    detail_path = out_dir / "stitch_upper_bound_per_seed.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        header = ["seed", "n_cold", "n_hot", "n_overall"]
        for bucket in ("cold", "hot", "overall"):
            header += [f"{bucket}_{m}" for m in METRIC_COLS]
        w.writerow(header)
        for s in per_seed:
            row = [s["seed"], s["n_cold"], s["n_hot"], s["n_overall"]]
            for bucket in ("cold", "hot", "overall"):
                row += [f"{s[bucket][m]:.6f}" for m in METRIC_COLS]  # type: ignore[index]
            w.writerow(row)

    # 汇总 CSV
    summary_path = out_dir / "stitch_upper_bound_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "metric", "mean", "std"])
        for bucket in ("cold", "hot", "overall"):
            for m in METRIC_COLS:
                cell = agg[bucket][m]
                w.writerow([bucket, m, f"{cell['mean']:.6f}", f"{cell['std']:.6f}"])

    # 溯源 JSON
    prov = {
        "description": "Metric-stitch optimistic upper bound: CBI cold bucket + Hot Graph Expert hot bucket, item-macro full-ranking test, count-weighted overall.",
        "is_optimistic_upper_bound": True,
        "note": "Deployable candidate-level routing will score below this bound; needs score calibration. This artifact requires none because Overall is a weighted average of two independently computed scalars.",
        "seeds": list(args.seeds),
        "cold_source_template": str(COLD_CSV_TMPL),
        "hot_source_template": str(HOT_CSV_TMPL),
        "cold_model": "USIM-Feedback-FAST3-ContentDelta (CBI+simulator)",
        "hot_model": "Frozen Hot Graph Expert (test-only replay)",
        "overall_formula": "(cold*n_cold + hot*n_hot)/(n_cold+n_hot)",
        "per_seed": per_seed,
        "aggregate": agg,
    }
    (out_dir / "stitch_upper_bound_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 控制台打印关键行
    print("=== 指标拼接上界 (mean±std over seeds {}) ===".format(args.seeds))
    print(f"{'bucket':8s} {'R@10':>16s} {'N@10':>16s} {'R@20':>16s} {'N@20':>16s}")
    for bucket in ("cold", "hot", "overall"):
        def fmt(m: str) -> str:
            c = agg[bucket][m]
            return f"{c['mean']:.4f}±{c['std']:.4f}"
        print(f"{bucket:8s} {fmt('R@10'):>16s} {fmt('N@10'):>16s} {fmt('R@20'):>16s} {fmt('N@20'):>16s}")
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
