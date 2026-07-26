"""Cold-oriented 软路由重扫：验证"软路由能否在不牺牲 Overall 的前提下把 Cold 顶更高"。

背景：evaluate_cbi_hot_routing_soft._fit_gate_on_val 的准则是
  "val Cold >= CGRC 底线 后，最大化 val Overall N@10"
一旦 Cold 过线优化器就把资源全投 Overall，导致软路由 Cold 停在 ~0.284、离 oracle(~0.375)差 24%。

本脚本换准则(约束-Pareto)：
  约束 = val Overall R@10 >= 协同基线(g≡0) val Overall R@10  AND  val Overall N@10 >= 协同基线 N@10
  目标 = 在可行域内最大化 val Cold R@10
即"Overall 不劣于纯协同专家的前提下，Cold 能顶多高"。单冷分支因 Overall 崩溃落在可行域外，
不再是反例，故这是一个良定义的约束-Pareto 保证。

同口径：完全复用 soft 的 build/precompute/_eval_gate 与 single_branch 的 _eval_fixed_g。冻结后一次性上 test。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_cbi_hot_routing_soft as soft
from evaluate_cbi_hot_routing_soft import _eval_gate, _precompute_split, _read_val_df
import evaluate_single_branch_baselines as sb
from evaluate_cbi_hot_routing_stage1 import (
    K_LIST, build_cbi_expert, build_hot_expert, _REPO_ROOT, _resolve_torch_device,
)

_METRIC_ORDER = [f"{m}@{k}" for k in K_LIST for m in ("R", "N")]


def _fit_coldmax(val_cache, device, tau_grid, T_grid, hot_val_overall):
    """Overall(R@10 & N@10) 不劣于协同基线 约束下最大化 Cold R@10。回退：仅 N@10 约束 -> 无约束最大 Cold。"""
    guard_r10 = hot_val_overall["overall"]["R@10"]
    guard_n10 = hot_val_overall["overall"]["N@10"]
    scan = []
    for tau in tau_grid:
        for T in T_grid:
            res, _, _ = _eval_gate(val_cache, float(tau), float(T), device)
            scan.append({
                "tau": float(tau), "T": float(T),
                "cold_R10": res["cold"]["R@10"], "cold_N10": res["cold"]["N@10"],
                "overall_R10": res["overall"]["R@10"], "overall_N10": res["overall"]["N@10"],
            })
    def pick(pred):
        cand = [r for r in scan if pred(r)]
        return max(cand, key=lambda r: r["cold_R10"]) if cand else None
    strict = pick(lambda r: r["overall_R10"] >= guard_r10 and r["overall_N10"] >= guard_n10)
    if strict is not None:
        return strict, "strict(R10&N10)", scan, (guard_r10, guard_n10)
    relaxed = pick(lambda r: r["overall_N10"] >= guard_n10)
    if relaxed is not None:
        return relaxed, "relaxed(N10-only)", scan, (guard_r10, guard_n10)
    unc = max(scan, key=lambda r: r["cold_R10"])
    return unc, "unconstrained(no-feasible)", scan, (guard_r10, guard_n10)


def run_seed(seed, device, batch_size, tau_grid, T_grid):
    print(f"\n===== cold-max soft-routing seed {seed} =====", flush=True)
    hot = build_hot_expert(seed, device)
    cbi = build_cbi_expert(seed, device)
    cold_threshold = cbi["cold_threshold"]
    cold_item_mask = torch.from_numpy(cbi["train_pop"] < cold_threshold).to(device)

    val_df = _read_val_df(cbi)
    test_df = cbi["test_df"].reset_index(drop=True)
    print(f"  precomputing val ({len(val_df)}) and test ({len(test_df)})...", flush=True)
    val_cache = _precompute_split(val_df, cbi, hot, cold_item_mask, device, batch_size)
    test_cache = _precompute_split(test_df, cbi, hot, cold_item_mask, device, batch_size)

    # 协同基线(g≡0) 在 val 上的 overall，作为约束阈值
    hot_val, _, _ = sb._eval_fixed_g(val_cache, 0.0, device)
    print(f"  [val hot-only guard] overall R@10={hot_val['overall']['R@10']:.4f} "
          f"N@10={hot_val['overall']['N@10']:.4f}", flush=True)

    best, mode, scan, guard = _fit_coldmax(val_cache, device, tau_grid, T_grid, hot_val)
    print(f"  [val fit cold-max/{mode}] tau={best['tau']:.3f} T={best['T']:.3f} "
          f"| val cold R@10={best['cold_R10']:.4f} N@10={best['cold_N10']:.4f} "
          f"overall R@10={best['overall_R10']:.4f} N@10={best['overall_N10']:.4f}", flush=True)

    test_res, n_cold, n_hot = _eval_gate(test_cache, best["tau"], best["T"], device)
    print(f"  [TEST cold-max] cold R@10={test_res['cold']['R@10']:.4f} N@10={test_res['cold']['N@10']:.4f} "
          f"| hot R@10={test_res['hot']['R@10']:.4f} | overall R@10={test_res['overall']['R@10']:.4f} "
          f"N@10={test_res['overall']['N@10']:.4f}", flush=True)

    return {
        "seed": seed, "fit": best, "fit_mode": mode,
        "test": test_res, "n_cold": n_cold, "n_hot": n_hot,
        "guard_overall_R10": guard[0], "guard_overall_N10": guard[1],
    }


def write_outputs(seed_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "coldmax_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "metric", "mean", "std"])
        for b in ("cold", "hot", "overall"):
            for m in _METRIC_ORDER:
                vals = [sr["test"][b][m] for sr in seed_results]
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                w.writerow([b, m, f"{mean:.6f}", f"{std:.6f}"])
    prov = {
        "experiment": "soft_routing_coldmax",
        "objective": "max val Cold R@10 s.t. val Overall (R@10 & N@10) >= hot-only(g=0) baseline",
        "per_seed_fit": [{"seed": sr["seed"], "mode": sr["fit_mode"], **sr["fit"],
                          "guard_overall_R10": sr["guard_overall_R10"],
                          "guard_overall_N10": sr["guard_overall_N10"]} for sr in seed_results],
        "seeds": [sr["seed"] for sr in seed_results],
    }
    (out_dir / "coldmax_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote outputs to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "soft_routing_coldmax")
    args = ap.parse_args()
    device = _resolve_torch_device()
    print(f"device={device} torch={torch.__version__} seeds={args.seeds}", flush=True)
    tau_grid = [round(x, 2) for x in np.arange(0.0, 6.01, 0.5)]
    T_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    seed_results = [run_seed(s, device, args.batch_size, tau_grid, T_grid) for s in args.seeds]
    write_outputs(seed_results, args.output_dir)


if __name__ == "__main__":
    main()
