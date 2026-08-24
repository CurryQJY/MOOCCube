"""单分支基线：与软路由(evaluate_cbi_hot_routing_soft)完全同口径，只把融合门控固定为常量。

目的：为"拼接无损下界"判生死提供两个端点对照——
  ① 单冷分支 (g≡1): s = zscore(s_cbi)              纯内容专家
  ② 单热分支 (g≡0): s = zscore(s_hot)              纯图协同专家
再与 soft 脚本已产出的 ③软路由 / ④硬路由 拼成四配置对照表，即可判定：
  A. 融合有增益:  soft >= max(单冷,单热)  且 Cold/Overall 都不降  -> bound 是钢筋
  B. 融合无功:    soft ≈ max(单冷,单热)                          -> bound 退化平凡
  C. 融合有害:    soft <  max(单冷,单热)                          -> 放弃这条 spine

同口径保证：直接复用 soft 脚本的 build_cbi/build_hot(冻结同批 ckpt)、_precompute_split(同 static split、
item 0..697、cold_threshold、seen 掩码 -1e9 保留目标、per-user zscore、item-macro)。唯一区别是门控 g 为常量。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_cbi_hot_routing_soft as soft
from evaluate_cbi_hot_routing_stage1 import (
    K_LIST,
    NEG_INF,
    MacroAccumulator,
    build_cbi_expert,
    build_hot_expert,
    _REPO_ROOT,
    _resolve_torch_device,
)
from hin_eval_common import compute_ranking_metric_values

_METRIC_ORDER = [f"{m}@{k}" for k in K_LIST for m in ("R", "N")]


def _eval_fixed_g(cache, g_value: float, device, batch_size=4096):
    """与 soft._eval_gate 完全相同的融合/掩码/度量流程，但 g 为常量标量 (逐候选相同)。

    g_value=1.0 -> 纯 cbi_z (单冷分支)；g_value=0.0 -> 纯 hot_z (单热分支)。
    """
    n_items = cache["cbi_z"].shape[1]
    g = torch.full((n_items,), float(g_value), device=device)
    acc = MacroAccumulator()
    N = cache["cbi_z"].shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cbi_z = cache["cbi_z"][start:end].to(device)
        hot_z = cache["hot_z"][start:end].to(device)
        i = cache["tgt_i"][start:end].to(device)
        tcold = cache["tgt_cold"][start:end].to(device)
        valid = (cbi_z != 0) | (hot_z != 0)  # z==0 marks invalid/seen (post-zscore), 与 soft 一致
        s_route = g.unsqueeze(0) * cbi_z + (1.0 - g).unsqueeze(0) * hot_z
        rowsB = torch.arange(i.size(0), device=device)
        tgt_val = s_route[rowsB, i].clone()
        s_route = torch.where(valid, s_route, torch.full_like(s_route, NEG_INF))
        s_route[rowsB, i] = tgt_val
        metric_vals = compute_ranking_metric_values(s_route, i, k_list=K_LIST)
        acc.add_batch(tcold, i, metric_vals)
    res, n_cold, n_hot = acc.finalize()
    return res, n_cold, n_hot


def run_seed(seed, device, batch_size):
    print(f"\n===== single-branch seed {seed} =====", flush=True)
    hot = build_hot_expert(seed, device)
    cbi = build_cbi_expert(seed, device)
    cold_threshold = cbi["cold_threshold"]
    cold_item_mask = torch.from_numpy(cbi["train_pop"] < cold_threshold).to(device)

    test_df = cbi["test_df"].reset_index(drop=True)
    print(f"  precomputing test ({len(test_df)}) scores (same protocol as soft)...", flush=True)
    test_cache = soft._precompute_split(test_df, cbi, hot, cold_item_mask, device, batch_size)

    cold_res, n_cold, n_hot = _eval_fixed_g(test_cache, 1.0, device)  # ① 单冷分支
    hot_res, _, _ = _eval_fixed_g(test_cache, 0.0, device)            # ② 单热分支

    def _fmt(tag, r):
        print(f"  [TEST {tag}] cold R@10={r['cold']['R@10']:.4f} N@10={r['cold']['N@10']:.4f} "
              f"| hot R@10={r['hot']['R@10']:.4f} N@10={r['hot']['N@10']:.4f} "
              f"| overall R@10={r['overall']['R@10']:.4f} N@10={r['overall']['N@10']:.4f}", flush=True)
    _fmt("cold-only (g=1)", cold_res)
    _fmt("hot-only  (g=0)", hot_res)

    return {
        "seed": seed,
        "n_cold": n_cold,
        "n_hot": n_hot,
        "cold_only": cold_res,   # ① 单冷分支
        "hot_only": hot_res,     # ② 单热分支
        "cbi_ckpt": str(cbi["ckpt_path"]),
        "hot_ckpt": str(hot["ckpt_path"]),
    }


def write_outputs(seed_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "single_branch_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "bucket", "metric", "mean", "std"])
        for variant, key in (("cold_only", "cold_only"), ("hot_only", "hot_only")):
            for b in ("cold", "hot", "overall"):
                for m in _METRIC_ORDER:
                    vals = [sr[key][b][m] for sr in seed_results]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    w.writerow([variant, b, m, f"{mean:.6f}", f"{std:.6f}"])
    prov = {
        "experiment": "single_branch_baselines",
        "description": "Fixed-gate endpoints g=1 (cold-only) and g=0 (hot-only), same protocol as soft routing.",
        "seeds": [sr["seed"] for sr in seed_results],
        "checkpoints": [{"seed": sr["seed"], "cbi_ckpt": sr["cbi_ckpt"], "hot_ckpt": sr["hot_ckpt"]}
                        for sr in seed_results],
    }
    (out_dir / "single_branch_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote outputs to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025])
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--output-dir", type=Path,
                    default=_REPO_ROOT / "outputs" / "single_branch_baselines")
    args = ap.parse_args()
    device = _resolve_torch_device()
    print(f"device={device} torch={torch.__version__} seeds={args.seeds}", flush=True)
    seed_results = [run_seed(s, device, args.batch_size) for s in args.seeds]
    write_outputs(seed_results, args.output_dir)


if __name__ == "__main__":
    main()
