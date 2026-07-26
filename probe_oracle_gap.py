"""可行性探针：oracle 拼接上界能否用可观测信号(pop)闭合？

核心问题：现有软/硬路由的 Cold 停在 ~0.28，离单冷专家 0.375 差一大截。是"特征不够"
还是"门控设定错"？关键事实：cold item ⟺ train_pop==0（cold_threshold=1），而 pop 完全可观测。
故"按真 cold/hot 标签路由"的 oracle 其实可用 pop 直接实现：cold->信内容(g=1)、hot->信协同(g=0)。

本探针在 test 上直接测若干 per-item 门控向量 g[n_items]（同 soft 口径的融合/掩码/度量）：
  - oracle_label : g = 1{pop==0}          真标签硬阶跃（可观测，pop 实现）
  - cold_only    : g ≡ 1                   复现校验（应=单冷 0.375）
  - hot_only     : g ≡ 0                   复现校验（应=单热 0.230）
若 oracle_label 同时拿到 Cold≈0.375 且 Overall≈0.245 -> gap 用 pop 即可闭合(绿灯，且比想象更强)。
若 oracle_label 的 Overall 反而崩 -> 说明 cold 桶硬切会伤 hot 排序，gap 需更聪明的按候选校准(仍绿灯，指向新方法)。
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

import evaluate_cbi_hot_routing_soft as soft
from evaluate_cbi_hot_routing_soft import _precompute_split, _read_val_df
from evaluate_cbi_hot_routing_stage1 import (
    K_LIST, NEG_INF, MacroAccumulator, build_cbi_expert, build_hot_expert, _resolve_torch_device,
)
from hin_eval_common import compute_ranking_metric_values


def _eval_g_vec(cache, g_vec, device, batch_size=4096):
    """与 soft._eval_gate 完全同流程，但门控是任意给定的 per-item 向量 g_vec[n_items]。"""
    g = g_vec.to(device)
    acc = MacroAccumulator()
    N = cache["cbi_z"].shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cbi_z = cache["cbi_z"][start:end].to(device)
        hot_z = cache["hot_z"][start:end].to(device)
        i = cache["tgt_i"][start:end].to(device)
        tcold = cache["tgt_cold"][start:end].to(device)
        valid = (cbi_z != 0) | (hot_z != 0)
        s_route = g.unsqueeze(0) * cbi_z + (1.0 - g).unsqueeze(0) * hot_z
        rowsB = torch.arange(i.size(0), device=device)
        tgt_val = s_route[rowsB, i].clone()
        s_route = torch.where(valid, s_route, torch.full_like(s_route, NEG_INF))
        s_route[rowsB, i] = tgt_val
        metric_vals = compute_ranking_metric_values(s_route, i, k_list=K_LIST)
        acc.add_batch(tcold, i, metric_vals)
    res, _, _ = acc.finalize()
    return res


def run_seed(seed, device, batch_size):
    print(f"\n===== oracle-gap probe seed {seed} =====", flush=True)
    hot = build_hot_expert(seed, device)
    cbi = build_cbi_expert(seed, device)
    cold_threshold = cbi["cold_threshold"]
    train_pop = cbi["train_pop"]
    cold_item = torch.from_numpy((train_pop < cold_threshold).astype(np.float32))  # 1 for cold(pop==0)
    n_items = cbi["n_items"]
    print(f"  n_items={n_items} n_cold_items={int(cold_item.sum())} "
          f"({100*float(cold_item.mean()):.1f}%)", flush=True)

    cold_item_mask = cold_item.bool().to(device)
    test_df = cbi["test_df"].reset_index(drop=True)
    test_cache = _precompute_split(test_df, cbi, hot, cold_item_mask, device, batch_size)

    configs = {
        "oracle_label": cold_item,                       # g=1 for cold(pop==0), 0 for hot
        "cold_only   ": torch.ones(n_items),
        "hot_only    ": torch.zeros(n_items),
    }
    for name, g_vec in configs.items():
        r = _eval_g_vec(test_cache, g_vec, device, batch_size)
        print(f"  [{name}] cold R@10={r['cold']['R@10']:.4f} N@10={r['cold']['N@10']:.4f} "
              f"| hot R@10={r['hot']['R@10']:.4f} | overall R@10={r['overall']['R@10']:.4f} "
              f"N@10={r['overall']['N@10']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    device = _resolve_torch_device()
    print(f"device={device} torch={torch.__version__} seeds={args.seeds}", flush=True)
    for s in args.seeds:
        run_seed(s, device, args.batch_size)


if __name__ == "__main__":
    main()
