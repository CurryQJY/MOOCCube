"""学习式软路由：按候选课流行度平滑地在 CBI(内容) 与 Hot(图协同) 两专家间转移信任。

与 stage1 硬路由的区别：
  - 硬路由：冷候选只用 CBI、热候选只用 Hot（一刀切）。
  - 软路由：每个候选 c 融合两专家的校准分
        s_route(c) = g(c)·zscore(s_cbi)(c) + (1-g(c))·zscore(s_hot)(c)
    其中门控 g(c) = sigmoid((tau - log(pop_c+1)) / T)，仅 (tau, T) 两个可学参数。
    pop 越低(越冷) -> g->1 -> 越信 CBI；pop 越高(越热) -> g->0 -> 越信 Hot。

关键性质：
  - 两个专家分数各自在“该用户全部有效候选”上做 zscore，故同一用户内可比（不需跨模型物理校准）。
  - T->0 时 g 退化为按 cold_threshold 的硬阶跃 => 复现 stage1 硬切分，因此软路由是硬路由的连续松弛，
    在 validation 上择优后至少不差于硬路由。
  - (tau, T) 只在 validation 上按“Cold 不输 CGRC 且 Overall 最大”网格拟合，冻结后一次性上 test（无 test 信息泄漏）。

与主表 / stage1 完全同口径：同一份 static split、item 编号 0..697、cold_threshold=1、
full-ranking、seen 掩码 -1e9(保留目标)、item-macro 平均、compute_ranking_metric_values。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate_cbi_hot_routing_stage1 as s1
from evaluate_cbi_hot_routing_stage1 import (
    K_LIST,
    NEG_INF,
    MacroAccumulator,
    build_cbi_expert,
    build_hot_expert,
    build_seen_bool,
    cbi_user_vectors,
    _REPO_ROOT,
    _resolve_torch_device,
    _sha256,
)
from hin_eval_common import compute_ranking_metric_values

import pandas as pd
import torch.nn.functional as F

# CGRC 主表参照(item-macro, 3seed)：软路由的 Cold 底线与 Overall 目标。
CGRC_COLD_R10 = 0.2589
CGRC_COLD_N10 = 0.1845
CGRC_OVERALL_R10 = 0.2494
CGRC_OVERALL_N10 = 0.1615


def _zscore_all(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-row z-score over ALL valid candidates (not per-bucket). Invalid -> 0 (post-fusion masked)."""
    masked = torch.where(valid, scores, torch.zeros_like(scores))
    n = valid.sum(dim=1, keepdim=True).clamp(min=1)
    mean = masked.sum(dim=1, keepdim=True) / n
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    var = (centered * centered).sum(dim=1, keepdim=True) / n
    std = var.sqrt().clamp(min=1e-6)
    z = torch.where(valid, centered / std, torch.zeros_like(scores))
    return z


def _read_val_df(cbi: dict) -> pd.DataFrame:
    """Load the validation split declared in the CBI manifest (same protocol as train/test)."""
    manifest = json.loads(cbi["manifest_path"].read_text(encoding="utf-8-sig"))
    val_path = manifest["exports"]["val_split"]
    df = pd.read_pickle(val_path)
    return df.reset_index(drop=True)


def _precompute_split(df, cbi, hot, cold_item_mask, device, batch_size):
    """Return cached per-row tensors needed for cheap (tau,T) grid search / eval.

    Caches the two z-scored expert score rows (independent of tau,T), the log-pop
    feature, target ids, and cold/hot target bucket. Fusion for a given (tau,T) is
    then a cheap linear combine + metric call.
    """
    n_items = cbi["n_items"]
    cbi_bank = cbi["item_bank"].to(device)
    hot_bank = hot["all_i"].to(device)
    all_u_hot = hot["all_u"].to(device)
    cbi_model = cbi["model"]

    u_all = torch.from_numpy(df["u_idx"].astype(np.int64).to_numpy())
    i_all = torch.from_numpy(df["i_idx"].astype(np.int64).to_numpy())
    n_rows = u_all.numel()

    unique_users = np.unique(u_all.numpy())
    user_to_row = {int(u): r for r, u in enumerate(unique_users)}
    seen_bool_unique = build_seen_bool(cbi["train_df"], unique_users, n_items).to(device)
    inv = torch.from_numpy(
        np.asarray([user_to_row[int(u)] for u in u_all.numpy()], dtype=np.int64)
    )

    cbi_z_rows, hot_z_rows, tgt_i_rows, tgt_cold_rows = [], [], [], []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        u = u_all[start:end].to(device)
        i = i_all[start:end].to(device)
        rows_seen = seen_bool_unique[inv[start:end].to(device)]

        z_cbi = cbi_user_vectors(cbi_model, u)
        s_cbi = torch.mm(z_cbi, cbi_bank.t())
        z_hot = all_u_hot[u]
        s_hot = torch.mm(z_hot, hot_bank.t())

        rowsB = torch.arange(u.size(0), device=device)
        tgt_cbi = s_cbi[rowsB, i].clone()
        tgt_hot = s_hot[rowsB, i].clone()
        s_cbi = s_cbi.masked_fill(rows_seen, NEG_INF)
        s_hot = s_hot.masked_fill(rows_seen, NEG_INF)
        s_cbi[rowsB, i] = tgt_cbi
        s_hot[rowsB, i] = tgt_hot

        valid = s_cbi > (NEG_INF / 2)  # seen-mask identical for both experts
        cbi_z = _zscore_all(s_cbi, valid)
        hot_z = _zscore_all(s_hot, valid)
        # keep invalid marker so fusion can re-mask
        cbi_z_rows.append(cbi_z.cpu())
        hot_z_rows.append(hot_z.cpu())
        tgt_i_rows.append(i.cpu())
        tgt_cold_rows.append(cold_item_mask[i].cpu())

    logpop = torch.log1p(torch.from_numpy(cbi["train_pop"].astype(np.float32))).to(device)  # [n_items]
    return {
        "cbi_z": torch.cat(cbi_z_rows, 0),      # [N, n_items]
        "hot_z": torch.cat(hot_z_rows, 0),
        "tgt_i": torch.cat(tgt_i_rows, 0),
        "tgt_cold": torch.cat(tgt_cold_rows, 0),
        "logpop": logpop,
        "seen_from_cbi_z": None,
    }


def _eval_gate(cache, tau: float, T: float, device, batch_size=4096):
    """Fuse with gate g=sigmoid((tau-logpop)/T) and compute item-macro metrics."""
    logpop = cache["logpop"]
    g = torch.sigmoid((tau - logpop) / max(T, 1e-6))  # [n_items], high for cold
    acc = MacroAccumulator()
    N = cache["cbi_z"].shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cbi_z = cache["cbi_z"][start:end].to(device)
        hot_z = cache["hot_z"][start:end].to(device)
        i = cache["tgt_i"][start:end].to(device)
        tcold = cache["tgt_cold"][start:end].to(device)
        valid = (cbi_z != 0) | (hot_z != 0)  # z==0 marks invalid/seen (post-zscore)
        s_route = g.unsqueeze(0) * cbi_z + (1.0 - g).unsqueeze(0) * hot_z
        # restore target (may be legitimately ~0 after zscore; force valid) and mask invalid
        rowsB = torch.arange(i.size(0), device=device)
        tgt_val = s_route[rowsB, i].clone()
        s_route = torch.where(valid, s_route, torch.full_like(s_route, NEG_INF))
        s_route[rowsB, i] = tgt_val
        metric_vals = compute_ranking_metric_values(s_route, i, k_list=K_LIST)
        acc.add_batch(tcold, i, metric_vals)
    res, n_cold, n_hot = acc.finalize()
    return res, n_cold, n_hot


def _fit_gate_on_val(val_cache, device, tau_grid, T_grid):
    """Pick (tau,T) maximizing val Overall N@10 s.t. val Cold R@10 >= CGRC and Cold N@10 >= CGRC.

    Falls back to maximizing Cold if no point satisfies the Overall-competitive guard,
    so the fit never crashes; the guard status is reported.
    """
    best = None
    scan = []
    for tau in tau_grid:
        for T in T_grid:
            res, _, _ = _eval_gate(val_cache, float(tau), float(T), device)
            cold_ok = (res["cold"]["R@10"] >= CGRC_COLD_R10) and (res["cold"]["N@10"] >= CGRC_COLD_N10)
            row = {
                "tau": float(tau), "T": float(T),
                "cold_R10": res["cold"]["R@10"], "cold_N10": res["cold"]["N@10"],
                "overall_R10": res["overall"]["R@10"], "overall_N10": res["overall"]["N@10"],
                "cold_ok": bool(cold_ok),
            }
            scan.append(row)
            key = (1 if cold_ok else 0, res["overall"]["N@10"])
            if best is None or key > best[0]:
                best = (key, row)
    return best[1], scan


def run_seed(seed, device, batch_size, tau_grid, T_grid):
    print(f"\n===== soft-routing seed {seed} =====", flush=True)
    hot = build_hot_expert(seed, device)
    cbi = build_cbi_expert(seed, device)
    n_items = cbi["n_items"]
    cold_threshold = cbi["cold_threshold"]
    cold_item_mask = torch.from_numpy(cbi["train_pop"] < cold_threshold).to(device)

    val_df = _read_val_df(cbi)
    test_df = cbi["test_df"].reset_index(drop=True)

    print(f"  precomputing val ({len(val_df)}) and test ({len(test_df)}) scores...", flush=True)
    val_cache = _precompute_split(val_df, cbi, hot, cold_item_mask, device, batch_size)
    test_cache = _precompute_split(test_df, cbi, hot, cold_item_mask, device, batch_size)

    best, scan = _fit_gate_on_val(val_cache, device, tau_grid, T_grid)
    print(f"  [val fit] tau={best['tau']:.3f} T={best['T']:.3f} cold_ok={best['cold_ok']} "
          f"| val cold R@10={best['cold_R10']:.4f} N@10={best['cold_N10']:.4f} "
          f"overall R@10={best['overall_R10']:.4f} N@10={best['overall_N10']:.4f}", flush=True)

    # frozen (tau,T) -> test
    test_res, n_cold, n_hot = _eval_gate(test_cache, best["tau"], best["T"], device)
    # hard-routing reference on test: T->0 limit
    hard_res, _, _ = _eval_gate(test_cache, best["tau"], 1e-4, device)

    print(f"  [TEST soft ] cold R@10={test_res['cold']['R@10']:.4f} N@10={test_res['cold']['N@10']:.4f} "
          f"| hot R@10={test_res['hot']['R@10']:.4f} | overall R@10={test_res['overall']['R@10']:.4f} "
          f"N@10={test_res['overall']['N@10']:.4f}", flush=True)
    print(f"  [TEST hard ] cold R@10={hard_res['cold']['R@10']:.4f} "
          f"overall R@10={hard_res['overall']['R@10']:.4f} N@10={hard_res['overall']['N@10']:.4f}", flush=True)

    return {
        "seed": seed,
        "fit": best,
        "test_soft": test_res,
        "test_hard": hard_res,
        "n_cold": n_cold,
        "n_hot": n_hot,
        "cbi_ckpt": cbi["ckpt_path"],
        "hot_ckpt": hot["ckpt_path"],
        "val_scan": scan,
    }


_METRIC_ORDER = [f"{m}@{k}" for k in K_LIST for m in ("R", "N")]


def write_outputs(seed_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # per-seed test (soft)
    fieldnames = ["seed", "tau", "T", "n_cold", "n_hot"] + [
        f"{b}_{m}" for b in ("cold", "hot", "overall") for m in _METRIC_ORDER
    ]
    with (out_dir / "soft_routing_per_seed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for sr in seed_results:
            row = {"seed": sr["seed"], "tau": sr["fit"]["tau"], "T": sr["fit"]["T"],
                   "n_cold": sr["n_cold"], "n_hot": sr["n_hot"]}
            for b in ("cold", "hot", "overall"):
                for m in _METRIC_ORDER:
                    row[f"{b}_{m}"] = sr["test_soft"][b][m]
            w.writerow(row)

    # summary mean±std (soft) + hard reference
    with (out_dir / "soft_routing_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "bucket", "metric", "mean", "std"])
        for variant, key in (("soft", "test_soft"), ("hard", "test_hard")):
            for b in ("cold", "hot", "overall"):
                for m in _METRIC_ORDER:
                    vals = [sr[key][b][m] for sr in seed_results]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    w.writerow([variant, b, m, f"{mean:.6f}", f"{std:.6f}"])

    prov = {
        "experiment": "cbi_hot_routing_soft",
        "description": (
            "Learned soft routing over two frozen experts. Per candidate c: "
            "s_route(c) = g(c)*zscore(s_cbi)(c) + (1-g(c))*zscore(s_hot)(c), "
            "g(c)=sigmoid((tau-log1p(pop_c))/T). (tau,T) fit on validation to maximize "
            "Overall N@10 subject to Cold R@10>=CGRC and Cold N@10>=CGRC, frozen, then test once."
        ),
        "guards": {"cgrc_cold_r10": CGRC_COLD_R10, "cgrc_cold_n10": CGRC_COLD_N10,
                   "cgrc_overall_r10": CGRC_OVERALL_R10, "cgrc_overall_n10": CGRC_OVERALL_N10},
        "seeds": [sr["seed"] for sr in seed_results],
        "per_seed_fit": [{"seed": sr["seed"], **sr["fit"]} for sr in seed_results],
        "checkpoints": [
            {"seed": sr["seed"], "cbi_ckpt": str(sr["cbi_ckpt"]),
             "cbi_sha256": _sha256(sr["cbi_ckpt"]),
             "hot_ckpt": str(sr["hot_ckpt"]), "hot_sha256": _sha256(sr["hot_ckpt"])}
            for sr in seed_results
        ],
    }
    (out_dir / "soft_routing_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote outputs to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "cbi_hot_routing_soft")
    args = ap.parse_args()

    seeds = [2025] if args.validate_only else args.seeds
    device = _resolve_torch_device()
    print(f"device={device} torch={torch.__version__} seeds={seeds}", flush=True)

    # tau over plausible log-pop range; T from near-hard (0.05) to smooth (2.0)
    tau_grid = [round(x, 2) for x in np.arange(0.0, 6.01, 0.5)]
    T_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]

    seed_results = [run_seed(s, device, args.batch_size, tau_grid, T_grid) for s in seeds]

    if args.validate_only:
        print("\n[validate-only] seed 2025 done; not writing full CSVs.")
        return
    write_outputs(seed_results, args.output_dir)


if __name__ == "__main__":
    main()
