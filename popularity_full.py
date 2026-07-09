"""
popularity_full.py — Popularity baseline under streaming protocol.

数据格式与 lightgcn_full_hin.py 完全一致:
  - processed_data_hin/{meta.json, stream_data.pkl, content_emb.pt}
  - 按月切分, warmup_periods=2, 累计训练
  - cold_threshold=5 (基于交互行的 popularity 字段)
  - 同时输出 sampled (1+200) 和 full ranking 结果, cold/hot 分组

打分规则: 所有用户共享同一个 item 流行度向量 score = log(1+count),
其中 count 来自截至当前 period 的累计 train data.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    split_dataframe_by_periods,
)
from hin_eval_common import compute_ranking_metric_values, compute_ranking_metrics, print_final_report


class Config:
    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items

        self.cold_threshold = int(os.environ.get("POP_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("POP_EVAL_N_NEG", "200"))
        self.warmup_periods = int(os.environ.get("POP_FULL_WARMUP", "2"))
        self.period_type = os.environ.get("POP_PERIOD_TYPE", "M")
        self.use_cumulative = os.environ.get("POP_USE_CUMULATIVE", "1") == "1"
        self.batch_size = int(os.environ.get("POP_BATCH_SIZE", "2048"))
        # 极小噪声打破完全相同的流行度造成的位置偏置
        self.tie_break_noise = float(os.environ.get("POP_TIE_BREAK_NOISE", "1e-6"))


def build_pop_score(train_df: pd.DataFrame, n_items: int, device: torch.device) -> torch.Tensor:
    counts = np.zeros(n_items, dtype=np.float64)
    if len(train_df) > 0:
        items = train_df["i_idx"].to_numpy(np.int64)
        np.add.at(counts, items, 1)
    score = np.log1p(counts).astype(np.float32)
    return torch.from_numpy(score).to(device)


def evaluate_popularity_ranker(
    loader,
    device: torch.device,
    n_items: int,
    cold_threshold: int,
    pop_score: torch.Tensor,
    k_list=(5, 10, 20),
    n_neg: int = 200,
    eval_type: str = "cold",
    full_ranking: bool = False,
    user_seen_items: Dict[int, set] = None,
    tie_break_noise: float = 1e-6,
    average_mode: str = "interaction",
    export_item_metrics_path: str = None,
):
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    accum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in k_list}
    total_samples = 0
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    item_counts: Dict[int, int] = {}
    seen_tensor_cache: Dict[int, torch.Tensor] = {}
    all_item_idx = torch.arange(n_items, device=device, dtype=torch.long)

    with torch.no_grad():
        for batch, pop in loader:
            if eval_type == "cold":
                mask = pop < cold_threshold
            elif eval_type == "hot":
                mask = pop >= cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = int(mask.sum().item())
            if n_sel < 1:
                continue

            batch_sel = {k: v[mask].to(device) for k, v in batch.items()}
            i = batch_sel["i"]
            user_ids = [int(x) for x in batch_sel["u"].detach().cpu().tolist()]

            if user_seen_items is not None:
                for uid in user_ids:
                    if uid in seen_tensor_cache:
                        continue
                    seen_items = user_seen_items.get(uid)
                    if seen_items:
                        seen_idx = [x for x in seen_items if 0 <= x < n_items]
                        seen_tensor_cache[uid] = (
                            torch.tensor(seen_idx, dtype=torch.long, device=device)
                            if seen_idx else None
                        )
                    else:
                        seen_tensor_cache[uid] = None

            if full_ranking:
                # 全库打分: 所有用户共享同一行
                scores = pop_score.unsqueeze(0).expand(n_sel, n_items).clone()
                scores = scores + torch.randn_like(scores) * tie_break_noise
                if user_seen_items is not None:
                    row_idx = torch.arange(n_sel, device=device)
                    target_scores = scores[row_idx, i].clone()
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache.get(uid)
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        seen_ex_tgt = int((seen_idx != i[row]).sum().item())
                        avail = n_items - 1 - seen_ex_tgt
                    avail_counts.append(max(1, avail))

                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is not None and seen_idx.numel() > 0:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]

                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                # 打乱候选顺序避免位置偏置
                perm = torch.argsort(
                    torch.rand(n_sel, cand_idx.size(1), device=device),
                    dim=1,
                )
                cand_idx = cand_idx.gather(1, perm)
                scores = pop_score[cand_idx]
                scores = scores + torch.randn_like(scores) * tie_break_noise
                target_indices = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            batch_values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
            if average_mode == "item_macro":
                item_ids = [int(x) for x in i.detach().cpu().tolist()]
                for row, item_id in enumerate(item_ids):
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    for key, values in batch_values.items():
                        per_item = item_accum[key]
                        per_item[item_id] = per_item.get(item_id, 0.0) + float(values[row].detach().cpu().item())
            else:
                for k, values in batch_values.items():
                    accum[k] += float(values.sum().detach().cpu().item())
            total_samples += n_sel

    if total_samples < 1:
        return None, 0
    if average_mode == "item_macro":
        if not item_counts:
            return None, 0
        macro = {}
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        if export_item_metrics_path:
            out_path = Path(export_item_metrics_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["item_id", "count"] + [f"{m}@{k}" for m in ["R", "N"] for k in k_list]
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item_id in sorted(item_counts):
                    count = max(1, item_counts[item_id])
                    row = {"item_id": int(item_id), "count": int(item_counts[item_id])}
                    for key, per_item in item_accum.items():
                        row[key] = float(per_item.get(item_id, 0.0) / count)
                    writer.writerow(row)
        return macro, len(item_counts)
    return {k: v / total_samples for k, v in accum.items()}, total_samples


def _weighted_add(dst: Dict[str, float], src: Dict[str, float], weight: int):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v * weight


def _weighted_avg(src: Dict[str, float], count: int):
    if count < 1:
        return {}
    return {k: v / count for k, v in src.items()}


def main():
    setup_seed(int(os.environ.get("POP_SEED", "2025")))
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, _ = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"])
    periods = split_dataframe_by_periods(df, period_type=cfg.period_type)
    print(
        f"Streaming periods: {len(periods)} | warmup={cfg.warmup_periods} | "
        f"cumulative_train={cfg.use_cumulative}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: Popularity stream | device={device}")

    k_list = [5, 10, 20]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    accum_sample_cold = {k: 0.0 for k in metrics_keys}
    accum_sample_hot = {k: 0.0 for k in metrics_keys}
    accum_full_cold = {k: 0.0 for k in metrics_keys}
    accum_full_hot = {k: 0.0 for k in metrics_keys}
    count_sample_cold = 0
    count_sample_hot = 0
    count_full_cold = 0
    count_full_hot = 0

    user_seen_items: Dict[int, set] = {}
    accumulated_dfs = []
    train_df_for_eval: pd.DataFrame = pd.DataFrame(columns=df.columns)

    for t, p_df in enumerate(periods):
        print(f"\n[Period {t + 1}/{len(periods)}] size={len(p_df)}")
        eval_loader = DataLoader(
            InteractionDataset(p_df),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_interactions,
        )

        if t >= cfg.warmup_periods and len(train_df_for_eval) > 0:
            pop_score = build_pop_score(train_df_for_eval, cfg.n_items, device)

            sample_cold, n_sc = evaluate_popularity_ranker(
                eval_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
                k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
                user_seen_items=user_seen_items, tie_break_noise=cfg.tie_break_noise,
            )
            sample_hot, n_sh = evaluate_popularity_ranker(
                eval_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
                k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
                user_seen_items=user_seen_items, tie_break_noise=cfg.tie_break_noise,
            )
            full_cold, n_fc = evaluate_popularity_ranker(
                eval_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
                k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, tie_break_noise=cfg.tie_break_noise,
            )
            full_hot, n_fh = evaluate_popularity_ranker(
                eval_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
                k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
                user_seen_items=user_seen_items, tie_break_noise=cfg.tie_break_noise,
            )

            sample_cold = sample_cold or {}
            sample_hot = sample_hot or {}
            full_cold = full_cold or {}
            full_hot = full_hot or {}

            _weighted_add(accum_sample_cold, sample_cold, n_sc)
            _weighted_add(accum_sample_hot, sample_hot, n_sh)
            _weighted_add(accum_full_cold, full_cold, n_fc)
            _weighted_add(accum_full_hot, full_hot, n_fh)
            count_sample_cold += n_sc
            count_sample_hot += n_sh
            count_full_cold += n_fc
            count_full_hot += n_fh

            print(
                "  Eval: "
                f"sample_cold_R@10={sample_cold.get('R@10', 0.0):.4f}, "
                f"sample_hot_R@10={sample_hot.get('R@10', 0.0):.4f}, "
                f"full_cold_R@10={full_cold.get('R@10', 0.0):.4f}, "
                f"full_hot_R@10={full_hot.get('R@10', 0.0):.4f}"
            )
        else:
            print("  Warmup period: skip evaluation")

        if cfg.use_cumulative:
            accumulated_dfs.append(p_df)
            train_df_for_eval = pd.concat(accumulated_dfs, ignore_index=True)
        else:
            train_df_for_eval = p_df
        add_user_seen_from_df(user_seen_items, p_df)

    final_sample_cold = _weighted_avg(accum_sample_cold, count_sample_cold)
    final_sample_hot = _weighted_avg(accum_sample_hot, count_sample_hot)
    final_full_cold = _weighted_avg(accum_full_cold, count_full_cold)
    final_full_hot = _weighted_avg(accum_full_hot, count_full_hot)

    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=final_sample_cold,
        sample_hot=final_sample_hot,
        full_cold=final_full_cold,
        full_hot=final_full_hot,
        count_sample_cold=count_sample_cold,
        count_sample_hot=count_sample_hot,
        count_full_cold=count_full_cold,
        count_full_hot=count_full_hot,
        title="Popularity Streaming HIN",
    )

    out = {
        "sample_cold": final_sample_cold,
        "sample_hot": final_sample_hot,
        "full_cold": final_full_cold,
        "full_hot": final_full_hot,
        "count_sample_cold": count_sample_cold,
        "count_sample_hot": count_sample_hot,
        "count_full_cold": count_full_cold,
        "count_full_hot": count_full_hot,
        "periods": len(periods),
        "warmup_periods": cfg.warmup_periods,
        "use_cumulative": cfg.use_cumulative,
    }
    pd.DataFrame([out]).to_json("popularity_full_result.json", orient="records", force_ascii=False)
    print("Saved: popularity_full_result.json")


if __name__ == "__main__":
    main()
