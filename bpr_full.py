"""
bpr_full.py — Pure BPR Matrix Factorization baseline under streaming protocol.

数据格式与 lightgcn_full_hin.py 一致, 但模型为纯 ID-based MF (无图传播, 无 content):
  - processed_data_hin/{meta.json, stream_data.pkl, content_emb.pt}
  - 按月切分, warmup_periods=2, 累计训练
  - cold_threshold=5, sampled (1+200) 与 full ranking 同时输出, cold/hot 分组
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bpr_static import BPRMFModel
from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    split_dataframe_by_periods,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report
from lightgcn_static_hin import (
    compute_bpr_loss,
    prepare_train_cache,
    sample_negatives,
)


class Config:
    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items

        self.emb_dim = int(os.environ.get("BPR_EMB_DIM", "128"))
        self.lr = float(os.environ.get("BPR_LR", "1e-3"))
        self.reg_weight = float(os.environ.get("BPR_REG", "1e-4"))
        self.n_epochs = int(os.environ.get("BPR_FULL_EPOCHS", "2"))
        self.batch_size = int(os.environ.get("BPR_BATCH_SIZE", "2048"))

        self.cold_threshold = int(os.environ.get("BPR_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("BPR_EVAL_N_NEG", "200"))
        self.warmup_periods = int(os.environ.get("BPR_FULL_WARMUP", "2"))
        self.period_type = os.environ.get("BPR_PERIOD_TYPE", "M")
        self.use_cumulative = os.environ.get("BPR_USE_CUMULATIVE", "1") == "1"


def _weighted_add(dst: Dict[str, float], src: Dict[str, float], weight: int):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v * weight


def _weighted_avg(src: Dict[str, float], count: int):
    if count < 1:
        return {}
    return {k: v / count for k, v in src.items()}


def main():
    setup_seed(int(os.environ.get("BPR_SEED", "2025")))
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, _ = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"])
    periods = split_dataframe_by_periods(df, period_type=cfg.period_type)
    print(
        f"Streaming periods: {len(periods)} | warmup={cfg.warmup_periods} | "
        f"cumulative_train={cfg.use_cumulative} | epochs/period={cfg.n_epochs}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRMFModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: BPR-MF stream | device={device} | emb_dim={cfg.emb_dim}")

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

    for t, p_df in enumerate(periods):
        print(f"\n[Period {t + 1}/{len(periods)}] size={len(p_df)}")
        eval_loader = DataLoader(
            InteractionDataset(p_df),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_interactions,
        )

        if t >= cfg.warmup_periods:
            model.eval()
            with torch.no_grad():
                z_u, z_i = model.all_embeddings()
                all_u = F.normalize(z_u, dim=1)
                all_i = F.normalize(z_i, dim=1)
                get_user_fn = lambda b: all_u[b["u"]]

                sample_cold, n_sc = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
                    user_seen_items=user_seen_items,
                )
                sample_hot, n_sh = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
                    user_seen_items=user_seen_items,
                )
                full_cold, n_fc = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
                    user_seen_items=user_seen_items,
                )
                full_hot, n_fh = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
                    user_seen_items=user_seen_items,
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
            train_df = pd.concat(accumulated_dfs, ignore_index=True)
        else:
            train_df = p_df
        if len(train_df) < 1:
            add_user_seen_from_df(user_seen_items, p_df)
            continue

        train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
        train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
        train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)

        model.train()
        for ep in range(cfg.n_epochs):
            neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
            train_neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            z_u, z_i = model.all_embeddings()
            loss = compute_bpr_loss(
                z_u, z_i, train_users_t, train_pos_t, train_neg_t,
                reg_weight=cfg.reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            print(f"  Train epoch [{ep + 1}/{cfg.n_epochs}] loss={float(loss.item()):.4f}")

        del train_users_t, train_pos_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        title="BPR-MF Streaming HIN",
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
    pd.DataFrame([out]).to_json("bpr_full_result.json", orient="records", force_ascii=False)
    print("Saved: bpr_full_result.json")


if __name__ == "__main__":
    main()
