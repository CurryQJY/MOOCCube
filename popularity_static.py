"""
popularity_static.py — Popularity baseline under static 8/1/1 split.

数据格式与 lightgcn_static_hin.py 完全一致:
  - processed_data_hin/{meta.json, stream_data.pkl, content_emb.pt}
  - 8/1/1 random split (train_ratio=0.8, val_ratio=0.1)
  - 在 train_df 上统计 item count, 在 test_df 上评估
  - cold_threshold=5, sampled (1+200) 与 full ranking 同时输出, cold/hot 分组
"""

import os
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    static_result_path,
    static_split_df,
)
from hin_eval_common import print_final_report
from popularity_full import build_pop_score, evaluate_popularity_ranker


class Config:
    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items

        self.cold_threshold = int(os.environ.get("POP_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("POP_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.batch_size = int(os.environ.get("POP_BATCH_SIZE", "2048"))
        self.static_seed = int(os.environ.get("POP_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.train_ratio = float(os.environ.get("POP_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("POP_STATIC_VAL_RATIO", "0.1"))
        self.tie_break_noise = float(os.environ.get("POP_TIE_BREAK_NOISE", "1e-6"))


def main():
    setup_seed(int(os.environ.get("POP_SEED", "2025")))
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, _ = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"])

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    print(
        f"Static split done: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}"
    )

    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        add_user_seen_from_df(test_seen, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pop_score = build_pop_score(train_df, cfg.n_items, device)
    print(f"Model: Popularity static | device={device}")

    k_list = [5, 10, 20]

    sample_cold, n_sc = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
    )
    sample_hot, n_sh = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
    )
    full_cold, n_fc = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
    )
    full_hot, n_fh = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
    )
    full_cold_item_macro, n_fc_item_macro = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
        average_mode="item_macro",
    )
    full_hot_item_macro, n_fh_item_macro = evaluate_popularity_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, pop_score,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, tie_break_noise=cfg.tie_break_noise,
        average_mode="item_macro",
    )

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
    full_cold_item_macro = full_cold_item_macro or {}
    full_hot_item_macro = full_hot_item_macro or {}
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=sample_cold,
        sample_hot=sample_hot,
        full_cold=full_cold,
        full_hot=full_hot,
        count_sample_cold=n_sc,
        count_sample_hot=n_sh,
        count_full_cold=n_fc,
        count_full_hot=n_fh,
        title="Popularity Static HIN",
    )

    out = {
        "model": "Popularity",
        "protocol": "static_item_cold",
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item_macro,
        "full_hot_item_macro": full_hot_item_macro,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item_macro,
        "count_full_hot_item_macro": n_fh_item_macro,
        "best_epoch": None,
        "best_metric": "non_trainable",
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
    }
    result_path = static_result_path("popularity_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
