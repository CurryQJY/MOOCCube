"""
Content-profile baseline for the shared static item cold-start split.

Each user is represented by the mean content embedding of items observed in the
training set. Candidate items are scored by dot product with normalized content
embeddings. This is a non-trainable side-information baseline and intentionally
separate from MARec.
"""

import os

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
from hin_eval_common import evaluate_embedding_ranker, print_final_report


class Config:
    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = int(os.environ.get("CONTENT_PROFILE_BATCH_SIZE", "4096"))
        self.cold_threshold = int(
            os.environ.get("CONTENT_PROFILE_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5"))
        )
        self.eval_n_neg = int(
            os.environ.get("CONTENT_PROFILE_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200"))
        )
        self.static_seed = int(
            os.environ.get("CONTENT_PROFILE_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025"))
        )
        self.seed = int(os.environ.get("CONTENT_PROFILE_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("CONTENT_PROFILE_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("CONTENT_PROFILE_STATIC_VAL_RATIO", "0.1"))


def build_user_content_profiles(train_seen, item_vectors: torch.Tensor, n_users: int, device) -> torch.Tensor:
    profiles = torch.zeros((n_users, item_vectors.shape[1]), dtype=torch.float32, device=device)
    for uid, seen in train_seen.items():
        if uid < 0 or uid >= n_users or not seen:
            continue
        seen_idx = [int(item) for item in seen if 0 <= int(item) < item_vectors.shape[0]]
        if not seen_idx:
            continue
        idx = torch.tensor(seen_idx, dtype=torch.long, device=device)
        profiles[int(uid)] = item_vectors[idx].mean(dim=0)
    return torch.nn.functional.normalize(profiles, dim=1)


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"])
    setup_seed(cfg.seed)

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

    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
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
    all_i = torch.nn.functional.normalize(content_emb.float().to(device), dim=1)
    all_u = build_user_content_profiles(train_seen, all_i, cfg.n_users, device)
    get_user_fn = lambda batch: all_u[batch["u"]]
    k_list = [5, 10, 20]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    print(f"Model: ContentProfile static | device={device}")
    val_full_cold, _ = evaluate_embedding_ranker(
        val_loader,
        device=device,
        n_items=cfg.n_items,
        cold_threshold=cfg.cold_threshold,
        get_user_vectors_fn=get_user_fn,
        all_item_vectors=all_i,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=train_seen,
    )
    val_key = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0

    sample_cold, n_sc = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
        user_seen_items=test_seen,
    )
    sample_hot, n_sh = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
        user_seen_items=test_seen,
    )
    full_cold, n_fc = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen,
    )
    full_hot, n_fh = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen,
    )
    full_cold_item_macro, n_fc_item_macro = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, average_mode="item_macro",
    )
    full_hot_item_macro, n_fh_item_macro = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, average_mode="item_macro",
    )

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
    full_cold_item_macro = full_cold_item_macro or {}
    full_hot_item_macro = full_hot_item_macro or {}
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
        title="ContentProfile Static HIN",
    )

    out = {
        "model": "ContentProfile",
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
        "best_val_full_cold_n10": val_key,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "note": "Mean normalized train-item content profile; no validation-tuned parameters.",
    }
    result_path = static_result_path("content_profile_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
