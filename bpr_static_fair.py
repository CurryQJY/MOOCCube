"""
bpr_static.py — Pure BPR Matrix Factorization baseline under static 8/1/1 split.

数据格式与 lightgcn_static_hin.py 一致, 但模型为纯 ID-based MF (无图传播, 无 content):
  - processed_data_hin/{meta.json, stream_data.pkl, content_emb.pt}
  - 8/1/1 random split
  - cold_threshold=5, sampled (1+200) 与 full ranking 同时输出, cold/hot 分组
  - 验证集选择 best epoch (full_cold_N@10), 评估 test
"""

import copy
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.n_epochs = int(os.environ.get("BPR_STATIC_EPOCHS", "200"))
        self.batch_size = int(os.environ.get("BPR_BATCH_SIZE", "4096"))
        self.eval_interval = int(os.environ.get("BPR_EVAL_INTERVAL", "10"))

        self.cold_threshold = int(os.environ.get("BPR_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("BPR_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("BPR_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("BPR_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("BPR_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("BPR_STATIC_VAL_RATIO", "0.1"))


class BPRMFModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_emb.weight, self.item_emb.weight


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, _ = load_hin_processed(data_dir)
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

    train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRMFModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: BPR-MF static | device={device} | epochs={cfg.n_epochs} | emb_dim={cfg.emb_dim}")

    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    k_list = [5, 10, 20]

    n_train = train_users_t.numel()
    for epoch in range(cfg.n_epochs):
        model.train()
        train_neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
        train_neg_t = torch.tensor(train_neg_np, dtype=torch.long, device=device)

        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            u_batch = train_users_t[idx]
            p_batch = train_pos_t[idx]
            n_batch = train_neg_t[idx]

            optimizer.zero_grad()
            z_u, z_i = model.all_embeddings()
            loss = compute_bpr_loss(
                z_u,
                z_i,
                u_batch,
                p_batch,
                n_batch,
                reg_weight=cfg.reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)

        do_eval = ((epoch + 1) % cfg.eval_interval == 0) or (epoch + 1 == cfg.n_epochs)
        val_key = float("nan")
        if do_eval:
            model.eval()
            with torch.no_grad():
                z_u, z_i = model.all_embeddings()
                all_u = F.normalize(z_u, dim=1)
                all_i = F.normalize(z_i, dim=1)
                get_user_fn = lambda b: all_u[b["u"]]
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
                if val_key > best_val:
                    best_val = val_key
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
            print(
                f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f} | "
                f"val_full_cold_N@10={val_key:.4f}"
            )
        else:
            print(f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restore best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    model.eval()
    with torch.no_grad():
        z_u, z_i = model.all_embeddings()
        all_u = F.normalize(z_u, dim=1)
        all_i = F.normalize(z_i, dim=1)
        get_user_fn = lambda b: all_u[b["u"]]

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

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
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
        title="BPR-MF Static HIN",
    )

    out = {
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
    }
    result_path = static_result_path("bpr_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
