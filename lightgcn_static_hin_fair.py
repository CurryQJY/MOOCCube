import copy
import os
from collections import defaultdict
from typing import Dict, Tuple

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


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("LIGHTGCN_EMB_DIM", "128"))
        self.hidden_dim = int(os.environ.get("LIGHTGCN_HIDDEN_DIM", "256"))
        self.n_layers = int(os.environ.get("LIGHTGCN_N_LAYERS", "2"))
        self.content_weight = float(os.environ.get("LIGHTGCN_CONTENT_WEIGHT", "0.35"))

        self.lr = float(os.environ.get("LIGHTGCN_LR", "1e-3"))
        self.reg_weight = float(os.environ.get("LIGHTGCN_REG", "1e-4"))
        self.n_epochs = int(os.environ.get("LIGHTGCN_STATIC_EPOCHS", "300"))
        self.batch_size = int(os.environ.get("LIGHTGCN_BATCH_SIZE", "4096"))
        self.eval_interval = int(os.environ.get("LIGHTGCN_EVAL_INTERVAL", "10"))

        self.cold_threshold = int(os.environ.get("LIGHTGCN_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("LIGHTGCN_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("LIGHTGCN_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("LIGHTGCN_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("LIGHTGCN_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("LIGHTGCN_STATIC_VAL_RATIO", "0.1"))


def build_norm_adj(
    n_users: int,
    n_items: int,
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    size = n_users + n_items
    if u_idx.size < 1:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, val, (size, size), device=device).coalesce()

    u_t = torch.tensor(u_idx, dtype=torch.long, device=device)
    i_t = torch.tensor(i_idx, dtype=torch.long, device=device) + n_users

    row = torch.cat([u_t, i_t], dim=0)
    col = torch.cat([i_t, u_t], dim=0)
    val = torch.ones(row.numel(), dtype=torch.float32, device=device)

    deg = torch.zeros(size, dtype=torch.float32, device=device)
    deg.scatter_add_(0, row, val)
    deg_inv = deg.clamp_min(1e-12).pow(-0.5)
    norm_val = deg_inv[row] * val * deg_inv[col]

    idx = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(idx, norm_val, (size, size), device=device).coalesce()


def prepare_train_cache(
    train_df: pd.DataFrame,
    n_items: int
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    users = train_df["u_idx"].to_numpy(np.int64, copy=True)
    pos_items = train_df["i_idx"].to_numpy(np.int64, copy=True)

    user_rows = defaultdict(list)
    for row, uid in enumerate(users.tolist()):
        user_rows[int(uid)].append(row)
    user_rows = {uid: np.asarray(rows, dtype=np.int64) for uid, rows in user_rows.items()}

    user_seen = build_user_seen(train_df)
    train_item_pool = np.unique(pos_items).astype(np.int64, copy=False)
    if train_item_pool.size < 1:
        raise ValueError("Cannot prepare training cache from an empty training item pool")
    user_neg_pool = {}
    for uid in user_rows.keys():
        seen = user_seen.get(uid, set())
        if len(seen) >= train_item_pool.size:
            pool = train_item_pool
        else:
            seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen)) if seen else np.empty(0, dtype=np.int64)
            pool = np.setdiff1d(train_item_pool, seen_arr, assume_unique=False)
            if pool.size < 1:
                pool = train_item_pool
        user_neg_pool[uid] = pool

    return users, pos_items, user_rows, user_neg_pool


def sample_negatives(
    pos_items: np.ndarray,
    user_rows: Dict[int, np.ndarray],
    user_neg_pool: Dict[int, np.ndarray],
    n_items: int
) -> np.ndarray:
    if n_items <= 1:
        return np.zeros_like(pos_items)

    neg = np.empty_like(pos_items)
    for uid, rows in user_rows.items():
        pool = user_neg_pool.get(uid)
        if pool is None or pool.size < 1:
            pool = np.arange(n_items, dtype=np.int64)
        chosen = np.random.choice(pool, size=rows.size, replace=True)
        same = chosen == pos_items[rows]
        if same.any():
            same_locs = np.where(same)[0]
            for loc in same_locs:
                row = rows[loc]
                alt_pool = pool[pool != pos_items[row]]
                if alt_pool.size < 1:
                    alt_pool = pool
                chosen[loc] = np.random.choice(alt_pool)
        neg[rows] = chosen
    return neg


def compute_bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    u_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
    reg_weight: float
) -> torch.Tensor:
    z_u = user_emb[u_idx]
    z_p = item_emb[pos_idx]
    z_n = item_emb[neg_idx]

    pos_score = (z_u * z_p).sum(dim=1)
    neg_score = (z_u * z_n).sum(dim=1)
    loss_rec = -F.logsigmoid(pos_score - neg_score).mean()

    loss_reg = (
        z_u.pow(2).sum(dim=1)
        + z_p.pow(2).sum(dim=1)
        + z_n.pow(2).sum(dim=1)
    ).mean()
    return loss_rec + reg_weight * loss_reg


class LightGCNStaticModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )

    def get_base_item_bank(self) -> torch.Tensor:
        item_id = self.item_emb.weight
        item_con = self.content_proj(self.item_con_emb.weight)
        return item_id + self.cfg.content_weight * item_con

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat([self.user_emb.weight, self.get_base_item_bank()], dim=0)
        outs = [ego]
        for _ in range(self.cfg.n_layers):
            ego = torch.sparse.mm(norm_adj, ego)
            outs.append(ego)
        out = torch.stack(outs, dim=0).mean(dim=0)
        return out[:self.cfg.n_users], out[self.cfg.n_users:]


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"], content_dim=content_emb.shape[1])
    setup_seed(cfg.seed)

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio
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
    model = LightGCNStaticModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_adj = build_norm_adj(
        cfg.n_users,
        cfg.n_items,
        train_df["u_idx"].to_numpy(np.int64),
        train_df["i_idx"].to_numpy(np.int64),
        device,
    )
    print(f"Model: LightGCN static | device={device} | epochs={cfg.n_epochs}")

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
            z_u, z_i = model.propagate(train_adj)
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
                all_u, all_i = model.propagate(train_adj)
                all_u = F.normalize(all_u, dim=1)
                all_i = F.normalize(all_i, dim=1)
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
                    user_seen_items=train_seen
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
        all_u, all_i = model.propagate(train_adj)
        all_u = F.normalize(all_u, dim=1)
        all_i = F.normalize(all_i, dim=1)
        get_user_fn = lambda b: all_u[b["u"]]

        sample_cold, n_sc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
            user_seen_items=test_seen
        )
        sample_hot, n_sh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
            user_seen_items=test_seen
        )
        full_cold, n_fc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
            user_seen_items=test_seen
        )
        full_hot, n_fh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_i,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
            user_seen_items=test_seen
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
        title="LightGCN Static HIN"
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
    result_path = static_result_path("lightgcn_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
