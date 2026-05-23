"""
ColdRec-derived CGRC baseline for static item cold-start.

This is a local adaptation of the CGRC implementation in YuanchenBei/ColdRec
for the paper "Content-based Graph Reconstruction for Cold-start Item
Recommendation" (SIGIR 2024). It keeps this repository's strict static split
and full-ranking evaluator, while following CGRC's main training and inference
path:

1. mask warm items as pseudo-cold, reconstruct their user-item edges from item
   content and propagated user states;
2. train a ranking loss on the original training graph;
3. at inference, reconstruct top-K user edges for real cold items and run
   LightGCN on the reconstructed graph G_hat.
"""

import copy
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
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
from lightgcn_static_hin import prepare_train_cache
from baseline_checkpoint import checkpoint_config, maybe_resume_checkpoint, save_checkpoint


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("CGRC_EMB_DIM", "64"))
        self.mlp_hidden = int(os.environ.get("CGRC_MLP_HIDDEN", "64"))
        self.layers_gprime = int(os.environ.get("CGRC_LAYERS_GPRIME", "2"))
        self.layers_full = int(os.environ.get("CGRC_LAYERS_FULL", "2"))
        self.layers_ghat = int(os.environ.get("CGRC_LAYERS_GHAT", "2"))

        self.mask_rho = float(
            os.environ.get("CGRC_MASK_RHO", os.environ.get("CGRC_MASK_ITEM_RATIO", "0.3"))
        )
        self.recon_topk = int(os.environ.get("CGRC_RECON_TOPK", "20"))
        self.lambda_e = float(os.environ.get("CGRC_LAMBDA_E", "1.0"))
        self.tau = float(os.environ.get("CGRC_TAU", "0.5"))
        self.le_max_edges = int(os.environ.get("CGRC_LE_MAX_EDGES", "4096"))
        self.ranking_neg_per_user = int(os.environ.get("CGRC_RANKING_NEG_PER_USER", "32"))
        self.recon_user_chunk = int(os.environ.get("CGRC_RECON_USER_CHUNK", "4096"))

        self.lr = float(os.environ.get("CGRC_LR", "1e-3"))
        self.reg_weight = float(os.environ.get("CGRC_REG", "1e-4"))
        self.grad_clip = float(os.environ.get("CGRC_GRAD_CLIP", "0.0"))
        self.n_epochs = int(os.environ.get("CGRC_STATIC_EPOCHS", "50"))
        self.batch_size = int(os.environ.get("CGRC_BATCH_SIZE", "4096"))

        self.cold_threshold = int(
            os.environ.get("CGRC_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5"))
        )
        self.eval_n_neg = int(
            os.environ.get("CGRC_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200"))
        )
        self.static_seed = int(
            os.environ.get("CGRC_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025"))
        )
        self.seed = int(os.environ.get("CGRC_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("CGRC_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("CGRC_STATIC_VAL_RATIO", "0.1"))
        self.ckpt = checkpoint_config("CGRC")


class CGRCNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        content_dim: int,
        emb_dim: int,
        mlp_hidden: int,
        item_content: torch.Tensor,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.register_buffer("item_content", item_content.float())
        self.user_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, emb_dim)))
        self.item_lin = nn.Linear(content_dim, emb_dim, bias=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def item_x(self) -> torch.Tensor:
        return self.item_lin(self.item_content)

    def edge_logits_broadcast(
        self,
        h_u_bar: torch.Tensor,
        x_cold: torch.Tensor,
        cold_ids: torch.Tensor,
    ) -> torch.Tensor:
        cold_x = x_cold[cold_ids]
        n_user, dim = h_u_bar.shape
        n_cold = cold_x.shape[0]
        user_part = h_u_bar.unsqueeze(1).expand(n_user, n_cold, dim).reshape(n_user * n_cold, dim)
        item_part = cold_x.unsqueeze(0).expand(n_user, n_cold, dim).reshape(n_user * n_cold, dim)
        return self.edge_mlp(torch.cat([user_part, item_part], dim=1)).view(n_user, n_cold)


def _normalize_graph_mat(adj_mat: sp.spmatrix) -> sp.csr_matrix:
    adj_mat = adj_mat.tocsr()
    rowsum = np.asarray(adj_mat.sum(1)).flatten()
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    np.power(rowsum, -0.5, out=d_inv, where=rowsum != 0)
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocsr()


def _sparse_adj_tensor(adj_mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = adj_mat.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False)).float()
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def _build_interaction_csr(train_df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    users = train_df["u_idx"].to_numpy(np.int64, copy=False)
    items = train_df["i_idx"].to_numpy(np.int64, copy=False)
    data = np.ones(users.size, dtype=np.float32)
    mat = sp.csr_matrix((data, (users, items)), shape=(n_users, n_items), dtype=np.float32)
    mat.eliminate_zeros()
    if mat.nnz:
        mat.data = np.ones_like(mat.data, dtype=np.float32)
    return mat


def _bip_adj_from_R(R: sp.csr_matrix, n_users: int, n_items: int) -> sp.csr_matrix:
    R_coo = R.tocoo()
    rows = R_coo.row.astype(np.int64, copy=False)
    cols = R_coo.col.astype(np.int64, copy=False)
    row_idx = np.concatenate([rows, cols + n_users])
    col_idx = np.concatenate([cols + n_users, rows])
    data = np.ones(row_idx.size, dtype=np.float32)
    return sp.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_users + n_items, n_users + n_items),
        dtype=np.float32,
    )


def _drop_edges_to_items(R_csr: sp.csr_matrix, cold_items: Iterable[int]) -> sp.csr_matrix:
    cold_arr = np.asarray(list(cold_items), dtype=np.int64)
    if cold_arr.size < 1:
        return R_csr.tocsr(copy=True)
    coo = R_csr.tocoo()
    mask = ~np.isin(coo.col, cold_arr)
    return sp.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])),
        shape=R_csr.shape,
        dtype=np.float32,
    )


def _add_edges_to_R(R_csr: sp.csr_matrix, pairs: Sequence[Tuple[int, int]]) -> sp.csr_matrix:
    if not pairs:
        return R_csr.tocsr(copy=True)
    coo = R_csr.tocoo()
    pair_users = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pair_items = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    row = np.concatenate([coo.row.astype(np.int64, copy=False), pair_users])
    col = np.concatenate([coo.col.astype(np.int64, copy=False), pair_items])
    data = np.concatenate([coo.data.astype(np.float32, copy=False), np.ones(len(pairs), dtype=np.float32)])
    out = sp.csr_matrix((data, (row, col)), shape=R_csr.shape, dtype=np.float32)
    out.eliminate_zeros()
    if out.nnz:
        out.data = np.ones_like(out.data, dtype=np.float32)
    return out


def _lightgcn_mean_all_layers(
    adj_t: torch.Tensor,
    user_emb: torch.Tensor,
    item_x: torch.Tensor,
    n_users: int,
    n_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ego = torch.cat([user_emb, item_x], dim=0)
    embs = [ego]
    h = ego
    for _ in range(n_layers):
        h = torch.sparse.mm(adj_t, h)
        embs.append(h)
    out = torch.stack(embs, dim=1).mean(dim=1)
    return out[:n_users], out[n_users:]


def _propagate_gprime_frozen_cold(
    adj_t: torch.Tensor,
    user_emb: torch.Tensor,
    item_x: torch.Tensor,
    n_users: int,
    n_layers: int,
    cold_item_idx: torch.Tensor,
) -> List[torch.Tensor]:
    ego = torch.cat([user_emb, item_x], dim=0)
    out = [ego]
    h = ego
    cold_rows = cold_item_idx + n_users if cold_item_idx.numel() > 0 else None
    for _ in range(n_layers):
        h = torch.sparse.mm(adj_t, h)
        if cold_rows is not None:
            h[cold_rows] = item_x[cold_item_idx]
        out.append(h)
    return out


def _user_mean_layers_1_to_L(layer_list: Sequence[torch.Tensor], n_users: int, n_layers: int) -> torch.Tensor:
    if n_layers <= 0:
        return layer_list[0][:n_users]
    stacks = torch.stack([layer_list[layer][:n_users] for layer in range(1, n_layers + 1)], dim=0)
    return stacks.mean(dim=0)


def _build_user_rated(train_df: pd.DataFrame, n_users: int) -> List[set]:
    user_rated = [set() for _ in range(n_users)]
    for uid, iid in zip(train_df["u_idx"].values, train_df["i_idx"].values):
        user_rated[int(uid)].add(int(iid))
    return user_rated


def _sample_cold_items(train_item_pool: np.ndarray, rho: float, device: torch.device) -> torch.Tensor:
    if rho <= 0.0 or train_item_pool.size < 1:
        return torch.empty(0, dtype=torch.long, device=device)
    mask = np.random.rand(train_item_pool.size) < rho
    chosen = train_item_pool[mask]
    if chosen.size < 1:
        chosen = np.random.choice(train_item_pool, size=1, replace=False)
    return torch.tensor(chosen, dtype=torch.long, device=device)


def _masked_edges(
    R_coo_row: np.ndarray,
    R_coo_col: np.ndarray,
    cold_ids_cpu: np.ndarray,
) -> List[Tuple[int, int]]:
    if cold_ids_cpu.size < 1:
        return []
    mask = np.isin(R_coo_col, cold_ids_cpu)
    if not mask.any():
        return []
    return list(zip(R_coo_row[mask].tolist(), R_coo_col[mask].tolist()))


def _reconstruction_loss(
    logits: torch.Tensor,
    cold_ids: torch.Tensor,
    masked_edges: Sequence[Tuple[int, int]],
    u_indices: torch.Tensor,
    user_rated: Sequence[set],
) -> torch.Tensor:
    if not masked_edges:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    cold_list = [int(x) for x in cold_ids.detach().cpu().tolist()]
    col_of = {cid: col for col, cid in enumerate(cold_list)}
    row_map = {int(u_indices[row].item()): row for row in range(u_indices.numel())}
    by_user: Dict[int, List[int]] = defaultdict(list)
    for uid, iid in masked_edges:
        by_user[int(uid)].append(int(iid))

    valid_rows = []
    for row in range(u_indices.numel()):
        uid = int(u_indices[row].item())
        rated = user_rated[uid]
        valid_rows.append([cid not in rated for cid in cold_list])
    valid_mask = torch.tensor(valid_rows, device=logits.device, dtype=torch.bool)
    if not valid_mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    neg_inf = torch.finfo(logits.dtype).min
    log_denom = torch.logsumexp(logits.masked_fill(~valid_mask, neg_inf), dim=1)

    row_inds = []
    col_inds = []
    for uid, items in by_user.items():
        row = row_map.get(uid)
        if row is None or not bool(valid_mask[row].any().item()):
            continue
        for iid in items:
            col = col_of.get(iid)
            if col is not None:
                row_inds.append(row)
                col_inds.append(col)

    if not row_inds:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    row_t = torch.tensor(row_inds, device=logits.device, dtype=torch.long)
    col_t = torch.tensor(col_inds, device=logits.device, dtype=torch.long)
    return -(logits[row_t, col_t] - log_denom[row_t]).mean()


def _ranking_loss(
    z_u: torch.Tensor,
    z_i: torch.Tensor,
    u_idx: Sequence[int],
    i_pos: Sequence[int],
    B_list: Sequence[int],
    user_rated: Sequence[set],
    tau: float,
) -> torch.Tensor:
    if not u_idx or not B_list:
        return torch.zeros((), device=z_u.device, dtype=z_u.dtype)

    u_t = torch.tensor(u_idx, dtype=torch.long, device=z_u.device)
    B_t = torch.tensor(B_list, dtype=torch.long, device=z_u.device)
    sim = torch.matmul(z_u[u_t], z_i[B_t].transpose(0, 1)) / tau

    B_map = {int(item): pos for pos, item in enumerate(B_list)}
    pos_cols = torch.tensor([B_map.get(int(iid), -1) for iid in i_pos], dtype=torch.long, device=z_u.device)

    neg_rows = []
    for uid in u_idx:
        rated = user_rated[int(uid)]
        neg_rows.append([int(item) not in rated for item in B_list])
    neg_mask = torch.tensor(neg_rows, dtype=torch.bool, device=z_u.device)

    valid = (pos_cols >= 0) & neg_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=z_u.device, dtype=z_u.dtype)

    neg_inf = torch.finfo(sim.dtype).min
    log_denom = torch.logsumexp(sim.masked_fill(~neg_mask, neg_inf), dim=1)
    row_idx = torch.arange(len(u_idx), dtype=torch.long, device=z_u.device)
    return -(sim[row_idx[valid], pos_cols[valid]] - log_denom[valid]).mean()


def _l2_reg_loss(reg_weight: float, *embs: torch.Tensor) -> torch.Tensor:
    if reg_weight <= 0.0:
        return torch.zeros((), device=embs[0].device, dtype=embs[0].dtype)
    reg = torch.zeros((), device=embs[0].device, dtype=embs[0].dtype)
    for emb in embs:
        reg = reg + torch.norm(emb, p=2) / max(1, emb.shape[0])
    return reg * reg_weight


def _iter_cgrc_batches(
    train_users: np.ndarray,
    train_pos: np.ndarray,
    user_rated: Sequence[set],
    user_neg_pool: Dict[int, np.ndarray],
    train_item_pool: np.ndarray,
    batch_size: int,
    ranking_neg_per_user: int,
):
    order = np.random.permutation(train_users.size)
    for start in range(0, order.size, batch_size):
        idx = order[start:start + batch_size]
        u_idx = train_users[idx].astype(np.int64, copy=False).tolist()
        i_idx = train_pos[idx].astype(np.int64, copy=False).tolist()
        B_set = set(int(item) for item in i_idx)
        if ranking_neg_per_user > 0:
            for uid in u_idx:
                pool = user_neg_pool.get(int(uid), train_item_pool)
                if pool.size < 1:
                    continue
                chosen = np.random.choice(
                    pool,
                    size=ranking_neg_per_user,
                    replace=pool.size < ranking_neg_per_user,
                )
                B_set.update(int(x) for x in chosen.tolist())
        yield u_idx, i_idx, list(B_set)


def _cold_items_in(df: pd.DataFrame, cold_threshold: int) -> np.ndarray:
    cold_df = df[df["popularity"].astype(int) < int(cold_threshold)]
    if cold_df.empty:
        return np.empty(0, dtype=np.int64)
    return np.sort(cold_df["i_idx"].astype(np.int64).unique())


def _topk_users_for_cold(
    model: CGRCNet,
    h_u_bar: torch.Tensor,
    x_all: torch.Tensor,
    cold_t: torch.Tensor,
    topk: int,
    user_chunk: int,
) -> torch.Tensor:
    k = min(topk, h_u_bar.shape[0])
    n_cold = cold_t.numel()
    if k <= 0 or n_cold < 1:
        return torch.empty((0, n_cold), dtype=torch.long, device=h_u_bar.device)

    best_vals = torch.empty((0, n_cold), dtype=h_u_bar.dtype, device=h_u_bar.device)
    best_rows = torch.empty((0, n_cold), dtype=torch.long, device=h_u_bar.device)
    chunk = max(1, int(user_chunk))
    for start in range(0, h_u_bar.shape[0], chunk):
        end = min(start + chunk, h_u_bar.shape[0])
        logits = model.edge_logits_broadcast(h_u_bar[start:end], x_all, cold_t)
        row_ids = torch.arange(start, end, dtype=torch.long, device=h_u_bar.device).view(-1, 1)
        row_ids = row_ids.expand(-1, n_cold)

        cand_vals = torch.cat([best_vals, logits], dim=0)
        cand_rows = torch.cat([best_rows, row_ids], dim=0)
        keep = min(k, cand_vals.shape[0])
        best_vals, pos = torch.topk(cand_vals, k=keep, dim=0)
        best_rows = cand_rows.gather(0, pos)
    return best_rows


def _build_ghat_embeddings(
    model: CGRCNet,
    cfg: Config,
    R_base: sp.csr_matrix,
    sparse_full: torch.Tensor,
    device: torch.device,
    eval_cold_items: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    x_all = model.item_x()
    empty_cold = torch.empty(0, dtype=torch.long, device=device)
    layers = _propagate_gprime_frozen_cold(
        sparse_full,
        model.user_emb,
        x_all,
        cfg.n_users,
        cfg.layers_gprime,
        empty_cold,
    )
    h_u_bar = _user_mean_layers_1_to_L(layers, cfg.n_users, cfg.layers_gprime)

    cold_np = np.asarray(eval_cold_items, dtype=np.int64)
    pairs = []
    if cold_np.size > 0 and cfg.recon_topk > 0:
        cold_t = torch.tensor(cold_np, dtype=torch.long, device=device)
        top_rows = _topk_users_for_cold(
            model,
            h_u_bar,
            x_all,
            cold_t,
            cfg.recon_topk,
            cfg.recon_user_chunk,
        )
        top_rows_cpu = top_rows.detach().cpu().numpy()
        for col, item_id in enumerate(cold_np.tolist()):
            for row in top_rows_cpu[:, col].tolist():
                pairs.append((int(row), int(item_id)))

    R_hat = _add_edges_to_R(R_base, pairs)
    adj_hat = _normalize_graph_mat(_bip_adj_from_R(R_hat, cfg.n_users, cfg.n_items))
    sparse_hat = _sparse_adj_tensor(adj_hat, device)
    z_u, z_i = _lightgcn_mean_all_layers(sparse_hat, model.user_emb, x_all, cfg.n_users, cfg.layers_ghat)
    return torch.nn.functional.normalize(z_u, dim=1), torch.nn.functional.normalize(z_i, dim=1), len(pairs)


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

    train_users_np, train_pos_np, _, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    train_item_pool = np.unique(train_pos_np).astype(np.int64, copy=False)
    user_rated = _build_user_rated(train_df, cfg.n_users)
    R_base = _build_interaction_csr(train_df, cfg.n_users, cfg.n_items)
    R_coo = R_base.tocoo()
    R_coo_row = R_coo.row.astype(np.int64, copy=False)
    R_coo_col = R_coo.col.astype(np.int64, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGRCNet(
        cfg.n_users,
        cfg.n_items,
        cfg.content_dim,
        cfg.emb_dim,
        cfg.mlp_hidden,
        content_emb,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    sparse_full = _sparse_adj_tensor(
        _normalize_graph_mat(_bip_adj_from_R(R_base, cfg.n_users, cfg.n_items)),
        device,
    )
    val_cold_items = _cold_items_in(val_df, cfg.cold_threshold)
    test_cold_items = _cold_items_in(test_df, cfg.cold_threshold)

    print(
        "Model: CGRC static (ColdRec-derived) | "
        f"device={device} | epochs={cfg.n_epochs} | batch={cfg.batch_size} | "
        f"rho={cfg.mask_rho} | topk={cfg.recon_topk} | "
        f"val_cold_items={val_cold_items.size} | test_cold_items={test_cold_items.size}"
    )

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_recon_edges = 0
    k_list = [5, 10, 20]
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.ckpt, model, optimizer, device)
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)
    best_recon_edges = int(ckpt_state.get("best_recon_edges", best_recon_edges))

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        loss_sum = 0.0
        loss_e_sum = 0.0
        loss_r_sum = 0.0
        n_batches = 0

        for u_idx, i_idx, B_list in _iter_cgrc_batches(
            train_users_np,
            train_pos_np,
            user_rated,
            user_neg_pool,
            train_item_pool,
            cfg.batch_size,
            cfg.ranking_neg_per_user,
        ):
            x_all = model.item_x()
            cold_ids = _sample_cold_items(train_item_pool, cfg.mask_rho, device)
            loss_e = torch.zeros((), device=device, dtype=torch.float32)

            if cold_ids.numel() > 0:
                cold_cpu = cold_ids.detach().cpu().numpy()
                edges = _masked_edges(R_coo_row, R_coo_col, cold_cpu)
                if len(edges) > cfg.le_max_edges:
                    keep_idx = np.random.choice(len(edges), size=cfg.le_max_edges, replace=False)
                    edges = [edges[int(pos)] for pos in keep_idx]

                if edges:
                    R_masked = _drop_edges_to_items(R_base, set(int(x) for x in cold_cpu.tolist()))
                    sparse_masked = _sparse_adj_tensor(
                        _normalize_graph_mat(_bip_adj_from_R(R_masked, cfg.n_users, cfg.n_items)),
                        device,
                    )
                    layers = _propagate_gprime_frozen_cold(
                        sparse_masked,
                        model.user_emb,
                        x_all,
                        cfg.n_users,
                        cfg.layers_gprime,
                        cold_ids,
                    )
                    h_u_bar = _user_mean_layers_1_to_L(layers, cfg.n_users, cfg.layers_gprime)
                    needed_users = sorted({int(uid) for uid, _ in edges})
                    needed_t = torch.tensor(needed_users, dtype=torch.long, device=device)
                    logits = model.edge_logits_broadcast(h_u_bar[needed_t], x_all, cold_ids)
                    loss_e = _reconstruction_loss(logits, cold_ids, edges, needed_t, user_rated)

            z_u, z_i = _lightgcn_mean_all_layers(
                sparse_full,
                model.user_emb,
                x_all,
                cfg.n_users,
                cfg.layers_full,
            )
            loss_r = _ranking_loss(z_u, z_i, u_idx, i_idx, B_list, user_rated, cfg.tau)
            u_t = torch.tensor(u_idx, dtype=torch.long, device=device)
            i_t = torch.tensor(i_idx, dtype=torch.long, device=device)
            reg = _l2_reg_loss(cfg.reg_weight, model.user_emb[u_t], x_all[i_t])
            loss = cfg.lambda_e * loss_e + loss_r + reg

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            loss_sum += float(loss.item())
            loss_e_sum += float(loss_e.item())
            loss_r_sum += float(loss_r.item())
            n_batches += 1

        model.eval()
        improved = False
        with torch.no_grad():
            all_u_val, all_i_val, recon_edges = _build_ghat_embeddings(
                model,
                cfg,
                R_base,
                sparse_full,
                device,
                val_cold_items,
            )
            get_user_fn = lambda batch: all_u_val[batch["u"]]
            val_full_cold, _ = evaluate_embedding_ranker(
                val_loader,
                device=device,
                n_items=cfg.n_items,
                cold_threshold=cfg.cold_threshold,
                get_user_vectors_fn=get_user_fn,
                all_item_vectors=all_i_val,
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
                best_recon_edges = recon_edges
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                improved = True
        if cfg.ckpt.save and improved:
            save_checkpoint(
                cfg.ckpt,
                "best.pt",
                epoch + 1,
                model,
                optimizer,
                best_state=best_state,
                extra={
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "best_recon_edges": best_recon_edges,
                },
            )

        denom = max(1, n_batches)
        print(
            f"Epoch [{epoch + 1}/{cfg.n_epochs}] "
            f"loss={loss_sum / denom:.4f} | L_E={loss_e_sum / denom:.4f} | "
            f"L_R={loss_r_sum / denom:.4f} | val_full_cold_N@10={val_key:.4f} | "
            f"val_recon_edges={recon_edges}"
        )
        if cfg.ckpt.save:
            save_checkpoint(
                cfg.ckpt,
                "latest.pt",
                epoch + 1,
                model,
                optimizer,
                best_state=best_state,
                extra={
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "best_recon_edges": best_recon_edges,
                },
            )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(
            f"Restore best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}, "
            f"val_recon_edges={best_recon_edges}"
        )

    model.eval()
    with torch.no_grad():
        all_u, all_i, test_recon_edges = _build_ghat_embeddings(
            model,
            cfg,
            R_base,
            sparse_full,
            device,
            test_cold_items,
        )
        get_user_fn = lambda batch: all_u[batch["u"]]

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
    metrics_keys = [f"{metric}@{k}" for metric in ["R", "N"] for k in k_list]

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
        title="CGRC Static HIN (ColdRec-derived)",
    )

    out = {
        "model": "CGRC",
        "source": "ColdRec-derived third-party adaptation",
        "protocol": "static_item_cold",
        "best_metric": "cold",
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
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_val_recon_edges": best_recon_edges,
        "test_recon_edges": test_recon_edges,
        "emb_dim": cfg.emb_dim,
        "mlp_hidden": cfg.mlp_hidden,
        "layers_gprime": cfg.layers_gprime,
        "layers_full": cfg.layers_full,
        "layers_ghat": cfg.layers_ghat,
        "mask_rho": cfg.mask_rho,
        "recon_topk": cfg.recon_topk,
        "lambda_e": cfg.lambda_e,
        "tau": cfg.tau,
        "le_max_edges": cfg.le_max_edges,
        "ranking_neg_per_user": cfg.ranking_neg_per_user,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "checkpoint_dir": cfg.ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
    }
    result_path = static_result_path("cgrc_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
