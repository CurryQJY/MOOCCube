"""
LightGCL static item-cold adaptation for the shared HIN split.

The official LightGCL code is pulled under third_party/LightGCL. Its public
runner expects prebuilt scipy matrices for standard collaborative filtering
datasets. This script keeps the local static item-cold protocol and adapts the
official LightGCL training objective:

  - normalized user-item graph propagation;
  - low-rank SVD graph contrastive view;
  - BPR loss + SVD contrastive loss;
  - content projection added to the item initial representation so
    zero-interaction cold items can be ranked.
"""

import copy
import os
from typing import Dict, Optional, Tuple

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
from lightgcn_static_hin import prepare_train_cache, sample_negatives
from baseline_checkpoint import checkpoint_config, maybe_resume_checkpoint, save_checkpoint


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("LIGHTGCL_EMB_DIM", "64"))
        self.hidden_dim = int(os.environ.get("LIGHTGCL_HIDDEN_DIM", "128"))
        self.n_layers = int(os.environ.get("LIGHTGCL_N_LAYERS", "2"))
        self.svd_rank = int(os.environ.get("LIGHTGCL_SVD_RANK", "5"))
        self.content_weight = float(os.environ.get("LIGHTGCL_CONTENT_WEIGHT", "1.0"))

        self.lr = float(os.environ.get("LIGHTGCL_LR", "1e-3"))
        self.dropout = float(os.environ.get("LIGHTGCL_DROPOUT", "0.0"))
        self.temp = float(os.environ.get("LIGHTGCL_TEMP", "0.2"))
        self.lambda1 = float(os.environ.get("LIGHTGCL_LAMBDA1", "0.2"))
        self.lambda2 = float(os.environ.get("LIGHTGCL_LAMBDA2", "1e-7"))

        self.n_epochs = int(os.environ.get("LIGHTGCL_STATIC_EPOCHS", "80"))
        self.eval_interval = int(os.environ.get("LIGHTGCL_EVAL_INTERVAL", "5"))
        self.batch_size = int(os.environ.get("LIGHTGCL_BATCH_SIZE", "1024"))
        self.eval_batch_size = int(os.environ.get("LIGHTGCL_EVAL_BATCH_SIZE", "4096"))
        self.bank_batch_size = int(os.environ.get("LIGHTGCL_BANK_BATCH_SIZE", "32768"))

        self.cold_threshold = int(os.environ.get("LIGHTGCL_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("LIGHTGCL_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("LIGHTGCL_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("LIGHTGCL_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("LIGHTGCL_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("LIGHTGCL_STATIC_VAL_RATIO", "0.1"))
        self.ckpt = checkpoint_config("LIGHTGCL")


def sparse_dropout(mat: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout <= 0.0:
        return mat
    mat = mat.coalesce()
    values = F.dropout(mat.values(), p=dropout, training=True)
    return torch.sparse_coo_tensor(mat.indices(), values, mat.size(), device=mat.device).coalesce()


def build_norm_adj(
    n_users: int,
    n_items: int,
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if u_idx.size < 1:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, val, (n_users, n_items), device=device).coalesce()

    u_t = torch.tensor(u_idx, dtype=torch.long, device=device)
    i_t = torch.tensor(i_idx, dtype=torch.long, device=device)
    val = torch.ones(u_t.numel(), dtype=torch.float32, device=device)

    deg_u = torch.zeros(n_users, dtype=torch.float32, device=device)
    deg_i = torch.zeros(n_items, dtype=torch.float32, device=device)
    deg_u.scatter_add_(0, u_t, val)
    deg_i.scatter_add_(0, i_t, val)
    norm_val = deg_u[u_t].clamp_min(1e-8).pow(-0.5) * deg_i[i_t].clamp_min(1e-8).pow(-0.5)
    idx = torch.stack([u_t, i_t], dim=0)
    return torch.sparse_coo_tensor(idx, norm_val, (n_users, n_items), device=device).coalesce()


class LightGCLStaticFairModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(cfg.n_users, cfg.emb_dim)))
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(cfg.n_items, cfg.emb_dim)))
        self.register_buffer("content_features", content_emb.float())
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.5),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )
        for module in self.content_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def base_items(self) -> torch.Tensor:
        return self.item_embedding + self.cfg.content_weight * self.content_proj(self.content_features)

    def propagate(
        self,
        adj_norm: torch.Tensor,
        u_mul_s: torch.Tensor,
        v_mul_s: torch.Tensor,
        ut: torch.Tensor,
        vt: torch.Tensor,
        training_view: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e_u = self.user_embedding
        e_i = self.base_items()
        e_u_list = [e_u]
        e_i_list = [e_i]
        g_u_list = [e_u]
        g_i_list = [e_i]

        for _ in range(self.cfg.n_layers):
            adj = sparse_dropout(adj_norm, self.cfg.dropout) if training_view else adj_norm
            z_u = torch.sparse.mm(adj, e_i)
            z_i = torch.sparse.mm(adj.t(), e_u)

            g_u = u_mul_s @ (vt @ e_i)
            g_i = v_mul_s @ (ut @ e_u)

            e_u, e_i = z_u, z_i
            e_u_list.append(e_u)
            e_i_list.append(e_i)
            g_u_list.append(g_u)
            g_i_list.append(g_i)

        # The official implementation sums layer outputs instead of averaging.
        return (
            torch.stack(e_u_list, dim=0).sum(dim=0),
            torch.stack(e_i_list, dim=0).sum(dim=0),
            torch.stack(g_u_list, dim=0).sum(dim=0),
            torch.stack(g_i_list, dim=0).sum(dim=0),
        )

    def loss(
        self,
        adj_norm: torch.Tensor,
        u_mul_s: torch.Tensor,
        v_mul_s: torch.Tensor,
        ut: torch.Tensor,
        vt: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        e_u, e_i, g_u, g_i = self.propagate(adj_norm, u_mul_s, v_mul_s, ut, vt, training_view=True)
        item_ids = torch.cat([pos_items, neg_items], dim=0)

        user_logits = g_u[users] @ e_u.t() / self.cfg.temp
        item_logits = g_i[item_ids] @ e_i.t() / self.cfg.temp
        neg_score = torch.logsumexp(user_logits, dim=1).mean() + torch.logsumexp(item_logits, dim=1).mean()
        pos_score = (
            torch.clamp((g_u[users] * e_u[users]).sum(dim=1) / self.cfg.temp, -5.0, 5.0).mean()
            + torch.clamp((g_i[item_ids] * e_i[item_ids]).sum(dim=1) / self.cfg.temp, -5.0, 5.0).mean()
        )
        loss_s = -pos_score + neg_score

        u_emb = e_u[users]
        pos_emb = e_i[pos_items]
        neg_emb = e_i[neg_items]
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        loss_r = -F.logsigmoid(pos_scores - neg_scores).mean()

        loss_reg = torch.zeros((), dtype=torch.float32, device=users.device)
        for param in self.parameters():
            loss_reg = loss_reg + param.norm(2).square()
        loss_reg = loss_reg * self.cfg.lambda2

        loss = loss_r + self.cfg.lambda1 * loss_s + loss_reg
        parts = {
            "loss_r": float(loss_r.detach().cpu().item()),
            "loss_s": float((self.cfg.lambda1 * loss_s).detach().cpu().item()),
            "loss_reg": float(loss_reg.detach().cpu().item()),
        }
        return loss, parts


def compute_svd_factors(adj_norm: torch.Tensor, q: int):
    q_eff = min(q, max(1, adj_norm._nnz() - 1), adj_norm.size(0), adj_norm.size(1))
    print(f"Performing LightGCL SVD: q={q_eff}, nnz={adj_norm._nnz()}")
    with torch.no_grad():
        svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=q_eff, niter=2)
        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)
    return u_mul_s, v_mul_s, svd_u.t(), svd_v.t()


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def precompute_banks(
    cfg: Config,
    model: LightGCLStaticFairModel,
    adj_norm: torch.Tensor,
    u_mul_s: torch.Tensor,
    v_mul_s: torch.Tensor,
    ut: torch.Tensor,
    vt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        user_bank, item_bank, _, _ = model.propagate(adj_norm, u_mul_s, v_mul_s, ut, vt, training_view=False)
        user_bank = F.normalize(user_bank, dim=1)
        item_bank = F.normalize(item_bank, dim=1)
    return user_bank, item_bank


def evaluate_split(
    cfg: Config,
    model: LightGCLStaticFairModel,
    loader: DataLoader,
    device: torch.device,
    user_bank: torch.Tensor,
    item_bank: torch.Tensor,
    user_seen_items: Optional[Dict[int, set]],
    full_ranking: bool,
    average_mode: str = "interaction",
):
    def get_user_vectors(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return user_bank[batch["u"]]

    cold, n_cold = evaluate_embedding_ranker(
        loader,
        device=device,
        n_items=cfg.n_items,
        cold_threshold=cfg.cold_threshold,
        get_user_vectors_fn=get_user_vectors,
        all_item_vectors=item_bank,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=full_ranking,
        user_seen_items=user_seen_items,
        normalize_user=False,
        average_mode=average_mode,
    )
    hot, n_hot = evaluate_embedding_ranker(
        loader,
        device=device,
        n_items=cfg.n_items,
        cold_threshold=cfg.cold_threshold,
        get_user_vectors_fn=get_user_vectors,
        all_item_vectors=item_bank,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=full_ranking,
        user_seen_items=user_seen_items,
        normalize_user=False,
        average_mode=average_mode,
    )
    return cold or {}, n_cold, hot or {}, n_hot


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
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
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        add_user_seen_from_df(test_seen, val_df)

    train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_emb = content_emb.float().to(device)
    model = LightGCLStaticFairModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    adj_norm = build_norm_adj(
        cfg.n_users,
        cfg.n_items,
        train_df["u_idx"].to_numpy(np.int64),
        train_df["i_idx"].to_numpy(np.int64),
        device,
    )
    u_mul_s, v_mul_s, ut, vt = compute_svd_factors(adj_norm, cfg.svd_rank)
    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    n_train = train_users_t.numel()
    print(
        f"Model: LightGCL official-adapted | device={device} | epochs={cfg.n_epochs} | "
        f"emb_dim={cfg.emb_dim}, batch={cfg.batch_size}, lambda1={cfg.lambda1}, temp={cfg.temp}"
    )

    best_val = -1.0
    best_epoch = -1
    best_state = None
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in [5, 10, 20]]
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.ckpt, model, optimizer, device)
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)

    for epoch in range(start_epoch + 1, cfg.n_epochs + 1):
        model.train()
        neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
        neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        loss_parts = {"loss_r": 0.0, "loss_s": 0.0, "loss_reg": 0.0}
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss, parts = model.loss(
                adj_norm,
                u_mul_s,
                v_mul_s,
                ut,
                vt,
                train_users_t[idx],
                train_pos_t[idx],
                neg_t[idx],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"LightGCL loss became non-finite at epoch={epoch}, batch={n_batches}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            for key in loss_parts:
                loss_parts[key] += parts[key]
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        part_msg = " ".join([f"{k}={v / max(1, n_batches):.4f}" for k, v in loss_parts.items()])
        do_eval = (epoch % cfg.eval_interval == 0) or (epoch == cfg.n_epochs)
        if do_eval:
            improved = False
            user_bank, item_bank = precompute_banks(cfg, model, adj_norm, u_mul_s, v_mul_s, ut, vt)
            val_cold, n_vc, _, _ = evaluate_split(
                cfg,
                model,
                val_loader,
                device,
                user_bank,
                item_bank,
                user_seen_items=train_seen,
                full_ranking=True,
            )
            val_key = val_cold.get("N@10", 0.0)
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
                improved = True
            if cfg.ckpt.save and improved:
                save_checkpoint(
                    cfg.ckpt,
                    "best.pt",
                    epoch,
                    model,
                    optimizer,
                    best_state=best_state,
                    extra={"best_val": best_val, "best_epoch": best_epoch},
                )
            print(
                f"LightGCL Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg} | "
                f"val_full_cold_N@10={val_key:.4f} | val_cold_count={n_vc}"
            )
        else:
            print(f"LightGCL Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg}")
        if cfg.ckpt.save:
            save_checkpoint(
                cfg.ckpt,
                "latest.pt",
                epoch,
                model,
                optimizer,
                best_state=best_state,
                extra={"best_val": best_val, "best_epoch": best_epoch},
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Restore LightGCL best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    user_bank, item_bank = precompute_banks(cfg, model, adj_norm, u_mul_s, v_mul_s, ut, vt)
    sample_cold, n_sc, sample_hot, n_sh = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        user_bank,
        item_bank,
        user_seen_items=test_seen,
        full_ranking=False,
    )
    full_cold, n_fc, full_hot, n_fh = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        user_bank,
        item_bank,
        user_seen_items=test_seen,
        full_ranking=True,
    )
    full_cold_item_macro, n_fc_item_macro, full_hot_item_macro, n_fh_item_macro = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        user_bank,
        item_bank,
        user_seen_items=test_seen,
        full_ranking=True,
        average_mode="item_macro",
    )

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
        title="LightGCL Static HIN (official-adapted)",
    )

    out = {
        "model": "LightGCL",
        "model_display": "LightGCL (official-adapted)",
        "source": "Official LightGCL source pulled under third_party/LightGCL; PyTorch static-HIN adaptation.",
        "protocol": "static_item_cold",
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item_macro or {},
        "full_hot_item_macro": full_hot_item_macro or {},
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item_macro,
        "count_full_hot_item_macro": n_fh_item_macro,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_metric": "cold",
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "lambda1": cfg.lambda1,
        "lambda2": cfg.lambda2,
        "temp": cfg.temp,
        "svd_rank": cfg.svd_rank,
        "content_weight": cfg.content_weight,
        "checkpoint_dir": cfg.ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
        "note": (
            "Official LightGCL BPR + SVD contrastive objective is adapted to "
            "the shared static item-cold protocol with a content projection for cold items."
        ),
    }
    result_path = static_result_path("lightgcl_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
