"""
CCFCRec static item-cold adaptation for the shared HIN split.

The official CCFCRec code is pulled under third_party/CCFCRec. Its public
scripts are hard-wired for ML-20M / Amazon-VG and evaluate item-to-user
ranking. This wrapper keeps the static item-cold protocol used by the local
main table and adapts the core CCFCRec idea:

  - learn ID-based user/item collaborative embeddings from warm interactions;
  - learn a content-to-collaborative item encoder;
  - align content-generated item embeddings with co-occurring item ID
    embeddings using contrastive learning;
  - rank cold items with content-generated vectors and hot items with ID
    vectors under the shared full-ranking evaluator.
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

        self.emb_dim = int(os.environ.get("CCFCREC_EMB_DIM", "128"))
        self.hidden_dim = int(os.environ.get("CCFCREC_HIDDEN_DIM", "256"))
        self.batch_size = int(os.environ.get("CCFCREC_BATCH_SIZE", "4096"))
        self.eval_batch_size = int(os.environ.get("CCFCREC_EVAL_BATCH_SIZE", str(self.batch_size)))
        self.bank_batch_size = int(os.environ.get("CCFCREC_BANK_BATCH_SIZE", "32768"))

        self.n_epochs = int(os.environ.get("CCFCREC_STATIC_EPOCHS", "80"))
        self.lr = float(os.environ.get("CCFCREC_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("CCFCREC_WEIGHT_DECAY", "1e-4"))
        self.eval_interval = int(os.environ.get("CCFCREC_EVAL_INTERVAL", "5"))

        self.positive_number = int(os.environ.get("CCFCREC_POSITIVE_NUMBER", "10"))
        self.negative_number = int(os.environ.get("CCFCREC_NEGATIVE_NUMBER", "40"))
        self.self_negative_number = int(os.environ.get("CCFCREC_SELF_NEGATIVE_NUMBER", "40"))
        self.tau = float(os.environ.get("CCFCREC_TAU", "0.1"))
        self.lambda1 = float(os.environ.get("CCFCREC_LAMBDA1", "0.5"))

        self.cold_threshold = int(os.environ.get("CCFCREC_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("CCFCREC_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("CCFCREC_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("CCFCREC_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("CCFCREC_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("CCFCREC_STATIC_VAL_RATIO", "0.1"))
        self.eval_item_mode = os.environ.get("CCFCREC_EVAL_ITEM_MODE", "mixed").strip().lower()
        if self.eval_item_mode not in {"mixed", "content", "id"}:
            raise ValueError("CCFCREC_EVAL_ITEM_MODE must be one of: mixed, content, id")
        self.ckpt = checkpoint_config("CCFCREC")


class CCFCRecStaticModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("content_features", content_emb.float())
        self.user_embedding = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_embedding = nn.Embedding(cfg.n_items, cfg.emb_dim)
        self.content_encoder = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        for module in self.content_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_content(self, item_idx: torch.Tensor) -> torch.Tensor:
        return self.content_encoder(self.content_features[item_idx])

    def training_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        co_pos_items: torch.Tensor,
        co_neg_items: torch.Tensor,
        self_neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        user_vec = self.user_embedding(users)
        pos_id = self.item_embedding(pos_items)
        neg_id = self.item_embedding(neg_items)
        pos_content = self.encode_content(pos_items)
        neg_content = self.encode_content(neg_items)

        id_pos_score = (user_vec * pos_id).sum(dim=1)
        id_neg_score = (user_vec * neg_id).sum(dim=1)
        rank_id = -F.logsigmoid(id_pos_score - id_neg_score).mean()

        content_pos_score = (user_vec * pos_content).sum(dim=1)
        content_neg_score = (user_vec * neg_content).sum(dim=1)
        rank_content = -F.logsigmoid(content_pos_score - content_neg_score).mean()

        query = F.normalize(pos_content, dim=1)
        co_pos = F.normalize(self.item_embedding(co_pos_items), dim=2)
        co_neg = F.normalize(self.item_embedding(co_neg_items), dim=2)
        pos_logits = torch.bmm(co_pos, query.unsqueeze(2)).squeeze(2) / self.cfg.tau
        neg_logits = torch.bmm(co_neg, query.unsqueeze(2)).squeeze(2) / self.cfg.tau
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        coll_contrast = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(all_logits, dim=1)).mean()

        self_pos = (query * F.normalize(pos_id, dim=1)).sum(dim=1, keepdim=True) / self.cfg.tau
        self_neg = F.normalize(self.item_embedding(self_neg_items), dim=2)
        self_neg_logits = torch.bmm(self_neg, query.unsqueeze(2)).squeeze(2) / self.cfg.tau
        self_logits = torch.cat([self_pos, self_neg_logits], dim=1)
        self_contrast = -(self_pos.squeeze(1) - torch.logsumexp(self_logits, dim=1)).mean()

        rank_loss = rank_id + rank_content
        contrast_loss = coll_contrast + self_contrast
        total = self.cfg.lambda1 * contrast_loss + (1.0 - self.cfg.lambda1) * rank_loss
        parts = {
            "rank_id": float(rank_id.detach().cpu().item()),
            "rank_content": float(rank_content.detach().cpu().item()),
            "coll_contrast": float(coll_contrast.detach().cpu().item()),
            "self_contrast": float(self_contrast.detach().cpu().item()),
        }
        return total, parts


def build_item_train_counts(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    counts = torch.zeros(n_items, dtype=torch.long)
    value_counts = train_df["i_idx"].astype(int).value_counts()
    for item_id, count in value_counts.items():
        idx = int(item_id)
        if 0 <= idx < n_items:
            counts[idx] = int(count)
    return counts


def build_sampling_pools(train_df: pd.DataFrame, n_items: int) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    train_item_pool = np.unique(train_df["i_idx"].to_numpy(np.int64, copy=True))
    if train_item_pool.size < 1:
        raise ValueError("Cannot build CCFCRec sampling pools from an empty train item pool")

    item_pos_sets = {int(item): {int(item)} for item in train_item_pool.tolist()}
    for items in train_df.groupby("u_idx")["i_idx"]:
        unique_items = np.unique(items[1].to_numpy(np.int64, copy=False))
        if unique_items.size < 1:
            continue
        as_set = set(int(x) for x in unique_items.tolist())
        for item in unique_items.tolist():
            item_pos_sets.setdefault(int(item), set()).update(as_set)

    item_pos_pool = {
        int(item): np.asarray(sorted(values), dtype=np.int64)
        for item, values in item_pos_sets.items()
        if values
    }
    item_neg_pool = {}
    for item in train_item_pool.tolist():
        pool = train_item_pool[train_item_pool != int(item)]
        item_neg_pool[int(item)] = pool if pool.size > 0 else train_item_pool
    return train_item_pool, item_pos_pool, item_neg_pool


def sample_contrast_items(
    pos_items: np.ndarray,
    cfg: Config,
    train_item_pool: np.ndarray,
    item_pos_pool: Dict[int, np.ndarray],
    item_neg_pool: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz = pos_items.shape[0]
    co_pos = np.empty((bsz, cfg.positive_number), dtype=np.int64)
    co_neg = np.empty((bsz, cfg.negative_number), dtype=np.int64)
    self_neg = np.empty((bsz, cfg.self_negative_number), dtype=np.int64)
    for item in np.unique(pos_items).tolist():
        item = int(item)
        rows = np.where(pos_items == item)[0]
        pos_pool = item_pos_pool.get(item)
        if pos_pool is None or pos_pool.size < 1:
            pos_pool = np.asarray([int(item)], dtype=np.int64)
        neg_pool = item_neg_pool.get(item)
        if neg_pool is None or neg_pool.size < 1:
            neg_pool = train_item_pool
        co_pos[rows] = np.random.choice(
            pos_pool,
            size=(rows.size, cfg.positive_number),
            replace=True,
        )
        co_neg[rows] = np.random.choice(
            neg_pool,
            size=(rows.size, cfg.negative_number),
            replace=True,
        )
        self_neg[rows] = np.random.choice(
            neg_pool,
            size=(rows.size, cfg.self_negative_number),
            replace=True,
        )
    return co_pos, co_neg, self_neg


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def precompute_ccfc_item_bank(
    cfg: Config,
    model: CCFCRecStaticModel,
    item_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    item_is_cold = (item_counts < cfg.cold_threshold).to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, cfg.n_items, cfg.bank_batch_size):
            end = min(start + cfg.bank_batch_size, cfg.n_items)
            idx = torch.arange(start, end, dtype=torch.long, device=device)
            id_vec = model.item_embedding(idx)
            content_vec = model.encode_content(idx)
            if cfg.eval_item_mode == "content":
                item_vec = content_vec
            elif cfg.eval_item_mode == "id":
                item_vec = id_vec
            else:
                cold_mask = item_is_cold[idx].view(-1, 1)
                item_vec = torch.where(cold_mask, content_vec, id_vec)
            chunks.append(F.normalize(item_vec, dim=1).detach())
    return torch.cat(chunks, dim=0)


def evaluate_split(
    cfg: Config,
    model: CCFCRecStaticModel,
    loader: DataLoader,
    device: torch.device,
    item_bank: torch.Tensor,
    user_seen_items: Optional[Dict[int, set]],
    full_ranking: bool,
    average_mode: str = "interaction",
):
    def get_user_vectors(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model.user_embedding(batch["u"])

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
        normalize_user=True,
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
        normalize_user=True,
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
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}, eval_item_mode={cfg.eval_item_mode}"
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

    train_users_np, train_pos_np, user_rows, rank_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    train_item_pool, item_pos_pool, item_neg_pool = build_sampling_pools(train_df, cfg.n_items)
    item_counts = build_item_train_counts(train_df, cfg.n_items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCFCRecStaticModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    n_train = train_users_t.numel()
    print(
        f"Model: CCFCRec official-adapted | device={device} | epochs={cfg.n_epochs} | "
        f"lambda1={cfg.lambda1}, tau={cfg.tau}, pos={cfg.positive_number}, neg={cfg.negative_number}"
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
        rank_neg_np = sample_negatives(train_pos_np, user_rows, rank_neg_pool, cfg.n_items)
        rank_neg_t = torch.tensor(rank_neg_np, dtype=torch.long, device=device)
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        loss_parts = {"rank_id": 0.0, "rank_content": 0.0, "coll_contrast": 0.0, "self_contrast": 0.0}
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            idx_np = idx.detach().cpu().numpy()
            co_pos_np, co_neg_np, self_neg_np = sample_contrast_items(
                pos_items=train_pos_np[idx_np],
                cfg=cfg,
                train_item_pool=train_item_pool,
                item_pos_pool=item_pos_pool,
                item_neg_pool=item_neg_pool,
            )
            co_pos = torch.tensor(co_pos_np, dtype=torch.long, device=device)
            co_neg = torch.tensor(co_neg_np, dtype=torch.long, device=device)
            self_neg = torch.tensor(self_neg_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss, parts = model.training_loss(
                users=train_users_t[idx],
                pos_items=train_pos_t[idx],
                neg_items=rank_neg_t[idx],
                co_pos_items=co_pos,
                co_neg_items=co_neg,
                self_neg_items=self_neg,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"CCFCRec loss became non-finite at epoch={epoch}, batch={n_batches}")
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
            item_bank = precompute_ccfc_item_bank(cfg, model, item_counts, device)
            val_cold, n_vc, _, _ = evaluate_split(
                cfg,
                model,
                val_loader,
                device,
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
                f"CCFCRec Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg} | "
                f"val_full_cold_N@10={val_key:.4f} | val_cold_count={n_vc}"
            )
        else:
            print(f"CCFCRec Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg}")
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
    print(f"Restore CCFCRec best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    item_bank = precompute_ccfc_item_bank(cfg, model, item_counts, device)
    sample_cold, n_sc, sample_hot, n_sh = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        item_bank,
        user_seen_items=test_seen,
        full_ranking=False,
    )
    full_cold, n_fc, full_hot, n_fh = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
        item_bank,
        user_seen_items=test_seen,
        full_ranking=True,
    )
    full_cold_item_macro, n_fc_item_macro, full_hot_item_macro, n_fh_item_macro = evaluate_split(
        cfg,
        model,
        test_loader,
        device,
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
        title="CCFCRec Static HIN (official-adapted)",
    )

    out = {
        "model": "CCFCRec",
        "model_display": "CCFCRec (official-adapted)",
        "source": "Official CCFCRec source pulled under third_party/CCFCRec; PyTorch static-HIN adaptation.",
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
        "tau": cfg.tau,
        "positive_number": cfg.positive_number,
        "negative_number": cfg.negative_number,
        "self_negative_number": cfg.self_negative_number,
        "eval_item_mode": cfg.eval_item_mode,
        "checkpoint_dir": cfg.ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
        "note": (
            "Core CCFCRec contrastive content-to-CF idea is adapted to the shared "
            "user-to-item static item-cold protocol; checkpoint selected by validation full cold N@10."
        ),
    }
    result_path = static_result_path("ccfcrec_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
