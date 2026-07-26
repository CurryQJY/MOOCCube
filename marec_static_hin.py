"""
MARec-style metadata-alignment baseline for the shared static cold-start split.

This is a local reproduction adapted to the MOOCCube HIN preprocessing.  The
paper's key idea is to align collaborative item-item similarity from clicks with
metadata/item-content similarity; here the collaborative backbone is an
EASE-style closed-form item-item model, and the metadata side uses course
content embeddings.
"""

import os
from typing import Dict, Iterable, Tuple

import numpy as np
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
        self.batch_size = int(os.environ.get("MAREC_BATCH_SIZE", "2048"))
        self.cold_threshold = int(os.environ.get("MAREC_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("MAREC_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("MAREC_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("MAREC_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("MAREC_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("MAREC_STATIC_VAL_RATIO", "0.1"))
        self.lambdas = _parse_float_list(os.environ.get("MAREC_LAMBDAS", "100,300,1000"))
        self.alphas = _parse_float_list(os.environ.get("MAREC_ALPHAS", "0.05,0.1,0.2,0.5,1.0"))
        self.meta_topk = int(os.environ.get("MAREC_META_TOPK", "100"))
        self.pop_gamma = float(os.environ.get("MAREC_POP_GAMMA", "0.5"))
        self.mode = os.environ.get("MAREC_MODE", "ridge_prior").strip().lower()
        self.content_betas = _parse_float_list(os.environ.get("MAREC_CONTENT_BETAS", "0,0.5,0.75,1.0"))
        self.normalize_item_vectors = os.environ.get("MAREC_NORMALIZE_ITEM", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }


def _parse_float_list(text: str) -> Tuple[float, ...]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError(f"Empty float list: {text!r}")
    return tuple(vals)


def _build_item_gram(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    gram = np.zeros((n_items, n_items), dtype=np.float64)
    for _, group in train_df.groupby("u_idx", sort=False):
        items = group["i_idx"].astype(np.int64).unique()
        if items.size < 1:
            continue
        gram[np.ix_(items, items)] += 1.0
    return gram


def _build_user_dense_history(user_ids: Iterable[int], train_seen: Dict[int, set], n_items: int, device) -> torch.Tensor:
    user_ids = [int(uid) for uid in user_ids]
    hist = torch.zeros((len(user_ids), n_items), dtype=torch.float32, device=device)
    rows = []
    cols = []
    for row, uid in enumerate(user_ids):
        seen = train_seen.get(uid)
        if not seen:
            continue
        for item in seen:
            item = int(item)
            if 0 <= item < n_items:
                rows.append(row)
                cols.append(item)
    if rows:
        hist[torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(cols, dtype=torch.long, device=device)] = 1.0
    return hist


def _build_user_content_profile(
    user_ids: Iterable[int],
    train_seen: Dict[int, set],
    content_vectors: torch.Tensor,
    device,
) -> torch.Tensor:
    rows = []
    zero = torch.zeros(content_vectors.shape[1], dtype=content_vectors.dtype, device=device)
    for uid in user_ids:
        seen = train_seen.get(int(uid))
        if not seen:
            rows.append(zero)
            continue
        seen_idx = [int(item) for item in seen if 0 <= int(item) < content_vectors.shape[0]]
        if not seen_idx:
            rows.append(zero)
            continue
        idx = torch.tensor(seen_idx, dtype=torch.long, device=device)
        rows.append(content_vectors[idx].mean(dim=0))
    return torch.stack(rows, dim=0)


def _metadata_similarity(content_emb: torch.Tensor, topk: int) -> np.ndarray:
    content = content_emb.float()
    content = torch.nn.functional.normalize(content, dim=1)
    sim = torch.mm(content, content.t()).cpu().numpy().astype(np.float64)
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0.0] = 0.0
    if topk > 0 and topk < sim.shape[0]:
        keep = np.zeros_like(sim, dtype=bool)
        idx = np.argpartition(sim, -topk, axis=0)[-topk:, :]
        cols = np.arange(sim.shape[1])[None, :]
        keep[idx, cols] = True
        sim = np.where(keep, sim, 0.0)
    col_sum = sim.sum(axis=0, keepdims=True)
    return sim / np.maximum(col_sum, 1e-12)


def _row_normalize_np(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norm, 1e-12)).astype(np.float32)


def _ease_backbone(gram: np.ndarray, reg: float) -> np.ndarray:
    mat = gram.copy()
    diag = np.diag_indices_from(mat)
    mat[diag] += reg
    inv = np.linalg.inv(mat)
    denom = -np.diag(inv).copy()
    denom[np.abs(denom) < 1e-12] = -1e-12
    coef = inv / denom
    coef[diag] = 0.0
    return coef.astype(np.float32)


def _marec_prior(gram: np.ndarray, meta_sim: np.ndarray, item_pop: np.ndarray, reg: float, alpha: float, gamma: float) -> np.ndarray:
    n_items = gram.shape[0]
    diag = np.diag_indices(n_items)
    col_weight = np.power(1.0 + item_pop.astype(np.float64), -gamma)
    prior = meta_sim * col_weight.reshape(1, -1)
    mat = gram.copy()
    mat[diag] += reg + alpha
    rhs = gram + alpha * prior
    coef = np.linalg.solve(mat, rhs)
    coef[diag] = 0.0
    return coef.astype(np.float32)


def _make_item_vectors(gram: np.ndarray, meta_sim: np.ndarray, item_pop: np.ndarray, reg: float, alpha: float, cfg: Config) -> np.ndarray:
    if cfg.mode == "blend":
        backbone = _ease_backbone(gram, reg)
        col_weight = np.power(1.0 + item_pop.astype(np.float64), -cfg.pop_gamma)
        prior = (meta_sim * col_weight.reshape(1, -1)).astype(np.float32)
        coef = (1.0 - alpha) * backbone + alpha * prior
        np.fill_diagonal(coef, 0.0)
        return coef.T.astype(np.float32)
    coef = _marec_prior(gram, meta_sim, item_pop, reg, alpha, cfg.pop_gamma)
    return coef.T.astype(np.float32)


def _evaluate(
    item_vectors_np: np.ndarray,
    content_vectors: torch.Tensor,
    content_beta: float,
    train_seen: Dict[int, set],
    val_loader,
    test_loader,
    cfg: Config,
    device,
    k_list,
    test_seen: Dict[int, set],
):
    beta = float(max(0.0, min(1.0, content_beta)))
    item_part = torch.tensor(item_vectors_np, dtype=torch.float32, device=device)
    if cfg.normalize_item_vectors:
        item_part = torch.nn.functional.normalize(item_part, dim=1)
    content_part = torch.nn.functional.normalize(content_vectors.to(device).float(), dim=1)
    if beta <= 0.0:
        all_item_vectors = item_part
    elif beta >= 1.0:
        all_item_vectors = content_part
    else:
        all_item_vectors = torch.cat(
            [
                ((1.0 - beta) ** 0.5) * item_part,
                (beta ** 0.5) * content_part,
            ],
            dim=1,
        )

    def get_user_fn(batch):
        user_ids = batch["u"].detach().cpu().tolist()
        hist = _build_user_dense_history(user_ids, train_seen, cfg.n_items, device)
        hist = torch.nn.functional.normalize(hist, dim=1)
        if beta <= 0.0:
            return hist
        profile = _build_user_content_profile(user_ids, train_seen, content_part, device)
        profile = torch.nn.functional.normalize(profile, dim=1)
        if beta >= 1.0:
            return profile
        return torch.cat(
            [
                ((1.0 - beta) ** 0.5) * hist,
                (beta ** 0.5) * profile,
            ],
            dim=1,
        )

    val_full_cold, _ = evaluate_embedding_ranker(
        val_loader,
        device=device,
        n_items=cfg.n_items,
        cold_threshold=cfg.cold_threshold,
        get_user_vectors_fn=get_user_fn,
        all_item_vectors=all_item_vectors,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=train_seen,
    )
    val_key = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0

    sample_cold, n_sc = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_item_vectors,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
        user_seen_items=test_seen,
    )
    sample_hot, n_sh = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_item_vectors,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
        user_seen_items=test_seen,
    )
    full_cold, n_fc = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_item_vectors,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen,
    )
    full_hot, n_fh = evaluate_embedding_ranker(
        test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, all_item_vectors,
        k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen,
    )
    return val_key, sample_cold, n_sc, sample_hot, n_sh, full_cold, n_fc, full_hot, n_fh


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

    item_pop = np.zeros(cfg.n_items, dtype=np.float64)
    counts = train_df["i_idx"].astype(int).value_counts()
    for item_id, count in counts.items():
        if 0 <= int(item_id) < cfg.n_items:
            item_pop[int(item_id)] = float(count)

    print("Building MARec item-item matrices ...")
    gram = _build_item_gram(train_df, cfg.n_items)
    meta_sim = _metadata_similarity(content_emb, topk=cfg.meta_topk)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_list = [5, 10, 20]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]
    best = None

    print(f"Model: MARec static | device={device} | mode={cfg.mode}")
    for reg in cfg.lambdas:
        for alpha in cfg.alphas:
            item_vecs = _make_item_vectors(gram, meta_sim, item_pop, reg=reg, alpha=alpha, cfg=cfg)
            if cfg.normalize_item_vectors:
                item_vecs = _row_normalize_np(item_vecs)
            for content_beta in cfg.content_betas:
                val_key, sample_cold, n_sc, sample_hot, n_sh, full_cold, n_fc, full_hot, n_fh = _evaluate(
                    item_vecs,
                    content_emb,
                    content_beta,
                    train_seen,
                    val_loader,
                    test_loader,
                    cfg,
                    device,
                    k_list,
                    test_seen,
                )
                print(
                    f"  reg={reg:g}, alpha={alpha:g}, content_beta={content_beta:g} | "
                    f"val_full_cold_N@10={val_key:.4f}"
                )
                if best is None or val_key > best["val"]:
                    best = {
                        "val": val_key,
                        "reg": reg,
                        "alpha": alpha,
                        "content_beta": content_beta,
                        "sample_cold": sample_cold or {},
                        "sample_hot": sample_hot or {},
                        "full_cold": full_cold or {},
                        "full_hot": full_hot or {},
                        "counts": (n_sc, n_sh, n_fc, n_fh),
                    }

    n_sc, n_sh, n_fc, n_fh = best["counts"]
    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=best["sample_cold"],
        sample_hot=best["sample_hot"],
        full_cold=best["full_cold"],
        full_hot=best["full_hot"],
        count_sample_cold=n_sc,
        count_sample_hot=n_sh,
        count_full_cold=n_fc,
        count_full_hot=n_fh,
        title="MARec Static HIN",
    )

    out = {
        "model": "MARec",
        "protocol": "static",
        "sample_cold": best["sample_cold"],
        "sample_hot": best["sample_hot"],
        "full_cold": best["full_cold"],
        "full_hot": best["full_hot"],
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "best_val_full_cold_n10": best["val"],
        "best_reg": best["reg"],
        "best_alpha": best["alpha"],
        "best_content_beta": best["content_beta"],
        "mode": cfg.mode,
        "meta_topk": cfg.meta_topk,
        "pop_gamma": cfg.pop_gamma,
        "normalize_item_vectors": cfg.normalize_item_vectors,
        "eval_n_neg": cfg.eval_n_neg,
    }
    result_path = static_result_path("marec_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
