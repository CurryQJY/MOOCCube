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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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

        # Best-epoch selection strategy: cold | hot | combined | weighted | last
        # cold (default) -> back-compat: pick epoch with max val_full_cold_N@10
        # combined        -> pick epoch with max (cold_N@10 + hot_N@10)
        # weighted        -> pick epoch with max (alpha*cold + (1-alpha)*hot)
        # hot             -> pick epoch with max val_full_hot_N@10
        # last            -> always pick the last epoch (no early stop)
        self.best_metric = os.environ.get("BASELINE_BEST_METRIC", "cold").strip().lower()
        self.best_alpha = float(os.environ.get("BASELINE_BEST_ALPHA", "0.5"))
        self.early_stop_average_mode = os.environ.get(
            "BASELINE_EARLY_STOP_AVG_MODE",
            os.environ.get("USIM_EARLY_STOP_AVG_MODE", "interaction"),
        ).strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("BASELINE_EARLY_STOP_AVG_MODE/USIM_EARLY_STOP_AVG_MODE must be interaction or item_macro")

        default_ckpt_dir = os.environ.get("BASELINE_CKPT_DIR", "").strip()
        self.ckpt_dir = os.environ.get("BPR_CKPT_DIR", default_ckpt_dir).strip()
        self.save_ckpt = _env_flag("BPR_SAVE_CKPT", _env_flag("BASELINE_SAVE_CKPT", bool(self.ckpt_dir)))
        self.auto_resume = _env_flag("BPR_AUTO_RESUME", _env_flag("BASELINE_AUTO_RESUME", bool(self.ckpt_dir)))
        self.force_fresh = _env_flag("BPR_FORCE_FRESH", _env_flag("BASELINE_FORCE_FRESH", False))
        self.save_opt_state = _env_flag("BPR_SAVE_OPT_STATE", _env_flag("BASELINE_SAVE_OPT_STATE", True))


def _save_checkpoint(
    cfg: Config,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    best_epoch: int,
    best_state: dict,
    best_cold_at_best: float,
    best_hot_at_best: float,
) -> None:
    if not cfg.ckpt_dir:
        return
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "n_epochs": int(cfg.n_epochs),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if cfg.save_opt_state else None,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "best_state": best_state,
        "best_cold_at_best": float(best_cold_at_best),
        "best_hot_at_best": float(best_hot_at_best),
        "best_metric": cfg.best_metric,
        "best_average_mode": cfg.early_stop_average_mode,
        "static_seed": cfg.static_seed,
        "seed": cfg.seed,
    }
    torch.save(payload, os.path.join(cfg.ckpt_dir, filename))


def _try_resume_checkpoint(
    cfg: Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    if not cfg.ckpt_dir:
        return 0, -1.0, -1, None, float("nan"), float("nan")
    print(
        f"Checkpoint: save={cfg.save_ckpt} resume={cfg.auto_resume} "
        f"force_fresh={cfg.force_fresh} save_opt={cfg.save_opt_state} dir={cfg.ckpt_dir}"
    )
    if cfg.force_fresh or not cfg.auto_resume:
        return 0, -1.0, -1, None, float("nan"), float("nan")

    latest_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if not os.path.exists(latest_path):
        return 0, -1.0, -1, None, float("nan"), float("nan")

    try:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    opt_state = ckpt.get("optimizer_state")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    start_epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", -1.0))
    best_epoch = int(ckpt.get("best_epoch", -1))
    best_state = ckpt.get("best_state")
    best_cold_at_best = float(ckpt.get("best_cold_at_best", float("nan")))
    best_hot_at_best = float(ckpt.get("best_hot_at_best", float("nan")))
    print(
        f"Resume checkpoint: latest_epoch={start_epoch} | best_epoch={best_epoch} | "
        f"best_score={best_val:.6f}"
    )
    return start_epoch, best_val, best_epoch, best_state, best_cold_at_best, best_hot_at_best


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
    best_cold_at_best = float("nan")
    best_hot_at_best = float("nan")
    k_list = [5, 10, 20]
    print(
        f"Best-epoch strategy: {cfg.best_metric}"
        + (f" (alpha={cfg.best_alpha})" if cfg.best_metric == "weighted" else "")
        + f" | avg_mode={cfg.early_stop_average_mode}"
    )
    start_epoch, best_val, best_epoch, best_state, best_cold_at_best, best_hot_at_best = _try_resume_checkpoint(
        cfg,
        model,
        optimizer,
        device,
    )

    n_train = train_users_t.numel()
    for epoch in range(start_epoch, cfg.n_epochs):
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
        val_cold_n10 = float("nan")
        val_hot_n10 = float("nan")
        if do_eval:
            model.eval()
            improved = False
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
                    average_mode=cfg.early_stop_average_mode,
                )
                val_cold_n10 = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0
                # Also evaluate hot when best metric needs it (otherwise skip for speed)
                if cfg.best_metric in {"hot", "combined", "weighted"}:
                    val_full_hot, _ = evaluate_embedding_ranker(
                        val_loader,
                        device=device,
                        n_items=cfg.n_items,
                        cold_threshold=cfg.cold_threshold,
                        get_user_vectors_fn=get_user_fn,
                        all_item_vectors=all_i,
                        k_list=k_list,
                        n_neg=cfg.eval_n_neg,
                        eval_type="hot",
                        full_ranking=True,
                        user_seen_items=train_seen,
                        average_mode=cfg.early_stop_average_mode,
                    )
                    val_hot_n10 = val_full_hot.get("N@10", 0.0) if val_full_hot else 0.0
                # Compute val_key according to strategy
                if cfg.best_metric == "cold":
                    val_key = val_cold_n10
                elif cfg.best_metric == "hot":
                    val_key = val_hot_n10
                elif cfg.best_metric == "combined":
                    val_key = val_cold_n10 + val_hot_n10
                elif cfg.best_metric == "weighted":
                    val_key = cfg.best_alpha * val_cold_n10 + (1.0 - cfg.best_alpha) * val_hot_n10
                elif cfg.best_metric == "last":
                    # Strictly increasing key -> always picks the latest evaluated epoch
                    val_key = float(epoch + 1)
                else:
                    val_key = val_cold_n10  # fallback
                if val_key > best_val:
                    best_val = val_key
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    best_cold_at_best = val_cold_n10
                    best_hot_at_best = val_hot_n10
                    improved = True
            if cfg.save_ckpt and improved:
                _save_checkpoint(
                    cfg,
                    "best.pt",
                    epoch + 1,
                    model,
                    optimizer,
                    best_val,
                    best_epoch,
                    best_state,
                    best_cold_at_best,
                    best_hot_at_best,
                )
            if cfg.best_metric in {"hot", "combined", "weighted"}:
                print(
                    f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f} | "
                    f"val_cold_N@10={val_cold_n10:.4f} | val_hot_N@10={val_hot_n10:.4f} | "
                    f"val_key={val_key:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f} | "
                    f"val_full_cold_N@10={val_cold_n10:.4f}"
                )
        else:
            print(f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f}")
        if cfg.save_ckpt:
            _save_checkpoint(
                cfg,
                "latest.pt",
                epoch + 1,
                model,
                optimizer,
                best_val,
                best_epoch,
                best_state,
                best_cold_at_best,
                best_hot_at_best,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restore best epoch={best_epoch} (metric={cfg.best_metric}, score={best_val:.4f}, "
            f"cold@best={best_cold_at_best:.4f}, hot@best={best_hot_at_best:.4f})"
        )

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
        "model": "BPR",
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
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_cold_at_best if cfg.best_metric != "cold" else best_val,
        "best_metric": cfg.best_metric,
        "best_average_mode": cfg.early_stop_average_mode,
        "best_alpha": cfg.best_alpha if cfg.best_metric == "weighted" else None,
        "best_score": best_val,
        "best_cold_n10_at_best_epoch": best_cold_at_best,
        "best_hot_n10_at_best_epoch": best_hot_at_best,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "checkpoint_dir": cfg.ckpt_dir or None,
        "resumed_from_epoch": start_epoch,
    }
    result_path = static_result_path("bpr_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
