"""
Official-code SEMCo adapter for the shared static HIN item-cold split.

The SEMCo repository is stored under third_party/SEMCo. This adapter reuses its
SEMCo_Learner module and its sparsemax/entmax contrastive objective, while
replacing the original CSV layout and ColdRec evaluator with this repository's
strict static split and full-ranking evaluator.
"""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEMCO_ROOT = Path(__file__).resolve().parent / "third_party" / "SEMCo"
if str(SEMCO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMCO_ROOT))

from models.SEMCo import SEMCo_Learner  # noqa: E402
from util.utils import omega_alpha  # noqa: E402
from entmax import entmax15, sparsemax  # noqa: E402

from hin_data_common import (  # noqa: E402
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
from hin_eval_common import evaluate_embedding_ranker, print_final_report  # noqa: E402


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim
        self.fn = os.environ.get("SEMCO_OFFICIAL_FN", "sparsemax").strip().lower()
        if self.fn not in {"sparsemax", "entmax15", "softmax"}:
            raise ValueError("SEMCO_OFFICIAL_FN must be sparsemax, entmax15, or softmax")

        default_scale = {"sparsemax": "12", "entmax15": "2.5", "softmax": "0.2"}[self.fn]
        default_reg = {"sparsemax": "0.001", "entmax15": "0.005", "softmax": "0.005"}[self.fn]
        self.sm_scale = float(os.environ.get("SEMCO_OFFICIAL_SM_SCALE", default_scale))
        self.reg = float(os.environ.get("SEMCO_OFFICIAL_REG", default_reg))

        self.hidden_size = int(os.environ.get("SEMCO_OFFICIAL_HIDDEN", "192"))
        self.emb_size = int(os.environ.get("SEMCO_OFFICIAL_EMB", "64"))
        self.lr = float(os.environ.get("SEMCO_OFFICIAL_LR", "0.001"))
        self.batch_size = int(os.environ.get("SEMCO_OFFICIAL_BATCH_SIZE", "2048"))
        self.max_epoch = int(os.environ.get("SEMCO_OFFICIAL_EPOCHS", "15"))
        self.eval_batch_size = int(os.environ.get("SEMCO_OFFICIAL_EVAL_BATCH_SIZE", "8192"))
        self.progress_interval = int(os.environ.get("SEMCO_OFFICIAL_PROGRESS_INTERVAL", "0"))
        self.grad_clip = float(os.environ.get("SEMCO_OFFICIAL_GRAD_CLIP", "5.0"))

        self.cold_threshold = int(os.environ.get("SEMCO_OFFICIAL_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1")))
        self.eval_n_neg = int(os.environ.get("SEMCO_OFFICIAL_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("SEMCO_OFFICIAL_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("SEMCO_OFFICIAL_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("SEMCO_OFFICIAL_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("SEMCO_OFFICIAL_STATIC_VAL_RATIO", "0.1"))
        self.run_sampled_eval = _bool_env("SEMCO_OFFICIAL_RUN_SAMPLED_EVAL", False)
        self.average_mode = os.environ.get(
            "SEMCO_OFFICIAL_EARLY_STOP_AVG_MODE",
            os.environ.get("USIM_EARLY_STOP_AVG_MODE", "item_macro"),
        ).strip().lower()
        if self.average_mode not in {"interaction", "item_macro"}:
            raise ValueError("SEMCO_OFFICIAL_EARLY_STOP_AVG_MODE must be interaction or item_macro")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _remap_users(*dfs: pd.DataFrame) -> Tuple[Dict[int, int], Tuple[pd.DataFrame, ...]]:
    users = sorted(set().union(*(set(df["u_idx"].astype(int).tolist()) for df in dfs)))
    user_map = {old: new for new, old in enumerate(users)}
    remapped = []
    for df in dfs:
        out = df.copy()
        out["u_idx"] = out["u_idx"].astype(int).map(user_map).astype(np.int64)
        out["i_idx"] = out["i_idx"].astype(np.int64)
        remapped.append(out)
    return user_map, tuple(remapped)


def _build_interaction_mat(train_df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    rows = train_df["u_idx"].to_numpy(np.int64, copy=False)
    cols = train_df["i_idx"].to_numpy(np.int64, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float32)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    mat.sum_duplicates()
    return mat


def _build_official_data(train_df: pd.DataFrame, n_users: int, n_items: int, content_emb: torch.Tensor):
    warm_items = np.sort(train_df["i_idx"].astype(int).unique()).astype(np.int64)
    interaction_mat = _build_interaction_mat(train_df, n_users, n_items)
    row_counts = np.asarray(interaction_mat.sum(axis=1)).reshape(-1)
    if np.any(row_counts <= 0):
        raise ValueError("Official SEMCo adapter requires every mapped user to have train history")
    return SimpleNamespace(
        mapped_item_content=[content_emb.detach().cpu().numpy().astype(np.float32, copy=True)],
        mapped_warm_item_idx=warm_items,
        interaction_mat=interaction_mat,
    )


def official_loss(cfg: Config, user_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
    if cfg.fn == "softmax":
        pos_score = (user_vecs @ item_vecs.T) / cfg.sm_scale
        return -torch.diag(F.log_softmax(pos_score, dim=1)).mean()

    if cfg.fn == "sparsemax":
        fn = sparsemax
        alpha = 2.0
    else:
        fn = entmax15
        alpha = 1.5
    z = (user_vecs @ item_vecs.T) / cfg.sm_scale
    p_star = fn(z / cfg.sm_scale, dim=1)
    entropy = omega_alpha(p_star, alpha).mean()
    eye = torch.eye(z.shape[0], dtype=z.dtype, device=z.device)
    alignment_loss = torch.einsum("ij,ij->i", p_star - eye, z).mean()
    return alignment_loss + entropy


def _state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate_split(
    cfg: Config,
    loader: DataLoader,
    device: torch.device,
    user_bank: torch.Tensor,
    item_bank: torch.Tensor,
    user_seen_items: Optional[Dict[int, set]],
    full_ranking: bool,
    average_mode: str = "interaction",
    export_cold_item_metrics_path: Optional[str] = None,
    export_hot_item_metrics_path: Optional[str] = None,
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
        export_item_metrics_path=export_cold_item_metrics_path,
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
        export_item_metrics_path=export_hot_item_metrics_path,
    )
    return cold or {}, n_cold, hot or {}, n_hot


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Official SEMCo code uses .cuda() internally; CUDA is required for this adapter")

    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    print(f"Loading data from {data_dir} ...", flush=True)
    meta, df, content_emb = load_hin_processed(data_dir)
    raw_cfg = Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
    setup_seed(raw_cfg.seed)

    train_df, val_df, test_df = static_split_df(
        df,
        seed=raw_cfg.static_seed,
        train_ratio=raw_cfg.train_ratio,
        val_ratio=raw_cfg.val_ratio,
    )
    user_map, (train_df, val_df, test_df) = _remap_users(train_df, val_df, test_df)
    cfg = copy.copy(raw_cfg)
    cfg.n_users = len(user_map)
    print(
        f"Static split remapped: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"users={cfg.n_users}, items={cfg.n_items}, fn={cfg.fn}, scale={cfg.sm_scale}, "
        f"epochs={cfg.max_epoch}",
        flush=True,
    )

    official_data = _build_official_data(train_df, cfg.n_users, cfg.n_items, content_emb)
    idx_map = {int(item): pos for pos, item in enumerate(official_data.mapped_warm_item_idx.tolist())}
    train_users_np = train_df["u_idx"].to_numpy(np.int64, copy=True)
    train_items_np = train_df["i_idx"].to_numpy(np.int64, copy=True)
    train_local_items_np = np.asarray([idx_map[int(item)] for item in train_items_np], dtype=np.int64)

    device = torch.device("cuda:0")
    model = SEMCo_Learner(official_data, cfg.emb_size, cfg.hidden_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, (len(train_users_np) // cfg.batch_size) * cfg.max_epoch),
        eta_min=0.0,
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

    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_local_items_t = torch.tensor(train_local_items_np, dtype=torch.long, device=device)
    n_train = train_users_t.numel()
    best_val = -1.0
    best_epoch = -1
    best_state = None
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in [5, 10, 20]]

    print(
        f"Model: SEMCo official-code adapted | device={device} | "
        f"hidden={cfg.hidden_size}, emb={cfg.emb_size}, lr={cfg.lr}, reg={cfg.reg}",
        flush=True,
    )

    for epoch in range(1, cfg.max_epoch + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        total_batches = (n_train + cfg.batch_size - 1) // cfg.batch_size
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            user_vecs, feats = model(True, train_users_t[idx].detach().cpu().tolist())
            loss = official_loss(cfg, user_vecs, feats[train_local_items_t[idx]])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Official SEMCo loss is non-finite at epoch={epoch}, batch={n_batches}")
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
            if cfg.progress_interval > 0 and n_batches % cfg.progress_interval == 0:
                print(
                    f"  [SEMCo-official] epoch={epoch}/{cfg.max_epoch} "
                    f"batch={n_batches}/{total_batches} loss={epoch_loss / max(1, n_batches):.4f}",
                    flush=True,
                )

        with torch.no_grad():
            user_bank, item_bank = model.eval()(perturbed=False)
            val_cold, n_vc, _, _ = evaluate_split(
                cfg,
                val_loader,
                device,
                user_bank=user_bank,
                item_bank=item_bank,
                user_seen_items=train_seen,
                full_ranking=True,
                average_mode=cfg.average_mode,
            )
        val_key = val_cold.get("N@10", 0.0)
        if val_key > best_val:
            best_val = val_key
            best_epoch = epoch
            best_state = _state_dict_to_cpu(model)
        print(
            f"SEMCo-official Epoch [{epoch}/{cfg.max_epoch}] loss={epoch_loss / max(1, n_batches):.4f} | "
            f"val_full_cold_N@10({cfg.average_mode})={val_key:.4f} | val_cold_count={n_vc}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Restore SEMCo official best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}", flush=True)

    with torch.no_grad():
        user_bank, item_bank = model.eval()(perturbed=False)
        if cfg.run_sampled_eval:
            sample_cold, n_sc, sample_hot, n_sh = evaluate_split(
                cfg,
                test_loader,
                device,
                user_bank,
                item_bank,
                user_seen_items=test_seen,
                full_ranking=False,
            )
        else:
            sample_cold, n_sc, sample_hot, n_sh = {}, 0, {}, 0
        full_cold, n_fc, full_hot, n_fh = evaluate_split(
            cfg,
            test_loader,
            device,
            user_bank,
            item_bank,
            user_seen_items=test_seen,
            full_ranking=True,
        )
        full_cold_item_macro, n_fc_item_macro, full_hot_item_macro, n_fh_item_macro = evaluate_split(
            cfg,
            test_loader,
            device,
            user_bank,
            item_bank,
            user_seen_items=test_seen,
            full_ranking=True,
            average_mode="item_macro",
            export_cold_item_metrics_path=static_result_path("per_item_full_cold_semco_official_static.csv"),
            export_hot_item_metrics_path=static_result_path("per_item_full_hot_semco_official_static.csv"),
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
        title="SEMCo Official-Code Static HIN",
    )

    out = {
        "model": "SEMCo",
        "model_display": "SEMCo (official-code adapted)",
        "source": "Official SEMCo code from third_party/SEMCo; strict-HIN data/evaluator adapter.",
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
        "best_average_mode": cfg.average_mode,
        "static_seed": cfg.static_seed,
        "fn": cfg.fn,
        "sm_scale": cfg.sm_scale,
        "reg": cfg.reg,
        "lr": cfg.lr,
        "hidden_size": cfg.hidden_size,
        "emb_size": cfg.emb_size,
        "max_epoch": cfg.max_epoch,
        "per_item_full_cold_path": static_result_path("per_item_full_cold_semco_official_static.csv"),
        "per_item_full_hot_path": static_result_path("per_item_full_hot_semco_official_static.csv"),
        "note": (
            "Reuses official SEMCo_Learner and official sparse/entmax objective; "
            "data loading and final full-ranking metrics follow this repository's strict static protocol."
        ),
    }
    result_path = static_result_path("semco_official_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    with open(static_result_path("semco_official_static_config.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if not isinstance(v, dict)}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
