"""
SEMCo-style static item-cold adapter for the shared HIN split.

This is a lightweight implementation of the SEMCo idea for the local strict
item-cold evaluator: learn a content-only item similarity space with sampled
entmax, then rank each target item against the full catalog using a profile
formed from the user's observed training items.

The original SEMCo paper emphasizes item-item similarity learned from content
features rather than mapping cold items into an ID-collaborative embedding.
This adapter keeps that spirit while reusing the repository's static split and
full-ranking evaluation utilities.
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

from baseline_checkpoint import checkpoint_config, maybe_resume_checkpoint, save_checkpoint
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


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.encoder_mode = os.environ.get("SEMCO_ENCODER_MODE", "mlp").strip().lower()
        if self.encoder_mode not in {"mlp", "raw_residual", "raw"}:
            raise ValueError("SEMCO_ENCODER_MODE must be one of: mlp, raw_residual, raw")
        default_emb_dim = str(content_dim) if self.encoder_mode in {"raw_residual", "raw"} else "128"
        self.emb_dim = int(os.environ.get("SEMCO_EMB_DIM", default_emb_dim))
        if self.encoder_mode in {"raw_residual", "raw"} and self.emb_dim != content_dim:
            raise ValueError("SEMCO_ENCODER_MODE=raw_residual/raw requires SEMCO_EMB_DIM == content_dim")
        self.hidden_dim = int(os.environ.get("SEMCO_HIDDEN_DIM", "256"))
        self.dropout = float(os.environ.get("SEMCO_DROPOUT", "0.10"))
        self.raw_delta_scale = float(os.environ.get("SEMCO_RAW_DELTA_SCALE", "0.05"))
        self.batch_size = int(os.environ.get("SEMCO_BATCH_SIZE", "4096"))
        self.eval_batch_size = int(os.environ.get("SEMCO_EVAL_BATCH_SIZE", str(self.batch_size)))
        self.bank_batch_size = int(os.environ.get("SEMCO_BANK_BATCH_SIZE", "32768"))

        self.n_epochs = int(os.environ.get("SEMCO_STATIC_EPOCHS", "20"))
        self.lr = float(os.environ.get("SEMCO_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("SEMCO_WEIGHT_DECAY", "1e-4"))
        self.grad_clip = float(os.environ.get("SEMCO_GRAD_CLIP", "5.0"))
        self.eval_interval = int(os.environ.get("SEMCO_EVAL_INTERVAL", "1"))

        self.n_neg = int(os.environ.get("SEMCO_NEGATIVE_NUMBER", "64"))
        self.temperature = float(os.environ.get("SEMCO_TAU", "0.10"))
        self.entmax_alpha = float(os.environ.get("SEMCO_ENTMAX_ALPHA", "1.5"))
        self.entmax_iter = int(os.environ.get("SEMCO_ENTMAX_ITER", "50"))
        self.loss_mode = os.environ.get("SEMCO_LOSS_MODE", "fy").strip().lower()
        if self.loss_mode not in {"fy", "nll"}:
            raise ValueError("SEMCO_LOSS_MODE must be 'fy' or 'nll'")

        self.detach_query = _bool_env("SEMCO_DETACH_QUERY", True)
        self.exclude_train_target = _bool_env("SEMCO_EXCLUDE_TRAIN_TARGET", True)
        self.singleton_policy = os.environ.get("SEMCO_SINGLETON_POLICY", "global").strip().lower()
        if self.singleton_policy not in {"global", "keep"}:
            raise ValueError("SEMCO_SINGLETON_POLICY must be 'global' or 'keep'")

        self.cold_threshold = int(os.environ.get("SEMCO_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("SEMCO_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("SEMCO_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("SEMCO_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("SEMCO_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("SEMCO_STATIC_VAL_RATIO", "0.1"))
        self.early_stop_average_mode = os.environ.get(
            "SEMCO_EARLY_STOP_AVG_MODE",
            os.environ.get(
                "BASELINE_EARLY_STOP_AVG_MODE",
                os.environ.get("USIM_EARLY_STOP_AVG_MODE", "item_macro"),
            ),
        ).strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("SEMCO_EARLY_STOP_AVG_MODE must be interaction or item_macro")
        self.run_sampled_eval = _bool_env("SEMCO_RUN_SAMPLED_EVAL", False)
        self.export_item_only = _bool_env("SEMCO_EXPORT_ITEM_ONLY", False)
        self.ckpt = checkpoint_config("SEMCO")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class SEMCoStaticModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("content_features", content_emb.float())
        if cfg.encoder_mode == "raw":
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Sequential(
                nn.LayerNorm(cfg.content_dim),
                nn.Linear(cfg.content_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.cfg.encoder_mode == "raw":
            return
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        if self.cfg.encoder_mode == "raw_residual":
            last = self.encoder[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def encode_all_items(self) -> torch.Tensor:
        raw = F.normalize(self.content_features, dim=1)
        if self.cfg.encoder_mode == "raw":
            return raw
        encoded = self.encoder(self.content_features)
        if self.cfg.encoder_mode == "raw_residual":
            return F.normalize(raw + self.cfg.raw_delta_scale * encoded, dim=1)
        return F.normalize(encoded, dim=1)


def entmax_bisect(logits: torch.Tensor, alpha: float = 1.5, n_iter: int = 50) -> torch.Tensor:
    if abs(alpha - 1.0) < 1e-6:
        return torch.softmax(logits, dim=-1)
    if alpha <= 1.0:
        raise ValueError("entmax alpha must be > 1")

    x = logits * (alpha - 1.0)
    x = x - x.max(dim=-1, keepdim=True).values
    tau_lo = x.min(dim=-1, keepdim=True).values - 1.0
    tau_hi = x.max(dim=-1, keepdim=True).values
    power = 1.0 / (alpha - 1.0)

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) * 0.5
        probs = torch.clamp(x - tau, min=0.0).pow(power)
        too_large = probs.sum(dim=-1, keepdim=True) >= 1.0
        tau_lo = torch.where(too_large, tau, tau_lo)
        tau_hi = torch.where(too_large, tau_hi, tau)

    tau = (tau_lo + tau_hi) * 0.5
    probs = torch.clamp(x - tau, min=0.0).pow(power)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def sampled_entmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.5,
    n_iter: int = 50,
    mode: str = "fy",
) -> torch.Tensor:
    if abs(alpha - 1.0) < 1e-6:
        return F.cross_entropy(logits, target)

    probs = entmax_bisect(logits, alpha=alpha, n_iter=n_iter)
    if mode == "nll":
        target_prob = probs.gather(1, target.view(-1, 1)).squeeze(1)
        return -torch.log(target_prob.clamp_min(1e-12)).mean()

    omega = (probs.pow(alpha).sum(dim=-1) - 1.0) / (alpha * (alpha - 1.0))
    target_score = logits.gather(1, target.view(-1, 1)).squeeze(1)
    loss = (probs * logits).sum(dim=-1) - omega - target_score
    return loss.mean()


def build_user_item_count_matrix(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    users_np = train_df["u_idx"].to_numpy(np.int64, copy=True)
    items_np = train_df["i_idx"].to_numpy(np.int64, copy=True)
    indices = torch.tensor(np.vstack([users_np, items_np]), dtype=torch.long, device=device)
    values = torch.ones(users_np.shape[0], dtype=torch.float32, device=device)
    count_mat = torch.sparse_coo_tensor(indices, values, (n_users, n_items), device=device).coalesce()

    hist_counts = torch.zeros(n_users, dtype=torch.float32, device=device)
    hist_counts.scatter_add_(0, torch.tensor(users_np, dtype=torch.long, device=device), values)

    pair_counts = train_df.groupby(["u_idx", "i_idx"])["i_idx"].transform("size").to_numpy(np.float32, copy=True)
    return count_mat, hist_counts, pair_counts


def sample_semco_negatives(
    batch_users: np.ndarray,
    batch_pos: np.ndarray,
    user_neg_pool: Dict[int, np.ndarray],
    n_items: int,
    n_neg: int,
) -> np.ndarray:
    neg = np.empty((batch_pos.shape[0], n_neg), dtype=np.int64)
    fallback = np.arange(n_items, dtype=np.int64)
    for row, (uid, pos) in enumerate(zip(batch_users.tolist(), batch_pos.tolist())):
        pool = user_neg_pool.get(int(uid))
        if pool is None or pool.size < 1:
            pool = fallback
        chosen = np.random.choice(pool, size=n_neg, replace=True)
        same = chosen == int(pos)
        if same.any():
            alt_pool = pool[pool != int(pos)]
            if alt_pool.size < 1:
                alt_pool = fallback[fallback != int(pos)]
            if alt_pool.size < 1:
                alt_pool = pool
            chosen[same] = np.random.choice(alt_pool, size=int(same.sum()), replace=True)
        neg[row] = chosen
    return neg


def build_profile_sum_bank(
    user_item_counts: torch.Tensor,
    item_vectors: torch.Tensor,
    detach_query: bool = True,
) -> torch.Tensor:
    profile_items = item_vectors.detach() if detach_query else item_vectors
    return torch.sparse.mm(user_item_counts, profile_items)


def build_eval_profile_bank(
    profile_sum_bank: torch.Tensor,
    hist_counts: torch.Tensor,
    item_vectors: torch.Tensor,
) -> torch.Tensor:
    global_profile = item_vectors.detach().mean(dim=0, keepdim=True)
    denom = hist_counts.view(-1, 1).clamp_min(1.0)
    profiles = profile_sum_bank / denom
    no_history = hist_counts.view(-1, 1) <= 0.0
    profiles = torch.where(no_history, global_profile.expand_as(profiles), profiles)
    return F.normalize(profiles, dim=1)


def training_profiles(
    cfg: Config,
    profile_sum_bank: torch.Tensor,
    hist_counts: torch.Tensor,
    item_vectors: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    pos_pair_counts: torch.Tensor,
) -> torch.Tensor:
    query_item_vectors = item_vectors.detach() if cfg.detach_query else item_vectors
    raw_sum = profile_sum_bank[users]
    raw_count = hist_counts[users]

    if cfg.exclude_train_target:
        pos_count = pos_pair_counts.to(dtype=raw_sum.dtype, device=raw_sum.device)
        loo_sum = raw_sum - query_item_vectors[pos_items] * pos_count.view(-1, 1)
        loo_count = raw_count - pos_count
        if cfg.singleton_policy == "global":
            global_profile = query_item_vectors.mean(dim=0, keepdim=True).expand_as(loo_sum)
            profiles = torch.where(
                (loo_count > 0).view(-1, 1),
                loo_sum / loo_count.clamp_min(1.0).view(-1, 1),
                global_profile,
            )
        else:
            profiles = torch.where(
                (loo_count > 0).view(-1, 1),
                loo_sum / loo_count.clamp_min(1.0).view(-1, 1),
                raw_sum / raw_count.clamp_min(1.0).view(-1, 1),
            )
    else:
        profiles = raw_sum / raw_count.clamp_min(1.0).view(-1, 1)

    return F.normalize(profiles, dim=1)


def semco_batch_loss(
    cfg: Config,
    item_vectors: torch.Tensor,
    profile_sum_bank: torch.Tensor,
    hist_counts: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    pos_pair_counts: torch.Tensor,
    neg_items: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    query = training_profiles(
        cfg,
        profile_sum_bank=profile_sum_bank,
        hist_counts=hist_counts,
        item_vectors=item_vectors,
        users=users,
        pos_items=pos_items,
        pos_pair_counts=pos_pair_counts,
    )
    candidates = torch.cat([pos_items.view(-1, 1), neg_items], dim=1)
    cand_vec = item_vectors[candidates]
    logits = torch.bmm(cand_vec, query.unsqueeze(2)).squeeze(2) / max(cfg.temperature, 1e-6)
    target = torch.zeros(users.numel(), dtype=torch.long, device=users.device)
    loss = sampled_entmax_loss(
        logits,
        target,
        alpha=cfg.entmax_alpha,
        n_iter=cfg.entmax_iter,
        mode=cfg.loss_mode,
    )
    with torch.no_grad():
        probs = entmax_bisect(logits, alpha=cfg.entmax_alpha, n_iter=cfg.entmax_iter)
        active = (probs > 1e-8).float().sum(dim=1).mean()
        pos_prob = probs[:, 0].mean()
        margin = (logits[:, 0] - logits[:, 1:].max(dim=1).values).mean()
    return loss, {
        "active": float(active.detach().cpu().item()),
        "pos_prob": float(pos_prob.detach().cpu().item()),
        "margin": float(margin.detach().cpu().item()),
    }


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate_split(
    cfg: Config,
    model: SEMCoStaticModel,
    loader: DataLoader,
    device: torch.device,
    item_bank: torch.Tensor,
    profile_bank: torch.Tensor,
    user_seen_items: Optional[Dict[int, set]],
    full_ranking: bool,
    average_mode: str = "interaction",
    export_cold_item_metrics_path: Optional[str] = None,
    export_hot_item_metrics_path: Optional[str] = None,
):
    def get_user_vectors(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return profile_bank[batch["u"]]

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
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...", flush=True)
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
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}",
        flush=True,
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
    model = SEMCoStaticModel(cfg, content_emb).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = (
        torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if trainable_params
        else None
    )
    if optimizer is None and cfg.n_epochs > 0:
        print("SEMCo raw encoder has no trainable parameters; skip training loop.", flush=True)
        cfg.n_epochs = 0
    user_item_counts, hist_counts, train_pos_pair_counts_np = build_user_item_count_matrix(
        train_df,
        cfg.n_users,
        cfg.n_items,
        device,
    )
    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    train_pair_counts_t = torch.tensor(train_pos_pair_counts_np, dtype=torch.float32, device=device)
    n_train = train_users_t.numel()

    print(
        f"Model: SEMCo static | device={device} | epochs={cfg.n_epochs} | "
        f"mode={cfg.encoder_mode}, alpha={cfg.entmax_alpha:.2f}, tau={cfg.temperature:.3f}, neg={cfg.n_neg}, "
        f"detach_query={int(cfg.detach_query)}, exclude_train_target={int(cfg.exclude_train_target)}",
        flush=True,
    )

    best_val = -1.0
    best_epoch = -1
    best_state = None
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in [5, 10, 20]]
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.ckpt, model, optimizer, device)
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)

    if start_epoch == 0:
        model.eval()
        with torch.no_grad():
            item_bank = model.encode_all_items()
            profile_sum_eval = build_profile_sum_bank(user_item_counts, item_bank, detach_query=True)
            profile_bank = build_eval_profile_bank(profile_sum_eval, hist_counts, item_bank)
            init_cold, n_vc, _, _ = evaluate_split(
                cfg,
                model,
                val_loader,
                device,
                item_bank=item_bank,
                profile_bank=profile_bank,
                user_seen_items=train_seen,
                full_ranking=True,
                average_mode=cfg.early_stop_average_mode,
            )
        init_val = init_cold.get("N@10", 0.0)
        if init_val > best_val:
            best_val = init_val
            best_epoch = 0
            best_state = state_dict_to_cpu(model)
        print(
            f"SEMCo Init loss=NA | "
            f"val_full_cold_N@10({cfg.early_stop_average_mode})={init_val:.4f} | "
            f"val_cold_count={n_vc}",
            flush=True,
        )

    for epoch in range(start_epoch + 1, cfg.n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        stat_sums = {"active": 0.0, "pos_prob": 0.0, "margin": 0.0}
        n_batches = 0

        item_vectors = model.encode_all_items()
        profile_sum_bank = build_profile_sum_bank(
            user_item_counts,
            item_vectors,
            detach_query=cfg.detach_query,
        )
        if cfg.detach_query:
            profile_sum_bank = profile_sum_bank.detach()

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            idx_np = idx.detach().cpu().numpy()
            neg_np = sample_semco_negatives(
                batch_users=train_users_np[idx_np],
                batch_pos=train_pos_np[idx_np],
                user_neg_pool=user_neg_pool,
                n_items=cfg.n_items,
                n_neg=cfg.n_neg,
            )
            neg_items = torch.tensor(neg_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            item_vectors = model.encode_all_items()
            if cfg.detach_query:
                profile_sum_for_batch = profile_sum_bank
            else:
                profile_sum_for_batch = build_profile_sum_bank(user_item_counts, item_vectors, detach_query=False)
            loss, stats = semco_batch_loss(
                cfg,
                item_vectors=item_vectors,
                profile_sum_bank=profile_sum_for_batch,
                hist_counts=hist_counts,
                users=train_users_t[idx],
                pos_items=train_pos_t[idx],
                pos_pair_counts=train_pair_counts_t[idx],
                neg_items=neg_items,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"SEMCo loss became non-finite at epoch={epoch}, batch={n_batches}")
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            for key, value in stats.items():
                stat_sums[key] += value
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_stats = {key: value / max(1, n_batches) for key, value in stat_sums.items()}
        do_eval = (epoch % max(1, cfg.eval_interval) == 0) or (epoch == cfg.n_epochs)
        if do_eval:
            model.eval()
            with torch.no_grad():
                item_bank = model.encode_all_items()
                profile_sum_eval = build_profile_sum_bank(user_item_counts, item_bank, detach_query=True)
                profile_bank = build_eval_profile_bank(profile_sum_eval, hist_counts, item_bank)
                val_cold, n_vc, _, _ = evaluate_split(
                    cfg,
                    model,
                    val_loader,
                    device,
                    item_bank=item_bank,
                    profile_bank=profile_bank,
                    user_seen_items=train_seen,
                    full_ranking=True,
                    average_mode=cfg.early_stop_average_mode,
                )
            val_key = val_cold.get("N@10", 0.0)
            improved = val_key > best_val
            if improved:
                best_val = val_key
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
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
                f"SEMCo Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | "
                f"active={avg_stats['active']:.2f} pos_p={avg_stats['pos_prob']:.4f} "
                f"margin={avg_stats['margin']:.4f} | "
                f"val_full_cold_N@10({cfg.early_stop_average_mode})={val_key:.4f} | "
                f"val_cold_count={n_vc}",
                flush=True,
            )
        else:
            print(
                f"SEMCo Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | "
                f"active={avg_stats['active']:.2f} pos_p={avg_stats['pos_prob']:.4f} "
                f"margin={avg_stats['margin']:.4f}",
                flush=True,
            )
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
    print(
        f"Restore SEMCo best epoch={best_epoch}, "
        f"val_full_cold_N@10({cfg.early_stop_average_mode})={best_val:.4f}",
        flush=True,
    )

    model.eval()
    with torch.no_grad():
        item_bank = model.encode_all_items()
        profile_sum_eval = build_profile_sum_bank(user_item_counts, item_bank, detach_query=True)
        profile_bank = build_eval_profile_bank(profile_sum_eval, hist_counts, item_bank)

        if cfg.export_item_only:
            print("SEMCO_EXPORT_ITEM_ONLY=1: skip sampled and interaction-macro final eval.", flush=True)
            sample_cold, n_sc, sample_hot, n_sh = {}, 0, {}, 0
            full_cold, n_fc, full_hot, n_fh = {}, 0, {}, 0
        else:
            if cfg.run_sampled_eval:
                sample_cold, n_sc, sample_hot, n_sh = evaluate_split(
                    cfg,
                    model,
                    test_loader,
                    device,
                    item_bank,
                    profile_bank,
                    user_seen_items=test_seen,
                    full_ranking=False,
                )
            else:
                sample_cold, n_sc, sample_hot, n_sh = {}, 0, {}, 0
            full_cold, n_fc, full_hot, n_fh = evaluate_split(
                cfg,
                model,
                test_loader,
                device,
                item_bank,
                profile_bank,
                user_seen_items=test_seen,
                full_ranking=True,
            )

        full_cold_item_macro, n_fc_item_macro, full_hot_item_macro, n_fh_item_macro = evaluate_split(
            cfg,
            model,
            test_loader,
            device,
            item_bank,
            profile_bank,
            user_seen_items=test_seen,
            full_ranking=True,
            average_mode="item_macro",
            export_cold_item_metrics_path=static_result_path("per_item_full_cold_semco_static.csv"),
            export_hot_item_metrics_path=static_result_path("per_item_full_hot_semco_static.csv"),
        )

    if not cfg.export_item_only:
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
            title="SEMCo Static HIN",
        )

    out = {
        "model": "SEMCo",
        "model_display": "SEMCo (sampled entmax static-HIN adaptation)",
        "source": (
            "Local content-only sampled-entmax adapter for the shared static item-cold "
            "protocol; no official source was integrated."
        ),
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
        "best_average_mode": cfg.early_stop_average_mode,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "entmax_alpha": cfg.entmax_alpha,
        "tau": cfg.temperature,
        "negative_number": cfg.n_neg,
        "loss_mode": cfg.loss_mode,
        "detach_query": cfg.detach_query,
        "exclude_train_target": cfg.exclude_train_target,
        "singleton_policy": cfg.singleton_policy,
        "checkpoint_dir": cfg.ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
        "per_item_full_cold_path": static_result_path("per_item_full_cold_semco_static.csv"),
        "per_item_full_hot_path": static_result_path("per_item_full_hot_semco_static.csv"),
        "note": (
            "Ranks with content-only item vectors and train-history content profiles; "
            "checkpoint selected by validation full cold N@10."
        ),
    }
    result_path = static_result_path("semco_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
