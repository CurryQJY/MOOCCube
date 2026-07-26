"""
Standalone SAGERec baseline for the shared static item-cold protocol.

This is intentionally independent from FAST3/SAGE-lite. It implements the
SAGERec ideas that matter for a baseline comparison:
  - trainable sampling over each user's interaction history,
  - two expert user representations with different history aggregation,
  - tail-aware gating by target-item popularity bucket.
"""

import copy
import os
from dataclasses import asdict, dataclass
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
from hin_eval_common import compute_ranking_metric_values, print_final_report
from lightgcn_static_hin import prepare_train_cache, sample_negatives


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class SageRecConfig:
    n_users: int
    n_items: int
    content_dim: int
    emb_dim: int = 128
    hidden_dim: int = 256
    gate_hidden_dim: int = 64
    sample_top_n: int = 15
    max_hist_len: int = 100
    bucket_count: int = 20
    content_weight: float = 0.35
    sampler_tau: float = 1.0
    lr: float = 1e-3
    reg_weight: float = 1e-4
    n_epochs: int = 60
    batch_size: int = 4096
    eval_interval: int = 5
    cold_threshold: int = 1
    eval_n_neg: int = 200
    static_seed: int = 2025
    seed: int = 2025
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    best_metric: str = "cold"
    best_alpha: float = 0.5
    early_stop_average_mode: str = "item_macro"
    ckpt_dir: str = ""
    save_ckpt: bool = False
    auto_resume: bool = False
    force_fresh: bool = False
    save_opt_state: bool = True

    def __post_init__(self) -> None:
        self.sample_top_n = max(1, int(self.sample_top_n))
        self.max_hist_len = max(1, int(self.max_hist_len))
        self.bucket_count = max(1, int(self.bucket_count))
        self.best_metric = self.best_metric.strip().lower()
        if self.best_metric not in {"cold", "hot", "combined", "weighted", "last"}:
            raise ValueError("best_metric must be cold, hot, combined, weighted, or last")
        self.early_stop_average_mode = self.early_stop_average_mode.strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("early_stop_average_mode must be interaction or item_macro")

    @classmethod
    def from_env(cls, n_users: int, n_items: int, content_dim: int) -> "SageRecConfig":
        static_seed = int(os.environ.get("SAGEREC_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        default_ckpt_dir = os.environ.get("BASELINE_CKPT_DIR", "").strip()
        ckpt_dir = os.environ.get("SAGEREC_CKPT_DIR", default_ckpt_dir).strip()
        save_ckpt = _env_flag("SAGEREC_SAVE_CKPT", _env_flag("BASELINE_SAVE_CKPT", bool(ckpt_dir)))
        auto_resume = _env_flag("SAGEREC_AUTO_RESUME", _env_flag("BASELINE_AUTO_RESUME", bool(ckpt_dir)))
        return cls(
            n_users=n_users,
            n_items=n_items,
            content_dim=content_dim,
            emb_dim=int(os.environ.get("SAGEREC_EMB_DIM", "128")),
            hidden_dim=int(os.environ.get("SAGEREC_HIDDEN_DIM", "256")),
            gate_hidden_dim=int(os.environ.get("SAGEREC_GATE_HIDDEN_DIM", "64")),
            sample_top_n=int(os.environ.get("SAGEREC_SAMPLE_TOP_N", "15")),
            max_hist_len=int(os.environ.get("SAGEREC_MAX_HIST_LEN", "100")),
            bucket_count=int(os.environ.get("SAGEREC_BUCKET_COUNT", "20")),
            content_weight=float(os.environ.get("SAGEREC_CONTENT_WEIGHT", "0.35")),
            sampler_tau=float(os.environ.get("SAGEREC_SAMPLER_TAU", "1.0")),
            lr=float(os.environ.get("SAGEREC_LR", "1e-3")),
            reg_weight=float(os.environ.get("SAGEREC_REG", "1e-4")),
            n_epochs=int(os.environ.get("SAGEREC_STATIC_EPOCHS", "60")),
            batch_size=int(os.environ.get("SAGEREC_BATCH_SIZE", "4096")),
            eval_interval=int(os.environ.get("SAGEREC_EVAL_INTERVAL", "5")),
            cold_threshold=int(os.environ.get("SAGEREC_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1"))),
            eval_n_neg=int(os.environ.get("SAGEREC_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200"))),
            static_seed=static_seed,
            seed=int(os.environ.get("SAGEREC_SEED", str(static_seed))),
            train_ratio=float(os.environ.get("SAGEREC_STATIC_TRAIN_RATIO", "0.8")),
            val_ratio=float(os.environ.get("SAGEREC_STATIC_VAL_RATIO", "0.1")),
            best_metric=os.environ.get("BASELINE_BEST_METRIC", "cold"),
            best_alpha=float(os.environ.get("BASELINE_BEST_ALPHA", "0.5")),
            early_stop_average_mode=os.environ.get(
                "BASELINE_EARLY_STOP_AVG_MODE",
                os.environ.get("USIM_EARLY_STOP_AVG_MODE", "item_macro"),
            ),
            ckpt_dir=ckpt_dir,
            save_ckpt=save_ckpt,
            auto_resume=auto_resume,
            force_fresh=_env_flag("SAGEREC_FORCE_FRESH", _env_flag("BASELINE_FORCE_FRESH", False)),
            save_opt_state=_env_flag("SAGEREC_SAVE_OPT_STATE", _env_flag("BASELINE_SAVE_OPT_STATE", True)),
        )


def build_padded_user_history(
    train_df: pd.DataFrame,
    n_users: int,
    max_hist_len: int,
    pad_item_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_hist_len < 1:
        raise ValueError("max_hist_len must be positive")

    hist = torch.full((n_users, max_hist_len), int(pad_item_id), dtype=torch.long)
    mask = torch.zeros((n_users, max_hist_len), dtype=torch.bool)
    if train_df.empty:
        return hist, mask

    cols = ["u_idx", "i_idx"]
    if "timestamp" in train_df.columns:
        cols.append("timestamp")
        ordered = train_df[cols].sort_values(["u_idx", "timestamp"], kind="mergesort")
    else:
        ordered = train_df[cols].copy()
        ordered["__row_order"] = np.arange(len(ordered))
        ordered = ordered.sort_values(["u_idx", "__row_order"], kind="mergesort")

    for uid, group in ordered.groupby("u_idx", sort=False):
        uid_i = int(uid)
        if uid_i < 0 or uid_i >= n_users:
            continue
        items = [int(x) for x in group["i_idx"].tolist()]
        items = items[-max_hist_len:]
        if not items:
            continue
        hist[uid_i, : len(items)] = torch.tensor(items, dtype=torch.long)
        mask[uid_i, : len(items)] = True
    return hist, mask


class SageRecModel(nn.Module):
    def __init__(self, cfg: SageRecConfig, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.pad_item_id = cfg.n_items

        content = self._fit_content_matrix(content_emb.float())
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.emb_dim, padding_idx=self.pad_item_id)
        self.item_con_emb = nn.Embedding.from_pretrained(content, freeze=True, padding_idx=self.pad_item_id)

        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )

        self.sampler_user = nn.ModuleList(
            [nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False) for _ in range(2)]
        )
        self.sampler_item = nn.ModuleList(
            [nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False) for _ in range(2)]
        )

        self.expert_user = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
                    nn.GELU(),
                    nn.LayerNorm(cfg.emb_dim),
                )
                for _ in range(2)
            ]
        )
        self.expert_item = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.emb_dim, cfg.emb_dim),
                    nn.GELU(),
                    nn.LayerNorm(cfg.emb_dim),
                )
                for _ in range(2)
            ]
        )

        self.gate_bucket_emb = nn.Embedding(cfg.bucket_count, cfg.gate_hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.gate_hidden_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 2),
        )

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[self.pad_item_id].zero_()

    def _fit_content_matrix(self, content_emb: torch.Tensor) -> torch.Tensor:
        if content_emb.dim() != 2:
            raise ValueError("content_emb must be a 2D tensor")
        if content_emb.size(1) != self.cfg.content_dim:
            raise ValueError(
                f"content_emb dim mismatch: got {content_emb.size(1)}, expected {self.cfg.content_dim}"
            )
        if content_emb.size(0) >= self.cfg.n_items:
            content = content_emb[: self.cfg.n_items]
        else:
            pad_rows = torch.zeros(
                self.cfg.n_items - content_emb.size(0),
                self.cfg.content_dim,
                dtype=content_emb.dtype,
                device=content_emb.device,
            )
            content = torch.cat([content_emb, pad_rows], dim=0)
        pad = torch.zeros(1, self.cfg.content_dim, dtype=content.dtype, device=content.device)
        return torch.cat([content, pad], dim=0)

    def get_item_bank(self, include_pad: bool = False) -> torch.Tensor:
        item_id = self.item_emb.weight
        item_con = self.content_proj(self.item_con_emb.weight)
        bank = item_id + self.cfg.content_weight * item_con
        bank = bank.clone()
        bank[self.pad_item_id] = 0.0
        if include_pad:
            return bank
        return bank[: self.cfg.n_items]

    def _history_vectors(self, hist: torch.Tensor) -> torch.Tensor:
        item_bank = self.get_item_bank(include_pad=True)
        safe_hist = hist.clamp(min=0, max=self.pad_item_id)
        return item_bank[safe_hist]

    def _sampler_logits(
        self,
        expert_idx: int,
        base_user: torch.Tensor,
        hist_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.sampler_user[expert_idx](base_user)
        k = self.sampler_item[expert_idx](hist_vec)
        logits = (k * q.unsqueeze(1)).sum(dim=-1) / max(1.0, self.cfg.emb_dim ** 0.5)
        return logits.masked_fill(~mask, -1e9)

    def _soft_weights(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tau = max(1e-6, float(self.cfg.sampler_tau))
        soft = torch.softmax(logits / tau, dim=1) * mask.float()
        return soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _topn_hard_weights(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hist_len = logits.size(1)
        k = min(self.cfg.sample_top_n, hist_len)
        top_idx = torch.topk(logits, k=k, dim=1).indices
        hard = torch.zeros_like(logits).scatter(1, top_idx, 1.0) * mask.float()
        return hard / hard.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _expert_history_reps(
        self,
        base_user: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_vec = self._history_vectors(hist)

        logits_e1 = self._sampler_logits(0, base_user, hist_vec, mask)
        soft_e1 = self._soft_weights(logits_e1, mask)
        hard_e1 = self._topn_hard_weights(logits_e1, mask)
        st_e1 = hard_e1 + soft_e1 - soft_e1.detach()
        z1 = torch.bmm(st_e1.unsqueeze(1), hist_vec).squeeze(1)

        logits_e2 = self._sampler_logits(1, base_user, hist_vec, mask)
        soft_e2 = self._soft_weights(logits_e2, mask)
        hard_e2 = self._topn_hard_weights(logits_e2, mask)
        top_soft = soft_e2 * (hard_e2 > 0).float()
        top_soft = top_soft / top_soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
        z2 = torch.bmm(top_soft.unsqueeze(1), hist_vec).squeeze(1)

        return z1, z2

    def expert_user_representations(
        self,
        users: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_user = self.user_emb(users)
        z1, z2 = self._expert_history_reps(base_user, hist, mask)
        e1 = self.expert_user[0](torch.cat([base_user, z1], dim=1))
        e2 = self.expert_user[1](torch.cat([base_user, z2], dim=1))
        return F.normalize(e1, dim=1), F.normalize(e2, dim=1)

    def _lookup_popularity(self, item_idx: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
        pop = popularity.to(device=item_idx.device, dtype=torch.float32)
        if pop.dim() == 1 and pop.numel() >= self.cfg.n_items:
            safe_idx = item_idx.clamp(min=0, max=self.cfg.n_items - 1)
            return pop[safe_idx]
        if pop.numel() == item_idx.numel():
            return pop.reshape(item_idx.shape)
        if pop.numel() == 1:
            return pop.reshape(1).expand_as(item_idx).float()
        raise ValueError("popularity must be a full item vector, scalar, or match item_idx shape")

    def gate_weights_for_items(self, item_idx: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
        pop_values = self._lookup_popularity(item_idx, popularity)
        pop_full = popularity.to(device=item_idx.device, dtype=torch.float32).reshape(-1)
        max_pop = pop_full.max().clamp_min(0.0)
        bucket = torch.floor(pop_values * self.cfg.bucket_count / (max_pop + 1.0)).long()
        bucket = bucket.clamp(min=0, max=self.cfg.bucket_count - 1)
        gate_logits = self.gate_mlp(self.gate_bucket_emb(bucket))
        return torch.softmax(gate_logits, dim=-1)

    def _expert_item_representations(self, item_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        item_bank = self.get_item_bank(include_pad=False)
        item_vec = item_bank[item_idx.clamp(min=0, max=self.cfg.n_items - 1)]
        e1 = F.normalize(self.expert_item[0](item_vec), dim=-1)
        e2 = F.normalize(self.expert_item[1](item_vec), dim=-1)
        return e1, e2

    def score_items(
        self,
        users: torch.Tensor,
        item_idx: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
        popularity: torch.Tensor,
    ) -> torch.Tensor:
        e1_user, e2_user = self.expert_user_representations(users, hist, mask)
        e1_item, e2_item = self._expert_item_representations(item_idx)
        gate = self.gate_weights_for_items(item_idx, popularity)

        if item_idx.dim() == 1:
            s1 = torch.mm(e1_user, e1_item.t())
            s2 = torch.mm(e2_user, e2_item.t())
            return gate[:, 0].unsqueeze(0) * s1 + gate[:, 1].unsqueeze(0) * s2

        if item_idx.dim() == 2:
            s1 = (e1_user.unsqueeze(1) * e1_item).sum(dim=-1)
            s2 = (e2_user.unsqueeze(1) * e2_item).sum(dim=-1)
            return gate[..., 0] * s1 + gate[..., 1] * s2

        raise ValueError("item_idx must be 1D or 2D")

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
        popularity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cand = torch.stack([pos_items, neg_items], dim=1)
        scores = self.score_items(users, cand, hist, mask, popularity)
        return scores[:, 0], scores[:, 1]


def build_item_popularity(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    pop = torch.zeros(n_items, dtype=torch.float32)
    if train_df.empty:
        return pop
    counts = train_df.groupby("i_idx").size()
    for item_id, count in counts.items():
        item_i = int(item_id)
        if 0 <= item_i < n_items:
            pop[item_i] = float(count)
    return pop


def _mask_history_targets(
    hist: torch.Tensor,
    mask: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
) -> torch.Tensor:
    return mask & (hist != pos_items.unsqueeze(1)) & (hist != neg_items.unsqueeze(1))


def _bpr_loss(
    model: SageRecModel,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    hist: torch.Tensor,
    mask: torch.Tensor,
    popularity: torch.Tensor,
    reg_weight: float,
) -> torch.Tensor:
    pos_scores, neg_scores = model(users, pos_items, neg_items, hist, mask, popularity)
    loss_rec = -F.logsigmoid(pos_scores - neg_scores).mean()
    loss_reg = (
        model.user_emb(users).pow(2).sum(dim=1)
        + model.item_emb(pos_items).pow(2).sum(dim=1)
        + model.item_emb(neg_items).pow(2).sum(dim=1)
    ).mean()
    return loss_rec + reg_weight * loss_reg


def _state_dict_to_cpu(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_checkpoint(
    cfg: SageRecConfig,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    best_epoch: int,
    best_state: Optional[dict],
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
        "best_state": copy.deepcopy(best_state),
        "best_cold_at_best": float(best_cold_at_best),
        "best_hot_at_best": float(best_hot_at_best),
        "config": asdict(cfg),
    }
    torch.save(payload, os.path.join(cfg.ckpt_dir, filename))


def _try_resume_checkpoint(
    cfg: SageRecConfig,
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


def _best_score(cfg: SageRecConfig, epoch: int, cold_n10: float, hot_n10: float) -> float:
    if cfg.best_metric == "cold":
        return cold_n10
    if cfg.best_metric == "hot":
        return hot_n10
    if cfg.best_metric == "combined":
        return cold_n10 + hot_n10
    if cfg.best_metric == "weighted":
        return cfg.best_alpha * cold_n10 + (1.0 - cfg.best_alpha) * hot_n10
    if cfg.best_metric == "last":
        return float(epoch)
    return cold_n10


def evaluate_sagerec_ranker(
    loader,
    model: SageRecModel,
    hist_all: torch.Tensor,
    mask_all: torch.Tensor,
    item_popularity: torch.Tensor,
    device: torch.device,
    n_items: int,
    cold_threshold: int,
    k_list=(5, 10, 20),
    n_neg: int = 200,
    eval_type: str = "cold",
    full_ranking: bool = False,
    user_seen_items: Optional[Dict[int, set]] = None,
    average_mode: str = "interaction",
) -> Tuple[Optional[Dict[str, float]], int]:
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be interaction or item_macro")

    accum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in k_list}
    total_samples = 0
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    item_counts: Dict[int, int] = {}
    seen_tensor_cache: Dict[int, Optional[torch.Tensor]] = {}

    model.eval()
    hist_all = hist_all.to(device)
    mask_all = mask_all.to(device)
    item_popularity = item_popularity.to(device)

    with torch.no_grad():
        all_item_idx = torch.arange(n_items, device=device, dtype=torch.long)

        for batch, pop in loader:
            if eval_type == "cold":
                sel = pop < cold_threshold
            elif eval_type == "hot":
                sel = pop >= cold_threshold
            else:
                sel = torch.ones_like(pop, dtype=torch.bool)

            n_sel = int(sel.sum().item())
            if n_sel < 1:
                continue

            u = batch["u"][sel].to(device)
            i = batch["i"][sel].to(device)
            hist = hist_all[u]
            mask = mask_all[u]
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

            if user_seen_items is not None:
                for uid in user_ids:
                    if uid in seen_tensor_cache:
                        continue
                    seen_items = user_seen_items.get(uid)
                    if seen_items:
                        seen_idx = [x for x in seen_items if 0 <= x < n_items]
                        seen_tensor_cache[uid] = (
                            torch.tensor(seen_idx, dtype=torch.long, device=device)
                            if seen_idx else None
                        )
                    else:
                        seen_tensor_cache[uid] = None

            if full_ranking:
                scores = model.score_items(u, all_item_idx, hist, mask, item_popularity)
                if user_seen_items is not None:
                    row_idx = torch.arange(n_sel, device=device)
                    target_scores = scores[row_idx, i].clone()
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache.get(uid)
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        seen_ex_tgt = int((seen_idx != i[row]).sum().item())
                        avail = n_items - 1 - seen_ex_tgt
                    avail_counts.append(max(1, avail))

                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is not None and seen_idx.numel() > 0:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]

                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                perm = torch.argsort(torch.rand(n_sel, cand_idx.size(1), device=device), dim=1)
                cand_idx = cand_idx.gather(1, perm)
                scores = model.score_items(u, cand_idx, hist, mask, item_popularity)
                target_indices = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            batch_values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
            if average_mode == "item_macro":
                item_ids = [int(x) for x in i.detach().cpu().tolist()]
                for row, item_id in enumerate(item_ids):
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    for key, values in batch_values.items():
                        per_item = item_accum[key]
                        per_item[item_id] = per_item.get(item_id, 0.0) + float(values[row].detach().cpu().item())
            else:
                for key, values in batch_values.items():
                    accum[key] += float(values.sum().detach().cpu().item())
            total_samples += n_sel

    if total_samples < 1:
        return None, 0
    if average_mode == "item_macro":
        if not item_counts:
            return None, 0
        macro = {}
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        return macro, len(item_counts)
    return {key: val / total_samples for key, val in accum.items()}, total_samples


def main() -> None:
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    print(f"Loading data from {data_dir} ...")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = SageRecConfig.from_env(meta["n_users"], meta["n_items"], content_dim=int(content_emb.shape[1]))
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
    test_history_df = train_df
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        add_user_seen_from_df(test_seen, val_df)
        test_history_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    train_hist, train_hist_mask = build_padded_user_history(
        train_df,
        n_users=cfg.n_users,
        max_hist_len=cfg.max_hist_len,
        pad_item_id=cfg.n_items,
    )
    test_hist, test_hist_mask = build_padded_user_history(
        test_history_df,
        n_users=cfg.n_users,
        max_hist_len=cfg.max_hist_len,
        pad_item_id=cfg.n_items,
    )
    item_popularity = build_item_popularity(train_df, cfg.n_items)

    train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SageRecModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(
        "Model: SAGERec static | "
        f"device={device} | epochs={cfg.n_epochs} | emb_dim={cfg.emb_dim} | "
        f"top_n={cfg.sample_top_n} | max_hist_len={cfg.max_hist_len} | buckets={cfg.bucket_count}"
    )
    print(
        f"Best-epoch strategy: {cfg.best_metric}"
        + (f" (alpha={cfg.best_alpha})" if cfg.best_metric == "weighted" else "")
        + f" | avg_mode={cfg.early_stop_average_mode}"
    )

    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    train_hist = train_hist.to(device)
    train_hist_mask = train_hist_mask.to(device)
    item_popularity = item_popularity.to(device)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_cold_at_best = float("nan")
    best_hot_at_best = float("nan")
    start_epoch, best_val, best_epoch, best_state, best_cold_at_best, best_hot_at_best = _try_resume_checkpoint(
        cfg,
        model,
        optimizer,
        device,
    )

    k_list = [5, 10, 20]
    n_train = train_users_t.numel()
    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        train_neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
        train_neg_t = torch.tensor(train_neg_np, dtype=torch.long, device=device)

        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            u_batch = train_users_t[idx]
            p_batch = train_pos_t[idx]
            n_batch = train_neg_t[idx]
            hist = train_hist[u_batch]
            hist_mask = _mask_history_targets(train_hist[u_batch], train_hist_mask[u_batch], p_batch, n_batch)

            optimizer.zero_grad()
            loss = _bpr_loss(
                model,
                u_batch,
                p_batch,
                n_batch,
                hist,
                hist_mask,
                item_popularity,
                reg_weight=cfg.reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)

        do_eval = ((epoch + 1) % cfg.eval_interval == 0) or (epoch + 1 == cfg.n_epochs)
        if do_eval:
            val_full_cold, _ = evaluate_sagerec_ranker(
                val_loader,
                model=model,
                hist_all=train_hist,
                mask_all=train_hist_mask,
                item_popularity=item_popularity,
                device=device,
                n_items=cfg.n_items,
                cold_threshold=cfg.cold_threshold,
                k_list=k_list,
                n_neg=cfg.eval_n_neg,
                eval_type="cold",
                full_ranking=True,
                user_seen_items=train_seen,
                average_mode=cfg.early_stop_average_mode,
            )
            val_cold_n10 = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0
            val_hot_n10 = float("nan")
            if cfg.best_metric in {"hot", "combined", "weighted"}:
                val_full_hot, _ = evaluate_sagerec_ranker(
                    val_loader,
                    model=model,
                    hist_all=train_hist,
                    mask_all=train_hist_mask,
                    item_popularity=item_popularity,
                    device=device,
                    n_items=cfg.n_items,
                    cold_threshold=cfg.cold_threshold,
                    k_list=k_list,
                    n_neg=cfg.eval_n_neg,
                    eval_type="hot",
                    full_ranking=True,
                    user_seen_items=train_seen,
                    average_mode=cfg.early_stop_average_mode,
                )
                val_hot_n10 = val_full_hot.get("N@10", 0.0) if val_full_hot else 0.0
            val_key = _best_score(cfg, epoch + 1, val_cold_n10, val_hot_n10)
            improved = val_key > best_val
            if improved:
                best_val = val_key
                best_epoch = epoch + 1
                best_state = _state_dict_to_cpu(model)
                best_cold_at_best = val_cold_n10
                best_hot_at_best = val_hot_n10
                if cfg.save_ckpt:
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

    if best_state is None:
        best_epoch = cfg.n_epochs
        best_state = _state_dict_to_cpu(model)
    model.load_state_dict(best_state)
    print(
        f"Restore best epoch={best_epoch} (metric={cfg.best_metric}, score={best_val:.4f}, "
        f"cold@best={best_cold_at_best:.4f}, hot@best={best_hot_at_best:.4f})"
    )

    model.eval()
    sample_cold, n_sc = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=False,
        user_seen_items=test_seen,
    )
    sample_hot, n_sh = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=False,
        user_seen_items=test_seen,
    )
    full_cold, n_fc = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=test_seen,
    )
    full_hot, n_fh = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=test_seen,
    )
    full_cold_item_macro, n_fc_item_macro = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=test_seen,
        average_mode="item_macro",
    )
    full_hot_item_macro, n_fh_item_macro = evaluate_sagerec_ranker(
        test_loader,
        model,
        test_hist,
        test_hist_mask,
        item_popularity,
        device,
        cfg.n_items,
        cfg.cold_threshold,
        k_list=k_list,
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=test_seen,
        average_mode="item_macro",
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
        title="SAGERec Static HIN",
    )

    out = {
        "model": "SAGERec",
        "model_display": "SAGERec",
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
        "config": asdict(cfg),
    }
    result_path = static_result_path("sagerec_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
