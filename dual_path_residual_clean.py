"""AlphaFuse-style dual-path residual scorer under the clean static protocol.

Reference (local PDF + arXiv 2504.19218, SIGIR 2025):
  AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space
  of Language Embeddings.

Core idea we take from AlphaFuse (single model, no late fusion of two models):
  1) SVD on content (language) embeddings -> semantic-rich row space +
     semantic-sparse null space.
  2) Frozen standardized language codes occupy the full emb_dim = d_s + d_n.
  3) Trainable ID residual lives only in the last d_n (null) dimensions.
  4) Final item vector = language_code + [0_{d_s} || id_residual].
  5) Strict cold items: residual forced to 0 (pure content path).
  6) Warm items: residual free (content + collaborative null residual).

What we keep from this repo's clean protocol:
  - load_shared_static_split (strict_item_cold_balanced)
  - full-catalog single-vector ranking, item-macro cold/hot/overall
  - early-stop on val cold N@10
  - dual-tower InfoNCE (user ID emb + item residual bank), not SASRec sequence

Isolation: does not modify static_content_scorer_clean.py / static_semco_clean.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast3_delta.static_protocol import load_shared_static_split


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class ResidualConfig:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.content_dim = int(content_dim)
        # AlphaFuse defaults for discriminative backbone (paper: emb=128, id=64)
        self.emb_dim = 128          # d_s + d_n
        self.null_dim = 64          # d_n  (learnable residual rank)
        # null_threshold: if set, singular values below this (after norm max=1)
        # are treated as null; otherwise take the smallest null_dim directions
        # after ranking by singular value descending (paper alternative).
        self.null_threshold: float | None = None
        # Loss / training (aligned with static_content_scorer_clean spirit)
        self.temp = 0.07
        self.margin = 0.15
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        # Optional: residual dropout during training (simulate cold on warm)
        self.residual_dropout_prob = 0.0
        # protocol
        self.cold_threshold = 1
        self.batch_size = 2048
        self.eval_batch_users = 512
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.epochs = 60
        self.patience = 20
        self.min_delta = 1e-4

    @property
    def row_dim(self) -> int:
        return int(self.emb_dim - self.null_dim)


# --------------------------------------------------------------------------- #
# AlphaFuse language-space decomposition
# --------------------------------------------------------------------------- #
def decompose_language_space(
    content_emb: torch.Tensor,
    emb_dim: int,
    null_dim: int,
    null_threshold: float | None = None,
) -> dict:
    """SVD of content covariance -> frozen language codes in R^{emb_dim}.

    Paper Eq.(5)-(9):
      mu = mean(E)
      Sigma = cov(E)
      U S U^T = SVD(Sigma)   (we use eigh; eigenvalues = squared singular vals)
      E_lang = (E - mu) @ U[:, :d_s+d_n] @ S^{-1/2}[:d_s+d_n]

    Dimensions:
      emb_dim = d_s + d_n  (final item dim)
      null_dim = d_n       (last dims host learnable ID residual)
      row_dim  = d_s       (first dims pure frozen semantics)

    Returns dict with language_codes (n_items, emb_dim) and diagnostics.
    """
    if content_emb.dim() != 2:
        raise ValueError(f"content_emb must be 2D, got {tuple(content_emb.shape)}")
    n_items, content_dim = content_emb.shape
    if emb_dim > content_dim:
        raise ValueError(f"emb_dim={emb_dim} > content_dim={content_dim}")
    if null_dim < 0 or null_dim > emb_dim:
        raise ValueError(f"null_dim={null_dim} invalid for emb_dim={emb_dim}")
    if n_items < 2:
        raise ValueError("need at least 2 items for covariance")

    x = content_emb.float()
    mu = x.mean(dim=0)
    xc = x - mu
    # Paper uses weighted cov; we use uniform (p(v)=1/N) unbiased estimator.
    cov = (xc.t() @ xc) / max(1.0, float(n_items - 1))
    # Symmetric eigh: eigenvalues ascending; flip to descending.
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.flip(0).clamp_min(0.0)
    evecs = evecs.flip(1)

    # Singular values ~ sqrt(eigenvalues of cov). Normalize for thresholding.
    svals = torch.sqrt(evals.clamp_min(0.0))
    smax = float(svals.max().item()) if svals.numel() else 1.0
    s_norm = svals / max(smax, 1e-12)

    if null_threshold is not None:
        # Subspaces with normalized singular value < threshold -> null.
        rich_mask = s_norm >= float(null_threshold)
        n_rich = int(rich_mask.sum().item())
        # Need at least emb_dim directions after clip: take top emb_dim, of which
        # the last null_dim are "null clip" host for residual.
        # If threshold yields too few rich dims, fall back to fixed split.
        d_s = min(n_rich, emb_dim - null_dim)
        d_s = max(d_s, 1)
        d_n = emb_dim - d_s
    else:
        d_s = emb_dim - null_dim
        d_n = null_dim

    take = d_s + d_n
    U_take = evecs[:, :take]                          # (content_dim, take)
    # Standardization: divide by sqrt(s_i) = svals^{1/2} wait: paper uses S^{-1/2}
    # on squared singular values of the centered design; with cov eigh, eigenvalue
    # lambda_i = s_i^2 of the (weighted) design, so S^{-1/2} on lambda is 1/sqrt(lambda).
    scale = 1.0 / torch.sqrt(evals[:take].clamp_min(1e-8))
    # E_lang = (E - mu) @ U @ diag(lambda^{-1/2})
    language_codes = xc @ U_take * scale.unsqueeze(0)  # (n_items, take)

    # Diagnostics: how much variance in row vs null host dims
    var_all = language_codes.var(dim=0)
    row_var = float(var_all[:d_s].sum().item()) if d_s > 0 else 0.0
    null_var = float(var_all[d_s:].sum().item()) if d_n > 0 else 0.0

    return {
        "language_codes": language_codes.contiguous(),
        "mu": mu.contiguous(),
        "U_take": U_take.contiguous(),
        "scale": scale.contiguous(),
        "evals": evals.detach().cpu(),
        "s_norm": s_norm.detach().cpu(),
        "d_s": int(d_s),
        "d_n": int(d_n),
        "row_var": row_var,
        "null_host_var": null_var,
        "smax": smax,
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DualPathResidualScorer(nn.Module):
    """User ID tower + AlphaFuse item bank (frozen language + null residual)."""

    def __init__(self, cfg: ResidualConfig, language_codes: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        if language_codes.shape != (cfg.n_items, cfg.emb_dim):
            raise ValueError(
                f"language_codes shape {tuple(language_codes.shape)} != "
                f"({cfg.n_items}, {cfg.emb_dim})"
            )
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.user_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.user_proj.weight)
        nn.init.zeros_(self.user_proj.bias)

        # Frozen language codes (full emb_dim = row || null-host init).
        self.register_buffer("language_codes", language_codes.float())
        # Learnable residual only in null dims (last d_n). Init near zero so
        # warm items start close to pure content (paper allows zero or random).
        self.item_null_residual = nn.Embedding(cfg.n_items, cfg.null_dim)
        nn.init.normal_(self.item_null_residual.weight, mean=0.0, std=0.01)

        self.d_s = cfg.row_dim
        self.d_n = cfg.null_dim

    def user_vector(self, u_idx: torch.Tensor) -> torch.Tensor:
        return self.user_proj(self.user_emb(u_idx))

    def residual_vector(
        self,
        i_idx: torch.Tensor,
        force_cold=False,
        apply_residual_dropout: bool = False,
    ) -> torch.Tensor:
        """(B, d_n) residual; zeroed for cold / dropout rows."""
        res = self.item_null_residual(i_idx)
        n = res.size(0)
        mask = torch.zeros((n, 1), dtype=torch.bool, device=res.device)
        if isinstance(force_cold, torch.Tensor):
            fm = force_cold.to(device=res.device)
            if fm.dtype != torch.bool:
                fm = fm > 0
            mask = mask | fm.view(-1, 1)
        elif force_cold:
            mask = torch.ones((n, 1), dtype=torch.bool, device=res.device)
        if (
            apply_residual_dropout
            and self.training
            and self.cfg.residual_dropout_prob > 0
        ):
            drop = torch.rand((n, 1), device=res.device) < float(
                self.cfg.residual_dropout_prob
            )
            mask = mask | drop
        if mask.any():
            res = torch.where(mask, torch.zeros_like(res), res)
        return res

    def item_vector(
        self,
        i_idx: torch.Tensor,
        force_cold=False,
        apply_residual_dropout: bool = False,
        return_parts: bool = False,
    ):
        """item = language + pad(residual)  [AlphaFuse Eq.(10)].

        force_cold: bool or bool tensor — residual zeroed (strict cold path).
        """
        lang = self.language_codes.index_select(0, i_idx)  # (B, emb_dim)
        res = self.residual_vector(
            i_idx, force_cold=force_cold, apply_residual_dropout=apply_residual_dropout
        )
        # pad residual into last d_n dims: [0_{d_s} || res]
        pad = torch.zeros(res.size(0), self.d_s, device=res.device, dtype=res.dtype)
        residual_full = torch.cat([pad, res], dim=-1)  # (B, emb_dim)
        fused = lang + residual_full
        if return_parts:
            return fused, lang, res
        return fused

    def encode_all(
        self,
        cold_item_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full catalog item bank (n_items, emb_dim), residual zeroed for cold."""
        n = self.cfg.n_items
        idx = torch.arange(n, device=self.language_codes.device)
        if cold_item_mask is None:
            return self.item_vector(idx, force_cold=False)
        return self.item_vector(idx, force_cold=cold_item_mask)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def build_user_seen(train_df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for u, i in zip(train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def compute_train_popularity(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int64)
    vc = train_df["i_idx"].value_counts()
    pop[vc.index.to_numpy()] = vc.to_numpy()
    return pop


# --------------------------------------------------------------------------- #
# Metrics (item-macro), same contract as static_content_scorer_clean
# --------------------------------------------------------------------------- #
def _dcg_hit(rank_pos: int, k: int) -> float:
    if rank_pos < k:
        return 1.0 / np.log2(rank_pos + 2.0)
    return 0.0


@torch.no_grad()
def evaluate_clean(
    model: DualPathResidualScorer,
    eval_df: pd.DataFrame,
    device: torch.device,
    train_pop: np.ndarray,
    user_seen: dict[int, set[int]],
    k_list=(5, 10, 20),
) -> dict:
    cfg = model.cfg
    model.eval()
    n_items = cfg.n_items
    cold_np = train_pop < cfg.cold_threshold
    cold_mask_t = torch.as_tensor(cold_np, dtype=torch.bool, device=device)

    # One vector per item. Cold residual = 0; hot residual free.
    bank = F.normalize(model.encode_all(cold_item_mask=cold_mask_t), dim=1)

    max_k = max(k_list)
    acc = {f"{m}@{k}": {} for m in ("R", "N") for k in k_list}
    cnt: dict[int, int] = {}
    users = eval_df["u_idx"].to_numpy()
    items = eval_df["i_idx"].to_numpy()
    bs = cfg.eval_batch_users
    for start in range(0, len(users), bs):
        u_batch = users[start : start + bs]
        i_batch = items[start : start + bs]
        u_t = torch.as_tensor(u_batch, dtype=torch.long, device=device)
        z_u = F.normalize(model.user_vector(u_t), dim=1)
        scores = torch.mm(z_u, bank.t())
        for r, u in enumerate(u_batch):
            seen = user_seen.get(int(u))
            if seen:
                pos = int(i_batch[r])
                for it in seen:
                    if it != pos:
                        scores[r, it] = -1e30
        _, topk = torch.topk(scores, max_k, dim=1)
        topk = topk.cpu().numpy()
        for r in range(len(u_batch)):
            tgt = int(i_batch[r])
            row = topk[r]
            rank_pos = (
                int(np.where(row == tgt)[0][0]) if (row == tgt).any() else max_k + 1
            )
            cnt[tgt] = cnt.get(tgt, 0) + 1
            for k in k_list:
                hit = 1.0 if rank_pos < k else 0.0
                acc[f"R@{k}"][tgt] = acc[f"R@{k}"].get(tgt, 0.0) + hit
                acc[f"N@{k}"][tgt] = acc[f"N@{k}"].get(tgt, 0.0) + _dcg_hit(rank_pos, k)

    def macro(metric_key: str, item_filter) -> float:
        vals = []
        for it, c in cnt.items():
            if not item_filter(it):
                continue
            vals.append(acc[metric_key].get(it, 0.0) / max(1, c))
        return float(np.mean(vals)) if vals else 0.0

    out = {}
    for split_name, filt in (
        ("cold", lambda it: cold_np[it]),
        ("hot", lambda it: not cold_np[it]),
    ):
        for k in k_list:
            out[f"{split_name}_R@{k}"] = macro(f"R@{k}", filt)
            out[f"{split_name}_N@{k}"] = macro(f"N@{k}", filt)
        out[f"{split_name}_count"] = int(sum(1 for it in cnt if filt(it)))
    for k in k_list:
        for m in ("R", "N"):
            cc, hc = out["cold_count"], out["hot_count"]
            tot = max(1, cc + hc)
            out[f"overall_{m}@{k}"] = (
                out[f"cold_{m}@{k}"] * cc + out[f"hot_{m}@{k}"] * hc
            ) / tot
    return out


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def infonce_loss(
    model: DualPathResidualScorer,
    u_idx: torch.Tensor,
    i_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    cfg = model.cfg
    z_u = F.normalize(model.user_vector(u_idx), dim=1)
    item_out = model.item_vector(
        i_idx, force_cold=False, apply_residual_dropout=True, return_parts=False
    )
    z_i = F.normalize(item_out, dim=1)
    logits = torch.mm(z_u, z_i.t()) / cfg.temp
    n = logits.size(0)
    labels = torch.arange(n, device=device)
    pos_mask = torch.eye(n, dtype=torch.bool, device=device)
    logits_m = logits.clone()
    logits_m[pos_mask] -= cfg.margin / cfg.temp
    if n <= 1:
        return F.cross_entropy(logits_m, labels)
    max_neg = n - 1
    n_total = min(cfg.train_num_negs, max_neg)
    if n_total <= 0:
        return F.cross_entropy(logits_m, labels)
    n_hard = max(0, min(int(n_total * cfg.hard_neg_ratio), n_total))
    n_rand = n_total - n_hard
    neg = logits_m.clone()
    neg[pos_mask] = -1e9
    hard_idx = torch.empty(n, 0, dtype=torch.long, device=device)
    rand_idx = torch.empty(n, 0, dtype=torch.long, device=device)
    if n_hard > 0:
        _, hard_idx = torch.topk(neg, k=n_hard, dim=1)
    if n_rand > 0:
        rs = torch.rand_like(neg)
        rs[pos_mask] = -1e9
        if n_hard > 0:
            rs.scatter_(1, hard_idx, -1e9)
        _, rand_idx = torch.topk(rs, k=n_rand, dim=1)
    cand = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
    cand_logits = logits_m.gather(1, cand)
    targets = torch.zeros(n, dtype=torch.long, device=device)
    return F.cross_entropy(cand_logits, targets)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(
    cfg: ResidualConfig,
    train_df,
    val_df,
    test_df,
    content_emb,
    device,
    out_dir: Path,
    decomp_meta: dict | None = None,
):
    print(
        f"[dual-path] decomposing language space emb_dim={cfg.emb_dim} "
        f"null_dim={cfg.null_dim} threshold={cfg.null_threshold}",
        flush=True,
    )
    decomp = decompose_language_space(
        content_emb.cpu(),
        emb_dim=cfg.emb_dim,
        null_dim=cfg.null_dim,
        null_threshold=cfg.null_threshold,
    )
    # Align cfg dims if threshold mode adjusted d_s/d_n
    cfg.null_dim = int(decomp["d_n"])
    # emb_dim stays d_s + d_n
    language_codes = decomp["language_codes"].to(device)
    print(
        f"[dual-path] d_s={decomp['d_s']} d_n={decomp['d_n']} "
        f"row_var={decomp['row_var']:.4f} null_host_var={decomp['null_host_var']:.4f} "
        f"smax={decomp['smax']:.6f}",
        flush=True,
    )

    model = DualPathResidualScorer(cfg, language_codes).to(device)
    # Only train user tower + null residual (language frozen as buffer).
    opt = torch.optim.AdamW(
        [
            {"params": model.user_emb.parameters()},
            {"params": model.user_proj.parameters()},
            {"params": model.item_null_residual.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_pop = compute_train_popularity(train_df, cfg.n_items)
    user_seen = build_user_seen(train_df)
    n_cold = int((train_pop < cfg.cold_threshold).sum())
    print(
        f"[dual-path] cold items (train pop==0): {n_cold}/{cfg.n_items}",
        flush=True,
    )

    u_all = torch.as_tensor(train_df["u_idx"].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(train_df["i_idx"].to_numpy(), dtype=torch.long)
    n_train = u_all.size(0)

    best_score = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    val_history = []

    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist decomposition diagnostics (not full U) for audit
    decomp_save = {
        "d_s": decomp["d_s"],
        "d_n": decomp["d_n"],
        "row_var": decomp["row_var"],
        "null_host_var": decomp["null_host_var"],
        "smax": decomp["smax"],
        "emb_dim": cfg.emb_dim,
        "null_threshold": cfg.null_threshold,
        "evals_top20": decomp["evals"][:20].tolist(),
        "s_norm_top20": decomp["s_norm"][:20].tolist(),
        "s_norm_tail20": decomp["s_norm"][-20:].tolist(),
    }
    if decomp_meta:
        decomp_save["meta"] = decomp_meta
    with open(out_dir / "language_decomp.json", "w", encoding="utf-8") as f:
        json.dump(decomp_save, f, indent=2)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        nb = 0
        t0 = time.time()
        for start in range(0, n_train, cfg.batch_size):
            b = perm[start : start + cfg.batch_size]
            u_idx = u_all[b].to(device)
            i_idx = i_all[b].to(device)
            opt.zero_grad()
            loss = infonce_loss(model, u_idx, i_idx, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={nb}")
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.detach())
            nb += 1

        val = evaluate_clean(model, val_df, device, train_pop, user_seen)
        score = val["cold_N@10"]
        val_history.append(
            {
                "epoch": epoch,
                "cold_N@10": score,
                "cold_R@10": val["cold_R@10"],
                "hot_N@10": val["hot_N@10"],
                "hot_R@10": val["hot_R@10"],
            }
        )
        print(
            f"[epoch {epoch}/{cfg.epochs}] loss={total_loss/max(1,nb):.4f} "
            f"time={time.time()-t0:.1f}s | val cold R@10={val['cold_R@10']:.4f} "
            f"N@10={val['cold_N@10']:.4f} | hot R@10={val['hot_R@10']:.4f} "
            f"N@10={val['hot_N@10']:.4f}",
            flush=True,
        )
        if score > best_score + cfg.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
            torch.save(best_state, out_dir / "best.pt")
            with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
                json.dump(val_history, f, indent=2)
            print(
                f"  * new best epoch={best_epoch} cold_N@10={best_score:.4f} "
                f"-> {out_dir / 'best.pt'}",
                flush=True,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(
                    f"Early stop at epoch {epoch} (best {best_epoch}, "
                    f"cold N@10={best_score:.4f})",
                    flush=True,
                )
                break

    with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
        json.dump(val_history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best.pt")

    test = evaluate_clean(model, test_df, device, train_pop, user_seen)
    print(f"\n=== TEST dual-path residual — best epoch {best_epoch} ===")
    for split in ("cold", "hot", "overall"):
        print(
            f"  {split:8s} R@10={test[f'{split}_R@10']:.4f}  "
            f"N@10={test[f'{split}_N@10']:.4f}"
        )

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "test": test,
                "method": "dual_path_residual_clean",
                "reference": "AlphaFuse SIGIR 2025 arXiv:2504.19218",
                "emb_dim": cfg.emb_dim,
                "null_dim": cfg.null_dim,
                "row_dim": cfg.row_dim,
                "null_threshold": cfg.null_threshold,
                "residual_dropout_prob": cfg.residual_dropout_prob,
                "temp": cfg.temp,
                "evaluator": "clean_single_vector_full_catalog_item_macro",
                "decomp": decomp_save,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_dir}", flush=True)
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--null-dim", type=int, default=64)
    ap.add_argument(
        "--null-threshold",
        type=float,
        default=None,
        help="optional: normalized singular-value threshold for null space",
    )
    ap.add_argument("--residual-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    content_emb = torch.load(
        os.path.join(args.data_dir, "content_emb.pt"), map_location="cpu"
    )
    if not isinstance(content_emb, torch.Tensor):
        content_emb = torch.as_tensor(content_emb)
    content_emb = content_emb.float()

    train_df, val_df, test_df, split_info = load_shared_static_split(args.split_dir)
    cfg = ResidualConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.emb_dim = int(args.emb_dim)
    cfg.null_dim = int(args.null_dim)
    cfg.null_threshold = args.null_threshold
    cfg.residual_dropout_prob = float(args.residual_dropout)
    cfg.lr = float(args.lr)
    cfg.patience = int(args.patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"n_users={cfg.n_users} n_items={cfg.n_items} content_dim={cfg.content_dim} "
        f"device={device} emb_dim={cfg.emb_dim} null_dim={cfg.null_dim}"
    )
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(
        f"split_info seed={split_info.get('seed')} "
        f"cold_def={split_info.get('cold_definition')}"
    )

    if args.dry_run:
        train_pop = compute_train_popularity(train_df, cfg.n_items)
        n_cold = int((train_pop < cfg.cold_threshold).sum())
        decomp = decompose_language_space(
            content_emb, cfg.emb_dim, cfg.null_dim, cfg.null_threshold
        )
        print(f"[dry-run] cold items: {n_cold}/{cfg.n_items}")
        print(
            f"[dry-run] decomp d_s={decomp['d_s']} d_n={decomp['d_n']} "
            f"codes={tuple(decomp['language_codes'].shape)}"
        )
        # smoke: cold residual zero, warm residual nonzero after random init
        model = DualPathResidualScorer(cfg, decomp["language_codes"])
        cold_mask = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool)
        with torch.no_grad():
            bank = model.encode_all(cold_item_mask=cold_mask)
            lang = model.language_codes
            # cold items must equal language codes exactly
            if cold_mask.any():
                diff = (bank[cold_mask] - lang[cold_mask]).abs().max().item()
                assert diff < 1e-6, f"cold residual leak: max_diff={diff}"
            # residual table is nonzero (random init)
            assert model.item_null_residual.weight.abs().sum().item() > 0
        print("[dry-run] OK")
        return

    try:
        train(
            cfg,
            train_df,
            val_df,
            test_df,
            content_emb,
            device,
            Path(args.output_dir),
            decomp_meta={"split_dir": args.split_dir, "seed": args.seed},
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
