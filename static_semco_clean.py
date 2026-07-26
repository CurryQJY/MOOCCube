"""SEMCo-style pure-content cold scorer under the clean static protocol.

Reference (local PDF + arXiv 2604.12990, SIGIR 2026):
  Sparse Contrastive Learning for Content-Based Cold Item Recommendation
  (SEMCo): content-only item encoder; user = RY (history of content vectors);
  sampled alpha-entmax contrastive loss; no CF-ID alignment of cold items.

What we take from SEMCo:
  - No item ID embedding / no ID<->content InfoNCE aux
  - Item bank = content encoder only (L2-normalized)
  - User vector = mean of train-history content vectors (leave-one-out in train)
  - Sampled entmax (alpha=1.5 default) instead of dense softmax

What we keep from this repo's clean protocol:
  - load_shared_static_split (strict_item_cold_balanced)
  - full-catalog single-vector ranking, item-macro cold/hot/overall
  - early-stop on val cold N@10

Isolation: does not modify static_content_scorer_clean.py or semco_static_hin.py.
Entmax helpers are inlined (same math as tests/test_semco_static_hin.py).
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
class SEMCoCleanConfig:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.content_dim = int(content_dim)
        self.emb_dim = 128
        self.hidden_dim = 256
        self.dropout = 0.1
        # SEMCo training
        self.temp = 0.10
        self.entmax_alpha = 1.5          # 1.0=softmax, 1.5=entmax, 2.0=sparsemax
        self.entmax_iter = 50
        self.loss_mode = "fy"            # Fenchel-Young (paper default spirit)
        self.n_neg = 64
        self.detach_query = True
        self.exclude_train_target = True
        self.singleton_policy = "global"  # if LOO empties history, use global mean
        # protocol
        self.cold_threshold = 1
        self.batch_size = 2048
        self.eval_batch_users = 512
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.grad_clip = 5.0
        self.epochs = 60
        self.patience = 15
        self.min_delta = 1e-4


# --------------------------------------------------------------------------- #
# Model: content encoder only
# --------------------------------------------------------------------------- #
class ContentOnlyEncoder(nn.Module):
    def __init__(self, cfg: SEMCoCleanConfig, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("content_features", content_emb.float())
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg.content_dim),
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_all(self) -> torch.Tensor:
        """(n_items, d) L2-normalized content codes Y."""
        return F.normalize(self.encoder(self.content_features), dim=1)

    def encode_idx(self, i_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(self.content_features[i_idx]), dim=1)


# --------------------------------------------------------------------------- #
# Entmax (alpha-entmax via bisection) + sampled loss
# --------------------------------------------------------------------------- #
def entmax_bisect(logits: torch.Tensor, alpha: float = 1.5, n_iter: int = 50) -> torch.Tensor:
    if abs(alpha - 1.0) < 1e-6:
        return torch.softmax(logits, dim=-1)
    if alpha <= 1.0:
        raise ValueError("entmax alpha must be > 1 for sparse branch")
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
    """Paper Eq.(3)/(7): Fenchel-Young loss for alpha-entmax; alpha=1 -> CE."""
    if abs(alpha - 1.0) < 1e-6:
        return F.cross_entropy(logits, target)
    probs = entmax_bisect(logits, alpha=alpha, n_iter=n_iter)
    if mode == "nll":
        target_prob = probs.gather(1, target.view(-1, 1)).squeeze(1)
        return -torch.log(target_prob.clamp_min(1e-12)).mean()
    # FY: (p̂·z) - H_alpha(p̂) - p·z  with H via omega form used in repo
    omega = (probs.pow(alpha).sum(dim=-1) - 1.0) / (alpha * (alpha - 1.0))
    target_score = logits.gather(1, target.view(-1, 1)).squeeze(1)
    loss = (probs * logits).sum(dim=-1) - omega - target_score
    return loss.mean()


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


def build_user_item_count_matrix(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    users_np = train_df["u_idx"].to_numpy(np.int64, copy=True)
    items_np = train_df["i_idx"].to_numpy(np.int64, copy=True)
    indices = torch.tensor(np.vstack([users_np, items_np]), dtype=torch.long, device=device)
    values = torch.ones(users_np.shape[0], dtype=torch.float32, device=device)
    count_mat = torch.sparse_coo_tensor(indices, values, (n_users, n_items), device=device).coalesce()
    hist_counts = torch.zeros(n_users, dtype=torch.float32, device=device)
    hist_counts.scatter_add_(0, torch.tensor(users_np, dtype=torch.long, device=device), values)
    pair_counts = (
        train_df.groupby(["u_idx", "i_idx"])["i_idx"].transform("size").to_numpy(np.float32, copy=True)
    )
    return count_mat, hist_counts, pair_counts


def sample_negatives_fast(batch_pos: np.ndarray, n_items: int, n_neg: int) -> np.ndarray:
    """Uniform catalog negatives (vectorized). Collisions with positives are shifted.

    For n_items~700 this is nearly as clean as per-user seen-exclusion and far
    cheaper than Python per-row pools over 199k users.
    """
    b = batch_pos.shape[0]
    neg = np.random.randint(0, n_items, size=(b, n_neg), dtype=np.int64)
    pos = batch_pos.reshape(-1, 1)
    hit = neg == pos
    if hit.any():
        neg = np.where(hit, (pos + 1 + np.random.randint(0, max(1, n_items - 1), size=neg.shape)) % n_items, neg)
    return neg


def training_profiles(
    cfg: SEMCoCleanConfig,
    profile_sum_bank: torch.Tensor,
    hist_counts: torch.Tensor,
    item_vectors: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    pos_pair_counts: torch.Tensor,
) -> torch.Tensor:
    """User query = mean content of history (optional LOO of the positive)."""
    query_items = item_vectors.detach() if cfg.detach_query else item_vectors
    raw_sum = profile_sum_bank[users]
    raw_count = hist_counts[users]
    if cfg.exclude_train_target:
        pos_count = pos_pair_counts.to(dtype=raw_sum.dtype, device=raw_sum.device)
        loo_sum = raw_sum - query_items[pos_items] * pos_count.view(-1, 1)
        loo_count = raw_count - pos_count
        if cfg.singleton_policy == "global":
            global_profile = query_items.mean(dim=0, keepdim=True).expand_as(loo_sum)
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


def build_profile_sum_bank(
    user_item_counts: torch.Tensor,
    item_vectors: torch.Tensor,
    detach_query: bool = True,
) -> torch.Tensor:
    items = item_vectors.detach() if detach_query else item_vectors
    return torch.sparse.mm(user_item_counts, items)


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


# --------------------------------------------------------------------------- #
# Metrics (item-macro), same contract as static_content_scorer_clean
# --------------------------------------------------------------------------- #
def _dcg_hit(rank_pos: int, k: int) -> float:
    if rank_pos < k:
        return 1.0 / np.log2(rank_pos + 2.0)
    return 0.0


@torch.no_grad()
def evaluate_clean(
    model: ContentOnlyEncoder,
    eval_df: pd.DataFrame,
    device: torch.device,
    train_pop: np.ndarray,
    user_seen: dict[int, set[int]],
    profile_bank: torch.Tensor,
    item_bank: torch.Tensor | None = None,
    k_list=(5, 10, 20),
) -> dict:
    cfg = model.cfg
    model.eval()
    if item_bank is None:
        item_bank = model.encode_all()
    bank = item_bank  # already L2-normalized
    max_k = max(k_list)
    acc = {f"{m}@{k}": {} for m in ("R", "N") for k in k_list}
    cnt: dict[int, int] = {}
    cold_np = train_pop < cfg.cold_threshold
    users = eval_df["u_idx"].to_numpy()
    items = eval_df["i_idx"].to_numpy()
    bs = cfg.eval_batch_users
    for start in range(0, len(users), bs):
        u_batch = users[start:start + bs]
        i_batch = items[start:start + bs]
        u_t = torch.as_tensor(u_batch, dtype=torch.long, device=device)
        z_u = profile_bank.index_select(0, u_t)
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
            rank_pos = int(np.where(row == tgt)[0][0]) if (row == tgt).any() else max_k + 1
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
# Train
# --------------------------------------------------------------------------- #
def train(
    cfg: SEMCoCleanConfig,
    train_df,
    val_df,
    test_df,
    content_emb,
    device,
    out_dir: Path,
):
    model = ContentOnlyEncoder(cfg, content_emb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_pop = compute_train_popularity(train_df, cfg.n_items)
    user_seen = build_user_seen(train_df)
    user_item_counts, hist_counts, pair_counts_np = build_user_item_count_matrix(
        train_df, cfg.n_users, cfg.n_items, device
    )

    u_all_np = train_df["u_idx"].to_numpy(np.int64, copy=True)
    i_all_np = train_df["i_idx"].to_numpy(np.int64, copy=True)
    u_all = torch.as_tensor(u_all_np, dtype=torch.long, device=device)
    i_all = torch.as_tensor(i_all_np, dtype=torch.long, device=device)
    pair_all = torch.as_tensor(pair_counts_np, dtype=torch.float32, device=device)
    n_train = u_all.size(0)

    best_score = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    val_history = []

    print(
        f"[semco-clean] alpha={cfg.entmax_alpha} temp={cfg.temp} n_neg={cfg.n_neg} "
        f"detach_query={cfg.detach_query} epochs={cfg.epochs}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        nb = 0
        t0 = time.time()
        # Epoch-level content bank for user profiles (SEMCo detach_query).
        with torch.no_grad():
            y_epoch = model.encode_all()
            profile_sum = build_profile_sum_bank(
                user_item_counts, y_epoch, detach_query=True
            ).detach()
            y_epoch = y_epoch.detach()

        for start in range(0, n_train, cfg.batch_size):
            b = perm[start:start + cfg.batch_size]
            b_cpu = b.detach().cpu().numpy()
            u_idx = u_all[b]
            i_idx = i_all[b]
            pair_c = pair_all[b]
            neg_np = sample_negatives_fast(i_all_np[b_cpu], cfg.n_items, cfg.n_neg)
            neg_items = torch.as_tensor(neg_np, dtype=torch.long, device=device)

            opt.zero_grad()
            # Grads only through pos/neg content codes; query is detached history mean.
            query = training_profiles(
                cfg, profile_sum, hist_counts, y_epoch, u_idx, i_idx, pair_c
            )
            candidates = torch.cat([i_idx.view(-1, 1), neg_items], dim=1)
            cand_vec = model.encode_idx(candidates.reshape(-1)).view(
                candidates.size(0), candidates.size(1), -1
            )
            # SEMCo: cosine / temperature (same as repo semco_static_hin).
            logits = torch.bmm(cand_vec, query.unsqueeze(2)).squeeze(2) / max(cfg.temp, 1e-6)
            target = torch.zeros(u_idx.numel(), dtype=torch.long, device=device)
            loss = sampled_entmax_loss(
                logits, target, alpha=cfg.entmax_alpha, n_iter=cfg.entmax_iter, mode=cfg.loss_mode
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={nb}")
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.detach())
            nb += 1
            if nb == 1 or (nb % 50 == 0):
                print(
                    f"  epoch {epoch} batch {nb} loss={float(loss.detach()):.4f}",
                    flush=True,
                )

        with torch.no_grad():
            item_bank = model.encode_all()
            psum = build_profile_sum_bank(user_item_counts, item_bank, detach_query=True)
            profile_bank = build_eval_profile_bank(psum, hist_counts, item_bank)
            val = evaluate_clean(
                model, val_df, device, train_pop, user_seen, profile_bank, item_bank
            )
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Persist best immediately so a mid-run crash still leaves a checkpoint.
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out_dir / "best.pt")
            with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
                json.dump(val_history, f, indent=2)
            print(
                f"  * new best epoch={best_epoch} cold_N@10={best_score:.4f} -> {out_dir/'best.pt'}",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
        json.dump(val_history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best.pt")
    with torch.no_grad():
        item_bank = model.encode_all()
        psum = build_profile_sum_bank(user_item_counts, item_bank, detach_query=True)
        profile_bank = build_eval_profile_bank(psum, hist_counts, item_bank)
        test = evaluate_clean(
            model, test_df, device, train_pop, user_seen, profile_bank, item_bank
        )
    print(f"\n=== TEST SEMCo-clean — best epoch {best_epoch} ===")
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
                "method": "static_semco_clean",
                "reference": "SEMCo SIGIR 2026 arXiv:2604.12990",
                "entmax_alpha": cfg.entmax_alpha,
                "temp": cfg.temp,
                "n_neg": cfg.n_neg,
                "evaluator": "clean_single_vector_full_catalog_item_macro",
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
    ap.add_argument("--alpha", type=float, default=1.5, help="entmax alpha (1=softmax, 1.5, 2=sparsemax)")
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--n-neg", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    content_emb = torch.load(os.path.join(args.data_dir, "content_emb.pt"), map_location="cpu")
    if not isinstance(content_emb, torch.Tensor):
        content_emb = torch.as_tensor(content_emb)
    content_emb = content_emb.float()

    train_df, val_df, test_df, split_info = load_shared_static_split(args.split_dir)
    cfg = SEMCoCleanConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.entmax_alpha = float(args.alpha)
    cfg.temp = float(args.temp)
    cfg.n_neg = int(args.n_neg)
    cfg.patience = int(args.patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"n_users={cfg.n_users} n_items={cfg.n_items} content_dim={cfg.content_dim} "
        f"device={device} alpha={cfg.entmax_alpha}"
    )
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"split_info seed={split_info.get('seed')} cold_def={split_info.get('cold_definition')}")

    if args.dry_run:
        train_pop = compute_train_popularity(train_df, cfg.n_items)
        n_cold = int((train_pop < cfg.cold_threshold).sum())
        print(f"[dry-run] cold items: {n_cold}/{cfg.n_items}")
        # smoke: one forward of entmax
        z = torch.randn(4, 8)
        p = entmax_bisect(z, alpha=cfg.entmax_alpha)
        assert torch.allclose(p.sum(dim=1), torch.ones(4), atol=1e-4)
        print("[dry-run] OK")
        return

    try:
        train(cfg, train_df, val_df, test_df, content_emb, device, Path(args.output_dir))
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
