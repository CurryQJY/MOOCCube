"""Clean static content-behavior cold-start scorer (no USIM / RL / course code).

This is an isolated re-implementation of the "static content-masked scorer"
configuration that, in the legacy USIM ablation, matched or beat the full
CKG-RL model on strict item-cold recommendation. The goal here is to reproduce
that backbone WITHOUT any of the USIM rollout / PPO / course-reward machinery,
and — critically — under a CLEAN full-catalog single-vector evaluator (no
legacy dual-vector-per-cold-item trick).

Model (exactly the pieces the static ablation actually used):
  user vector : user_proj(user_emb[u])
  item vector : gate(alpha)*id_e + (1-alpha)*content_e
                 - id_e : item_id_emb[i], zeroed per-row with prob dropout_prob
                          during training (this is the cold simulation)
                 - content_e : content_proj(frozen content_emb[i])
  loss        : in-batch InfoNCE (z_u . z_i^T / temp) with a positive margin
                and mixed hard/random negatives.

Clean evaluation contract (the whole point):
  - Build ONE item bank for the entire catalog (each item exactly one vector).
  - Cold items get id_e zeroed (force_cold); hot items keep id_e.
  - Score = user . bank^T over the full catalog (full ranking).
  - The positive's score is read straight from that same bank — the positive
    item is NEVER re-encoded into a second, different vector. This removes the
    legacy inflation where a cold positive got its own refined vector.
  - Train history is masked out of the candidate scores.
  - Metrics are item-macro (per-item mean, then mean over items) so cold (few
    items) and hot are comparable to the paper's main tables.

Isolation: imports only torch/pandas/numpy + `load_shared_static_split` (to
reuse the exact same strict split, guaranteeing comparability). No fast3 model
code, no eval.py, no course artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
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
class ScorerConfig:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.content_dim = int(content_dim)
        # Architecture (matches the legacy static-scorer defaults).
        self.emb_dim = 128
        self.hidden_dim = 256
        # Loss / training.
        self.temp = 0.07
        self.margin = 0.15
        self.dropout_prob = 0.35          # per-row ID dropout = cold simulation
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        # Auxiliary loss. The legacy static-scorer's run.log shows the
        # ID<->content alignment InfoNCE (aux) was active the whole run
        # (aux=4.6 -> ~2.x), while its prereq-aux was dead (prereq=0.0000
        # every epoch). aux InfoNCE is a real, effective component.
        self.aux_weight = 0.3             # ID<->content tower alignment InfoNCE
        # Prereq aux (experiment A): pull each warm item's vector toward the
        # ID-centroid of its prerequisite courses (course metadata, no leak).
        # Legacy left this dead; here we activate it. 0 disables.
        self.prereq_aux_weight = 1.0
        self.prereq_aux_margin = 0.05
        self.prereq_path = "outputs/prereq_target/prereq_index_topk10.pt"
        self.cold_threshold = 1           # item is cold iff train popularity == 0
        self.batch_size = 2048
        self.eval_batch_users = 512
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.epochs = 60
        self.patience = 60                # early-stop on cold item-macro N@10
        self.min_delta = 1e-4


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class StaticContentScorer(nn.Module):
    def __init__(self, cfg: ScorerConfig, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)
        self.user_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        # Frozen content tower input, trainable projection to emb space.
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim),
        )
        # Fusion gate over [id_e, content_e] -> scalar alpha in (0,1).
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.emb_dim, 1),
            nn.Sigmoid(),
        )

        # Prerequisite index (course-course prereq relations from MOOCCube
        # metadata; NOT from interactions/splits -> leak-free, available at
        # deploy time for any cold course). prereq_idx: (n_items, topk) long,
        # -1 padded; has_prereq: (n_items,) bool. Loaded lazily; may be None.
        self.prereq_idx = None
        self.prereq_mask = None
        if getattr(cfg, "prereq_aux_weight", 0.0) > 0.0 and cfg.prereq_path:
            if os.path.exists(cfg.prereq_path):
                blob = torch.load(cfg.prereq_path, map_location="cpu", weights_only=False)
                self.prereq_idx = blob["prereq_idx"].long()
                self.prereq_mask = blob["has_prereq"].bool()
                print(f"[clean-static] loaded prereq index: {cfg.prereq_path} "
                      f"has_prereq={int(self.prereq_mask.sum())}/{self.prereq_mask.numel()}",
                      flush=True)
            else:
                print(f"[clean-static] WARN prereq_path missing: {cfg.prereq_path} "
                      f"-> prereq aux disabled", flush=True)

    def user_vector(self, u_idx: torch.Tensor) -> torch.Tensor:
        return self.user_proj(self.user_emb(u_idx))

    def _content_e(self, i_idx: torch.Tensor) -> torch.Tensor:
        return self.content_proj(self.item_con_emb(i_idx))

    def item_vector(
        self,
        i_idx: torch.Tensor,
        force_cold=False,
        apply_id_dropout: bool = False,
        return_towers: bool = False,
    ):
        """Fused item vector.

        force_cold: bool or bool tensor (per row). When true, id_e is zeroed so
                    the item relies purely on content — exactly what a strict
                    cold item must do at inference.
        apply_id_dropout: during training, randomly zero id_e per row with
                    prob dropout_prob (the cold-start simulation on warm items).
        return_towers: also return (id_e_true, content_e) for the auxiliary
                    ID<->content InfoNCE. id_e_true is the UN-masked ID vector.
        """
        id_e_true = self.item_id_emb(i_idx)
        id_e = id_e_true
        n = id_e.size(0)
        mask_id = torch.zeros((n, 1), dtype=torch.bool, device=id_e.device)
        if isinstance(force_cold, torch.Tensor):
            fm = force_cold.to(device=id_e.device)
            mask_id = mask_id | (fm > 0 if fm.dtype != torch.bool else fm).view(-1, 1)
        elif force_cold:
            mask_id = torch.ones((n, 1), dtype=torch.bool, device=id_e.device)
        if apply_id_dropout and self.training and self.cfg.dropout_prob > 0:
            drop = torch.rand((n, 1), device=id_e.device) < float(self.cfg.dropout_prob)
            mask_id = mask_id | drop
        if mask_id.any():
            id_e = torch.where(mask_id, torch.zeros_like(id_e), id_e)
        content_e = self._content_e(i_idx)
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        fused = alpha * id_e + (1.0 - alpha) * content_e
        if return_towers:
            return fused, id_e_true, content_e
        return fused


# --------------------------------------------------------------------------- #
# Data
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
# Metrics (item-macro): per-item accumulate R@k / N@k, then mean over items.
# --------------------------------------------------------------------------- #
def _dcg_hit(rank_pos: int, k: int) -> float:
    # target found at 0-based rank_pos within top-k -> 1/log2(rank+2)
    if rank_pos < k:
        return 1.0 / np.log2(rank_pos + 2.0)
    return 0.0


@torch.no_grad()
def evaluate_clean(
    model: StaticContentScorer,
    eval_df: pd.DataFrame,
    device: torch.device,
    train_pop: np.ndarray,
    user_seen: dict[int, set[int]],
    k_list=(5, 10, 20),
) -> dict:
    """Clean full-catalog single-vector evaluation, item-macro.

    Cold vs hot split by TRAIN popularity (cold iff pop < threshold). The
    positive item's score comes straight from the shared bank — never re-encoded.
    """
    cfg = model.cfg
    model.eval()
    n_items = cfg.n_items
    all_idx = torch.arange(n_items, device=device)

    # One vector per item. Cold items (train pop == 0) drop id_e; hot keep it.
    cold_item_mask_t = torch.as_tensor(
        train_pop < cfg.cold_threshold, dtype=torch.bool, device=device
    )
    bank_parts = []
    for start in range(0, n_items, 4096):
        idx = all_idx[start:start + 4096]
        fc = cold_item_mask_t[start:start + 4096]
        vec = model.item_vector(idx, force_cold=fc, apply_id_dropout=False)
        bank_parts.append(F.normalize(vec, dim=1))
    bank = torch.cat(bank_parts, dim=0)  # (n_items, d), each item exactly once

    max_k = max(k_list)
    # per-item accumulators
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
        z_u = F.normalize(model.user_vector(u_t), dim=1)
        scores = torch.mm(z_u, bank.t())  # (B, n_items)
        # mask train history so seen items can't occupy top-k
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
    # overall = count-weighted (item-macro) mean of cold and hot
    for k in k_list:
        for m in ("R", "N"):
            cc, hc = out["cold_count"], out["hot_count"]
            tot = max(1, cc + hc)
            out[f"overall_{m}@{k}"] = (
                out[f"cold_{m}@{k}"] * cc + out[f"hot_{m}@{k}"] * hc
            ) / tot
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except OSError:
        return False


def write_run_manifest(out_dir: Path, args, cfg: ScorerConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "script_sha256": source_hash,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "argv": list(sys.argv),
        "seed": int(args.seed),
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir),
        "prereq_weight": float(cfg.prereq_aux_weight),
        "aux_weight": float(cfg.aux_weight),
        "prereq_path": str(cfg.prereq_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _aux_infonce(cfg, id_e_true, content_e, device) -> torch.Tensor:
    """ID<->content tower alignment InfoNCE (legacy aux, weight cfg.aux_weight).

    Pulls each item's content-tower vector toward its own ID-tower vector in a
    full-batch InfoNCE. This teaches the content tower to reconstruct the
    behavior-space geometry, which is exactly what a cold item (no ID) must
    fall back on. id_e_true is the UN-masked ID vector.
    """
    if cfg.aux_weight <= 0.0 or id_e_true.size(0) <= 1:
        return id_e_true.new_zeros(())
    z_id = F.normalize(id_e_true, dim=1)
    z_con = F.normalize(content_e, dim=1)
    logits_aux = torch.mm(z_id, z_con.t()) / cfg.temp
    labels_aux = torch.arange(logits_aux.size(0), device=device)
    return cfg.aux_weight * F.cross_entropy(logits_aux, labels_aux)


def _prereq_aux_loss(model, i_idx, item_vec_norm, device) -> torch.Tensor:
    """Prereq-anchoring aux (margin ranking).

    For batch items that have prerequisite courses, pull the item's (normalized)
    vector toward the centroid of its prerequisite courses' ID vectors. Prereq
    relations come from MOOCCube course metadata (course->course), NOT from
    interactions — no leak. Prereq neighbors that are themselves cold (untrained
    ID) are masked out so we anchor only to real trained behavior geometry.
    """
    cfg = model.cfg
    if cfg.prereq_aux_weight <= 0.0 or model.prereq_idx is None:
        return item_vec_norm.new_zeros(())
    idx = i_idx.view(-1)
    has_pre = model.prereq_mask.to(device).index_select(0, idx)
    if not bool(has_pre.any().item()):
        return item_vec_norm.new_zeros(())
    rows = has_pre.nonzero(as_tuple=False).view(-1)
    tgt = idx.index_select(0, rows)
    pre_idx = model.prereq_idx.to(device).index_select(0, tgt)          # (R, topk)
    valid = pre_idx >= 0
    pre_safe = pre_idx.clamp(min=0)
    id_table = model.item_id_emb.weight                                    # dynamic, trained
    pre_vecs = id_table[pre_safe]                                          # (R, topk, D)
    vmask = valid.float().unsqueeze(-1)
    denom = vmask.sum(dim=1).clamp(min=1.0)
    centroid = (pre_vecs * vmask).sum(dim=1) / denom                       # (R, D)
    # rows with no valid (all-cold) prereq neighbors: skip
    keep = valid.any(dim=1)
    if not bool(keep.any().item()):
        return item_vec_norm.new_zeros(())
    centroid = F.normalize(centroid[keep], dim=1)
    anchor = item_vec_norm.index_select(0, rows[keep])
    pos = (anchor * centroid).sum(dim=1)
    loss = F.relu(cfg.prereq_aux_margin - pos).mean()
    return cfg.prereq_aux_weight * loss


def infonce_loss_parts(model: StaticContentScorer, u_idx, i_idx, device):
    cfg = model.cfg
    z_u = F.normalize(model.user_vector(u_idx), dim=1)
    item_out, id_e_true, content_e = model.item_vector(
        i_idx, force_cold=False, apply_id_dropout=True, return_towers=True
    )
    z_i = F.normalize(item_out, dim=1)
    prereq = _prereq_aux_loss(model, i_idx, z_i, device)
    logits = torch.mm(z_u, z_i.t()) / cfg.temp
    n = logits.size(0)
    labels = torch.arange(n, device=device)
    pos_mask = torch.eye(n, dtype=torch.bool, device=device)
    logits_m = logits.clone()
    logits_m[pos_mask] -= cfg.margin / cfg.temp
    aux = _aux_infonce(cfg, id_e_true, content_e, device)
    if n <= 1:
        main = F.cross_entropy(logits_m, labels)
    else:
        max_neg = n - 1
        n_total = min(cfg.train_num_negs, max_neg)
        if n_total <= 0:
            main = F.cross_entropy(logits_m, labels)
        else:
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
            main = F.cross_entropy(cand_logits, targets)
    total = main + aux + prereq
    return total, {"main": main, "aux": aux, "prereq": prereq}


def infonce_loss(model: StaticContentScorer, u_idx, i_idx, device) -> torch.Tensor:
    return infonce_loss_parts(model, u_idx, i_idx, device)[0]


def train(cfg: ScorerConfig, train_df, val_df, test_df, content_emb, device, out_dir: Path):
    model = StaticContentScorer(cfg, content_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_pop = compute_train_popularity(train_df, cfg.n_items)
    user_seen = build_user_seen(train_df)

    u_all = torch.as_tensor(train_df["u_idx"].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(train_df["i_idx"].to_numpy(), dtype=torch.long)
    n_train = u_all.size(0)

    best_score = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    val_history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        total_main = 0.0
        total_aux = 0.0
        total_prereq = 0.0
        nb = 0
        t0 = time.time()
        for start in range(0, n_train, cfg.batch_size):
            b = perm[start:start + cfg.batch_size]
            u_idx = u_all[b].to(device)
            i_idx = i_all[b].to(device)
            opt.zero_grad()
            loss, parts = infonce_loss_parts(model, u_idx, i_idx, device)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            total_main += float(parts["main"].detach())
            total_aux += float(parts["aux"].detach())
            total_prereq += float(parts["prereq"].detach())
            nb += 1
        val = evaluate_clean(model, val_df, device, train_pop, user_seen)
        score = val["cold_N@10"]  # early-stop on cold item-macro N@10
        denom = max(1, nb)
        val_history.append({"epoch": epoch, "loss": total_loss / denom,
                            "main_loss": total_main / denom,
                            "aux_loss": total_aux / denom,
                            "prereq_loss": total_prereq / denom,
                            "cold_N@10": score,
                            "cold_R@10": val["cold_R@10"],
                            "hot_N@10": val["hot_N@10"],
                            "hot_R@10": val["hot_R@10"]})
        print(f"[epoch {epoch}/{cfg.epochs}] loss={total_loss/denom:.4f} "
              f"main={total_main/denom:.4f} aux={total_aux/denom:.4f} "
              f"prereq={total_prereq/denom:.4f} "
              f"time={time.time()-t0:.1f}s | val cold R@10={val['cold_R@10']:.4f} "
              f"N@10={val['cold_N@10']:.4f} | hot R@10={val['hot_R@10']:.4f} "
              f"N@10={val['hot_N@10']:.4f}", flush=True)
        if score > best_score + cfg.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch}, "
                      f"cold N@10={best_score:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_clean(model, test_df, device, train_pop, user_seen)
    print("\n=== TEST (clean single-vector, item-macro) — best epoch "
          f"{best_epoch} ===")
    for split in ("cold", "hot", "overall"):
        print(f"  {split:8s} R@10={test[f'{split}_R@10']:.4f}  "
              f"N@10={test[f'{split}_N@10']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
        json.dump(val_history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "test": test,
                   "evaluator": "clean_single_vector_full_catalog_item_macro"},
                  f, indent=2)
    print(f"\nSaved to {out_dir}")
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    ap.add_argument("--split-dir", required=True,
                    help="strict item-cold split dir (static_train/val/test.pkl)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--dropout-prob", type=float, default=None,
                    help="override ID-dropout prob (default 0.35 from config)")
    ap.add_argument("--prereq-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--prereq-path",
                    default="outputs/prereq_target/prereq_index_topk10.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    content_emb = torch.load(os.path.join(data_dir, "content_emb.pt"))
    if not isinstance(content_emb, torch.Tensor):
        content_emb = torch.as_tensor(content_emb)
    content_emb = content_emb.float()

    train_df, val_df, test_df, split_info = load_shared_static_split(args.split_dir)

    cfg = ScorerConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.prereq_aux_weight = float(args.prereq_weight)
    cfg.aux_weight = float(args.aux_weight)
    cfg.prereq_path = str(args.prereq_path)
    if args.dropout_prob is not None:
        cfg.dropout_prob = float(args.dropout_prob)
        print(f"[clean-static] dropout_prob overridden to {cfg.dropout_prob}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"n_users={cfg.n_users} n_items={cfg.n_items} content_dim={cfg.content_dim} "
          f"device={device}")
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"prereq_weight={cfg.prereq_aux_weight} aux_weight={cfg.aux_weight} "
          f"prereq_path={cfg.prereq_path}", flush=True)

    if args.dry_run:
        train_pop = compute_train_popularity(train_df, cfg.n_items)
        n_cold = int((train_pop < cfg.cold_threshold).sum())
        print(f"[dry-run] cold items (train pop==0): {n_cold} / {cfg.n_items}")
        if cfg.prereq_aux_weight > 0.0 and os.path.exists(cfg.prereq_path):
            blob = torch.load(cfg.prereq_path, map_location="cpu", weights_only=False)
            print(f"[dry-run] prereq index has_prereq="
                  f"{int(blob['has_prereq'].sum())}/{len(blob['has_prereq'])}")
        elif cfg.prereq_aux_weight > 0.0:
            raise FileNotFoundError(f"prerequisite index not found: {cfg.prereq_path}")
        print("[dry-run] OK — no training performed.")
        return

    out_dir = Path(args.output_dir)
    write_run_manifest(out_dir, args, cfg)
    train(cfg, train_df, val_df, test_df, content_emb, device, out_dir)


if __name__ == "__main__":
    main()
