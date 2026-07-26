"""Clean graph-augmented content-behavior cold-start scorer.

Extends static_content_scorer_clean.py by adding a standard LightGCN-style
propagation layer on the user-item bipartite graph (built from TRAIN
interactions only -> leak-free). Rationale (verified 2026-07-23): the static
double-tower core already matches CGRC on COLD, but lags on HOT/OVERALL because
it has NO collaborative signal. A dropout scan confirmed dropout is not the
bottleneck. CGRC itself is LightGCN + its cold mechanism; so a standard graph
backbone is fair infrastructure and the novelty stays in the cold-start path.

Design:
  item ego vector : gate(alpha)*id_e + (1-alpha)*content_e   (unchanged core;
                    cold items force_cold -> id_e zeroed; train ID-dropout)
  user ego vector : user_emb (raw)
  propagation     : L layers of D^-1/2 A D^-1/2 over bipartite graph, mean of
                    layers 0..L  -> final z_u, z_i (LightGCN)
  scoring         : normalize(z_u) . normalize(z_i)
  cold items have no train edges -> propagation leaves them at their ego vector
                    -> cold path (content+aux+prereq) preserved.
  losses          : main InfoNCE (in-batch, on propagated z) + aux ID<->content
                    (on ego) + prereq anchor (on ego). Same as A experiment.

Isolation: does NOT modify static_content_scorer_clean.py. Reuses only
load_shared_static_split. No legacy USIM/CGRC code imported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast3_delta.static_protocol import load_shared_static_split


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class GraphScorerConfig:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.content_dim = int(content_dim)
        self.emb_dim = 128
        self.hidden_dim = 256
        self.temp = 0.07
        self.margin = 0.15
        self.dropout_prob = 0.35
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        self.aux_weight = 0.3
        self.prereq_aux_weight = 1.0
        self.prereq_aux_margin = 0.05
        self.prereq_path = "outputs/prereq_target/prereq_index_topk10.pt"
        self.n_layers = 2                 # LightGCN propagation depth
        self.gate_k = 5.0                 # degree-gated fusion beta=deg/(deg+K):
                                          # cold(deg=0)->beta=0->pure ego(content),
                                          # hot(high deg)->beta->1->pure post-graph
        self.cold_threshold = 1
        self.batch_size = 2048
        self.eval_batch_users = 512
        self.lr = 1e-3
        self.epochs = 60
        self.patience = 60
        self.min_delta = 1e-4


# --------------------------------------------------------------------------- #
# Bipartite graph: normalized sparse adjacency from TRAIN interactions only.
# --------------------------------------------------------------------------- #
def build_norm_adj(train_df: pd.DataFrame, n_users: int, n_items: int,
                   device: torch.device) -> torch.Tensor:
    """Symmetric-normalized bipartite adjacency D^-1/2 A D^-1/2 as sparse tensor.

    Node order: [0..n_users) users, [n_users..n_users+n_items) items.
    Only TRAIN edges -> strict-cold items (no train interactions) get zero
    degree and stay isolated (propagation leaves them at their ego vector).
    """
    u = train_df["u_idx"].to_numpy(np.int64, copy=False)
    i = train_df["i_idx"].to_numpy(np.int64, copy=False)
    n = n_users + n_items
    rows = np.concatenate([u, i + n_users])
    cols = np.concatenate([i + n_users, u])
    data = np.ones(rows.size, dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    A.data[:] = 1.0  # binary
    deg = np.asarray(A.sum(axis=1)).flatten()
    dinv = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    dinv[nz] = np.power(deg[nz], -0.5)
    D = sp.diags(dinv)
    norm = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GraphContentScorer(nn.Module):
    def __init__(self, cfg: GraphScorerConfig, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim), nn.LayerNorm(cfg.emb_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim), nn.GELU(),
            nn.Linear(cfg.emb_dim, 1), nn.Sigmoid(),
        )
        # prereq index (course metadata; leak-free). May be None.
        self.prereq_idx = None
        self.prereq_mask = None
        if cfg.prereq_aux_weight > 0.0 and cfg.prereq_path and os.path.exists(cfg.prereq_path):
            blob = torch.load(cfg.prereq_path, map_location="cpu", weights_only=False)
            self.prereq_idx = blob["prereq_idx"].long()
            self.prereq_mask = blob["has_prereq"].bool()
            print(f"[graph] loaded prereq index has_prereq="
                  f"{int(self.prereq_mask.sum())}/{self.prereq_mask.numel()}", flush=True)
        # item train degree -> degree-gated fusion beta. set by set_item_degree().
        self.register_buffer("item_beta", torch.zeros(cfg.n_items))

    def set_item_degree(self, item_pop: np.ndarray):
        """beta_i = deg_i / (deg_i + K). cold(deg=0)->0, hot->~1."""
        deg = torch.as_tensor(item_pop, dtype=torch.float32)
        beta = deg / (deg + float(self.cfg.gate_k))
        self.item_beta = beta.to(self.item_id_emb.weight.device)

    def item_ego(self, apply_id_dropout: bool, cold_mask_all: torch.Tensor):
        """Ego (pre-propagation) item vectors for ALL items.

        cold_mask_all: (n_items,) bool — strict-cold items force id_e=0.
        Returns fused (n_items,d), id_e_true (n_items,d), content_e (n_items,d).
        """
        all_idx = torch.arange(self.cfg.n_items, device=self.item_id_emb.weight.device)
        id_e_true = self.item_id_emb(all_idx)
        id_e = id_e_true
        mask = cold_mask_all.view(-1, 1).clone()
        if apply_id_dropout and self.training and self.cfg.dropout_prob > 0:
            drop = torch.rand((self.cfg.n_items, 1), device=id_e.device) < float(self.cfg.dropout_prob)
            mask = mask | drop
        id_e = torch.where(mask, torch.zeros_like(id_e), id_e)
        content_e = self.content_proj(self.item_con_emb(all_idx))
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        fused = alpha * id_e + (1.0 - alpha) * content_e
        return fused, id_e_true, content_e

    def propagate(self, adj: torch.Tensor, item_ego: torch.Tensor):
        """LightGCN + degree-gated fusion.

        Users: pure post-graph (all users have interactions).
        Items: beta*post_graph + (1-beta)*ego. Cold items (beta=0) keep their
        pure content ego (no graph dilution); hot items (beta~1) use the
        collaborative post-graph vector. Returns z_u_all, z_i_all.
        """
        ego = torch.cat([self.user_emb.weight, item_ego], dim=0)
        embs = [ego]
        h = ego
        for _ in range(self.cfg.n_layers):
            h = torch.sparse.mm(adj, h)
            embs.append(h)
        out = torch.stack(embs, dim=1).mean(dim=1)
        z_u = out[:self.cfg.n_users]
        z_i_post = out[self.cfg.n_users:]
        beta = self.item_beta.view(-1, 1)
        z_i = beta * z_i_post + (1.0 - beta) * item_ego
        return z_u, z_i


# --------------------------------------------------------------------------- #
# Losses (on ego vectors, same as A experiment)
# --------------------------------------------------------------------------- #
def _aux_infonce(cfg, id_e_true, content_e, device):
    if cfg.aux_weight <= 0.0 or id_e_true.size(0) <= 1:
        return id_e_true.new_zeros(())
    z_id = F.normalize(id_e_true, dim=1)
    z_con = F.normalize(content_e, dim=1)
    logits = torch.mm(z_id, z_con.t()) / cfg.temp
    labels = torch.arange(logits.size(0), device=device)
    return cfg.aux_weight * F.cross_entropy(logits, labels)


def _prereq_aux(model, ego_norm_all, device):
    """Prereq anchor on ALL items' ego vectors. ego_norm_all: (n_items,d) normalized."""
    cfg = model.cfg
    if cfg.prereq_aux_weight <= 0.0 or model.prereq_idx is None:
        return ego_norm_all.new_zeros(())
    mask = model.prereq_mask.to(device)
    if not bool(mask.any().item()):
        return ego_norm_all.new_zeros(())
    rows = mask.nonzero(as_tuple=False).view(-1)
    pre_idx = model.prereq_idx.to(device).index_select(0, rows)   # (R, topk)
    valid = pre_idx >= 0
    pre_safe = pre_idx.clamp(min=0)
    id_table = model.item_id_emb.weight
    pre_vecs = id_table[pre_safe]                                 # (R, topk, d)
    vmask = valid.float().unsqueeze(-1)
    denom = vmask.sum(dim=1).clamp(min=1.0)
    centroid = (pre_vecs * vmask).sum(dim=1) / denom
    keep = valid.any(dim=1)
    if not bool(keep.any().item()):
        return ego_norm_all.new_zeros(())
    centroid = F.normalize(centroid[keep], dim=1)
    anchor = ego_norm_all.index_select(0, rows[keep])
    pos = (anchor * centroid).sum(dim=1)
    return cfg.prereq_aux_weight * F.relu(cfg.prereq_aux_margin - pos).mean()


def graph_loss(model, adj, cold_mask_all, u_idx, i_idx, device):
    cfg = model.cfg
    # full-graph ego + propagation (LightGCN computes all nodes once)
    item_ego, id_e_true, content_e = model.item_ego(apply_id_dropout=True,
                                                     cold_mask_all=cold_mask_all)
    z_u_all, z_i_all = model.propagate(adj, item_ego)
    z_u = F.normalize(z_u_all.index_select(0, u_idx), dim=1)
    z_i = F.normalize(z_i_all.index_select(0, i_idx), dim=1)
    logits = torch.mm(z_u, z_i.t()) / cfg.temp
    n = logits.size(0)
    labels = torch.arange(n, device=device)
    pos_mask = torch.eye(n, dtype=torch.bool, device=device)
    logits_m = logits.clone()
    logits_m[pos_mask] -= cfg.margin / cfg.temp
    # aux on batch ego towers; prereq on all-item ego (normalized)
    aux = _aux_infonce(cfg, id_e_true.index_select(0, i_idx),
                       content_e.index_select(0, i_idx), device)
    prereq = _prereq_aux(model, F.normalize(item_ego, dim=1), device)
    if n <= 1:
        return F.cross_entropy(logits_m, labels) + aux + prereq
    max_neg = n - 1
    n_total = min(cfg.train_num_negs, max_neg)
    if n_total <= 0:
        return F.cross_entropy(logits_m, labels) + aux + prereq
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
    return F.cross_entropy(cand_logits, targets) + aux + prereq


# --------------------------------------------------------------------------- #
# Data helpers + metrics
# --------------------------------------------------------------------------- #
def build_user_seen(train_df):
    seen = {}
    for u, i in zip(train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def compute_train_popularity(train_df, n_items):
    pop = np.zeros(n_items, dtype=np.int64)
    vc = train_df["i_idx"].value_counts()
    pop[vc.index.to_numpy()] = vc.to_numpy()
    return pop


def _dcg_hit(rank_pos, k):
    return 1.0 / np.log2(rank_pos + 2.0) if rank_pos < k else 0.0


@torch.no_grad()
def evaluate_clean(model, adj, eval_df, device, train_pop, user_seen, k_list=(5, 10, 20)):
    """Clean full-catalog single-vector eval after graph propagation."""
    cfg = model.cfg
    model.eval()
    cold_mask_all = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool, device=device)
    item_ego, _, _ = model.item_ego(apply_id_dropout=False, cold_mask_all=cold_mask_all)
    z_u_all, z_i_all = model.propagate(adj, item_ego)
    bank = F.normalize(z_i_all, dim=1)                 # (n_items,d) one vector per item

    max_k = max(k_list)
    acc = {f"{m}@{k}": {} for m in ("R", "N") for k in k_list}
    cnt = {}
    cold_np = train_pop < cfg.cold_threshold
    users = eval_df["u_idx"].to_numpy()
    items = eval_df["i_idx"].to_numpy()
    bs = cfg.eval_batch_users
    for start in range(0, len(users), bs):
        u_batch = users[start:start + bs]
        i_batch = items[start:start + bs]
        u_t = torch.as_tensor(u_batch, dtype=torch.long, device=device)
        z_u = F.normalize(z_u_all.index_select(0, u_t), dim=1)
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
                acc[f"R@{k}"][tgt] = acc[f"R@{k}"].get(tgt, 0.0) + (1.0 if rank_pos < k else 0.0)
                acc[f"N@{k}"][tgt] = acc[f"N@{k}"].get(tgt, 0.0) + _dcg_hit(rank_pos, k)

    def macro(key, filt):
        vals = [acc[key].get(it, 0.0) / max(1, c) for it, c in cnt.items() if filt(it)]
        return float(np.mean(vals)) if vals else 0.0

    out = {}
    for name, filt in (("cold", lambda it: cold_np[it]), ("hot", lambda it: not cold_np[it])):
        for k in k_list:
            out[f"{name}_R@{k}"] = macro(f"R@{k}", filt)
            out[f"{name}_N@{k}"] = macro(f"N@{k}", filt)
        out[f"{name}_count"] = int(sum(1 for it in cnt if filt(it)))
    for k in k_list:
        for m in ("R", "N"):
            cc, hc = out["cold_count"], out["hot_count"]
            out[f"overall_{m}@{k}"] = (out[f"cold_{m}@{k}"] * cc + out[f"hot_{m}@{k}"] * hc) / max(1, cc + hc)
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg, train_df, val_df, test_df, content_emb, device, out_dir):
    model = GraphContentScorer(cfg, content_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_pop = compute_train_popularity(train_df, cfg.n_items)
    user_seen = build_user_seen(train_df)
    adj = build_norm_adj(train_df, cfg.n_users, cfg.n_items, device)
    print(f"[graph] adj built: nnz={adj._nnz()} n_layers={cfg.n_layers}", flush=True)
    model.set_item_degree(train_pop)
    _b = model.item_beta
    print(f"[gated] beta: cold(deg=0) count={int((_b==0).sum())} "
          f"mean={float(_b.mean()):.3f} max={float(_b.max()):.3f} K={cfg.gate_k}", flush=True)
    cold_mask_all = torch.as_tensor(train_pop < cfg.cold_threshold, dtype=torch.bool, device=device)

    u_all = torch.as_tensor(train_df["u_idx"].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(train_df["i_idx"].to_numpy(), dtype=torch.long)
    n_train = u_all.size(0)
    best_score, best_epoch, best_state, no_improve = -1.0, -1, None, 0
    hist = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, n_train, cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            u_idx = u_all[b].to(device)
            i_idx = i_all[b].to(device)
            opt.zero_grad()
            loss = graph_loss(model, adj, cold_mask_all, u_idx, i_idx, device)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        val = evaluate_clean(model, adj, val_df, device, train_pop, user_seen)
        score = val["cold_N@10"]
        hist.append({"epoch": epoch, "cold_N@10": score, "cold_R@10": val["cold_R@10"],
                     "hot_N@10": val["hot_N@10"], "hot_R@10": val["hot_R@10"]})
        print(f"[epoch {epoch}/{cfg.epochs}] loss={tot/max(1,nb):.4f} time={time.time()-t0:.1f}s "
              f"| val cold R@10={val['cold_R@10']:.4f} N@10={val['cold_N@10']:.4f} "
              f"| hot R@10={val['hot_R@10']:.4f} N@10={val['hot_N@10']:.4f}", flush=True)
        if score > best_score + cfg.min_delta:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop epoch {epoch} (best {best_epoch}, cold N@10={best_score:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_clean(model, adj, test_df, device, train_pop, user_seen)
    print(f"\n=== TEST (graph, clean single-vector) best epoch {best_epoch} ===")
    for sp_ in ("cold", "hot", "overall"):
        print(f"  {sp_:8s} R@10={test[f'{sp_}_R@10']:.4f}  N@10={test[f'{sp_}_N@10']:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "val_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "test": test,
                   "evaluator": "clean_single_vector_full_catalog_item_macro_graph",
                   "n_layers": cfg.n_layers}, f, indent=2)
    print(f"\nSaved to {out_dir}")
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--prereq-weight", dest="prereq_weight", type=float, default=1.0,
                    help="weight of prereq anchor aux loss (cfg.prereq_aux_weight)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    content_emb = torch.load(os.path.join(args.data_dir, "content_emb.pt"))
    if not isinstance(content_emb, torch.Tensor):
        content_emb = torch.as_tensor(content_emb)
    content_emb = content_emb.float()
    train_df, val_df, test_df, _ = load_shared_static_split(args.split_dir)

    cfg = GraphScorerConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.n_layers = args.n_layers
    cfg.prereq_aux_weight = float(args.prereq_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"n_users={cfg.n_users} n_items={cfg.n_items} content_dim={cfg.content_dim} "
          f"device={device} n_layers={cfg.n_layers} prereq_weight={cfg.prereq_aux_weight}",
          flush=True)
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    if args.dry_run:
        tp = compute_train_popularity(train_df, cfg.n_items)
        print(f"[dry-run] cold items={int((tp < cfg.cold_threshold).sum())}/{cfg.n_items} "
              f"prereq_weight={cfg.prereq_aux_weight}", flush=True)
        print("[dry-run] OK", flush=True)
        return
    train(cfg, train_df, val_df, test_df, content_emb, device, Path(args.output_dir))


if __name__ == "__main__":
    main()


