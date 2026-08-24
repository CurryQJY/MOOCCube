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
        # ID-dropout scope during training:
        #   "uniform" — every item is eligible (legacy default)
        #   "tail"    — only items with train_pop in [cold_threshold, dropout_max_pop]
        #               are eligible; head items never get forced-cold. Matches the
        #               diagnose_pseudocold_operator hard constraint (rand* ~0 gain).
        self.dropout_mode = "uniform"
        self.dropout_max_pop = 25
        # When an item is ID-dropped, also force beta=0 on that forward so the
        # train path matches the strict-cold inference path (content ego only).
        self.dropout_force_beta0 = False
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
        # ---- Train/eval-consistent radial calibration ----------------------
        # Strict-cold items have no interaction rows, so beta=0 leaves their
        # radius (projection on the user-cloud direction) as an uncontrolled
        # content-encoder by-product. When enabled, keep each item's direction
        # and predict its radius from warm prerequisite neighbours. The layer is
        # disabled by default to preserve every legacy checkpoint bit-for-bit.
        self.radial_calibration = False
        self.radial_target_fallback = "warm_mean"
        self.radial_eps = 1e-6
        # ---- KG-propagated cold path (item-item knowledge channel) ----
        # The user-item graph gives cold items ZERO signal by construction:
        # build_norm_adj uses train edges only (cold deg=0) and beta=deg/(deg+K)
        # zeroes their post-graph term. So a cold item's only neighbour
        # information has to squeeze through the SHARED content encoder, which
        # cannot represent per-item displacements that content does not predict.
        # That is the wall the op-align / RL / soft-refine attempts all hit.
        # Prereq metadata is the one structure where cold items DO have edges,
        # so we propagate the collaborative item vectors over it, giving each
        # cold item its own path: cold -> related warm course -> that course's
        # users. kg_weight=0 (default) reproduces the tail arm bit-identically.
        self.kg_weight = 0.0              # 0 -> channel off, exact legacy path
        self.kg_layers = 1                # item-item propagation depth
        self.kg_scope = "cold_tail"       # "cold" | "cold_tail" | "all"
        self.kg_tail_max_pop = 25         # upper pop bound for the "tail" part
        self.kg_knn_fallback = 0          # >0: content-kNN edges for items with
                                          # no prereq edge (k neighbours)
        # "missing": kNN edges only for items that have NO prereq edge (legacy,
        # bit-identical to before this flag existed). "all": every item also gets
        # k content-kNN edges, which densifies a sparse prereq graph -- the test
        # for whether propagation gain is limited by graph density.
        self.kg_knn_scope = "missing"
        self.kg_gate_bias = -2.0          # init bias of the gate's last layer;
                                          # negative -> starts near the known-good
                                          # tail behaviour and learns to open
        # Learned-gate escape hatch. The gate turns out to be structurally
        # unlearnable for its own target population: strict-cold items have
        # train_pop=0, so they contribute ZERO training rows, and the gate's
        # gradient comes almost entirely from tail items simulated cold by
        # ID-dropout -- for which the channel is at best neutral. Measured
        # outcome: w drifts DOWN from its 0.119 init to ~0.055-0.064 across all
        # three seeds, i.e. training closes a channel that helps cold items
        # (125 win / 3 loss when swept post-hoc). So >= 0 here replaces the
        # gate with a constant w, to be selected on val like any other
        # hyper-parameter. < 0 keeps the learned gate.
        self.kg_fixed_w = -1.0
        # NCER edge reweighting strength (see build_knowledge_adj). 0 = off and
        # bit-identical to the binary graph.
        self.kg_ncer_kappa = 0.0
        self.kg_ncer_form = "linear"      # "linear" = paper's (1+kc); "exp" =
                                          # exp(kc), removes the ~2x ceiling
        self.kg_edge_type_w = -1.0        # kNN-edge weight (prereq=1); <0 = off
        self.kg_min_neighbor_pop = 0.0    # drop KG edges where neither endpoint
                                          # has train_pop >= this; 0 = off
        self.kg_cold_rewire_seed = -1     # random control: resample each cold
                                          # item's neighbours from warm items,
                                          # degree preserved; <0 = off
        self.kg_cold_centroid = False     # null model: replace graph aggregation
                                          # with the plain warm-centroid vector
        # ---- differentiable cold pathway (DiffCold-style, no diffusion) -------
        # The KG mix above is gradient-free: kg_scope_w only covers rows with zero
        # training interactions, so it appears in no loss term. This replaces the
        # fixed neighbour aggregate with a LEARNED generator and trains it on
        # pseudo-cold rows against their real warm ID embedding, which is the one
        # piece of DiffCold (arXiv 2606.12245) that transfers here -- their
        # retrieval aggregator is what we already have, and their conditional
        # diffusion needs far more warm items than the 596 this catalogue offers.
        # The existing aux InfoNCE aligns content_e -> id_e_true but EXCLUDES
        # pseudo-cold rows and never sees the neighbour aggregate, so this is a
        # genuinely different objective, not a re-parameterisation of it.
        self.cold_gen_enable = False
        self.cold_gen_hidden = 256
        self.cold_gen_align_weight = 1.0
        # Fraction of warm items sampled per step for the alignment loss only.
        # Kept independent of dropout_mode="tail" so the verified ranking-path
        # behaviour is untouched: tail dropout gives ~33 pseudo-cold rows/step,
        # too thin to fit a generator, while DiffCold masks 30% of all warm items.
        self.cold_gen_mask_rho = 0.30
        # Training-only donor supervision: when a tail item is ID-dropped,
        # temporarily enable its KG row so its ranking loss can train the
        # warm neighbours that donate collaborative evidence. Evaluation keeps
        # the static kg_scope_w unchanged.
        self.kg_train_pseudocold = False
        # ---- DIF (Denoising Implicit Feedback) label-noise reweighting ----
        # Applied to ALL training positives (which are warm by construction:
        # strict-cold items have train_pop=0 and therefore zero train rows),
        # gated by degree so the correction concentrates on low-frequency items
        # and vanishes on the head. gamma_i = exp(-dif_alpha * g_i), where
        # g_i = log1p(deg_i)/log1p(max_deg) in [0,1].
        # t_i = r_i * gamma_i (r_i = detached per-sample CE residual),
        # w_i = t_i/(t_i+1), sample_weight = 1 - dif_max_w * w_i.
        # dif_max_w < 1 is the uncertainty guard: we are not certain a
        # high-residual sample is mislabeled, so the down-weight is capped.
        self.dif_enable = False
        self.dif_alpha = 3.0
        self.dif_max_w = 0.5
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


def build_knowledge_adj(prereq_idx: torch.Tensor, n_items: int, device: torch.device,
                        content_emb: torch.Tensor | None = None,
                        knn_k: int = 0, knn_scope: str = "missing",
                        ncer_kappa: float = 0.0, ncer_form: str = "linear",
                        edge_type_w: float = -1.0,
                        item_pop=None, min_neighbor_pop: float = 0.0,
                        cold_rewire_seed: int = -1):
    """Symmetric-normalized item-item knowledge adjacency D^-1/2 A D^-1/2.

    Edges come from course prerequisite metadata only -- course-level side
    information, never interaction data, so this is leak-free in exactly the
    same sense as the content embeddings.

    The reason this graph exists: in the user-item graph cold items are
    isolated by construction (build_norm_adj uses train edges only), so the
    collaborative channel is dead for them. Here they DO have edges, which
    gives each cold item its own path to real users:
        cold item -> related warm course -> users who took that course.

    prereq_idx: (n_items, topk) long, -1 padded.
    knn_k > 0 : items left with no prereq edge get content-kNN edges instead.
    ncer_kappa > 0 : Neighborhood-Consistency Edge Reweighting (IIMRec, ACM MM'26,
        doi 10.1145/3767308.3836382). Triadic closure -- an edge (i,j) is more
        trustworthy when i and j share many neighbours, since it is then backed
        by several independent paths; an isolated edge is more likely noise.
        Motivation here: our own group ablation showed edge utility is strongly
        heterogeneous (prereq edges worth ~4x a kNN edge per edge, and dropping
        low-popularity-neighbour edges did not hurt), so equal weights are
        leaving signal on the table. Crucially this is a PREPROCESSING step with
        no parameters and no gradient, so it sidesteps the wall that killed the
        learned gate: cold items have zero training rows but they DO have
        knowledge edges, so their consistency scores are computable.
        kappa=0 reproduces the binary graph bit-identically.

    Returns (sparse adj (n_items,n_items), item degree array).
    """
    pi = prereq_idx.detach().cpu().numpy()
    valid = pi >= 0
    rows = np.repeat(np.arange(pi.shape[0], dtype=np.int64), valid.sum(axis=1))
    cols = pi[valid].astype(np.int64)
    keep = (cols >= 0) & (cols < n_items) & (rows != cols)
    rows, cols = rows[keep], cols[keep]

    n_prereq_dir = rows.size
    if knn_k > 0 and content_emb is not None:
        scope = str(knn_scope or "missing").lower()
        if scope == "missing":
            have = np.zeros(n_items, dtype=bool)
            have[rows] = True
            have[cols] = True
            targets = np.nonzero(~have)[0]
        elif scope == "all":
            targets = np.arange(n_items, dtype=np.int64)
        else:
            raise ValueError(f"unknown knn_scope={knn_scope!r}; use 'missing' or 'all'")
        k = int(min(knn_k, n_items - 1))
        if targets.size and k > 0:
            z = F.normalize(content_emb.float(), dim=1)
            sim = (z[torch.as_tensor(targets)] @ z.t()).detach().cpu().numpy()
            sim[np.arange(targets.size), targets] = -np.inf
            nb = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            rows = np.concatenate([rows, np.repeat(targets, k)])
            cols = np.concatenate([cols, nb.reshape(-1).astype(np.int64)])
            print(f"[kg] content-kNN scope={scope} applied to {targets.size} items "
                  f"(k={k})", flush=True)

    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                      shape=(n_items, n_items), dtype=np.float32)
    A = A.maximum(A.T).tocsr()      # undirected
    A.data[:] = 1.0                 # binary

    # ---- topology pruning: drop edges where NEITHER endpoint carries user ----
    # signal. A cold item borrows z_i_post from its neighbours; a neighbour with
    # no interactions of its own has nothing to lend, so such an edge transfers
    # noise. Rule is max(pop_i,pop_j) < P, which is symmetric and never touches
    # a cold->warm edge (cold pop=0 but the warm end carries the max).
    # This changes TOPOLOGY, unlike edge reweighting, which degree normalisation
    # cancels for uniformly-scaled rows (measured: 4 reweighting arms all null,
    # and all 258 kNN-backfill targets sit in pure-kNN rows where alpha is an
    # exact no-op). 0 = off, bit-identical.
    if float(min_neighbor_pop) > 0.0 and item_pop is not None:
        pop = np.asarray(item_pop, dtype=np.float32).reshape(-1)
        Ac0 = A.tocoo()
        r0, c0 = Ac0.row, Ac0.col
        keep_e = np.maximum(pop[r0], pop[c0]) >= float(min_neighbor_pop)
        dropped = int((~keep_e).sum())
        A = sp.csr_matrix((np.ones(int(keep_e.sum()), dtype=np.float32),
                           (r0[keep_e], c0[keep_e])),
                          shape=(n_items, n_items), dtype=np.float32)
        deg_after = np.asarray(A.sum(axis=1)).flatten()
        print(f"[kg] prune min_neighbor_pop={min_neighbor_pop}: dropped {dropped}/"
              f"{r0.size} directed entries; items_with_edges "
              f"{int((deg_after > 0).sum())}/{n_items}", flush=True)

    deg = np.asarray(A.sum(axis=1)).flatten()

    # ---- RANDOM CONTROL: keep each cold item's degree, resample WHICH warm ----
    # items it borrows from. Because kg_scope="cold" zeroes w on every warm row,
    # only the cold rows of this adjacency ever reach the output, so randomising
    # exactly those rows is the sharpest test of the paper's central claim --
    # "a cold item's evidence sits in its (prerequisite / content-similar)
    # neighbours". If cold N@10 survives this, the edge SEMANTICS carry nothing
    # and the channel is only pulling cold items toward the warm collaborative
    # mass. Degree and warm-neighbour count are held fixed so the comparison
    # isolates identity, not connectivity. < 0 = off, bit-identical.
    if int(cold_rewire_seed) >= 0 and item_pop is not None:
        pop_r = np.asarray(item_pop, dtype=np.float32).reshape(-1)
        cold_ids = np.nonzero(pop_r < 1.0)[0]
        warm_ids = np.nonzero(pop_r >= 1.0)[0]
        rng = np.random.default_rng(int(cold_rewire_seed))
        L = A.tolil()
        n_re = 0
        for i in cold_ids:
            old_nb = np.asarray(L.rows[i], dtype=np.int64)
            d_i = old_nb.size
            if d_i == 0:
                continue
            new_nb = rng.choice(warm_ids, size=min(d_i, warm_ids.size), replace=False)
            for j in old_nb:
                L[i, j] = 0.0
                L[j, i] = 0.0
            for j in new_nb:
                L[i, j] = 1.0
                L[j, i] = 1.0
            n_re += 1
        A = L.tocsr()
        A.eliminate_zeros()
        deg2 = np.asarray(A.sum(axis=1)).flatten()
        print(f"[kg] COLD-REWIRE seed={cold_rewire_seed}: rewired {n_re} cold rows "
              f"to random warm neighbours | cold mean_deg {deg[cold_ids].mean():.2f}"
              f"->{deg2[cold_ids].mean():.2f} | nnz {A.nnz}", flush=True)
        deg = deg2

    # Edge provenance, needed by both reweighting schemes below. Symmetrising
    # loses it, so rebuild the prereq-only mask from the pre-kNN slice.
    Pm = sp.csr_matrix((np.ones(n_prereq_dir, dtype=np.float32),
                        (rows[:n_prereq_dir], cols[:n_prereq_dir])),
                       shape=(n_items, n_items), dtype=np.float32)
    Pm = Pm.maximum(Pm.T).tocsr()

    Ac = A.tocoo()
    r, c = Ac.row, Ac.col
    is_pre = np.asarray(Pm[r, c]).flatten() > 0
    w_e = np.ones(r.size, dtype=np.float32)

    # ---- (B) edge-type prior: prereq edges keep weight 1, kNN-only get alpha --
    # Motivated directly by our own group ablation: per edge, prereq edges are
    # worth ~4.3x a kNN edge (-2.44e-5 vs -5.7e-6 cold N@10 per edge). This
    # encodes that measurement with one scalar, no proxy.  < 0 disables.
    if float(edge_type_w) >= 0.0:
        w_e = np.where(is_pre, 1.0, float(edge_type_w)).astype(np.float32)

    # ---- (A) NCER: reweight by neighbourhood consistency (kappa=0 = no-op) ----
    # Paper normalises the shared-neighbour count by the kNN parameter k. That
    # does not transfer here: our graph mixes prereq and kNN edges so degrees
    # (mean 18.96) run well above k=10 and c_ij would exceed 1. We use
    # min(deg_i, deg_j), the tight upper bound on |N(i) cap N(j)|.
    # form="linear" is the paper's (1+kappa*c). Measured here: c_ij separates
    # prereq from kNN edges well (AUC 0.806, means 0.491 vs 0.240) BUT the
    # linear form caps the attainable prereq/kNN weight ratio at 0.491/0.240 =
    # 2.04x as kappa->inf, while the ablation says ~4x is wanted -- which is why
    # kappa=1 moved cold N@10 by only +0.0006. form="exp" removes that ceiling.
    if float(ncer_kappa) > 0.0:
        shared = (A @ A.T).tocsr()
        sh = np.asarray(shared[r, c]).flatten().astype(np.float32)
        denom = np.minimum(deg[r], deg[c]).astype(np.float32)
        denom[denom <= 0] = 1.0
        cons = np.clip(sh / denom, 0.0, 1.0)
        form = str(ncer_form or "linear").lower()
        if form == "linear":
            mult = 1.0 + float(ncer_kappa) * cons
        elif form == "exp":
            mult = np.exp(float(ncer_kappa) * cons)
        else:
            raise ValueError(f"unknown ncer_form={ncer_form!r}; use 'linear' or 'exp'")
        w_e = (w_e * mult).astype(np.float32)
        print(f"[kg] NCER form={form} kappa={ncer_kappa} c_ij mean={cons.mean():.4f} "
              f"p50={np.median(cons):.4f} max={cons.max():.4f}", flush=True)

    if float(edge_type_w) >= 0.0 or float(ncer_kappa) > 0.0:
        A = sp.csr_matrix((w_e, (r, c)), shape=(n_items, n_items), dtype=np.float32)
        wdeg = np.asarray(A.sum(axis=1)).flatten()
        wp, wk = w_e[is_pre], w_e[~is_pre]
        print(f"[kg] edge weights: prereq n={is_pre.sum()} mean={wp.mean():.4f} | "
              f"knn n={(~is_pre).sum()} mean={wk.mean():.4f} | "
              f"ratio={wp.mean()/max(wk.mean(),1e-9):.3f}x", flush=True)
    else:
        wdeg = deg

    # Degree used for the symmetric normalisation follows the (possibly
    # reweighted) graph; the RETURNED deg stays the binary count so that
    # kg_has_edge / kg_deg_norm keep their previous meaning.
    dinv = np.zeros_like(wdeg, dtype=np.float32)
    nz = wdeg > 0
    dinv[nz] = np.power(wdeg[nz], -0.5)
    D = sp.diags(dinv)
    norm = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (n_items, n_items)).coalesce().to(device)
    return adj, deg


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
        # Needed by the prereq aux loss AND by the KG channel, so load it when
        # either is enabled.
        self.prereq_idx = None
        self.prereq_mask = None
        kg_w = float(getattr(cfg, "kg_weight", 0.0))
        need_prereq = (cfg.prereq_aux_weight > 0.0) or (kg_w > 0.0)
        if need_prereq and cfg.prereq_path and os.path.exists(cfg.prereq_path):
            blob = torch.load(cfg.prereq_path, map_location="cpu", weights_only=False)
            self.prereq_idx = blob["prereq_idx"].long()
            self.prereq_mask = blob["has_prereq"].bool()
            print(f"[graph] loaded prereq index has_prereq="
                  f"{int(self.prereq_mask.sum())}/{self.prereq_mask.numel()}", flush=True)
        # item train degree -> degree-gated fusion beta. set by set_item_degree().
        self.register_buffer("item_beta", torch.zeros(cfg.n_items))
        # log-normalized train degree in [0,1], used by the DIF gate.
        self.register_buffer("item_deg_norm", torch.zeros(cfg.n_items))
        # raw train popularity (for tail-only ID-dropout eligibility).
        self.register_buffer("item_train_pop", torch.zeros(cfg.n_items))
        # ---- KG channel (off unless cfg.kg_weight > 0) ----
        # Buffers are non-persistent so checkpoints stay key-compatible with
        # legacy (kg-off) runs.
        self.kg_gate = None
        self.cold_gen = None
        self.register_buffer("kg_adj", None, persistent=False)
        self.register_buffer("kg_deg_norm", torch.zeros(cfg.n_items), persistent=False)
        self.register_buffer("kg_has_prereq", torch.zeros(cfg.n_items), persistent=False)
        self.register_buffer("kg_has_edge", torch.zeros(cfg.n_items), persistent=False)
        self.register_buffer("kg_scope_w", torch.zeros(cfg.n_items), persistent=False)
        if kg_w > 0.0:
            if self.prereq_idx is None:
                raise ValueError(
                    "cfg.kg_weight>0 requires a prereq index; "
                    f"missing or unreadable: {cfg.prereq_path!r}")
            adj_k, deg_k = build_knowledge_adj(
                self.prereq_idx, cfg.n_items,
                torch.device("cpu"),
                content_emb=content_emb,
                knn_k=int(getattr(cfg, "kg_knn_fallback", 0)),
                knn_scope=str(getattr(cfg, "kg_knn_scope", "missing")),
                ncer_kappa=float(getattr(cfg, "kg_ncer_kappa", 0.0)),
                ncer_form=str(getattr(cfg, "kg_ncer_form", "linear")),
                edge_type_w=float(getattr(cfg, "kg_edge_type_w", -1.0)),
                item_pop=getattr(cfg, "item_train_pop_np", None),
                min_neighbor_pop=float(getattr(cfg, "kg_min_neighbor_pop", 0.0)),
                cold_rewire_seed=int(getattr(cfg, "kg_cold_rewire_seed", -1)),
            )
            self.kg_adj = adj_k
            deg_t = torch.as_tensor(deg_k, dtype=torch.float32)
            log_k = torch.log1p(deg_t)
            kmax = float(log_k.max().item())
            self.kg_deg_norm = log_k / kmax if kmax > 0 else torch.zeros_like(log_k)
            self.kg_has_prereq = self.prereq_mask.float()
            self.kg_has_edge = (deg_t > 0).float()
            self.kg_gate = nn.Sequential(
                nn.Linear(3, 16), nn.GELU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            )
            # Start small (sigmoid(kg_gate_bias)) and learn to open up. Do NOT
            # zero-init the last weight: that makes w item-independent at init
            # and starves the gate's first layer of gradient, defeating the
            # point of a PER-ITEM gate. The negative bias alone gives the
            # "start near the known-good tail behaviour" property.
            nn.init.constant_(self.kg_gate[2].bias, float(getattr(cfg, "kg_gate_bias", -2.0)))
            if bool(getattr(cfg, "cold_gen_enable", False)):
                h = int(getattr(cfg, "cold_gen_hidden", 256))
                self.cold_gen = nn.Sequential(
                    nn.Linear(cfg.emb_dim * 2, h), nn.GELU(), nn.Linear(h, cfg.emb_dim))
                print(f"[cold-gen] learned cold pathway ON hidden={h} "
                      f"align_w={getattr(cfg, 'cold_gen_align_weight', 1.0)} "
                      f"mask_rho={getattr(cfg, 'cold_gen_mask_rho', 0.3)}", flush=True)
            print(f"[kg] knowledge adj nnz={self.kg_adj._nnz()} "
                  f"items_with_edges={int((deg_t > 0).sum())}/{cfg.n_items} "
                  f"mean_deg={float(deg_t.mean()):.2f} "
                  f"kg_weight={kg_w} layers={cfg.kg_layers} scope={cfg.kg_scope}",
                  flush=True)

    def set_item_degree(self, item_pop: np.ndarray):
        """beta_i = deg_i / (deg_i + K). cold(deg=0)->0, hot->~1."""
        deg = torch.as_tensor(item_pop, dtype=torch.float32)
        beta = deg / (deg + float(self.cfg.gate_k))
        dev = self.item_id_emb.weight.device
        self.item_beta = beta.to(dev)
        self.item_train_pop = deg.to(dev)
        log_deg = torch.log1p(deg)
        denom = float(log_deg.max().item())
        self.item_deg_norm = (log_deg / denom if denom > 0
                              else torch.zeros_like(log_deg)).to(dev)
        # Which items may receive knowledge propagation. Cold items are the
        # target; "cold_tail" also lets low-pop warm items in, matching the
        # tail-dropout scope used elsewhere in this line of experiments.
        scope = str(getattr(self.cfg, "kg_scope", "cold_tail") or "cold_tail").lower()
        cold_thr = float(self.cfg.cold_threshold)
        if scope == "all":
            scope_w = torch.ones_like(deg)
        elif scope == "cold":
            scope_w = (deg < cold_thr).float()
        elif scope == "cold_tail":
            max_pop = float(getattr(self.cfg, "kg_tail_max_pop", 25))
            scope_w = ((deg < cold_thr) | (deg <= max_pop)).float()
        else:
            raise ValueError(f"unknown kg_scope={scope!r}; "
                             "use 'cold', 'cold_tail' or 'all'")
        self.kg_scope_w = scope_w.to(dev)

    def item_ego(self, apply_id_dropout: bool, cold_mask_all: torch.Tensor,
                 return_drop_mask: bool = False):
        """Ego (pre-propagation) item vectors for ALL items.

        cold_mask_all: (n_items,) bool — strict-cold items force id_e=0.
        Returns fused (n_items,d), id_e_true (n_items,d), content_e (n_items,d).
        If return_drop_mask=True, also returns the stochastic ID-drop mask
        applied this forward (excludes permanent strict-cold rows).
        """
        all_idx = torch.arange(self.cfg.n_items, device=self.item_id_emb.weight.device)
        id_e_true = self.item_id_emb(all_idx)
        id_e = id_e_true
        mask = cold_mask_all.view(-1, 1).clone()
        drop_mask = torch.zeros((self.cfg.n_items, 1), dtype=torch.bool, device=id_e.device)
        if apply_id_dropout and self.training and self.cfg.dropout_prob > 0:
            drop = torch.rand((self.cfg.n_items, 1), device=id_e.device) < float(self.cfg.dropout_prob)
            mode = str(getattr(self.cfg, "dropout_mode", "uniform") or "uniform").lower()
            if mode == "tail":
                # Only low-pop warm items are eligible; head items never forced cold.
                max_pop = float(getattr(self.cfg, "dropout_max_pop", 25))
                cold_thr = float(getattr(self.cfg, "cold_threshold", 1))
                pop = self.item_train_pop.view(-1, 1)
                eligible = (pop >= cold_thr) & (pop <= max_pop)
                drop = drop & eligible
            elif mode != "uniform":
                raise ValueError(f"unknown dropout_mode={mode!r}; use 'uniform' or 'tail'")
            drop_mask = drop
            mask = mask | drop
        id_e = torch.where(mask, torch.zeros_like(id_e), id_e)
        content_e = self.content_proj(self.item_con_emb(all_idx))
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        fused = alpha * id_e + (1.0 - alpha) * content_e
        if return_drop_mask:
            return fused, id_e_true, content_e, drop_mask.view(-1)
        return fused, id_e_true, content_e

    @torch.no_grad()
    def _radial_target(self, item_unit: torch.Tensor,
                       u_hat: torch.Tensor,
                       source_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return a detached radius target from warm prerequisite neighbours.

        Only warm neighbours (train_pop >= cold_threshold) may supervise a
        target. This prevents a cold-to-cold recursive target and makes the
        inference path depend only on information available before test.
        """
        radius = (item_unit.detach() @ u_hat.detach()).detach()
        warm = self.item_train_pop >= float(self.cfg.cold_threshold)
        if source_mask is not None:
            warm = warm & source_mask.to(device=warm.device, dtype=torch.bool)
        warm_radius = radius[warm]
        if warm_radius.numel() == 0:
            fallback = radius.mean()
        else:
            fallback = warm_radius.mean()
        if self.prereq_idx is None:
            return torch.full_like(radius, fallback)
        pi = self.prereq_idx.to(device=radius.device)
        valid = pi >= 0
        safe = pi.clamp_min(0).clamp_max(radius.numel() - 1)
        nb_warm = warm.index_select(0, safe.view(-1)).view_as(safe)
        use = valid & nb_warm
        nb_radius = radius.index_select(0, safe.view(-1)).view_as(safe)
        counts = use.sum(dim=1)
        summed = (nb_radius * use.to(nb_radius.dtype)).sum(dim=1)
        target = torch.where(
            counts > 0, summed / counts.clamp_min(1).to(summed.dtype),
            torch.full_like(summed, fallback),
        )
        return target

    def _calibrate_radius(self, z_i: torch.Tensor,
                          beta: torch.Tensor,
                          z_u: torch.Tensor) -> torch.Tensor:
        """Calibrate only beta==0 rows while preserving their direction.

        The user-cloud axis is detached and the target is computed from warm
        prerequisite neighbours. The returned rows are unit vectors; scoring
        normalizes banks anyway, and the unit parameterization makes the
        radial/tangential decomposition exact.
        """
        if not bool(getattr(self.cfg, "radial_calibration", False)):
            return z_i
        z_unit = F.normalize(z_i, dim=1)
        u_hat = F.normalize(F.normalize(z_u, dim=1).mean(dim=0), dim=0).detach()
        eps = float(getattr(self.cfg, "radial_eps", 1e-6))
        target = self._radial_target(
            z_unit, u_hat, source_mask=beta.view(-1) > eps).clamp(-0.999, 0.999)
        rows = beta.view(-1) <= eps
        row_ids = torch.nonzero(rows, as_tuple=False).view(-1)
        if row_ids.numel() == 0:
            return z_i
        z = z_unit.index_select(0, row_ids)
        perp = z - (z @ u_hat).unsqueeze(1) * u_hat.unsqueeze(0)
        perp = F.normalize(perp, dim=1)
        a = target.index_select(0, row_ids).unsqueeze(1)
        calibrated = a * u_hat.unsqueeze(0) + (1.0 - a.square()).sqrt() * perp
        out = z_i.clone()
        out.index_copy_(0, row_ids, F.normalize(calibrated, dim=1))
        return out

    def cold_gen_vectors(self):
        """Learned reconstruction of an item's warm ID vector from its KG
        neighbourhood plus its own content, for ALL items.

        Input is the warm-only neighbour aggregate over kg_adj (zero diagonal, so
        an item never reads itself) concatenated with its content projection. A
        cold item's own ID embedding never enters, so this is leak-free by the
        same argument as the content channel; the neighbours' ID embeddings are
        detached so the generator cannot reshape the collaborative space to make
        its own regression easier.
        """
        if self.cold_gen is None or self.kg_adj is None:
            return None
        src = self.item_id_emb.weight.detach()
        warm = (self.item_train_pop >= float(self.cfg.cold_threshold)).to(src.dtype).view(-1, 1)
        agg = torch.sparse.mm(self.kg_adj, src * warm)
        content_e = self.content_proj(self.item_con_emb.weight)
        return self.cold_gen(torch.cat([F.normalize(agg, dim=1), content_e], dim=1))

    def propagate(self, adj: torch.Tensor, item_ego: torch.Tensor,
                  beta_override: torch.Tensor | None = None,
                  kg_scope_override: torch.Tensor | None = None):
        """LightGCN + degree-gated fusion.

        Users: pure post-graph (all users have interactions).
        Items: beta*post_graph + (1-beta)*ego. Cold items (beta=0) keep their
        pure content ego (no graph dilution); hot items (beta~1) use the
        collaborative post-graph vector. Returns z_u_all, z_i_all.

        beta_override: optional (n_items,) or (n_items,1) replacement for
        self.item_beta on this forward only (used by train-time cold simulation).

        kg_scope_override: optional (n_items,) or (n_items,1) replacement for
        the static KG receiver scope on this forward only. It is used by the
        pseudo-cold donor-training path; evaluation callers leave it unset.

        When cfg.kg_weight > 0 a third channel is mixed in: item vectors are
        additionally propagated over the prereq knowledge graph, where cold
        items (unlike in the user-item graph) actually have neighbours. The
        mix is z_i <- (1-w)*z_i + w*z_i_kg with a learned per-item gate w, so
        kg_weight=0 reproduces the legacy path exactly.
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
        if beta_override is None:
            beta = self.item_beta.view(-1, 1)
        else:
            beta = beta_override.view(-1, 1).to(dtype=z_i_post.dtype, device=z_i_post.device)
        z_i = beta * z_i_post + (1.0 - beta) * item_ego
        z_i = self._calibrate_radius(z_i, beta, z_u)
        if self.kg_adj is not None and float(getattr(self.cfg, "kg_weight", 0.0)) > 0.0:
            # Propagate the COLLABORATIVE item vectors over the knowledge graph
            # so a cold item aggregates its warm prereq neighbours' post-graph
            # vectors -- i.e. it reaches real users through those courses.
            # Same edges at train and eval, so train/inference stay consistent.
            # Pure NEIGHBOUR aggregate: the layer-0 self term is deliberately
            # excluded, because z_i already carries this item's own ego and
            # folding it in again would halve the rotation that w buys -- the
            # same "mechanism present but mechanically weakened" trap that
            # sank the soft-refine attempt.
            h_k = z_i_post
            kg_embs = []
            for _ in range(int(getattr(self.cfg, "kg_layers", 1))):
                h_k = torch.sparse.mm(self.kg_adj, h_k)
                kg_embs.append(h_k)
            z_i_kg = torch.stack(kg_embs, dim=1).mean(dim=1)
            # ---- NULL MODEL: drop the graph entirely -------------------------
            # The cold-rewire control showed that randomising WHICH warm items a
            # cold item borrows from costs only 7.6% of the gain (0.2266 ->
            # 0.2216 vs 0.1606 without the channel), i.e. neighbour identity is
            # nearly irrelevant. This goes one step further and removes selection
            # altogether: every cold item receives the plain mean of all warm
            # collaborative vectors. If cold N@10 still lands near 0.22 then the
            # whole item-item graph reduces to a single centroid shift and must
            # be reported as such rather than as knowledge propagation.
            if bool(getattr(self.cfg, "kg_cold_centroid", False)):
                warm_m = (self.item_train_pop
                          >= float(self.cfg.cold_threshold)).to(z_i_post.dtype).view(-1, 1)
                centroid = (z_i_post * warm_m).sum(dim=0, keepdim=True) / warm_m.sum().clamp(min=1.0)
                z_i_kg = centroid.expand_as(z_i_post)
            # ---- learned cold pathway: replace the fixed aggregate ------------
            # Same mixing algebra and the same scope/edge gating as before, so
            # cold_gen_enable=0 is bit-identical. The difference is that z_i_kg
            # now carries parameters, which is what makes the cold row reachable
            # by gradient once kg_train_pseudocold opens the scope on dropped rows.
            if self.cold_gen is not None:
                gen = self.cold_gen_vectors()
                if gen is not None:
                    z_i_kg = gen
            fixed_w = float(getattr(self.cfg, "kg_fixed_w", -1.0))
            if fixed_w >= 0.0:
                # Constant channel strength: no gate to collapse. See the
                # cfg.kg_fixed_w comment for why the learned gate cannot work
                # here. The channel then has no parameters of its own; the rest
                # of the model still trains WITH it active, which is the
                # train/inference consistency this line of work is about.
                gate = torch.full((self.cfg.n_items, 1), fixed_w,
                                  device=z_i.device, dtype=z_i.dtype)
            else:
                feats = torch.stack(
                    [self.item_deg_norm, self.kg_has_prereq, self.kg_deg_norm], dim=1)
                gate = self.kg_gate(feats)
            # kg_has_edge makes items without knowledge edges an exact no-op
            # instead of perturbing them toward their own post-graph vector,
            # and keeps the normalize() below away from zero rows.
            if kg_scope_override is None:
                scope_w = self.kg_scope_w.view(-1, 1)
            else:
                scope_w = kg_scope_override.view(-1, 1).to(
                    device=z_i.device, dtype=z_i.dtype)
                if scope_w.size(0) != self.cfg.n_items:
                    raise ValueError(
                        "kg_scope_override must have one value per item"
                    )
            w = (gate
                 * float(self.cfg.kg_weight)
                 * scope_w
                 * self.kg_has_edge.view(-1, 1))
            # Scale-match before mixing. Without this the mix is silently
            # weakened by the layer-mean, so w would not mean what it says.
            scale = z_i.norm(dim=1, keepdim=True).detach()
            z_i_kg = F.normalize(z_i_kg, dim=1) * scale
            z_i = (1.0 - w) * z_i + w * z_i_kg
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


def _cold_gen_align(model, device):
    """DiffCold-style simulation alignment for the learned cold pathway.

    Sample mask_rho of the WARM items as pseudo-cold and require the generator --
    which sees only their neighbourhood aggregate and their own content, never
    their own ID embedding -- to reproduce their real warm ID vector. InfoNCE
    rather than MSE so the objective is about ranking geometry, matching how the
    vectors are consumed at scoring time.

    Independent of dropout_mode="tail": that pool is 95 items and yields ~33 rows
    per step, too thin to fit a generator, and restricting to the tail would teach
    the generator only the low-popularity corner of the warm manifold.
    """
    cfg = model.cfg
    if model.cold_gen is None:
        return torch.zeros((), device=device)
    w = float(getattr(cfg, "cold_gen_align_weight", 1.0))
    rho = float(getattr(cfg, "cold_gen_mask_rho", 0.30))
    if w <= 0.0 or rho <= 0.0:
        return torch.zeros((), device=device)
    warm = torch.nonzero(
        (model.item_train_pop >= float(cfg.cold_threshold))
        & (model.kg_has_edge > 0), as_tuple=False).view(-1)
    if warm.numel() <= 1:
        return torch.zeros((), device=device)
    k = max(2, int(round(rho * float(warm.numel()))))
    pick = warm[torch.randperm(warm.numel(), device=warm.device)[:k]]
    gen = model.cold_gen_vectors()
    if gen is None:
        return torch.zeros((), device=device)
    z_gen = F.normalize(gen.index_select(0, pick), dim=1)
    z_tgt = F.normalize(model.item_id_emb.weight.detach().index_select(0, pick), dim=1)
    logits = torch.mm(z_gen, z_tgt.t()) / cfg.temp
    labels = torch.arange(logits.size(0), device=device)
    return w * F.cross_entropy(logits, labels)


def _prereq_aux(model, ego_norm_all, device,
                excluded_source_mask: torch.Tensor | None = None):
    """Prereq anchor on all ego vectors using only allowed neighbour ID sources."""
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
    if excluded_source_mask is not None:
        excluded = excluded_source_mask.to(device=device, dtype=torch.bool).view(-1)
        if excluded.numel() != ego_norm_all.size(0):
            raise ValueError("excluded_source_mask must have one entry per item")
        valid = valid & ~excluded.index_select(0, pre_safe.view(-1)).view_as(pre_safe)
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


def _dif_ce(cfg, model, logits, targets, i_idx):
    """Cross-entropy with DIF degree-gated, uncertainty-capped sample weights.

    Falls back to plain mean CE when DIF is disabled, so default behaviour of
    every existing run is bit-identical.
    """
    if not getattr(cfg, "dif_enable", False):
        return F.cross_entropy(logits, targets)
    per = F.cross_entropy(logits, targets, reduction="none")
    with torch.no_grad():
        g = model.item_deg_norm.index_select(0, i_idx)      # [0,1], head->1
        gamma = torch.exp(-float(cfg.dif_alpha) * g)        # low-freq->1, head->0
        r = per.detach()                                    # label-fit residual
        t = r * gamma
        w = t / (t + 1.0)
        weight = 1.0 - float(cfg.dif_max_w) * w             # uncertainty cap
    return (per * weight).sum() / weight.sum().clamp_min(1e-8)


def graph_loss(model, adj, cold_mask_all, u_idx, i_idx, device,
               return_item_bank: bool = False, refine_fn=None,
               return_pseudocold_state: bool = False,
               pseudocold_adj_fn=None):
    """InfoNCE + aux + prereq on propagated z.

    refine_fn: optional differentiable callable z_i_all -> z_i_all' that
    replaces a subset of item rows (the align rows) with operator-refined
    vectors IN-GRAPH, before scoring. When None (every legacy caller), the
    forward is bit-identical to the original. This is the train-time hook for
    the soft-refine (route-A) arm: the InfoNCE ranking then sees refined cold
    rows exactly as inference does, so training and inference share one module.

    return_pseudocold_state returns the stochastic drop mask and both propagated
    banks for counterfactual supervision. pseudocold_adj_fn(drop_mask) may
    provide a re-normalized graph with every dropped item's train edge removed;
    it is never called on the legacy path.
    """
    cfg = model.cfg
    # full-graph ego + propagation (LightGCN computes all nodes once)
    force_beta0 = bool(getattr(cfg, "dropout_force_beta0", False))
    need_drop_mask = force_beta0 or return_pseudocold_state or pseudocold_adj_fn is not None
    if need_drop_mask:
        item_ego, id_e_true, content_e, drop_mask = model.item_ego(
            apply_id_dropout=True, cold_mask_all=cold_mask_all, return_drop_mask=True)
        # Match strict-cold inference path on stochastically dropped rows:
        # id already zeroed; also kill graph contribution via beta=0.
        beta = model.item_beta.clone()
        if force_beta0 and bool(drop_mask.any().item()):
            beta[drop_mask] = 0.0
        effective_adj = pseudocold_adj_fn(drop_mask) if pseudocold_adj_fn is not None else adj
        kg_scope_override = None
        if (bool(getattr(cfg, "kg_train_pseudocold", False))
                and model.training and bool(drop_mask.any().item())):
            kg_scope_override = model.kg_scope_w.clone()
            kg_scope_override[drop_mask] = 1.0
        z_u_all, z_i_all = model.propagate(
            effective_adj,
            item_ego,
            beta_override=beta if force_beta0 else None,
            kg_scope_override=kg_scope_override,
        )
    else:
        drop_mask = torch.zeros(model.cfg.n_items, dtype=torch.bool, device=device)
        item_ego, id_e_true, content_e = model.item_ego(
            apply_id_dropout=True, cold_mask_all=cold_mask_all)
        z_u_all, z_i_all = model.propagate(adj, item_ego)
    if refine_fn is not None:
        # Differentiable in-graph refinement of the align rows (route A). The
        # returned tensor must be a full (n_items,d) bank with only align rows
        # changed; gradient flows back through z_i_all into content_proj/gate.
        z_i_all = refine_fn(z_i_all)
    z_u = F.normalize(z_u_all.index_select(0, u_idx), dim=1)
    z_i = F.normalize(z_i_all.index_select(0, i_idx), dim=1)
    logits = torch.mm(z_u, z_i.t()) / cfg.temp
    n = logits.size(0)
    labels = torch.arange(n, device=device)
    pos_mask = torch.eye(n, dtype=torch.bool, device=device)
    logits_m = logits.clone()
    logits_m[pos_mask] -= cfg.margin / cfg.temp
    # A pseudo-cold target must not recover its masked ID through the auxiliary
    # ID-content objective. Legacy calls keep every batch row exactly as before.
    aux_i_idx = i_idx
    if return_pseudocold_state and bool(drop_mask.any().item()):
        aux_i_idx = i_idx[~drop_mask.index_select(0, i_idx)]
    if aux_i_idx.numel() == 0:
        aux = item_ego.sum() * 0.0
    else:
        aux = _aux_infonce(cfg, id_e_true.index_select(0, aux_i_idx),
                           content_e.index_select(0, aux_i_idx), device)
    prereq = _prereq_aux(
        model, F.normalize(item_ego, dim=1), device,
        excluded_source_mask=drop_mask if return_pseudocold_state else None)
    cgen = _cold_gen_align(model, device)

    def finish(total_loss):
        if return_pseudocold_state:
            return total_loss, z_i_all, drop_mask, z_u_all
        if return_item_bank:
            return total_loss, z_i_all
        return total_loss

    if n <= 1:
        return finish(_dif_ce(cfg, model, logits_m, labels, i_idx) + aux + prereq + cgen)
    max_neg = n - 1
    n_total = min(cfg.train_num_negs, max_neg)
    if n_total <= 0:
        return finish(_dif_ce(cfg, model, logits_m, labels, i_idx) + aux + prereq + cgen)
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
    return finish(_dif_ce(cfg, model, cand_logits, targets, i_idx) + aux + prereq + cgen)


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
    ap.add_argument("--kg-weight", dest="kg_weight", type=float, default=0.0,
                    help="knowledge-channel mix weight; 0 (default) = legacy path")
    ap.add_argument("--kg-layers", dest="kg_layers", type=int, default=1,
                    help="item-item propagation depth on the knowledge graph")
    ap.add_argument("--kg-scope", dest="kg_scope", default="cold_tail",
                    choices=["cold", "cold_tail", "all"],
                    help="which items may receive knowledge propagation")
    ap.add_argument("--kg-knn-fallback", dest="kg_knn_fallback", type=int, default=0,
                    help="content-kNN edges for items with no prereq edge (0=off)")
    ap.add_argument("--kg-ncer-kappa", dest="kg_ncer_kappa", type=float, default=0.0,
                    help="NCER edge reweighting strength (triadic closure); "
                         "0 (default) = binary graph, bit-identical to before")
    ap.add_argument("--kg-fixed-w", dest="kg_fixed_w", type=float, default=-1.0,
                    help="constant channel strength instead of the learned gate "
                         "(>=0 enables; the gate closes on its own because cold "
                         "items contribute no training rows)")
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
    cfg.kg_weight = float(args.kg_weight)
    cfg.kg_layers = int(args.kg_layers)
    cfg.kg_scope = str(args.kg_scope)
    cfg.kg_knn_fallback = int(args.kg_knn_fallback)
    cfg.kg_ncer_kappa = float(args.kg_ncer_kappa)
    cfg.kg_fixed_w = float(args.kg_fixed_w)
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

