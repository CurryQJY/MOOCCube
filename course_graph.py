import math

import torch
import torch.nn as nn

from usim import build_course_artifacts


def _row_topk_neighbors(scores, topk, min_value=0.0, exclude_self=False):
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")

    n_rows, n_cols = scores.shape
    if n_rows < 1 or n_cols < 1 or topk < 1:
        idx = torch.full((n_rows, max(1, topk)), -1, dtype=torch.long)
        w = torch.zeros((n_rows, max(1, topk)), dtype=torch.float32)
        return idx, w

    work = scores.clone()
    if exclude_self and n_rows == n_cols:
        diag = torch.arange(n_rows)
        work[diag, diag] = 0.0

    k = min(int(topk), n_cols)
    top_w, top_idx = torch.topk(work, k=k, dim=1)
    valid = top_w > float(min_value)
    top_idx = top_idx.masked_fill(~valid, -1)
    top_w = top_w.masked_fill(~valid, 0.0)

    if k < topk:
        pad_idx = torch.full((n_rows, topk - k), -1, dtype=torch.long)
        pad_w = torch.zeros((n_rows, topk - k), dtype=torch.float32)
        top_idx = torch.cat([top_idx, pad_idx], dim=1)
        top_w = torch.cat([top_w, pad_w], dim=1)

    denom = top_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    top_w = top_w / denom
    return top_idx.long(), top_w.float()


def _compute_item_difficulty(df, n_items):
    difficulty = torch.zeros(n_items, dtype=torch.float32)
    if "i_idx" not in df.columns or "popularity" not in df.columns:
        return difficulty

    pop = (
        df[["i_idx", "popularity"]]
        .dropna()
        .groupby("i_idx", sort=False)["popularity"]
        .median()
    )
    if pop.empty:
        return difficulty

    pop_values = torch.zeros(n_items, dtype=torch.float32)
    for i_idx, value in pop.items():
        item_idx = int(i_idx)
        if 0 <= item_idx < n_items:
            pop_values[item_idx] = float(value)

    max_pop = float(pop_values.max().item())
    if max_pop <= 0.0:
        return difficulty

    # Lower popularity is used as a weak difficulty prior.
    norm = torch.log1p(pop_values) / math.log1p(max_pop)
    difficulty = 1.0 - norm
    return difficulty.clamp(0.0, 1.0)


def _compute_item_popularity(df, n_items):
    popularity = torch.zeros(n_items, dtype=torch.float32)
    if "i_idx" not in df.columns or "popularity" not in df.columns:
        return popularity

    pop = (
        df[["i_idx", "popularity"]]
        .dropna()
        .groupby("i_idx", sort=False)["popularity"]
        .median()
    )
    if pop.empty:
        return popularity

    for i_idx, value in pop.items():
        item_idx = int(i_idx)
        if 0 <= item_idx < n_items:
            popularity[item_idx] = float(value)
    return popularity


def build_graph_course_artifacts(
    df,
    n_items,
    relation_dir="MOOCCube/relations",
    prereq_min_support=30,
    prereq_max_per_item=5,
    prereq_min_items=1,
    prereq_max_forward=20,
    graph_topk_prereq=5,
    graph_topk_semantic=8,
):
    artifacts, stats = build_course_artifacts(
        df,
        n_items,
        relation_dir=relation_dir,
        prereq_min_support=prereq_min_support,
        prereq_max_per_item=prereq_max_per_item,
        prereq_min_items=prereq_min_items,
        prereq_max_forward=prereq_max_forward,
    )

    prereq_idx, prereq_w = _row_topk_neighbors(
        artifacts["item_prereq_item_mat"],
        topk=graph_topk_prereq,
        min_value=0.0,
        exclude_self=True,
    )
    semantic_idx, semantic_w = _row_topk_neighbors(
        artifacts["item_concept_overlap"],
        topk=graph_topk_semantic,
        min_value=0.0,
        exclude_self=True,
    )

    artifacts = dict(artifacts)
    artifacts["graph_prereq_idx"] = prereq_idx
    artifacts["graph_prereq_w"] = prereq_w
    artifacts["graph_semantic_idx"] = semantic_idx
    artifacts["graph_semantic_w"] = semantic_w
    artifacts["item_difficulty"] = _compute_item_difficulty(df, n_items)
    artifacts["item_popularity"] = _compute_item_popularity(df, n_items)

    stats = dict(stats)
    stats["graph_topk_prereq"] = int(graph_topk_prereq)
    stats["graph_topk_semantic"] = int(graph_topk_semantic)
    stats["graph_items_with_prereq_neighbors"] = int((prereq_w.sum(dim=1) > 0).sum().item())
    stats["graph_items_with_semantic_neighbors"] = int((semantic_w.sum(dim=1) > 0).sum().item())
    return artifacts, stats


class CourseGraphEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, prereq_weight=1.0, semantic_weight=1.0):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.prereq_weight = float(prereq_weight)
        self.semantic_weight = float(semantic_weight)

        self.delta_net = nn.Sequential(
            nn.Linear(self.emb_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.emb_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(self.emb_dim * 3, self.emb_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(self.emb_dim)

        self.register_buffer("graph_prereq_idx", None, persistent=False)
        self.register_buffer("graph_prereq_w", None, persistent=False)
        self.register_buffer("graph_semantic_idx", None, persistent=False)
        self.register_buffer("graph_semantic_w", None, persistent=False)

    def set_artifacts(self, artifacts, device):
        self.graph_prereq_idx = None
        self.graph_prereq_w = None
        self.graph_semantic_idx = None
        self.graph_semantic_w = None

        if not artifacts:
            return

        prereq_idx = artifacts.get("graph_prereq_idx")
        prereq_w = artifacts.get("graph_prereq_w")
        semantic_idx = artifacts.get("graph_semantic_idx")
        semantic_w = artifacts.get("graph_semantic_w")

        if prereq_idx is not None:
            self.graph_prereq_idx = prereq_idx.to(device)
        if prereq_w is not None:
            self.graph_prereq_w = prereq_w.to(device)
        if semantic_idx is not None:
            self.graph_semantic_idx = semantic_idx.to(device)
        if semantic_w is not None:
            self.graph_semantic_w = semantic_w.to(device)

    def _aggregate(self, item_idx, neighbor_idx, neighbor_w, seed_lookup_fn):
        batch_size = item_idx.size(0)
        device = item_idx.device
        if neighbor_idx is None or neighbor_w is None:
            return torch.zeros((batch_size, self.emb_dim), device=device)

        idx = neighbor_idx[item_idx]
        w = neighbor_w[item_idx]
        valid = idx >= 0
        if not valid.any():
            return torch.zeros((batch_size, self.emb_dim), device=device)

        safe_idx = idx.masked_fill(~valid, 0)
        neighbor_seed = seed_lookup_fn(safe_idx.view(-1)).view(batch_size, -1, self.emb_dim)
        masked_seed = neighbor_seed * valid.unsqueeze(-1).float()
        weights = w.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        return (masked_seed * weights).sum(dim=1) / denom

    def forward(self, item_idx, self_seed, seed_lookup_fn):
        prereq_ctx = self._aggregate(
            item_idx,
            self.graph_prereq_idx,
            self.graph_prereq_w,
            seed_lookup_fn,
        ) * self.prereq_weight
        semantic_ctx = self._aggregate(
            item_idx,
            self.graph_semantic_idx,
            self.graph_semantic_w,
            seed_lookup_fn,
        ) * self.semantic_weight

        fusion = torch.cat([self_seed, prereq_ctx, semantic_ctx], dim=-1)
        delta = self.delta_net(fusion)
        gate = self.gate_net(fusion)
        return self.out_norm(self_seed + gate * delta)
