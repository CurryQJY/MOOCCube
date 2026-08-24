import math
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn.functional as F


def _xavier_row_norm(n_users: int, emb_dim: int) -> float:
    fan_sum = max(1, int(n_users) + int(emb_dim))
    return math.sqrt(float(emb_dim) * 2.0 / float(fan_sum))


def _project_content(content_emb: torch.Tensor, emb_dim: int) -> torch.Tensor:
    x = torch.as_tensor(content_emb).detach().float().cpu()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return torch.zeros((int(x.shape[0]), int(emb_dim)), dtype=torch.float32)
    if int(x.shape[1]) == int(emb_dim):
        return F.normalize(x, dim=1)

    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = x / std
    x = x - x.mean(dim=0, keepdim=True)
    rank = min(int(emb_dim), int(x.shape[0]), int(x.shape[1]))
    if rank <= 0:
        return torch.zeros((int(x.shape[0]), int(emb_dim)), dtype=torch.float32)
    try:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.pca_lowrank(x, q=rank, center=False)
        vh = vh.t()
    z = torch.matmul(x, vh[:rank].t())
    if rank < int(emb_dim):
        pad = torch.zeros((z.size(0), int(emb_dim) - rank), dtype=z.dtype)
        z = torch.cat([z, pad], dim=1)
    return F.normalize(z, dim=1)


def _semantic_kmeans(
    item_vecs: torch.Tensor,
    cluster_k: int,
    seed: int,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_items = int(item_vecs.size(0))
    emb_dim = int(item_vecs.size(1))
    k = max(1, min(int(cluster_k), max(1, n_items)))
    if n_items == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros((0, emb_dim), dtype=torch.float32)
    if k == 1:
        assignments = torch.zeros(n_items, dtype=torch.long)
        centroid = F.normalize(item_vecs.mean(dim=0, keepdim=True), dim=1)
        return assignments, centroid

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    init_ids = torch.randperm(n_items, generator=generator)[:k]
    centroids = item_vecs.index_select(0, init_ids).clone()

    assignments = torch.zeros(n_items, dtype=torch.long)
    for _ in range(max(1, int(max_iter))):
        scores = torch.matmul(item_vecs, centroids.t())
        new_assignments = scores.argmax(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        next_centroids = centroids.clone()
        for cid in range(k):
            mask = assignments == cid
            if mask.any():
                next_centroids[cid] = item_vecs[mask].mean(dim=0)
        centroids = F.normalize(next_centroids, dim=1)

    return assignments, centroids


def build_sg_urinit_weights(
    train_df: pd.DataFrame,
    content_emb: torch.Tensor,
    n_users: int,
    emb_dim: int,
    cluster_k: int = 32,
    local_weight: float = 0.7,
    global_weight: float = 0.3,
    target_norm: float = 0.0,
    seed: int = 2025,
    max_iter: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build SG-URInit-style user weights from users' interacted item content."""
    n_users = int(n_users)
    emb_dim = int(emb_dim)
    item_vecs = _project_content(content_emb, emb_dim)
    n_items = int(item_vecs.size(0))
    assignments, centroids = _semantic_kmeans(item_vecs, cluster_k, seed, max_iter)

    local_weight = max(0.0, float(local_weight))
    global_weight = max(0.0, float(global_weight))
    if local_weight == 0.0 and global_weight == 0.0:
        local_weight = 1.0
    target_norm = float(target_norm) if float(target_norm) > 0.0 else _xavier_row_norm(n_users, emb_dim)

    weights = torch.zeros((n_users, emb_dim), dtype=torch.float32)
    initialized = torch.zeros(n_users, dtype=torch.bool)
    if train_df is None or len(train_df) == 0 or n_items == 0 or n_users == 0:
        return weights, initialized, {
            "enabled": True,
            "initialized_users": 0,
            "cold_users": n_users,
            "cluster_k": int(centroids.size(0)),
            "target_norm": target_norm,
        }

    pair_df = train_df[["u_idx", "i_idx"]].dropna()
    pair_df = pair_df.astype({"u_idx": "int64", "i_idx": "int64"})
    pair_df = pair_df[
        (pair_df["u_idx"] >= 0)
        & (pair_df["u_idx"] < n_users)
        & (pair_df["i_idx"] >= 0)
        & (pair_df["i_idx"] < n_items)
    ]

    for user_id, group in pair_df.groupby("u_idx", sort=False):
        item_ids = torch.as_tensor(group["i_idx"].drop_duplicates().to_numpy(), dtype=torch.long)
        if item_ids.numel() == 0:
            continue
        local_vec = item_vecs.index_select(0, item_ids).mean(dim=0)
        cluster_ids = assignments.index_select(0, item_ids)
        global_vec = centroids.index_select(0, cluster_ids).mean(dim=0)
        user_vec = local_weight * local_vec + global_weight * global_vec
        norm = user_vec.norm()
        if not torch.isfinite(norm) or norm <= 1e-12:
            continue
        weights[int(user_id)] = user_vec / norm * target_norm
        initialized[int(user_id)] = True

    return weights, initialized, {
        "enabled": True,
        "initialized_users": int(initialized.sum().item()),
        "cold_users": int(n_users - initialized.sum().item()),
        "cluster_k": int(centroids.size(0)),
        "local_weight": float(local_weight),
        "global_weight": float(global_weight),
        "target_norm": float(target_norm),
    }


@torch.no_grad()
def apply_sg_urinit_(model: Any, train_df: pd.DataFrame, content_emb: torch.Tensor, cfg: Any) -> Dict[str, Any]:
    if not bool(getattr(cfg, "use_sg_urinit", False)):
        return {"enabled": False, "initialized_users": 0}

    weights, initialized, stats = build_sg_urinit_weights(
        train_df=train_df,
        content_emb=content_emb,
        n_users=int(getattr(cfg, "n_users")),
        emb_dim=int(getattr(cfg, "emb_dim")),
        cluster_k=int(getattr(cfg, "sg_urinit_cluster_k", 32)),
        local_weight=float(getattr(cfg, "sg_urinit_local_weight", 0.7)),
        global_weight=float(getattr(cfg, "sg_urinit_global_weight", 0.3)),
        target_norm=float(getattr(cfg, "sg_urinit_target_norm", 0.0)),
        seed=int(getattr(cfg, "sg_urinit_seed", 2025)),
        max_iter=int(getattr(cfg, "sg_urinit_max_iter", 20)),
    )
    target = model.user_emb.weight
    mask = initialized.to(device=target.device)
    target[mask] = weights.to(device=target.device, dtype=target.dtype)[mask]
    stats["enabled"] = True
    return stats
