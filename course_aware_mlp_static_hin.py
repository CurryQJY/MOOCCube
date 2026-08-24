"""Supervised non-RL course-aware MLP baseline for strict item-cold ranking.

This control uses the same course-knowledge feature family as CKG-RL
(concepts, prerequisites, difficulty, and redundancy), but removes retrieval
simulation, actor-critic learning, and PPO. It is intended as a direct
course-aware reranking/scoring control under the shared static item-cold split.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fast3_delta.course_artifacts import build_course_artifacts
from hin_data_common import (
    InteractionDataset,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    static_result_path,
    static_split_df,
)
from hin_eval_common import compute_ranking_metric_values


K_LIST = (5, 10, 20)
METRIC_KEYS = [f"{m}@{k}" for m in ("R", "N") for k in K_LIST]


@dataclass
class Config:
    n_users: int
    n_items: int
    content_dim: int
    hidden_dim: int = int(os.environ.get("COURSE_MLP_HIDDEN_DIM", "64"))
    lr: float = float(os.environ.get("COURSE_MLP_LR", "1e-3"))
    weight_decay: float = float(os.environ.get("COURSE_MLP_WEIGHT_DECAY", "1e-5"))
    n_epochs: int = int(os.environ.get("COURSE_MLP_STATIC_EPOCHS", "40"))
    batch_size: int = int(os.environ.get("COURSE_MLP_BATCH_SIZE", "4096"))
    eval_batch_size: int = int(os.environ.get("COURSE_MLP_EVAL_BATCH_SIZE", "2048"))
    eval_interval: int = int(os.environ.get("COURSE_MLP_EVAL_INTERVAL", "5"))
    cold_threshold: int = int(os.environ.get("COURSE_MLP_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1")))
    static_seed: int = int(os.environ.get("COURSE_MLP_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
    seed: int = int(os.environ.get("COURSE_MLP_SEED", os.environ.get("USIM_SEED", "2025")))
    train_ratio: float = float(os.environ.get("COURSE_MLP_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio: float = float(os.environ.get("COURSE_MLP_STATIC_VAL_RATIO", "0.1"))
    relation_dir: str = os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations")
    prereq_graph_source: str = os.environ.get("USIM_PREREQ_GRAPH_SOURCE", "concept")
    prereq_max_per_item: int = int(os.environ.get("USIM_PREREQ_MAX_PER_ITEM", "5"))
    prereq_min_support: int = int(os.environ.get("USIM_PREREQ_MIN_SUPPORT", "30"))
    prereq_min_items: int = int(os.environ.get("USIM_PREREQ_MIN_ITEMS", "1"))
    prereq_max_forward: int = int(os.environ.get("USIM_PREREQ_MAX_FORWARD", "20"))
    prereq_concept_score_thr: float = float(os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", "0.10"))
    prereq_concept_min_hits: int = int(os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", "1"))
    prereq_concept_file: str = os.environ.get("USIM_PREREQ_CONCEPT_FILE", "prerequisite-dependency.json")
    concept_overlap_mode: str = os.environ.get("USIM_CONCEPT_OVERLAP_MODE", "plain")
    content_prior_weight: float = float(os.environ.get("COURSE_MLP_CONTENT_PRIOR_WEIGHT", "1.0"))
    correction_scale: float = float(os.environ.get("COURSE_MLP_CORRECTION_SCALE", "0.2"))


class CourseAwareScorer(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, content_prior_weight: float, correction_scale: float):
        super().__init__()
        self.content_prior_weight = float(content_prior_weight)
        self.correction_scale = float(correction_scale)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        content_prior = features[:, 0]
        correction = self.net(features).squeeze(-1)
        return self.content_prior_weight * content_prior + self.correction_scale * correction


class FeatureBuilder:
    def __init__(
        self,
        cfg: Config,
        train_df: pd.DataFrame,
        full_df: pd.DataFrame,
        content_emb: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.content = F.normalize(content_emb.float(), dim=1).to(device)

        train_pop = torch.zeros(cfg.n_items, dtype=torch.float32)
        counts = train_df["i_idx"].value_counts()
        for item_id, count in counts.items():
            item_id = int(item_id)
            if 0 <= item_id < cfg.n_items:
                train_pop[item_id] = float(count)
        max_log = torch.log1p(train_pop).max().clamp_min(1.0)
        self.item_difficulty = (1.0 - torch.log1p(train_pop) / max_log).clamp(0.0, 1.0).to(device)

        seen = torch.zeros((cfg.n_users, cfg.n_items), dtype=torch.bool)
        for uid, item_id in zip(train_df["u_idx"].values, train_df["i_idx"].values):
            uid = int(uid)
            item_id = int(item_id)
            if 0 <= uid < cfg.n_users and 0 <= item_id < cfg.n_items:
                seen[uid, item_id] = True
        self.seen_cpu = seen
        self.seen = seen.to(device)

        user_content = torch.zeros((cfg.n_users, cfg.content_dim), dtype=torch.float32)
        content_cpu = self.content.cpu()
        chunk = 4096
        seen_float = seen.float()
        for start in range(0, cfg.n_users, chunk):
            end = min(cfg.n_users, start + chunk)
            hist = seen_float[start:end].sum(dim=1, keepdim=True).clamp_min(1.0)
            user_content[start:end] = torch.mm(seen_float[start:end], content_cpu) / hist
        self.user_content = F.normalize(user_content, dim=1).to(device)
        max_hist = seen_float.sum(dim=1).max().clamp_min(1.0)
        self.max_hist_log = float(torch.log1p(max_hist).item())

        artifacts, stats = build_course_artifacts(
            full_df,
            cfg.n_items,
            relation_dir=cfg.relation_dir,
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
            concept_overlap_mode=cfg.concept_overlap_mode,
            prereq_graph_source=cfg.prereq_graph_source,
            prereq_concept_score_thr=cfg.prereq_concept_score_thr,
            prereq_concept_min_hits=cfg.prereq_concept_min_hits,
            prereq_concept_file=cfg.prereq_concept_file,
        )
        self.artifact_stats = stats
        self.prereq_mat = artifacts["item_prereq_item_mat"].float().to(device)
        self.prereq_cnt = artifacts["item_prereq_item_cnt"].float().to(device)
        self.concept_overlap = artifacts["item_concept_overlap"].float().to(device)
        self.video_contain = artifacts["item_video_contain"].float().to(device)
        self.same_family = artifacts["item_same_family"].float().to(device)

    def sample_negatives(self, users: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        users_cpu = users.detach().cpu()
        pos_cpu = positives.detach().cpu()
        neg = torch.randint(0, self.cfg.n_items, pos_cpu.shape, dtype=torch.long)
        for _ in range(20):
            bad = self.seen_cpu[users_cpu, neg] | (neg == pos_cpu)
            if not bool(bad.any()):
                break
            neg[bad] = torch.randint(0, self.cfg.n_items, (int(bad.sum()),), dtype=torch.long)
        bad = self.seen_cpu[users_cpu, neg] | (neg == pos_cpu)
        if bool(bad.any()):
            all_items = torch.arange(self.cfg.n_items)
            for idx in bad.nonzero(as_tuple=False).view(-1).tolist():
                allowed = ~(self.seen_cpu[int(users_cpu[idx])]) & (all_items != int(pos_cpu[idx]))
                choices = all_items[allowed]
                neg[idx] = choices[torch.randint(0, choices.numel(), (1,)).item()]
        return neg.to(self.device)

    def pair_features(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        seen = self.seen.index_select(0, users).float()
        hist = seen.sum(dim=1, keepdim=True)
        hist_safe = hist.clamp_min(1.0)
        user_vec = self.user_content.index_select(0, users)
        item_vec = self.content.index_select(0, items)

        content_sim = (user_vec * item_vec).sum(dim=1, keepdim=True)
        concept = torch.mm(seen, self.concept_overlap.t()).gather(1, items.view(-1, 1)) / hist_safe
        prereq_seen = torch.mm(seen, self.prereq_mat.t()).gather(1, items.view(-1, 1))
        prereq_cnt = self.prereq_cnt.index_select(0, items).view(-1, 1)
        prereq_violation = torch.where(
            prereq_cnt > 0,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)
        readiness = (torch.log1p(hist_safe) / max(1e-6, self.max_hist_log)).clamp(0.0, 1.0)
        difficulty = self.item_difficulty.index_select(0, items).view(-1, 1)
        difficulty_gap = F.relu(difficulty - readiness)
        family = (self.same_family.index_select(0, items) * seen).amax(dim=1, keepdim=True)
        video = (self.video_contain.index_select(0, items) * seen).amax(dim=1, keepdim=True)
        redundancy = torch.maximum(family, video)
        hist_norm = readiness
        return torch.cat(
            [content_sim, concept, prereq_violation, difficulty, difficulty_gap, redundancy, hist_norm],
            dim=1,
        )

    def full_features(self, users: torch.Tensor) -> torch.Tensor:
        seen = self.seen.index_select(0, users).float()
        hist = seen.sum(dim=1, keepdim=True)
        hist_safe = hist.clamp_min(1.0)
        user_vec = self.user_content.index_select(0, users)

        content_sim = torch.mm(user_vec, self.content.t())
        concept = torch.mm(seen, self.concept_overlap.t()) / hist_safe
        prereq_seen = torch.mm(seen, self.prereq_mat.t())
        prereq_cnt = self.prereq_cnt.view(1, -1)
        prereq_violation = torch.where(
            prereq_cnt > 0,
            1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
            torch.zeros_like(prereq_seen),
        ).clamp(0.0, 1.0)
        readiness = (torch.log1p(hist_safe) / max(1e-6, self.max_hist_log)).clamp(0.0, 1.0)
        difficulty = self.item_difficulty.view(1, -1).expand_as(content_sim)
        difficulty_gap = F.relu(difficulty - readiness)
        family = torch.matmul(seen, self.same_family.t()).clamp(max=1.0)
        video = torch.matmul(seen, self.video_contain.t()).clamp(max=1.0)
        redundancy = torch.maximum(family, video)
        hist_norm = readiness.expand_as(content_sim)
        return torch.stack(
            [content_sim, concept, prereq_violation, difficulty, difficulty_gap, redundancy, hist_norm],
            dim=2,
        )


def evaluate_full(
    model: CourseAwareScorer,
    builder: FeatureBuilder,
    loader: DataLoader,
    device: torch.device,
    eval_type: str,
    user_seen_items: Dict[int, set],
    export_item_metrics_path: str | None = None,
) -> tuple[dict[str, float] | None, int, int]:
    accum = {key: 0.0 for key in METRIC_KEYS}
    item_accum = {key: {} for key in METRIC_KEYS}
    item_counts: dict[int, int] = {}
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, pop in loader:
            if eval_type == "cold":
                mask = pop < builder.cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= builder.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)
            if int(mask.sum()) < 1:
                continue
            users = batch["u"][mask].to(device)
            items = batch["i"][mask].to(device)
            feats = builder.full_features(users)
            scores = model(feats.view(-1, feats.size(-1))).view(users.size(0), builder.cfg.n_items)
            row_idx = torch.arange(users.size(0), device=device)
            target_scores = scores[row_idx, items].clone()
            seen_rows = builder.seen.index_select(0, users)
            scores = scores.masked_fill(seen_rows, -1e9)
            scores[row_idx, items] = target_scores
            values = compute_ranking_metric_values(scores, items, k_list=K_LIST)
            item_ids = [int(x) for x in items.detach().cpu().tolist()]
            for key, val in values.items():
                accum[key] += float(val.sum().detach().cpu().item())
            for row, item_id in enumerate(item_ids):
                item_counts[item_id] = item_counts.get(item_id, 0) + 1
                for key, val in values.items():
                    per_item = item_accum[key]
                    per_item[item_id] = per_item.get(item_id, 0.0) + float(val[row].detach().cpu().item())
            total += int(users.size(0))
    if total < 1:
        return None, 0, 0
    macro = {
        key: sum(per_item.get(item_id, 0.0) / count for item_id, count in item_counts.items())
        / max(1, len(item_counts))
        for key, per_item in item_accum.items()
    }
    if export_item_metrics_path:
        rows = []
        for item_id in sorted(item_counts):
            row = {"item_id": int(item_id), "count": int(item_counts[item_id])}
            for key, per_item in item_accum.items():
                row[key] = float(per_item.get(item_id, 0.0) / max(1, item_counts[item_id]))
            rows.append(row)
        pd.DataFrame(rows).to_csv(export_item_metrics_path, index=False)
    return macro, len(item_counts), total


def train_one() -> None:
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = static_split_df(df, seed=cfg.static_seed, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)
    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)

    builder = FeatureBuilder(cfg, train_df, df, content_emb, device)
    model = CourseAwareScorer(
        n_features=7,
        hidden_dim=cfg.hidden_dim,
        content_prior_weight=cfg.content_prior_weight,
        correction_scale=cfg.correction_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(InteractionDataset(train_df), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_interactions)
    val_loader = DataLoader(InteractionDataset(val_df), batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_interactions)
    test_loader = DataLoader(InteractionDataset(test_df), batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_interactions)

    best_state = None
    best_epoch = 0
    best_val = -1.0
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        losses = []
        for batch, _ in train_loader:
            users = batch["u"].to(device)
            pos = batch["i"].to(device)
            neg = builder.sample_negatives(users, pos)
            pos_score = model(builder.pair_features(users, pos))
            neg_score = model(builder.pair_features(users, neg))
            loss = F.softplus(neg_score - pos_score).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        do_eval = epoch == 1 or epoch == cfg.n_epochs or epoch % cfg.eval_interval == 0
        if do_eval:
            val_macro, _, _ = evaluate_full(model, builder, val_loader, device, "cold", train_seen)
            val_key = val_macro.get("N@10", 0.0) if val_macro else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(f"Epoch [{epoch}/{cfg.n_epochs}] loss={np.mean(losses):.4f} val_cold_item_macro_N@10={val_key:.4f}")
        else:
            print(f"Epoch [{epoch}/{cfg.n_epochs}] loss={np.mean(losses):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    cold_macro, cold_items, cold_cases = evaluate_full(
        model,
        builder,
        test_loader,
        device,
        "cold",
        test_seen,
        export_item_metrics_path=static_result_path("per_item_full_cold_course_aware_mlp_static.csv"),
    )
    hot_macro, hot_items, hot_cases = evaluate_full(
        model,
        builder,
        test_loader,
        device,
        "hot",
        test_seen,
        export_item_metrics_path=static_result_path("per_item_full_hot_course_aware_mlp_static.csv"),
    )
    cold_macro = cold_macro or {}
    hot_macro = hot_macro or {}
    out = {
        "model": "CourseAware-MLP",
        "protocol": "static_item_cold",
        "best_metric": "full_cold_N@10_item_macro",
        "best_epoch": int(best_epoch),
        "best_val_full_cold_n10": float(best_val),
        "full_cold_item_macro": cold_macro,
        "full_hot_item_macro": hot_macro,
        "count_full_cold_item_macro": int(cold_items),
        "count_full_hot_item_macro": int(hot_items),
        "count_full_cold": int(cold_cases),
        "count_full_hot": int(hot_cases),
        "hidden_dim": cfg.hidden_dim,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
        "eval_interval": cfg.eval_interval,
        "content_prior_weight": cfg.content_prior_weight,
        "correction_scale": cfg.correction_scale,
        "static_seed": cfg.static_seed,
        "features": [
            "content_similarity",
            "concept_match",
            "prerequisite_violation",
            "difficulty",
            "difficulty_gap",
            "redundancy",
            "history_length",
        ],
        "course_artifact_stats": builder.artifact_stats,
    }
    pd.DataFrame([out]).to_json(static_result_path("course_aware_mlp_static_result.json"), orient="records")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    train_one()
