"""Isolated V3.3 rank-distilled counterfactual USIM route.

The module deliberately builds on the clean V3.2 split/evaluation contracts.
Its additions are limited to train-only teacher rank panels, rank-calibrated
generator supervision, and a PPO reward defined by incremental panel KL gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, FrozenSet, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

import ckg_rl_usim_v32_clean as clean


_HARD_RETRIEVAL_ITEM_CHUNK = 32


@dataclass(frozen=True)
class RankDistillRunConfig(clean.CleanRunConfig):
    """V3.3 settings layered over the clean V3.2 protocol."""

    rank_temperature: float = 0.20
    generator_rank_weight: float = 1.0
    panel_size: int = 48
    panel_positive_count: int = 8
    panel_hard_count: int = 16
    course_reward_weight: float = 0.25
    delta_weight: float = 0.02

    @classmethod
    def for_seed(cls, seed: int) -> "RankDistillRunConfig":
        base_config = clean.CleanRunConfig.for_seed(int(seed))
        fields = dict(base_config.__dict__)
        fields["output_dir"] = f"outputs/ckg_rl_usim_v33_rank_distill/seed{int(seed)}"
        fields["checkpoint_dir"] = f"checkpoints/ckg_rl_usim_v33_rank_distill/seed{int(seed)}"
        return cls(**fields)


def _validate_rank_distill_config(config: RankDistillRunConfig) -> None:
    clean._validate_run_config(config)
    if float(config.rank_temperature) <= 0.0:
        raise ValueError("rank_temperature must be positive")
    if float(config.generator_rank_weight) < 0.0:
        raise ValueError("generator_rank_weight must be non-negative")
    if int(config.panel_size) < 1 or int(config.panel_positive_count) < 0 or int(config.panel_hard_count) < 0:
        raise ValueError("panel size/source counts are invalid")
    if float(config.course_reward_weight) < 0.0 or float(config.delta_weight) < 0.0:
        raise ValueError("rank reward weights must be non-negative")


@dataclass(frozen=True)
class RankPanels:
    """Fixed, deterministic teacher-user panels indexed by warm item ID."""

    item_ids: tuple[int, ...]
    panel_ids: torch.Tensor
    positive_counts: tuple[int, ...]
    hard_counts: tuple[int, ...]
    panel_size: int
    seed: int

    def __post_init__(self) -> None:
        if tuple(sorted(int(item_id) for item_id in self.item_ids)) != self.item_ids:
            raise ValueError("rank panel item_ids must be sorted and unique")
        panel_ids = torch.as_tensor(self.panel_ids, dtype=torch.long).cpu()
        if panel_ids.ndim != 2 or panel_ids.size(0) != len(self.item_ids):
            raise ValueError("panel_ids must be [item_count, panel_size]")
        if panel_ids.size(1) != int(self.panel_size) or int(self.panel_size) < 1:
            raise ValueError("panel_size must match a positive panel_ids width")
        if len(self.positive_counts) != len(self.item_ids) or len(self.hard_counts) != len(self.item_ids):
            raise ValueError("rank panel source counts must align with item_ids")
        for row in panel_ids.tolist():
            if len(set(int(user_id) for user_id in row)) != len(row):
                raise ValueError("each rank panel must contain unique user IDs")
        object.__setattr__(self, "panel_ids", panel_ids)

    def panel_for(self, item_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        positions = {int(item_id): index for index, item_id in enumerate(self.item_ids)}
        requested = [int(item_id) for item_id in torch.as_tensor(item_ids, dtype=torch.long).view(-1).tolist()]
        missing = [item_id for item_id in requested if item_id not in positions]
        if missing:
            raise KeyError(f"rank panel is missing item IDs: {sorted(set(missing))}")
        indices = torch.tensor([positions[item_id] for item_id in requested], dtype=torch.long)
        return self.panel_ids.index_select(0, indices).to(device=device)

    def digest(self, item_ids: Sequence[int] | None = None) -> str:
        if item_ids is None:
            selected = self.panel_ids
            selected_ids = self.item_ids
        else:
            positions = {int(item_id): index for index, item_id in enumerate(self.item_ids)}
            selected_ids = tuple(sorted(int(item_id) for item_id in item_ids))
            selected = self.panel_ids.index_select(
                0, torch.tensor([positions[item_id] for item_id in selected_ids], dtype=torch.long)
            )
        payload = ("|".join(str(item_id) for item_id in selected_ids)).encode("ascii")
        return hashlib.sha256(payload + selected.contiguous().numpy().tobytes()).hexdigest()


def _stable_rank(seed: int, item_id: int, user_id: int, role: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{int(item_id)}|{int(user_id)}|{role}".encode("ascii")).hexdigest()


def _deterministic_fill(
    *,
    n_users: int,
    needed: int,
    excluded: set[int],
    seed: int,
    item_id: int,
    role: str,
) -> list[int]:
    """Choose unseen user IDs without touching process-global RNG state."""
    if needed <= 0:
        return []
    rng = random.Random(f"{int(seed)}|{int(item_id)}|{role}")
    selected: list[int] = []
    attempts = 0
    max_attempts = max(64, int(needed) * 64)
    while len(selected) < needed and attempts < max_attempts:
        candidate = int(rng.randrange(int(n_users)))
        attempts += 1
        if candidate not in excluded:
            excluded.add(candidate)
            selected.append(candidate)
    if len(selected) < needed:
        start = int(hashlib.sha256(f"{int(seed)}|{int(item_id)}|{role}|fill".encode("ascii")).hexdigest(), 16)
        for offset in range(int(n_users)):
            candidate = (start + offset) % int(n_users)
            if candidate not in excluded:
                excluded.add(candidate)
                selected.append(candidate)
                if len(selected) == needed:
                    break
    if len(selected) != needed:
        raise RuntimeError("unable to build a unique fixed-width rank panel")
    return selected


@torch.no_grad()
def _teacher_hard_users(
    teacher: clean.CleanTeacher,
    *,
    item_ids: Sequence[int],
    positive_users: Mapping[int, torch.Tensor],
    hard_count: int,
) -> dict[int, list[int]]:
    """Batched full-teacher top-k without a per-item CPU user-bank scan."""
    requested = tuple(int(item_id) for item_id in item_ids)
    if int(hard_count) <= 0:
        return {item_id: [] for item_id in requested}
    user_vectors = teacher.user_vectors().detach()
    item_vectors = teacher.item_vectors().detach()
    n_users = int(user_vectors.size(0))
    result: dict[int, list[int]] = {}
    for start in range(0, len(requested), _HARD_RETRIEVAL_ITEM_CHUNK):
        batch_item_ids = requested[start:start + _HARD_RETRIEVAL_ITEM_CHUNK]
        ids = torch.tensor(batch_item_ids, dtype=torch.long, device=user_vectors.device)
        scores = torch.matmul(user_vectors, item_vectors.index_select(0, ids).t())
        for column, item_id in enumerate(batch_item_ids):
            positive_ids = torch.as_tensor(
                positive_users.get(int(item_id), torch.empty(0, dtype=torch.long)),
                dtype=torch.long,
                device=scores.device,
            ).view(-1)
            if positive_ids.numel() and (
                int(positive_ids.min().item()) < 0 or int(positive_ids.max().item()) >= n_users
            ):
                raise ValueError("H_train has a user outside the teacher user table")
            available = n_users - int(torch.unique(positive_ids).numel())
            count = min(int(hard_count), max(0, available))
            if count == 0:
                result[int(item_id)] = []
                continue
            column_scores = scores[:, column]
            if positive_ids.numel():
                column_scores = column_scores.clone()
                column_scores.index_fill_(0, positive_ids.unique(), -torch.inf)
            result[int(item_id)] = [
                int(user_id) for user_id in torch.topk(column_scores, k=count).indices.detach().cpu().tolist()
            ]
    return result


@torch.no_grad()
def build_rank_panels(
    teacher: clean.CleanTeacher,
    h_train: pd.DataFrame,
    *,
    item_ids: FrozenSet[int] | set[int] | Sequence[int],
    seed: int,
    panel_size: int,
    positive_count: int,
    hard_count: int,
) -> RankPanels:
    """Build fixed warm-item panels from a teacher and H_train only."""
    required = {"u_idx", "i_idx"}
    if not required.issubset(h_train.columns):
        raise ValueError("H_train must contain u_idx and i_idx")
    requested = tuple(sorted({int(item_id) for item_id in item_ids}))
    if not requested:
        raise ValueError("rank panels require at least one H_train item")
    observed_items = {int(item_id) for item_id in h_train["i_idx"].astype(int).unique()}
    if not set(requested).issubset(observed_items):
        raise ValueError("rank panels may be built only for items observed in H_train")
    if int(panel_size) < 1 or int(positive_count) < 0 or int(hard_count) < 0:
        raise ValueError("panel_size must be positive and source counts non-negative")

    n_users = int(teacher.user_emb.num_embeddings)
    width = min(int(panel_size), n_users)
    if width < 1:
        raise ValueError("teacher must expose at least one user")
    positives = clean._positive_users_by_item(h_train)
    hard_by_item = _teacher_hard_users(
        teacher,
        item_ids=requested,
        positive_users=positives,
        hard_count=min(int(hard_count), width),
    )
    rows: list[list[int]] = []
    positive_counts: list[int] = []
    hard_counts: list[int] = []

    for item_id in requested:
        observed_positive = sorted(int(user_id) for user_id in positives.get(int(item_id), torch.empty(0)).tolist())
        ordered_positive = sorted(
            observed_positive,
            key=lambda user_id: _stable_rank(seed, item_id, user_id, "positive"),
        )
        selected = ordered_positive[: min(int(positive_count), width)]
        positive_counts.append(len(selected))
        all_positive = set(observed_positive)
        wanted_hard = min(int(hard_count), width - len(selected))
        hard_users = [
            int(user_id)
            for user_id in hard_by_item[int(item_id)][:wanted_hard]
            if int(user_id) not in selected
        ]
        selected.extend(hard_users)
        hard_counts.append(len(hard_users))

        excluded = set(selected) | all_positive
        negative_capacity = max(0, n_users - len(excluded))
        selected.extend(
            _deterministic_fill(
                n_users=n_users,
                needed=min(width - len(selected), negative_capacity),
                excluded=excluded,
                seed=seed,
                item_id=item_id,
                role="negative",
            )
        )
        if len(selected) < width:
            selected.extend(
                _deterministic_fill(
                    n_users=n_users,
                    needed=width - len(selected),
                    excluded=set(selected),
                    seed=seed,
                    item_id=item_id,
                    role="fallback",
                )
            )
        rows.append(selected)

    return RankPanels(
        item_ids=requested,
        panel_ids=torch.tensor(rows, dtype=torch.long),
        positive_counts=tuple(positive_counts),
        hard_counts=tuple(hard_counts),
        panel_size=width,
        seed=int(seed),
    )


def panel_distribution(
    user_vectors: torch.Tensor,
    state: torch.Tensor,
    panel_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return the normalized teacher-user affinity distribution on each panel."""
    if float(temperature) <= 0.0:
        raise ValueError("rank temperature must be positive")
    users = torch.as_tensor(user_vectors, dtype=state.dtype, device=state.device)
    ids = torch.as_tensor(panel_ids, dtype=torch.long, device=state.device)
    if users.ndim != 2 or state.ndim != 2 or ids.ndim != 2:
        raise ValueError("user_vectors/state/panel_ids must be rank 2")
    if state.size(0) != ids.size(0) or state.size(1) != users.size(1):
        raise ValueError("rank-panel dimensions must match state and user vectors")
    if ids.numel() < 1 or int(ids.min().item()) < 0 or int(ids.max().item()) >= users.size(0):
        raise ValueError("panel_ids contain an out-of-range user")
    selected_users = users.index_select(0, ids.reshape(-1)).view(ids.size(0), ids.size(1), users.size(1))
    scores = torch.einsum("bkd,bd->bk", F.normalize(selected_users, dim=2), F.normalize(state, dim=1))
    return F.softmax(scores / float(temperature), dim=1)


def rank_kl(target_q: torch.Tensor, state_q: torch.Tensor) -> torch.Tensor:
    """Per-row KL(target distribution || state distribution)."""
    target = torch.as_tensor(target_q)
    state = torch.as_tensor(state_q, dtype=target.dtype, device=target.device)
    if target.shape != state.shape or target.ndim != 2:
        raise ValueError("target_q and state_q must share [batch, panel] shape")
    epsilon = torch.finfo(target.dtype).eps
    return (target * (target.clamp_min(epsilon).log() - state.clamp_min(epsilon).log())).sum(dim=1, keepdim=True)


def incremental_rank_gain(target_q: torch.Tensor, before_q: torch.Tensor, after_q: torch.Tensor) -> torch.Tensor:
    return rank_kl(target_q, before_q) - rank_kl(target_q, after_q)


def generator_rank_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    user_vectors: torch.Tensor,
    panel_ids: torch.Tensor,
    *,
    temperature: float,
    rank_weight: float,
) -> torch.Tensor:
    if float(rank_weight) < 0.0:
        raise ValueError("rank_weight must be non-negative")
    vector_loss = clean._generator_objective(prediction, target)
    with torch.no_grad():
        target_q = panel_distribution(user_vectors, target.detach(), panel_ids, temperature=temperature)
    prediction_q = panel_distribution(user_vectors, prediction, panel_ids, temperature=temperature)
    return vector_loss + float(rank_weight) * rank_kl(target_q, prediction_q).mean()


def _generator_rank_loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    user_vectors: torch.Tensor,
    panel_ids: torch.Tensor,
    *,
    temperature: float,
    rank_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vector_loss = clean._generator_objective(prediction, target)
    with torch.no_grad():
        target_q = panel_distribution(user_vectors, target.detach(), panel_ids, temperature=temperature)
    prediction_q = panel_distribution(user_vectors, prediction, panel_ids, temperature=temperature)
    rank_loss = rank_kl(target_q, prediction_q).mean()
    return vector_loss + float(rank_weight) * rank_loss, vector_loss, rank_loss


def train_rank_calibrated_generator(
    teacher: clean.CleanTeacher,
    content: torch.Tensor,
    *,
    generator_item_ids: FrozenSet[int],
    rank_panels: RankPanels,
    config: RankDistillRunConfig,
) -> tuple[clean.ContentGenerator, dict[str, Any]]:
    """Fit the content generator only on H_G item labels and rank panels."""
    _validate_rank_distill_config(config)
    device = teacher.item_emb.weight.device
    train_ids, validation_ids = clean._generator_train_validation_ids(
        generator_item_ids,
        seed=config.seed,
        validation_fraction=config.generator_val_fraction,
    )
    all_ids = set(train_ids) | set(validation_ids)
    if all_ids != {int(item_id) for item_id in generator_item_ids}:
        raise RuntimeError("generator rank training must use exactly the H_G item set")
    # This lookup deliberately fails before training if any H_G label lacks a
    # panel, rather than silently falling back to a non-comparable objective.
    rank_panels.panel_for(torch.tensor(sorted(all_ids)), device=device)

    generator = clean.ContentGenerator(
        content_dim=int(content.size(1)), emb_dim=config.emb_dim, hidden_dim=config.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(config.generator_lr))
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    with torch.no_grad():
        teacher_items = teacher.item_vectors().detach()
        teacher_users = teacher.user_vectors().detach()
    best_state = clean._copy_state_dict(generator)
    best_key = (float("inf"), float("inf"))
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config.generator_epochs) + 1):
        generator.train()
        train_losses: list[float] = []
        for batch_ids in clean._item_id_batches(train_ids, config.batch_size, seed=config.seed + epoch):
            ids = batch_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _generator_rank_loss_terms(
                generator(frozen_content.index_select(0, ids)),
                teacher_items.index_select(0, ids),
                teacher_users,
                rank_panels.panel_for(ids, device=device),
                temperature=config.rank_temperature,
                rank_weight=config.generator_rank_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        generator.eval()
        validation_source = validation_ids if validation_ids else train_ids
        with torch.no_grad():
            ids = torch.tensor(validation_source, dtype=torch.long, device=device)
            validation_total, validation_vector, validation_rank = _generator_rank_loss_terms(
                generator(frozen_content.index_select(0, ids)),
                teacher_items.index_select(0, ids),
                teacher_users,
                rank_panels.panel_for(ids, device=device),
                temperature=config.rank_temperature,
                rank_weight=config.generator_rank_weight,
            )
        row = {
            "epoch": float(epoch),
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "validation_loss": float(validation_total.item()),
            "validation_vector_loss": float(validation_vector.item()),
            "validation_rank_kl": float(validation_rank.item()),
        }
        history.append(row)
        key = (row["validation_rank_kl"], row["validation_vector_loss"])
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = clean._copy_state_dict(generator)

    generator.load_state_dict(best_state)
    generator.eval()
    return generator, {
        "selected_epoch": int(best_epoch),
        "selection_rank_kl": float(best_key[0]),
        "selection_vector_loss": float(best_key[1]),
        "train_item_ids": list(train_ids),
        "validation_item_ids": list(validation_ids),
        "train_panel_sha256": rank_panels.digest(train_ids),
        "validation_panel_sha256": rank_panels.digest(validation_ids or train_ids),
        "history": history,
        "protocol": "h_g_only_vector_and_rank_distillation",
    }


class RankDistilledUSIMEngine(clean.CleanUSIMEngine):
    """V3.2 legal rollout whose training reward is panel ranking improvement."""

    def __init__(
        self,
        *,
        rank_panels: RankPanels,
        rank_temperature: float,
        course_reward_weight: float,
        delta_weight: float,
        **kwargs: object,
    ) -> None:
        if float(rank_temperature) <= 0.0:
            raise ValueError("rank_temperature must be positive")
        if float(course_reward_weight) < 0.0 or float(delta_weight) < 0.0:
            raise ValueError("course_reward_weight and delta_weight must be non-negative")
        super().__init__(**kwargs)
        self.rank_panels = rank_panels
        self.rank_temperature = float(rank_temperature)
        self.course_reward_weight = float(course_reward_weight)
        self.delta_weight = float(delta_weight)

    def _panel_ids(self, item_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        return self.rank_panels.panel_for(item_ids, device=device)

    def rank_diagnostics(
        self,
        *,
        before_state: torch.Tensor,
        after_state: torch.Tensor,
        target_emb: torch.Tensor,
        user_bank: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        panel_ids = self._panel_ids(item_ids, device=before_state.device)
        target_q = panel_distribution(user_bank, target_emb.detach(), panel_ids, temperature=self.rank_temperature)
        before_q = panel_distribution(user_bank, before_state, panel_ids, temperature=self.rank_temperature)
        after_q = panel_distribution(user_bank, after_state, panel_ids, temperature=self.rank_temperature)
        initial_kl = rank_kl(target_q, before_q)
        final_kl = rank_kl(target_q, after_q)
        return {
            "initial_kl": initial_kl,
            "final_kl": final_kl,
            "rank_gain": initial_kl - final_kl,
        }

    def _training_reward(
        self,
        before_state: torch.Tensor,
        after_state: torch.Tensor,
        target_emb: torch.Tensor,
        positive_user_ids: Sequence[torch.Tensor],
        user_bank: torch.Tensor,
        active_user: torch.Tensor,
        selected_user_ids: torch.Tensor,
        item_ids: torch.Tensor | None,
        user_history: Mapping[int, set[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del positive_user_ids
        if item_ids is None:
            raise ValueError("rank-distilled training reward requires item IDs")
        diagnostics = self.rank_diagnostics(
            before_state=before_state,
            after_state=after_state,
            target_emb=target_emb,
            user_bank=user_bank,
            item_ids=item_ids,
        )
        rank_gain = diagnostics["rank_gain"]
        raw_course_reward = self._course_reward(selected_user_ids, item_ids, user_history)
        weighted_course_reward = self.course_reward_weight * raw_course_reward
        delta_penalty = self.delta_weight * (after_state - before_state).norm(dim=1, keepdim=True)
        active = active_user.float().view(-1, 1)
        reward = (rank_gain + weighted_course_reward - self.step_penalty - delta_penalty) * active
        embedding_gain = torch.zeros_like(rank_gain)
        return reward, embedding_gain, rank_gain * active, weighted_course_reward * active

    def rollout(
        self,
        initial_state: torch.Tensor,
        *,
        user_bank: torch.Tensor,
        training: bool,
        target_emb: torch.Tensor | None = None,
        positive_user_ids: Sequence[torch.Tensor] | None = None,
        item_ids: torch.Tensor | None = None,
        user_history: Mapping[int, set[int]] | None = None,
    ) -> clean.CleanRolloutResult:
        if training and positive_user_ids is None:
            positive_user_ids = tuple(
                torch.empty(0, dtype=torch.long, device=initial_state.device)
                for _ in range(initial_state.size(0))
            )
        return super().rollout(
            initial_state,
            user_bank=user_bank,
            training=training,
            target_emb=target_emb,
            positive_user_ids=positive_user_ids,
            item_ids=item_ids,
            user_history=user_history,
        )


def create_rank_distilled_engine(
    config: RankDistillRunConfig,
    *,
    rank_panels: RankPanels,
    course_signal: clean.CleanCourseSignal | None = None,
) -> RankDistilledUSIMEngine:
    if config.use_course_signal and course_signal is None:
        raise ValueError("use_course_signal requires a CleanCourseSignal")
    return RankDistilledUSIMEngine(
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        max_steps=config.max_steps,
        candidate_count=config.candidate_count,
        step_size=config.step_size,
        step_penalty=config.step_penalty,
        max_delta=config.max_delta,
        retrieval_chunk=config.retrieval_chunk,
        course_bias_fn=None if course_signal is None else course_signal.candidate_bias,
        course_reward_fn=None if course_signal is None else course_signal.reward,
        rank_panels=rank_panels,
        rank_temperature=config.rank_temperature,
        course_reward_weight=config.course_reward_weight,
        delta_weight=config.delta_weight,
    )


@torch.no_grad()
def _policy_rank_diagnostics(
    teacher: clean.CleanTeacher,
    generator: clean.ContentGenerator,
    engine: RankDistilledUSIMEngine,
    *,
    item_ids: Sequence[int],
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    policy_epoch: int,
    batch_size: int,
) -> dict[str, float]:
    """Measure rollout calibration only on warm pseudo-validation items."""
    ids_all = sorted(int(item_id) for item_id in item_ids)
    if not ids_all:
        raise ValueError("rank diagnostics require at least one pseudo item")
    device = teacher.item_emb.weight.device
    user_bank = teacher.user_vectors().detach()
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    initial_kls: list[torch.Tensor] = []
    final_kls: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []
    for batch_ids in clean._item_id_batches(ids_all, batch_size, seed=0):
        ids = batch_ids.to(device)
        initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
        if clean._policy_mode(policy_epoch) == "identity_generator":
            final = initial
        else:
            rollout = engine.rollout(
                initial,
                user_bank=user_bank,
                training=False,
                item_ids=ids,
                user_history=user_history,
            )
            final = F.normalize(rollout.final_state, dim=1)
        diagnostics = engine.rank_diagnostics(
            before_state=initial,
            after_state=final,
            target_emb=teacher.item_vectors(ids).detach(),
            user_bank=user_bank,
            item_ids=ids,
        )
        initial_kls.append(diagnostics["initial_kl"].detach().cpu())
        final_kls.append(diagnostics["final_kl"].detach().cpu())
        gains.append(diagnostics["rank_gain"].detach().cpu())
    return {
        "p_val_initial_rank_kl": float(torch.cat(initial_kls).mean().item()),
        "p_val_final_rank_kl": float(torch.cat(final_kls).mean().item()),
        "p_val_rank_gain": float(torch.cat(gains).mean().item()),
    }


def _select_rank_policy_row(
    rows: Sequence[dict[str, Any]], *, hot_retention_tolerance: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rank policy selection requires validation rows")
    baseline = rows[0]
    hot_r_floor = float(baseline["hot_r10"]) - float(hot_retention_tolerance)
    hot_n_floor = float(baseline["hot_n10"]) - float(hot_retention_tolerance)
    hot_eligible = [
        row for row in rows
        if float(row["hot_r10"]) >= hot_r_floor and float(row["hot_n10"]) >= hot_n_floor
    ]
    eligible = [
        row for row in hot_eligible
        if int(row["epoch"]) == 0 or float(row["p_val_rank_gain"]) >= -1e-12
    ]
    if not eligible:
        raise RuntimeError("no rank policy epoch preserves hot validation metrics")
    return max(
        eligible,
        key=lambda row: (
            float(row["cold_n10"]),
            float(row["cold_r10"]),
            float(row["overall_n10"]),
            float(row["p_val_rank_gain"]),
            -float(row["p_val_final_rank_kl"]),
            -int(row["epoch"]),
        ),
    )


def train_rank_distilled_policy(
    teacher: clean.CleanTeacher,
    generator: clean.ContentGenerator,
    engine: RankDistilledUSIMEngine,
    views: clean.CleanStageViews,
    *,
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    validation_callback: Callable[[int], dict[str, float | int | str]],
    config: RankDistillRunConfig,
) -> tuple[RankDistilledUSIMEngine, dict[str, Any], list[dict[str, Any]]]:
    """Fit PPO on P_train and select only with P_val/C_val/H_val diagnostics."""
    _validate_rank_distill_config(config)
    device = teacher.item_emb.weight.device
    engine.policy.to(device)
    teacher.eval()
    generator.eval()
    for module in (teacher, generator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    user_bank = teacher.user_vectors().detach()
    train_item_ids = sorted(int(item_id) for item_id in views.policy_train_item_ids)
    validation_item_ids = sorted(int(item_id) for item_id in views.policy_val_item_ids) or train_item_ids
    if not train_item_ids:
        raise ValueError("P_train must contain at least one pseudo-cold item")
    engine.rank_panels.panel_for(torch.tensor(train_item_ids), device=device)
    engine.rank_panels.panel_for(torch.tensor(validation_item_ids), device=device)

    optimizer = torch.optim.Adam(engine.policy.parameters(), lr=float(config.policy_lr))
    ppo = clean.CleanRecPPO(
        engine.policy,
        replay_capacity=config.replay_capacity,
        replay_batch_size=config.replay_batch_size,
        gamma=config.ppo_gamma,
        clip_ratio=config.ppo_clip_ratio,
        value_weight=config.ppo_value_weight,
        terminal_value_weight=config.ppo_terminal_value_weight,
        entropy_weight=config.ppo_entropy_weight,
    )
    state_by_epoch: dict[int, dict[str, torch.Tensor]] = {0: clean._copy_state_dict(engine.policy)}
    rows: list[dict[str, Any]] = []

    def collect_validation(epoch: int, train_loss: float, train_rank_gain: float) -> None:
        engine.policy.eval()
        outer = validation_callback(epoch)
        if outer.get("policy_mode") != clean._policy_mode(epoch):
            raise RuntimeError("rank policy validation mode does not match policy epoch")
        diagnostics = _policy_rank_diagnostics(
            teacher,
            generator,
            engine,
            item_ids=validation_item_ids,
            content=frozen_content,
            user_history=user_history,
            policy_epoch=epoch,
            batch_size=config.batch_size,
        )
        rows.append({
            "epoch": int(epoch),
            "policy_mode": clean._policy_mode(epoch),
            "train_loss": float(train_loss),
            "train_rank_gain": float(train_rank_gain),
            **diagnostics,
            **{name: float(value) for name, value in outer.items() if name != "policy_mode"},
        })

    collect_validation(0, 0.0, 0.0)
    for epoch in range(1, int(config.policy_epochs) + 1):
        engine.policy.train()
        losses: list[float] = []
        rank_gains: list[float] = []
        for batch_ids in clean._item_id_batches(train_item_ids, config.batch_size, seed=config.seed + epoch):
            ids = batch_ids.to(device)
            with torch.no_grad():
                initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
                target = teacher.item_vectors(ids).detach()
            rollout = engine.rollout(
                initial,
                user_bank=user_bank,
                training=True,
                target_emb=target,
                item_ids=ids,
                user_history=user_history,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = ppo.loss(rollout.trajectory, user_bank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.policy.parameters(), max_norm=5.0)
            optimizer.step()
            ppo.sync_target()
            losses.append(float(loss.detach().item()))
            rank_gains.append(float(rollout.stats["recommendation_reward"]))
        state_by_epoch[epoch] = clean._copy_state_dict(engine.policy)
        collect_validation(
            epoch,
            float(sum(losses) / max(1, len(losses))),
            float(sum(rank_gains) / max(1, len(rank_gains))),
        )

    selected = _select_rank_policy_row(rows, hot_retention_tolerance=config.hot_retention_tolerance)
    engine.policy.load_state_dict(state_by_epoch[int(selected["epoch"])])
    engine.policy.eval()
    return engine, selected, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _config_payload(config: RankDistillRunConfig) -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in config.__dict__.items()
    }


def _write_rank_panel_manifest(
    output_dir: Path,
    panels: RankPanels,
    *,
    generator_item_ids: FrozenSet[int],
    policy_train_item_ids: FrozenSet[int],
    policy_val_item_ids: FrozenSet[int],
) -> None:
    clean._write_json(output_dir / "rank_panel_manifest.json", {
        "source": "H_train_only_frozen_teacher",
        "seed": int(panels.seed),
        "panel_size": int(panels.panel_size),
        "item_count": int(len(panels.item_ids)),
        "all_panel_sha256": panels.digest(),
        "generator_panel_sha256": panels.digest(generator_item_ids),
        "policy_train_panel_sha256": panels.digest(policy_train_item_ids),
        "policy_val_panel_sha256": panels.digest(policy_val_item_ids),
        "positive_count_sum": int(sum(panels.positive_counts)),
        "hard_count_sum": int(sum(panels.hard_counts)),
    })


def _write_v33_manifest(
    output_dir: Path,
    config: RankDistillRunConfig,
    partitions: clean.CleanPartitions,
    *,
    stage_hashes: Mapping[str, str],
    selected_policy: Mapping[str, Any],
    course_stats: Mapping[str, Any],
) -> None:
    clean._write_json(output_dir / "v33_manifest.json", {
        "route": "ckg_rl_usim_v33_rank_distill",
        "seed": int(config.seed),
        "legacy_warm_checkpoint": None,
        "random_id_dropout": False,
        "main_candidate_mode": "legal_state_retrieval",
        "inference_oracle_access": False,
        "teacher_protocol": "H_train_only_selected_on_H_val",
        "generator_protocol": "H_G_only_vector_and_rank_distillation",
        "policy_protocol": "P_train_rank_gain_reward_legal_candidates",
        "policy_rank_gain_gate": "ppo_epoch_requires_nonnegative_p_val_rank_gain",
        "legacy_embedding_reward": False,
        "legacy_positive_score_reward": False,
        "test_loaded_after_policy_selection": True,
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": clean._policy_mode(int(selected_policy["epoch"])),
        "course_signal": dict(course_stats),
        "stage_hashes": dict(stage_hashes),
        "partitions": clean._partition_manifest_payload(partitions),
        "config": _config_payload(config),
    })


def run_rank_distill_pipeline(config: RankDistillRunConfig) -> dict[str, Any]:
    """Run V3.3; outer test is loaded exactly after policy selection."""
    clean._validate_clean_route_environment()
    _validate_rank_distill_config(config)
    output_dir = clean._resolve_path(config.output_dir)
    checkpoint_dir = clean._resolve_path(config.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty V3.3 output directory: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty V3.3 checkpoint directory: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    clean.setup_seed(config.seed)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    empty_test = val_df.iloc[0:0].copy()
    partitions = clean.build_clean_partitions(
        train_df,
        val_df,
        empty_test,
        n_items=int(content.size(0)),
        seed=config.seed,
        pseudo_ratio=config.pseudo_ratio,
        pseudo_val_fraction=config.pseudo_val_fraction,
        min_popularity=config.pseudo_min_popularity,
    )
    views = clean.build_stage_views(partitions)
    user_history = clean.build_user_seen(partitions.h_train)
    teacher, teacher_meta = clean.train_clean_teacher(
        views,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        user_history=user_history,
        config=config,
    )
    teacher_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "teacher.pt", stage="teacher", module=teacher, metadata=teacher_meta
    )

    panel_item_ids = views.generator_item_ids | views.policy_train_item_ids | views.policy_val_item_ids
    rank_panels = build_rank_panels(
        teacher,
        views.teacher_train,
        item_ids=panel_item_ids,
        seed=config.seed,
        panel_size=config.panel_size,
        positive_count=config.panel_positive_count,
        hard_count=config.panel_hard_count,
    )
    _write_rank_panel_manifest(
        output_dir,
        rank_panels,
        generator_item_ids=views.generator_item_ids,
        policy_train_item_ids=views.policy_train_item_ids,
        policy_val_item_ids=views.policy_val_item_ids,
    )
    generator, generator_meta = train_rank_calibrated_generator(
        teacher,
        content,
        generator_item_ids=views.generator_item_ids,
        rank_panels=rank_panels,
        config=config,
    )
    generator_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "generator.pt", stage="generator", module=generator, metadata=generator_meta
    )
    _write_csv(
        output_dir / "generator_rank_epochs.csv",
        generator_meta["history"],
        ("epoch", "train_loss", "validation_loss", "validation_vector_loss", "validation_rank_kl"),
    )

    course_signal, course_stats = clean.build_clean_course_signal(
        partitions.h_train,
        n_items=int(content.size(0)),
        config=config,
    )
    engine = create_rank_distilled_engine(config, rank_panels=rank_panels, course_signal=course_signal)
    warm_item_ids = partitions.g_item_ids | partitions.p_train_item_ids | partitions.p_val_item_ids

    def validation_callback(epoch: int) -> dict[str, float | int | str]:
        return clean.evaluate_clean_route(
            teacher,
            generator,
            engine,
            hot_frame=partitions.h_val,
            cold_frame=partitions.c_val,
            warm_item_ids=warm_item_ids,
            content=content,
            user_history=user_history,
            config=config,
            policy_epoch=epoch,
            export_dir=output_dir,
            export_prefix=f"val_epoch_{int(epoch):03d}_",
        )

    engine, selected_policy, policy_rows = train_rank_distilled_policy(
        teacher,
        generator,
        engine,
        views,
        content=content,
        user_history=user_history,
        validation_callback=validation_callback,
        config=config,
    )
    policy_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "policy.pt",
        stage="policy",
        module=engine.policy,
        metadata={"selected": selected_policy, "validation_rows": policy_rows},
    )
    _write_csv(
        output_dir / "policy_rank_epochs.csv",
        policy_rows,
        (
            "epoch", "policy_mode", "train_loss", "train_rank_gain", "p_val_initial_rank_kl",
            "p_val_final_rank_kl", "p_val_rank_gain", "cold_r10", "cold_n10", "hot_r10", "hot_n10",
            "overall_r10", "overall_n10", "cold_item_count", "hot_item_count",
        ),
    )
    validation = dict(next(row for row in policy_rows if int(row["epoch"]) == int(selected_policy["epoch"])))

    # This is intentionally the first test-file read in the route.
    test_df = clean.load_clean_test_inputs(config, partitions.h_train)
    partitions = clean.attach_clean_test_rows(partitions, test_df)
    test_metrics = clean.evaluate_clean_route(
        teacher,
        generator,
        engine,
        hot_frame=partitions.h_test,
        cold_frame=partitions.c_test,
        warm_item_ids=warm_item_ids,
        content=content,
        user_history=user_history,
        config=config,
        policy_epoch=int(selected_policy["epoch"]),
        export_dir=output_dir,
        export_prefix="test_",
    )
    clean._write_json(output_dir / "clean_partition.json", clean._partition_manifest_payload(partitions))
    clean._write_json(output_dir / "final_metrics.json", {
        "validation": validation,
        "test": test_metrics,
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": clean._policy_mode(int(selected_policy["epoch"])),
    })
    _write_v33_manifest(
        output_dir,
        config,
        partitions,
        stage_hashes={"teacher": teacher_hash, "generator": generator_hash, "policy": policy_hash},
        selected_policy=selected_policy,
        course_stats=course_stats,
    )
    return {
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": clean._policy_mode(int(selected_policy["epoch"])),
        "validation": validation,
        "test": test_metrics,
        "output_dir": str(output_dir),
    }


def _parse_rank_distill_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated CKG-RL V3.3 rank-distilled USIM")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data-dir")
    parser.add_argument("--split-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--course-relation-dir")
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--generator-epochs", type=int)
    parser.add_argument("--policy-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--emb-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--pseudo-ratio", type=float)
    parser.add_argument("--pseudo-val-fraction", type=float)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--retrieval-chunk", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rank-temperature", type=float)
    parser.add_argument("--generator-rank-weight", type=float)
    parser.add_argument("--panel-size", type=int)
    parser.add_argument("--panel-positive-count", type=int)
    parser.add_argument("--panel-hard-count", type=int)
    parser.add_argument("--course-reward-weight", type=float)
    parser.add_argument("--delta-weight", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-course-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _rank_config_from_args(args: argparse.Namespace) -> RankDistillRunConfig:
    config = RankDistillRunConfig.for_seed(int(args.seed))
    replacements: dict[str, Any] = {}
    for field, value in {
        "data_dir": args.data_dir,
        "split_dir": args.split_dir,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "course_relation_dir": args.course_relation_dir,
        "teacher_epochs": args.teacher_epochs,
        "generator_epochs": args.generator_epochs,
        "policy_epochs": args.policy_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "pseudo_ratio": args.pseudo_ratio,
        "pseudo_val_fraction": args.pseudo_val_fraction,
        "candidate_count": args.candidate_count,
        "retrieval_chunk": args.retrieval_chunk,
        "max_steps": args.max_steps,
        "rank_temperature": args.rank_temperature,
        "generator_rank_weight": args.generator_rank_weight,
        "panel_size": args.panel_size,
        "panel_positive_count": args.panel_positive_count,
        "panel_hard_count": args.panel_hard_count,
        "course_reward_weight": args.course_reward_weight,
        "delta_weight": args.delta_weight,
        "device": args.device,
    }.items():
        if value is not None:
            replacements[field] = value
    if args.use_course_signal:
        replacements["use_course_signal"] = True
    config = replace(config, **replacements)
    if args.smoke:
        config = replace(
            config,
            teacher_epochs=min(1, int(config.teacher_epochs)),
            generator_epochs=min(1, int(config.generator_epochs)),
            policy_epochs=min(1, int(config.policy_epochs)),
        )
    return config


def _dry_run_rank_distill_pipeline(config: RankDistillRunConfig) -> dict[str, Any]:
    clean._validate_clean_route_environment()
    _validate_rank_distill_config(config)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    partitions = clean.build_clean_partitions(
        train_df,
        val_df,
        val_df.iloc[0:0].copy(),
        n_items=int(content.size(0)),
        seed=config.seed,
        pseudo_ratio=config.pseudo_ratio,
        pseudo_val_fraction=config.pseudo_val_fraction,
        min_popularity=config.pseudo_min_popularity,
    )
    return {
        "status": "dry_run_ok",
        "route": "ckg_rl_usim_v33_rank_distill",
        "seed": int(config.seed),
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "h_train_rows": int(len(partitions.h_train)),
        "h_val_rows": int(len(partitions.h_val)),
        "c_val_rows": int(len(partitions.c_val)),
        "g_item_count": int(len(partitions.g_item_ids)),
        "p_train_item_count": int(len(partitions.p_train_item_ids)),
        "p_val_item_count": int(len(partitions.p_val_item_ids)),
        "panel_size": int(config.panel_size),
        "course_signal": bool(config.use_course_signal),
        "output_dir": str(clean._resolve_path(config.output_dir)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_rank_distill_args(argv)
    config = _rank_config_from_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run_rank_distill_pipeline(config), ensure_ascii=True, sort_keys=True))
        return 0
    result = run_rank_distill_pipeline(config)
    if args.smoke:
        output_dir = Path(result["output_dir"])
        clean._write_json(output_dir / "smoke_report.json", {
            "status": "completed",
            "route": "ckg_rl_usim_v33_rank_distill",
            "selected_policy_epoch": int(result["selected_policy_epoch"]),
            "target_free_inference": True,
            "test_selection_used": False,
        })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
