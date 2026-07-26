"""Isolated clean T -> G -> V3.2 pipeline primitives.

This module intentionally does not import or resume the historical V2/V3
routes.  The first layer locks the outer strict-cold split and derives all
student-stage views from H_train alone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, FrozenSet, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from ckg_rl_clean_method import CleanMethodConfig, build_clean_method_manifest
from fast3_delta.pseudocold import build_pseudocold_plan
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions, setup_seed
from hin_eval_common import evaluate_embedding_ranker


_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CleanPartitions:
    """Outer strict-cold rows and predeclared warm pseudo-item partitions."""

    h_train: pd.DataFrame
    h_val: pd.DataFrame
    h_test: pd.DataFrame
    c_val: pd.DataFrame
    c_test: pd.DataFrame
    g_item_ids: FrozenSet[int]
    p_train_item_ids: FrozenSet[int]
    p_val_item_ids: FrozenSet[int]


@dataclass(frozen=True)
class CleanStageViews:
    """Exactly the rows and item IDs allowed at each clean training stage."""

    teacher_train: pd.DataFrame
    teacher_val: pd.DataFrame
    generator_train: pd.DataFrame
    generator_item_ids: FrozenSet[int]
    policy_train: pd.DataFrame
    policy_train_item_ids: FrozenSet[int]
    policy_val: pd.DataFrame
    policy_val_item_ids: FrozenSet[int]


class CleanTeacher(nn.Module):
    """Behavior-only IV embedding teacher used as the frozen offline oracle."""

    def __init__(self, *, n_users: int, n_items: int, emb_dim: int, temperature: float = 0.07):
        super().__init__()
        if int(n_users) < 1 or int(n_items) < 1 or int(emb_dim) < 1:
            raise ValueError("n_users, n_items, and emb_dim must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        self.user_emb = nn.Embedding(int(n_users), int(emb_dim))
        self.item_emb = nn.Embedding(int(n_items), int(emb_dim))
        self.temperature = float(temperature)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def user_vectors(self, user_ids: torch.Tensor | None = None) -> torch.Tensor:
        source = self.user_emb.weight if user_ids is None else self.user_emb(user_ids.long())
        return F.normalize(source, dim=-1)

    def item_vectors(self, item_ids: torch.Tensor | None = None) -> torch.Tensor:
        source = self.item_emb.weight if item_ids is None else self.item_emb(item_ids.long())
        return F.normalize(source, dim=-1)

    def ranking_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_ids = torch.as_tensor(user_ids, device=self.user_emb.weight.device, dtype=torch.long).view(-1)
        item_ids = torch.as_tensor(item_ids, device=self.item_emb.weight.device, dtype=torch.long).view(-1)
        if user_ids.numel() != item_ids.numel() or user_ids.numel() < 1:
            raise ValueError("user_ids and item_ids must have equal nonzero length")
        users = self.user_vectors(user_ids)
        items = self.item_vectors(item_ids)
        labels = torch.arange(users.size(0), dtype=torch.long, device=users.device)
        logits = torch.matmul(users, items.t()) / self.temperature
        return F.cross_entropy(logits, labels)


class ContentGenerator(nn.Module):
    """Content-only mapping into the frozen teacher item embedding space."""

    def __init__(self, *, content_dim: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        if int(content_dim) < 1 or int(emb_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError("content_dim, emb_dim, and hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(int(content_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(emb_dim)),
        )

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return self.net(content.float())


def full_positive_score_gain(
    before_state: torch.Tensor,
    after_state: torch.Tensor,
    teacher_item: torch.Tensor,
    positive_user_ids: Sequence[torch.Tensor],
    teacher_user_bank: torch.Tensor,
) -> torch.Tensor:
    """Mean score-error reduction over every observed positive user per item."""
    if before_state.shape != after_state.shape or before_state.shape != teacher_item.shape:
        raise ValueError("before_state, after_state, and teacher_item must share shape")
    if len(positive_user_ids) != before_state.size(0):
        raise ValueError("positive_user_ids must contain one user set per batch row")
    gains = []
    for row, user_ids in enumerate(positive_user_ids):
        ids = torch.as_tensor(user_ids, dtype=torch.long, device=teacher_user_bank.device).view(-1)
        if ids.numel() == 0:
            gains.append(before_state.new_zeros(()))
            continue
        users = teacher_user_bank.index_select(0, ids).to(
            device=before_state.device,
            dtype=before_state.dtype,
        )
        target_scores = torch.matmul(users, teacher_item[row])
        before_error = torch.abs(torch.matmul(users, before_state[row]) - target_scores).mean()
        after_error = torch.abs(torch.matmul(users, after_state[row]) - target_scores).mean()
        gains.append(before_error - after_error)
    return torch.stack(gains, dim=0).view(-1, 1)


def target_excluded_history(
    user_history: Mapping[int, set[int]],
    *,
    target_item_ids: torch.Tensor,
    selected_user_ids: torch.Tensor,
) -> list[set[int]]:
    """Copy selected users' histories while removing each row's target item."""
    target_ids = torch.as_tensor(target_item_ids, dtype=torch.long).view(-1).tolist()
    user_ids = torch.as_tensor(selected_user_ids, dtype=torch.long).view(-1).tolist()
    if len(target_ids) != len(user_ids):
        raise ValueError("target_item_ids and selected_user_ids must have equal length")
    return [
        {int(item_id) for item_id in user_history.get(int(user_id), set()) if int(item_id) != int(target_id)}
        for target_id, user_id in zip(target_ids, user_ids)
    ]


def project_displacement(
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    *,
    max_delta: float,
) -> torch.Tensor:
    """Cap the final state change without changing its direction."""
    if initial_state.shape != final_state.shape:
        raise ValueError("initial_state and final_state must share shape")
    if float(max_delta) < 0.0:
        raise ValueError("max_delta must be non-negative")
    delta = final_state - initial_state
    norm = delta.norm(dim=1, keepdim=True)
    scale = torch.clamp(float(max_delta) / norm.clamp_min(1e-12), max=1.0)
    return initial_state + delta * scale


class LegalUSIMPolicy(nn.Module):
    """Actor/critic over inference-legal users and a separate END action."""

    def __init__(self, *, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(int(emb_dim) + 1, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.query = nn.Linear(int(hidden_dim), int(emb_dim))
        self.user_key = nn.Linear(int(emb_dim), int(emb_dim), bias=False)
        self.end_head = nn.Linear(int(hidden_dim), 1)
        self.value_head = nn.Linear(int(hidden_dim), 1)

    def action_value(
        self,
        state: torch.Tensor,
        remaining_steps: torch.Tensor,
        candidates: torch.Tensor,
        *,
        candidate_logit_bias: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or candidates.ndim != 3:
            raise ValueError("state must be [batch, dim] and candidates must be [batch, count, dim]")
        if candidates.size(0) != state.size(0) or candidates.size(2) != state.size(1):
            raise ValueError("candidate dimensions must match state")
        remaining = torch.as_tensor(
            remaining_steps, dtype=state.dtype, device=state.device
        ).view(-1, 1)
        if remaining.size(0) != state.size(0):
            raise ValueError("remaining_steps must have one value per state row")
        features = self.common(torch.cat([state, remaining], dim=1))
        user_logits = torch.matmul(
            self.query(features).unsqueeze(1),
            self.user_key(candidates).transpose(1, 2),
        ).squeeze(1)
        if candidate_logit_bias is not None:
            bias = torch.as_tensor(
                candidate_logit_bias,
                dtype=user_logits.dtype,
                device=user_logits.device,
            )
            if bias.shape != user_logits.shape:
                raise ValueError("candidate_logit_bias must be [batch, candidate_count]")
            user_logits = user_logits + bias
        logits = torch.cat([user_logits, self.end_head(features)], dim=1)
        distribution = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=1) if deterministic else distribution.sample()
        else:
            action = torch.as_tensor(action, dtype=torch.long, device=state.device).view(-1)
        return action, distribution.log_prob(action), self.value_head(features), distribution.entropy()


@dataclass(frozen=True)
class CleanRolloutResult:
    final_state: torch.Tensor
    raw_final_state: torch.Tensor
    trajectory: dict[str, list[torch.Tensor]]
    stats: dict[str, float]


CourseBiasFn = Callable[[torch.Tensor, torch.Tensor, list[set[int]]], torch.Tensor]
CourseRewardFn = Callable[[torch.Tensor, torch.Tensor, list[set[int]]], torch.Tensor]


class CleanUSIMEngine:
    """USIM episode engine with one legal action pool for train and inference."""

    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        max_steps: int,
        candidate_count: int,
        step_size: float,
        step_penalty: float,
        max_delta: float,
        retrieval_chunk: int = 8192,
        course_bias_fn: CourseBiasFn | None = None,
        course_reward_fn: CourseRewardFn | None = None,
    ):
        if int(max_steps) < 1 or int(candidate_count) < 1 or int(retrieval_chunk) < 1:
            raise ValueError("max_steps, candidate_count, and retrieval_chunk must be positive")
        if float(step_size) <= 0.0 or float(step_penalty) < 0.0 or float(max_delta) < 0.0:
            raise ValueError("step_size must be positive; penalties and max_delta non-negative")
        self.policy = LegalUSIMPolicy(emb_dim=int(emb_dim), hidden_dim=int(hidden_dim))
        self.max_steps = int(max_steps)
        self.candidate_count = int(candidate_count)
        self.step_size = float(step_size)
        self.step_penalty = float(step_penalty)
        self.max_delta = float(max_delta)
        self.retrieval_chunk = int(retrieval_chunk)
        self.course_bias_fn = course_bias_fn
        self.course_reward_fn = course_reward_fn

    @staticmethod
    def _candidate_vectors(user_bank: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        return user_bank.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0), candidate_ids.size(1), user_bank.size(1)
        )

    def legal_candidate_ids(self, state: torch.Tensor, user_bank: torch.Tensor) -> torch.Tensor:
        """Retrieve state-nearest users. No target/residual/positive inputs exist here."""
        if state.ndim != 2 or user_bank.ndim != 2 or state.size(1) != user_bank.size(1):
            raise ValueError("state and user_bank must be two-dimensional with matching width")
        if user_bank.size(0) < 1:
            raise ValueError("user_bank must contain at least one user")
        count = min(self.candidate_count, int(user_bank.size(0)))
        query = F.normalize(state, dim=1)
        top_scores = None
        top_ids = None
        for start in range(0, int(user_bank.size(0)), self.retrieval_chunk):
            end = min(start + self.retrieval_chunk, int(user_bank.size(0)))
            chunk_scores = torch.matmul(query, F.normalize(user_bank[start:end], dim=1).t())
            chunk_ids = torch.arange(start, end, dtype=torch.long, device=state.device).view(1, -1)
            chunk_ids = chunk_ids.expand(state.size(0), -1)
            if top_scores is None:
                merged_scores, merged_ids = chunk_scores, chunk_ids
            else:
                merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
                merged_ids = torch.cat([top_ids, chunk_ids], dim=1)
            keep = min(count, int(merged_scores.size(1)))
            top_scores, positions = torch.topk(merged_scores, k=keep, dim=1)
            top_ids = merged_ids.gather(1, positions)
        return top_ids

    def _candidate_logit_bias(
        self,
        candidate_ids: torch.Tensor,
        item_ids: torch.Tensor | None,
        user_history: Mapping[int, set[int]] | None,
    ) -> torch.Tensor:
        zero = torch.zeros(candidate_ids.shape, device=candidate_ids.device, dtype=torch.float32)
        if self.course_bias_fn is None or item_ids is None or user_history is None:
            return zero
        flat_item_ids = item_ids.view(-1, 1).expand_as(candidate_ids).reshape(-1)
        histories = target_excluded_history(
            user_history,
            target_item_ids=flat_item_ids,
            selected_user_ids=candidate_ids.reshape(-1),
        )
        bias = self.course_bias_fn(candidate_ids, item_ids, histories)
        bias = torch.as_tensor(bias, dtype=torch.float32, device=candidate_ids.device)
        if bias.shape != candidate_ids.shape:
            raise ValueError("course_bias_fn must return [batch, candidate_count]")
        return torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)

    def _course_reward(
        self,
        selected_user_ids: torch.Tensor,
        item_ids: torch.Tensor | None,
        user_history: Mapping[int, set[int]] | None,
    ) -> torch.Tensor:
        zero = torch.zeros((selected_user_ids.size(0), 1), device=selected_user_ids.device)
        if self.course_reward_fn is None or item_ids is None or user_history is None:
            return zero
        histories = target_excluded_history(
            user_history,
            target_item_ids=item_ids,
            selected_user_ids=selected_user_ids,
        )
        reward = self.course_reward_fn(selected_user_ids, item_ids, histories)
        reward = torch.as_tensor(reward, dtype=zero.dtype, device=zero.device).view(-1, 1)
        if reward.shape != zero.shape:
            raise ValueError("course_reward_fn must return [batch, 1]")
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

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
        embedding_gain = (before_state - target_emb).norm(dim=1, keepdim=True) - (
            after_state - target_emb
        ).norm(dim=1, keepdim=True)
        recommendation_gain = full_positive_score_gain(
            before_state,
            after_state,
            target_emb,
            positive_user_ids,
            user_bank,
        )
        course_reward = self._course_reward(selected_user_ids, item_ids, user_history)
        active = active_user.float().view(-1, 1)
        reward = (embedding_gain + recommendation_gain - self.step_penalty + course_reward) * active
        return reward, embedding_gain * active, recommendation_gain * active, course_reward * active

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
    ) -> CleanRolloutResult:
        """Run a train-supervised or target-free deployment episode."""
        if initial_state.ndim != 2:
            raise ValueError("initial_state must be [batch, dim]")
        if not training and (target_emb is not None or positive_user_ids is not None):
            raise ValueError("inference rollout forbids oracle target and positive users")
        if training and (target_emb is None or positive_user_ids is None):
            raise ValueError("training rollout requires oracle target and positive users")
        if target_emb is not None and target_emb.shape != initial_state.shape:
            raise ValueError("target_emb must match initial_state")
        if positive_user_ids is not None and len(positive_user_ids) != initial_state.size(0):
            raise ValueError("positive_user_ids must have one entry per state row")
        if item_ids is not None and torch.as_tensor(item_ids).numel() != initial_state.size(0):
            raise ValueError("item_ids must have one entry per state row")

        user_bank = torch.as_tensor(user_bank, dtype=initial_state.dtype, device=initial_state.device)
        state = initial_state
        done = torch.zeros(initial_state.size(0), dtype=torch.bool, device=initial_state.device)
        trajectory: dict[str, list[torch.Tensor]] = {
            "states": [], "next_states": [], "candidate_ids": [], "candidate_logit_bias": [],
            "actions": [], "rewards": [], "done": [], "old_log_probs": [],
            "remaining_steps": [], "next_remaining_steps": [], "terminal_states": [],
            "terminal_remaining_steps": [],
        }
        active_steps = 0
        end_actions = 0
        embedding_reward = 0.0
        recommendation_reward = 0.0
        course_reward = 0.0

        for step in range(self.max_steps):
            remaining = torch.full(
                (state.size(0), 1),
                float(self.max_steps - step) / float(self.max_steps),
                dtype=state.dtype,
                device=state.device,
            )
            candidate_ids = self.legal_candidate_ids(state, user_bank)
            candidates = self._candidate_vectors(user_bank, candidate_ids)
            candidate_bias = self._candidate_logit_bias(candidate_ids, item_ids, user_history)
            action, log_prob, _, _ = self.policy.action_value(
                state,
                remaining,
                candidates,
                candidate_logit_bias=candidate_bias,
                deterministic=not training,
            )
            end_action = candidates.size(1)
            safe_action = action.clamp(max=end_action - 1)
            row_index = torch.arange(state.size(0), device=state.device)
            selected_users = candidates[row_index, safe_action]
            selected_user_ids = candidate_ids[row_index, safe_action]
            active_before = ~done
            active_user = active_before & action.ne(end_action)
            next_state = torch.where(
                active_user.view(-1, 1),
                state + self.step_size * selected_users,
                state,
            )
            done_after = done | action.eq(end_action)
            if training:
                reward, emb_reward, rec_reward, ckg_reward = self._training_reward(
                    state,
                    next_state,
                    target_emb,
                    positive_user_ids,
                    user_bank,
                    active_user,
                    selected_user_ids,
                    item_ids,
                    user_history,
                )
                embedding_reward += float(emb_reward.mean().detach().item())
                recommendation_reward += float(rec_reward.mean().detach().item())
                course_reward += float(ckg_reward.mean().detach().item())
            else:
                reward = state.new_zeros((state.size(0), 1))
            next_remaining = torch.clamp(remaining - 1.0 / float(self.max_steps), min=0.0)
            trajectory["states"].append(state.detach())
            trajectory["next_states"].append(next_state.detach())
            trajectory["candidate_ids"].append(candidate_ids.detach())
            trajectory["candidate_logit_bias"].append(candidate_bias.detach())
            trajectory["actions"].append(action.detach())
            trajectory["rewards"].append(reward.detach())
            trajectory["done"].append(done_after.detach())
            trajectory["old_log_probs"].append(log_prob.detach())
            trajectory["remaining_steps"].append(remaining.detach())
            trajectory["next_remaining_steps"].append(next_remaining.detach())
            terminal_state = target_emb.detach() if training else next_state.detach()
            trajectory["terminal_states"].append(terminal_state)
            trajectory["terminal_remaining_steps"].append(torch.zeros_like(remaining))
            active_steps += int(active_user.sum().detach().item())
            end_actions += int((active_before & action.eq(end_action)).sum().detach().item())
            state, done = next_state, done_after

        projected = project_displacement(initial_state, state, max_delta=self.max_delta)
        denominator = max(1, initial_state.size(0))
        return CleanRolloutResult(
            final_state=projected,
            raw_final_state=state,
            trajectory=trajectory,
            stats={
                "end_rate": float(end_actions / denominator),
                "active_steps": float(active_steps / denominator),
                "embedding_reward": float(embedding_reward / self.max_steps),
                "recommendation_reward": float(recommendation_reward / self.max_steps),
                "course_reward": float(course_reward / self.max_steps),
                "rollout_delta_l2": float((projected - initial_state).norm(dim=1).mean().detach().item()),
            },
        )


class CleanCourseSignal:
    """Observable course compatibility signal over target-excluded histories."""

    def __init__(
        self,
        *,
        item_concept_overlap: torch.Tensor | None,
        item_prereq_item_mat: torch.Tensor | None,
        item_prereq_item_count: torch.Tensor | None,
        item_popularity: torch.Tensor,
        concept_weight: float = 0.04,
        prereq_weight: float = 0.08,
        difficulty_weight: float = 0.03,
        redundant_weight: float = 0.02,
        warm_seen: float = 5.0,
        concept_minimum: float = 0.12,
        redundant_threshold: float = 0.70,
        bias_scale: float = 0.20,
    ):
        popularity = torch.as_tensor(item_popularity, dtype=torch.float32).view(-1).cpu()
        if popularity.numel() < 1 or torch.any(popularity < 0):
            raise ValueError("item_popularity must be a nonempty non-negative vector")
        n_items = int(popularity.numel())
        for name, matrix in (
            ("item_concept_overlap", item_concept_overlap),
            ("item_prereq_item_mat", item_prereq_item_mat),
        ):
            if matrix is not None and torch.as_tensor(matrix).shape != (n_items, n_items):
                raise ValueError(f"{name} must be [n_items, n_items]")
        if item_prereq_item_count is not None and torch.as_tensor(item_prereq_item_count).numel() != n_items:
            raise ValueError("item_prereq_item_count must have n_items entries")
        if float(warm_seen) <= 0.0 or float(bias_scale) < 0.0:
            raise ValueError("warm_seen must be positive and bias_scale non-negative")
        self.item_concept_overlap = (
            None if item_concept_overlap is None else torch.as_tensor(item_concept_overlap, dtype=torch.float32).cpu()
        )
        self.item_prereq_item_mat = (
            None if item_prereq_item_mat is None else torch.as_tensor(item_prereq_item_mat, dtype=torch.float32).cpu()
        )
        self.item_prereq_item_count = (
            None if item_prereq_item_count is None else torch.as_tensor(item_prereq_item_count, dtype=torch.float32).view(-1).cpu()
        )
        self.item_popularity = popularity
        max_log = torch.log1p(popularity.max()).clamp_min(1.0)
        self.item_difficulty = 1.0 - torch.log1p(popularity) / max_log
        self.concept_weight = float(concept_weight)
        self.prereq_weight = float(prereq_weight)
        self.difficulty_weight = float(difficulty_weight)
        self.redundant_weight = float(redundant_weight)
        self.warm_seen = float(warm_seen)
        self.concept_minimum = float(concept_minimum)
        self.redundant_threshold = float(redundant_threshold)
        self.bias_scale = float(bias_scale)

    def _compatibility(
        self,
        item_ids: torch.Tensor,
        histories: Sequence[set[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        ids = torch.as_tensor(item_ids, dtype=torch.long).view(-1).cpu()
        if ids.numel() != len(histories):
            raise ValueError("item_ids and histories must have equal length")
        values = torch.zeros(ids.numel(), dtype=torch.float32)
        for row, (item_id, history) in enumerate(zip(ids.tolist(), histories)):
            if item_id < 0 or item_id >= self.item_popularity.numel():
                raise ValueError("course signal item ID is out of range")
            seen = sorted(
                int(seen_item)
                for seen_item in history
                if 0 <= int(seen_item) < self.item_popularity.numel()
            )
            if not seen:
                continue
            seen_ids = torch.tensor(seen, dtype=torch.long)
            readiness = min(1.0, len(seen) / self.warm_seen)
            prereq_gap = 0.0
            if self.item_prereq_item_mat is not None and self.item_prereq_item_count is not None:
                count = float(self.item_prereq_item_count[item_id].item())
                if count > 0.0:
                    covered = float(self.item_prereq_item_mat[item_id, seen_ids].sum().item())
                    prereq_gap = max(0.0, min(1.0, 1.0 - covered / count))
            concept_bonus = 0.0
            redundancy = 0.0
            if self.item_concept_overlap is not None:
                concept_match = float(self.item_concept_overlap[item_id, seen_ids].mean().item())
                band = max(1e-6, self.redundant_threshold - self.concept_minimum)
                concept_bonus = max(0.0, min(1.0, (concept_match - self.concept_minimum) / band))
                redundancy = max(
                    0.0,
                    min(1.0, (concept_match - self.redundant_threshold) / max(1e-6, 1.0 - self.redundant_threshold)),
                )
            difficulty_gap = max(0.0, float(self.item_difficulty[item_id].item()) - readiness)
            values[row] = (
                self.concept_weight * concept_bonus
                - self.prereq_weight * prereq_gap
                - self.difficulty_weight * difficulty_gap
                - self.redundant_weight * redundancy
            )
        return values.to(device=device)

    def candidate_bias(
        self,
        candidate_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: list[set[int]],
    ) -> torch.Tensor:
        if candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must be [batch, candidate_count]")
        flat_item_ids = item_ids.view(-1, 1).expand_as(candidate_ids).reshape(-1)
        raw = self._compatibility(flat_item_ids, histories, device=candidate_ids.device).view_as(candidate_ids)
        return self.bias_scale * raw / raw.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

    def reward(
        self,
        selected_user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: list[set[int]],
    ) -> torch.Tensor:
        del selected_user_ids
        return self._compatibility(item_ids, histories, device=item_ids.device).view(-1, 1)


class CleanReplayBuffer:
    """Bounded detached transition storage for the clean RecPPO update."""

    required_keys = (
        "state",
        "next_state",
        "candidate_ids",
        "candidate_logit_bias",
        "action",
        "reward",
        "done",
        "old_log_prob",
        "remaining_steps",
        "next_remaining_steps",
        "terminal_state",
        "terminal_remaining_steps",
    )

    def __init__(self, capacity: int):
        if int(capacity) < 1:
            raise ValueError("replay capacity must be positive")
        self._records: deque[dict[str, torch.Tensor]] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._records)

    def append(self, transition: Mapping[str, torch.Tensor]) -> None:
        missing = [key for key in self.required_keys if key not in transition]
        if missing:
            raise KeyError(f"replay transition is missing {missing}")
        batch_size = int(torch.as_tensor(transition["action"]).view(-1).numel())
        for key in self.required_keys:
            if torch.as_tensor(transition[key]).shape[0] != batch_size:
                raise ValueError(f"replay field {key!r} has an incompatible batch dimension")
        for row in range(batch_size):
            self._records.append(
                {
                    key: torch.as_tensor(transition[key][row]).detach().cpu().clone()
                    for key in self.required_keys
                }
            )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if not self._records:
            raise RuntimeError("cannot sample an empty replay buffer")
        count = min(int(batch_size), len(self._records))
        rows = random.sample(list(self._records), count)
        return {
            key: torch.stack([row[key] for row in rows], dim=0).to(device=device)
            for key in self.required_keys
        }


class CleanRecPPO:
    """Detached replay PPO with a frozen target critic and terminal anchor."""

    def __init__(
        self,
        policy: LegalUSIMPolicy,
        *,
        replay_capacity: int,
        replay_batch_size: int,
        gamma: float,
        clip_ratio: float,
        value_weight: float,
        terminal_value_weight: float,
        entropy_weight: float,
    ):
        if int(replay_batch_size) < 1:
            raise ValueError("replay_batch_size must be positive")
        if not 0.0 <= float(gamma) <= 1.0 or float(clip_ratio) <= 0.0:
            raise ValueError("gamma must be in [0, 1] and clip_ratio positive")
        self.policy = policy
        self.target_policy = copy.deepcopy(policy).eval()
        for parameter in self.target_policy.parameters():
            parameter.requires_grad_(False)
        self.replay = CleanReplayBuffer(replay_capacity)
        self.replay_batch_size = int(replay_batch_size)
        self.gamma = float(gamma)
        self.clip_ratio = float(clip_ratio)
        self.value_weight = float(value_weight)
        self.terminal_value_weight = float(terminal_value_weight)
        self.entropy_weight = float(entropy_weight)
        self.last_stats: dict[str, float] = {}

    def sync_target(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

    def _append_trajectory(self, trajectory: Mapping[str, Sequence[torch.Tensor]]) -> None:
        required = (
            "states", "next_states", "candidate_ids", "candidate_logit_bias", "actions", "rewards",
            "done", "old_log_probs", "remaining_steps", "next_remaining_steps", "terminal_states",
            "terminal_remaining_steps",
        )
        missing = [key for key in required if key not in trajectory]
        if missing:
            raise KeyError(f"rollout trajectory is missing {missing}")
        step_count = len(trajectory["rewards"])
        if any(len(trajectory[key]) != step_count for key in required):
            raise ValueError("rollout trajectory lists must have equal length")
        for step in range(step_count):
            self.replay.append(
                {
                    "state": trajectory["states"][step],
                    "next_state": trajectory["next_states"][step],
                    "candidate_ids": trajectory["candidate_ids"][step],
                    "candidate_logit_bias": trajectory["candidate_logit_bias"][step],
                    "action": trajectory["actions"][step],
                    "reward": trajectory["rewards"][step],
                    "done": trajectory["done"][step],
                    "old_log_prob": trajectory["old_log_probs"][step],
                    "remaining_steps": trajectory["remaining_steps"][step],
                    "next_remaining_steps": trajectory["next_remaining_steps"][step],
                    "terminal_state": trajectory["terminal_states"][step],
                    "terminal_remaining_steps": trajectory["terminal_remaining_steps"][step],
                }
            )

    def loss(self, trajectory: Mapping[str, Sequence[torch.Tensor]], user_bank: torch.Tensor) -> torch.Tensor:
        self._append_trajectory(trajectory)
        device = next(self.policy.parameters()).device
        sample = self.replay.sample(self.replay_batch_size, device)
        user_bank = torch.as_tensor(user_bank, dtype=sample["state"].dtype, device=device)
        candidate_ids = sample["candidate_ids"].long()
        candidates = user_bank.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0), candidate_ids.size(1), user_bank.size(1)
        )
        _, new_log_prob, value, entropy = self.policy.action_value(
            sample["state"],
            sample["remaining_steps"],
            candidates,
            candidate_logit_bias=sample["candidate_logit_bias"],
            action=sample["action"],
        )
        dummy_action = torch.zeros(sample["state"].size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, target_value, _ = self.target_policy.action_value(
                sample["state"],
                sample["remaining_steps"],
                candidates,
                candidate_logit_bias=sample["candidate_logit_bias"],
                action=dummy_action,
            )
            _, _, target_next_value, _ = self.target_policy.action_value(
                sample["next_state"],
                sample["next_remaining_steps"],
                candidates,
                candidate_logit_bias=sample["candidate_logit_bias"],
                action=dummy_action,
            )
            done = sample["done"].float().view(-1, 1)
            td_target = sample["reward"] + self.gamma * (1.0 - done) * target_next_value
            advantage = (td_target - target_value).detach().view(-1)
        ratio = torch.exp(new_log_prob - sample["old_log_prob"].detach().view(-1))
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        actor_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
        critic_loss = F.mse_loss(value, td_target.detach())
        _, _, terminal_value, _ = self.policy.action_value(
            sample["terminal_state"],
            sample["terminal_remaining_steps"],
            candidates,
            candidate_logit_bias=sample["candidate_logit_bias"],
            action=dummy_action,
        )
        terminal_value_loss = terminal_value.pow(2).mean()
        total = (
            actor_loss
            + self.value_weight * critic_loss
            + self.terminal_value_weight * terminal_value_loss
            - self.entropy_weight * entropy.mean()
        )
        self.last_stats = {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "terminal_value_loss": float(terminal_value_loss.detach().item()),
            "entropy": float(entropy.mean().detach().item()),
            "replay_size": float(len(self.replay)),
        }
        return total


def _policy_mode(policy_epoch: int) -> str:
    if int(policy_epoch) == 0:
        return "identity_generator"
    if int(policy_epoch) > 0:
        return "ppo_rollout"
    raise ValueError("policy_epoch must be non-negative")


@torch.no_grad()
def build_clean_item_bank(
    teacher: CleanTeacher,
    generator: ContentGenerator,
    engine: CleanUSIMEngine,
    content: torch.Tensor,
    *,
    strict_cold_item_ids: torch.Tensor,
    user_history: Mapping[int, set[int]] | None,
    policy_epoch: int,
    item_batch_size: int = 128,
) -> torch.Tensor:
    """Return a teacher-identical hot bank with identity or policy cold replacements."""
    if int(item_batch_size) < 1:
        raise ValueError("item_batch_size must be positive")
    policy_mode = _policy_mode(policy_epoch)
    device = teacher.item_emb.weight.device
    content = torch.as_tensor(content, dtype=torch.float32, device=device)
    teacher_bank = teacher.item_vectors().detach()
    if content.ndim != 2 or content.size(0) != teacher_bank.size(0):
        raise ValueError("content must have one row for every teacher item")
    cold_ids = torch.as_tensor(strict_cold_item_ids, dtype=torch.long, device=device).view(-1)
    if cold_ids.numel() == 0:
        return teacher_bank.clone()
    if int(cold_ids.min().item()) < 0 or int(cold_ids.max().item()) >= teacher_bank.size(0):
        raise ValueError("strict_cold_item_ids contains an out-of-range item")
    cold_ids = torch.unique(cold_ids, sorted=True)
    item_bank = teacher_bank.clone()
    user_bank = teacher.user_vectors().detach()
    use_policy_rollout = policy_mode == "ppo_rollout"
    generator_was_training = generator.training
    policy_was_training = engine.policy.training
    generator.eval()
    if use_policy_rollout:
        engine.policy.eval()
    try:
        for start in range(0, cold_ids.numel(), int(item_batch_size)):
            item_ids = cold_ids[start:start + int(item_batch_size)]
            initial_state = F.normalize(generator(content.index_select(0, item_ids)), dim=1)
            if not use_policy_rollout:
                item_bank.index_copy_(0, item_ids, initial_state)
                continue
            rollout = engine.rollout(
                initial_state,
                user_bank=user_bank,
                training=False,
                target_emb=None,
                positive_user_ids=None,
                item_ids=item_ids,
                user_history=user_history,
            )
            item_bank.index_copy_(0, item_ids, F.normalize(rollout.final_state, dim=1))
    finally:
        generator.train(generator_was_training)
        if use_policy_rollout:
            engine.policy.train(policy_was_training)
    return item_bank


@dataclass(frozen=True)
class CleanRunConfig:
    """All route choices for the isolated clean teacher/generator/policy run."""

    seed: int
    data_dir: str | Path
    split_dir: str | Path
    output_dir: str | Path
    checkpoint_dir: str | Path
    teacher_epochs: int = 20
    generator_epochs: int = 30
    policy_epochs: int = 15
    batch_size: int = 1024
    eval_batch_size: int = 2048
    emb_dim: int = 128
    hidden_dim: int = 256
    teacher_lr: float = 5e-4
    generator_lr: float = 1e-3
    policy_lr: float = 3e-4
    pseudo_ratio: float = 0.30
    pseudo_val_fraction: float = 0.20
    pseudo_min_popularity: int = 1
    generator_val_fraction: float = 0.10
    candidate_count: int = 20
    retrieval_chunk: int = 8192
    max_steps: int = 5
    step_size: float = 0.05
    step_penalty: float = 0.01
    max_delta: float = 0.25
    replay_capacity: int = 8192
    replay_batch_size: int = 512
    ppo_gamma: float = 0.99
    ppo_clip_ratio: float = 0.20
    ppo_value_weight: float = 0.50
    ppo_terminal_value_weight: float = 1.0
    ppo_entropy_weight: float = 0.01
    hot_retention_tolerance: float = 0.003
    cold_threshold: int = 1
    device: str = ""
    use_course_signal: bool = False
    course_relation_dir: str | Path = "MOOCCube/relations"
    course_bias_scale: float = 0.20
    course_concept_weight: float = 0.04
    course_prereq_weight: float = 0.08
    course_difficulty_weight: float = 0.03
    course_redundant_weight: float = 0.02

    @classmethod
    def for_seed(cls, seed: int) -> "CleanRunConfig":
        seed = int(seed)
        split_dir = (
            "outputs/content_delta_pop5/static_item_cold_balanced/"
            f"strict_item_cold_balanced_thr1_seed_{seed}"
        )
        return cls(
            seed=seed,
            data_dir="processed_data_hin_clean_pop5",
            split_dir=split_dir,
            output_dir=f"outputs/ckg_rl_usim_v32_clean/seed{seed}",
            checkpoint_dir=f"checkpoints/ckg_rl_usim_v32_clean/seed{seed}",
        )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (_REPO_ROOT / path)


def _resolve_device(requested: str) -> torch.device:
    raw = str(requested).strip().lower()
    if raw:
        if raw == "cpu":
            return torch.device("cpu")
        if raw.startswith("cuda") and torch.cuda.is_available():
            return torch.device(raw)
        raise RuntimeError(f"requested unavailable device: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_run_config(config: CleanRunConfig) -> None:
    positive_ints = {
        "teacher_epochs": config.teacher_epochs,
        "generator_epochs": config.generator_epochs,
        "policy_epochs": config.policy_epochs,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "emb_dim": config.emb_dim,
        "hidden_dim": config.hidden_dim,
        "candidate_count": config.candidate_count,
        "retrieval_chunk": config.retrieval_chunk,
        "max_steps": config.max_steps,
        "replay_capacity": config.replay_capacity,
        "replay_batch_size": config.replay_batch_size,
        "cold_threshold": config.cold_threshold,
    }
    invalid = [name for name, value in positive_ints.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"clean run requires positive values for {invalid}")
    if not 0.0 < float(config.pseudo_ratio) <= 1.0:
        raise ValueError("pseudo_ratio must be in (0, 1]")
    if not 0.0 <= float(config.pseudo_val_fraction) < 1.0:
        raise ValueError("pseudo_val_fraction must be in [0, 1)")
    if not 0.0 <= float(config.generator_val_fraction) < 1.0:
        raise ValueError("generator_val_fraction must be in [0, 1)")
    if float(config.step_size) <= 0.0 or float(config.step_penalty) < 0.0:
        raise ValueError("step_size must be positive and step_penalty non-negative")
    if float(config.max_delta) < 0.0:
        raise ValueError("max_delta must be non-negative")
    if not 0.0 <= float(config.ppo_gamma) <= 1.0:
        raise ValueError("ppo_gamma must be in [0, 1]")
    if float(config.course_bias_scale) < 0.0:
        raise ValueError("course_bias_scale must be non-negative")


def _torch_load_tensor(path: Path) -> torch.Tensor:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, torch.Tensor):
        raise ValueError(f"expected a tensor in {path}")
    return loaded.float()


def _with_train_popularity(
    train_df: pd.DataFrame,
    *frames: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    train = train_df.copy()
    counts = train["i_idx"].astype(int).value_counts().astype(int)
    output = []
    for frame in (train, *frames):
        copied = frame.copy()
        copied["popularity"] = copied["i_idx"].astype(int).map(counts).fillna(0).astype(int)
        if "_row_id" not in copied.columns:
            copied["_row_id"] = range(len(copied))
        output.append(copied)
    return tuple(output)


def load_clean_train_val_inputs(
    config: CleanRunConfig,
) -> tuple[dict[str, Any], torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Load only catalog, H_train, and outer validation before policy selection."""
    data_dir = _resolve_path(config.data_dir)
    split_dir = _resolve_path(config.split_dir)
    meta_path = data_dir / "meta.json"
    content_path = data_dir / "content_emb.pt"
    train_path = split_dir / "static_train.pkl"
    val_path = split_dir / "static_val.pkl"
    for path in (meta_path, content_path, train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(f"clean train/validation input is missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    content = _torch_load_tensor(content_path)
    train = pd.read_pickle(train_path).copy()
    validation = pd.read_pickle(val_path).copy()
    for name, frame in (("static_train.pkl", train), ("static_val.pkl", validation)):
        _require_interaction_columns(frame, name)
    if int(meta.get("n_items", -1)) != int(content.size(0)):
        raise ValueError("meta n_items does not match content row count")
    if int(meta.get("content_dim", -1)) != int(content.size(1)):
        raise ValueError("meta content_dim does not match content width")
    train, validation = _with_train_popularity(train, validation)
    return meta, content, train, validation


def load_clean_test_inputs(config: CleanRunConfig, train_df: pd.DataFrame) -> pd.DataFrame:
    """Load the outer test split only after all policy selection is complete."""
    test_path = _resolve_path(config.split_dir) / "static_test.pkl"
    if not test_path.is_file():
        raise FileNotFoundError(f"clean test input is missing: {test_path}")
    test = pd.read_pickle(test_path).copy()
    _require_interaction_columns(test, "static_test.pkl")
    _, test = _with_train_popularity(train_df, test)
    return test


def attach_clean_test_rows(partitions: CleanPartitions, test_df: pd.DataFrame) -> CleanPartitions:
    """Classify delayed outer-test rows with the already-fixed warm catalog."""
    warm_ids = partitions.g_item_ids | partitions.p_train_item_ids | partitions.p_val_item_ids
    h_test, c_test = _split_rows_by_item_ids(test_df, warm_ids)
    return replace(partitions, h_test=h_test, c_test=c_test)


def build_clean_course_signal(
    train_df: pd.DataFrame,
    *,
    n_items: int,
    config: CleanRunConfig,
) -> tuple[CleanCourseSignal | None, dict[str, Any]]:
    """Build course features from catalog metadata and relation files only."""
    if not config.use_course_signal:
        return None, {"enabled": False}
    from fast3_delta.course_artifacts import build_course_artifacts

    metadata_path = _resolve_path(config.data_dir) / "stream_data.pkl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"course metadata source is missing: {metadata_path}")
    metadata = pd.read_pickle(metadata_path)
    required = {"i_idx", "course_id"}
    if not required.issubset(metadata.columns):
        raise ValueError("course metadata source must include i_idx and course_id")
    # Pass only catalog identity metadata. Concept prerequisites do not consume
    # interaction order or outer cold labels.
    catalog_metadata = metadata.loc[:, ["i_idx", "course_id"]].drop_duplicates("i_idx").copy()
    artifacts, stats = build_course_artifacts(
        catalog_metadata,
        int(n_items),
        relation_dir=str(_resolve_path(config.course_relation_dir)),
        prereq_graph_source="concept",
    )
    popularity = torch.zeros(int(n_items), dtype=torch.float32)
    for item_id, count in train_df["i_idx"].astype(int).value_counts().items():
        popularity[int(item_id)] = float(count)
    signal = CleanCourseSignal(
        item_concept_overlap=artifacts.get("item_concept_overlap"),
        item_prereq_item_mat=artifacts.get("item_prereq_item_mat"),
        item_prereq_item_count=artifacts.get("item_prereq_item_cnt"),
        item_popularity=popularity,
        concept_weight=config.course_concept_weight,
        prereq_weight=config.course_prereq_weight,
        difficulty_weight=config.course_difficulty_weight,
        redundant_weight=config.course_redundant_weight,
        bias_scale=config.course_bias_scale,
    )
    return signal, {"enabled": True, "source": "catalog_metadata_concept_relations", **dict(stats)}


def create_clean_engine(
    config: CleanRunConfig,
    *,
    course_signal: CleanCourseSignal | None = None,
) -> CleanUSIMEngine:
    """Create the policy engine with only optional observable course features."""
    if config.use_course_signal and course_signal is None:
        raise ValueError("use_course_signal requires a CleanCourseSignal")
    return CleanUSIMEngine(
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
    )


def _build_eval_loader(frame: pd.DataFrame, batch_size: int) -> DataLoader:
    return DataLoader(
        InteractionDataset(frame),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_interactions,
    )


def _evaluate_frame(
    frame: pd.DataFrame,
    *,
    teacher: CleanTeacher,
    item_bank: torch.Tensor,
    user_history: Mapping[int, set[int]],
    batch_size: int,
    device: torch.device,
    export_item_metrics_path: str | None = None,
) -> tuple[dict[str, float] | None, int]:
    if frame.empty:
        return None, 0
    loader = _build_eval_loader(frame, batch_size)
    return evaluate_embedding_ranker(
        loader,
        device=device,
        n_items=item_bank.size(0),
        cold_threshold=1,
        get_user_vectors_fn=lambda batch: teacher.user_vectors(batch["u"]),
        all_item_vectors=item_bank,
        k_list=(5, 10, 20),
        eval_type="all",
        full_ranking=True,
        user_seen_items=dict(user_history),
        average_mode="item_macro",
        export_item_metrics_path=export_item_metrics_path,
    )


def _overall_metric(
    cold: dict[str, float] | None,
    cold_count: int,
    hot: dict[str, float] | None,
    hot_count: int,
    key: str,
) -> float:
    total = int(cold_count) + int(hot_count)
    if total < 1:
        return 0.0
    cold_value = 0.0 if cold is None else float(cold.get(key, 0.0))
    hot_value = 0.0 if hot is None else float(hot.get(key, 0.0))
    return (cold_value * int(cold_count) + hot_value * int(hot_count)) / total


def evaluate_clean_route(
    teacher: CleanTeacher,
    generator: ContentGenerator,
    engine: CleanUSIMEngine,
    *,
    hot_frame: pd.DataFrame,
    cold_frame: pd.DataFrame,
    warm_item_ids: FrozenSet[int],
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    config: CleanRunConfig,
    policy_epoch: int,
    export_dir: Path | None = None,
    export_prefix: str = "",
) -> dict[str, float | int | str]:
    """Evaluate hot teacher parity and strict-cold target-free adaptation."""
    all_item_ids = set(range(int(content.size(0))))
    strict_cold_ids = torch.tensor(
        sorted(all_item_ids - set(warm_item_ids)), dtype=torch.long, device=teacher.item_emb.weight.device
    )
    item_bank = build_clean_item_bank(
        teacher,
        generator,
        engine,
        content,
        strict_cold_item_ids=strict_cold_ids,
        user_history=user_history,
        policy_epoch=policy_epoch,
        item_batch_size=min(int(config.eval_batch_size), 128),
    )
    hot_export = None if export_dir is None else str(export_dir / f"{export_prefix}per_item_hot.csv")
    cold_export = None if export_dir is None else str(export_dir / f"{export_prefix}per_item_cold.csv")
    device = teacher.item_emb.weight.device
    hot, hot_count = _evaluate_frame(
        hot_frame,
        teacher=teacher,
        item_bank=item_bank,
        user_history=user_history,
        batch_size=config.eval_batch_size,
        device=device,
        export_item_metrics_path=hot_export,
    )
    cold, cold_count = _evaluate_frame(
        cold_frame,
        teacher=teacher,
        item_bank=item_bank,
        user_history=user_history,
        batch_size=config.eval_batch_size,
        device=device,
        export_item_metrics_path=cold_export,
    )
    return {
        "policy_mode": _policy_mode(policy_epoch),
        "cold_r10": 0.0 if cold is None else float(cold.get("R@10", 0.0)),
        "cold_n10": 0.0 if cold is None else float(cold.get("N@10", 0.0)),
        "hot_r10": 0.0 if hot is None else float(hot.get("R@10", 0.0)),
        "hot_n10": 0.0 if hot is None else float(hot.get("N@10", 0.0)),
        "overall_r10": _overall_metric(cold, cold_count, hot, hot_count, "R@10"),
        "overall_n10": _overall_metric(cold, cold_count, hot, hot_count, "N@10"),
        "cold_item_count": int(cold_count),
        "hot_item_count": int(hot_count),
    }


def _copy_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _save_stage_checkpoint(
    path: Path,
    *,
    stage: str,
    module: nn.Module,
    metadata: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "model_state": _copy_state_dict(module),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _item_id_batches(item_ids: Sequence[int], batch_size: int, *, seed: int) -> list[torch.Tensor]:
    ids = torch.tensor(sorted(int(item_id) for item_id in item_ids), dtype=torch.long)
    if ids.numel() < 1:
        return []
    order = torch.randperm(ids.numel(), generator=torch.Generator().manual_seed(int(seed)))
    shuffled = ids.index_select(0, order)
    return [shuffled[start:start + int(batch_size)] for start in range(0, shuffled.numel(), int(batch_size))]


def _generator_train_validation_ids(
    item_ids: FrozenSet[int], *, seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    ordered = sorted(
        (int(item_id) for item_id in item_ids),
        key=lambda item_id: hashlib.sha256(f"generator|{int(seed)}|{item_id}".encode("ascii")).hexdigest(),
    )
    if not ordered:
        raise ValueError("generator requires at least one H_G item")
    validation_count = 0
    if len(ordered) > 1 and float(validation_fraction) > 0.0:
        validation_count = min(
            len(ordered) - 1,
            max(1, int(round(len(ordered) * float(validation_fraction)))),
        )
    return ordered[validation_count:], ordered[:validation_count]


def train_clean_teacher(
    views: CleanStageViews,
    *,
    n_users: int,
    n_items: int,
    user_history: Mapping[int, set[int]],
    config: CleanRunConfig,
) -> tuple[CleanTeacher, dict[str, Any]]:
    """Fit behavioral IV embeddings using H_train and select only on H_val."""
    device = _resolve_device(config.device)
    teacher = CleanTeacher(n_users=n_users, n_items=n_items, emb_dim=config.emb_dim).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=float(config.teacher_lr))
    train_loader = DataLoader(
        InteractionDataset(views.teacher_train),
        batch_size=int(config.batch_size),
        shuffle=True,
        collate_fn=collate_interactions,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )
    best_state = _copy_state_dict(teacher)
    best_epoch = 0
    best_score = -float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.teacher_epochs) + 1):
        teacher.train()
        losses = []
        for batch, _ in train_loader:
            user_ids = batch["u"].to(device)
            item_ids = batch["i"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = teacher.ranking_loss(user_ids, item_ids)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        teacher.eval()
        val_metrics, _ = _evaluate_frame(
            views.teacher_val,
            teacher=teacher,
            item_bank=teacher.item_vectors().detach(),
            user_history=user_history,
            batch_size=config.eval_batch_size,
            device=device,
        )
        score = -float(sum(losses) / max(1, len(losses)))
        if val_metrics is not None:
            score = float(val_metrics.get("N@10", score))
        row = {
            "epoch": float(epoch),
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "validation_n10": float(score),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = _copy_state_dict(teacher)
    teacher.load_state_dict(best_state)
    teacher.eval()
    return teacher, {
        "selected_epoch": int(best_epoch),
        "selection_score": float(best_score),
        "history": history,
        "protocol": "behavioral_h_train_to_h_val",
    }


def _generator_objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(prediction, dim=1)
    tgt = F.normalize(target, dim=1)
    return F.mse_loss(pred, tgt) + (1.0 - (pred * tgt).sum(dim=1).mean())


def train_content_generator(
    teacher: CleanTeacher,
    content: torch.Tensor,
    generator_item_ids: FrozenSet[int],
    *,
    config: CleanRunConfig,
) -> tuple[ContentGenerator, dict[str, Any]]:
    """Map content to the frozen teacher space using H_G items only."""
    device = teacher.item_emb.weight.device
    train_ids, validation_ids = _generator_train_validation_ids(
        generator_item_ids,
        seed=config.seed,
        validation_fraction=config.generator_val_fraction,
    )
    generator = ContentGenerator(
        content_dim=int(content.size(1)), emb_dim=config.emb_dim, hidden_dim=config.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(config.generator_lr))
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    with torch.no_grad():
        targets = teacher.item_vectors().detach()
    best_state = _copy_state_dict(generator)
    best_epoch = 0
    best_score = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.generator_epochs) + 1):
        generator.train()
        losses = []
        for batch_ids in _item_id_batches(train_ids, config.batch_size, seed=config.seed + epoch):
            item_ids = batch_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _generator_objective(
                generator(frozen_content.index_select(0, item_ids)),
                targets.index_select(0, item_ids),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        generator.eval()
        validation_source = validation_ids if validation_ids else train_ids
        with torch.no_grad():
            eval_ids = torch.tensor(validation_source, dtype=torch.long, device=device)
            score = float(
                _generator_objective(
                    generator(frozen_content.index_select(0, eval_ids)),
                    targets.index_select(0, eval_ids),
                ).item()
            )
        history.append({
            "epoch": float(epoch),
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "validation_loss": score,
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = _copy_state_dict(generator)
    generator.load_state_dict(best_state)
    generator.eval()
    return generator, {
        "selected_epoch": int(best_epoch),
        "selection_score": float(best_score),
        "train_item_count": int(len(train_ids)),
        "validation_item_count": int(len(validation_ids)),
        "history": history,
        "protocol": "content_h_g_to_frozen_teacher_items",
    }


def _positive_users_by_item(frame: pd.DataFrame) -> dict[int, torch.Tensor]:
    positives: dict[int, set[int]] = {}
    for user_id, item_id in zip(frame["u_idx"].astype(int), frame["i_idx"].astype(int)):
        positives.setdefault(int(item_id), set()).add(int(user_id))
    return {
        item_id: torch.tensor(sorted(user_ids), dtype=torch.long)
        for item_id, user_ids in positives.items()
        if user_ids
    }


@torch.no_grad()
def _policy_proxy_l2(
    teacher: CleanTeacher,
    generator: ContentGenerator,
    engine: CleanUSIMEngine,
    *,
    item_ids: Sequence[int],
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    policy_epoch: int,
    batch_size: int,
) -> float:
    if not item_ids:
        return 0.0
    policy_mode = _policy_mode(policy_epoch)
    device = teacher.item_emb.weight.device
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    user_bank = teacher.user_vectors().detach()
    values = []
    for batch_ids in _item_id_batches(item_ids, batch_size, seed=0):
        ids = batch_ids.to(device)
        initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
        if policy_mode == "identity_generator":
            final_state = initial
        else:
            rollout = engine.rollout(
                initial,
                user_bank=user_bank,
                training=False,
                item_ids=ids,
                user_history=user_history,
            )
            final_state = rollout.final_state
        target = teacher.item_vectors(ids).detach()
        values.append((final_state - target).norm(dim=1).detach().cpu())
    return float(torch.cat(values).mean().item())


def _select_policy_row(
    rows: Sequence[dict[str, Any]], *, hot_retention_tolerance: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy selection requires at least one validation row")
    baseline = rows[0]
    hot_r_floor = float(baseline["hot_r10"]) - float(hot_retention_tolerance)
    hot_n_floor = float(baseline["hot_n10"]) - float(hot_retention_tolerance)
    eligible = [
        row for row in rows
        if float(row["hot_r10"]) >= hot_r_floor and float(row["hot_n10"]) >= hot_n_floor
    ]
    if not eligible:
        raise RuntimeError("no policy epoch preserves hot validation metrics")
    return max(
        eligible,
        key=lambda row: (
            float(row["cold_n10"]),
            float(row["cold_r10"]),
            float(row["overall_n10"]),
            -float(row["proxy_l2"]),
            -int(row["epoch"]),
        ),
    )


def train_clean_policy(
    teacher: CleanTeacher,
    generator: ContentGenerator,
    engine: CleanUSIMEngine,
    views: CleanStageViews,
    *,
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    validation_callback: Callable[[int], dict[str, float | int | str]],
    config: CleanRunConfig,
) -> tuple[CleanUSIMEngine, dict[str, Any], list[dict[str, Any]]]:
    """Optimize only actor/critic on P_train and select with P_val plus C_val."""
    device = teacher.item_emb.weight.device
    engine.policy.to(device)
    teacher.eval()
    generator.eval()
    for module in (teacher, generator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    user_bank = teacher.user_vectors().detach()
    positives = _positive_users_by_item(views.teacher_train)
    train_item_ids = sorted(int(item_id) for item_id in views.policy_train_item_ids)
    validation_item_ids = sorted(int(item_id) for item_id in views.policy_val_item_ids) or train_item_ids
    if not train_item_ids:
        raise ValueError("P_train must contain at least one pseudo-cold item")
    if any(item_id not in positives for item_id in train_item_ids):
        raise RuntimeError("each P_train item must have H_train positive users")
    optimizer = torch.optim.Adam(engine.policy.parameters(), lr=float(config.policy_lr))
    ppo = CleanRecPPO(
        engine.policy,
        replay_capacity=config.replay_capacity,
        replay_batch_size=config.replay_batch_size,
        gamma=config.ppo_gamma,
        clip_ratio=config.ppo_clip_ratio,
        value_weight=config.ppo_value_weight,
        terminal_value_weight=config.ppo_terminal_value_weight,
        entropy_weight=config.ppo_entropy_weight,
    )
    state_by_epoch: dict[int, dict[str, torch.Tensor]] = {0: _copy_state_dict(engine.policy)}
    rows: list[dict[str, Any]] = []

    def collect_validation(epoch: int, train_loss: float) -> None:
        engine.policy.eval()
        outer = validation_callback(epoch)
        policy_mode = _policy_mode(epoch)
        if outer.get("policy_mode") != policy_mode:
            raise RuntimeError("validation callback reported a policy mode inconsistent with its epoch")
        proxy = _policy_proxy_l2(
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
            "train_loss": float(train_loss),
            "proxy_l2": float(proxy),
            "policy_mode": policy_mode,
            **{name: float(value) for name, value in outer.items() if name != "policy_mode"},
        })

    collect_validation(0, 0.0)
    for epoch in range(1, int(config.policy_epochs) + 1):
        engine.policy.train()
        losses = []
        for batch_ids in _item_id_batches(train_item_ids, config.batch_size, seed=config.seed + epoch):
            ids = batch_ids.to(device)
            with torch.no_grad():
                initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
                target = teacher.item_vectors(ids).detach()
            rollout = engine.rollout(
                initial,
                user_bank=user_bank,
                training=True,
                target_emb=target,
                positive_user_ids=[positives[int(item_id)] for item_id in ids.detach().cpu().tolist()],
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
        state_by_epoch[epoch] = _copy_state_dict(engine.policy)
        collect_validation(epoch, float(sum(losses) / max(1, len(losses))))
    selected = _select_policy_row(rows, hot_retention_tolerance=config.hot_retention_tolerance)
    engine.policy.load_state_dict(state_by_epoch[int(selected["epoch"])])
    engine.policy.eval()
    return engine, selected, rows


def _partition_manifest_payload(partitions: CleanPartitions) -> dict[str, Any]:
    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        rows = frame[["u_idx", "i_idx"]].astype(int).to_csv(index=False).encode("utf-8")
        return {"rows": int(len(frame)), "sha256": hashlib.sha256(rows).hexdigest()}

    return {
        "h_train": summarize(partitions.h_train),
        "h_val": summarize(partitions.h_val),
        "h_test": summarize(partitions.h_test),
        "c_val": summarize(partitions.c_val),
        "c_test": summarize(partitions.c_test),
        "g_item_ids": sorted(int(item_id) for item_id in partitions.g_item_ids),
        "p_train_item_ids": sorted(int(item_id) for item_id in partitions.p_train_item_ids),
        "p_val_item_ids": sorted(int(item_id) for item_id in partitions.p_val_item_ids),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_validation_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "epoch", "policy_mode", "train_loss", "proxy_l2", "cold_r10", "cold_n10", "hot_r10", "hot_n10",
        "overall_r10", "overall_n10", "cold_item_count", "hot_item_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_clean_manifest(
    output_dir: Path,
    config: CleanRunConfig,
    partitions: CleanPartitions,
    *,
    stage_hashes: Mapping[str, str],
    selected_policy_epoch: int,
    test_loaded_after_policy_selection: bool,
    course_stats: Mapping[str, Any],
) -> None:
    _write_json(output_dir / "clean_manifest.json", {
        "route": "ckg_rl_usim_v32_clean",
        "seed": int(config.seed),
        "legacy_warm_checkpoint": None,
        "random_id_dropout": False,
        "main_candidate_mode": "legal_state_retrieval",
        "inference_oracle_access": False,
        "method_contract": build_clean_method_manifest(
            CleanMethodConfig(
                candidate_mode="legal_state_retrieval",
                max_policy_delta=float(config.max_delta),
                stability_anchor_count=0,
            )
        ),
        "teacher_protocol": "H_train_only_selected_on_H_val",
        "generator_protocol": "H_G_only_excludes_p_train_and_p_val",
        "policy_protocol": "P_train_only_teacher_reward_legal_candidates",
        "test_loaded_after_policy_selection": bool(test_loaded_after_policy_selection),
        "course_signal": dict(course_stats),
        "selected_policy_epoch": int(selected_policy_epoch),
        "selected_policy_mode": _policy_mode(selected_policy_epoch),
        "stage_hashes": dict(stage_hashes),
        "partitions": _partition_manifest_payload(partitions),
        "config": {
            name: (str(value) if isinstance(value, Path) else value)
            for name, value in config.__dict__.items()
        },
    })


def run_clean_pipeline(config: CleanRunConfig) -> dict[str, Any]:
    """Run the clean three-stage route; outer test is delayed until selection ends."""
    _validate_clean_route_environment()
    _validate_run_config(config)
    output_dir = _resolve_path(config.output_dir)
    checkpoint_dir = _resolve_path(config.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty clean output directory: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty clean checkpoint directory: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(config.seed)
    meta, content, train_df, val_df = load_clean_train_val_inputs(config)
    if int(meta.get("n_users", 0)) < 1:
        raise ValueError("meta must contain a positive n_users")
    empty_test = val_df.iloc[0:0].copy()
    partitions = build_clean_partitions(
        train_df,
        val_df,
        empty_test,
        n_items=int(content.size(0)),
        seed=config.seed,
        pseudo_ratio=config.pseudo_ratio,
        pseudo_val_fraction=config.pseudo_val_fraction,
        min_popularity=config.pseudo_min_popularity,
    )
    views = build_stage_views(partitions)
    user_history = build_user_seen(partitions.h_train)
    teacher, teacher_meta = train_clean_teacher(
        views,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        user_history=user_history,
        config=config,
    )
    teacher_hash = _save_stage_checkpoint(
        checkpoint_dir / "teacher.pt", stage="teacher", module=teacher, metadata=teacher_meta
    )
    generator, generator_meta = train_content_generator(
        teacher, content, views.generator_item_ids, config=config
    )
    generator_hash = _save_stage_checkpoint(
        checkpoint_dir / "generator.pt", stage="generator", module=generator, metadata=generator_meta
    )
    course_signal, course_stats = build_clean_course_signal(
        partitions.h_train,
        n_items=int(content.size(0)),
        config=config,
    )
    engine = create_clean_engine(config, course_signal=course_signal)
    warm_item_ids = partitions.g_item_ids | partitions.p_train_item_ids | partitions.p_val_item_ids

    def validation_callback(epoch: int) -> dict[str, float | int]:
        return evaluate_clean_route(
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

    engine, selected_policy, validation_rows = train_clean_policy(
        teacher,
        generator,
        engine,
        views,
        content=content,
        user_history=user_history,
        validation_callback=validation_callback,
        config=config,
    )
    policy_hash = _save_stage_checkpoint(
        checkpoint_dir / "policy.pt",
        stage="policy",
        module=engine.policy,
        metadata={
            "selected": selected_policy,
            "selected_policy_mode": _policy_mode(int(selected_policy["epoch"])),
            "validation_rows": validation_rows,
        },
    )
    _write_validation_rows(output_dir / "validation_epochs.csv", validation_rows)
    validation = dict(next(row for row in validation_rows if int(row["epoch"]) == int(selected_policy["epoch"])))

    # No test frame has been loaded above this point. It now becomes a one-shot final read.
    test_df = load_clean_test_inputs(config, partitions.h_train)
    partitions = attach_clean_test_rows(partitions, test_df)
    test_metrics = evaluate_clean_route(
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
    _write_json(output_dir / "clean_partition.json", _partition_manifest_payload(partitions))
    _write_json(output_dir / "final_metrics.json", {
        "validation": validation,
        "test": test_metrics,
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": _policy_mode(int(selected_policy["epoch"])),
    })
    write_clean_manifest(
        output_dir,
        config,
        partitions,
        stage_hashes={"teacher": teacher_hash, "generator": generator_hash, "policy": policy_hash},
        selected_policy_epoch=int(selected_policy["epoch"]),
        test_loaded_after_policy_selection=True,
        course_stats=course_stats,
    )
    return {
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": _policy_mode(int(selected_policy["epoch"])),
        "validation": validation,
        "test": test_metrics,
        "output_dir": str(output_dir),
    }


def _require_interaction_columns(frame: pd.DataFrame, name: str) -> None:
    required = {"u_idx", "i_idx"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required interaction columns: {missing}")


def _split_rows_by_item_ids(
    frame: pd.DataFrame,
    warm_item_ids: FrozenSet[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warm_mask = frame["i_idx"].astype(int).isin(warm_item_ids)
    return frame.loc[warm_mask].copy(), frame.loc[~warm_mask].copy()


def _pseudo_item_sets(
    train_df: pd.DataFrame,
    *,
    n_items: int,
    seed: int,
    pseudo_ratio: float,
    pseudo_val_fraction: float,
    min_popularity: int,
) -> tuple[FrozenSet[int], FrozenSet[int], FrozenSet[int]]:
    if not 0.0 <= float(pseudo_ratio) <= 1.0:
        raise ValueError("pseudo_ratio must be in [0, 1]")
    if not 0.0 <= float(pseudo_val_fraction) < 1.0:
        raise ValueError("pseudo_val_fraction must be in [0, 1)")
    if int(min_popularity) < 1:
        raise ValueError("min_popularity must be at least one")

    item_counts = train_df["i_idx"].astype(int).value_counts()
    warm_item_ids = frozenset(int(item_id) for item_id in item_counts.index)
    if not warm_item_ids:
        raise ValueError("H_train must contain at least one warm item")
    if min(warm_item_ids) < 0 or max(warm_item_ids) >= int(n_items):
        raise ValueError("H_train contains an item ID outside [0, n_items)")

    popularity = torch.zeros(int(n_items), dtype=torch.float32)
    for item_id, count in item_counts.items():
        popularity[int(item_id)] = float(count)
    eligible_count = sum(int(count) >= int(min_popularity) for count in item_counts.values)
    requested = int(torch.ceil(torch.tensor(float(eligible_count) * float(pseudo_ratio))).item())

    # The content generator must always retain at least one warm target item.
    selected_count = min(requested, max(0, len(warm_item_ids) - 1))
    if selected_count == 0:
        return frozenset(), frozenset(), warm_item_ids

    plan = build_pseudocold_plan(
        popularity,
        target_count=selected_count,
        min_popularity=float(min_popularity),
        cold_threshold=float(min_popularity),
        seed=int(seed),
    )
    selected = tuple(int(item_id) for item_id in plan.selected_item_ids)
    if len(selected) != selected_count:
        raise RuntimeError("pseudo-cold plan did not produce the requested number of items")

    # The predeclared P_val split is deterministic and leaves P_train nonempty.
    p_val_count = 0
    if len(selected) > 1 and pseudo_val_fraction > 0.0:
        p_val_count = min(
            len(selected) - 1,
            max(1, int(round(len(selected) * float(pseudo_val_fraction)))),
        )
    p_val_ids = frozenset(selected[:p_val_count])
    p_train_ids = frozenset(selected[p_val_count:])
    g_item_ids = warm_item_ids - p_train_ids - p_val_ids
    if not g_item_ids:
        raise RuntimeError("pseudo allocation left no warm item for content-generator training")
    return p_train_ids, p_val_ids, g_item_ids


def build_clean_partitions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_items: int,
    seed: int,
    pseudo_ratio: float,
    pseudo_val_fraction: float,
    min_popularity: int,
) -> CleanPartitions:
    """Partition a fixed strict-cold split without looking at outer held-out rows.

    `train_df` is H_train from the already-created outer split.  Its item IDs
    define warm items; every non-warm validation/test row is strict-cold.
    Pseudo item selection reads only `train_df` counts.
    """
    for name, frame in (("train_df", train_df), ("val_df", val_df), ("test_df", test_df)):
        _require_interaction_columns(frame, name)
    if int(n_items) < 1:
        raise ValueError("n_items must be positive")

    h_train = train_df.copy()
    warm_item_ids = frozenset(int(value) for value in h_train["i_idx"].astype(int).unique())
    h_val, c_val = _split_rows_by_item_ids(val_df, warm_item_ids)
    h_test, c_test = _split_rows_by_item_ids(test_df, warm_item_ids)
    p_train_ids, p_val_ids, g_item_ids = _pseudo_item_sets(
        h_train,
        n_items=int(n_items),
        seed=int(seed),
        pseudo_ratio=float(pseudo_ratio),
        pseudo_val_fraction=float(pseudo_val_fraction),
        min_popularity=int(min_popularity),
    )
    if (g_item_ids & p_train_ids) or (g_item_ids & p_val_ids) or (p_train_ids & p_val_ids):
        raise RuntimeError("clean warm item partitions are not disjoint")
    if g_item_ids | p_train_ids | p_val_ids != warm_item_ids:
        raise RuntimeError("clean warm item partitions do not cover H_train items")
    return CleanPartitions(
        h_train=h_train,
        h_val=h_val,
        h_test=h_test,
        c_val=c_val,
        c_test=c_test,
        g_item_ids=g_item_ids,
        p_train_item_ids=p_train_ids,
        p_val_item_ids=p_val_ids,
    )


def build_stage_views(partitions: CleanPartitions) -> CleanStageViews:
    """Return the only interaction views each clean stage is allowed to read."""
    generator_train = partitions.h_train.loc[
        partitions.h_train["i_idx"].astype(int).isin(partitions.g_item_ids)
    ].copy()
    policy_train = partitions.h_train.loc[
        partitions.h_train["i_idx"].astype(int).isin(partitions.p_train_item_ids)
    ].copy()
    policy_val = partitions.h_train.loc[
        partitions.h_train["i_idx"].astype(int).isin(partitions.p_val_item_ids)
    ].copy()
    return CleanStageViews(
        teacher_train=partitions.h_train.copy(),
        teacher_val=partitions.h_val.copy(),
        generator_train=generator_train,
        generator_item_ids=partitions.g_item_ids,
        policy_train=policy_train,
        policy_train_item_ids=partitions.p_train_item_ids,
        policy_val=policy_val,
        policy_val_item_ids=partitions.p_val_item_ids,
    )


def _validate_clean_route_environment() -> None:
    random_mask = os.environ.get("USIM_CLEAN_RANDOM_ID_DROPOUT", "0").strip()
    if random_mask not in {"0", "false", "False", "off", "OFF"}:
        raise ValueError("clean route forbids random ID dropout")
    candidate_mode = os.environ.get("USIM_CLEAN_CANDIDATE_MODE", "legal_state_retrieval").strip().lower()
    if candidate_mode != "legal_state_retrieval":
        raise ValueError("clean route requires legal_state_retrieval candidates")


def _parse_clean_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated clean CKG-RL V3.2 route")
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-course-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> CleanRunConfig:
    config = CleanRunConfig.for_seed(int(args.seed))
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


def _dry_run_clean_pipeline(config: CleanRunConfig) -> dict[str, Any]:
    _validate_clean_route_environment()
    _validate_run_config(config)
    meta, content, train_df, val_df = load_clean_train_val_inputs(config)
    partitions = build_clean_partitions(
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
        "seed": int(config.seed),
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "h_train_rows": int(len(partitions.h_train)),
        "h_val_rows": int(len(partitions.h_val)),
        "c_val_rows": int(len(partitions.c_val)),
        "g_item_count": int(len(partitions.g_item_ids)),
        "p_train_item_count": int(len(partitions.p_train_item_ids)),
        "p_val_item_count": int(len(partitions.p_val_item_ids)),
        "course_signal": bool(config.use_course_signal),
        "output_dir": str(_resolve_path(config.output_dir)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_clean_args(argv)
    config = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run_clean_pipeline(config), ensure_ascii=True, sort_keys=True))
        return 0
    _validate_clean_route_environment()
    result = run_clean_pipeline(config)
    if args.smoke:
        output_dir = Path(result["output_dir"])
        _write_json(output_dir / "smoke_report.json", {
            "status": "completed",
            "selected_policy_epoch": int(result["selected_policy_epoch"]),
            "target_free_inference": True,
            "test_selection_used": False,
        })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
