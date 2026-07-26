"""CKG-RL V3: USIM-consistent user-sequence simulation primitives.

The full static-runner adapter is added incrementally below these independently
testable primitives.  The module is intentionally separate from the V2 route
so historical results remain reproducible.
"""

from __future__ import annotations

import copy
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as base


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def apply_usim_transition(
    state: torch.Tensor,
    selected_user: torch.Tensor,
    *,
    action: torch.Tensor,
    done_before: torch.Tensor,
    end_action_index: int,
    step_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply USIM Eq. (6) while respecting explicit terminal actions."""
    action = action.to(device=state.device, dtype=torch.long).view(-1)
    done_before = done_before.to(device=state.device, dtype=torch.bool).view(-1)
    if action.numel() != state.size(0) or done_before.numel() != state.size(0):
        raise ValueError("action and done_before must contain one value per state row")
    if selected_user.shape != state.shape:
        raise ValueError("selected_user must have the same shape as state")

    active_user = (~done_before) & action.ne(int(end_action_index))
    proposed_state = state + float(step_size) * selected_user
    next_state = torch.where(active_user.view(-1, 1), proposed_state, state)
    done_after = done_before | action.eq(int(end_action_index))
    return next_state, done_after, active_user


def full_positive_recommendation_gain(
    before_state: torch.Tensor,
    after_state: torch.Tensor,
    teacher_item: torch.Tensor,
    positive_user_ids: Sequence[torch.Tensor],
    teacher_user_bank: torch.Tensor,
) -> torch.Tensor:
    """Return USIM's score-error improvement averaged over every positive user."""
    if before_state.shape != after_state.shape or before_state.shape != teacher_item.shape:
        raise ValueError("before_state, after_state, and teacher_item must share shape")
    if len(positive_user_ids) != before_state.size(0):
        raise ValueError("positive_user_ids must contain one user set per state row")

    gains: List[torch.Tensor] = []
    for row, users in enumerate(positive_user_ids):
        ids = torch.as_tensor(users, device=teacher_user_bank.device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            gains.append(before_state.new_zeros(()))
            continue
        user_vectors = teacher_user_bank.index_select(0, ids).to(
            device=before_state.device, dtype=before_state.dtype
        )
        target_scores = torch.matmul(user_vectors, teacher_item[row])
        before_error = torch.abs(torch.matmul(user_vectors, before_state[row]) - target_scores).mean()
        after_error = torch.abs(torch.matmul(user_vectors, after_state[row]) - target_scores).mean()
        gains.append(before_error - after_error)
    return torch.stack(gains, dim=0).view(-1, 1)


class EndAwareRecActorCritic(nn.Module):
    """Candidate-user policy with a separate learnable terminal logit."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.common = nn.Sequential(
            nn.Linear(self.embedding_dim + 1, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.actor_query = nn.Linear(int(hidden_dim), self.embedding_dim)
        self.user_key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.end_head = nn.Linear(int(hidden_dim), 1)
        self.critic_head = nn.Linear(int(hidden_dim), 1)

    def action_value(
        self,
        state: torch.Tensor,
        remaining_steps: torch.Tensor,
        candidates: torch.Tensor,
        *,
        candidate_logit_bias: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or candidates.ndim != 3:
            raise ValueError("state must be [batch, dim] and candidates must be [batch, count, dim]")
        if candidates.size(0) != state.size(0) or candidates.size(2) != state.size(1):
            raise ValueError("candidate shape must match state batch and embedding dimensions")
        remaining_steps = torch.as_tensor(
            remaining_steps, dtype=state.dtype, device=state.device
        ).view(-1, 1)
        if remaining_steps.size(0) != state.size(0):
            raise ValueError("remaining_steps must contain one value per state row")

        features = self.common(torch.cat([state, remaining_steps], dim=1))
        query = self.actor_query(features).unsqueeze(1)
        keys = self.user_key(candidates)
        user_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        if candidate_logit_bias is None:
            candidate_logit_bias = user_logits.new_zeros(user_logits.shape)
        else:
            candidate_logit_bias = torch.as_tensor(
                candidate_logit_bias,
                dtype=user_logits.dtype,
                device=user_logits.device,
            )
            if candidate_logit_bias.shape != user_logits.shape:
                raise ValueError("candidate_logit_bias must be [batch, candidate_count]")
        user_logits = user_logits + candidate_logit_bias
        logits = torch.cat([user_logits, self.end_head(features)], dim=1)
        distribution = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=1) if deterministic else distribution.sample()
        else:
            action = torch.as_tensor(action, device=state.device, dtype=torch.long).view(-1)
        return action, distribution.log_prob(action), self.critic_head(features), distribution.entropy()


class V3ReplayBuffer:
    """Bounded row-level replay storage with detached rollout behaviour data."""

    _REQUIRED_KEYS = (
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
            raise ValueError("capacity must be positive")
        self._records: deque[Dict[str, torch.Tensor]] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._records)

    def append(self, transition: Mapping[str, torch.Tensor]) -> None:
        missing = [key for key in self._REQUIRED_KEYS if key not in transition]
        if missing:
            raise KeyError(f"missing replay transition keys: {missing}")
        batch_size = int(torch.as_tensor(transition["action"]).shape[0])
        for key in self._REQUIRED_KEYS:
            if int(torch.as_tensor(transition[key]).shape[0]) != batch_size:
                raise ValueError(f"replay field {key!r} has incompatible batch size")
        for row in range(batch_size):
            self._records.append(
                {
                    key: torch.as_tensor(transition[key][row]).detach().cpu().clone()
                    for key in self._REQUIRED_KEYS
                }
            )

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self._records:
            raise RuntimeError("cannot sample an empty V3 replay buffer")
        count = min(int(batch_size), len(self._records))
        records = random.sample(list(self._records), count)
        return {
            key: torch.stack([record[key] for record in records], dim=0).to(device=device)
            for key in self._REQUIRED_KEYS
        }


def _v3_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _v3_env_float(name: str, default: float, *, allow_zero: bool = False) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < 0.0 or (not allow_zero and value == 0.0):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _v3_training_candidate_quotas(count: int) -> Tuple[int, int, int, int]:
    """Allocate train candidates to residual/positive/state/random support."""
    count = int(count)
    if count < 1:
        raise ValueError("candidate count must be positive")
    weights = (0.30, 0.30, 0.30, 0.10)
    quotas = [int(count * weight) for weight in weights]
    for index in range(count - sum(quotas)):
        quotas[index % len(quotas)] += 1
    return tuple(quotas)


class CKGRLV3USIM(base.Fast3FeedbackUSIM):
    """Current CKG-RL model with an explicit USIM-compatible V3 simulator."""

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.agent = EndAwareRecActorCritic(config.emb_dim)
        self.v3_target_agent = copy.deepcopy(self.agent).eval()
        for parameter in self.v3_target_agent.parameters():
            parameter.requires_grad_(False)
        self.v3_replay = V3ReplayBuffer(_v3_env_int("USIM_V3_REPLAY_CAPACITY", 4096))
        self._v3_train_item_users: Dict[int, torch.Tensor] | None = None
        self._v3_sync_target_on_next_forward = False
        self.v3_last_ppo_stats: Dict[str, float] = {}
        self._v3_last_candidate_stats: Dict[str, float] = {}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a warm teacher without accidentally importing its legacy actor."""
        has_v3_actor = any(str(key).startswith("v3_target_agent.") for key in state_dict)
        has_legacy_actor = any(str(key).startswith("agent.") for key in state_dict)
        if has_legacy_actor and not has_v3_actor:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not str(key).startswith("agent.")
            }
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def v3_candidate_count(self) -> int:
        requested = _v3_env_int("USIM_V3_CANDIDATES", int(getattr(self.cfg, "n_candidates", 20)))
        return min(requested, max(1, int(self.cfg.n_users)))

    @property
    def v3_step_size(self) -> float:
        return _v3_env_float("USIM_V3_STEP_SIZE", 0.05)

    @property
    def v3_step_penalty(self) -> float:
        return _v3_env_float("USIM_V3_STEP_PENALTY", 0.01, allow_zero=True)

    def _v3_train_users_for_items(self, item_idx, user_seen_items) -> List[torch.Tensor]:
        if self._v3_train_item_users is None:
            inverted: Dict[int, List[int]] = {}
            for user_id, seen_items in (user_seen_items or {}).items():
                for seen_item in seen_items:
                    inverted.setdefault(int(seen_item), []).append(int(user_id))
            self._v3_train_item_users = {
                item_id: torch.tensor(sorted(set(users)), dtype=torch.long)
                for item_id, users in inverted.items()
                if users
            }
        result: List[torch.Tensor] = []
        for value in torch.as_tensor(item_idx, dtype=torch.long).detach().cpu().tolist():
            result.append(self._v3_train_item_users.get(int(value), torch.empty(0, dtype=torch.long)))
        return result

    def _v3_sync_target_agent_if_needed(self) -> None:
        if self._v3_sync_target_on_next_forward:
            self.v3_target_agent.load_state_dict(self.agent.state_dict())
            self.v3_target_agent.eval()
            self._v3_sync_target_on_next_forward = False

    def forward(self, *args, **kwargs):
        self._v3_sync_target_agent_if_needed()
        return super().forward(*args, **kwargs)

    @staticmethod
    def _v3_topk_user_ids(
        query: torch.Tensor,
        user_bank_raw: torch.Tensor,
        count: int,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if user_bank_raw.size(0) < 1:
            raise ValueError("user_bank_raw must contain at least one user")
        limit = min(max(1, int(count)), int(user_bank_raw.size(0)))
        norm_query = F.normalize(query, dim=1)
        chunk_size = int(chunk_size or _v3_env_int("USIM_V3_RETRIEVAL_CHUNK", 8192))
        running_scores = None
        running_ids = None
        for start in range(0, int(user_bank_raw.size(0)), chunk_size):
            end = min(start + chunk_size, int(user_bank_raw.size(0)))
            chunk_scores = torch.matmul(
                norm_query,
                F.normalize(user_bank_raw[start:end], dim=1).t(),
            )
            chunk_ids = torch.arange(start, end, device=query.device, dtype=torch.long).view(1, -1)
            chunk_ids = chunk_ids.expand(query.size(0), -1)
            if running_scores is None:
                merged_scores = chunk_scores
                merged_ids = chunk_ids
            else:
                merged_scores = torch.cat([running_scores, chunk_scores], dim=1)
                merged_ids = torch.cat([running_ids, chunk_ids], dim=1)
            keep = min(limit, int(merged_scores.size(1)))
            running_scores, positions = torch.topk(merged_scores, k=keep, dim=1)
            running_ids = merged_ids.gather(1, positions)
        return running_ids

    @staticmethod
    def _v3_append_unique(
        destination: List[int],
        seen: set[int],
        source: Iterable[int] | torch.Tensor,
        *,
        quota: int,
        n_users: int,
    ) -> int:
        added = 0
        if quota < 1:
            return added
        values = torch.as_tensor(source, dtype=torch.long).view(-1)
        for candidate in values:
            candidate = int(candidate.detach().cpu().item())
            if 0 <= candidate < n_users and candidate not in seen:
                destination.append(candidate)
                seen.add(candidate)
                added += 1
                if added >= quota:
                    break
        return added

    @classmethod
    def _v3_append_random_unique(
        cls,
        destination: List[int],
        seen: set[int],
        *,
        sampled: torch.Tensor,
        fallback_start: int,
        quota: int,
        n_users: int,
    ) -> int:
        """Draw a bounded random supplement without permuting the whole user bank."""
        if quota < 1 or n_users < 1:
            return 0
        before = len(destination)
        cls._v3_append_unique(
            destination,
            seen,
            sampled,
            quota=quota,
            n_users=n_users,
        )
        remaining = int(quota) - (len(destination) - before)
        if remaining > 0:
            start = int(fallback_start) % n_users
            scan_limit = min(n_users, len(seen) + remaining)
            for offset in range(scan_limit):
                candidate = (start + offset) % n_users
                if candidate not in seen:
                    destination.append(candidate)
                    seen.add(candidate)
                    if len(destination) - before >= quota:
                        break
        return len(destination) - before

    def _v3_course_logit_bias(
        self,
        candidate_ids: torch.Tensor,
        *,
        item_idx: torch.Tensor | None,
        target_pop: torch.Tensor | None,
        user_seen_items,
    ) -> torch.Tensor:
        beta = float(getattr(self.cfg, "feedback_course_sample_beta", 0.0))
        zero = torch.zeros(candidate_ids.shape, dtype=torch.float32, device=candidate_ids.device)
        if beta <= 0.0 or item_idx is None:
            return zero
        fit_score = self._compute_candidate_course_fit(
            candidate_ids,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        fit_score = torch.as_tensor(fit_score, dtype=torch.float32, device=candidate_ids.device)
        if fit_score.shape != candidate_ids.shape:
            raise ValueError("course fit must match candidate_ids shape")
        fit_score = torch.nan_to_num(fit_score, nan=0.0, posinf=0.0, neginf=0.0)
        return beta * fit_score / fit_score.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

    def _v3_build_candidates(
        self,
        current_h: torch.Tensor,
        user_bank_raw: torch.Tensor,
        *,
        training: bool,
        target_emb: torch.Tensor | None,
        positive_user_ids: Sequence[torch.Tensor] | None,
        item_idx: torch.Tensor | None,
        target_pop: torch.Tensor | None,
        user_seen_items,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = self.v3_candidate_count
        batch_size = current_h.size(0)
        state_top = self._v3_topk_user_ids(current_h, user_bank_raw, count)
        residual_top = None
        if training:
            if target_emb is None or positive_user_ids is None:
                raise ValueError("V3 training candidates require target_emb and positive_user_ids")
            residual_top = self._v3_topk_user_ids(target_emb.detach() - current_h, user_bank_raw, count)

        rows: List[List[int]] = []
        n_users = int(user_bank_raw.size(0))
        state_top_cpu = state_top.detach().cpu()
        residual_top_cpu = residual_top.detach().cpu() if residual_top is not None else None
        random_draws = (
            torch.randint(
                0,
                n_users,
                (batch_size, max(8, count * 8)),
                device="cpu",
            )
            if training
            else None
        )
        source_totals = {"residual": 0, "positive": 0, "state": 0, "random": 0}
        valid_candidate_count = 0
        residual_quota, positive_quota, state_quota, random_quota = _v3_training_candidate_quotas(count)
        for row in range(batch_size):
            unique: List[int] = []
            seen = set()
            row_counts = {"residual": 0, "positive": 0, "state": 0, "random": 0}
            if training:
                row_counts["residual"] += self._v3_append_unique(
                    unique,
                    seen,
                    residual_top_cpu[row],
                    quota=residual_quota,
                    n_users=n_users,
                )
                positive_source = torch.as_tensor(
                    positive_user_ids[row], dtype=torch.long
                ).view(-1)
                row_counts["positive"] += self._v3_append_unique(
                    unique,
                    seen,
                    positive_source,
                    quota=positive_quota,
                    n_users=n_users,
                )
                row_counts["state"] += self._v3_append_unique(
                    unique,
                    seen,
                    state_top_cpu[row],
                    quota=state_quota,
                    n_users=n_users,
                )
                row_counts["random"] += self._v3_append_random_unique(
                    unique,
                    seen,
                    sampled=random_draws[row],
                    fallback_start=int(random_draws[row, 0].item()),
                    quota=random_quota,
                    n_users=n_users,
                )
                if len(unique) < count:
                    row_counts["state"] += self._v3_append_unique(
                        unique,
                        seen,
                        state_top_cpu[row],
                        quota=count - len(unique),
                        n_users=n_users,
                    )
                if len(unique) < count:
                    row_counts["residual"] += self._v3_append_unique(
                        unique,
                        seen,
                        residual_top_cpu[row],
                        quota=count - len(unique),
                        n_users=n_users,
                    )
                if len(unique) < count:
                    row_counts["random"] += self._v3_append_random_unique(
                        unique,
                        seen,
                        sampled=random_draws[row],
                        fallback_start=int(random_draws[row, 0].item()),
                        quota=count - len(unique),
                        n_users=n_users,
                    )
            else:
                self._v3_append_unique(
                    unique,
                    seen,
                    state_top_cpu[row],
                    quota=count,
                    n_users=n_users,
                )
            if not unique:
                unique = [0]
            valid_candidate_count += len(unique)
            while len(unique) < count:
                unique.append(unique[-1])
            for source_name, source_count in row_counts.items():
                source_totals[source_name] += source_count
            rows.append(unique)
        candidate_ids = torch.tensor(rows, dtype=torch.long, device=current_h.device)
        candidates = user_bank_raw.index_select(0, candidate_ids.reshape(-1)).view(
            batch_size, count, -1
        )
        candidate_logit_bias = self._v3_course_logit_bias(
            candidate_ids,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        denominator = max(1, valid_candidate_count)
        self._v3_last_candidate_stats = {
            "v3_course_logit_bias_abs": float(candidate_logit_bias.abs().mean().detach().item()),
            "v3_train_residual_share": float(source_totals["residual"] / denominator) if training else 0.0,
            "v3_train_positive_share": float(source_totals["positive"] / denominator) if training else 0.0,
            "v3_train_state_share": float(source_totals["state"] / denominator) if training else 0.0,
            "v3_train_random_share": float(source_totals["random"] / denominator) if training else 0.0,
        }
        return candidates, candidate_ids, candidate_logit_bias

    def run_usim_episode(
        self,
        init_item_emb: torch.Tensor,
        target_emb: torch.Tensor | None = None,
        user_bank_raw=None,
        item_idx: torch.Tensor | None = None,
        target_pop: torch.Tensor | None = None,
        user_seen_items=None,
        **_,
    ):
        training = bool(self.training and target_emb is not None)
        if self._original_v2_teacher_user_emb is not None:
            user_bank_raw = self._original_v2_teacher_user_emb.to(self.device)
        elif isinstance(user_bank_raw, tuple):
            user_bank_raw = user_bank_raw[0]
        if user_bank_raw is None:
            user_bank_raw = self._build_user_bank_raw()[0]
        user_bank_raw = torch.as_tensor(
            user_bank_raw, dtype=init_item_emb.dtype, device=self.device
        ).detach()
        if item_idx is None:
            raise ValueError("V3 rollout requires item_idx")
        item_idx = torch.as_tensor(item_idx, dtype=torch.long, device=self.device).view(-1)
        if item_idx.numel() != init_item_emb.size(0):
            raise ValueError("item_idx must contain one value per initial state")

        if training:
            target_emb = torch.as_tensor(target_emb, dtype=init_item_emb.dtype, device=self.device).detach()
            if target_emb.shape != init_item_emb.shape:
                raise ValueError("target_emb must match init_item_emb during V3 training")
            positive_user_ids = self._v3_train_users_for_items(item_idx, user_seen_items)
        else:
            positive_user_ids = None

        current_h = init_item_emb.clone()
        done = torch.zeros(current_h.size(0), dtype=torch.bool, device=self.device)
        trajectory = {
            "states": [],
            "next_states": [],
            "candidate_ids": [],
            "candidate_logit_bias": [],
            "actions": [],
            "rewards": [],
            "done": [],
            "old_log_probs": [],
            "remaining_steps": [],
            "next_remaining_steps": [],
            "terminal_states": [],
            "terminal_remaining_steps": [],
        }
        active_total = 0
        end_total = 0
        reward_emb_total = 0.0
        reward_rec_total = 0.0
        reward_ckg_total = 0.0
        course_logit_bias_abs_total = 0.0
        train_residual_share_total = 0.0
        train_positive_share_total = 0.0
        train_state_share_total = 0.0
        train_random_share_total = 0.0

        for step in range(int(self.cfg.usim_steps)):
            remaining = torch.full(
                (current_h.size(0), 1),
                float(int(self.cfg.usim_steps) - step),
                dtype=current_h.dtype,
                device=self.device,
            )
            candidates, candidate_ids, candidate_logit_bias = self._v3_build_candidates(
                current_h,
                user_bank_raw,
                training=training,
                target_emb=target_emb if training else None,
                positive_user_ids=positive_user_ids if training else None,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            candidate_stats = self._v3_last_candidate_stats
            course_logit_bias_abs_total += float(candidate_stats.get("v3_course_logit_bias_abs", 0.0))
            train_residual_share_total += float(candidate_stats.get("v3_train_residual_share", 0.0))
            train_positive_share_total += float(candidate_stats.get("v3_train_positive_share", 0.0))
            train_state_share_total += float(candidate_stats.get("v3_train_state_share", 0.0))
            train_random_share_total += float(candidate_stats.get("v3_train_random_share", 0.0))
            if candidate_logit_bias is None:
                candidate_logit_bias = current_h.new_zeros(
                    (current_h.size(0), candidates.size(1))
                )
            action, log_prob, value, entropy = self.agent.action_value(
                current_h,
                remaining,
                candidates,
                candidate_logit_bias=candidate_logit_bias,
                deterministic=not training,
            )
            safe_action = action.clamp_max(candidates.size(1) - 1)
            selected_user = candidates[
                torch.arange(current_h.size(0), device=self.device), safe_action
            ]
            next_h, done_after, active_user = apply_usim_transition(
                current_h,
                selected_user,
                action=action,
                done_before=done,
                end_action_index=candidates.size(1),
                step_size=self.v3_step_size,
            )
            active_before = ~done
            reward = current_h.new_zeros((current_h.size(0), 1))
            terminal_state = current_h.detach()
            if training:
                emb_gain = (current_h - target_emb).norm(dim=1, keepdim=True) - (
                    next_h - target_emb
                ).norm(dim=1, keepdim=True)
                rec_gain = full_positive_recommendation_gain(
                    current_h,
                    next_h,
                    target_emb,
                    positive_user_ids,
                    user_bank_raw,
                )
                reward = (emb_gain + rec_gain - self.v3_step_penalty) * active_before.view(-1, 1)
                reward_emb_total += float(emb_gain.mean().detach().item())
                reward_rec_total += float(rec_gain.mean().detach().item())
                terminal_state = target_emb.detach()
                if bool(getattr(self.cfg, "use_course_reward", False)):
                    selected_user_ids = candidate_ids[
                        torch.arange(current_h.size(0), device=self.device), safe_action
                    ]
                    course_terms = self._compute_course_reward_terms(
                        selected_user_ids,
                        item_idx=item_idx,
                        target_pop=target_pop,
                        user_seen_items=user_seen_items,
                    )
                    ckg_reward = (
                        float(getattr(self.cfg, "feedback_course_concept_weight", 0.0))
                        * course_terms["concept_bonus"]
                        - float(getattr(self.cfg, "feedback_course_prereq_weight", 0.0))
                        * course_terms["prereq_gap"]
                        - float(getattr(self.cfg, "feedback_course_difficulty_weight", 0.0))
                        * course_terms["difficulty_gap"]
                        - float(getattr(self.cfg, "feedback_course_redundant_weight", 0.0))
                        * course_terms["redundant"]
                    ) * active_user.view(-1, 1)
                    reward = reward + ckg_reward
                    reward_ckg_total += float(ckg_reward.mean().detach().item())

            next_remaining = torch.clamp(remaining - 1.0, min=0.0)
            trajectory["states"].append(current_h.detach())
            trajectory["next_states"].append(next_h.detach())
            trajectory["candidate_ids"].append(candidate_ids.detach())
            trajectory["candidate_logit_bias"].append(candidate_logit_bias.detach())
            trajectory["actions"].append(action.detach())
            trajectory["rewards"].append(reward.detach())
            trajectory["done"].append(done_after.detach())
            trajectory["old_log_probs"].append(log_prob.detach())
            trajectory["remaining_steps"].append(remaining.detach())
            trajectory["next_remaining_steps"].append(next_remaining.detach())
            trajectory["terminal_states"].append(terminal_state)
            trajectory["terminal_remaining_steps"].append(next_remaining.detach())
            active_total += int(active_before.sum().detach().item())
            end_total += int((active_before & action.eq(candidates.size(1))).sum().detach().item())
            current_h, done = next_h, done_after

        steps = max(1, int(self.cfg.usim_steps))
        stats = {
            "steps": steps,
            "v3_end_rate": float(end_total / max(1, active_total)),
            "v3_active_steps": float(active_total / max(1, init_item_emb.size(0))),
            "v3_embedding_reward": float(reward_emb_total / steps),
            "v3_recommendation_reward": float(reward_rec_total / steps),
            "v3_course_reward": float(reward_ckg_total / steps),
            "v3_rollout_delta_l2": float((current_h - init_item_emb).norm(dim=1).mean().detach().item()),
            "v3_course_logit_bias_abs": float(course_logit_bias_abs_total / steps),
            "v3_train_residual_share": float(train_residual_share_total / steps),
            "v3_train_positive_share": float(train_positive_share_total / steps),
            "v3_train_state_share": float(train_state_share_total / steps),
            "v3_train_random_share": float(train_random_share_total / steps),
        }
        return current_h, trajectory, stats

    def _v3_replay_user_bank(self) -> torch.Tensor:
        if self._original_v2_teacher_user_emb is not None:
            return self._original_v2_teacher_user_emb.to(self.device).detach()
        built = self._build_user_bank_raw()
        if isinstance(built, tuple):
            built = built[0]
        return torch.as_tensor(built, device=self.device).detach()

    def compute_ppo_loss(self, trajectory):
        if not trajectory.get("rewards"):
            return next(self.parameters()).sum() * 0.0
        for step in range(len(trajectory["rewards"])):
            self.v3_replay.append(
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

        sample = self.v3_replay.sample(
            _v3_env_int("USIM_V3_REPLAY_BATCH_SIZE", 512), self.device
        )
        user_bank = self._v3_replay_user_bank().to(dtype=sample["state"].dtype)
        candidate_ids = sample["candidate_ids"].long()
        candidates = user_bank.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0), candidate_ids.size(1), -1
        )
        action, new_log_prob, value, entropy = self.agent.action_value(
            sample["state"],
            sample["remaining_steps"],
            candidates,
            candidate_logit_bias=sample["candidate_logit_bias"],
            action=sample["action"],
        )
        del action
        with torch.no_grad():
            dummy_action = torch.zeros(
                sample["state"].size(0), dtype=torch.long, device=self.device
            )
            _, _, target_value, _ = self.v3_target_agent.action_value(
                sample["state"],
                sample["remaining_steps"],
                candidates,
                candidate_logit_bias=sample["candidate_logit_bias"],
                action=dummy_action,
            )
            _, _, target_next_value, _ = self.v3_target_agent.action_value(
                sample["next_state"],
                sample["next_remaining_steps"],
                candidates,
                candidate_logit_bias=sample["candidate_logit_bias"],
                action=dummy_action,
            )
            done = sample["done"].float().view(-1, 1)
            gamma = _v3_env_float("USIM_V3_GAMMA", 0.99)
            td_target = sample["reward"] + gamma * (1.0 - done) * target_next_value
            advantage = (td_target - target_value).detach().view(-1)

        old_log_prob = sample["old_log_prob"].detach().view(-1)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clip = _v3_env_float("USIM_V3_PPO_CLIP", 0.20)
        actor_loss = -torch.minimum(
            ratio * advantage,
            torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
        ).mean()
        critic_loss = F.mse_loss(value, td_target.detach())
        _, _, terminal_value, _ = self.agent.action_value(
            sample["terminal_state"],
            sample["terminal_remaining_steps"],
            candidates,
            candidate_logit_bias=sample["candidate_logit_bias"],
            action=torch.zeros(sample["state"].size(0), dtype=torch.long, device=self.device),
        )
        terminal_value_loss = terminal_value.pow(2).mean()
        value_weight = _v3_env_float("USIM_V3_VALUE_WEIGHT", 0.5, allow_zero=True)
        terminal_weight = _v3_env_float("USIM_V3_TERMINAL_VALUE_WEIGHT", 1.0, allow_zero=True)
        entropy_weight = _v3_env_float("USIM_V3_ENTROPY_WEIGHT", 0.01, allow_zero=True)
        total_loss = (
            actor_loss
            + value_weight * critic_loss
            + terminal_weight * terminal_value_loss
            - entropy_weight * entropy.mean()
        )
        self.v3_last_ppo_stats = {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "terminal_value_loss": float(terminal_value_loss.detach().item()),
            "entropy": float(entropy.mean().detach().item()),
            "replay_size": float(len(self.v3_replay)),
        }
        self._v3_sync_target_on_next_forward = True
        return total_loss


_BASE_STATIC_EXPERIMENT = base.run_static_experiment


def _write_v3_engine_manifest() -> None:
    output_root = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
    if not output_root:
        return
    destination = Path(output_root) / "v3_engine_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = _v3_env_int("USIM_V3_CANDIDATES", 20)
    residual_quota, positive_quota, state_quota, random_quota = _v3_training_candidate_quotas(
        candidate_count
    )
    payload = {
        "route": "ckg_rl_usim_v3",
        "engine_revision": os.environ.get("USIM_V3_ENGINE_REVISION", "v3.1"),
        "inference_oracle_access": False,
        "step_size": _v3_env_float("USIM_V3_STEP_SIZE", 0.05),
        "step_penalty": _v3_env_float("USIM_V3_STEP_PENALTY", 0.01, allow_zero=True),
        "candidate_count": candidate_count,
        "train_candidate_quotas": {
            "residual": residual_quota,
            "positive": positive_quota,
            "state": state_quota,
            "random": random_quota,
        },
        "course_policy_integration": "normalized_observable_course_fit_logit_bias",
        "replay_capacity": _v3_env_int("USIM_V3_REPLAY_CAPACITY", 4096),
        "replay_batch_size": _v3_env_int("USIM_V3_REPLAY_BATCH_SIZE", 512),
        "teacher_checkpoint": os.environ.get("USIM_FB_INIT_CKPT_DIR", ""),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_static_experiment(*args, **kwargs):
    if os.environ.get("USIM_V3_CORE", "0") != "1":
        raise RuntimeError("CKG-RL V3 requires USIM_V3_CORE=1")
    previous_model_class = base.Fast3FeedbackUSIM
    base.Fast3FeedbackUSIM = CKGRLV3USIM
    try:
        result = _BASE_STATIC_EXPERIMENT(*args, **kwargs)
        _write_v3_engine_manifest()
        return result
    finally:
        base.Fast3FeedbackUSIM = previous_model_class


def main():
    previous_static_experiment = base.run_static_experiment
    base.run_static_experiment = run_static_experiment
    try:
        return base.main()
    finally:
        base.run_static_experiment = previous_static_experiment


if __name__ == "__main__":
    base.setup_seed(int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025"))))
    main()
