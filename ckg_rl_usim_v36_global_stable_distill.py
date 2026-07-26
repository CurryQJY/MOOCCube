"""V3.6 globally stable counterfactual action distillation.

This route keeps the V3.5 teacher, generator, legal action space, and
deployment rollout.  Offline action labels additionally penalize score drift
on a deterministic bank of users observed by the H_G generator stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank
import ckg_rl_usim_v35_action_distill as v35


V35_P_VAL_GAIN = 0.0047182762
V35_TRAIN_ACTION_AGREEMENT = 0.23880598
V35_TEACHER_SHA256 = "8816d9b3670ed0bd5d76b6336d250e15ad375b86a486824c86c3976874bc156a"
V35_GENERATOR_SHA256 = "198450f7c5eb4c14e0d60b932f5e6c243f8ca5f35673f4c516d52bc84e6ab082"


@dataclass(frozen=True)
class GlobalStableConfig(v35.ActionDistillConfig):
    """V3.6 controls; both method additions remain independently ablatable."""

    global_anchor_count: int = 128
    global_stability_weight: float = 10.0
    expert_action_fraction: float = 0.5

    @classmethod
    def for_seed(cls, seed: int) -> "GlobalStableConfig":
        base = v35.ActionDistillConfig.for_seed(int(seed))
        fields = dict(base.__dict__)
        fields.update({
            "output_dir": f"outputs/ckg_rl_usim_v36_global_stable_distill/seed{int(seed)}",
            "checkpoint_dir": f"checkpoints/ckg_rl_usim_v36_global_stable_distill/seed{int(seed)}",
            "global_anchor_count": 128,
            "global_stability_weight": 10.0,
            "expert_action_fraction": 0.5,
        })
        return cls(**fields)


def validate_global_stable_config(config: GlobalStableConfig) -> None:
    v35.validate_action_distill_config(config)
    if int(config.global_anchor_count) < 1:
        raise ValueError("global anchor count must be positive")
    if float(config.global_stability_weight) < 0.0:
        raise ValueError("global stability weight must be non-negative")
    if not 0.0 <= float(config.expert_action_fraction) <= 1.0:
        raise ValueError("expert action fraction must be in [0, 1]")


@dataclass(frozen=True)
class GlobalAnchorBank:
    """Immutable identities of the H_G users used for score-drift checks."""

    user_ids: tuple[int, ...]
    seed: int
    source_user_count: int

    def digest(self) -> str:
        payload = {
            "seed": int(self.seed),
            "source_user_count": int(self.source_user_count),
            "user_ids": [int(user_id) for user_id in self.user_ids],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def vectors(self, user_bank: torch.Tensor) -> torch.Tensor:
        users = torch.as_tensor(user_bank)
        if users.ndim != 2:
            raise ValueError("user_bank must be [users, dim]")
        if not self.user_ids:
            raise ValueError("global anchor bank must contain at least one user")
        ids = torch.tensor(self.user_ids, dtype=torch.long, device=users.device)
        if int(ids.min()) < 0 or int(ids.max()) >= users.size(0):
            raise ValueError("global anchor user ID is outside user_bank")
        return users.index_select(0, ids).detach()


def build_global_anchor_bank(
    h_g_rows: pd.DataFrame,
    *,
    seed: int,
    anchor_count: int,
) -> GlobalAnchorBank:
    """Select anchors only from H_G interaction users by stable SHA-256 order."""
    if int(anchor_count) < 1:
        raise ValueError("anchor_count must be positive")
    if "u_idx" not in h_g_rows.columns:
        raise ValueError("H_G rows must contain u_idx")
    source_users = sorted(int(user_id) for user_id in h_g_rows["u_idx"].dropna().astype(int).unique())
    if not source_users:
        raise ValueError("H_G rows must contain at least one user")

    def stable_key(user_id: int) -> str:
        value = f"v36-global-anchor:{int(seed)}:{int(user_id)}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    selected = tuple(sorted(source_users, key=lambda user_id: (stable_key(user_id), user_id))[:anchor_count])
    return GlobalAnchorBank(
        user_ids=selected,
        seed=int(seed),
        source_user_count=int(len(source_users)),
    )


@torch.no_grad()
def stable_counterfactual_action_targets(
    engine: rank.RankDistilledUSIMEngine,
    *,
    state: torch.Tensor,
    target_emb: torch.Tensor,
    user_bank: torch.Tensor,
    item_ids: torch.Tensor,
    anchor_vectors: torch.Tensor,
    action_temperature: float,
    global_stability_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return V3.5 local action labels with a global score-drift penalty."""
    if float(global_stability_weight) < 0.0:
        raise ValueError("global_stability_weight must be non-negative")
    current_state = torch.as_tensor(state)
    anchors = torch.as_tensor(
        anchor_vectors, dtype=current_state.dtype, device=current_state.device
    )
    if anchors.ndim != 2 or anchors.size(0) < 1 or anchors.size(1) != current_state.size(1):
        raise ValueError("anchor_vectors must be nonempty [anchors, dim] matching state")

    candidate_ids, _, local_utilities = v35.counterfactual_action_targets(
        engine,
        state=current_state,
        target_emb=target_emb,
        user_bank=user_bank,
        item_ids=item_ids,
        action_temperature=action_temperature,
    )
    users = torch.as_tensor(user_bank, dtype=current_state.dtype, device=current_state.device)
    targets = torch.as_tensor(target_emb, dtype=current_state.dtype, device=current_state.device)
    candidate_vectors = engine._candidate_vectors(users, candidate_ids)
    candidate_states = current_state.unsqueeze(1) + engine.step_size * candidate_vectors

    before_error = torch.matmul(current_state - targets, anchors.t()).pow(2).mean(dim=1)
    after_error = torch.matmul(
        candidate_states - targets.unsqueeze(1), anchors.t()
    ).pow(2).mean(dim=2)
    candidate_delta = after_error - before_error.unsqueeze(1)
    end_zero = torch.zeros(
        (current_state.size(0), 1), dtype=current_state.dtype, device=current_state.device
    )
    stability_delta = torch.cat((candidate_delta, end_zero), dim=1)
    utilities = local_utilities - float(global_stability_weight) * stability_delta
    utilities[:, -1] = 0.0
    if not torch.isfinite(utilities).all():
        raise RuntimeError("globally stable action utilities must be finite")
    target_probs = torch.softmax(utilities / float(action_temperature), dim=1)
    if not torch.isfinite(target_probs).all():
        raise RuntimeError("globally stable action targets must be finite")
    return (
        candidate_ids.detach(),
        target_probs.detach(),
        utilities.detach(),
        stability_delta.detach(),
    )


@dataclass(frozen=True)
class StableActionStep:
    """One stable-teacher label and the independently chosen rollout action."""

    state: torch.Tensor
    remaining_steps: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logit_bias: torch.Tensor
    target_probs: torch.Tensor
    utilities: torch.Tensor
    stability_delta: torch.Tensor
    actor_actions: torch.Tensor
    expert_actions: torch.Tensor
    rollout_actions: torch.Tensor
    expert_mask: torch.Tensor


def deterministic_expert_mask(
    item_ids: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    step: int,
    fraction: float,
) -> torch.Tensor:
    """Choose expert transitions without depending on process-global RNG."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("expert action fraction must be in [0, 1]")
    ids = torch.as_tensor(item_ids, dtype=torch.long).view(-1)
    if float(fraction) == 0.0:
        return torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
    if float(fraction) == 1.0:
        return torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    threshold = int(float(fraction) * (1 << 64))
    values = []
    for item_id in ids.detach().cpu().tolist():
        key = f"v36-expert:{int(seed)}:{int(epoch)}:{int(step)}:{int(item_id)}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big", signed=False)
        values.append(value < threshold)
    return torch.tensor(values, dtype=torch.bool, device=ids.device)


@torch.no_grad()
def collect_mixed_action_steps(
    engine: rank.RankDistilledUSIMEngine,
    *,
    initial_state: torch.Tensor,
    target_emb: torch.Tensor,
    item_ids: torch.Tensor,
    user_bank: torch.Tensor,
    user_history: Mapping[int, set[int]],
    anchor_vectors: torch.Tensor,
    action_temperature: float,
    global_stability_weight: float,
    expert_action_fraction: float,
    seed: int,
    epoch: int,
) -> list[StableActionStep]:
    """Label actor states and advance them with a deterministic actor/expert mix."""
    if not 0.0 <= float(expert_action_fraction) <= 1.0:
        raise ValueError("expert action fraction must be in [0, 1]")
    state = torch.as_tensor(initial_state).detach()
    targets = torch.as_tensor(target_emb, dtype=state.dtype, device=state.device).detach()
    active_ids = torch.as_tensor(item_ids, dtype=torch.long, device=state.device).view(-1)
    users = torch.as_tensor(user_bank, dtype=state.dtype, device=state.device)
    if targets.shape != state.shape or active_ids.numel() != state.size(0):
        raise ValueError("initial stable-action batch dimensions are inconsistent")

    steps: list[StableActionStep] = []
    for step_index in range(engine.max_steps):
        if state.size(0) == 0:
            break
        candidate_ids, target_probs, utilities, stability_delta = (
            stable_counterfactual_action_targets(
                engine,
                state=state,
                target_emb=targets,
                user_bank=users,
                item_ids=active_ids,
                anchor_vectors=anchor_vectors,
                action_temperature=action_temperature,
                global_stability_weight=global_stability_weight,
            )
        )
        candidate_vectors = engine._candidate_vectors(users, candidate_ids)
        candidate_bias = engine._candidate_logit_bias(candidate_ids, active_ids, user_history)
        remaining = torch.full(
            (state.size(0), 1),
            float(engine.max_steps - step_index) / float(engine.max_steps),
            dtype=state.dtype,
            device=state.device,
        )
        actor_actions, _, _, _ = engine.policy.action_value(
            state,
            remaining,
            candidate_vectors,
            candidate_logit_bias=candidate_bias,
            deterministic=True,
        )
        expert_actions = utilities.argmax(dim=1)
        expert_mask = deterministic_expert_mask(
            active_ids,
            seed=seed,
            epoch=epoch,
            step=step_index,
            fraction=expert_action_fraction,
        )
        rollout_actions = torch.where(expert_mask, expert_actions, actor_actions)
        steps.append(StableActionStep(
            state=state.detach(),
            remaining_steps=remaining.detach(),
            candidate_ids=candidate_ids.detach(),
            candidate_logit_bias=candidate_bias.detach(),
            target_probs=target_probs.detach(),
            utilities=utilities.detach(),
            stability_delta=stability_delta.detach(),
            actor_actions=actor_actions.detach(),
            expert_actions=expert_actions.detach(),
            rollout_actions=rollout_actions.detach(),
            expert_mask=expert_mask.detach(),
        ))

        end_action = candidate_vectors.size(1)
        safe_actions = rollout_actions.clamp_max(end_action - 1)
        selected_users = candidate_vectors[
            torch.arange(state.size(0), device=state.device), safe_actions
        ]
        keep = rollout_actions.ne(end_action)
        keep_ids = torch.nonzero(keep, as_tuple=False).view(-1)
        state = (state + engine.step_size * selected_users).index_select(0, keep_ids)
        targets = targets.index_select(0, keep_ids)
        active_ids = active_ids.index_select(0, keep_ids)
    return steps


def select_stable_policy_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select only by non-negative P_val rank gain, with identity eligible."""
    return v35.select_action_distill_policy_row(rows)


def _stable_action_step_statistics(steps: Sequence[StableActionStep]) -> dict[str, float]:
    if not steps:
        return {
            "train_examples": 0.0,
            "train_expert_gain": 0.0,
            "train_actor_gain": 0.0,
            "train_action_agreement": 0.0,
            "train_actor_end_rate": 0.0,
            "train_rollout_end_rate": 0.0,
            "train_expert_transition_rate": 0.0,
            "train_actor_stability_delta": 0.0,
        }
    utilities = torch.cat([step.utilities for step in steps], dim=0)
    stability_delta = torch.cat([step.stability_delta for step in steps], dim=0)
    actor_actions = torch.cat([step.actor_actions for step in steps], dim=0)
    rollout_actions = torch.cat([step.rollout_actions for step in steps], dim=0)
    expert_mask = torch.cat([step.expert_mask for step in steps], dim=0)
    actor_gain = utilities.gather(1, actor_actions.view(-1, 1)).view(-1)
    actor_drift = stability_delta.gather(1, actor_actions.view(-1, 1)).view(-1)
    end_action = utilities.size(1) - 1
    return {
        "train_examples": float(utilities.size(0)),
        "train_expert_gain": float(utilities.max(dim=1).values.mean().item()),
        "train_actor_gain": float(actor_gain.mean().item()),
        "train_action_agreement": float(
            actor_actions.eq(utilities.argmax(dim=1)).float().mean().item()
        ),
        "train_actor_end_rate": float(actor_actions.eq(end_action).float().mean().item()),
        "train_rollout_end_rate": float(rollout_actions.eq(end_action).float().mean().item()),
        "train_expert_transition_rate": float(expert_mask.float().mean().item()),
        "train_actor_stability_delta": float(actor_drift.mean().item()),
    }


def _global_anchor_drift(
    before_state: torch.Tensor,
    after_state: torch.Tensor,
    target_emb: torch.Tensor,
    anchor_vectors: torch.Tensor,
) -> torch.Tensor:
    anchors = torch.as_tensor(
        anchor_vectors, dtype=before_state.dtype, device=before_state.device
    )
    before = torch.matmul(before_state - target_emb, anchors.t()).pow(2).mean(dim=1)
    after = torch.matmul(after_state - target_emb, anchors.t()).pow(2).mean(dim=1)
    return after - before


@torch.no_grad()
def _stable_policy_diagnostics(
    teacher: clean.CleanTeacher,
    generator: clean.ContentGenerator,
    engine: rank.RankDistilledUSIMEngine,
    *,
    item_ids: Sequence[int],
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    anchor_vectors: torch.Tensor,
    policy_epoch: int,
    config: GlobalStableConfig,
) -> dict[str, float]:
    """Add stability diagnostics while preserving V3.5 P_val rank semantics."""
    diagnostics = rank._policy_rank_diagnostics(
        teacher,
        generator,
        engine,
        item_ids=item_ids,
        content=content,
        user_history=user_history,
        policy_epoch=policy_epoch,
        batch_size=config.batch_size,
    )
    if int(policy_epoch) == 0:
        return {
            **diagnostics,
            "p_val_anchor_drift": 0.0,
            "p_val_action_agreement": 0.0,
            "p_val_actor_end_rate": 0.0,
        }

    device = teacher.item_emb.weight.device
    users = teacher.user_vectors().detach()
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    drift_values: list[torch.Tensor] = []
    action_steps: list[StableActionStep] = []
    for batch_ids in clean._item_id_batches(item_ids, config.batch_size, seed=0):
        ids = batch_ids.to(device)
        initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
        target = teacher.item_vectors(ids).detach()
        rollout = engine.rollout(
            initial,
            user_bank=users,
            training=False,
            item_ids=ids,
            user_history=user_history,
        )
        final = F.normalize(rollout.final_state, dim=1)
        drift_values.append(
            _global_anchor_drift(initial, final, target, anchor_vectors).detach().cpu()
        )
        action_steps.extend(collect_mixed_action_steps(
            engine,
            initial_state=initial,
            target_emb=target,
            item_ids=ids,
            user_bank=users,
            user_history=user_history,
            anchor_vectors=anchor_vectors,
            action_temperature=config.action_temperature,
            global_stability_weight=config.global_stability_weight,
            expert_action_fraction=0.0,
            seed=config.seed,
            epoch=policy_epoch,
        ))
    action_stats = _stable_action_step_statistics(action_steps)
    return {
        **diagnostics,
        "p_val_anchor_drift": float(torch.cat(drift_values).mean().item()),
        "p_val_action_agreement": float(action_stats["train_action_agreement"]),
        "p_val_actor_end_rate": float(action_stats["train_actor_end_rate"]),
    }


def _stable_policy_mode(epoch: int) -> str:
    return "identity_generator" if int(epoch) == 0 else "global_stable_action_distill_rollout"


def train_global_stable_policy(
    teacher: clean.CleanTeacher,
    generator: clean.ContentGenerator,
    engine: rank.RankDistilledUSIMEngine,
    views: clean.CleanStageViews,
    *,
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    anchor_bank: GlobalAnchorBank,
    config: GlobalStableConfig,
) -> tuple[rank.RankDistilledUSIMEngine, dict[str, Any], list[dict[str, Any]]]:
    """Fit on P_train mixed states and select exclusively with P_val rank gain."""
    validate_global_stable_config(config)
    device = teacher.item_emb.weight.device
    engine.policy.to(device)
    teacher.eval()
    generator.eval()
    for module in (teacher, generator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    frozen_content = torch.as_tensor(content, dtype=torch.float32, device=device)
    user_bank = teacher.user_vectors().detach()
    anchor_vectors = anchor_bank.vectors(user_bank)
    train_item_ids = sorted(int(item_id) for item_id in views.policy_train_item_ids)
    validation_item_ids = sorted(int(item_id) for item_id in views.policy_val_item_ids) or train_item_ids
    if not train_item_ids:
        raise ValueError("P_train must contain at least one pseudo-cold item")
    engine.rank_panels.panel_for(torch.tensor(train_item_ids), device=device)
    engine.rank_panels.panel_for(torch.tensor(validation_item_ids), device=device)

    optimizer = torch.optim.Adam(engine.policy.parameters(), lr=float(config.policy_lr))
    state_by_epoch: dict[int, dict[str, torch.Tensor]] = {
        0: clean._copy_state_dict(engine.policy)
    }
    rows: list[dict[str, Any]] = []

    def collect_validation(epoch: int, train_values: Mapping[str, float]) -> None:
        engine.policy.eval()
        diagnostics = _stable_policy_diagnostics(
            teacher,
            generator,
            engine,
            item_ids=validation_item_ids,
            content=frozen_content,
            user_history=user_history,
            anchor_vectors=anchor_vectors,
            policy_epoch=epoch,
            config=config,
        )
        rows.append({
            "epoch": int(epoch),
            "policy_mode": _stable_policy_mode(epoch),
            **{name: float(value) for name, value in train_values.items()},
            **diagnostics,
        })

    stat_names = tuple(_stable_action_step_statistics(()).keys())
    collect_validation(0, {"train_loss": 0.0, **{name: 0.0 for name in stat_names}})

    for epoch in range(1, int(config.policy_epochs) + 1):
        losses: list[float] = []
        batch_stats: list[dict[str, float]] = []
        for batch_ids in clean._item_id_batches(
            train_item_ids, config.batch_size, seed=config.seed + epoch
        ):
            ids = batch_ids.to(device)
            with torch.no_grad():
                initial = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
                target = teacher.item_vectors(ids).detach()
            engine.policy.eval()
            steps = collect_mixed_action_steps(
                engine,
                initial_state=initial,
                target_emb=target,
                item_ids=ids,
                user_bank=user_bank,
                user_history=user_history,
                anchor_vectors=anchor_vectors,
                action_temperature=config.action_temperature,
                global_stability_weight=config.global_stability_weight,
                expert_action_fraction=config.expert_action_fraction,
                seed=config.seed,
                epoch=epoch,
            )
            if not steps:
                raise RuntimeError("stable action collection produced no trainable states")
            engine.policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([
                v35.action_distillation_loss(engine, step, user_bank=user_bank)
                for step in steps
            ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.policy.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            batch_stats.append(_stable_action_step_statistics(steps))

        def average_stat(name: str) -> float:
            return float(sum(row[name] for row in batch_stats) / max(1, len(batch_stats)))

        state_by_epoch[epoch] = clean._copy_state_dict(engine.policy)
        collect_validation(epoch, {
            "train_loss": float(sum(losses) / max(1, len(losses))),
            **{name: average_stat(name) for name in stat_names},
        })

    selected = select_stable_policy_row(rows)
    engine.policy.load_state_dict(state_by_epoch[int(selected["epoch"])])
    engine.policy.eval()
    return engine, selected, rows


def _write_global_anchor_manifest(
    output_dir: Path,
    anchor_bank: GlobalAnchorBank,
    *,
    requested_count: int,
) -> None:
    clean._write_json(output_dir / "global_anchor_manifest.json", {
        "source": "H_G_interaction_users_only",
        "ordering": "sha256_seed_user_id",
        "seed": int(anchor_bank.seed),
        "requested_user_count": int(requested_count),
        "source_user_count": int(anchor_bank.source_user_count),
        "selected_user_count": int(len(anchor_bank.user_ids)),
        "user_ids": [int(user_id) for user_id in anchor_bank.user_ids],
        "anchor_sha256": anchor_bank.digest(),
    })


def _source_hash_gate(stage_hashes: Mapping[str, str]) -> bool:
    return bool(
        str(stage_hashes.get("teacher")) == V35_TEACHER_SHA256
        and str(stage_hashes.get("generator")) == V35_GENERATOR_SHA256
    )


def _viability_gate(
    selected_policy: Mapping[str, Any],
    *,
    source_hash_gate_passed: bool,
) -> bool:
    return bool(
        source_hash_gate_passed
        and int(selected_policy["epoch"]) > 0
        and float(selected_policy["p_val_rank_gain"]) > V35_P_VAL_GAIN
        and float(selected_policy["train_action_agreement"]) > V35_TRAIN_ACTION_AGREEMENT
    )


def _write_v36_manifest(
    output_dir: Path,
    config: GlobalStableConfig,
    partitions: clean.CleanPartitions,
    *,
    anchor_bank: GlobalAnchorBank,
    stage_hashes: Mapping[str, str],
    selected_policy: Mapping[str, Any],
    course_stats: Mapping[str, Any],
) -> None:
    source_gate = _source_hash_gate(stage_hashes)
    clean._write_json(output_dir / "v36_manifest.json", {
        "route": "ckg_rl_usim_v36_global_stable_distill",
        "control_type": "v35_action_distillation_plus_expert_states_and_global_stability",
        "seed": int(config.seed),
        "legacy_warm_checkpoint": None,
        "random_id_dropout": False,
        "main_candidate_mode": "legal_state_retrieval",
        "inference_oracle_access": False,
        "teacher_protocol": "H_train_only_selected_on_H_val",
        "generator_protocol": "H_G_only_vector_teacher_reconstruction_v32_exact",
        "generator_rank_loss": False,
        "policy_protocol": "P_train_global_stable_action_distillation_mixed_states",
        "policy_optimizer": "globally_stable_action_distillation",
        "selection_protocol": "p_val_rank_gain_only",
        "outer_c_val_evaluated": False,
        "test_loaded": False,
        "global_anchor_source": "H_G_interaction_users_only",
        "global_anchor_sha256": anchor_bank.digest(),
        "global_anchor_count": int(config.global_anchor_count),
        "global_stability_weight": float(config.global_stability_weight),
        "expert_action_fraction": float(config.expert_action_fraction),
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": _stable_policy_mode(int(selected_policy["epoch"])),
        "selected_p_val_rank_gain": float(selected_policy["p_val_rank_gain"]),
        "selected_train_action_agreement": float(selected_policy["train_action_agreement"]),
        "v35_reference": {
            "p_val_rank_gain": V35_P_VAL_GAIN,
            "train_action_agreement": V35_TRAIN_ACTION_AGREEMENT,
            "teacher_sha256": V35_TEACHER_SHA256,
            "generator_sha256": V35_GENERATOR_SHA256,
        },
        "source_hash_gate_passed": source_gate,
        "viability_gate_passed": _viability_gate(
            selected_policy, source_hash_gate_passed=source_gate
        ),
        "course_signal": dict(course_stats),
        "stage_hashes": dict(stage_hashes),
        "partitions": clean._partition_manifest_payload(partitions),
        "config": v35._config_payload(config),
    })


def run_global_stable_pipeline(config: GlobalStableConfig) -> dict[str, Any]:
    """Run the V3.6 P_train/P_val screen without opening outer cold or test rows."""
    clean._validate_clean_route_environment()
    validate_global_stable_config(config)
    output_dir = clean._resolve_path(config.output_dir)
    checkpoint_dir = clean._resolve_path(config.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty V3.6 output directory: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty V3.6 checkpoint directory: {checkpoint_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    clean.setup_seed(config.seed)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    partitions = v35._build_p_only_partitions(
        train_df,
        val_df,
        n_items=int(content.size(0)),
        config=config,
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
        checkpoint_dir / "teacher.pt",
        stage="teacher",
        module=teacher,
        metadata=teacher_meta,
    )

    generator, generator_meta = clean.train_content_generator(
        teacher, content, views.generator_item_ids, config=config
    )
    generator_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "generator.pt",
        stage="generator",
        module=generator,
        metadata=generator_meta,
    )
    rank._write_csv(
        output_dir / "generator_vector_epochs.csv",
        generator_meta["history"],
        ("epoch", "train_loss", "validation_loss"),
    )

    panels = v35.build_action_distill_panels(
        teacher,
        views.teacher_train,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
        config=config,
    )
    v35._write_action_panel_manifest(
        output_dir,
        panels,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
    )
    anchor_bank = build_global_anchor_bank(
        views.generator_train,
        seed=config.seed,
        anchor_count=config.global_anchor_count,
    )
    _write_global_anchor_manifest(
        output_dir, anchor_bank, requested_count=config.global_anchor_count
    )
    course_signal, course_stats = clean.build_clean_course_signal(
        partitions.h_train,
        n_items=int(content.size(0)),
        config=config,
    )
    engine = rank.create_rank_distilled_engine(
        config, rank_panels=panels, course_signal=course_signal
    )
    engine, selected_policy, policy_rows = train_global_stable_policy(
        teacher,
        generator,
        engine,
        views,
        content=content,
        user_history=user_history,
        anchor_bank=anchor_bank,
        config=config,
    )
    policy_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "policy.pt",
        stage="policy",
        module=engine.policy,
        metadata={"selected": selected_policy, "validation_rows": policy_rows},
    )
    rank._write_csv(
        output_dir / "policy_stable_action_epochs.csv",
        policy_rows,
        (
            "epoch", "policy_mode", "train_loss", "train_examples",
            "train_expert_gain", "train_actor_gain", "train_action_agreement",
            "train_actor_end_rate", "train_rollout_end_rate",
            "train_expert_transition_rate", "train_actor_stability_delta",
            "p_val_initial_rank_kl", "p_val_final_rank_kl", "p_val_rank_gain",
            "p_val_anchor_drift", "p_val_action_agreement", "p_val_actor_end_rate",
        ),
    )
    selected_row = dict(next(
        row for row in policy_rows if int(row["epoch"]) == int(selected_policy["epoch"])
    ))
    stage_hashes = {
        "teacher": teacher_hash,
        "generator": generator_hash,
        "policy": policy_hash,
    }
    source_gate = _source_hash_gate(stage_hashes)
    selected_row["source_hash_gate_passed"] = source_gate
    selected_row["viability_gate_passed"] = _viability_gate(
        selected_row, source_hash_gate_passed=source_gate
    )
    clean._write_json(output_dir / "p_val_selected_metrics.json", selected_row)
    clean._write_json(
        output_dir / "policy_partition.json", clean._partition_manifest_payload(partitions)
    )
    _write_v36_manifest(
        output_dir,
        config,
        partitions,
        anchor_bank=anchor_bank,
        stage_hashes=stage_hashes,
        selected_policy=selected_row,
        course_stats=course_stats,
    )
    return {
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": _stable_policy_mode(int(selected_policy["epoch"])),
        "p_val": selected_row,
        "test_loaded": False,
        "output_dir": str(output_dir),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V3.6 globally stable action-distillation P-only screen"
    )
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
    parser.add_argument("--panel-size", type=int)
    parser.add_argument("--panel-positive-count", type=int)
    parser.add_argument("--panel-hard-count", type=int)
    parser.add_argument("--action-temperature", type=float)
    parser.add_argument("--global-anchor-count", type=int)
    parser.add_argument("--global-stability-weight", type=float)
    parser.add_argument("--expert-action-fraction", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-course-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> GlobalStableConfig:
    config = GlobalStableConfig.for_seed(int(args.seed))
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
        "panel_size": args.panel_size,
        "panel_positive_count": args.panel_positive_count,
        "panel_hard_count": args.panel_hard_count,
        "action_temperature": args.action_temperature,
        "global_anchor_count": args.global_anchor_count,
        "global_stability_weight": args.global_stability_weight,
        "expert_action_fraction": args.expert_action_fraction,
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


def _dry_run(config: GlobalStableConfig) -> dict[str, Any]:
    clean._validate_clean_route_environment()
    validate_global_stable_config(config)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    partitions = v35._build_p_only_partitions(
        train_df,
        val_df,
        n_items=int(content.size(0)),
        config=config,
    )
    views = clean.build_stage_views(partitions)
    anchor_bank = build_global_anchor_bank(
        views.generator_train,
        seed=config.seed,
        anchor_count=config.global_anchor_count,
    )
    return {
        "status": "dry_run_ok",
        "route": "ckg_rl_usim_v36_global_stable_distill",
        "seed": int(config.seed),
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "g_item_count": int(len(partitions.g_item_ids)),
        "p_train_item_count": int(len(partitions.p_train_item_ids)),
        "p_val_item_count": int(len(partitions.p_val_item_ids)),
        "global_anchor_count": int(len(anchor_bank.user_ids)),
        "global_anchor_sha256": anchor_bank.digest(),
        "global_stability_weight": float(config.global_stability_weight),
        "expert_action_fraction": float(config.expert_action_fraction),
        "generator_rank_loss": False,
        "test_loaded": False,
        "output_dir": str(clean._resolve_path(config.output_dir)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run(config), ensure_ascii=True, sort_keys=True))
        return 0
    result = run_global_stable_pipeline(config)
    if args.smoke:
        output_dir = Path(result["output_dir"])
        clean._write_json(output_dir / "smoke_report.json", {
            "status": "completed",
            "route": "ckg_rl_usim_v36_global_stable_distill",
            "selected_policy_epoch": int(result["selected_policy_epoch"]),
            "target_free_inference": True,
            "test_loaded": False,
        })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
