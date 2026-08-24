"""V3.5 target-free USIM with counterfactual action distillation.

The V3.2 teacher and vector generator remain fixed.  A frozen teacher creates
soft one-step action labels only for pseudo-cold warm items during training;
deployment uses the same legal retrieval rollout without teacher targets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, FrozenSet, Mapping, Sequence

import torch
import torch.nn.functional as F

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank


@dataclass(frozen=True)
class ActionDistillConfig(rank.RankDistillRunConfig):
    """V3.5 configuration with vector-only generation and soft action labels."""

    generator_rank_weight: float = 0.0
    action_temperature: float = 0.005

    @classmethod
    def for_seed(cls, seed: int) -> "ActionDistillConfig":
        base = rank.RankDistillRunConfig.for_seed(int(seed))
        fields = dict(base.__dict__)
        fields.update({
            "output_dir": f"outputs/ckg_rl_usim_v35_action_distill/seed{int(seed)}",
            "checkpoint_dir": f"checkpoints/ckg_rl_usim_v35_action_distill/seed{int(seed)}",
            "generator_rank_weight": 0.0,
            "action_temperature": 0.005,
        })
        return cls(**fields)


@dataclass(frozen=True)
class ActionDistillStep:
    """One active, target-labeled policy state collected without gradients."""

    state: torch.Tensor
    remaining_steps: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logit_bias: torch.Tensor
    target_probs: torch.Tensor
    utilities: torch.Tensor
    actor_actions: torch.Tensor


def validate_action_distill_config(config: ActionDistillConfig) -> None:
    """Reject configurations that would change the controlled V3.2 generator."""
    rank._validate_rank_distill_config(config)
    if abs(float(config.generator_rank_weight)) > 1e-12:
        raise ValueError("action distillation requires generator rank weight equal to zero")
    if float(config.action_temperature) <= 0.0:
        raise ValueError("action temperature must be positive")


def build_action_distill_panels(
    teacher: clean.CleanTeacher,
    h_train,
    *,
    p_train_item_ids: FrozenSet[int],
    p_val_item_ids: FrozenSet[int],
    config: ActionDistillConfig,
) -> rank.RankPanels:
    """Build panels only for pseudo-warm policy supervision and validation."""
    item_ids = frozenset(int(item_id) for item_id in p_train_item_ids | p_val_item_ids)
    if not item_ids:
        raise ValueError("action distillation requires at least one policy pseudo item")
    return rank.build_rank_panels(
        teacher,
        h_train,
        item_ids=item_ids,
        seed=config.seed,
        panel_size=config.panel_size,
        positive_count=config.panel_positive_count,
        hard_count=config.panel_hard_count,
    )


def _build_p_only_partitions(
    train_df,
    validation_df,
    *,
    n_items: int,
    config: ActionDistillConfig,
) -> clean.CleanPartitions:
    """Materialize only H_val from the mixed validation file for this screen."""
    warm_item_ids = frozenset(int(item_id) for item_id in train_df["i_idx"].astype(int).unique())
    h_val = validation_df.loc[
        validation_df["i_idx"].astype(int).isin(warm_item_ids)
    ].copy()
    return clean.build_clean_partitions(
        train_df,
        h_val,
        h_val.iloc[0:0].copy(),
        n_items=int(n_items),
        seed=config.seed,
        pseudo_ratio=config.pseudo_ratio,
        pseudo_val_fraction=config.pseudo_val_fraction,
        min_popularity=config.pseudo_min_popularity,
    )


@torch.no_grad()
def counterfactual_action_targets(
    engine: rank.RankDistilledUSIMEngine,
    *,
    state: torch.Tensor,
    target_emb: torch.Tensor,
    user_bank: torch.Tensor,
    item_ids: torch.Tensor,
    action_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legal candidate IDs, soft target probabilities, and utilities.

    The final utility column corresponds to the legal ``END`` action.  It has
    zero one-step rank gain because it leaves the current state unchanged.
    """
    if float(action_temperature) <= 0.0:
        raise ValueError("action temperature must be positive")
    current_state = torch.as_tensor(state)
    if current_state.ndim != 2:
        raise ValueError("state must be [batch, dim]")
    device = current_state.device
    targets = torch.as_tensor(target_emb, dtype=current_state.dtype, device=device)
    if targets.shape != current_state.shape:
        raise ValueError("target_emb must match state")
    ids = torch.as_tensor(item_ids, dtype=torch.long, device=device).view(-1)
    if ids.numel() != current_state.size(0):
        raise ValueError("item_ids must have one value per state row")
    users = torch.as_tensor(user_bank, dtype=current_state.dtype, device=device)
    if users.ndim != 2 or users.size(1) != current_state.size(1):
        raise ValueError("user_bank must be [users, dim] matching state")

    candidate_ids = engine.legal_candidate_ids(current_state, users)
    candidate_vectors = engine._candidate_vectors(users, candidate_ids)
    batch_size, candidate_count, emb_dim = candidate_vectors.shape
    if batch_size != current_state.size(0) or emb_dim != current_state.size(1):
        raise RuntimeError("legal candidate vectors do not match the requested state batch")

    panel_ids = engine.rank_panels.panel_for(ids, device=device)
    target_q = rank.panel_distribution(
        users, targets, panel_ids, temperature=engine.rank_temperature
    )
    before_q = rank.panel_distribution(
        users, current_state, panel_ids, temperature=engine.rank_temperature
    )
    before_kl = rank.rank_kl(target_q, before_q)

    # Keep the action axis explicit: flatten only batch x candidates, then
    # reshape the per-action KL back to [batch, candidate_count].
    candidate_states = current_state.unsqueeze(1) + engine.step_size * candidate_vectors
    flat_states = candidate_states.reshape(batch_size * candidate_count, emb_dim)
    flat_panels = panel_ids.unsqueeze(1).expand(-1, candidate_count, -1).reshape(
        batch_size * candidate_count, panel_ids.size(1)
    )
    flat_targets = target_q.unsqueeze(1).expand(-1, candidate_count, -1).reshape(
        batch_size * candidate_count, target_q.size(1)
    )
    after_q = rank.panel_distribution(
        users, flat_states, flat_panels, temperature=engine.rank_temperature
    )
    after_kl = rank.rank_kl(flat_targets, after_q).reshape(batch_size, candidate_count)
    candidate_gain = before_kl - after_kl
    utilities = torch.cat(
        (candidate_gain, torch.zeros((batch_size, 1), dtype=current_state.dtype, device=device)),
        dim=1,
    )
    if not torch.isfinite(utilities).all():
        raise RuntimeError("counterfactual action utilities must be finite")
    target_probs = torch.softmax(utilities / float(action_temperature), dim=1)
    if not torch.isfinite(target_probs).all() or not torch.allclose(
        target_probs.sum(dim=1), torch.ones(batch_size, device=device, dtype=target_probs.dtype), atol=1e-5
    ):
        raise RuntimeError("counterfactual action targets must be finite probability distributions")
    return candidate_ids.detach(), target_probs.detach(), utilities.detach()


def _action_log_probs(
    engine: rank.RankDistilledUSIMEngine,
    step: ActionDistillStep,
    *,
    user_bank: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every legal policy action using the existing actor interface."""
    state = step.state
    candidate_ids = step.candidate_ids
    candidates = engine._candidate_vectors(user_bank, candidate_ids)
    batch_size, candidate_count, emb_dim = candidates.shape
    action_count = candidate_count + 1
    if step.target_probs.shape != (batch_size, action_count):
        raise ValueError("target_probs must include every candidate plus END")
    actions = torch.arange(action_count, dtype=torch.long, device=state.device).view(1, -1)
    actions = actions.expand(batch_size, -1).reshape(-1)
    repeated_state = state.unsqueeze(1).expand(-1, action_count, -1).reshape(
        batch_size * action_count, emb_dim
    )
    repeated_remaining = step.remaining_steps.unsqueeze(1).expand(-1, action_count, -1).reshape(
        batch_size * action_count, 1
    )
    repeated_candidates = candidates.unsqueeze(1).expand(-1, action_count, -1, -1).reshape(
        batch_size * action_count, candidate_count, emb_dim
    )
    repeated_bias = step.candidate_logit_bias.unsqueeze(1).expand(-1, action_count, -1).reshape(
        batch_size * action_count, candidate_count
    )
    _, log_probs, _, _ = engine.policy.action_value(
        repeated_state,
        repeated_remaining,
        repeated_candidates,
        candidate_logit_bias=repeated_bias,
        action=actions,
    )
    return log_probs.reshape(batch_size, action_count)


def action_distillation_loss(
    engine: rank.RankDistilledUSIMEngine,
    step: ActionDistillStep,
    *,
    user_bank: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy from the frozen counterfactual action teacher to actor."""
    log_probs = _action_log_probs(engine, step, user_bank=user_bank)
    target_probs = step.target_probs.to(dtype=log_probs.dtype, device=log_probs.device)
    if target_probs.shape != log_probs.shape:
        raise ValueError("action target shape must match actor action log probabilities")
    return -(target_probs * log_probs).sum(dim=1).mean()


@torch.no_grad()
def _collect_on_policy_action_steps(
    engine: rank.RankDistilledUSIMEngine,
    *,
    initial_state: torch.Tensor,
    target_emb: torch.Tensor,
    item_ids: torch.Tensor,
    user_bank: torch.Tensor,
    user_history: Mapping[int, set[int]],
    action_temperature: float,
) -> list[ActionDistillStep]:
    """Label the current actor's greedy state distribution without gradients."""
    state = torch.as_tensor(initial_state).detach()
    targets = torch.as_tensor(target_emb, dtype=state.dtype, device=state.device).detach()
    active_ids = torch.as_tensor(item_ids, dtype=torch.long, device=state.device).view(-1)
    if targets.shape != state.shape or active_ids.numel() != state.size(0):
        raise ValueError("initial action-distillation batch dimensions are inconsistent")

    steps: list[ActionDistillStep] = []
    for step_index in range(engine.max_steps):
        if state.size(0) == 0:
            break
        candidate_ids, target_probs, utilities = counterfactual_action_targets(
            engine,
            state=state,
            target_emb=targets,
            user_bank=user_bank,
            item_ids=active_ids,
            action_temperature=action_temperature,
        )
        candidate_vectors = engine._candidate_vectors(user_bank, candidate_ids)
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
        steps.append(ActionDistillStep(
            state=state.detach(),
            remaining_steps=remaining.detach(),
            candidate_ids=candidate_ids.detach(),
            candidate_logit_bias=candidate_bias.detach(),
            target_probs=target_probs.detach(),
            utilities=utilities.detach(),
            actor_actions=actor_actions.detach(),
        ))

        end_action = candidate_vectors.size(1)
        safe_actions = actor_actions.clamp_max(end_action - 1)
        selected_users = candidate_vectors[
            torch.arange(state.size(0), device=state.device), safe_actions
        ]
        keep = actor_actions.ne(end_action)
        state = (state + engine.step_size * selected_users).index_select(
            0, torch.nonzero(keep, as_tuple=False).view(-1)
        )
        targets = targets.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))
        active_ids = active_ids.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))
    return steps


def _action_step_statistics(steps: Sequence[ActionDistillStep]) -> dict[str, float]:
    """Summarize target-space behavior collected before the optimization step."""
    if not steps:
        return {
            "train_examples": 0.0,
            "train_expert_gain": 0.0,
            "train_actor_gain": 0.0,
            "train_action_agreement": 0.0,
            "train_end_rate": 0.0,
        }
    utilities = torch.cat([step.utilities for step in steps], dim=0)
    actions = torch.cat([step.actor_actions for step in steps], dim=0)
    actor_gain = utilities.gather(1, actions.view(-1, 1)).view(-1)
    return {
        "train_examples": float(utilities.size(0)),
        "train_expert_gain": float(utilities.max(dim=1).values.mean().item()),
        "train_actor_gain": float(actor_gain.mean().item()),
        "train_action_agreement": float(actions.eq(utilities.argmax(dim=1)).float().mean().item()),
        "train_end_rate": float(actions.eq(utilities.size(1) - 1).float().mean().item()),
    }


def _action_policy_mode(epoch: int) -> str:
    return "identity_generator" if int(epoch) == 0 else "action_distill_rollout"


def select_action_distill_policy_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select only with P_val counterfactual rank gain, never outer cold rows."""
    if not rows:
        raise ValueError("action-distillation selection requires P_val rows")
    baseline = next((row for row in rows if int(row["epoch"]) == 0), None)
    if baseline is None:
        raise ValueError("action-distillation selection requires an epoch-0 identity row")
    eligible = [
        row for row in rows
        if int(row["epoch"]) == 0 or float(row["p_val_rank_gain"]) >= -1e-12
    ]
    if not eligible:
        raise RuntimeError("no action-distillation epoch has non-negative P_val rank gain")
    return max(
        eligible,
        key=lambda row: (float(row["p_val_rank_gain"]), -int(row["epoch"])),
    )


def train_action_distilled_policy(
    teacher: clean.CleanTeacher,
    generator: clean.ContentGenerator,
    engine: rank.RankDistilledUSIMEngine,
    views: clean.CleanStageViews,
    *,
    content: torch.Tensor,
    user_history: Mapping[int, set[int]],
    config: ActionDistillConfig,
) -> tuple[rank.RankDistilledUSIMEngine, dict[str, Any], list[dict[str, Any]]]:
    """Fit actor on P_train counterfactual actions and select only with P_val."""
    validate_action_distill_config(config)
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
    state_by_epoch: dict[int, dict[str, torch.Tensor]] = {0: clean._copy_state_dict(engine.policy)}
    rows: list[dict[str, Any]] = []

    def collect_validation(epoch: int, train_values: Mapping[str, float]) -> None:
        engine.policy.eval()
        diagnostics = rank._policy_rank_diagnostics(
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
            "policy_mode": _action_policy_mode(epoch),
            "train_loss": float(train_values["train_loss"]),
            "train_examples": float(train_values["train_examples"]),
            "train_expert_gain": float(train_values["train_expert_gain"]),
            "train_actor_gain": float(train_values["train_actor_gain"]),
            "train_action_agreement": float(train_values["train_action_agreement"]),
            "train_end_rate": float(train_values["train_end_rate"]),
            **diagnostics,
        })

    zero_values = {
        "train_loss": 0.0,
        "train_examples": 0.0,
        "train_expert_gain": 0.0,
        "train_actor_gain": 0.0,
        "train_action_agreement": 0.0,
        "train_end_rate": 0.0,
    }
    collect_validation(0, zero_values)

    for epoch in range(1, int(config.policy_epochs) + 1):
        losses: list[float] = []
        batch_stats: list[dict[str, float]] = []
        for batch_ids in clean._item_id_batches(train_item_ids, config.batch_size, seed=config.seed + epoch):
            ids = batch_ids.to(device)
            with torch.no_grad():
                initial_state = F.normalize(generator(frozen_content.index_select(0, ids)), dim=1)
                target = teacher.item_vectors(ids).detach()
            engine.policy.eval()
            steps = _collect_on_policy_action_steps(
                engine,
                initial_state=initial_state,
                target_emb=target,
                item_ids=ids,
                user_bank=user_bank,
                user_history=user_history,
                action_temperature=config.action_temperature,
            )
            if not steps:
                raise RuntimeError("counterfactual action collection produced no trainable states")
            engine.policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([
                action_distillation_loss(engine, step, user_bank=user_bank) for step in steps
            ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.policy.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            batch_stats.append(_action_step_statistics(steps))

        def average_stat(name: str) -> float:
            values = [row[name] for row in batch_stats]
            return float(sum(values) / max(1, len(values)))

        state_by_epoch[epoch] = clean._copy_state_dict(engine.policy)
        collect_validation(epoch, {
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "train_examples": average_stat("train_examples"),
            "train_expert_gain": average_stat("train_expert_gain"),
            "train_actor_gain": average_stat("train_actor_gain"),
            "train_action_agreement": average_stat("train_action_agreement"),
            "train_end_rate": average_stat("train_end_rate"),
        })

    selected = select_action_distill_policy_row(rows)
    engine.policy.load_state_dict(state_by_epoch[int(selected["epoch"])])
    engine.policy.eval()
    return engine, selected, rows


def _config_payload(config: ActionDistillConfig) -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in config.__dict__.items()
    }


def _write_action_panel_manifest(
    output_dir: Path,
    panels: rank.RankPanels,
    *,
    p_train_item_ids: FrozenSet[int],
    p_val_item_ids: FrozenSet[int],
) -> None:
    clean._write_json(output_dir / "rank_panel_manifest.json", {
        "source": "H_train_only_frozen_teacher_policy_pseudo_items",
        "seed": int(panels.seed),
        "panel_size": int(panels.panel_size),
        "item_count": int(len(panels.item_ids)),
        "all_panel_sha256": panels.digest(),
        "policy_train_panel_sha256": panels.digest(p_train_item_ids),
        "policy_val_panel_sha256": panels.digest(p_val_item_ids),
        "positive_count_sum": int(sum(panels.positive_counts)),
        "hard_count_sum": int(sum(panels.hard_counts)),
    })


def _write_action_distill_manifest(
    output_dir: Path,
    config: ActionDistillConfig,
    partitions: clean.CleanPartitions,
    *,
    stage_hashes: Mapping[str, str],
    selected_policy: Mapping[str, Any],
    course_stats: Mapping[str, Any],
) -> None:
    selected_epoch = int(selected_policy["epoch"])
    selected_gain = float(selected_policy["p_val_rank_gain"])
    clean._write_json(output_dir / "action_distill_manifest.json", {
        "route": "ckg_rl_usim_v35_action_distill",
        "control_type": "v32_vector_generator_plus_counterfactual_action_distillation",
        "seed": int(config.seed),
        "legacy_warm_checkpoint": None,
        "random_id_dropout": False,
        "main_candidate_mode": "legal_state_retrieval",
        "inference_oracle_access": False,
        "teacher_protocol": "H_train_only_selected_on_H_val",
        "generator_protocol": "H_G_only_vector_teacher_reconstruction_v32_exact",
        "generator_rank_loss": False,
        "policy_protocol": "P_train_counterfactual_rank_action_distillation_legal_candidates",
        "policy_optimizer": "counterfactual_action_distillation",
        "selection_protocol": "p_val_rank_gain_only",
        "outer_c_val_evaluated": False,
        "test_loaded": False,
        "selected_policy_epoch": selected_epoch,
        "selected_policy_mode": _action_policy_mode(selected_epoch),
        "selected_p_val_rank_gain": selected_gain,
        "viability_gate_passed": bool(selected_epoch > 0 and selected_gain > 0.0),
        "course_signal": dict(course_stats),
        "stage_hashes": dict(stage_hashes),
        "partitions": clean._partition_manifest_payload(partitions),
        "config": _config_payload(config),
    })


def run_action_distill_pipeline(config: ActionDistillConfig) -> dict[str, Any]:
    """Run the P_train/P_val viability screen without loading the test split."""
    clean._validate_clean_route_environment()
    validate_action_distill_config(config)
    output_dir = clean._resolve_path(config.output_dir)
    checkpoint_dir = clean._resolve_path(config.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty V3.5 output directory: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty V3.5 checkpoint directory: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    clean.setup_seed(config.seed)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    partitions = _build_p_only_partitions(
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
        checkpoint_dir / "teacher.pt", stage="teacher", module=teacher, metadata=teacher_meta
    )

    # Preserve the V3.2 generator RNG and H_G-only supervision exactly.
    generator, generator_meta = clean.train_content_generator(
        teacher, content, views.generator_item_ids, config=config
    )
    generator_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "generator.pt", stage="generator", module=generator, metadata=generator_meta
    )
    rank._write_csv(
        output_dir / "generator_vector_epochs.csv",
        generator_meta["history"],
        ("epoch", "train_loss", "validation_loss"),
    )

    panels = build_action_distill_panels(
        teacher,
        views.teacher_train,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
        config=config,
    )
    _write_action_panel_manifest(
        output_dir,
        panels,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
    )
    course_signal, course_stats = clean.build_clean_course_signal(
        partitions.h_train,
        n_items=int(content.size(0)),
        config=config,
    )
    engine = rank.create_rank_distilled_engine(
        config, rank_panels=panels, course_signal=course_signal
    )
    engine, selected_policy, policy_rows = train_action_distilled_policy(
        teacher,
        generator,
        engine,
        views,
        content=content,
        user_history=user_history,
        config=config,
    )
    policy_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "policy.pt",
        stage="policy",
        module=engine.policy,
        metadata={"selected": selected_policy, "validation_rows": policy_rows},
    )
    rank._write_csv(
        output_dir / "policy_action_epochs.csv",
        policy_rows,
        (
            "epoch", "policy_mode", "train_loss", "train_examples", "train_expert_gain",
            "train_actor_gain", "train_action_agreement", "train_end_rate",
            "p_val_initial_rank_kl", "p_val_final_rank_kl", "p_val_rank_gain",
        ),
    )
    selected_row = dict(next(
        row for row in policy_rows if int(row["epoch"]) == int(selected_policy["epoch"])
    ))
    selected_row["viability_gate_passed"] = bool(
        int(selected_row["epoch"]) > 0 and float(selected_row["p_val_rank_gain"]) > 0.0
    )
    clean._write_json(output_dir / "p_val_selected_metrics.json", selected_row)
    clean._write_json(output_dir / "policy_partition.json", clean._partition_manifest_payload(partitions))
    _write_action_distill_manifest(
        output_dir,
        config,
        partitions,
        stage_hashes={"teacher": teacher_hash, "generator": generator_hash, "policy": policy_hash},
        selected_policy=selected_policy,
        course_stats=course_stats,
    )
    return {
        "selected_policy_epoch": int(selected_policy["epoch"]),
        "selected_policy_mode": _action_policy_mode(int(selected_policy["epoch"])),
        "p_val": selected_row,
        "test_loaded": False,
        "output_dir": str(output_dir),
    }


def _parse_action_distill_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.5 counterfactual action-distillation viability screen")
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-course-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _action_distill_config_from_args(args: argparse.Namespace) -> ActionDistillConfig:
    config = ActionDistillConfig.for_seed(int(args.seed))
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


def _dry_run_action_distill_pipeline(config: ActionDistillConfig) -> dict[str, Any]:
    clean._validate_clean_route_environment()
    validate_action_distill_config(config)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
    partitions = _build_p_only_partitions(
        train_df,
        val_df,
        n_items=int(content.size(0)),
        config=config,
    )
    return {
        "status": "dry_run_ok",
        "route": "ckg_rl_usim_v35_action_distill",
        "seed": int(config.seed),
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "g_item_count": int(len(partitions.g_item_ids)),
        "p_train_item_count": int(len(partitions.p_train_item_ids)),
        "p_val_item_count": int(len(partitions.p_val_item_ids)),
        "generator_rank_loss": False,
        "test_loaded": False,
        "output_dir": str(clean._resolve_path(config.output_dir)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_action_distill_args(argv)
    config = _action_distill_config_from_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run_action_distill_pipeline(config), ensure_ascii=True, sort_keys=True))
        return 0
    result = run_action_distill_pipeline(config)
    if args.smoke:
        output_dir = Path(result["output_dir"])
        clean._write_json(output_dir / "smoke_report.json", {
            "status": "completed",
            "route": "ckg_rl_usim_v35_action_distill",
            "selected_policy_epoch": int(result["selected_policy_epoch"]),
            "target_free_inference": True,
            "test_loaded": False,
        })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
