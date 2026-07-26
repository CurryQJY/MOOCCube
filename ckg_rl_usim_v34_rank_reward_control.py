"""V3.4 causal control: V3.2 vector generator plus V3.3 rank-reward policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, FrozenSet, Sequence

import torch

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank


@dataclass(frozen=True)
class RankRewardControlConfig(rank.RankDistillRunConfig):
    """Configuration for the one-variable rank-reward control."""

    generator_rank_weight: float = 0.0

    @classmethod
    def for_seed(cls, seed: int) -> "RankRewardControlConfig":
        base = rank.RankDistillRunConfig.for_seed(int(seed))
        fields = dict(base.__dict__)
        fields.update({
            "output_dir": f"outputs/ckg_rl_usim_v34_rank_reward_control/seed{int(seed)}",
            "checkpoint_dir": f"checkpoints/ckg_rl_usim_v34_rank_reward_control/seed{int(seed)}",
            "generator_rank_weight": 0.0,
        })
        return cls(**fields)


def validate_rank_reward_control_config(config: RankRewardControlConfig) -> None:
    rank._validate_rank_distill_config(config)
    if abs(float(config.generator_rank_weight)) > 1e-12:
        raise ValueError("rank reward control requires generator rank weight equal to zero")


def build_policy_rank_panels(
    teacher: clean.CleanTeacher,
    h_train,
    *,
    p_train_item_ids: FrozenSet[int],
    p_val_item_ids: FrozenSet[int],
    config: RankRewardControlConfig,
) -> rank.RankPanels:
    """Build teacher panels only for policy pseudo items after generator fitting."""
    item_ids = frozenset(int(item_id) for item_id in p_train_item_ids | p_val_item_ids)
    if not item_ids:
        raise ValueError("rank reward control requires at least one policy pseudo item")
    return rank.build_rank_panels(
        teacher,
        h_train,
        item_ids=item_ids,
        seed=config.seed,
        panel_size=config.panel_size,
        positive_count=config.panel_positive_count,
        hard_count=config.panel_hard_count,
    )


def _config_payload(config: RankRewardControlConfig) -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in config.__dict__.items()
    }


def _write_control_panel_manifest(
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


def _write_control_manifest(
    output_dir: Path,
    config: RankRewardControlConfig,
    partitions: clean.CleanPartitions,
    *,
    stage_hashes: dict[str, str],
    selected_policy: dict[str, Any],
    course_stats: dict[str, Any],
) -> None:
    clean._write_json(output_dir / "control_manifest.json", {
        "route": "ckg_rl_usim_v34_rank_reward_control",
        "control_type": "v32_vector_generator_plus_v33_rank_reward_policy",
        "seed": int(config.seed),
        "legacy_warm_checkpoint": None,
        "random_id_dropout": False,
        "main_candidate_mode": "legal_state_retrieval",
        "inference_oracle_access": False,
        "teacher_protocol": "H_train_only_selected_on_H_val",
        "generator_protocol": "H_G_only_vector_teacher_reconstruction_v32_exact",
        "generator_rank_loss": False,
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


def run_rank_reward_control_pipeline(config: RankRewardControlConfig) -> dict[str, Any]:
    """Run the V3.4 causal control; test data stays unread until selection ends."""
    clean._validate_clean_route_environment()
    validate_rank_reward_control_config(config)
    output_dir = clean._resolve_path(config.output_dir)
    checkpoint_dir = clean._resolve_path(config.checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty V3.4 output directory: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty V3.4 checkpoint directory: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    clean.setup_seed(config.seed)
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

    # This call is intentionally unchanged from V3.2 and happens before any
    # policy-panel construction, preserving the generator RNG and supervision.
    generator, generator_meta = clean.train_content_generator(
        teacher,
        content,
        views.generator_item_ids,
        config=config,
    )
    generator_hash = clean._save_stage_checkpoint(
        checkpoint_dir / "generator.pt", stage="generator", module=generator, metadata=generator_meta
    )
    rank._write_csv(
        output_dir / "generator_vector_epochs.csv",
        generator_meta["history"],
        ("epoch", "train_loss", "validation_loss"),
    )

    panels = build_policy_rank_panels(
        teacher,
        views.teacher_train,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
        config=config,
    )
    _write_control_panel_manifest(
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
    engine = rank.create_rank_distilled_engine(config, rank_panels=panels, course_signal=course_signal)
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

    engine, selected_policy, policy_rows = rank.train_rank_distilled_policy(
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
    rank._write_csv(
        output_dir / "policy_rank_epochs.csv",
        policy_rows,
        (
            "epoch", "policy_mode", "train_loss", "train_rank_gain", "p_val_initial_rank_kl",
            "p_val_final_rank_kl", "p_val_rank_gain", "cold_r10", "cold_n10", "hot_r10", "hot_n10",
            "overall_r10", "overall_n10", "cold_item_count", "hot_item_count",
        ),
    )
    validation = dict(next(row for row in policy_rows if int(row["epoch"]) == int(selected_policy["epoch"])))

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
    _write_control_manifest(
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


def _parse_control_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.4 rank-reward-only causal control")
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
    parser.add_argument("--course-reward-weight", type=float)
    parser.add_argument("--delta-weight", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-course-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _control_config_from_args(args: argparse.Namespace) -> RankRewardControlConfig:
    config = RankRewardControlConfig.for_seed(int(args.seed))
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


def _dry_run_control_pipeline(config: RankRewardControlConfig) -> dict[str, Any]:
    clean._validate_clean_route_environment()
    validate_rank_reward_control_config(config)
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
        "route": "ckg_rl_usim_v34_rank_reward_control",
        "seed": int(config.seed),
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "g_item_count": int(len(partitions.g_item_ids)),
        "p_train_item_count": int(len(partitions.p_train_item_ids)),
        "p_val_item_count": int(len(partitions.p_val_item_ids)),
        "generator_rank_loss": False,
        "output_dir": str(clean._resolve_path(config.output_dir)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_control_args(argv)
    config = _control_config_from_args(args)
    if args.dry_run:
        print(json.dumps(_dry_run_control_pipeline(config), ensure_ascii=True, sort_keys=True))
        return 0
    result = run_rank_reward_control_pipeline(config)
    if args.smoke:
        output_dir = Path(result["output_dir"])
        clean._write_json(output_dir / "smoke_report.json", {
            "status": "completed",
            "route": "ckg_rl_usim_v34_rank_reward_control",
            "selected_policy_epoch": int(result["selected_policy_epoch"]),
            "target_free_inference": True,
            "test_selection_used": False,
        })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
