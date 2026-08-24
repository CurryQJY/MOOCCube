"""One-time diagnostic test replay for the frozen V3.5 action-distillation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank
import ckg_rl_usim_v35_action_distill as v35


@dataclass(frozen=True)
class V35TestReplayConfig:
    """Frozen source and fresh result root for a single test replay."""

    source_output_dir: str | Path
    source_checkpoint_dir: str | Path | None
    output_dir: str | Path
    device: str = ""


@dataclass(frozen=True)
class FrozenV35Source:
    """Verified source manifest and checkpoint payloads, before test access."""

    manifest_path: Path
    manifest: dict[str, Any]
    checkpoint_dir: Path
    checkpoint_hashes: dict[str, str]
    checkpoint_payloads: dict[str, dict[str, Any]]
    selected_policy_epoch: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path, *, expected_stage: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_stage} checkpoint must be a dictionary")
    if payload.get("stage") != expected_stage:
        raise ValueError(f"{expected_stage} checkpoint stage does not match source manifest")
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError(f"{expected_stage} checkpoint must contain model_state")
    return payload


def _manifest_path(source_output_dir: str | Path) -> Path:
    source = Path(source_output_dir)
    return source if source.name == "action_distill_manifest.json" else source / "action_distill_manifest.json"


def load_frozen_v35_source(
    source_output_dir: str | Path,
    source_checkpoint_dir: str | Path | None = None,
) -> FrozenV35Source:
    """Validate source provenance and all stage files before test rows are read."""
    manifest_path = _manifest_path(source_output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V3.5 source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("route") != "ckg_rl_usim_v35_action_distill":
        raise ValueError("test replay requires a V3.5 action-distillation source manifest")
    if bool(manifest.get("test_loaded")):
        raise ValueError("test replay source must be the original P-only V3.5 output")
    selected_epoch = int(manifest.get("selected_policy_epoch", -1))
    if selected_epoch < 0:
        raise ValueError("source manifest must provide a non-negative selected policy epoch")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("source manifest must contain its action-distillation config")
    raw_checkpoint_dir = source_checkpoint_dir if source_checkpoint_dir is not None else config.get("checkpoint_dir")
    if not raw_checkpoint_dir:
        raise ValueError("source manifest does not identify a checkpoint directory")
    checkpoint_dir = clean._resolve_path(raw_checkpoint_dir)
    expected_hashes = manifest.get("stage_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("source manifest must contain stage_hashes")

    actual_hashes: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for stage in ("teacher", "generator", "policy"):
        checkpoint_path = checkpoint_dir / f"{stage}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"source {stage} checkpoint is missing: {checkpoint_path}")
        actual_hash = _sha256(checkpoint_path)
        expected_hash = str(expected_hashes.get(stage, "")).lower()
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise ValueError(f"{stage} checkpoint sha256 does not match source manifest")
        actual_hashes[stage] = actual_hash
        payloads[stage] = _load_checkpoint(checkpoint_path, expected_stage=stage)
    return FrozenV35Source(
        manifest_path=manifest_path,
        manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        checkpoint_hashes=actual_hashes,
        checkpoint_payloads=payloads,
        selected_policy_epoch=selected_epoch,
    )


def _source_config(source: FrozenV35Source, *, device: str) -> v35.ActionDistillConfig:
    fields = dict(source.manifest["config"])
    if device:
        fields["device"] = str(device)
    config = v35.ActionDistillConfig(**fields)
    v35.validate_action_distill_config(config)
    if int(config.seed) != int(source.manifest.get("seed", -1)):
        raise ValueError("source config seed does not match source manifest")
    return config


def _load_frozen_modules(
    source: FrozenV35Source,
    config: v35.ActionDistillConfig,
    *,
    meta: Mapping[str, Any],
    content: torch.Tensor,
    partitions: clean.CleanPartitions,
    user_history: Mapping[int, set[int]],
) -> tuple[clean.CleanTeacher, clean.ContentGenerator, rank.RankDistilledUSIMEngine]:
    """Instantiate the exact source graph and load verified model states only."""
    device = clean._resolve_device(config.device)
    teacher = clean.CleanTeacher(
        n_users=int(meta["n_users"]), n_items=int(meta["n_items"]), emb_dim=int(config.emb_dim)
    ).to(device)
    teacher.load_state_dict(source.checkpoint_payloads["teacher"]["model_state"], strict=True)
    teacher.eval()

    generator = clean.ContentGenerator(
        content_dim=int(content.size(1)), emb_dim=int(config.emb_dim), hidden_dim=int(config.hidden_dim)
    ).to(device)
    generator.load_state_dict(source.checkpoint_payloads["generator"]["model_state"], strict=True)
    generator.eval()

    views = clean.build_stage_views(partitions)
    panels = v35.build_action_distill_panels(
        teacher,
        views.teacher_train,
        p_train_item_ids=views.policy_train_item_ids,
        p_val_item_ids=views.policy_val_item_ids,
        config=config,
    )
    course_signal, _ = clean.build_clean_course_signal(
        partitions.h_train,
        n_items=int(content.size(0)),
        config=config,
    )
    engine = rank.create_rank_distilled_engine(
        config, rank_panels=panels, course_signal=course_signal
    )
    engine.policy.to(device)
    engine.policy.load_state_dict(source.checkpoint_payloads["policy"]["model_state"], strict=True)
    engine.policy.eval()
    return teacher, generator, engine


def _write_replay_manifest(
    output_dir: Path,
    source: FrozenV35Source,
    *,
    policy_mode: str,
) -> None:
    clean._write_json(output_dir / "test_replay_manifest.json", {
        "route": "ckg_rl_usim_v35_test_replay",
        "diagnostic_only": True,
        "test_loaded": True,
        "source_manifest": str(source.manifest_path),
        "source_checkpoint_dir": str(source.checkpoint_dir),
        "source_selected_policy_epoch": int(source.selected_policy_epoch),
        "source_selected_policy_mode": str(policy_mode),
        "checkpoint_hashes": dict(source.checkpoint_hashes),
        "checkpoint_hashes_match_source": True,
        "selection_performed": False,
        "training_performed": False,
    })


def run_v35_test_replay(config: V35TestReplayConfig) -> dict[str, Any]:
    """Evaluate the frozen source once on test after checkpoint verification."""
    output_dir = clean._resolve_path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing V3.5 test replay: {output_dir}")

    source = load_frozen_v35_source(config.source_output_dir, config.source_checkpoint_dir)
    source_config = _source_config(source, device=config.device)
    clean._validate_clean_route_environment()
    clean.setup_seed(source_config.seed)
    meta, content, train_df, val_df = clean.load_clean_train_val_inputs(source_config)
    partitions = v35._build_p_only_partitions(
        train_df,
        val_df,
        n_items=int(content.size(0)),
        config=source_config,
    )
    user_history = clean.build_user_seen(partitions.h_train)
    teacher, generator, engine = _load_frozen_modules(
        source,
        source_config,
        meta=meta,
        content=content,
        partitions=partitions,
        user_history=user_history,
    )

    # Test rows are loaded only after source hashes and all frozen modules pass validation.
    test_df = clean.load_clean_test_inputs(source_config, partitions.h_train)
    partitions = clean.attach_clean_test_rows(partitions, test_df)
    warm_item_ids = (
        partitions.g_item_ids | partitions.p_train_item_ids | partitions.p_val_item_ids
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_mode = v35._action_policy_mode(source.selected_policy_epoch)
    metrics = clean.evaluate_clean_route(
        teacher,
        generator,
        engine,
        hot_frame=partitions.h_test,
        cold_frame=partitions.c_test,
        warm_item_ids=warm_item_ids,
        content=content,
        user_history=user_history,
        config=source_config,
        policy_epoch=source.selected_policy_epoch,
        export_dir=output_dir,
        export_prefix="test_",
    )
    metrics["policy_mode"] = policy_mode
    result = {
        "route": "ckg_rl_usim_v35_test_replay",
        "diagnostic_only": True,
        "test_loaded": True,
        "selected_policy_epoch": int(source.selected_policy_epoch),
        "policy_mode": policy_mode,
        "checkpoint_hashes": dict(source.checkpoint_hashes),
        **metrics,
    }
    clean._write_json(output_dir / "test_metrics.json", result)
    _write_replay_manifest(output_dir, source, policy_mode=policy_mode)
    return result


def build_test_replay_dry_run(config: V35TestReplayConfig) -> dict[str, Any]:
    """Validate frozen source provenance without reading test rows or writing output."""
    source = load_frozen_v35_source(config.source_output_dir, config.source_checkpoint_dir)
    return {
        "status": "dry_run_ok",
        "route": "ckg_rl_usim_v35_test_replay",
        "diagnostic_only": True,
        "test_loaded": False,
        "source_manifest": str(source.manifest_path),
        "selected_policy_epoch": int(source.selected_policy_epoch),
        "checkpoint_hashes": dict(source.checkpoint_hashes),
        "output_dir": str(clean._resolve_path(config.output_dir)),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one frozen V3.5 checkpoint on test")
    parser.add_argument(
        "--source-output-dir",
        default="outputs/ckg_rl_usim_v35_action_distill/seed2025",
    )
    parser.add_argument(
        "--source-checkpoint-dir",
        default="checkpoints/ckg_rl_usim_v35_action_distill/seed2025",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ckg_rl_usim_v35_action_distill/test_replay_seed2025",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = V35TestReplayConfig(
        source_output_dir=args.source_output_dir,
        source_checkpoint_dir=args.source_checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    result = build_test_replay_dry_run(config) if args.dry_run else run_v35_test_replay(config)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
