"""Contracts for the one-time frozen V3.5 diagnostic test replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import ckg_rl_usim_v32_clean as clean
from ckg_rl_usim_v35_action_distill import ActionDistillConfig, run_action_distill_pipeline
from ckg_rl_usim_v35_test_replay import (
    V35TestReplayConfig,
    build_test_replay_dry_run,
    load_frozen_v35_source,
    run_v35_test_replay,
)


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx"])
    frame["popularity"] = 1
    frame["_row_id"] = range(len(frame))
    return frame


def _write_frozen_source_manifest(root: Path) -> tuple[Path, Path]:
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for stage in ("teacher", "generator", "policy"):
        path = checkpoint_dir / f"{stage}.pt"
        torch.save({"stage": stage, "model_state": {}}, path)
        hashes[stage] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "action_distill_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "route": "ckg_rl_usim_v35_action_distill",
            "seed": 2025,
            "test_loaded": False,
            "selected_policy_epoch": 15,
            "stage_hashes": hashes,
            "config": {"checkpoint_dir": str(checkpoint_dir)},
        }),
        encoding="utf-8",
    )
    return manifest_path, checkpoint_dir


def _tiny_source_config(tmp_path: Path) -> ActionDistillConfig:
    return ActionDistillConfig(
        seed=7,
        data_dir=tmp_path / "data",
        split_dir=tmp_path / "split",
        output_dir=tmp_path / "source_output",
        checkpoint_dir=tmp_path / "source_checkpoint",
        teacher_epochs=1,
        generator_epochs=1,
        policy_epochs=1,
        batch_size=2,
        eval_batch_size=2,
        emb_dim=2,
        hidden_dim=4,
        pseudo_ratio=0.67,
        pseudo_val_fraction=0.5,
        pseudo_min_popularity=1,
        generator_val_fraction=0.5,
        candidate_count=2,
        retrieval_chunk=2,
        max_steps=1,
        panel_size=4,
        panel_positive_count=1,
        panel_hard_count=1,
        action_temperature=0.005,
        device="cpu",
    )


def _write_tiny_source_inputs(config: ActionDistillConfig) -> None:
    data_dir = Path(config.data_dir)
    split_dir = Path(config.split_dir)
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"n_users": 4, "n_items": 5, "content_dim": 3}), encoding="utf-8"
    )
    torch.save(torch.randn((5, 3)), data_dir / "content_emb.pt")
    _frame([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]).to_pickle(
        split_dir / "static_train.pkl"
    )
    _frame([(0, 0), (3, 3)]).to_pickle(split_dir / "static_val.pkl")
    _frame([(0, 1), (0, 4)]).to_pickle(split_dir / "static_test.pkl")


def test_replay_rejects_source_checkpoint_hash_drift(tmp_path):
    manifest_path, checkpoint_dir = _write_frozen_source_manifest(tmp_path / "source")
    (checkpoint_dir / "teacher.pt").write_bytes(b"hash drift")

    with pytest.raises(ValueError, match="sha256"):
        load_frozen_v35_source(manifest_path)


def test_replay_loads_frozen_v35_models_without_training_and_writes_test_metrics(tmp_path, monkeypatch):
    source_config = _tiny_source_config(tmp_path)
    _write_tiny_source_inputs(source_config)
    run_action_distill_pipeline(source_config)

    def fail_training(*args, **kwargs):
        raise AssertionError("frozen test replay must not train")

    monkeypatch.setattr(clean, "train_clean_teacher", fail_training)
    monkeypatch.setattr(clean, "train_content_generator", fail_training)
    replay_output = tmp_path / "replay_output"
    result = run_v35_test_replay(V35TestReplayConfig(
        source_output_dir=source_config.output_dir,
        source_checkpoint_dir=source_config.checkpoint_dir,
        output_dir=replay_output,
        device="cpu",
    ))
    manifest = json.loads((replay_output / "test_replay_manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (Path(source_config.output_dir) / "action_distill_manifest.json").read_text(encoding="utf-8")
    )

    assert result["diagnostic_only"] is True
    assert result["test_loaded"] is True
    assert result["policy_mode"] == source_manifest["selected_policy_mode"]
    assert result["cold_item_count"] == 1
    assert result["hot_item_count"] == 1
    assert manifest["source_selected_policy_epoch"] == result["selected_policy_epoch"]
    assert manifest["checkpoint_hashes_match_source"] is True
    assert (replay_output / "test_metrics.json").is_file()
    assert (replay_output / "test_per_item_hot.csv").is_file()
    assert (replay_output / "test_per_item_cold.csv").is_file()
    assert not (Path(source_config.output_dir) / "test_metrics.json").exists()


def test_replay_dry_run_does_not_create_output_and_launcher_is_fresh(tmp_path):
    manifest_path, checkpoint_dir = _write_frozen_source_manifest(tmp_path / "source")
    output_dir = tmp_path / "replay_output"
    payload = build_test_replay_dry_run(V35TestReplayConfig(
        source_output_dir=manifest_path.parent,
        source_checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        device="cpu",
    ))
    launcher = Path("run_ckg_rl_usim_v35_test_replay_seed2025.ps1").read_text(encoding="utf-8")

    assert payload["route"] == "ckg_rl_usim_v35_test_replay"
    assert payload["test_loaded"] is False
    assert payload["selected_policy_epoch"] == 15
    assert not output_dir.exists()
    assert 'ScriptPath = "ckg_rl_usim_v35_test_replay.py"' in launcher
    assert "test_replay_seed2025" in launcher
    assert "--dry-run" in launcher
