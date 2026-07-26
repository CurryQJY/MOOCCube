"""Contracts for the V3.4 vector-generator / rank-reward-only control."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import ckg_rl_usim_v32_clean as clean
from ckg_rl_usim_v34_rank_reward_control import (
    RankRewardControlConfig,
    build_policy_rank_panels,
    run_rank_reward_control_pipeline,
    validate_rank_reward_control_config,
)


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx"])
    frame["popularity"] = 1
    frame["_row_id"] = range(len(frame))
    return frame


def _teacher() -> clean.CleanTeacher:
    teacher = clean.CleanTeacher(n_users=5, n_items=5, emb_dim=2)
    with torch.no_grad():
        teacher.user_emb.weight.copy_(torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ]))
        teacher.item_emb.weight.copy_(torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5],
        ]))
    teacher.eval()
    return teacher


def _config(tmp_path) -> RankRewardControlConfig:
    return RankRewardControlConfig(
        seed=7,
        data_dir=tmp_path / "data",
        split_dir=tmp_path / "split",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
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
        generator_rank_weight=0.0,
        device="cpu",
    )


def test_control_config_forbids_generator_rank_loss_and_uses_v34_roots(tmp_path):
    default = RankRewardControlConfig.for_seed(2025)
    assert "v34_rank_reward_control" in str(default.output_dir)
    config = _config(tmp_path)

    validate_rank_reward_control_config(config)
    with pytest.raises(ValueError, match="rank weight"):
        validate_rank_reward_control_config(replace(config, generator_rank_weight=0.1))


def test_control_panels_are_limited_to_policy_pseudo_items(tmp_path):
    teacher = _teacher()
    train = _frame([(0, 1), (1, 1), (2, 2), (3, 2), (4, 3)])

    panels = build_policy_rank_panels(
        teacher,
        train,
        p_train_item_ids=frozenset({1}),
        p_val_item_ids=frozenset({2}),
        config=_config(tmp_path),
    )

    assert panels.item_ids == (1, 2)
    assert panels.panel_ids.shape == (2, 4)
    with pytest.raises(KeyError, match="panel"):
        panels.panel_for(torch.tensor([3]), device=torch.device("cpu"))


def test_control_pipeline_writes_vector_generator_and_rank_policy_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"n_users": 4, "n_items": 5, "content_dim": 3}), encoding="utf-8"
    )
    torch.save(torch.randn((5, 3)), data_dir / "content_emb.pt")
    _frame([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]).to_pickle(split_dir / "static_train.pkl")
    _frame([(0, 0), (3, 3)]).to_pickle(split_dir / "static_val.pkl")
    _frame([(1, 1), (0, 4)]).to_pickle(split_dir / "static_test.pkl")
    config = replace(
        _config(tmp_path),
        data_dir=data_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )

    result = run_rank_reward_control_pipeline(config)
    manifest = json.loads((output_dir / "control_manifest.json").read_text(encoding="utf-8"))

    assert result["selected_policy_epoch"] in {0, 1}
    assert manifest["route"] == "ckg_rl_usim_v34_rank_reward_control"
    assert manifest["generator_rank_loss"] is False
    assert manifest["policy_protocol"] == "P_train_rank_gain_reward_legal_candidates"
    assert manifest["test_loaded_after_policy_selection"] is True
    assert (output_dir / "generator_vector_epochs.csv").is_file()
    assert (output_dir / "policy_rank_epochs.csv").is_file()
    assert (output_dir / "rank_panel_manifest.json").is_file()
    assert (checkpoint_dir / "teacher.pt").is_file()
    assert (checkpoint_dir / "generator.pt").is_file()
    assert (checkpoint_dir / "policy.pt").is_file()


def test_v34_launcher_locks_vector_generator_control():
    launcher = Path("run_ckg_rl_usim_v34_rank_reward_control_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v34_rank_reward_control.py"' in launcher
    assert 'outputs\\ckg_rl_usim_v34_rank_reward_control' in launcher
    assert 'checkpoints\\ckg_rl_usim_v34_rank_reward_control' in launcher
    assert 'USIM_CLEAN_RANDOM_ID_DROPOUT' in launcher
    assert 'USIM_CLEAN_CANDIDATE_MODE' in launcher
    assert '--use-course-signal' in launcher
    assert '--dry-run' in launcher
    assert 'generator-rank-weight' not in launcher
    assert 'USIM_ORIGINAL_V2' not in launcher
    assert 'USIM_V3_CORE' not in launcher
