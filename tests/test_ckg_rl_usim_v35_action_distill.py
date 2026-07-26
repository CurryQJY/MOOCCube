"""Contracts for the isolated V3.5 counterfactual action-distillation route."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank
from ckg_rl_usim_v35_action_distill import (
    ActionDistillConfig,
    counterfactual_action_targets,
    run_action_distill_pipeline,
    select_action_distill_policy_row,
    validate_action_distill_config,
)


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx"])
    frame["popularity"] = 1
    frame["_row_id"] = range(len(frame))
    return frame


def _teacher(user_vectors: torch.Tensor, item_vectors: torch.Tensor) -> clean.CleanTeacher:
    teacher = clean.CleanTeacher(
        n_users=int(user_vectors.size(0)),
        n_items=int(item_vectors.size(0)),
        emb_dim=int(user_vectors.size(1)),
    )
    with torch.no_grad():
        teacher.user_emb.weight.copy_(user_vectors)
        teacher.item_emb.weight.copy_(item_vectors)
    teacher.eval()
    return teacher


def _engine(
    teacher: clean.CleanTeacher,
    *,
    candidate_count: int,
    item_id: int = 0,
) -> rank.RankDistilledUSIMEngine:
    panels = rank.RankPanels(
        item_ids=(int(item_id),),
        panel_ids=torch.tensor([[0, 1]]),
        positive_counts=(0,),
        hard_counts=(0,),
        panel_size=2,
        seed=7,
    )
    return rank.RankDistilledUSIMEngine(
        emb_dim=int(teacher.user_emb.embedding_dim),
        hidden_dim=4,
        max_steps=1,
        candidate_count=int(candidate_count),
        step_size=0.5,
        step_penalty=0.0,
        max_delta=1.0,
        rank_panels=panels,
        rank_temperature=0.2,
        course_reward_weight=0.0,
        delta_weight=0.0,
    )


def _config(tmp_path: Path) -> ActionDistillConfig:
    return ActionDistillConfig(
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
        action_temperature=0.005,
        device="cpu",
    )


def _write_tiny_split(data_dir: Path, split_dir: Path) -> None:
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


def test_counterfactual_targets_include_end_and_prefer_the_positive_legal_action():
    teacher = _teacher(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher, candidate_count=2)
    candidate_ids, labels, utilities = counterfactual_action_targets(
        engine,
        state=torch.tensor([[0.0, 1.0]]),
        target_emb=torch.tensor([[1.0, 0.0]]),
        user_bank=teacher.user_vectors(),
        item_ids=torch.tensor([0]),
        action_temperature=0.005,
    )

    positive_position = int(torch.nonzero(candidate_ids[0].eq(0), as_tuple=False).item())
    assert candidate_ids.shape == (1, 2)
    assert labels.shape == (1, 3)
    assert utilities.shape == (1, 3)
    assert labels.sum(dim=1).item() == pytest.approx(1.0)
    assert torch.isfinite(labels).all()
    assert utilities[0, positive_position] > utilities[0, -1]
    assert candidate_ids[0, int(utilities[0, :-1].argmax())].item() == 0


def test_counterfactual_targets_prefer_end_when_every_user_action_hurts():
    teacher = _teacher(
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher, candidate_count=1)
    _, labels, utilities = counterfactual_action_targets(
        engine,
        state=torch.tensor([[1.0, 0.0]]),
        target_emb=torch.tensor([[1.0, 0.0]]),
        user_bank=teacher.user_vectors(),
        item_ids=torch.tensor([0]),
        action_temperature=0.005,
    )

    assert utilities[0, 0].item() < 0.0
    assert utilities.argmax(dim=1).item() == 1
    assert labels.argmax(dim=1).item() == 1


def test_action_distill_config_requires_vector_generator_and_positive_temperature(tmp_path):
    config = _config(tmp_path)
    validate_action_distill_config(config)
    with pytest.raises(ValueError, match="rank weight"):
        validate_action_distill_config(replace(config, generator_rank_weight=0.1))
    with pytest.raises(ValueError, match="temperature"):
        validate_action_distill_config(replace(config, action_temperature=0.0))


def test_action_distill_selection_rejects_negative_p_val_gain():
    selected = select_action_distill_policy_row([
        {"epoch": 0, "p_val_rank_gain": 0.0},
        {"epoch": 1, "p_val_rank_gain": -0.01},
        {"epoch": 2, "p_val_rank_gain": -0.001},
    ])
    assert selected["epoch"] == 0


def test_target_free_v35_engine_accepts_unpanelled_cold_item_and_rejects_oracle_target():
    teacher = _teacher(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher, candidate_count=1)
    state = torch.tensor([[1.0, 0.0]])

    result = engine.rollout(
        state,
        user_bank=teacher.user_vectors(),
        training=False,
        item_ids=torch.tensor([99]),
        user_history={},
    )

    assert result.final_state.shape == state.shape
    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(
            state,
            user_bank=teacher.user_vectors(),
            training=False,
            target_emb=state,
            item_ids=torch.tensor([99]),
            user_history={},
        )


def test_v35_pipeline_writes_p_only_artifacts_and_never_evaluates_outer_or_test(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_tiny_split(Path(config.data_dir), Path(config.split_dir))

    def fail_test_load(*args, **kwargs):
        raise AssertionError("V3.5 viability route must not load static_test")

    def fail_outer_eval(*args, **kwargs):
        raise AssertionError("V3.5 viability route must not evaluate C_val during selection")

    monkeypatch.setattr(clean, "load_clean_test_inputs", fail_test_load)
    monkeypatch.setattr(clean, "evaluate_clean_route", fail_outer_eval)
    result = run_action_distill_pipeline(config)
    output_dir = Path(config.output_dir)
    manifest = json.loads((output_dir / "action_distill_manifest.json").read_text(encoding="utf-8"))
    selected_metrics = json.loads((output_dir / "p_val_selected_metrics.json").read_text(encoding="utf-8"))

    assert result["selected_policy_epoch"] in {0, 1}
    assert "test" not in result
    assert manifest["route"] == "ckg_rl_usim_v35_action_distill"
    assert manifest["test_loaded"] is False
    assert manifest["generator_rank_loss"] is False
    assert manifest["selection_protocol"] == "p_val_rank_gain_only"
    assert manifest["policy_optimizer"] == "counterfactual_action_distillation"
    assert manifest["partitions"]["c_val"]["rows"] == 0
    assert len(manifest["stage_hashes"]["teacher"]) == 64
    assert len(manifest["stage_hashes"]["generator"]) == 64
    assert selected_metrics["epoch"] == result["selected_policy_epoch"]
    assert (output_dir / "generator_vector_epochs.csv").is_file()
    assert (output_dir / "policy_action_epochs.csv").is_file()
    assert (output_dir / "rank_panel_manifest.json").is_file()
    assert not (output_dir / "final_metrics.json").exists()
    assert not any(output_dir.glob("test_*"))
    assert (Path(config.checkpoint_dir) / "teacher.pt").is_file()
    assert (Path(config.checkpoint_dir) / "generator.pt").is_file()
    assert (Path(config.checkpoint_dir) / "policy.pt").is_file()


def test_v35_launcher_is_fresh_and_locks_target_free_controls():
    launcher = Path("run_ckg_rl_usim_v35_action_distill_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v35_action_distill.py"' in launcher
    assert "outputs\\ckg_rl_usim_v35_action_distill" in launcher
    assert "checkpoints\\ckg_rl_usim_v35_action_distill" in launcher
    assert "USIM_CLEAN_RANDOM_ID_DROPOUT" in launcher
    assert "USIM_CLEAN_CANDIDATE_MODE" in launcher
    assert "--action-temperature" in launcher
    assert "--dry-run" in launcher
    assert "USIM_ORIGINAL_V2" not in launcher
    assert "USIM_V3_CORE" not in launcher
