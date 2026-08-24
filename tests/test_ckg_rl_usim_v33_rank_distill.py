"""Contracts for the isolated V3.3 rank-distilled USIM route."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest
import torch

import ckg_rl_usim_v32_clean as clean
from ckg_rl_usim_v33_rank_distill import (
    RankDistilledUSIMEngine,
    RankDistillRunConfig,
    RankPanels,
    build_rank_panels,
    generator_rank_objective,
    incremental_rank_gain,
    panel_distribution,
    rank_kl,
    _select_rank_policy_row,
    run_rank_distill_pipeline,
    train_rank_calibrated_generator,
)


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx"])
    frame["popularity"] = 1
    frame["_row_id"] = range(len(frame))
    return frame


def _teacher() -> clean.CleanTeacher:
    teacher = clean.CleanTeacher(n_users=5, n_items=4, emb_dim=2)
    with torch.no_grad():
        teacher.user_emb.weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]))
        teacher.item_emb.weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]))
    teacher.eval()
    return teacher


def test_rank_panels_are_deterministic_fixed_width_and_train_only():
    teacher = _teacher()
    train = _frame([(0, 0), (1, 0), (2, 1), (3, 1)])

    first = build_rank_panels(
        teacher,
        train,
        item_ids={0, 1},
        seed=7,
        panel_size=4,
        positive_count=1,
        hard_count=1,
    )
    second = build_rank_panels(
        teacher,
        train,
        item_ids={0, 1},
        seed=7,
        panel_size=4,
        positive_count=1,
        hard_count=1,
    )

    assert first.item_ids == (0, 1)
    assert torch.equal(first.panel_ids, second.panel_ids)
    assert first.panel_ids.shape == (2, 4)
    assert all(len(set(row.tolist())) == 4 for row in first.panel_ids)
    assert set(first.panel_ids[0].tolist()) & {0, 1}
    with pytest.raises(ValueError, match="H_train"):
        build_rank_panels(
            teacher,
            train,
            item_ids={3},
            seed=7,
            panel_size=4,
            positive_count=1,
            hard_count=1,
        )


def test_panel_kl_and_incremental_gain_have_the_correct_direction():
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    panels = torch.tensor([[0, 1]])
    target = torch.tensor([[1.0, 0.0]])
    before = torch.tensor([[0.0, 1.0]])
    after = target.clone()

    target_q = panel_distribution(users, target, panels, temperature=0.2)
    before_q = panel_distribution(users, before, panels, temperature=0.2)
    after_q = panel_distribution(users, after, panels, temperature=0.2)
    gain = incremental_rank_gain(target_q, before_q, after_q)

    assert rank_kl(target_q, after_q).item() == pytest.approx(0.0, abs=1e-7)
    assert rank_kl(target_q, before_q).item() > 0.0
    assert gain.item() > 0.0


def test_generator_rank_objective_is_zero_at_teacher_state_and_positive_away():
    users = torch.eye(2)
    panels = torch.tensor([[0, 1]])
    target = torch.tensor([[1.0, 0.0]])

    assert generator_rank_objective(
        target, target, users, panels, temperature=0.2, rank_weight=1.0
    ).item() == pytest.approx(0.0, abs=1e-7)
    assert generator_rank_objective(
        torch.tensor([[0.0, 1.0]]),
        target,
        users,
        panels,
        temperature=0.2,
        rank_weight=1.0,
    ).item() > 0.0


def test_rank_engine_reward_is_panel_kl_gain_not_legacy_vector_reward():
    panels = RankPanels(
        item_ids=(7,),
        panel_ids=torch.tensor([[0, 1]]),
        positive_counts=(0,),
        hard_counts=(0,),
        panel_size=2,
        seed=7,
    )
    engine = RankDistilledUSIMEngine(
        emb_dim=2,
        hidden_dim=4,
        max_steps=1,
        candidate_count=1,
        step_size=0.5,
        step_penalty=0.1,
        max_delta=1.0,
        rank_panels=panels,
        rank_temperature=0.2,
        course_reward_weight=0.0,
        delta_weight=0.0,
    )
    users = torch.eye(2)
    target = torch.tensor([[1.0, 0.0]])
    reward, embedding_gain, rank_gain, course_reward = engine._training_reward(
        torch.tensor([[0.0, 1.0]]),
        target,
        target,
        [torch.empty(0, dtype=torch.long)],
        users,
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([7]),
        {},
    )

    assert embedding_gain.item() == pytest.approx(0.0)
    assert rank_gain.item() > 0.0
    assert course_reward.item() == pytest.approx(0.0)
    assert reward.item() == pytest.approx(rank_gain.item() - 0.1)
    with pytest.raises(KeyError, match="panel"):
        engine._training_reward(
            torch.tensor([[0.0, 1.0]]),
            target,
            target,
            [torch.empty(0, dtype=torch.long)],
            users,
            torch.tensor([True]),
            torch.tensor([0]),
            torch.tensor([3]),
            {},
        )


def test_rank_engine_inference_is_target_free_even_without_a_cold_panel():
    panels = RankPanels(
        item_ids=(0,),
        panel_ids=torch.tensor([[0, 1]]),
        positive_counts=(1,),
        hard_counts=(1,),
        panel_size=2,
        seed=7,
    )
    engine = RankDistilledUSIMEngine(
        emb_dim=2,
        hidden_dim=4,
        max_steps=1,
        candidate_count=1,
        step_size=0.5,
        step_penalty=0.0,
        max_delta=1.0,
        rank_panels=panels,
        rank_temperature=0.2,
        course_reward_weight=0.0,
        delta_weight=0.0,
    )
    state = torch.tensor([[1.0, 0.0]])
    users = torch.eye(2)

    result = engine.rollout(
        state,
        user_bank=users,
        training=False,
        item_ids=torch.tensor([3]),
        user_history={},
    )

    assert result.final_state.shape == state.shape
    assert all(torch.equal(reward, torch.zeros_like(reward)) for reward in result.trajectory["rewards"])
    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(
            state,
            user_bank=users,
            training=False,
            target_emb=state,
            item_ids=torch.tensor([3]),
        )


def test_rank_generator_uses_only_the_supplied_h_g_items(tmp_path):
    teacher = _teacher()
    train = _frame([(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)])
    panels = build_rank_panels(
        teacher,
        train,
        item_ids={0, 1, 2},
        seed=7,
        panel_size=4,
        positive_count=1,
        hard_count=1,
    )
    config = RankDistillRunConfig(
        seed=7,
        data_dir=tmp_path / "data",
        split_dir=tmp_path / "split",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        generator_epochs=1,
        batch_size=2,
        emb_dim=2,
        hidden_dim=4,
        generator_val_fraction=0.5,
        device="cpu",
        panel_size=4,
        panel_positive_count=1,
        panel_hard_count=1,
    )

    generator, metadata = train_rank_calibrated_generator(
        teacher,
        torch.randn((4, 3)),
        generator_item_ids=frozenset({0, 1}),
        rank_panels=panels,
        config=config,
    )

    assert isinstance(generator, clean.ContentGenerator)
    assert set(metadata["train_item_ids"]) | set(metadata["validation_item_ids"]) == {0, 1}
    assert not (set(metadata["train_item_ids"]) | set(metadata["validation_item_ids"])) & {2}
    assert metadata["protocol"] == "h_g_only_vector_and_rank_distillation"


def test_rank_distill_pipeline_writes_isolated_audit_artifacts_and_delays_test_read(tmp_path):
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"n_users": 4, "n_items": 5, "content_dim": 3}), encoding="utf-8"
    )
    torch.save(torch.randn((5, 3)), data_dir / "content_emb.pt")
    _frame([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]).to_pickle(split_dir / "static_train.pkl")
    _frame([(0, 0), (3, 3)]).to_pickle(split_dir / "static_val.pkl")
    _frame([(1, 1), (0, 4)]).to_pickle(split_dir / "static_test.pkl")
    config = RankDistillRunConfig(
        seed=7,
        data_dir=data_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
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
        device="cpu",
    )

    result = run_rank_distill_pipeline(config)
    manifest = json.loads((output_dir / "v33_manifest.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((output_dir / "final_metrics.json").read_text(encoding="utf-8"))

    assert result["selected_policy_epoch"] in {0, 1}
    assert manifest["route"] == "ckg_rl_usim_v33_rank_distill"
    assert manifest["test_loaded_after_policy_selection"] is True
    assert manifest["inference_oracle_access"] is False
    assert (output_dir / "rank_panel_manifest.json").is_file()
    assert (output_dir / "generator_rank_epochs.csv").is_file()
    assert (output_dir / "policy_rank_epochs.csv").is_file()
    assert (checkpoint_dir / "teacher.pt").is_file()
    assert (checkpoint_dir / "generator.pt").is_file()
    assert (checkpoint_dir / "policy.pt").is_file()
    assert "validation" in final_metrics and "test" in final_metrics


def test_v33_launcher_is_fresh_and_locks_rank_distill_controls():
    launcher = Path("run_ckg_rl_usim_v33_rank_distill_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v33_rank_distill.py"' in launcher
    assert 'outputs\\ckg_rl_usim_v33_rank_distill' in launcher
    assert 'checkpoints\\ckg_rl_usim_v33_rank_distill' in launcher
    assert 'USIM_CLEAN_RANDOM_ID_DROPOUT' in launcher
    assert 'USIM_CLEAN_CANDIDATE_MODE' in launcher
    assert '--use-course-signal' in launcher
    assert '--dry-run' in launcher
    assert '[string]$RunTag = ""' in launcher
    assert 'USIM_ORIGINAL_V2' not in launcher
    assert 'USIM_V3_CORE' not in launcher
    assert 'USIM_FB_INIT_CKPT_DIR' not in launcher


def test_policy_selection_rejects_a_rollout_that_hurts_pseudo_cold_rank_kl():
    baseline = {
        "epoch": 0,
        "cold_n10": 0.10,
        "cold_r10": 0.10,
        "overall_n10": 0.10,
        "hot_r10": 0.20,
        "hot_n10": 0.20,
        "p_val_rank_gain": 0.0,
        "p_val_final_rank_kl": 0.5,
    }
    degraded_rollout = {
        **baseline,
        "epoch": 1,
        "cold_n10": 0.30,
        "cold_r10": 0.30,
        "overall_n10": 0.30,
        "p_val_rank_gain": -0.01,
        "p_val_final_rank_kl": 0.51,
    }

    selected = _select_rank_policy_row([baseline, degraded_rollout], hot_retention_tolerance=0.003)

    assert selected["epoch"] == 0
