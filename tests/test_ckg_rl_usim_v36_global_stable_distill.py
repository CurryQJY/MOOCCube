"""Contracts for V3.6 globally stable action distillation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import ckg_rl_usim_v32_clean as clean
import ckg_rl_usim_v33_rank_distill as rank
from ckg_rl_usim_v36_global_stable_distill import (
    GlobalStableConfig,
    build_global_anchor_bank,
    collect_mixed_action_steps,
    deterministic_expert_mask,
    run_global_stable_pipeline,
    select_stable_policy_row,
    stable_counterfactual_action_targets,
    validate_global_stable_config,
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
    max_steps: int = 1,
) -> rank.RankDistilledUSIMEngine:
    panels = rank.RankPanels(
        item_ids=(0,),
        panel_ids=torch.tensor([[0, 1]]),
        positive_counts=(0,),
        hard_counts=(0,),
        panel_size=2,
        seed=7,
    )
    return rank.RankDistilledUSIMEngine(
        emb_dim=int(teacher.user_emb.embedding_dim),
        hidden_dim=4,
        max_steps=max_steps,
        candidate_count=2,
        step_size=0.5,
        step_penalty=0.0,
        max_delta=1.0,
        rank_panels=panels,
        rank_temperature=0.2,
        course_reward_weight=0.0,
        delta_weight=0.0,
    )


def _config(tmp_path: Path) -> GlobalStableConfig:
    return GlobalStableConfig(
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
        global_anchor_count=2,
        global_stability_weight=10.0,
        expert_action_fraction=0.5,
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


def test_global_anchor_bank_is_h_g_only_deterministic_and_extracts_teacher_vectors():
    h_g_rows = pd.DataFrame({"u_idx": [4, 2, 4, 3], "i_idx": [0, 0, 1, 1]})

    first = build_global_anchor_bank(h_g_rows, seed=7, anchor_count=2)
    second = build_global_anchor_bank(h_g_rows.sample(frac=1.0, random_state=11), seed=7, anchor_count=2)

    assert first == second
    assert len(first.user_ids) == 2
    assert set(first.user_ids).issubset({2, 3, 4})
    assert len(first.digest()) == 64
    user_bank = torch.arange(10, dtype=torch.float32).view(5, 2)
    assert torch.equal(first.vectors(user_bank), user_bank[list(first.user_ids)])


def test_global_anchor_bank_rejects_non_h_g_or_empty_inputs():
    with pytest.raises(ValueError, match="u_idx"):
        build_global_anchor_bank(pd.DataFrame({"i_idx": [0]}), seed=7, anchor_count=1)
    with pytest.raises(ValueError, match="at least one"):
        build_global_anchor_bank(pd.DataFrame({"u_idx": [], "i_idx": []}), seed=7, anchor_count=1)
    with pytest.raises(ValueError, match="positive"):
        build_global_anchor_bank(pd.DataFrame({"u_idx": [0], "i_idx": [0]}), seed=7, anchor_count=0)


def test_global_stability_can_change_preference_without_changing_end_utility():
    teacher = _teacher(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher)
    kwargs = {
        "engine": engine,
        "state": torch.tensor([[0.0, 1.0]]),
        "target_emb": torch.tensor([[1.0, 0.0]]),
        "user_bank": teacher.user_vectors(),
        "item_ids": torch.tensor([0]),
        "anchor_vectors": torch.nn.functional.normalize(torch.tensor([[1.0, 1.0]]), dim=1),
        "action_temperature": 0.005,
    }

    candidate_ids, _, no_stability, no_stability_delta = stable_counterfactual_action_targets(
        **kwargs, global_stability_weight=0.0
    )
    _, labels, stable, stability_delta = stable_counterfactual_action_targets(
        **kwargs, global_stability_weight=100.0
    )

    positive_position = int(torch.nonzero(candidate_ids[0].eq(0), as_tuple=False).item())
    assert no_stability.argmax(dim=1).item() == positive_position
    assert stable.argmax(dim=1).item() != positive_position
    assert stable[0, -1].item() == pytest.approx(0.0)
    assert stability_delta[0, -1].item() == pytest.approx(0.0)
    assert stable[0, positive_position] == pytest.approx(
        no_stability[0, positive_position] - 100.0 * stability_delta[0, positive_position]
    )
    assert labels.sum(dim=1).item() == pytest.approx(1.0)
    assert no_stability_delta.shape == stable.shape


def test_global_stability_validates_anchor_shape_and_weight():
    teacher = _teacher(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher)
    kwargs = {
        "engine": engine,
        "state": torch.tensor([[0.0, 1.0]]),
        "target_emb": torch.tensor([[1.0, 0.0]]),
        "user_bank": teacher.user_vectors(),
        "item_ids": torch.tensor([0]),
        "action_temperature": 0.005,
    }
    with pytest.raises(ValueError, match="anchor_vectors"):
        stable_counterfactual_action_targets(
            **kwargs, anchor_vectors=torch.ones(1, 3), global_stability_weight=10.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        stable_counterfactual_action_targets(
            **kwargs, anchor_vectors=torch.ones(1, 2), global_stability_weight=-1.0
        )


def test_expert_mask_is_repeatable_and_respects_fraction_boundaries():
    item_ids = torch.tensor([10, 11, 12, 13, 14, 15])

    first = deterministic_expert_mask(
        item_ids, seed=7, epoch=3, step=1, fraction=0.5
    )
    second = deterministic_expert_mask(
        item_ids, seed=7, epoch=3, step=1, fraction=0.5
    )

    assert torch.equal(first, second)
    assert first.dtype == torch.bool
    assert not deterministic_expert_mask(
        item_ids, seed=7, epoch=3, step=1, fraction=0.0
    ).any()
    assert deterministic_expert_mask(
        item_ids, seed=7, epoch=3, step=1, fraction=1.0
    ).all()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        deterministic_expert_mask(
            item_ids, seed=7, epoch=3, step=1, fraction=1.1
        )


def test_expert_transition_advances_state_when_actor_selects_end():
    teacher = _teacher(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    engine = _engine(teacher, max_steps=2)
    with torch.no_grad():
        for parameter in engine.policy.parameters():
            parameter.zero_()
        engine.policy.end_head.bias.fill_(10.0)
    common = {
        "engine": engine,
        "initial_state": torch.tensor([[0.0, 1.0]]),
        "target_emb": torch.tensor([[1.0, 0.0]]),
        "item_ids": torch.tensor([0]),
        "user_bank": teacher.user_vectors(),
        "user_history": {},
        "anchor_vectors": torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1),
        "action_temperature": 0.005,
        "global_stability_weight": 0.0,
        "seed": 7,
        "epoch": 1,
    }

    actor_steps = collect_mixed_action_steps(**common, expert_action_fraction=0.0)
    expert_steps = collect_mixed_action_steps(**common, expert_action_fraction=1.0)

    assert len(actor_steps) == 1
    assert actor_steps[0].actor_actions.item() == actor_steps[0].utilities.size(1) - 1
    assert actor_steps[0].rollout_actions.item() == actor_steps[0].actor_actions.item()
    assert len(expert_steps) == 2
    assert expert_steps[0].expert_mask.item() is True
    assert expert_steps[0].rollout_actions.item() == expert_steps[0].expert_actions.item()
    assert not torch.equal(expert_steps[1].state, expert_steps[0].state)


def test_v36_selection_uses_only_nonnegative_p_val_gain():
    selected = select_stable_policy_row([
        {"epoch": 0, "p_val_rank_gain": 0.0, "p_val_anchor_drift": 0.0},
        {"epoch": 1, "p_val_rank_gain": -0.1, "p_val_anchor_drift": -10.0},
        {"epoch": 2, "p_val_rank_gain": 0.2, "p_val_anchor_drift": 10.0},
    ])

    assert selected["epoch"] == 2


def test_v36_config_locks_calibrated_components():
    config = GlobalStableConfig.for_seed(2025)

    assert config.global_anchor_count == 128
    assert config.global_stability_weight == pytest.approx(10.0)
    assert config.expert_action_fraction == pytest.approx(0.5)
    validate_global_stable_config(config)
    with pytest.raises(ValueError, match="anchor count"):
        validate_global_stable_config(replace(config, global_anchor_count=0))
    with pytest.raises(ValueError, match="stability weight"):
        validate_global_stable_config(replace(config, global_stability_weight=-1.0))
    with pytest.raises(ValueError, match="expert action fraction"):
        validate_global_stable_config(replace(config, expert_action_fraction=1.1))


def test_v36_pipeline_writes_anchor_manifest_and_never_loads_outer_or_test(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_tiny_split(Path(config.data_dir), Path(config.split_dir))

    def fail_test_load(*args, **kwargs):
        raise AssertionError("V3.6 P-only route must not load static_test")

    def fail_outer_eval(*args, **kwargs):
        raise AssertionError("V3.6 P-only route must not evaluate outer C_val")

    monkeypatch.setattr(clean, "load_clean_test_inputs", fail_test_load)
    monkeypatch.setattr(clean, "evaluate_clean_route", fail_outer_eval)
    result = run_global_stable_pipeline(config)
    output_dir = Path(config.output_dir)
    manifest = json.loads((output_dir / "v36_manifest.json").read_text(encoding="utf-8"))
    anchor_manifest = json.loads(
        (output_dir / "global_anchor_manifest.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (output_dir / "p_val_selected_metrics.json").read_text(encoding="utf-8")
    )

    assert result["selected_policy_epoch"] in {0, 1}
    assert "test" not in result
    assert manifest["route"] == "ckg_rl_usim_v36_global_stable_distill"
    assert manifest["test_loaded"] is False
    assert manifest["outer_c_val_evaluated"] is False
    assert manifest["selection_protocol"] == "p_val_rank_gain_only"
    assert manifest["policy_optimizer"] == "globally_stable_action_distillation"
    assert manifest["global_stability_weight"] == pytest.approx(10.0)
    assert manifest["expert_action_fraction"] == pytest.approx(0.5)
    assert manifest["partitions"]["c_val"]["rows"] == 0
    assert len(manifest["stage_hashes"]["teacher"]) == 64
    assert len(manifest["stage_hashes"]["generator"]) == 64
    assert anchor_manifest["source"] == "H_G_interaction_users_only"
    assert anchor_manifest["selected_user_count"] == 2
    assert len(anchor_manifest["anchor_sha256"]) == 64
    assert selected["epoch"] == result["selected_policy_epoch"]
    assert "p_val_anchor_drift" in selected
    assert "p_val_action_agreement" in selected
    assert "p_val_actor_end_rate" in selected
    assert (output_dir / "generator_vector_epochs.csv").is_file()
    assert (output_dir / "policy_stable_action_epochs.csv").is_file()
    assert not (output_dir / "final_metrics.json").exists()
    assert not any(output_dir.glob("test_*"))


def test_v36_launcher_is_fresh_and_pins_both_components():
    launcher = Path(
        "run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1"
    ).read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v36_global_stable_distill.py"' in launcher
    assert "outputs\\ckg_rl_usim_v36_global_stable_distill" in launcher
    assert "checkpoints\\ckg_rl_usim_v36_global_stable_distill" in launcher
    assert '"--global-anchor-count", "128"' in launcher
    assert '"--global-stability-weight", "10.0"' in launcher
    assert '"--expert-action-fraction", "0.5"' in launcher
    assert "USIM_CLEAN_RANDOM_ID_DROPOUT" in launcher
    assert "USIM_CLEAN_CANDIDATE_MODE" in launcher
    assert "--dry-run" in launcher
