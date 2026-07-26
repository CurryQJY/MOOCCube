"""Contracts for the isolated clean T -> G -> V3.2 route."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from ckg_rl_usim_v32_clean import (
    CleanCourseSignal,
    CleanRecPPO,
    CleanRunConfig,
    CleanUSIMEngine,
    CleanTeacher,
    ContentGenerator,
    build_clean_partitions,
    build_clean_item_bank,
    build_stage_views,
    create_clean_engine,
    full_positive_score_gain,
    project_displacement,
    run_clean_pipeline,
    target_excluded_history,
)


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx"])
    frame["popularity"] = 1
    frame["_row_id"] = range(len(frame))
    return frame


def _outer_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _frame(
        [
            (0, 0), (1, 0),
            (2, 1), (3, 1),
            (4, 2), (5, 2),
            (6, 3), (7, 3),
            (8, 4), (9, 4),
            (10, 5), (11, 5),
        ]
    )
    val = _frame([(0, 0), (2, 1), (12, 9)])
    test = _frame([(4, 2), (6, 3), (13, 10)])
    return train, val, test


def _partitions():
    train, val, test = _outer_split()
    return build_clean_partitions(
        train,
        val,
        test,
        n_items=11,
        seed=7,
        pseudo_ratio=0.50,
        pseudo_val_fraction=0.34,
        min_popularity=1,
    )


def test_clean_partitions_are_deterministic_disjoint_and_train_only():
    train, val, test = _outer_split()
    first = _partitions()
    second = build_clean_partitions(
        train,
        val,
        test,
        n_items=11,
        seed=7,
        pseudo_ratio=0.50,
        pseudo_val_fraction=0.34,
        min_popularity=1,
    )

    assert first.g_item_ids == second.g_item_ids
    assert first.p_train_item_ids == second.p_train_item_ids
    assert first.p_val_item_ids == second.p_val_item_ids
    assert first.g_item_ids.isdisjoint(first.p_train_item_ids)
    assert first.g_item_ids.isdisjoint(first.p_val_item_ids)
    assert first.p_train_item_ids.isdisjoint(first.p_val_item_ids)
    assert first.g_item_ids | first.p_train_item_ids | first.p_val_item_ids == set(range(6))
    assert set(first.h_val["i_idx"]) == {0, 1}
    assert set(first.c_val["i_idx"]) == {9}
    assert set(first.h_test["i_idx"]) == {2, 3}
    assert set(first.c_test["i_idx"]) == {10}
    assert_frame_equal(first.h_train, train)


def test_pseudo_selection_is_invariant_to_outer_validation_and_test_rows():
    train, val, test = _outer_split()
    baseline = _partitions()
    changed_val = pd.concat([val, _frame([(99, 8), (98, 7)])], ignore_index=True)
    changed_test = pd.concat([test, _frame([(97, 6), (96, 10)])], ignore_index=True)

    changed = build_clean_partitions(
        train,
        changed_val,
        changed_test,
        n_items=11,
        seed=7,
        pseudo_ratio=0.50,
        pseudo_val_fraction=0.34,
        min_popularity=1,
    )

    assert changed.p_train_item_ids == baseline.p_train_item_ids
    assert changed.p_val_item_ids == baseline.p_val_item_ids
    assert changed.g_item_ids == baseline.g_item_ids


def test_stage_views_keep_teacher_warm_only_and_generator_outside_pseudo_items():
    parts = _partitions()
    views = build_stage_views(parts)

    assert set(views.teacher_train["_row_id"]) == set(parts.h_train["_row_id"])
    assert set(views.teacher_val["_row_id"]) == set(parts.h_val["_row_id"])
    assert set(views.generator_item_ids) == set(parts.g_item_ids)
    assert not set(views.generator_item_ids) & set(parts.p_train_item_ids)
    assert not set(views.generator_item_ids) & set(parts.p_val_item_ids)
    assert set(views.policy_train_item_ids) == set(parts.p_train_item_ids)
    assert set(views.policy_val_item_ids) == set(parts.p_val_item_ids)
    assert not set(views.teacher_train["i_idx"]) & set(parts.c_val["i_idx"])
    assert not set(views.teacher_train["i_idx"]) & set(parts.c_test["i_idx"])


def test_clean_teacher_uses_only_behavioral_embedding_tables():
    teacher = CleanTeacher(n_users=3, n_items=4, emb_dim=2)
    loss = teacher.ranking_loss(torch.tensor([0, 1]), torch.tensor([1, 2]))

    assert torch.isfinite(loss)
    assert hasattr(teacher, "user_emb")
    assert hasattr(teacher, "item_emb")
    assert not hasattr(teacher, "agent")
    assert not hasattr(teacher, "content_proj")


def test_content_generator_has_no_item_id_parameters_and_maps_to_teacher_space():
    generator = ContentGenerator(content_dim=3, emb_dim=2, hidden_dim=4)
    output = generator(torch.zeros((2, 3), dtype=torch.float32))

    assert output.shape == (2, 2)
    assert all("item" not in name.lower() for name, _ in generator.named_parameters())


def _engine(*, max_steps: int = 2, max_delta: float = 1.0) -> CleanUSIMEngine:
    return CleanUSIMEngine(
        emb_dim=2,
        hidden_dim=4,
        max_steps=max_steps,
        candidate_count=2,
        step_size=0.5,
        step_penalty=0.0,
        max_delta=max_delta,
    )


def test_main_candidate_pool_is_state_retrieval_for_both_train_and_inference():
    engine = _engine()
    state = torch.tensor([[1.0, 0.0]])
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    candidate_ids = engine.legal_candidate_ids(state, users)

    assert torch.equal(candidate_ids, torch.tensor([[0, 1]]))
    assert torch.equal(candidate_ids, engine.legal_candidate_ids(state, users))


def test_chunked_legal_retrieval_matches_full_bank_topk():
    engine = CleanUSIMEngine(
        emb_dim=2,
        hidden_dim=4,
        max_steps=1,
        candidate_count=2,
        step_size=0.5,
        step_penalty=0.0,
        max_delta=1.0,
        retrieval_chunk=2,
    )
    state = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    users = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [-1.0, 0.0]]
    )
    expected = torch.topk(
        torch.nn.functional.normalize(state, dim=1)
        @ torch.nn.functional.normalize(users, dim=1).t(),
        k=2,
        dim=1,
    ).indices

    assert torch.equal(engine.legal_candidate_ids(state, users), expected)


def test_inference_rollout_rejects_oracle_inputs_and_does_not_compute_rewards(monkeypatch):
    engine = _engine()
    state = torch.zeros((1, 2))
    users = torch.eye(2).repeat(2, 1)[:3]

    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(state, user_bank=users, training=False, target_emb=torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(
            state,
            user_bank=users,
            training=False,
            positive_user_ids=[torch.tensor([0])],
        )

    monkeypatch.setattr(engine, "_training_reward", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference reward")))
    result = engine.rollout(state, user_bank=users, training=False)

    assert result.final_state.shape == (1, 2)
    assert all(torch.equal(reward, torch.zeros_like(reward)) for reward in result.trajectory["rewards"])


def test_end_action_freezes_state_and_suppresses_later_rewards(monkeypatch):
    engine = _engine(max_steps=3)
    initial = torch.tensor([[0.25, -0.25]])
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def choose_end(state, remaining_steps, candidates, **kwargs):
        batch = state.size(0)
        return (
            torch.full((batch,), candidates.size(1), dtype=torch.long),
            torch.zeros(batch),
            torch.zeros((batch, 1)),
            torch.zeros(batch),
        )

    monkeypatch.setattr(engine.policy, "action_value", choose_end)
    result = engine.rollout(
        initial,
        user_bank=users,
        training=True,
        target_emb=torch.zeros_like(initial),
        positive_user_ids=[torch.tensor([0])],
    )

    assert torch.equal(result.final_state, initial)
    assert result.trajectory["done"][0].tolist() == [True]
    assert all(torch.equal(reward, torch.zeros_like(reward)) for reward in result.trajectory["rewards"])


def test_training_reward_averages_all_positive_users():
    before = torch.tensor([[0.0, 0.0]])
    after = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[2.0, 0.0]])
    users = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

    gain = full_positive_score_gain(before, after, target, [torch.tensor([0, 1])], users)

    assert gain.item() == pytest.approx(2.0)


def test_target_history_exclusion_and_trust_projection_are_row_safe():
    histories = {0: {1, 2}, 1: {2, 3}}
    target_free = target_excluded_history(
        histories,
        target_item_ids=torch.tensor([2, 3]),
        selected_user_ids=torch.tensor([0, 1]),
    )
    projected = project_displacement(
        torch.zeros((1, 2)),
        torch.tensor([[3.0, 4.0]]),
        max_delta=2.0,
    )

    assert target_free == [{1}, {2}]
    assert torch.allclose(projected, torch.tensor([[1.2, 1.6]]))


def test_recppo_replays_detached_transitions_and_anchors_terminal_value():
    engine = _engine(max_steps=1)
    ppo = CleanRecPPO(
        engine.policy,
        replay_capacity=8,
        replay_batch_size=2,
        gamma=0.9,
        clip_ratio=0.2,
        value_weight=0.5,
        terminal_value_weight=1.0,
        entropy_weight=0.0,
    )
    old_log_prob = torch.tensor([-0.2, -0.3], requires_grad=True)
    trajectory = {
        "states": [torch.zeros((2, 2))],
        "next_states": [torch.zeros((2, 2))],
        "candidate_ids": [torch.tensor([[0, 1], [1, 2]])],
        "candidate_logit_bias": [torch.zeros((2, 2))],
        "actions": [torch.tensor([0, 2])],
        "rewards": [torch.tensor([[0.5], [0.0]])],
        "done": [torch.tensor([False, True])],
        "old_log_probs": [old_log_prob],
        "remaining_steps": [torch.ones((2, 1))],
        "next_remaining_steps": [torch.zeros((2, 1))],
        "terminal_states": [torch.zeros((2, 2))],
        "terminal_remaining_steps": [torch.zeros((2, 1))],
    }
    user_bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    loss = ppo.loss(trajectory, user_bank)
    loss.backward()

    assert torch.isfinite(loss)
    assert len(ppo.replay) == 2
    assert old_log_prob.grad is None
    assert ppo.last_stats["terminal_value_loss"] >= 0.0


def test_clean_item_bank_keeps_hot_teacher_vectors_and_calls_target_free_cold_rollout(monkeypatch):
    teacher = CleanTeacher(n_users=3, n_items=4, emb_dim=2)
    generator = ContentGenerator(content_dim=3, emb_dim=2, hidden_dim=4)
    engine = _engine(max_steps=1)
    content = torch.zeros((4, 3))
    teacher_bank = teacher.item_vectors().detach().clone()
    captured = {}

    def fake_rollout(initial_state, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(final_state=torch.tensor([[0.0, 3.0]]))

    monkeypatch.setattr(engine, "rollout", fake_rollout)
    item_bank = build_clean_item_bank(
        teacher,
        generator,
        engine,
        content,
        strict_cold_item_ids=torch.tensor([2]),
        user_history={0: {2}},
        policy_epoch=1,
    )

    assert torch.equal(item_bank[[0, 1, 3]], teacher_bank[[0, 1, 3]])
    assert torch.allclose(item_bank[2], torch.tensor([0.0, 1.0]))
    assert captured["training"] is False
    assert captured["target_emb"] is None
    assert captured["positive_user_ids"] is None


def test_policy_epoch_zero_bypasses_rollout_and_uses_normalized_generator_state(monkeypatch):
    teacher = CleanTeacher(n_users=3, n_items=4, emb_dim=2)
    generator = ContentGenerator(content_dim=3, emb_dim=2, hidden_dim=4)
    engine = _engine(max_steps=1)
    content = torch.zeros((4, 3))
    teacher_bank = teacher.item_vectors().detach().clone()
    with torch.no_grad():
        generator.net[0].weight.zero_()
        generator.net[0].bias.zero_()
        generator.net[2].weight.zero_()
        generator.net[2].bias.copy_(torch.tensor([3.0, 4.0]))

    def unexpected_rollout(*args, **kwargs):
        raise AssertionError("identity policy epoch must not call rollout")

    monkeypatch.setattr(engine, "rollout", unexpected_rollout)
    item_bank = build_clean_item_bank(
        teacher,
        generator,
        engine,
        content,
        strict_cold_item_ids=torch.tensor([2]),
        user_history={0: {2}},
        policy_epoch=0,
    )

    assert torch.equal(item_bank[[0, 1, 3]], teacher_bank[[0, 1, 3]])
    assert torch.allclose(item_bank[2], torch.tensor([0.6, 0.8]))


def test_clean_course_signal_uses_only_target_excluded_histories_in_policy_callbacks(monkeypatch):
    signal = CleanCourseSignal(
        item_concept_overlap=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        ),
        item_prereq_item_mat=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        item_prereq_item_count=torch.tensor([0.0, 1.0, 1.0]),
        item_popularity=torch.tensor([2.0, 2.0, 2.0]),
        bias_scale=0.2,
    )
    histories_seen = []
    engine = CleanUSIMEngine(
        emb_dim=2,
        hidden_dim=4,
        max_steps=1,
        candidate_count=2,
        step_size=0.5,
        step_penalty=0.0,
        max_delta=1.0,
        course_bias_fn=lambda ids, items, histories: histories_seen.extend(histories) or signal.candidate_bias(ids, items, histories),
        course_reward_fn=lambda ids, items, histories: histories_seen.extend(histories) or signal.reward(ids, items, histories),
    )
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def choose_first(state, remaining_steps, candidates, **kwargs):
        return (
            torch.zeros(state.size(0), dtype=torch.long),
            torch.zeros(state.size(0)),
            torch.zeros((state.size(0), 1)),
            torch.zeros(state.size(0)),
        )

    monkeypatch.setattr(engine.policy, "action_value", choose_first)
    engine.rollout(
        torch.zeros((1, 2)),
        user_bank=users,
        training=True,
        target_emb=torch.zeros((1, 2)),
        positive_user_ids=[torch.tensor([0])],
        item_ids=torch.tensor([2]),
        user_history={0: {1, 2}, 1: {2}},
    )

    assert histories_seen
    assert all(2 not in history for history in histories_seen)


def test_create_clean_engine_wires_the_optional_observable_course_signal(tmp_path):
    signal = CleanCourseSignal(
        item_concept_overlap=None,
        item_prereq_item_mat=None,
        item_prereq_item_count=None,
        item_popularity=torch.tensor([1.0, 2.0]),
    )
    config = CleanRunConfig(
        seed=7,
        data_dir=tmp_path / "data",
        split_dir=tmp_path / "split",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        emb_dim=2,
        hidden_dim=4,
        candidate_count=2,
        retrieval_chunk=2,
        max_steps=1,
        use_course_signal=True,
    )

    engine = create_clean_engine(config, course_signal=signal)

    assert engine.course_bias_fn is not None
    assert engine.course_reward_fn is not None


def test_clean_pipeline_writes_isolated_manifest_and_reads_test_after_selection(tmp_path):
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
    train = _frame([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)])
    val = _frame([(0, 0), (3, 3)])
    test = _frame([(1, 1), (0, 4)])
    train.to_pickle(split_dir / "static_train.pkl")
    val.to_pickle(split_dir / "static_val.pkl")
    test.to_pickle(split_dir / "static_test.pkl")
    config = CleanRunConfig(
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
        candidate_count=2,
        retrieval_chunk=2,
        max_steps=1,
        device="cpu",
    )

    result = run_clean_pipeline(config)
    manifest = json.loads((output_dir / "clean_manifest.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((output_dir / "final_metrics.json").read_text(encoding="utf-8"))
    expected_policy_mode = "identity_generator" if result["selected_policy_epoch"] == 0 else "ppo_rollout"

    assert result["selected_policy_epoch"] in {0, 1}
    assert manifest["legacy_warm_checkpoint"] is None
    assert manifest["random_id_dropout"] is False
    assert manifest["main_candidate_mode"] == "legal_state_retrieval"
    assert manifest["inference_oracle_access"] is False
    assert manifest["method_contract"]["evaluation"]["ranking_bank"] == "single_unified_catalog_bank"
    assert manifest["method_contract"]["inference"]["hot_route"] == "frozen_teacher_item_vectors"
    assert manifest["method_contract"]["inference"]["cold_route"] == "content_generator_then_target_free_policy"
    assert manifest["method_contract"]["policy_bounds"]["hot_items_mutable_in_policy"] is False
    assert manifest["test_loaded_after_policy_selection"] is True
    assert manifest["selected_policy_mode"] == expected_policy_mode
    assert (output_dir / "clean_partition.json").is_file()
    assert (checkpoint_dir / "teacher.pt").is_file()
    assert (checkpoint_dir / "generator.pt").is_file()
    assert (checkpoint_dir / "policy.pt").is_file()
    assert "validation" in final_metrics and "test" in final_metrics
    assert final_metrics["selected_policy_mode"] == expected_policy_mode


def test_v32_launcher_is_fresh_and_locks_clean_route_controls():
    launcher = Path("run_ckg_rl_usim_v32_clean_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v32_clean.py"' in launcher
    assert 'outputs\\ckg_rl_usim_v32_clean' in launcher
    assert 'checkpoints\\ckg_rl_usim_v32_clean' in launcher
    assert 'USIM_CLEAN_RANDOM_ID_DROPOUT' in launcher
    assert 'USIM_CLEAN_CANDIDATE_MODE' in launcher
    assert '--use-course-signal' in launcher
    assert '$outputRelative = "outputs\\ckg_rl_usim_v32_clean\\smoke_seed$Seed"' in launcher
    assert '$checkpointRelative = "checkpoints\\ckg_rl_usim_v32_clean\\smoke_seed$Seed"' in launcher
    assert '[string]$RunTag = ""' in launcher
    assert 'seed${Seed}_${RunTag}' in launcher
    assert 'USIM_ORIGINAL_V2' not in launcher
    assert 'USIM_V3_CORE' not in launcher
    assert 'USIM_FB_INIT_CKPT_DIR' not in launcher
