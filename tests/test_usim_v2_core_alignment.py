"""Regression tests for the isolated original-USIM core repair route."""

import torch
import pytest
from pathlib import Path

from fast3_delta.config import Fast3Config
from usim_feedback_fast3_content_delta_recovered_51ea_candidate import (
    Fast3FeedbackUSIM,
    _restore_original_v2_fresh_agent_state,
    _validate_original_v2_teacher_state,
)


def _v2_model(monkeypatch):
    monkeypatch.setenv("USIM_ORIGINAL_V2", "1")
    monkeypatch.setenv("USIM_ORIGINAL_V2_STEP_SIZE", "0.05")
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_USE_PSEUDO_COLD_TRAIN", "1")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MODE", "item_tail")
    monkeypatch.setenv("USIM_PSEUDO_COLD_RATIO", "0.50")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MIN_POP", "1")
    monkeypatch.setenv("USIM_TRAIN_FORCE_COLD", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    monkeypatch.setenv("USIM_AUX_WEIGHT", "0")
    monkeypatch.setenv("USIM_AUX_HOT_ONLY", "1")
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_USE_COURSE_REWARD", "0")
    monkeypatch.setenv("USIM_USE_PREREQ_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_SAGE_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_CGRC_RECON", "0")
    monkeypatch.setenv("USIM_USE_PAAC", "0")

    cfg = Fast3Config(n_users=4, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    cfg.cold_threshold = 1
    cfg.n_candidates = 3
    cfg.retrieve_top_m = 3
    model = Fast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model.item_popularity = torch.tensor([8.0, 8.0, 8.0])
    model._fixed_pseudo_cold_item_mask_cache = torch.tensor([False, True, False])
    return model


def test_v2_pseudocold_training_uses_behavior_target_and_only_refines_masked_rows(monkeypatch):
    model = _v2_model(monkeypatch)
    model.train()
    model._fixed_pseudo_cold_item_mask_cache = torch.tensor([False, False, True])
    base = torch.full((2, model.cfg.emb_dim), 0.25, dtype=torch.float32, requires_grad=True)
    behaviour = torch.full((2, model.cfg.emb_dim), 2.0, dtype=torch.float32, requires_grad=True)
    teacher_items = torch.arange(
        3 * model.cfg.emb_dim, dtype=torch.float32
    ).view(3, model.cfg.emb_dim)
    teacher_users = torch.full((4, model.cfg.emb_dim), 4.0, dtype=torch.float32)
    model._original_v2_teacher_item_emb = teacher_items
    model._original_v2_teacher_user_emb = teacher_users
    captured = {}

    def fake_get_item_vector(item_idx, llm_s, force_cold=False, disable_id_dropout=False):
        del item_idx, llm_s, disable_id_dropout
        captured["force_cold"] = force_cold.detach().clone()
        return base, behaviour, torch.zeros_like(base)

    def fake_episode(init_item_emb, target_emb=None, **kwargs):
        captured["episode_item_idx"] = kwargs["item_idx"].detach().clone()
        captured["target"] = target_emb.detach().clone()
        captured["oracle_user_idx"] = kwargs.get("oracle_user_idx")
        captured["oracle_user_emb"] = kwargs.get("oracle_user_emb")
        return init_item_emb + 2.0, {"rewards": []}, {"steps": 1}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_episode
    model.compute_ppo_loss = lambda trajectory: torch.zeros((), dtype=torch.float32)

    loss, _ = model(
        {"u": torch.tensor([0, 1]), "i": torch.tensor([0, 2])},
        torch.tensor([8.0, 8.0], dtype=torch.float32),
        torch.full((2,), -1.0, dtype=torch.float32),
    )

    assert torch.isfinite(loss)
    assert torch.equal(captured["force_cold"], torch.tensor([False, True]))
    assert captured["episode_item_idx"].tolist() == [2]
    assert torch.equal(captured["target"], teacher_items[2:3])
    assert captured["oracle_user_idx"].tolist() == [1]
    assert torch.equal(captured["oracle_user_emb"], teacher_users[1:2])


def test_v2_teacher_snapshot_freezes_iv_user_and_item_embeddings(monkeypatch):
    model = _v2_model(monkeypatch)
    item_before = model.item_id_emb.weight.detach().clone()
    user_before = model.user_proj(model.user_emb.weight).detach().clone()

    model.initialize_original_v2_teacher_()

    assert torch.equal(model._original_v2_teacher_item_emb, item_before)
    assert torch.equal(model._original_v2_teacher_user_emb, user_before)
    assert not model.item_id_emb.weight.requires_grad
    assert not model.user_emb.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in model.user_proj.parameters())


def test_v2_restores_a_fresh_policy_after_loading_the_warm_teacher(monkeypatch):
    model = _v2_model(monkeypatch)
    fresh_agent_state = {
        key: value.detach().clone() for key, value in model.agent.state_dict().items()
    }
    old_route_agent_state = {
        key: torch.full_like(value, 0.25)
        for key, value in model.agent.state_dict().items()
    }
    model.agent.load_state_dict(old_route_agent_state)

    _restore_original_v2_fresh_agent_state(model, fresh_agent_state)

    for key, expected in fresh_agent_state.items():
        assert torch.equal(model.agent.state_dict()[key], expected)


def test_v2_rejects_teacher_checkpoints_missing_or_shape_mismatching_iv_state(monkeypatch):
    model = _v2_model(monkeypatch)
    good = {key: value.detach().clone() for key, value in model.state_dict().items()}

    _validate_original_v2_teacher_state(model, good)

    missing = dict(good)
    del missing["item_id_emb.weight"]
    with pytest.raises(RuntimeError, match="item_id_emb.weight"):
        _validate_original_v2_teacher_state(model, missing)

    wrong_shape = dict(good)
    wrong_shape["user_emb.weight"] = wrong_shape["user_emb.weight"][:1]
    with pytest.raises(RuntimeError, match="user_emb.weight"):
        _validate_original_v2_teacher_state(model, wrong_shape)


def test_ppo_actor_gradient_treats_rollout_log_probs_as_old_policy(monkeypatch):
    model = _v2_model(monkeypatch)
    model.cfg.ppo_epochs = 1
    model.cfg.ppo_gamma = 0.0
    model.cfg.ppo_adv_norm = False
    model.cfg.ppo_coeffs = {"value": 0.0, "entropy": 0.0}
    torch.manual_seed(17)
    states = torch.randn(2, model.cfg.emb_dim)
    time_steps = torch.zeros((2, 1), dtype=torch.long)
    candidates = torch.randn(2, 3, model.cfg.emb_dim)
    actions = torch.tensor([0, 1], dtype=torch.long)
    _, old_log_probs, old_values, _ = model.agent.get_action_value(
        states,
        time_steps,
        candidates,
        action_idx=actions,
    )
    trajectory = {
        "rewards": [torch.ones((2, 1), dtype=torch.float32)],
        "log_probs": [old_log_probs],
        "values": [old_values],
        "states": [states.detach()],
        "time_steps": [time_steps],
        "candidates": [candidates.detach()],
        "actions": [actions],
    }

    model.compute_ppo_loss(trajectory).backward()

    actor_grad = model.agent.actor_head.weight.grad
    assert actor_grad is not None
    assert actor_grad.norm().item() > 0.0


def test_v2_transition_is_driven_by_the_selected_user_not_the_behavior_oracle(monkeypatch):
    model = _v2_model(monkeypatch)
    model.train()
    dim = model.cfg.emb_dim
    user_bank = torch.zeros((4, dim), dtype=torch.float32)
    user_bank[2, 0] = 1.0
    user_bank[3, 1] = 1.0

    def fake_candidates(item_emb, **kwargs):
        del item_emb, kwargs
        candidates = torch.zeros((1, 3, dim), dtype=torch.float32)
        candidate_ids = torch.tensor([[0, 1, 3]], dtype=torch.long)
        return candidates, candidate_ids, {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_action(current_h, time_step, candidates, fit_score=None, deterministic=False):
        del current_h, time_step, fit_score, deterministic
        action = torch.zeros((1,), dtype=torch.long)
        return action, candidates.new_zeros((1,)), candidates.new_zeros((1, 1)), candidates.new_zeros((1,))

    model.get_candidates = fake_candidates
    model._select_rollout_action = fake_action
    model._compute_course_reward_terms = lambda *args, **kwargs: {
        "prereq_gap": torch.zeros((1, 1)),
        "concept_bonus": torch.zeros((1, 1)),
        "difficulty_gap": torch.zeros((1, 1)),
        "redundant": torch.zeros((1, 1)),
    }

    initial = torch.zeros((1, dim), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((1, dim), dtype=torch.float32)
    target[0, 5] = 100.0
    final, _, _ = model.run_usim_episode(
        initial,
        target_emb=target,
        user_bank_raw=user_bank,
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([0.0]),
        oracle_user_idx=torch.tensor([2]),
        oracle_user_emb=user_bank[2:3],
    )

    expected = initial.detach().clone()
    expected[0, 0] = 0.05
    assert torch.allclose(final.detach(), expected)


def test_v2_bypasses_legacy_course_sampling_even_when_it_is_enabled(monkeypatch):
    model = _v2_model(monkeypatch)
    model.train()
    model.cfg.use_course_sample = True
    dim = model.cfg.emb_dim
    user_bank = torch.zeros((4, dim), dtype=torch.float32)
    user_bank[2, 0] = 1.0

    def fake_candidates(item_emb, **kwargs):
        del item_emb, kwargs
        return (
            torch.zeros((1, 3, dim), dtype=torch.float32),
            torch.tensor([[0, 1, 3]], dtype=torch.long),
            {"dup_rate": 0.0, "topm_coverage": 1.0},
        )

    def fake_action(current_h, time_step, candidates, fit_score=None, deterministic=False):
        del current_h, time_step, fit_score, deterministic
        action = torch.zeros((1,), dtype=torch.long)
        return action, candidates.new_zeros((1,)), candidates.new_zeros((1, 1)), candidates.new_zeros((1,))

    def fail_if_course_sampler_is_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("V2 must not mix CKG course sampling into USIM candidates")

    model.get_candidates = fake_candidates
    model._select_rollout_action = fake_action
    model._apply_course_sampling_bias = fail_if_course_sampler_is_called

    final, _, _ = model.run_usim_episode(
        torch.zeros((1, dim), dtype=torch.float32, requires_grad=True),
        target_emb=torch.ones((1, dim), dtype=torch.float32),
        user_bank_raw=user_bank,
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([0.0]),
        oracle_user_idx=torch.tensor([2]),
        oracle_user_emb=user_bank[2:3],
    )

    assert final.shape == (1, dim)


def test_original_v2_launcher_locks_the_isolated_usim_contract():
    launcher = Path(__file__).resolve().parents[1] / "run_usim_original_v2_seed2025.ps1"
    text = launcher.read_text(encoding="utf-8")

    for required in (
        '"USIM_ORIGINAL_V2" = "1"',
        '"USIM_ORIGINAL_V2_STEP_PENALTY" = "0.01"',
        '"USIM_ORIGINAL_V2_STEP_SIZE" = "0.05"',
        'ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"',
        'PseudoColdMode = "item_tail"',
        'TrainForceCold = $true',
        'UseCourseSample = $false',
        'UseCourseReward = $false',
        'AuxHotOnly = $true',
        'PpoLossWeight = 1.0',
        'RolloutPolicy = "ppo"',
        'CkgRlV1 = $false',
        'TeacherCheckpointDir',
        'InitCheckpointDir = $teacherCheckpointPath',
        'AutoResume = $false',
        'ForceFresh = $true',
        'outputs\\usim_original_v2',
        'checkpoints\\usim_original_v2',
    ):
        assert required in text


def test_v2_static_metrics_export_the_core_alignment_gates():
    source = (
        Path(__file__).resolve().parents[1]
        / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    ).read_text(encoding="utf-8")

    for required in (
        '"V2InitialTargetL2"',
        '"V2RolloutDeltaL2"',
        '"V2EmbeddingReward"',
        '"V2RecommendationReward"',
        'cand_info.get("v2_initial_target_l2", 0.0)',
        'cand_info.get("v2_rollout_delta_l2", 0.0)',
    ):
        assert required in source


def test_v2_manifest_records_teacher_and_transition_contract():
    source = (
        Path(__file__).resolve().parents[1]
        / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    ).read_text(encoding="utf-8")

    for required in (
        '"original_usim_v2"',
        '"original_usim_v2_step_size"',
        '"original_usim_v2_step_penalty"',
        '"original_usim_v2_teacher_checkpoint"',
    ):
        assert required in source
