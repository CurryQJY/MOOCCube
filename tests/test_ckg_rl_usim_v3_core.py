"""Behavior contracts for the CKG-RL V3 USIM-consistent simulation core."""

import pytest
import torch

from ckg_rl_usim_v3 import (
    CKGRLV3USIM,
    EndAwareRecActorCritic,
    V3ReplayBuffer,
    apply_usim_transition,
    full_positive_recommendation_gain,
)
from fast3_delta.config import Fast3Config


def _v3_model(monkeypatch, *, steps=2, n_users=4, candidates=3):
    monkeypatch.setenv("USIM_ORIGINAL_V2", "1")
    monkeypatch.setenv("USIM_USE_PSEUDO_COLD_TRAIN", "1")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MODE", "item_tail")
    monkeypatch.setenv("USIM_PSEUDO_COLD_RATIO", "0.50")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MIN_POP", "1")
    monkeypatch.setenv("USIM_TRAIN_FORCE_COLD", "1")
    monkeypatch.setenv("USIM_V3_STEP_SIZE", "0.05")
    monkeypatch.setenv("USIM_V3_STEP_PENALTY", "0.01")
    monkeypatch.setenv("USIM_V3_REPLAY_CAPACITY", "16")
    monkeypatch.setenv("USIM_V3_REPLAY_BATCH_SIZE", "2")
    cfg = Fast3Config(n_users=n_users, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.usim_steps = int(steps)
    cfg.n_candidates = int(candidates)
    cfg.retrieve_top_m = int(candidates)
    cfg.use_course_reward = False
    model = CKGRLV3USIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    return model


def test_active_user_transition_adds_only_selected_user_embedding():
    state = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    selected_user = torch.tensor([[2.0, -4.0]], dtype=torch.float32)

    next_state, done_after, active_user = apply_usim_transition(
        state,
        selected_user,
        action=torch.tensor([0]),
        done_before=torch.tensor([False]),
        end_action_index=3,
        step_size=0.05,
    )

    assert torch.allclose(next_state, torch.tensor([[0.1, 0.8]]))
    assert done_after.tolist() == [False]
    assert active_user.tolist() == [True]


def test_end_action_and_already_done_rows_leave_state_unchanged():
    state = torch.tensor([[0.3, -0.2], [1.0, 1.0]], dtype=torch.float32)
    selected_user = torch.tensor([[8.0, 8.0], [9.0, 9.0]], dtype=torch.float32)

    next_state, done_after, active_user = apply_usim_transition(
        state,
        selected_user,
        action=torch.tensor([2, 0]),
        done_before=torch.tensor([False, True]),
        end_action_index=2,
        step_size=0.05,
    )

    assert torch.equal(next_state, state)
    assert done_after.tolist() == [True, True]
    assert active_user.tolist() == [False, False]


def test_recommendation_gain_averages_errors_over_every_positive_user():
    before = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    after = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    teacher_item = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    positive_users = [torch.tensor([0, 1], dtype=torch.long)]
    teacher_user_bank = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)

    gain = full_positive_recommendation_gain(
        before,
        after,
        teacher_item,
        positive_users,
        teacher_user_bank,
    )

    assert torch.allclose(gain, torch.tensor([[2.0]]))


def test_actor_exposes_a_learned_terminal_action_in_deterministic_inference():
    actor = EndAwareRecActorCritic(embedding_dim=2, hidden_dim=4)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.end_head.bias.fill_(2.0)

    action, log_prob, value, entropy = actor.action_value(
        state=torch.zeros((1, 2)),
        remaining_steps=torch.tensor([[2.0]]),
        candidates=torch.zeros((1, 3, 2)),
        deterministic=True,
    )

    assert action.tolist() == [3]
    assert log_prob.shape == (1,)
    assert value.shape == (1, 1)
    assert entropy.shape == (1,)


def test_v31_actor_bias_changes_the_user_action_without_reordering():
    actor = EndAwareRecActorCritic(embedding_dim=2, hidden_dim=4)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.end_head.bias.fill_(-3.0)

    action, *_ = actor.action_value(
        torch.zeros((1, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 3, 2)),
        candidate_logit_bias=torch.tensor([[0.0, 1.0, 0.0]]),
        deterministic=True,
    )

    assert action.tolist() == [1]


def test_replay_buffer_stores_detached_rollout_tensors_and_done_flags():
    buffer = V3ReplayBuffer(capacity=4)
    state = torch.ones((2, 2), requires_grad=True)
    transition = {
        "state": state,
        "next_state": state + 1.0,
        "candidate_ids": torch.tensor([[0, 1], [1, 2]]),
        "candidate_logit_bias": torch.tensor([[0.0, 0.2], [0.1, -0.1]]),
        "action": torch.tensor([0, 2]),
        "reward": torch.tensor([[0.2], [0.0]]),
        "done": torch.tensor([False, True]),
        "old_log_prob": torch.tensor([-0.5, -0.7], requires_grad=True),
        "remaining_steps": torch.tensor([[2.0], [1.0]]),
        "next_remaining_steps": torch.tensor([[1.0], [0.0]]),
        "terminal_state": torch.zeros((2, 2)),
        "terminal_remaining_steps": torch.tensor([[1.0], [0.0]]),
    }

    buffer.append(transition)
    sample = buffer.sample(batch_size=2, device=torch.device("cpu"))

    assert len(buffer) == 2
    assert sample["state"].requires_grad is False
    assert sample["old_log_prob"].requires_grad is False
    assert sample["candidate_logit_bias"].requires_grad is False
    assert sorted(sample["done"].tolist()) == [False, True]


def test_v31_training_candidates_reserve_all_four_sources(monkeypatch):
    model = _v3_model(monkeypatch, n_users=30, candidates=20)
    model.train()
    model.cfg.feedback_course_sample_beta = 0.0
    state_top = torch.tensor([list(range(10, 30))], dtype=torch.long)
    residual_top = torch.tensor([list(range(0, 20))], dtype=torch.long)

    def fake_topk(query, *_args, **_kwargs):
        if torch.allclose(query, torch.zeros_like(query)):
            return state_top.to(query.device)
        return residual_top.to(query.device)

    monkeypatch.setattr(model, "_v3_topk_user_ids", fake_topk)
    _, candidate_ids, _ = model._v3_build_candidates(
        torch.zeros((1, model.cfg.emb_dim)),
        torch.randn(30, model.cfg.emb_dim),
        training=True,
        target_emb=torch.ones((1, model.cfg.emb_dim)),
        positive_user_ids=[torch.tensor([20, 21, 22, 23, 24, 25])],
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([0.0]),
        user_seen_items={},
    )

    chosen = set(candidate_ids[0].tolist())
    assert set(range(0, 6)).issubset(chosen)
    assert set(range(20, 26)).issubset(chosen)
    assert set(range(10, 16)).issubset(chosen)
    assert len(chosen) == 20


def test_v31_candidate_builder_uses_bounded_random_sampling(monkeypatch):
    model = _v3_model(monkeypatch, n_users=10_000, candidates=20)
    model.train()
    model.cfg.feedback_course_sample_beta = 0.0
    state_top = torch.arange(20, dtype=torch.long).view(1, -1)
    residual_top = torch.arange(20, 40, dtype=torch.long).view(1, -1)

    def fake_topk(query, *_args, **_kwargs):
        if torch.allclose(query, torch.zeros_like(query)):
            return state_top.to(query.device)
        return residual_top.to(query.device)

    def fail_full_permutation(*_args, **_kwargs):
        raise AssertionError("candidate construction must not materialize a full user permutation")

    monkeypatch.setattr(model, "_v3_topk_user_ids", fake_topk)
    monkeypatch.setattr(torch, "randperm", fail_full_permutation)
    _, candidate_ids, _ = model._v3_build_candidates(
        torch.zeros((1, model.cfg.emb_dim)),
        torch.randn(10_000, model.cfg.emb_dim),
        training=True,
        target_emb=torch.ones((1, model.cfg.emb_dim)),
        positive_user_ids=[torch.tensor([40])],
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([0.0]),
        user_seen_items={},
    )

    assert candidate_ids.shape == (1, 20)
    assert len(set(candidate_ids[0].tolist())) == 20


def test_v31_candidate_builder_draws_random_candidates_once_per_batch(monkeypatch):
    model = _v3_model(monkeypatch, n_users=100, candidates=20)
    model.train()
    model.cfg.feedback_course_sample_beta = 0.0
    state_top = torch.arange(20, dtype=torch.long).view(1, -1).expand(3, -1)
    residual_top = torch.arange(20, 40, dtype=torch.long).view(1, -1).expand(3, -1)
    original_randint = torch.randint
    calls = []

    def fake_topk(query, *_args, **_kwargs):
        if torch.allclose(query, torch.zeros_like(query)):
            return state_top.to(query.device)
        return residual_top.to(query.device)

    def capture_randint(low, high, size, *args, **kwargs):
        calls.append(tuple(size))
        return original_randint(low, high, size, *args, **kwargs)

    monkeypatch.setattr(model, "_v3_topk_user_ids", fake_topk)
    monkeypatch.setattr(torch, "randint", capture_randint)
    model._v3_build_candidates(
        torch.zeros((3, model.cfg.emb_dim)),
        torch.randn(100, model.cfg.emb_dim),
        training=True,
        target_emb=torch.ones((3, model.cfg.emb_dim)),
        positive_user_ids=[torch.tensor([40]), torch.tensor([41]), torch.tensor([42])],
        item_idx=torch.tensor([1, 1, 1]),
        target_pop=torch.zeros(3),
        user_seen_items={},
    )

    assert calls == [(3, 160)]


def test_v31_episode_reports_candidate_support_and_ckg_bias(monkeypatch):
    model = _v3_model(monkeypatch, steps=1, n_users=30, candidates=20)
    model.train()
    model.cfg.feedback_course_sample_beta = 0.2
    model._v3_train_item_users = {1: torch.tensor([20, 21, 22, 23, 24, 25])}
    state_top = torch.tensor([list(range(10, 30))], dtype=torch.long)
    residual_top = torch.tensor([list(range(0, 20))], dtype=torch.long)

    def fake_topk(query, *_args, **_kwargs):
        if torch.allclose(query, torch.zeros_like(query)):
            return state_top.to(query.device)
        return residual_top.to(query.device)

    def course_fit(candidate_ids, **_kwargs):
        return torch.linspace(-1.0, 1.0, candidate_ids.size(1)).view(1, -1)

    def choose_end(state, remaining_steps, candidates, **_kwargs):
        return (
            torch.full((state.size(0),), candidates.size(1), dtype=torch.long),
            torch.zeros(state.size(0)),
            torch.zeros((state.size(0), 1)),
            torch.zeros(state.size(0)),
        )

    monkeypatch.setattr(model, "_v3_topk_user_ids", fake_topk)
    monkeypatch.setattr(model, "_compute_candidate_course_fit", course_fit)
    monkeypatch.setattr(model.agent, "action_value", choose_end)
    _, _, stats = model.run_usim_episode(
        torch.zeros((1, model.cfg.emb_dim)),
        target_emb=torch.ones((1, model.cfg.emb_dim)),
        user_bank_raw=torch.randn(30, model.cfg.emb_dim),
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([0.0]),
        user_seen_items={},
    )

    assert stats["v3_train_residual_share"] == pytest.approx(0.30)
    assert stats["v3_train_positive_share"] == pytest.approx(0.30)
    assert stats["v3_train_state_share"] == pytest.approx(0.30)
    assert stats["v3_train_random_share"] == pytest.approx(0.10)
    assert stats["v3_course_logit_bias_abs"] > 0.0


def test_v3_inference_candidate_builder_receives_no_teacher_or_positive_users(monkeypatch):
    model = _v3_model(monkeypatch)
    model.eval()
    user_bank = torch.arange(4 * model.cfg.emb_dim, dtype=torch.float32).view(4, model.cfg.emb_dim)
    captured = {}

    def fake_candidates(current_h, user_bank_raw, **kwargs):
        captured["training"] = kwargs["training"]
        captured["target_emb"] = kwargs["target_emb"]
        captured["positive_user_ids"] = kwargs["positive_user_ids"]
        candidates = user_bank_raw[:3].unsqueeze(0).expand(current_h.size(0), -1, -1)
        candidate_ids = torch.tensor([[0, 1, 2]], dtype=torch.long).expand(current_h.size(0), -1)
        return candidates, candidate_ids, None

    def choose_end(state, remaining_steps, candidates, **kwargs):
        batch = state.size(0)
        return (
            torch.full((batch,), candidates.size(1), dtype=torch.long),
            torch.zeros(batch),
            torch.zeros((batch, 1)),
            torch.zeros(batch),
        )

    monkeypatch.setattr(model, "_v3_build_candidates", fake_candidates)
    monkeypatch.setattr(model.agent, "action_value", choose_end)

    result = model.infer_refined_item_vectors(
        torch.tensor([0]),
        force_cold=True,
        user_bank_raw=user_bank,
    )

    assert result.shape == (1, model.cfg.emb_dim)
    assert captured == {
        "training": False,
        "target_emb": None,
        "positive_user_ids": None,
    }


def test_v3_recppo_replays_detached_rollout_data_and_anchors_terminal_value(monkeypatch):
    model = _v3_model(monkeypatch)
    model.train()
    with torch.no_grad():
        for parameter in model.agent.parameters():
            parameter.zero_()
        model.agent.critic_head.bias.fill_(2.0)
    old_log_prob = torch.tensor([-0.4, -0.6], requires_grad=True)
    trajectory = {
        "states": [torch.zeros((2, model.cfg.emb_dim))],
        "next_states": [torch.zeros((2, model.cfg.emb_dim))],
        "candidate_ids": [torch.tensor([[0, 1, 2], [1, 2, 3]])],
        "candidate_logit_bias": [torch.zeros((2, 3))],
        "actions": [torch.tensor([0, 3])],
        "rewards": [torch.tensor([[0.2], [0.0]])],
        "done": [torch.tensor([False, True])],
        "old_log_probs": [old_log_prob],
        "remaining_steps": [torch.tensor([[2.0], [1.0]])],
        "next_remaining_steps": [torch.tensor([[1.0], [0.0]])],
        "terminal_states": [torch.zeros((2, model.cfg.emb_dim))],
        "terminal_remaining_steps": [torch.tensor([[1.0], [0.0]])],
    }

    loss = model.compute_ppo_loss(trajectory)
    loss.backward()

    assert torch.isfinite(loss)
    assert len(model.v3_replay) == 2
    assert old_log_prob.grad is None
    assert model.v3_last_ppo_stats["terminal_value_loss"] == pytest.approx(4.0)
    assert model._v3_sync_target_on_next_forward is True


def test_v31_replay_uses_the_same_detached_candidate_bias_for_ppo(monkeypatch):
    model = _v3_model(monkeypatch)
    model.train()
    trajectory = {
        "states": [torch.zeros((2, model.cfg.emb_dim))],
        "next_states": [torch.zeros((2, model.cfg.emb_dim))],
        "candidate_ids": [torch.tensor([[0, 1, 2], [1, 2, 3]])],
        "candidate_logit_bias": [torch.tensor([[0.0, 0.4, -0.4], [0.1, 0.0, -0.1]])],
        "actions": [torch.tensor([1, 0])],
        "rewards": [torch.tensor([[0.2], [0.0]])],
        "done": [torch.tensor([False, True])],
        "old_log_probs": [torch.tensor([-0.4, -0.6])],
        "remaining_steps": [torch.tensor([[2.0], [1.0]])],
        "next_remaining_steps": [torch.tensor([[1.0], [0.0]])],
        "terminal_states": [torch.zeros((2, model.cfg.emb_dim))],
        "terminal_remaining_steps": [torch.tensor([[1.0], [0.0]])],
    }
    observed = []

    def capture_actor(state, remaining_steps, candidates, *, candidate_logit_bias, action=None, **_kwargs):
        observed.append(candidate_logit_bias.detach().clone())
        scalar = next(model.parameters()).reshape(-1)[0]
        resolved_action = (
            torch.zeros(state.size(0), dtype=torch.long, device=state.device)
            if action is None
            else action
        )
        return (
            resolved_action,
            scalar.expand(state.size(0)),
            scalar.expand(state.size(0), 1),
            scalar.expand(state.size(0)),
        )

    monkeypatch.setattr(model.agent, "action_value", capture_actor)
    loss = model.compute_ppo_loss(trajectory)

    expected_rows = {tuple(row.tolist()) for row in trajectory["candidate_logit_bias"][0]}
    assert torch.isfinite(loss)
    assert any({tuple(row.tolist()) for row in value} == expected_rows for value in observed)


def test_v3_adds_course_reward_from_selected_user_history_without_oracle_leak(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.train()
    model.cfg.use_course_reward = True
    model.cfg.feedback_course_concept_weight = 0.5
    model.cfg.feedback_course_prereq_weight = 0.0
    model.cfg.feedback_course_difficulty_weight = 0.0
    model.cfg.feedback_course_redundant_weight = 0.0
    model._v3_train_item_users = {1: torch.tensor([0], dtype=torch.long)}
    captured = {}

    def course_terms(selected_user_ids, *, item_idx, target_pop, user_seen_items):
        captured["selected_user_ids"] = selected_user_ids.detach().clone()
        captured["item_idx"] = item_idx.detach().clone()
        captured["target_pop"] = target_pop.detach().clone()
        captured["history"] = user_seen_items
        zeros = torch.zeros((1, 1))
        return {
            "concept_bonus": torch.full((1, 1), 2.0),
            "prereq_gap": zeros,
            "difficulty_gap": zeros,
            "redundant": zeros,
        }

    def choose_user(state, remaining_steps, candidates, **kwargs):
        return (
            torch.zeros(state.size(0), dtype=torch.long),
            torch.zeros(state.size(0)),
            torch.zeros((state.size(0), 1)),
            torch.zeros(state.size(0)),
        )

    monkeypatch.setattr(model, "_compute_course_reward_terms", course_terms)
    monkeypatch.setattr(model.agent, "action_value", choose_user)
    _, trajectory, _ = model.run_usim_episode(
        torch.zeros((1, model.cfg.emb_dim)),
        target_emb=torch.zeros((1, model.cfg.emb_dim)),
        user_bank_raw=torch.zeros((4, model.cfg.emb_dim)),
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([5.0]),
        user_seen_items={0: {1}},
    )

    assert trajectory["rewards"][0].item() == pytest.approx(0.99)
    assert captured["selected_user_ids"].tolist() == [0]
    assert captured["item_idx"].tolist() == [1]
    assert captured["target_pop"].tolist() == [5.0]
    assert captured["history"] == {0: {1}}


def test_v31_inference_passes_observable_ckg_bias_to_actor(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.eval()
    model.cfg.feedback_course_sample_beta = 0.2
    captured = {}

    def course_fit(candidate_ids, *, item_idx, target_pop=None, user_seen_items=None):
        captured["candidate_ids"] = candidate_ids.detach().clone()
        captured["item_idx"] = item_idx.detach().clone()
        captured["target_pop"] = target_pop
        captured["history"] = user_seen_items
        return torch.tensor([[0.0, 2.0, -2.0]], dtype=torch.float32)

    def capture_actor(state, remaining_steps, candidates, *, candidate_logit_bias, **_kwargs):
        captured["bias"] = candidate_logit_bias.detach().clone()
        return (
            torch.full((state.size(0),), candidates.size(1), dtype=torch.long),
            torch.zeros(state.size(0)),
            torch.zeros((state.size(0), 1)),
            torch.zeros(state.size(0)),
        )

    monkeypatch.setattr(model, "_compute_candidate_course_fit", course_fit)
    monkeypatch.setattr(model.agent, "action_value", capture_actor)
    model.infer_refined_item_vectors(
        torch.tensor([0]),
        force_cold=True,
        user_bank_raw=torch.zeros((4, model.cfg.emb_dim)),
        user_seen_items={0: {1}},
    )

    assert captured["candidate_ids"].shape == (1, 3)
    assert captured["item_idx"].tolist() == [0]
    assert captured["target_pop"] is None
    assert captured["history"] == {0: {1}}
    assert torch.allclose(captured["bias"], torch.tensor([[0.0, 0.2, -0.2]]))


def test_v3_parent_forward_routes_only_pseudocold_rows_to_teacher_supervision(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.train()
    model._fixed_pseudo_cold_item_mask_cache = torch.tensor([False, False, True])
    model.item_popularity = torch.tensor([8.0, 8.0, 8.0])
    teacher_items = torch.arange(3 * model.cfg.emb_dim, dtype=torch.float32).view(3, model.cfg.emb_dim)
    teacher_users = torch.arange(4 * model.cfg.emb_dim, dtype=torch.float32).view(4, model.cfg.emb_dim)
    model._original_v2_teacher_item_emb = teacher_items
    model._original_v2_teacher_user_emb = teacher_users
    captured = {}

    def fake_episode(init_item_emb, target_emb=None, **kwargs):
        captured["item_idx"] = kwargs["item_idx"].detach().clone()
        captured["target_emb"] = target_emb.detach().clone()
        captured["oracle_user_idx"] = kwargs["oracle_user_idx"].detach().clone()
        return init_item_emb, {"rewards": []}, {"steps": 0}

    monkeypatch.setattr(model, "run_usim_episode", fake_episode)
    monkeypatch.setattr(model, "compute_ppo_loss", lambda trajectory: torch.zeros(()))

    loss, _ = model(
        {"u": torch.tensor([0, 1]), "i": torch.tensor([0, 2])},
        torch.tensor([8.0, 8.0]),
        torch.full((2,), -1.0),
        user_bank_raw=teacher_users,
        user_seen_items={},
    )

    assert torch.isfinite(loss)
    assert captured["item_idx"].tolist() == [2]
    assert torch.equal(captured["target_emb"], teacher_items[2:3])
    assert captured["oracle_user_idx"].tolist() == [1]


def test_v3_chunked_retrieval_matches_full_user_bank_topk():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    user_bank = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    expected = torch.topk(
        torch.nn.functional.normalize(query, dim=1)
        @ torch.nn.functional.normalize(user_bank, dim=1).t(),
        k=2,
        dim=1,
    ).indices

    actual = CKGRLV3USIM._v3_topk_user_ids(
        query,
        user_bank,
        count=2,
        chunk_size=2,
    )

    assert torch.equal(actual, expected)


def test_v3_warm_checkpoint_loader_preserves_teacher_weights_and_skips_legacy_actor_shape(monkeypatch):
    model = _v3_model(monkeypatch)
    teacher_items = torch.full_like(model.item_id_emb.weight, 3.0)
    teacher_users = torch.full_like(model.user_emb.weight, 4.0)
    legacy_state = {
        "item_id_emb.weight": teacher_items,
        "user_emb.weight": teacher_users,
        "agent.common.0.weight": torch.zeros((256, model.cfg.emb_dim + 5)),
    }

    model.load_state_dict(legacy_state, strict=False)

    assert torch.equal(model.item_id_emb.weight, teacher_items)
    assert torch.equal(model.user_emb.weight, teacher_users)


def test_v3_uses_frozen_teacher_user_bank_for_teacher_supervised_rollouts(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.train()
    model._v3_train_item_users = {1: torch.tensor([0], dtype=torch.long)}
    teacher_bank = torch.full((4, model.cfg.emb_dim), 7.0)
    model._original_v2_teacher_user_emb = teacher_bank
    captured = {}

    def capture_candidates(current_h, user_bank_raw, **kwargs):
        captured["bank"] = user_bank_raw.detach().clone()
        candidates = user_bank_raw[:3].unsqueeze(0).expand(current_h.size(0), -1, -1)
        candidate_ids = torch.tensor([[0, 1, 2]], dtype=torch.long).expand(current_h.size(0), -1)
        return candidates, candidate_ids, None

    def choose_end(state, remaining_steps, candidates, **kwargs):
        return (
            torch.full((state.size(0),), candidates.size(1), dtype=torch.long),
            torch.zeros(state.size(0)),
            torch.zeros((state.size(0), 1)),
            torch.zeros(state.size(0)),
        )

    monkeypatch.setattr(model, "_v3_build_candidates", capture_candidates)
    monkeypatch.setattr(model.agent, "action_value", choose_end)
    model.run_usim_episode(
        torch.zeros((1, model.cfg.emb_dim)),
        target_emb=torch.zeros((1, model.cfg.emb_dim)),
        user_bank_raw=torch.zeros((4, model.cfg.emb_dim)),
        item_idx=torch.tensor([1]),
        target_pop=torch.tensor([5.0]),
        user_seen_items={0: {1}},
    )

    assert torch.equal(captured["bank"], teacher_bank)


def test_v3_real_cpu_parent_forward_and_recppo_complete_one_pseudocold_batch(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.train()
    model.cfg.feedback_course_sample_beta = 0.0
    model._fixed_pseudo_cold_item_mask_cache = torch.tensor([False, False, True])
    model.item_popularity = torch.tensor([8.0, 8.0, 8.0])
    model._original_v2_teacher_item_emb = torch.randn(3, model.cfg.emb_dim)
    model._original_v2_teacher_user_emb = torch.randn(4, model.cfg.emb_dim)

    loss, info = model(
        {"u": torch.tensor([0, 1]), "i": torch.tensor([0, 2])},
        torch.tensor([8.0, 8.0]),
        torch.full((2,), -1.0),
        user_seen_items={0: {0}, 1: {2}},
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert info["pseudo_cold_count"] == 1
    assert len(model.v3_replay) >= 1
    assert model.v3_last_ppo_stats["replay_size"] >= 1.0
