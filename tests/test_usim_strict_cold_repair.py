from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast3_delta.config import FeedbackConfig
import fast3_delta.eval as eval_mod
from fast3_delta.eval import build_eval_item_vecs, build_eval_pos_item_vecs
from usim_feedback_fast3_content_delta import FastFeedbackUSIM
import usim_feedback_fast3_content_delta_repaired as repaired_mod
from usim_feedback_fast3_content_delta_repaired import (
    RepairedFast3Config,
    RepairedFast3FeedbackUSIM,
    repaired_strict_cold_item_mask,
)


class _FakeEvalConfig:
    n_items = 4
    emb_dim = 2
    cold_threshold = 1
    content_delta_cold_only = False
    content_delta_eval_bank_mode = "auto"
    legacy_train_protocol = False
    use_usim_refined_eval = True


class _FakeEvalModel:
    def __init__(self):
        self.cfg = _FakeEvalConfig()
        self.training = True
        self.item_popularity = torch.tensor([5.0, 0.0, 3.0, 0.0])
        self.base = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.8],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.refined = {
            1: torch.tensor([0.0, -1.0], dtype=torch.float32),
            3: torch.tensor([-1.0, 0.0], dtype=torch.float32),
        }

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def get_item_vector(self, idx_batch, llm_batch, force_cold=False):
        return self.base[idx_batch.cpu()].to(idx_batch.device), None, None

    def infer_refined_item_vectors(self, item_idx, llm_s=None, item_batch=1024, force_cold=True):
        rows = [self.refined[int(idx)] for idx in item_idx.detach().cpu().tolist()]
        return torch.stack(rows, dim=0).to(item_idx.device)


def test_eval_item_banks_replace_only_strict_cold_rows_with_refined_vectors():
    model = _FakeEvalModel()

    banks = build_eval_item_vecs(model, torch.device("cpu"), llm_scores=None, item_batch=2)

    for key in ("cold", "hot", "all"):
        bank = banks[key]
        assert torch.allclose(bank[0], model.base[0])
        assert torch.allclose(bank[2], model.base[2])
        assert torch.allclose(bank[1], model.refined[1])
        assert torch.allclose(bank[3], model.refined[3])


def test_eval_positive_vectors_use_refined_state_for_strict_cold_items():
    model = _FakeEvalModel()
    item_idx = torch.tensor([1, 2], dtype=torch.long)
    llm_s = torch.full((2,), -1.0, dtype=torch.float32)
    pop_sel = torch.tensor([0.0, 3.0], dtype=torch.float32)

    pos_vec = build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type="all")

    assert torch.allclose(pos_vec[0], model.refined[1])
    assert torch.allclose(pos_vec[1], model.base[2])


def test_legacy_train_protocol_keeps_eval_vectors_unrefined():
    model = _FakeEvalModel()
    model.cfg.legacy_train_protocol = True
    item_idx = torch.tensor([1], dtype=torch.long)
    llm_s = torch.full((1,), -1.0, dtype=torch.float32)
    pop_sel = torch.tensor([0.0], dtype=torch.float32)

    pos_vec = build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type="cold")

    assert torch.allclose(pos_vec[0], model.base[1])


def test_refined_eval_requires_item_popularity_when_enabled(monkeypatch):
    monkeypatch.setattr(eval_mod, "strict_cold_item_mask", repaired_strict_cold_item_mask)
    model = _FakeEvalModel()
    model.item_popularity = None

    with pytest.raises(RuntimeError, match="item_popularity"):
        build_eval_item_vecs(model, torch.device("cpu"), llm_scores=None, item_batch=2)


def test_repaired_defaults_enable_recppo_as_main_component(monkeypatch):
    monkeypatch.delenv("USIM_PPO_LOSS_WEIGHT", raising=False)
    monkeypatch.delenv("USIM_ROLLOUT_POLICY", raising=False)
    monkeypatch.delenv("USIM_USE_CONTENT_DELTA", raising=False)

    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    root = Path(__file__).resolve().parents[1]
    runner_text = (root / "run_usim_feedback_fast3_content_delta_repaired_static.ps1").read_text(encoding="utf-8")
    legacy_runner_text = (root / "run_usim_feedback_fast3_content_delta_static.ps1").read_text(encoding="utf-8")

    assert cfg.ppo_loss_weight == 1.0
    assert cfg.rollout_policy == "ppo"
    assert cfg.recppo_enabled is True
    assert cfg.use_content_delta is False
    assert cfg.recppo_terminal_value_weight > 0.0
    assert cfg.recppo_behavior_ce_weight >= 0.2
    assert not hasattr(cfg, "recppo_behavior_ce_final_weight")
    assert not hasattr(cfg, "recppo_behavior_ce_anneal_epochs")
    assert not hasattr(cfg, "recppo_embedding_gain_weight")
    assert not hasattr(cfg, "recppo_course_reward_scale")
    assert not hasattr(cfg, "recppo_course_reward_clip")
    assert cfg.recppo_teacher_force_behavior is False
    assert cfg.feedback_course_match_exclude_target is True
    assert cfg.reward_step_cost <= 0.01
    assert cfg.recppo_min_steps >= 1
    assert '$runnerArgs["PpoLossWeight"] = 1.0' in runner_text
    assert '$runnerArgs["RolloutPolicy"] = "ppo"' in runner_text
    assert '$runnerArgs["UseContentDelta"] = $false' in runner_text
    assert '$runnerArgs["Epochs"] = 30' in runner_text
    assert '$runnerArgs["Patience"] = 12' in runner_text
    assert '$runnerArgs["PseudoColdMode"] = "all_eligible"' in runner_text
    assert '$runnerArgs["PseudoColdRatio"] = 1.0' in runner_text
    assert "RecPpoBehaviorCeFinalWeight" not in runner_text
    assert "RecPpoBehaviorCeAnnealEpochs" not in runner_text
    assert "RecPpoEmbeddingGainWeight" not in runner_text
    assert "RecPpoCourseRewardScale" not in runner_text
    assert "RecPpoCourseRewardClip" not in runner_text
    assert '$runnerArgs["PseudoColdMinPop"] = 1' in runner_text
    assert '$runnerArgs["RlResidualScale"] = 0.30' in runner_text
    assert '$env:PYTHONHASHSEED = "0"' in runner_text
    assert '$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"' in runner_text
    assert 'USIM_RECPPO_STRICT_DETERMINISM' in runner_text
    assert 'USIM_RECPPO_EARLY_STOP_MODE' in runner_text
    assert '$recppoTrackedEnv = @(' in runner_text
    assert '$recppoOriginalEnv' in runner_text
    assert 'finally {' in runner_text
    assert "& $runner @runnerArgs" in runner_text
    assert '"usim_feedback_fast3_content_delta_repaired.py"' in runner_text
    assert "[double]$PpoLossWeight = 1.0" in legacy_runner_text
    assert '[string]$RolloutPolicy = "ppo"' in legacy_runner_text


def test_zero_ppo_weight_disables_recppo_phase(monkeypatch):
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_ROLLOUT_POLICY", "ppo")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))

    model._activate_recppo_phase()

    assert cfg.recppo_enabled is False
    assert model._recppo_phase_state.item() == 0
    assert model.user_emb.weight.requires_grad is True


def test_repaired_setup_seed_enables_strict_determinism(monkeypatch):
    calls = {}

    def fake_legacy_seed(seed):
        calls["seed"] = seed

    def fake_deterministic(enabled, warn_only=False):
        calls["deterministic"] = (enabled, warn_only)

    monkeypatch.setenv("USIM_RECPPO_STRICT_DETERMINISM", "1")
    monkeypatch.setattr(repaired_mod, "_legacy_setup_seed", fake_legacy_seed)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", fake_deterministic)

    repaired_mod.repaired_setup_seed(2025)

    assert calls == {"seed": 2025, "deterministic": (True, False)}


def test_repaired_main_seeds_before_delegating(monkeypatch):
    calls = []
    monkeypatch.setenv("USIM_STATIC_SEED", "2026")
    monkeypatch.setattr(repaired_mod, "install_repaired_bindings", lambda: calls.append("install"))
    monkeypatch.setattr(repaired_mod, "repaired_setup_seed", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(repaired_mod.legacy, "main", lambda: calls.append("main") or 0)

    assert repaired_mod.main() == 0
    assert calls == ["install", ("seed", 2026), "main"]


def test_infer_refined_item_vectors_runs_cold_start_episode_without_behavior_target(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._activate_recppo_phase()

    base = torch.full((2, cfg.emb_dim), 0.25, dtype=torch.float32)
    captured = {}

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        captured["force_cold"] = force_cold
        return base.clone(), torch.full_like(base, 9.0), torch.full_like(base, -3.0)

    def fake_run_usim_episode(init_item_emb, target_emb=None, **kwargs):
        captured["target_emb"] = target_emb
        captured["item_idx"] = kwargs.get("item_idx")
        return init_item_emb + 1.0, {"rewards": []}, {"steps": 0}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_run_usim_episode

    refined = model.infer_refined_item_vectors(torch.tensor([1, 2]), item_batch=16)

    assert captured["force_cold"] is True
    assert captured["target_emb"] is None
    assert captured["item_idx"].tolist() == [1, 2]
    expected = model._recppo_blend_state(base, base + 1.0)
    assert torch.allclose(refined, expected)


def test_warmup_inference_bypasses_untrained_policy(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    base = torch.full((1, cfg.emb_dim), 0.25, dtype=torch.float32)

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        return base.clone(), torch.full_like(base, 9.0), torch.full_like(base, -3.0)

    def fake_run_usim_episode(init_item_emb, target_emb=None, **kwargs):
        return init_item_emb + 10.0, {"rewards": []}, {"steps": 0}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_run_usim_episode

    refined = model.infer_refined_item_vectors(torch.tensor([1]), item_batch=16)

    assert torch.equal(refined, base)


def test_usim_transition_uses_selected_user_not_behavior_target(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.usim_lr = 0.5
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    selected_user = torch.ones((1, 1, cfg.emb_dim), dtype=torch.float32)

    def fake_get_candidates(*args, **kwargs):
        stats = {
            "dup_rate": 0.0,
            "topm_coverage": 1.0,
        }
        return selected_user, torch.tensor([[0]], dtype=torch.long), stats

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return (
            torch.tensor([0], dtype=torch.long),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select

    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    target = torch.full((1, cfg.emb_dim), 10.0, dtype=torch.float32)

    final_h, _, _ = model.run_usim_episode(init, target_emb=target, item_idx=torch.tensor([0]))

    assert torch.allclose(final_h, torch.full_like(final_h, 0.5))


def test_reward_is_delta_gain_minus_step_cost_without_absolute_distance(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.usim_lr = 0.0
    cfg.reward_gain_weight = 1.0
    cfg.reward_step_cost = 0.25
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    zero_user = torch.zeros((1, 1, cfg.emb_dim), dtype=torch.float32)

    def fake_get_candidates(*args, **kwargs):
        return zero_user, torch.tensor([[0]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return (
            torch.tensor([0], dtype=torch.long),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select

    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    final_h, trajectory, _ = model.run_usim_episode(init, target_emb=init.clone(), item_idx=torch.tensor([0]))

    assert torch.allclose(final_h, init)
    assert torch.allclose(trajectory["rewards"][0], torch.tensor([[-0.25]], dtype=torch.float32))


def test_recppo_episode_records_next_state_target_and_behavior_action(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.usim_lr = 0.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._recppo_behavior_user_idx = torch.tensor([1], dtype=torch.long)

    def fake_get_candidates(*args, **kwargs):
        candidates = torch.zeros((1, 2, cfg.emb_dim), dtype=torch.float32)
        return candidates, torch.tensor([[0, 2]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return (
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select

    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    target = torch.ones((1, cfg.emb_dim), dtype=torch.float32)
    _, trajectory, _ = model.run_usim_episode(init, target_emb=target, item_idx=torch.tensor([0]))

    assert torch.allclose(trajectory["target_emb"], target)
    assert len(trajectory["next_states"]) == 1
    assert len(trajectory["next_time_steps"]) == 1
    assert trajectory["dones"][0].item() is True
    assert trajectory["behavior_actions"][0].tolist() == [1]


def test_recppo_training_episode_teacher_forces_behavior_action(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.usim_lr = 1.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model.train()
    model.cfg.recppo_teacher_force_behavior = True
    model._recppo_behavior_user_idx = torch.tensor([1], dtype=torch.long)

    candidates = torch.stack(
        [
            torch.zeros((cfg.emb_dim,), dtype=torch.float32),
            torch.ones((cfg.emb_dim,), dtype=torch.float32),
        ],
        dim=0,
    ).view(1, 2, cfg.emb_dim)

    def fake_get_candidates(*args, **kwargs):
        return candidates, torch.tensor([[0, 1]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return (
            torch.tensor([0], dtype=torch.long),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select

    final_h, trajectory, _ = model.run_usim_episode(
        torch.zeros((1, cfg.emb_dim), dtype=torch.float32),
        target_emb=None,
        item_idx=torch.tensor([0]),
    )

    assert torch.allclose(final_h, torch.ones_like(final_h))
    assert trajectory["actions"][0].tolist() == [1]


def test_recppo_loss_has_terminal_value_and_behavior_supervision(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.recppo_terminal_value_weight = 0.5
    cfg.recppo_behavior_ce_weight = 0.5
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._recppo_behavior_user_idx = torch.tensor([1], dtype=torch.long)

    def fake_get_candidates(*args, **kwargs):
        candidates = torch.randn((1, 2, cfg.emb_dim), dtype=torch.float32)
        return candidates, torch.tensor([[0, 2]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    model.get_candidates = fake_get_candidates

    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    target = torch.ones((1, cfg.emb_dim), dtype=torch.float32)
    _, trajectory, _ = model.run_usim_episode(init, target_emb=target, item_idx=torch.tensor([0]))
    loss = model.compute_ppo_loss(trajectory)
    info = model._last_recppo_info

    assert loss.requires_grad
    assert info["recppo_terminal_value_loss"] >= 0.0
    assert info["recppo_behavior_ce_loss"] > 0.0
    assert info["recppo_has_next_state_bootstrap"] is True


def test_recppo_loss_bounds_extreme_critic_values(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.ppo_epochs = 1
    cfg.ppo_adv_norm = False
    cfg.ppo_value_clip = 0.2
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    def fake_logits_value(item_state, time_step, candidates_emb):
        logits = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        value = torch.full((1, 1), 1000.0, dtype=torch.float32, requires_grad=True)
        return logits, value

    model._agent_logits_value = fake_logits_value
    trajectory = {
        "rewards": [torch.zeros((1, 1), dtype=torch.float32)],
        "log_probs": [torch.zeros(1, dtype=torch.float32)],
        "values": [torch.full((1, 1), 1000.0, dtype=torch.float32)],
        "states": [torch.zeros((1, cfg.emb_dim), dtype=torch.float32)],
        "time_steps": [torch.zeros((1, 1), dtype=torch.float32)],
        "candidates": [torch.zeros((1, 2, cfg.emb_dim), dtype=torch.float32)],
        "actions": [torch.zeros(1, dtype=torch.long)],
        "dones": [torch.ones(1, dtype=torch.bool)],
    }

    loss = model.compute_ppo_loss(trajectory)

    assert torch.isfinite(loss)
    assert float(loss.detach().item()) < 10.0


def test_recppo_bounds_extreme_policy_logits(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    raw_logits = torch.tensor([[1000.0, -1000.0]], dtype=torch.float32)

    bounded = model._bound_recppo_logits(raw_logits)
    ce = torch.nn.functional.cross_entropy(bounded, torch.tensor([1], dtype=torch.long))

    assert float(bounded.abs().max().item()) <= cfg.recppo_logit_bound
    assert float(ce.item()) < 25.0


def test_forward_uses_behavior_id_embedding_as_usim_training_target(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "0")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_AUX_WEIGHT", "0")
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_USE_PAAC", "0")
    monkeypatch.setenv("USIM_USE_PREREQ_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_SAGE_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_CGRC_RECON", "0")
    monkeypatch.setenv("USIM_USE_PSEUDO_COLD_TRAIN", "1")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MODE", "all_eligible")
    monkeypatch.setenv("USIM_PSEUDO_COLD_RATIO", "1.0")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MIN_POP", "1")
    cfg = FeedbackConfig(n_users=2, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    cfg.cold_threshold = 1
    model = FastFeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model.train()

    z_i_base = torch.full((2, cfg.emb_dim), 0.25, dtype=torch.float32)
    id_e_true = torch.full((2, cfg.emb_dim), 2.0, dtype=torch.float32)
    content_e = torch.full((2, cfg.emb_dim), -1.0, dtype=torch.float32)
    captured = {}

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        return z_i_base.clone(), id_e_true.clone(), content_e.clone()

    def fake_run_usim_episode(init_item_emb, target_emb=None, **kwargs):
        captured["target_emb"] = target_emb
        return init_item_emb, {"rewards": []}, {"steps": 0}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_run_usim_episode
    model.compute_ppo_loss = lambda trajectory: torch.tensor(0.0, dtype=torch.float32)

    batch = {
        "u": torch.tensor([0, 1], dtype=torch.long),
        "i": torch.tensor([0, 1], dtype=torch.long),
    }
    pop = torch.tensor([5.0, 7.0], dtype=torch.float32)
    llm_s = torch.full((2,), -1.0, dtype=torch.float32)

    model.forward(batch, pop, llm_s)

    assert torch.allclose(captured["target_emb"], id_e_true)
    assert not torch.allclose(captured["target_emb"], z_i_base)
    assert captured["target_emb"].requires_grad is False


def test_recppo_optimizer_performs_real_multi_epoch_updates(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=8, n_items=3, content_dim=5)
    cfg.n_candidates = 3
    cfg.retrieve_top_m = 6
    cfg.ppo_epochs = 3
    cfg.recppo_target_kl = 0.0
    cfg.recppo_actor_lr = 1e-2
    cfg.recppo_critic_lr = 1e-2
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._reset_recppo_optimizer()
    model.train()
    model._recppo_behavior_user_idx = torch.tensor([1, 2], dtype=torch.long)

    init = torch.randn((2, cfg.emb_dim), dtype=torch.float32)
    target = torch.randn((2, cfg.emb_dim), dtype=torch.float32)
    _, trajectory, _ = model.run_usim_episode(
        init,
        target_emb=target,
        item_idx=torch.tensor([0, 1], dtype=torch.long),
    )
    before = model.agent.actor_head.weight.detach().clone()

    info = model.optimize_recppo(trajectory)

    assert info["recppo_update_epochs"] == 3
    assert info["recppo_max_ratio_deviation"] > 0.0
    assert not torch.allclose(before, model.agent.actor_head.weight.detach())
    assert all(param.grad is None for param in model._recppo_parameters())


def test_deterministic_refined_inference_is_repeatable(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=32, n_items=4, content_dim=5)
    cfg.n_candidates = 6
    cfg.retrieve_top_m = 16
    model = RepairedFast3FeedbackUSIM(cfg, torch.randn((4, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model.eval()
    item_idx = torch.tensor([1, 2], dtype=torch.long)

    first = model.infer_refined_item_vectors(item_idx, force_cold=True)
    second = model.infer_refined_item_vectors(item_idx, force_cold=True)

    assert torch.equal(first, second)


def test_repaired_eval_positive_vectors_reuse_cached_item_bank():
    model = _FakeEvalModel()
    banks = repaired_mod.repaired_build_eval_item_vecs(
        model,
        torch.device("cpu"),
        llm_scores=None,
        item_batch=2,
    )
    model.refined[1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    pos_vec = repaired_mod.repaired_build_eval_pos_item_vecs(
        model,
        torch.tensor([1], dtype=torch.long),
        torch.full((1,), -1.0, dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        eval_type="cold",
    )

    assert torch.equal(pos_vec[0], banks["cold"][1])


def test_stop_action_ends_episode_without_state_transition(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "3")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.n_candidates = 1
    cfg.usim_lr = 1.0
    cfg.recppo_min_steps = 0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    def fake_get_candidates(*args, **kwargs):
        users = torch.ones((1, 1, cfg.emb_dim), dtype=torch.float32)
        return users, torch.tensor([[0]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def select_stop(current_h, time_step, candidates, fit_score=None, deterministic=False):
        action = torch.tensor([candidates.size(1) - 1], dtype=torch.long)
        return action, torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = select_stop
    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)

    final_h, trajectory, stats = model.run_usim_episode(
        init,
        target_emb=torch.ones_like(init),
        item_idx=torch.tensor([0], dtype=torch.long),
    )

    assert torch.equal(final_h, init)
    assert len(trajectory["rewards"]) == 1
    assert trajectory["dones"][0].item() is True
    assert trajectory["valids"][0].item() is True
    assert stats["stop_rate"] == pytest.approx(1.0)


def test_stop_action_is_masked_before_minimum_steps(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "3")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.recppo_min_steps = 2
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    state = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    candidates = torch.randn((1, 3, cfg.emb_dim), dtype=torch.float32)
    candidates[:, -1, :] = 0.0

    early_logits, _ = model._agent_logits_value(state, torch.tensor([[0.0]]), candidates)
    available_logits, _ = model._agent_logits_value(state, torch.tensor([[2.0]]), candidates)

    assert float(early_logits[0, -1].item()) < -1e8
    assert torch.isfinite(available_logits[0, -1])


def test_behavior_supervision_only_labels_first_step(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=4, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.recppo_enable_stop = False
    cfg.feedback_course_sample_beta = 0.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._recppo_behavior_user_idx = torch.tensor([1], dtype=torch.long)

    def fake_get_candidates(*args, **kwargs):
        users = torch.zeros((1, 2, cfg.emb_dim), dtype=torch.float32)
        return users, torch.tensor([[0, 2]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return torch.tensor([0]), torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select
    model.run_usim_episode(
        torch.zeros((1, cfg.emb_dim), dtype=torch.float32),
        target_emb=torch.ones((1, cfg.emb_dim), dtype=torch.float32),
        item_idx=torch.tensor([0], dtype=torch.long),
    )
    labels = model._last_recppo_trajectory["behavior_actions"]
    candidates = model._last_recppo_trajectory["candidates"]

    assert labels[0].item() >= 0
    assert labels[1].item() == -100
    assert torch.count_nonzero(candidates[1]).item() == 0


def test_recppo_masks_previously_selected_users(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=4, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.recppo_enable_stop = False
    cfg.feedback_course_sample_beta = 0.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    def fake_get_candidates(*args, **kwargs):
        users = torch.randn((1, 2, cfg.emb_dim), dtype=torch.float32)
        return users, torch.tensor([[1, 2]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def choose_first_available(current_h, time_step, candidates, fit_score=None, deterministic=False):
        mask = model._recppo_rollout_action_mask
        action = mask.long().argmax(dim=1)
        return action, torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = choose_first_available
    _, trajectory, _ = model.run_usim_episode(
        torch.zeros((1, cfg.emb_dim), dtype=torch.float32),
        target_emb=torch.ones((1, cfg.emb_dim), dtype=torch.float32),
        item_idx=torch.tensor([0], dtype=torch.long),
    )

    assert trajectory["actions"][0].item() == 0
    assert trajectory["actions"][1].item() == 1
    assert trajectory["action_masks"][1].tolist() == [[False, True]]


def test_batch_duplicate_stat_is_not_part_of_action_reward(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.usim_lr = 0.0
    cfg.reward_step_cost = 0.25
    cfg.reward_dup_penalty_weight = 100.0
    cfg.recppo_enable_stop = False
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")

    def fake_get_candidates(*args, **kwargs):
        users = torch.zeros((1, 1, cfg.emb_dim), dtype=torch.float32)
        return users, torch.tensor([[0]], dtype=torch.long), {"dup_rate": 1.0, "topm_coverage": 0.0}

    model.get_candidates = fake_get_candidates
    init = torch.ones((1, cfg.emb_dim), dtype=torch.float32)
    _, trajectory, _ = model.run_usim_episode(
        init,
        target_emb=init.clone(),
        item_idx=torch.tensor([0], dtype=torch.long),
    )

    assert torch.allclose(trajectory["rewards"][0], torch.tensor([[-0.25]], dtype=torch.float32))


def test_repaired_defaults_train_content_path_on_all_warm_items(monkeypatch):
    for name in (
        "USIM_USE_PSEUDO_COLD_TRAIN",
        "USIM_PSEUDO_COLD_MODE",
        "USIM_PSEUDO_COLD_RATIO",
        "USIM_PSEUDO_COLD_MIN_POP",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)

    assert cfg.use_pseudo_cold_train is True
    assert cfg.pseudo_cold_mode == "all_eligible"
    assert cfg.pseudo_cold_ratio == pytest.approx(1.0)
    assert cfg.pseudo_cold_min_pop == 1


def test_recppo_default_schedule_reserves_half_of_training_for_warmup(monkeypatch):
    monkeypatch.setenv("USIM_N_EPOCHS", "30")
    monkeypatch.delenv("USIM_RECPPO_WARMUP_EPOCHS", raising=False)
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")

    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)

    assert cfg.recppo_warmup_epochs == 15
    assert cfg.n_epochs - cfg.recppo_warmup_epochs == 15


def test_recppo_policy_weight_scales_the_rl_objective(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=8, n_items=4, content_dim=5)
    cfg.n_candidates = 4
    cfg.retrieve_top_m = 8
    cfg.recppo_enable_stop = False
    cfg.recppo_behavior_ce_weight = 0.0
    cfg.recppo_terminal_value_weight = 0.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.randn((4, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    init = torch.randn((3, cfg.emb_dim), dtype=torch.float32)
    target = torch.randn_like(init)
    _, trajectory, _ = model.run_usim_episode(
        init,
        target_emb=target,
        item_idx=torch.tensor([0, 1, 2], dtype=torch.long),
    )
    prepared = model._prepare_recppo_targets(trajectory)

    cfg.ppo_loss_weight = 1.0
    full_loss, full_info = model._recppo_objective(trajectory, prepared)
    cfg.ppo_loss_weight = 0.25
    quarter_loss, quarter_info = model._recppo_objective(trajectory, prepared)

    assert full_info["recppo_policy_loss"] != pytest.approx(0.0)
    assert quarter_loss.item() == pytest.approx(0.25 * full_loss.item(), rel=1e-5, abs=1e-6)
    assert quarter_info["recppo_policy_loss"] == pytest.approx(full_info["recppo_policy_loss"])


def test_recppo_policy_weight_scales_independent_optimizer_learning_rates(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=4, n_items=3, content_dim=5)
    cfg.recppo_actor_lr = 5e-4
    cfg.recppo_critic_lr = 1e-3
    cfg.ppo_loss_weight = 0.25
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))

    optimizer = model._reset_recppo_optimizer()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.25e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-4)


def test_recppo_candidate_logits_use_scaled_dot_product(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=4, n_items=3, content_dim=5)
    cfg.recppo_enable_stop = False
    cfg.recppo_logit_bound = 0.0
    cfg.recppo_policy_temperature = 2.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    state = torch.randn((2, cfg.emb_dim), dtype=torch.float32)
    time_step = torch.zeros((2, 1), dtype=torch.long)
    candidates = torch.randn((2, 3, cfg.emb_dim), dtype=torch.float32)

    logits, _ = model._agent_logits_value(state, time_step, candidates)
    t_emb = F.one_hot(time_step.squeeze(1), num_classes=model.agent.time_dim).float()
    feat = model.agent.common(torch.cat([state, t_emb], dim=1))
    query = model.agent.actor_head(feat).unsqueeze(1)
    keys = model.agent.user_proj(candidates)
    expected = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
    expected = expected / (keys.size(-1) ** 0.5 * cfg.recppo_policy_temperature)

    assert torch.allclose(logits, expected)


def test_stage_guarded_selection_cannot_finish_on_warmup_checkpoint(monkeypatch):
    monkeypatch.setenv("USIM_N_EPOCHS", "4")
    monkeypatch.setenv("USIM_RECPPO_WARMUP_EPOCHS", "2")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
    model._recppo_epoch_state.fill_(1)
    cold = {"R@10": 0.25, "N@10": 0.18}
    hot = {"R@10": 0.08, "N@10": 0.04}

    warm_score = repaired_mod.repaired_compute_early_stop_score(
        cold, hot, 10, mode="recppo_stage_guarded"
    )
    model._recppo_phase_state.fill_(1)
    ppo_score = repaired_mod.repaired_compute_early_stop_score(
        cold, hot, 10, mode="recppo_stage_guarded"
    )

    assert warm_score < 0.0
    assert ppo_score > 0.0
    assert model._recppo_warm_hot_r.item() == pytest.approx(0.08)
    assert model._recppo_warm_hot_n.item() == pytest.approx(0.04)


def test_recppo_residual_is_ramped_after_phase_transition(monkeypatch):
    monkeypatch.setenv("USIM_N_EPOCHS", "10")
    monkeypatch.setenv("USIM_RECPPO_WARMUP_EPOCHS", "4")
    monkeypatch.setenv("USIM_RECPPO_RESIDUAL_RAMP_EPOCHS", "5")
    monkeypatch.setenv("USIM_RL_RESIDUAL_SCALE", "0.30")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
    model._recppo_phase_state.fill_(1)

    model._recppo_epoch_state.fill_(4)
    assert model._effective_recppo_residual_scale() == pytest.approx(0.06)
    model._recppo_epoch_state.fill_(8)
    assert model._effective_recppo_residual_scale() == pytest.approx(0.30)


def test_recppo_phase_freezes_warm_backbone_and_trains_only_policy_heads(monkeypatch):
    monkeypatch.setenv("USIM_N_EPOCHS", "2")
    monkeypatch.setenv("USIM_RECPPO_WARMUP_EPOCHS", "1")
    monkeypatch.setenv("USIM_STEPS", "0")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_AUX_WEIGHT", "0")
    monkeypatch.setenv("USIM_USE_PAAC", "0")
    monkeypatch.setenv("USIM_USE_PREREQ_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_SAGE_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_CGRC_RECON", "0")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    batch = {
        "u": torch.tensor([0, 1], dtype=torch.long),
        "i": torch.tensor([0, 1], dtype=torch.long),
    }
    pop = torch.tensor([4.0, 5.0], dtype=torch.float32)
    llm_s = torch.full((2,), -1.0, dtype=torch.float32)

    model.train()
    _, warm_stats = model(batch, pop, llm_s)
    model.eval()
    model.train()
    ppo_loss, ppo_stats = model(batch, pop, llm_s)

    assert warm_stats["recppo_phase"] == "warmup"
    assert ppo_stats["recppo_phase"] == "ppo"
    assert ppo_loss.requires_grad
    ppo_loss.backward()
    assert model.user_emb.weight.requires_grad is False
    assert model.item_id_emb.weight.requires_grad is False
    assert model.user_emb.weight.grad is None
    assert all(param.requires_grad for param in model.agent.parameters())


def test_recppo_phase_reports_policy_loss_without_outer_backbone_gradients(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model._activate_recppo_phase()
    model.train()

    def fake_legacy_forward(self, batch, pop, llm_s, **kwargs):
        self._last_recppo_trajectory = {"rewards": [torch.tensor([0.0])]}
        return self.user_emb.weight[0].sum(), {"main_loss": 1.25}

    def fake_optimize(_trajectory):
        model._last_recppo_info = {"recppo_total_loss": 0.375}
        return model._last_recppo_info

    monkeypatch.setattr(repaired_mod.legacy.Fast3FeedbackUSIM, "forward", fake_legacy_forward)
    monkeypatch.setattr(model, "optimize_recppo", fake_optimize)

    loss, stats = model(
        {"u": torch.tensor([0]), "i": torch.tensor([0])},
        torch.tensor([1.0]),
        torch.tensor([-1.0]),
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.375)
    assert stats["ppo_loss"] == pytest.approx(0.375)
    assert model.user_emb.weight.grad is None
    assert model.recppo_outer_anchor.grad is not None
    assert model.recppo_outer_anchor.grad.item() == pytest.approx(0.0)


def test_outer_optimizer_excludes_recppo_agent_parameters(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))

    optimizer = repaired_mod.repaired_make_fast3_optimizer(model, cfg)
    outer_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert outer_ids.isdisjoint(model.recppo_parameter_ids())
    assert id(model.user_emb.weight) in outer_ids


def test_guarded_early_stop_penalizes_hot_collapse():
    cold = {"R@10": 0.25, "N@10": 0.18}
    collapsed_hot = {"R@10": 0.0, "N@10": 0.0}
    viable_hot = {"R@10": 0.08, "N@10": 0.04}

    collapsed = repaired_mod.repaired_compute_early_stop_score(
        cold,
        collapsed_hot,
        10,
        mode="recppo_guarded",
    )
    viable = repaired_mod.repaired_compute_early_stop_score(
        cold,
        viable_hot,
        10,
        mode="recppo_guarded",
    )

    assert viable > collapsed
    assert collapsed == 0.0


def test_reward_uses_the_same_residual_blend_as_inference(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.recppo_enable_stop = False
    cfg.usim_lr = 1.0
    cfg.rl_residual_scale = 0.5
    cfg.reward_gain_weight = 1.0
    cfg.recppo_rank_gain_weight = 0.0
    cfg.reward_gain_clip = 1.0
    cfg.reward_step_cost = 0.0
    cfg.use_course_reward = False
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    selected = torch.zeros((1, 1, cfg.emb_dim), dtype=torch.float32)
    selected[0, 0, 1] = 1.0

    def fake_get_candidates(*args, **kwargs):
        return selected, torch.tensor([[0]], dtype=torch.long), {"dup_rate": 0.0, "topm_coverage": 1.0}

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return torch.tensor([0]), torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select
    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    init[0, 0] = 1.0
    target = torch.zeros_like(init)
    target[0, 1] = 1.0

    _, trajectory, _ = model.run_usim_episode(init, target_emb=target, item_idx=torch.tensor([0]))

    expected_gain = 0.5 / (1.0 + 0.5 ** 2) ** 0.5
    assert trajectory["rewards"][0].item() == pytest.approx(expected_gain, abs=1e-5)


def _make_global_rank_reward_model(monkeypatch, n_users=4):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=n_users, n_items=3, content_dim=5)
    cfg.recppo_rank_topk = 2
    cfg.recppo_rank_temperature = 0.5
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._recppo_phase_state.fill_(1)
    return model


def test_global_rank_reward_pool_contains_only_users_with_train_history(monkeypatch):
    model = _make_global_rank_reward_model(monkeypatch)
    bank = F.normalize(torch.randn((4, model.cfg.emb_dim), generator=torch.Generator().manual_seed(7)), dim=1)
    train_seen = {0: {1}, 2: {0, 1}, 3: set()}

    pool, user_ids = model._recppo_train_user_pool(bank, train_seen)

    assert user_ids.tolist() == [0, 2]
    assert torch.allclose(pool, bank[user_ids])


def test_global_rank_reward_caches_behavior_target_topk_users(monkeypatch):
    model = _make_global_rank_reward_model(monkeypatch)
    bank = torch.zeros((4, model.cfg.emb_dim), dtype=torch.float32)
    bank[0, 0] = 1.0
    bank[1, 0] = 0.8
    bank[1, 1] = 0.6
    bank[2, 0] = 0.6
    bank[2, 1] = 0.8
    bank[3, 1] = 1.0
    bank = F.normalize(bank, dim=1)
    train_seen = {0: {1}, 1: {1}, 2: {0}, 3: {2}}
    target = torch.zeros((1, model.cfg.emb_dim), dtype=torch.float32)
    target[0, 0] = 1.0
    item_idx = torch.tensor([1], dtype=torch.long)

    first = model._recppo_target_topk_user_ids(item_idx, target, bank, train_seen)
    misses_after_first = model._recppo_rank_cache_misses
    original_pool_builder = model._recppo_train_user_pool

    def fail_if_pool_is_rescanned(*args, **kwargs):
        raise AssertionError("a pure Top-K cache hit must not rescan the full training-user pool")

    model._recppo_train_user_pool = fail_if_pool_is_rescanned
    second = model._recppo_target_topk_user_ids(item_idx, target, bank, train_seen)
    model._recppo_train_user_pool = original_pool_builder

    assert first.tolist() == second.tolist()
    assert first[0].tolist() == [0, 1]
    assert misses_after_first == 1
    assert model._recppo_rank_cache_hits >= 1


def test_global_listwise_rank_gain_is_positive_when_state_moves_toward_behavior_target(monkeypatch):
    model = _make_global_rank_reward_model(monkeypatch, n_users=3)
    bank = torch.zeros((3, model.cfg.emb_dim), dtype=torch.float32)
    bank[0, 0] = 1.0
    bank[1, 1] = 1.0
    bank[2, 2] = 1.0
    train_seen = {0: {0}, 1: {1}, 2: {2}}
    target = bank[0].view(1, -1)
    prev_h = bank[1].view(1, -1)
    next_h = bank[0].view(1, -1)

    gain = model._global_train_user_listwise_gain(
        prev_h,
        next_h,
        target,
        item_idx=torch.tensor([0], dtype=torch.long),
        user_bank_norm=bank,
        user_seen_items=train_seen,
    )

    assert gain.shape == (1, 1)
    assert gain.item() > 0.0


def test_global_rank_cache_persists_across_recppo_epochs(monkeypatch):
    monkeypatch.setenv("USIM_N_EPOCHS", "4")
    monkeypatch.setenv("USIM_RECPPO_WARMUP_EPOCHS", "1")
    model = _make_global_rank_reward_model(monkeypatch)
    model._recppo_rank_topk_cache[1] = torch.tensor([2], dtype=torch.long)
    model._recppo_epoch_pending = True
    model.train()

    model._begin_pending_epoch()

    assert model._recppo_rank_topk_cache[1].tolist() == [2]


def test_recppo_objective_uses_fixed_behavior_ce_weight(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_RECPPO_BEHAVIOR_CE_W", "0.20")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=3, content_dim=5)
    cfg.n_candidates = 2
    cfg.recppo_enable_stop = False
    cfg.ppo_loss_weight = 0.0
    cfg.recppo_terminal_value_weight = 0.0
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    model._recppo_behavior_user_idx = torch.tensor([1], dtype=torch.long)

    def fake_get_candidates(*args, **kwargs):
        candidates = torch.randn((1, 2, cfg.emb_dim), dtype=torch.float32)
        return candidates, torch.tensor([[0, 2]], dtype=torch.long), {
            "dup_rate": 0.0,
            "topm_coverage": 1.0,
        }

    model.get_candidates = fake_get_candidates
    init = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
    target = torch.ones((1, cfg.emb_dim), dtype=torch.float32)
    _, trajectory, _ = model.run_usim_episode(
        init,
        target_emb=target,
        item_idx=torch.tensor([0]),
    )
    prepared = model._prepare_recppo_targets(trajectory)
    model._recppo_phase_state.fill_(1)

    model._recppo_epoch_state.fill_(5)
    early_loss, early_info = model._recppo_objective(trajectory, prepared)
    model._recppo_epoch_state.fill_(15)
    late_loss, late_info = model._recppo_objective(trajectory, prepared)

    expected = 0.20 * early_info["recppo_behavior_ce_loss"]
    assert early_loss.item() == pytest.approx(expected)
    assert late_loss.item() == pytest.approx(expected)
    assert late_info["recppo_behavior_ce_loss"] == pytest.approx(
        early_info["recppo_behavior_ce_loss"]
    )


def test_terminal_value_anchor_repeats_behavior_target_at_each_global_step(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
    captured = {}

    def fake_value(states, times):
        captured["states"] = states.detach().clone()
        captured["times"] = times.detach().clone()
        return torch.zeros((states.size(0), 1), dtype=states.dtype, requires_grad=True)

    model._agent_value = fake_value
    target = torch.randn((2, cfg.emb_dim), generator=torch.Generator().manual_seed(9))
    trajectory = {
        "target_emb": target,
        "time_steps": [
            torch.tensor([[0], [0]]),
            torch.tensor([[1], [1]]),
            torch.tensor([[2], [2]]),
        ],
        "dones": [
            torch.tensor([True, False]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
        ],
        "valids": [
            torch.tensor([True, True]),
            torch.tensor([False, True]),
            torch.tensor([False, True]),
        ],
    }

    model._terminal_value_loss(trajectory)

    assert torch.equal(captured["states"], torch.cat([target, target, target], dim=0))
    assert captured["times"].view(-1).tolist() == [0, 0, 1, 1, 2, 2]


def test_global_rank_gain_is_normalized_by_effective_transition_scale(monkeypatch):
    model = _make_global_rank_reward_model(monkeypatch, n_users=2)
    model.cfg.usim_lr = 0.3
    model.cfg.rl_residual_scale = 0.3
    model.cfg.recppo_residual_ramp_epochs = 5
    model.cfg.recppo_warmup_epochs = 0
    model._recppo_epoch_state.fill_(0)

    normalized = model._normalize_recppo_rank_gain(torch.tensor([[0.018]], dtype=torch.float32))

    assert model._recppo_rank_transition_scale() == pytest.approx(0.018)
    assert normalized.item() == pytest.approx(1.0)


def test_active_global_rank_reward_fails_closed_without_train_histories(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    model = _make_global_rank_reward_model(monkeypatch, n_users=2)
    model.cfg.recppo_enable_stop = False
    model.cfg.use_course_reward = False
    candidate = torch.zeros((1, 1, model.cfg.emb_dim), dtype=torch.float32)
    candidate[0, 0, 0] = 1.0

    def fake_get_candidates(*args, **kwargs):
        return candidate, torch.tensor([[0]], dtype=torch.long), {
            "dup_rate": 0.0,
            "topm_coverage": 1.0,
        }

    def fake_select(current_h, time_step, candidates, fit_score=None, deterministic=False):
        return torch.tensor([0]), torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)

    model.get_candidates = fake_get_candidates
    model._select_rollout_action = fake_select
    target = torch.zeros((1, model.cfg.emb_dim), dtype=torch.float32)
    target[0, 0] = 1.0

    with pytest.raises(RuntimeError, match="train-only user histories"):
        model.run_usim_episode(
            target.clone(),
            target_emb=target,
            item_idx=torch.tensor([0], dtype=torch.long),
            user_seen_items=None,
        )


def test_recppo_residual_blend_caps_vector_drift(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.rl_residual_scale = 1.0
    cfg.recppo_max_residual_norm = 0.5
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    base = torch.zeros((2, cfg.emb_dim), dtype=torch.float32)
    final = torch.full_like(base, 10.0)

    blended = model._blend_rl_episode_output(base, final)

    assert torch.all(blended.norm(dim=1) <= 0.50001)


def test_recppo_epoch_diagnostics_are_written(tmp_path, monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_FB_OUTPUT_DIR", str(tmp_path))
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    model = RepairedFast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
    model._recppo_epoch_state.fill_(2)
    model._accumulate_recppo_diagnostics(
        {
            "recppo_actor_loss": 0.2,
            "recppo_critic_loss": 0.3,
            "recppo_approx_kl": 0.01,
            "recppo_clip_fraction": 0.1,
            "recppo_entropy": 1.5,
            "recppo_reward_mean": 0.04,
            "recppo_stop_rate": 0.25,
            "recppo_rank_gain_std": 0.03,
        }
    )

    model._flush_recppo_diagnostics()

    text = (tmp_path / "recppo_epoch_metrics.csv").read_text(encoding="utf-8")
    assert "recppo_approx_kl" in text
    assert "recppo_stop_rate" in text
    assert "recppo_rank_gain_std" in text
    assert "0.25" in text


def test_repaired_manifest_records_entrypoint_and_recppo_config(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"

    def fake_legacy_writer(*args, **kwargs):
        manifest_path.write_text('{"script": {}, "model_config": {}}', encoding="utf-8")
        return str(manifest_path)

    monkeypatch.setattr(repaired_mod, "_legacy_write_static_manifest", fake_legacy_writer)
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)

    repaired_mod.repaired_write_static_manifest({}, {}, cfg, {}, "unused", object())
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["script"]["path"].endswith("usim_feedback_fast3_content_delta_repaired.py")
    assert payload["model_config"]["recppo"]["deterministic_eval_candidates"] is True
    assert payload["model_config"]["recppo"]["behavior_supervision"] == "first_step_only"
    assert (
        payload["model_config"]["recppo"]["rank_reward_source"]
        == "global_train_user_topk"
    )
    assert payload["model_config"]["recppo"]["joint_supervised_backbone"] is False
    assert payload["model_config"]["recppo"]["rank_topk"] == cfg.recppo_rank_topk
    assert payload["model_config"]["recppo"]["rank_temperature"] == pytest.approx(
        cfg.recppo_rank_temperature
    )


def test_repaired_checkpoint_fingerprint_covers_recppo_reward_controls(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)

    first, payload = repaired_mod.repaired_static_train_config_fingerprint(
        cfg,
        split_info={"split_mode": "strict_item_cold_balanced"},
        script_path=repaired_mod.__file__,
    )
    cfg.recppo_behavior_ce_weight += 0.01
    second, changed_payload = repaired_mod.repaired_static_train_config_fingerprint(
        cfg,
        split_info={"split_mode": "strict_item_cold_balanced"},
        script_path=repaired_mod.__file__,
    )

    assert payload["recppo_rank_reward_source"] == "global_train_user_topk"
    assert payload["recppo_joint_supervised_backbone"] is False
    expected_controls = {
        "recppo_fingerprint_schema",
        "reward_step_cost",
        "recppo_policy_temperature",
        "recppo_enable_stop",
        "recppo_min_steps",
        "recppo_stop_bias_init",
        "recppo_rank_normalize_transition",
        "recppo_bootstrap_next_value",
        "recppo_inject_behavior_user",
        "recppo_teacher_force_behavior",
        "recppo_value_bound",
        "recppo_logit_bound",
        "recppo_max_grad_norm",
        "ppo_gamma",
        "ppo_epochs",
        "ppo_lambda",
        "ppo_clip",
        "ppo_value_clip",
        "ppo_adv_norm",
        "ppo_coeffs",
        "usim_lr",
        "reward_gain_clip",
        "feedback_course_prereq_weight",
        "feedback_course_concept_weight",
        "feedback_course_difficulty_weight",
        "feedback_course_redundant_weight",
        "feedback_course_term_norm",
        "feedback_course_term_norm_clip",
        "feedback_course_term_norm_eps",
        "feedback_course_term_norm_ema_decay",
        "feedback_course_redundant_mode",
        "feedback_course_redundant_thr",
        "feedback_course_redundant_concept_gate",
        "feedback_course_prereq_gate",
        "feedback_prereq_weighted_edges",
        "feedback_prereq_soft_penalty",
        "feedback_course_sample_only_cold",
        "feedback_course_sample_topk",
        "feedback_course_sample_top_l",
        "recppo_guard_hot_ratio",
        "early_stop_min_delta",
        "rl_residual_scale",
    }
    assert expected_controls.issubset(payload)
    assert payload["rl_residual_scale"] == pytest.approx(cfg.rl_residual_scale)
    assert changed_payload["recppo_behavior_ce_weight"] != payload["recppo_behavior_ce_weight"]
    assert first != second


def test_recppo_optimizer_state_is_accepted_only_after_checkpoint_config_match(monkeypatch):
    candidate = {"state": {1: {"step": torch.tensor(2.0)}}, "param_groups": []}
    state = {"recppo_optimizer_state": candidate}
    monkeypatch.setattr(repaired_mod, "_legacy_load_feedback_checkpoint", lambda *args, **kwargs: state)

    loaded = repaired_mod.repaired_load_feedback_checkpoint("unused")
    assert loaded is state
    assert repaired_mod._pending_recppo_optimizer_state is None

    monkeypatch.setattr(
        repaired_mod,
        "_legacy_checkpoint_config_matches",
        lambda *args, **kwargs: (False, "mismatch", "new", "old"),
    )
    repaired_mod.repaired_checkpoint_config_matches(state, object())
    assert repaired_mod._pending_recppo_optimizer_state is None

    repaired_mod.repaired_load_feedback_checkpoint("unused")
    monkeypatch.setattr(
        repaired_mod,
        "_legacy_checkpoint_config_matches",
        lambda *args, **kwargs: (True, "match", "same", "same"),
    )
    repaired_mod.repaired_checkpoint_config_matches(state, object())
    assert repaired_mod._pending_recppo_optimizer_state is candidate


def test_finished_checkpoint_pairs_best_model_with_best_recppo_optimizer(monkeypatch):
    state_payload = {
        "status": "running",
        "next_epoch": 3,
        "es_best": {"epoch": 3},
        "es_best_state": {"weight": torch.tensor([3.0])},
    }

    class FakeOptimizer:
        marker = 1

        def state_dict(self):
            return {"marker": self.marker}

    model = SimpleNamespace(
        _recppo_optimizer=FakeOptimizer(),
        _recppo_best_optimizer_state=None,
    )
    monkeypatch.setattr(
        repaired_mod,
        "_legacy_build_feedback_ckpt_state",
        lambda *args, **kwargs: dict(state_payload),
    )

    running = repaired_mod.repaired_build_feedback_ckpt_state(model)
    assert running["recppo_optimizer_state"]["marker"] == 1
    assert running["recppo_best_optimizer_state"]["marker"] == 1

    model._recppo_optimizer.marker = 2
    state_payload.update({"status": "finished", "next_epoch": 5})
    finished = repaired_mod.repaired_build_feedback_ckpt_state(model)

    assert finished["recppo_optimizer_state"]["marker"] == 1
    assert finished["recppo_best_optimizer_state"]["marker"] == 1


def test_warmup_stage_payload_keeps_outer_state_and_drops_recppo_state():
    state = {
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {"state": {1: {"step": torch.tensor(3.0)}}},
        "recppo_optimizer_state": {"state": {2: {"step": torch.tensor(4.0)}}},
        "recppo_best_optimizer_state": {"state": {}},
        "next_epoch": 30,
        "train_config_payload": {"split_mode": "strict_item_cold_balanced"},
    }

    stage = repaired_mod.repaired_build_warmup_stage_state(state)

    assert stage["checkpoint_kind"] == "warmup_stage"
    assert stage["optimizer_state"] is state["optimizer_state"]
    assert stage["recppo_optimizer_state"] is None
    assert stage["recppo_best_optimizer_state"] is None
    assert stage["warmup_stage_epoch"] == 30


def test_warmup_stage_fingerprint_allows_ppo_branch_controls(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    _, base = repaired_mod.repaired_warmup_stage_fingerprint(
        cfg, split_info={"split_mode": "strict_item_cold_balanced"}
    )
    cfg.rl_residual_scale += 0.1
    cfg.ppo_loss_weight += 0.2
    cfg.n_epochs += 5
    cfg.early_stop_patience += 2
    _, branch = repaired_mod.repaired_warmup_stage_fingerprint(
        cfg, split_info={"split_mode": "strict_item_cold_balanced"}
    )

    assert branch == base


def test_warmup_stage_fingerprint_rejects_pseudo_cold_change(monkeypatch):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = RepairedFast3Config(n_users=3, n_items=4, content_dim=5)
    first, _ = repaired_mod.repaired_warmup_stage_fingerprint(
        cfg, split_info={"split_mode": "strict_item_cold_balanced"}
    )
    cfg.pseudo_cold_ratio = 0.5
    second, _ = repaired_mod.repaired_warmup_stage_fingerprint(
        cfg, split_info={"split_mode": "strict_item_cold_balanced"}
    )

    assert second != first


def test_repaired_runner_exposes_warmup_stage_checkpoint_argument():
    runner_text = Path("run_usim_feedback_fast3_content_delta_repaired_static.ps1").read_text(
        encoding="utf-8"
    )

    assert "WarmupStageCheckpoint" in runner_text
    assert "USIM_FB_WARMUP_STAGE_CKPT" in runner_text


def test_warmup_stage_is_only_used_when_branch_has_no_latest(monkeypatch, tmp_path):
    branch = tmp_path / "branch"
    branch.mkdir()
    stage = tmp_path / "warmup_stage.pt"
    torch.save({"checkpoint_kind": "warmup_stage", "next_epoch": 30}, stage)
    monkeypatch.setenv("USIM_FB_WARMUP_STAGE_CKPT", str(stage))
    monkeypatch.setattr(repaired_mod, "_legacy_load_feedback_checkpoint", lambda *_: None)

    loaded = repaired_mod.repaired_load_feedback_checkpoint(str(branch))
    assert loaded["checkpoint_kind"] == "warmup_stage"

    latest_state = {"checkpoint_kind": "branch_latest", "next_epoch": 32}
    monkeypatch.setattr(
        repaired_mod, "_legacy_load_feedback_checkpoint", lambda *_: latest_state
    )
    loaded = repaired_mod.repaired_load_feedback_checkpoint(str(branch))
    assert loaded is latest_state
