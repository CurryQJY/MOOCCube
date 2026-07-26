from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

import fast3_delta.eval as eval_mod
from fast3_delta.checkpoint import _static_train_config_fingerprint
from fast3_delta.config import Fast3Config
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM, FastFeedbackUSIM


def _candidate_model():
    """Build the smallest real FastFeedbackUSIM shell needed by get_candidates."""
    model = FastFeedbackUSIM.__new__(FastFeedbackUSIM)
    model.cfg = SimpleNamespace(
        n_candidates=6,
        candidate_strategy="retrieve_sample",
        n_users=32,
        retrieve_top_m=32,
        candidate_temp=0.20,
        candidate_epsilon=0.10,
        use_sage_lite=False,
        use_cgrc_recon=False,
        deterministic_eval_seed=1907,
    )
    model.device = torch.device("cpu")

    def fixed_top_m(query_norm, user_bank_raw, top_m, user_bank_norm=None):
        del user_bank_raw, user_bank_norm
        batch_size = query_norm.size(0)
        scores = torch.zeros((batch_size, top_m), dtype=torch.float32)
        indices = torch.arange(top_m, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        return scores, indices

    model._retrieve_topm_exact = fixed_top_m
    return model


def test_deterministic_candidate_sampling_is_item_step_stable_and_default_stays_stochastic():
    model = _candidate_model()
    item_state = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    item_ids = torch.tensor([7, 19], dtype=torch.long)
    user_bank = torch.arange(96, dtype=torch.float32).view(32, 3)

    first = model.get_candidates(
        item_state,
        user_bank_raw=user_bank,
        item_idx=item_ids,
        deterministic=True,
        rollout_step=0,
    )
    second = model.get_candidates(
        item_state,
        user_bank_raw=user_bank,
        item_idx=item_ids,
        deterministic=True,
        rollout_step=0,
    )
    permuted = model.get_candidates(
        item_state.flip(0),
        user_bank_raw=user_bank,
        item_idx=item_ids.flip(0),
        deterministic=True,
        rollout_step=0,
    )
    next_step = model.get_candidates(
        item_state,
        user_bank_raw=user_bank,
        item_idx=item_ids,
        deterministic=True,
        rollout_step=1,
    )

    assert torch.equal(first[1], second[1])
    assert torch.equal(first[1], permuted[1].flip(0))
    assert not torch.equal(first[1], next_step[1])

    torch.manual_seed(41)
    stochastic_first = model.get_candidates(item_state, user_bank_raw=user_bank, item_idx=item_ids)
    stochastic_second = model.get_candidates(item_state, user_bank_raw=user_bank, item_idx=item_ids)
    assert not torch.equal(stochastic_first[1], stochastic_second[1])


class _FullRankingBankModel:
    def __init__(self):
        self.cfg = SimpleNamespace(
            n_items=3,
            cold_threshold=1,
            eval_reuse_item_bank=True,
            use_course_rerank=False,
        )
        self.training = True
        self.captured_scores = None

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def user_emb(self, user_idx):
        vectors = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=user_idx.device)
        return vectors.index_select(0, user_idx)

    def user_proj(self, user_vectors):
        return user_vectors

    def apply_course_rerank(self, scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=None):
        del user_ids, seen_tensor_cache, cand_idx, target_pop
        self.captured_scores = scores.detach().clone()
        return scores


def test_corrected_full_ranking_reuses_catalog_bank_for_seen_positive(monkeypatch):
    model = _FullRankingBankModel()
    bank = torch.tensor(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    loader = [
        (
            {"u": torch.tensor([0], dtype=torch.long), "i": torch.tensor([1], dtype=torch.long)},
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([-1.0], dtype=torch.float32),
        )
    ]

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("corrected full ranking must reuse the catalog bank, not recompute positives")

    monkeypatch.setattr(eval_mod, "build_eval_pos_item_vecs", fail_if_called)
    metrics, count = eval_mod.evaluate_usim(
        model,
        loader,
        torch.device("cpu"),
        llm_scores=None,
        k_list=(1,),
        eval_type="cold",
        full_ranking=True,
        user_seen_items={0: {1}},
        all_item_vecs={"cold": bank},
    )

    assert count == 1
    assert metrics["R@1"] == 0.0
    assert model.captured_scores is not None
    expected_target_score = torch.dot(torch.tensor([1.0, 0.0]), bank[1])
    assert model.captured_scores[0, 1].item() == expected_target_score.item()


def _anchor_model(monkeypatch):
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_SIMULATOR_TARGET_MODE", "initial_state")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "0")
    monkeypatch.setenv("USIM_AUX_WEIGHT", "0")
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_USE_PAAC", "0")
    monkeypatch.setenv("USIM_USE_PREREQ_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_SAGE_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_CGRC_RECON", "0")
    cfg = Fast3Config(n_users=2, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    cfg.cold_threshold = 1
    model = Fast3FeedbackUSIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    return model


def test_initial_state_mode_routes_training_target_to_initial_item_state(monkeypatch):
    model = _anchor_model(monkeypatch)
    model.train()
    z_i_base = torch.full((2, model.cfg.emb_dim), 0.25, dtype=torch.float32, requires_grad=True)
    id_e_true = torch.full((2, model.cfg.emb_dim), 2.0, dtype=torch.float32, requires_grad=True)
    captured = {}

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        del i_idx, llm_s, force_cold, disable_id_dropout
        return z_i_base, id_e_true, torch.full_like(z_i_base, -1.0)

    def fake_run_usim_episode(init_item_emb, target_emb=None, **kwargs):
        del kwargs
        captured["initial"] = init_item_emb
        captured["target"] = target_emb
        return init_item_emb, {"rewards": []}, {"steps": 0}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_run_usim_episode
    model.compute_ppo_loss = lambda trajectory: torch.tensor(0.0, dtype=torch.float32)

    model.forward(
        {"u": torch.tensor([0, 1], dtype=torch.long), "i": torch.tensor([0, 1], dtype=torch.long)},
        torch.tensor([5.0, 7.0], dtype=torch.float32),
        torch.full((2,), -1.0, dtype=torch.float32),
    )

    assert torch.equal(captured["target"], z_i_base.detach())
    assert captured["target"].data_ptr() != z_i_base.data_ptr()
    assert not torch.equal(captured["target"], id_e_true)
    assert captured["target"].requires_grad is False


def test_initial_state_mode_routes_refined_inference_target_to_initial_item_state(monkeypatch):
    model = _anchor_model(monkeypatch)
    z_i_base = torch.full((2, model.cfg.emb_dim), 0.25, dtype=torch.float32, requires_grad=True)
    id_e_true = torch.full((2, model.cfg.emb_dim), 9.0, dtype=torch.float32, requires_grad=True)
    captured = {}

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        del i_idx, llm_s, force_cold, disable_id_dropout
        return z_i_base, id_e_true, torch.full_like(z_i_base, -3.0)

    def fake_run_usim_episode(init_item_emb, target_emb=None, **kwargs):
        captured["initial"] = init_item_emb
        captured["target"] = target_emb
        captured["item_idx"] = kwargs["item_idx"]
        return init_item_emb, {"rewards": []}, {"steps": 0}

    model.get_item_vector = fake_get_item_vector
    model.run_usim_episode = fake_run_usim_episode

    refined = model.infer_refined_item_vectors(torch.tensor([1, 2], dtype=torch.long), item_batch=16)

    assert torch.equal(refined, z_i_base.detach())
    assert captured["item_idx"].tolist() == [1, 2]
    assert isinstance(captured["target"], torch.Tensor)
    assert torch.equal(captured["target"], z_i_base.detach())
    assert captured["target"].data_ptr() != z_i_base.data_ptr()
    assert not torch.equal(captured["target"], id_e_true)
    assert captured["target"].requires_grad is False


def test_semantic_repair_controls_change_checkpoint_fingerprint():
    legacy_cfg = SimpleNamespace(
        cold_threshold=1,
        early_stop_score_mode="cold_only",
        early_stop_average_mode="item_macro",
        use_content_delta=False,
        content_delta_mode="embedding",
        content_delta_scale=0.25,
        rl_residual_scale=1.0,
        ppo_loss_weight=1.0,
        rollout_policy="ppo",
        usim_steps=5,
        use_pseudo_cold_train=False,
        pseudo_cold_mode="batch_random",
        pseudo_cold_ratio=0.0,
        pseudo_cold_min_pop=5,
        use_course_reward=True,
        use_course_sample=True,
        use_prereq_aux_loss=True,
        recppo_warmup_epochs=-1,
        recppo_enabled=True,
        emb_dim=8,
        n_users=4,
        n_items=6,
        deterministic_eval_candidates=False,
        eval_reuse_item_bank=False,
        simulator_target_mode="legacy_id",
        deterministic_eval_seed=2025,
    )
    repaired_cfg = deepcopy(legacy_cfg)
    repaired_cfg.deterministic_eval_candidates = True
    repaired_cfg.eval_reuse_item_bank = True
    repaired_cfg.simulator_target_mode = "initial_state"
    repaired_cfg.deterministic_eval_seed = 2026

    legacy_fingerprint, legacy_payload = _static_train_config_fingerprint(legacy_cfg)
    repaired_fingerprint, repaired_payload = _static_train_config_fingerprint(repaired_cfg)

    assert legacy_fingerprint != repaired_fingerprint
    assert legacy_payload["schema_version"] == 2
    assert "deterministic_eval_candidates" not in legacy_payload
    assert "eval_reuse_item_bank" not in legacy_payload
    assert "simulator_target_mode" not in legacy_payload
    assert "deterministic_eval_seed" not in legacy_payload
    assert repaired_payload["schema_version"] == 3
    assert repaired_payload["deterministic_eval_candidates"] is True
    assert repaired_payload["eval_reuse_item_bank"] is True
    assert repaired_payload["simulator_target_mode"] == "initial_state"
    assert repaired_payload["deterministic_eval_seed"] == 2026


def test_target_history_exclusion_can_be_enabled_without_v1(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "0")
    monkeypatch.setenv("USIM_V1_TARGET_HISTORY_EXCLUSION", "1")
    monkeypatch.setenv(
        "USIM_V1_TARGET_HISTORY_EXCLUSION_SCOPE", "all_course_terms"
    )

    cfg = Fast3Config(n_users=2, n_items=3, content_dim=5)

    assert cfg.v1_enabled is False
    assert cfg.v1_target_history_exclusion is True
    assert cfg.v1_target_history_exclusion_scope == "all_course_terms"


def test_independent_target_history_exclusion_changes_checkpoint_fingerprint():
    legacy_cfg = SimpleNamespace(
        cold_threshold=1,
        early_stop_score_mode="cold_only",
        early_stop_average_mode="item_macro",
        use_content_delta=False,
        content_delta_mode="embedding",
        content_delta_scale=0.25,
        rl_residual_scale=1.0,
        ppo_loss_weight=1.0,
        rollout_policy="ppo",
        usim_steps=5,
        use_pseudo_cold_train=False,
        pseudo_cold_mode="batch_random",
        pseudo_cold_ratio=0.0,
        pseudo_cold_min_pop=5,
        use_course_reward=True,
        use_course_sample=True,
        use_prereq_aux_loss=True,
        recppo_warmup_epochs=-1,
        recppo_enabled=True,
        emb_dim=8,
        n_users=4,
        n_items=6,
        v1_enabled=False,
        target_history_exclusion=False,
        target_history_exclusion_scope="all_course_terms",
    )
    repaired_cfg = deepcopy(legacy_cfg)
    repaired_cfg.target_history_exclusion = True

    legacy_fingerprint, legacy_payload = _static_train_config_fingerprint(legacy_cfg)
    repaired_fingerprint, repaired_payload = _static_train_config_fingerprint(repaired_cfg)

    assert legacy_fingerprint != repaired_fingerprint
    assert "target_history_exclusion" not in legacy_payload
    assert repaired_payload["target_history_exclusion"] is True
    assert repaired_payload["target_history_exclusion_scope"] == "all_course_terms"


def test_semantic_repair_launcher_locks_isolated_main_configuration():
    root = Path(__file__).resolve().parents[1]
    static_runner = (root / "run_usim_feedback_fast3_content_delta_static.ps1").read_text(
        encoding="utf-8"
    )
    launcher = (root / "run_ckg_rl_semantic_repair_seed2025.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$SimulatorTargetMode = \"legacy_id\"" in static_runner
    assert "[bool]$DeterministicEvalCandidates = $false" in static_runner
    assert "[bool]$EvalReuseItemBank = $false" in static_runner
    assert '"USIM_SIMULATOR_TARGET_MODE"' in static_runner
    assert '"USIM_DETERMINISTIC_EVAL_CANDIDATES"' in static_runner
    assert '"USIM_EVAL_REUSE_ITEM_BANK"' in static_runner
    assert 'SimulatorTargetMode = "initial_state"' in launcher
    assert "DeterministicEvalCandidates = $true" in launcher
    assert "EvalReuseItemBank = $true" in launcher
    assert "SaveCkpt = $true" in launcher
    assert "SaveOptState = $true" in launcher
    assert 'OutputRoot = "outputs\\ckg_rl_semantic_repair\\seed2025"' in launcher
    assert 'CheckpointRoot = "checkpoints\\ckg_rl_semantic_repair\\seed2025"' in launcher
    assert "[switch]$DryRun" in launcher
