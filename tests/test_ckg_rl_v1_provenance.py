from copy import deepcopy
from types import SimpleNamespace

import pytest

from fast3_delta.checkpoint import (
    CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
    V1_CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
    _static_train_config_fingerprint,
    checkpoint_fingerprint_schema_version,
    checkpoint_resume_decision,
)
from fast3_delta.config import Fast3Config
from fast3_delta.provenance import build_split_fingerprint


def _v1_cfg(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "1")
    monkeypatch.setenv("USIM_BATCH_SIZE", "512")
    monkeypatch.setenv("USIM_V1_PSEUDO_COLD_PLAN_HASH", "plan-sha256")
    monkeypatch.setenv("USIM_V1_PSEUDO_COLD_PLAN_COUNT", "37")
    monkeypatch.setenv("USIM_V1_PSEUDO_COLD_PLAN_SEED", "2025")
    monkeypatch.setenv("USIM_V1_SELECTOR_HOT_TOL", "0.003")
    monkeypatch.setenv("USIM_V1_SELECTOR_OVERALL_TOL", "0.004")
    monkeypatch.setenv("USIM_V1_REFERENCE_BATCH_SIZE", "384")
    monkeypatch.setenv("USIM_FB_COURSE_SAMPLE_SOFT", "1")
    monkeypatch.setenv("USIM_FB_COURSE_SAMPLE_BETA", "0.31")
    monkeypatch.setenv("USIM_FB_COURSE_SAMPLE_ONLY_COLD", "0")
    monkeypatch.setenv("USIM_FB_COURSE_SAMPLE_TOPK", "19")
    monkeypatch.setenv("USIM_FB_COURSE_SAMPLE_TOPL", "17")
    monkeypatch.setenv("USIM_CANDIDATE_STRATEGY", "retrieve_sample")
    monkeypatch.setenv("USIM_N_CANDIDATES", "13")
    monkeypatch.setenv("USIM_RETRIEVE_TOP_M", "41")
    monkeypatch.setenv("USIM_CANDIDATE_TEMP", "0.17")
    monkeypatch.setenv("USIM_CANDIDATE_EPSILON", "0.09")
    monkeypatch.setenv("USIM_PPO_EPOCHS", "1")
    monkeypatch.setenv("USIM_PPO_CLIP", "0.11")
    monkeypatch.setenv("USIM_PPO_GAMMA", "0.81")
    monkeypatch.setenv("USIM_PPO_LAMBDA", "0.88")
    monkeypatch.setenv("USIM_PPO_VALUE_CLIP", "0.16")
    monkeypatch.setenv("USIM_PPO_ADV_NORM", "0")
    monkeypatch.setenv("USIM_FB_REWARD_DUP_W", "0.0")
    return Fast3Config(n_users=7, n_items=11, content_dim=5)


def _split():
    return {
        "split_mode": "strict_item_cold_balanced",
        "seed": 2025,
        "train_rows": 100,
        "val_rows": 20,
        "test_rows": 30,
    }


def _checkpoint_state(cfg, split_info):
    train_fp, train_payload = _static_train_config_fingerprint(cfg, split_info=split_info)
    split_fp, split_payload = build_split_fingerprint(split_info)
    return {
        "fingerprint_schema_version": checkpoint_fingerprint_schema_version(cfg),
        "train_config_fingerprint": train_fp,
        "train_config_payload": train_payload,
        "split_fingerprint": split_fp,
        "split_payload": split_payload,
        "source_manifest": {"files": {"runner.py": {"sha256": "same"}}},
    }


def test_v1_reproducibility_contract_is_opt_in(monkeypatch):
    monkeypatch.delenv("USIM_CKG_RL_V1", raising=False)
    legacy_cfg = Fast3Config(n_users=7, n_items=11, content_dim=5)

    assert legacy_cfg.ckg_rl_v1_enabled is False
    assert legacy_cfg.feedback_course_match_exclude_target is False
    _, legacy_payload = _static_train_config_fingerprint(legacy_cfg, split_info=_split())
    assert legacy_payload["schema_version"] == CHECKPOINT_FINGERPRINT_SCHEMA_VERSION == 2
    assert "v1_enabled" not in legacy_payload

    cfg = _v1_cfg(monkeypatch)

    assert CHECKPOINT_FINGERPRINT_SCHEMA_VERSION == 2
    assert checkpoint_fingerprint_schema_version(cfg) == V1_CHECKPOINT_FINGERPRINT_SCHEMA_VERSION == 3
    assert cfg.ckg_rl_v1_enabled is True
    assert cfg.v1_reference_batch_size == 384
    assert cfg.v1_target_history_exclusion is True
    assert cfg.feedback_course_match_exclude_target is True
    assert cfg.v1_pseudo_cold_plan_hash == "plan-sha256"
    assert cfg.v1_pseudo_cold_plan_count == 37
    assert cfg.v1_pseudo_cold_plan_seed == 2025
    assert cfg.v1_selector_hot_tolerance == 0.003
    assert cfg.v1_selector_overall_tolerance == 0.004
    assert cfg.candidate_strategy == "retrieve_sample"
    assert cfg.candidate_temp == 0.17
    assert cfg.candidate_epsilon == 0.09
    assert cfg.ppo_epochs == 1
    assert cfg.ppo_clip == 0.11
    assert cfg.ppo_gamma == 0.81
    assert cfg.ppo_lambda == 0.88
    assert cfg.ppo_value_clip == 0.16
    assert cfg.ppo_adv_norm is False
    assert cfg.reward_dup_penalty_weight == 0.0


def test_legacy_config_ignores_v1_only_policy_and_candidate_overrides(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "0")
    monkeypatch.setenv("USIM_PPO_CLIP", "0.11")
    monkeypatch.setenv("USIM_PPO_GAMMA", "0.81")
    monkeypatch.setenv("USIM_PPO_VALUE_COEFF", "0.17")
    monkeypatch.setenv("USIM_PPO_ENTROPY_COEFF", "0.07")
    monkeypatch.setenv("USIM_CANDIDATE_STRATEGY", "uniform")
    monkeypatch.setenv("USIM_CANDIDATE_TEMP", "0.17")
    monkeypatch.setenv("USIM_CANDIDATE_EPSILON", "0.09")

    cfg = Fast3Config(n_users=7, n_items=11, content_dim=5)

    assert cfg.ppo_clip == 0.20
    assert cfg.ppo_gamma == 0.90
    assert cfg.ppo_coeffs == {"value": 0.5, "entropy": 0.01}
    assert cfg.candidate_strategy == "retrieve_sample"
    assert cfg.candidate_temp == 0.20
    assert cfg.candidate_epsilon == 0.10


def test_legacy_config_ignores_invalid_v1_contract_environment(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "0")
    monkeypatch.setenv("USIM_V1_TARGET_HISTORY_EXCLUSION_SCOPE", "invalid")
    monkeypatch.setenv("USIM_V1_PSEUDO_COLD_PLAN_COUNT", "-1")
    monkeypatch.setenv("USIM_V1_REFERENCE_BATCH_SIZE", "0")
    monkeypatch.setenv("USIM_V1_SELECTOR_HOT_TOL", "-0.1")

    cfg = Fast3Config(n_users=7, n_items=11, content_dim=5)

    assert cfg.ckg_rl_v1_enabled is False
    assert cfg.v1_target_history_exclusion is False
    assert cfg.v1_pseudo_cold_plan_count == 0
    assert cfg.v1_reference_batch_size == cfg.batch_size


def test_v1_selector_mode_is_explicit_and_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "1")
    monkeypatch.setenv("USIM_V1_SELECTOR_MODE", "cold_ndcg_running_retention")

    cfg = Fast3Config(n_users=7, n_items=11, content_dim=5)

    assert cfg.v1_selector_mode == "cold_ndcg_running_retention"

    monkeypatch.setenv("USIM_V1_SELECTOR_MODE", "unknown_selector")
    with pytest.raises(ValueError, match="USIM_V1_SELECTOR_MODE"):
        Fast3Config(n_users=7, n_items=11, content_dim=5)


def test_v1_fingerprint_binds_resolved_controls_and_uses_real_course_sampler(monkeypatch):
    cfg = _v1_cfg(monkeypatch)

    fingerprint, payload = _static_train_config_fingerprint(cfg, split_info=_split())

    required_keys = {
        "v1_enabled",
        "v1_contract_version",
        "v1_reference_batch_size",
        "v1_target_history_exclusion",
        "v1_target_history_exclusion_scope",
        "v1_pseudo_cold_plan_hash",
        "v1_pseudo_cold_plan_count",
        "v1_pseudo_cold_plan_seed",
        "v1_selector_mode",
        "v1_selector_hot_tolerance",
        "v1_selector_overall_tolerance",
        "use_epoch_early_stop",
        "early_stop_k",
        "early_stop_patience",
        "early_stop_min_delta",
        "aux_weight",
        "mask_known_pos_neg",
        "mask_same_item_neg",
        "use_course_rerank",
        "rerank_alpha",
        "rerank_lambda",
        "rerank_min_seen",
        "rerank_top_l",
        "rerank_penalty_cap",
        "rerank_only_cold",
        "concept_overlap_mode",
        "prereq_graph_source",
        "prereq_concept_score_thr",
        "prereq_concept_min_hits",
        "prereq_concept_file",
        "fast3_target_alpha_cold",
        "fast3_target_alpha_hot",
        "fast3_target_alpha_step",
        "fast3_target_alpha_entropy",
        "fast3_target_alpha_min",
        "fast3_target_alpha_max",
        "candidate_strategy",
        "n_candidates",
        "retrieve_top_m",
        "candidate_temp",
        "candidate_epsilon",
        "retrieval_user_chunk",
        "retrieval_query_chunk",
        "user_bank_refresh_steps",
        "usim_lr",
        "feedback_course_sample_soft",
        "feedback_course_sample_beta",
        "feedback_course_sample_only_cold",
        "feedback_course_sample_topk",
        "feedback_course_sample_top_l",
        "feedback_course_match_mode",
        "feedback_course_match_topk",
        "feedback_course_prereq_gate",
        "feedback_course_prereq_weight",
        "feedback_course_concept_weight",
        "feedback_course_difficulty_weight",
        "feedback_course_redundant_weight",
        "feedback_course_redundant_concept_gate",
        "feedback_course_term_norm",
        "reward_terminal_weight",
        "reward_gain_weight",
        "reward_gain_clip",
        "reward_dup_penalty_weight",
        "reward_cov_bonus_weight",
        "ppo_clip",
        "ppo_gamma",
        "ppo_epochs",
        "ppo_lambda",
        "ppo_value_clip",
        "ppo_adv_norm",
        "ppo_coeffs",
    }
    assert required_keys <= set(payload)
    assert "use_course_sample" not in payload
    assert payload["feedback_course_sample_soft"] is True
    assert payload["feedback_course_sample_beta"] == 0.31
    assert payload["v1_pseudo_cold_plan_hash"] == "plan-sha256"
    assert payload["v1_reference_batch_size"] == 384

    for attribute, replacement in (
        ("feedback_course_sample_beta", 0.32),
        ("v1_target_history_exclusion", False),
        ("v1_pseudo_cold_plan_hash", "different-plan"),
        ("v1_selector_overall_tolerance", 0.005),
        ("v1_reference_batch_size", 385),
        ("candidate_epsilon", 0.10),
        ("retrieval_query_chunk", 257),
        ("usim_lr", 0.25),
        ("feedback_course_prereq_weight", 0.09),
        ("reward_dup_penalty_weight", 0.01),
        ("early_stop_k", 11),
        ("ppo_epochs", 2),
        ("aux_weight", 0.25),
        ("mask_known_pos_neg", True),
        ("use_course_rerank", True),
        ("prereq_graph_source", "hybrid"),
        ("fast3_target_alpha_cold", 0.36),
    ):
        changed = deepcopy(cfg)
        setattr(changed, attribute, replacement)
        changed_fingerprint, _ = _static_train_config_fingerprint(changed, split_info=_split())
        assert changed_fingerprint != fingerprint, attribute


def test_v1_checkpoint_rejects_legacy_fingerprint_contract(monkeypatch):
    cfg = _v1_cfg(monkeypatch)
    split_info = _split()
    state = _checkpoint_state(cfg, split_info)

    decision = checkpoint_resume_decision(
        state,
        cfg,
        split_info,
        current_source_manifest=state["source_manifest"],
    )
    assert decision.ok is True

    outdated_schema = dict(state)
    outdated_schema["fingerprint_schema_version"] = 2
    decision = checkpoint_resume_decision(outdated_schema, cfg, split_info)
    assert decision.ok is False
    assert "legacy checkpoint" in decision.reason

    monkeypatch.setenv("USIM_FB_ALLOW_LEGACY_CKPT", "1")
    decision = checkpoint_resume_decision(outdated_schema, cfg, split_info)
    assert decision.ok is False
    assert "legacy checkpoint" in decision.reason

    legacy = SimpleNamespace(
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
        use_prereq_aux_loss=True,
        recppo_warmup_epochs=-1,
        recppo_enabled=False,
        emb_dim=128,
        n_users=7,
        n_items=11,
    )
    legacy_fp, legacy_payload = _static_train_config_fingerprint(legacy, split_info=split_info)
    legacy_state = dict(state)
    legacy_state["train_config_fingerprint"] = legacy_fp
    legacy_state["train_config_payload"] = legacy_payload
    legacy_state["fingerprint_schema_version"] = 2

    decision = checkpoint_resume_decision(legacy_state, cfg, split_info)
    assert decision.ok is False
    assert "legacy checkpoint" in decision.reason or "fingerprint mismatch" in decision.reason


def test_v1_checkpoint_rejects_source_provenance_change(monkeypatch):
    cfg = _v1_cfg(monkeypatch)
    state = _checkpoint_state(cfg, _split())
    state["source_manifest"] = {"files": {"runner.py": {"sha256": "old"}}}

    decision = checkpoint_resume_decision(
        state,
        cfg,
        _split(),
        current_source_manifest={"files": {"runner.py": {"sha256": "new"}}},
    )

    assert decision.ok is False
    assert "source provenance mismatch" in decision.reason


def test_v1_checkpoint_rejects_missing_source_provenance(monkeypatch):
    cfg = _v1_cfg(monkeypatch)
    state = _checkpoint_state(cfg, _split())
    state.pop("source_manifest")

    decision = checkpoint_resume_decision(
        state,
        cfg,
        _split(),
        current_source_manifest={"files": {"runner.py": {"sha256": "same"}}},
    )

    assert decision.ok is False
    assert "missing source provenance" in decision.reason
