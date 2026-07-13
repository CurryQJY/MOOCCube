from types import SimpleNamespace

from fast3_delta.checkpoint import (
    CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
    _static_train_config_fingerprint,
    checkpoint_resume_decision,
)
from fast3_delta.provenance import build_split_fingerprint


def _cfg(**overrides):
    values = {
        "cold_threshold": 1,
        "n_epochs": 60,
        "early_stop_patience": 60,
        "early_stop_score_mode": "cold_only",
        "early_stop_average_mode": "item_macro",
        "use_content_delta": False,
        "content_delta_mode": "embedding",
        "content_delta_scale": 0.25,
        "rl_residual_scale": 1.0,
        "ppo_loss_weight": 1.0,
        "rollout_policy": "ppo",
        "usim_steps": 5,
        "use_pseudo_cold_train": False,
        "pseudo_cold_mode": "batch_random",
        "pseudo_cold_ratio": 0.3,
        "pseudo_cold_min_pop": 5,
        "use_course_reward": True,
        "use_course_sample": True,
        "use_prereq_aux_loss": True,
        "recppo_warmup_epochs": -1,
        "recppo_enabled": False,
        "emb_dim": 64,
        "n_users": 10,
        "n_items": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _split(**overrides):
    values = {"split_mode": "strict_item_cold_balanced", "seed": 2025, "train_rows": 100, "val_rows": 20, "test_rows": 30}
    values.update(overrides)
    return values


def _state(cfg, split_info, source_manifest=None):
    train_fp, train_payload = _static_train_config_fingerprint(cfg, split_info=split_info)
    split_fp, split_payload = build_split_fingerprint(split_info)
    return {
        "fingerprint_schema_version": CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
        "train_config_fingerprint": train_fp,
        "train_config_payload": train_payload,
        "split_fingerprint": split_fp,
        "split_payload": split_payload,
        "source_manifest": source_manifest or {"files": {"train.py": {"sha256": "old"}}},
    }


def test_source_change_warns_but_does_not_reject_resume():
    cfg, split_info = _cfg(), _split()
    decision = checkpoint_resume_decision(
        _state(cfg, split_info), cfg, split_info, current_source_manifest={"files": {"train.py": {"sha256": "new"}}}
    )
    assert decision.ok is True
    assert "train.py" in decision.source_warning


def test_training_config_change_rejects_resume():
    cfg, split_info = _cfg(), _split()
    state = _state(cfg, split_info)
    cfg.ppo_loss_weight = 0.5
    decision = checkpoint_resume_decision(state, cfg, split_info)
    assert decision.ok is False
    assert "ppo_loss_weight" in decision.reason


def test_epoch_and_patience_extension_are_resume_compatible():
    cfg, split_info = _cfg(), _split()
    state = _state(cfg, split_info)
    cfg.n_epochs = 100
    cfg.early_stop_patience = 100
    assert checkpoint_resume_decision(state, cfg, split_info).ok is True


def test_split_change_rejects_resume():
    cfg, split_info = _cfg(), _split()
    state = _state(cfg, split_info)
    changed = dict(split_info, test_rows=31)
    decision = checkpoint_resume_decision(state, cfg, changed)
    assert decision.ok is False
    assert "test_rows" in decision.reason


def test_legacy_checkpoint_requires_explicit_override(monkeypatch):
    cfg, split_info = _cfg(), _split()
    decision = checkpoint_resume_decision({}, cfg, split_info)
    assert decision.ok is False
    monkeypatch.setenv("USIM_FB_ALLOW_LEGACY_CKPT", "1")
    decision = checkpoint_resume_decision({}, cfg, split_info)
    assert decision.ok is True
    assert decision.legacy_override is True
