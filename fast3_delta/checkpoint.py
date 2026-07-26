import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass

import torch

from .provenance import build_split_fingerprint, compare_source_manifests


# Keep the historical schema as the public default so pre-V1 static checkpoints
# retain their exact fingerprint and resume behavior. V1 owns its stricter
# contract through an explicit, isolated schema.
CHECKPOINT_FINGERPRINT_SCHEMA_VERSION = 2
V1_CHECKPOINT_FINGERPRINT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ResumeDecision:
    ok: bool
    reason: str
    train_fingerprint: str
    split_fingerprint: str
    source_warning: str = ""
    legacy_override: bool = False


def _feedback_ckpt_dir():
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "usim_feedback_fast3_content_delta"))


def _feedback_ckpt_enabled():
    return os.environ.get("USIM_FB_SAVE_CKPT", "1") == "1"


def _feedback_ckpt_auto_resume():
    return os.environ.get("USIM_FB_AUTO_RESUME", "1") == "1"


def _feedback_ckpt_force_fresh():
    return os.environ.get("USIM_FB_FORCE_FRESH", "0") == "1"


def _feedback_ckpt_save_optimizer_state():
    # Default ON so resumed runs keep optimizer momentum when possible.
    return os.environ.get("USIM_FB_SAVE_OPT_STATE", "1") == "1"


def _stable_json_fingerprint(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), payload


def _v1_enabled(cfg):
    return bool(getattr(cfg, "ckg_rl_v1_enabled", False) or getattr(cfg, "v1_enabled", False))


def _uses_extended_checkpoint_contract(cfg):
    """Whether a run opts into post-main-table resume semantics."""
    if _v1_enabled(cfg):
        return True
    return bool(
        getattr(cfg, "deterministic_eval_candidates", False)
        or getattr(cfg, "eval_reuse_item_bank", False)
        or getattr(cfg, "target_history_exclusion", False)
        or str(getattr(cfg, "simulator_target_mode", "legacy_id")) != "legacy_id"
    )


def checkpoint_fingerprint_schema_version(cfg):
    """Return the resume contract schema for this isolated training route."""
    if _uses_extended_checkpoint_contract(cfg):
        return V1_CHECKPOINT_FINGERPRINT_SCHEMA_VERSION
    return CHECKPOINT_FINGERPRINT_SCHEMA_VERSION


def _legacy_static_train_config_fingerprint(cfg, split_info=None, script_path=None):
    """Byte-compatible schema-2 fingerprint used by historical main-table runs."""
    split_info = split_info or {}

    def _cfg(name, default=None):
        return getattr(cfg, name, default)

    payload = {
        "schema_version": CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
        "data_dir": str(os.environ.get("USIM_DATA_DIR", "")),
        "seed": str(os.environ.get("USIM_SEED", os.environ.get("USIM_STATIC_SEED", ""))),
        "static_seed": str(os.environ.get("USIM_STATIC_SEED", "")),
        "split_mode": str(split_info.get("split_mode") or os.environ.get("USIM_STATIC_SPLIT_MODE", "")),
        "cold_threshold": int(_cfg("cold_threshold", int(os.environ.get("USIM_COLD_THRESHOLD", "1") or 1))),
        "early_stop_score_mode": str(_cfg("early_stop_score_mode", "")),
        "early_stop_average_mode": str(_cfg("early_stop_average_mode", "")),
        "use_content_delta": bool(_cfg("use_content_delta", False)),
        "content_delta_mode": str(_cfg("content_delta_mode", "")),
        "content_delta_scale": float(_cfg("content_delta_scale", 0.0) or 0.0),
        "rl_residual_scale": float(_cfg("rl_residual_scale", 1.0) or 1.0),
        "ppo_loss_weight": float(_cfg("ppo_loss_weight", 0.0) or 0.0),
        "rollout_policy": str(_cfg("rollout_policy", "")),
        "usim_steps": int(_cfg("usim_steps", _cfg("steps", 0)) or 0),
        "use_pseudo_cold_train": bool(_cfg("use_pseudo_cold_train", False)),
        "pseudo_cold_mode": str(_cfg("pseudo_cold_mode", "")),
        "pseudo_cold_ratio": float(_cfg("pseudo_cold_ratio", 0.0) or 0.0),
        "pseudo_cold_min_pop": int(_cfg("pseudo_cold_min_pop", 0) or 0),
        "use_course_reward": bool(_cfg("use_course_reward", False)),
        "use_course_sample": bool(_cfg("use_course_sample", False)),
        "use_prereq_aux_loss": bool(_cfg("use_prereq_aux_loss", False)),
        "recppo_warmup_epochs": int(
            _cfg("recppo_warmup_epochs", int(os.environ.get("USIM_RECPPO_WARMUP_EPOCHS", "-1") or -1)) or -1
        ),
        "recppo_enabled": bool(_cfg("recppo_enabled", False)),
        "emb_dim": int(_cfg("emb_dim", 0) or 0),
        "n_users": int(_cfg("n_users", 0) or 0),
        "n_items": int(_cfg("n_items", 0) or 0),
    }
    return _stable_json_fingerprint(payload)


def _static_train_config_fingerprint(cfg, split_info=None, script_path=None):
    """Fingerprint of train knobs that invalidate checkpoint resume when changed."""
    if not _uses_extended_checkpoint_contract(cfg):
        return _legacy_static_train_config_fingerprint(
            cfg,
            split_info=split_info,
            script_path=script_path,
        )

    split_info = split_info or {}

    def _cfg(name, default=None):
        return getattr(cfg, name, default)

    def _float_cfg(name, default=0.0):
        value = _cfg(name, default)
        return float(default if value is None else value)

    def _int_cfg(name, default=0):
        value = _cfg(name, default)
        return int(default if value is None else value)

    def _bool_cfg(name, default=False):
        value = _cfg(name, default)
        return bool(default if value is None else value)

    ppo_coeffs = _cfg("ppo_coeffs", {})
    if not isinstance(ppo_coeffs, dict):
        ppo_coeffs = {}
    v1_enabled = _v1_enabled(cfg)
    payload = {
        "schema_version": V1_CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
        "v1_enabled": v1_enabled,
        "data_dir": str(os.environ.get("USIM_DATA_DIR", "")),
        "seed": str(os.environ.get("USIM_SEED", os.environ.get("USIM_STATIC_SEED", ""))),
        "static_seed": str(os.environ.get("USIM_STATIC_SEED", "")),
        "split_mode": str(split_info.get("split_mode") or os.environ.get("USIM_STATIC_SPLIT_MODE", "")),
        "cold_threshold": _int_cfg(
            "cold_threshold", int(os.environ.get("USIM_COLD_THRESHOLD", "1") or 1)
        ),
        "early_stop_score_mode": str(_cfg("early_stop_score_mode", "")),
        "early_stop_average_mode": str(_cfg("early_stop_average_mode", "")),
        "use_content_delta": _bool_cfg("use_content_delta", False),
        "content_delta_mode": str(_cfg("content_delta_mode", "")),
        "content_delta_scale": _float_cfg("content_delta_scale", 0.0),
        "rl_residual_scale": _float_cfg("rl_residual_scale", 1.0),
        "ppo_loss_weight": _float_cfg("ppo_loss_weight", 0.0),
        "rollout_policy": str(_cfg("rollout_policy", "")),
        "usim_steps": _int_cfg("usim_steps", _cfg("steps", 0)),
        "deterministic_eval_candidates": _bool_cfg("deterministic_eval_candidates", False),
        "deterministic_eval_seed": _int_cfg("deterministic_eval_seed", 0),
        "eval_reuse_item_bank": _bool_cfg("eval_reuse_item_bank", False),
        "simulator_target_mode": str(_cfg("simulator_target_mode", "legacy_id")),
        "use_pseudo_cold_train": _bool_cfg("use_pseudo_cold_train", False),
        "pseudo_cold_mode": str(_cfg("pseudo_cold_mode", "")),
        "pseudo_cold_ratio": _float_cfg("pseudo_cold_ratio", 0.0),
        "pseudo_cold_min_pop": _int_cfg("pseudo_cold_min_pop", 0),
        "train_force_cold": _bool_cfg("train_force_cold", False),
        "aux_hot_only": _bool_cfg("aux_hot_only", False),
        "aux_weight": _float_cfg("aux_weight", 0.0),
        "mask_known_pos_neg": _bool_cfg("mask_known_pos_neg", False),
        "mask_same_item_neg": _bool_cfg("mask_same_item_neg", False),
        "use_structured_hard_neg": _bool_cfg("use_structured_hard_neg", False),
        "use_course_reward": _bool_cfg("use_course_reward", False),
        "use_prereq_aux_loss": _bool_cfg("use_prereq_aux_loss", False),
        "recppo_warmup_epochs": _int_cfg(
            "recppo_warmup_epochs",
            int(os.environ.get("USIM_RECPPO_WARMUP_EPOCHS", "-1") or -1),
        ),
        "recppo_enabled": _bool_cfg("recppo_enabled", False),
        "batch_size": _int_cfg("batch_size", 0),
        "n_candidates": _int_cfg("n_candidates", 0),
        "candidate_strategy": str(_cfg("candidate_strategy", "")),
        "retrieve_top_m": _int_cfg("retrieve_top_m", 0),
        "candidate_temp": _float_cfg("candidate_temp", 0.0),
        "candidate_epsilon": _float_cfg("candidate_epsilon", 0.0),
        "retrieval_user_chunk": _int_cfg("retrieval_user_chunk", 0),
        "retrieval_query_chunk": _int_cfg("retrieval_query_chunk", 0),
        "user_bank_refresh_steps": _int_cfg("user_bank_refresh_steps", 0),
        "usim_lr": _float_cfg("usim_lr", 0.0),
        "feedback_load_course_artifacts": _bool_cfg(
            "feedback_load_course_artifacts", False
        ),
        "feedback_course_sample_soft": _bool_cfg("feedback_course_sample_soft", False),
        "feedback_course_sample_beta": _float_cfg("feedback_course_sample_beta", 0.0),
        "feedback_course_sample_only_cold": _bool_cfg(
            "feedback_course_sample_only_cold", False
        ),
        "feedback_course_sample_topk": _int_cfg("feedback_course_sample_topk", 0),
        "feedback_course_sample_top_l": _int_cfg("feedback_course_sample_top_l", 0),
        "feedback_course_match_exclude_target": _bool_cfg(
            "feedback_course_match_exclude_target", False
        ),
        "feedback_course_only_cold": _bool_cfg("feedback_course_only_cold", False),
        "feedback_course_warm_seen": _int_cfg("feedback_course_warm_seen", 0),
        "feedback_course_concept_min": _float_cfg("feedback_course_concept_min", 0.0),
        "feedback_course_match_mode": str(_cfg("feedback_course_match_mode", "")),
        "feedback_course_match_topk": _int_cfg("feedback_course_match_topk", 0),
        "feedback_course_redundant_mode": str(
            _cfg("feedback_course_redundant_mode", "")
        ),
        "feedback_course_redundant_thr": _float_cfg(
            "feedback_course_redundant_thr", 0.0
        ),
        "feedback_course_struct_video_min": _float_cfg(
            "feedback_course_struct_video_min", 0.0
        ),
        "feedback_course_struct_chunk": _int_cfg("feedback_course_struct_chunk", 0),
        "feedback_course_prereq_gate": _float_cfg("feedback_course_prereq_gate", 0.0),
        "feedback_course_prereq_weight": _float_cfg("feedback_course_prereq_weight", 0.0),
        "feedback_prereq_weighted_edges": _bool_cfg(
            "feedback_prereq_weighted_edges", False
        ),
        "feedback_prereq_soft_penalty": _bool_cfg("feedback_prereq_soft_penalty", False),
        "feedback_course_concept_weight": _float_cfg("feedback_course_concept_weight", 0.0),
        "feedback_course_difficulty_weight": _float_cfg(
            "feedback_course_difficulty_weight", 0.0
        ),
        "feedback_course_redundant_weight": _float_cfg(
            "feedback_course_redundant_weight", 0.0
        ),
        "feedback_course_redundant_concept_gate": _float_cfg(
            "feedback_course_redundant_concept_gate", 0.0
        ),
        "feedback_course_term_norm": str(_cfg("feedback_course_term_norm", "")),
        "feedback_course_term_norm_clip": _float_cfg("feedback_course_term_norm_clip", 0.0),
        "feedback_course_term_norm_eps": _float_cfg("feedback_course_term_norm_eps", 0.0),
        "feedback_course_term_norm_ema_decay": _float_cfg(
            "feedback_course_term_norm_ema_decay", 0.0
        ),
        "use_course_rerank": _bool_cfg("use_course_rerank", False),
        "rerank_alpha": _float_cfg("rerank_alpha", 0.0),
        "rerank_lambda": _float_cfg("rerank_lambda", 0.0),
        "rerank_min_seen": _int_cfg("rerank_min_seen", 0),
        "rerank_top_l": _int_cfg("rerank_top_l", 0),
        "rerank_penalty_cap": _float_cfg("rerank_penalty_cap", 0.0),
        "rerank_only_cold": _bool_cfg("rerank_only_cold", False),
        "concept_overlap_mode": str(_cfg("concept_overlap_mode", "")),
        "prereq_graph_source": str(_cfg("prereq_graph_source", "")),
        "prereq_concept_score_thr": _float_cfg("prereq_concept_score_thr", 0.0),
        "prereq_concept_min_hits": _int_cfg("prereq_concept_min_hits", 0),
        "prereq_concept_file": str(_cfg("prereq_concept_file", "")),
        "reward_terminal_weight": _float_cfg("reward_terminal_weight", 0.0),
        "reward_gain_weight": _float_cfg("reward_gain_weight", 0.0),
        "reward_gain_clip": _float_cfg("reward_gain_clip", 0.0),
        "reward_dup_penalty_weight": _float_cfg("reward_dup_penalty_weight", 0.0),
        "reward_cov_bonus_weight": _float_cfg("reward_cov_bonus_weight", 0.0),
        "ppo_clip": _float_cfg("ppo_clip", 0.0),
        "ppo_gamma": _float_cfg("ppo_gamma", 0.0),
        "ppo_epochs": _int_cfg("ppo_epochs", 0),
        "ppo_lambda": _float_cfg("ppo_lambda", 0.0),
        "ppo_value_clip": _float_cfg("ppo_value_clip", 0.0),
        "ppo_adv_norm": _bool_cfg("ppo_adv_norm", False),
        "ppo_coeffs": {
            "value": float(ppo_coeffs.get("value", 0.0)),
            "entropy": float(ppo_coeffs.get("entropy", 0.0)),
        },
        "fast3_target_alpha_cold": _float_cfg("fast3_target_alpha_cold", 0.0),
        "fast3_target_alpha_hot": _float_cfg("fast3_target_alpha_hot", 0.0),
        "fast3_target_alpha_step": _float_cfg("fast3_target_alpha_step", 0.0),
        "fast3_target_alpha_entropy": _float_cfg("fast3_target_alpha_entropy", 0.0),
        "fast3_target_alpha_min": _float_cfg("fast3_target_alpha_min", 0.0),
        "fast3_target_alpha_max": _float_cfg("fast3_target_alpha_max", 0.0),
        "emb_dim": _int_cfg("emb_dim", 0),
        "n_users": _int_cfg("n_users", 0),
        "n_items": _int_cfg("n_items", 0),
    }
    target_history_exclusion = _bool_cfg("target_history_exclusion", False)
    if target_history_exclusion and not v1_enabled:
        payload.update(
            {
                "target_history_exclusion": True,
                "target_history_exclusion_scope": str(
                    _cfg("target_history_exclusion_scope", "")
                ),
            }
        )
    if v1_enabled:
        payload.update(
            {
                "v1_contract_version": _int_cfg("v1_contract_version", 1),
                "use_epoch_early_stop": _bool_cfg("use_epoch_early_stop", False),
                "early_stop_k": _int_cfg("early_stop_k", 0),
                "early_stop_patience": _int_cfg("early_stop_patience", 0),
                "early_stop_min_delta": _float_cfg("early_stop_min_delta", 0.0),
                "v1_reference_batch_size": _int_cfg("reference_batch_size", 0),
                "v1_target_history_exclusion": _bool_cfg(
                    "target_history_exclusion", False
                ),
                "v1_target_history_exclusion_scope": str(
                    _cfg("target_history_exclusion_scope", "")
                ),
                "v1_pseudo_cold_plan_hash": str(_cfg("pseudo_cold_plan_hash", "")),
                "v1_pseudo_cold_plan_count": _int_cfg("pseudo_cold_plan_count", 0),
                "v1_pseudo_cold_plan_seed": _int_cfg("pseudo_cold_plan_seed", 0),
                "v1_pseudo_cold_plan_strategy": str(
                    _cfg("pseudo_cold_plan_strategy", "")
                ),
                "v1_selector_mode": str(_cfg("selector_mode", "")),
                "v1_selector_hot_tolerance": _float_cfg("selector_hot_tolerance", 0.0),
                "v1_selector_overall_tolerance": _float_cfg(
                    "selector_overall_tolerance", 0.0
                ),
            }
        )
    return _stable_json_fingerprint(payload)


def _payload_differences(old_payload, new_payload):
    if not isinstance(old_payload, dict):
        return []
    differences = []
    for key in sorted(set(old_payload) | set(new_payload)):
        old = old_payload.get(key, "<missing>")
        new = new_payload.get(key, "<missing>")
        if old != new:
            differences.append(f"{key}:{old}->{new}")
    return differences


def checkpoint_resume_decision(resume_state, cfg, split_info=None, current_source_manifest=None, split_exports=None):
    split_info = split_info or {}
    train_fp, train_payload = _static_train_config_fingerprint(cfg, split_info=split_info)
    split_fp, split_payload = build_split_fingerprint(split_info, exports=split_exports)
    v1_enabled = bool(train_payload.get("v1_enabled", False))
    expected_schema = checkpoint_fingerprint_schema_version(cfg)
    if not isinstance(resume_state, dict):
        return ResumeDecision(False, "checkpoint missing or invalid", train_fp, split_fp)

    schema = resume_state.get("fingerprint_schema_version")
    ckpt_train_fp = resume_state.get("train_config_fingerprint")
    ckpt_split_fp = resume_state.get("split_fingerprint")
    if schema != expected_schema or not ckpt_train_fp or not ckpt_split_fp:
        if not v1_enabled and os.environ.get("USIM_FB_ALLOW_LEGACY_CKPT", "0") == "1":
            return ResumeDecision(True, "legacy checkpoint override", train_fp, split_fp, legacy_override=True)
        if v1_enabled:
            return ResumeDecision(False, "legacy checkpoint incompatible with V1", train_fp, split_fp)
        return ResumeDecision(False, "legacy checkpoint requires USIM_FB_ALLOW_LEGACY_CKPT=1", train_fp, split_fp)

    if str(ckpt_train_fp) != str(train_fp):
        diffs = _payload_differences(resume_state.get("train_config_payload"), train_payload)
        reason = "train config fingerprint mismatch"
        if diffs:
            reason += " | " + "; ".join(diffs[:12])
        return ResumeDecision(False, reason, train_fp, split_fp)

    if str(ckpt_split_fp) != str(split_fp):
        diffs = _payload_differences(resume_state.get("split_payload"), split_payload)
        reason = "split fingerprint mismatch"
        if diffs:
            reason += " | " + "; ".join(diffs[:12])
        return ResumeDecision(False, reason, train_fp, split_fp)

    warning = ""
    expected_source = resume_state.get("source_manifest")
    if v1_enabled:
        if not isinstance(expected_source, dict) or not isinstance(
            expected_source.get("files"), dict
        ):
            return ResumeDecision(
                False,
                "V1 checkpoint missing source provenance",
                train_fp,
                split_fp,
            )
        if not isinstance(current_source_manifest, dict) or not isinstance(
            current_source_manifest.get("files"), dict
        ):
            return ResumeDecision(
                False,
                "V1 resume missing current source provenance",
                train_fp,
                split_fp,
            )
    if isinstance(expected_source, dict) and isinstance(current_source_manifest, dict):
        changes = compare_source_manifests(expected_source, current_source_manifest)
        parts = [f"{kind}={','.join(paths)}" for kind, paths in changes.items() if paths]
        if parts:
            warning = "WARNING: source provenance differs | " + " | ".join(parts)
            if v1_enabled:
                return ResumeDecision(
                    False,
                    "source provenance mismatch | " + " | ".join(parts),
                    train_fp,
                    split_fp,
                    source_warning=warning,
                )
    return ResumeDecision(True, "fingerprint match", train_fp, split_fp, source_warning=warning)


def _checkpoint_config_matches(resume_state, cfg, split_info=None, script_path=None):
    """Return (ok, reason, current_fp, ckpt_fp)."""
    decision = checkpoint_resume_decision(resume_state, cfg, split_info=split_info)
    ckpt_fp = resume_state.get("train_config_fingerprint") if isinstance(resume_state, dict) else None
    return decision.ok, decision.reason, decision.train_fingerprint, None if ckpt_fp is None else str(ckpt_fp)


def _feedback_ckpt_snapshot_epochs():
    raw = os.environ.get("USIM_FB_SNAPSHOT_EPOCHS", "").strip()
    if not raw:
        return set()
    epochs = set()
    for part in raw.replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            epoch = int(text)
        except ValueError:
            continue
        if epoch > 0:
            epochs.add(epoch)
    return epochs


def _serialize_user_seen_items(user_seen_items):
    return {
        int(uid): sorted(int(it) for it in items)
        for uid, items in user_seen_items.items()
    }


def _deserialize_user_seen_items(payload):
    if not payload:
        return {}
    return {
        int(uid): set(int(it) for it in items)
        for uid, items in payload.items()
    }


def _latest_feedback_ckpt_path(ckpt_dir):
    return os.path.join(ckpt_dir, "latest.pt")


def _save_feedback_checkpoint(ckpt_dir, state, snapshot_name=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    tmp_path = latest_path + ".tmp"
    state = copy.deepcopy(state)
    state["saved_at"] = time.time()
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    if snapshot_name:
        snapshot_path = os.path.join(ckpt_dir, snapshot_name)
        torch.save(state, snapshot_path)
    return latest_path


def _load_feedback_checkpoint(ckpt_dir):
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    if not os.path.exists(latest_path):
        return None
    return torch.load(latest_path, map_location="cpu")


def _move_state_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _maybe_clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_feedback_ckpt_state(
    model,
    optimizer,
    history,
    accum_cold,
    accum_hot,
    count_cold,
    count_hot,
    full_cold,
    full_hot,
    fc_cold,
    fc_hot,
    user_seen_items,
    accumulated_periods,
    warmup_periods,
    total_periods,
    status,
    next_period,
    current_period=None,
    next_epoch=0,
    es_best=None,
    es_best_state=None,
    es_best_opt_state=None,
    es_no_improve=0,
):
    save_opt_state = _feedback_ckpt_save_optimizer_state()
    return {
        "version": 1,
        "status": status,
        "next_period": int(next_period),
        "current_period": None if current_period is None else int(current_period),
        "next_epoch": int(next_epoch),
        "accumulated_periods": int(accumulated_periods),
        "warmup_periods": int(warmup_periods),
        "total_periods": int(total_periods),
        "history": copy.deepcopy(history),
        "accum_cold": copy.deepcopy(accum_cold),
        "accum_hot": copy.deepcopy(accum_hot),
        "count_cold": int(count_cold),
        "count_hot": int(count_hot),
        "full_cold": copy.deepcopy(full_cold),
        "full_hot": copy.deepcopy(full_hot),
        "fc_cold": int(fc_cold),
        "fc_hot": int(fc_hot),
        "user_seen_items": _serialize_user_seen_items(user_seen_items),
        "model_state": _move_state_to_cpu(model.state_dict()),
        "optimizer_state": _move_state_to_cpu(optimizer.state_dict()) if save_opt_state else None,
        "es_best": copy.deepcopy(es_best),
        "es_best_state": _move_state_to_cpu(es_best_state),
        "es_best_opt_state": _move_state_to_cpu(es_best_opt_state) if save_opt_state else None,
        "es_no_improve": int(es_no_improve),
    }
