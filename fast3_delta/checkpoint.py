import copy
import hashlib
import json
import os
import time

import torch


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


def _static_train_config_fingerprint(cfg, split_info=None, script_path=None):
    """Fingerprint of train knobs that invalidate checkpoint resume when changed."""
    split_info = split_info or {}
    script_name = ""
    script_sha = ""
    if script_path:
        script_name = os.path.basename(os.path.abspath(script_path))
        try:
            with open(script_path, "rb") as handle:
                script_sha = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            script_sha = ""

    def _cfg(name, default=None):
        return getattr(cfg, name, default)

    payload = {
        "script_name": script_name,
        # Script body hash: code changes that affect training invalidate resume.
        "script_sha256": script_sha if os.environ.get("USIM_FB_CKPT_IGNORE_SCRIPT_HASH", "0") != "1" else "",
        "data_dir": str(os.environ.get("USIM_DATA_DIR", "")),
        "seed": str(os.environ.get("USIM_SEED", os.environ.get("USIM_STATIC_SEED", ""))),
        "static_seed": str(os.environ.get("USIM_STATIC_SEED", "")),
        "split_mode": str(split_info.get("split_mode") or os.environ.get("USIM_STATIC_SPLIT_MODE", "")),
        "cold_threshold": int(_cfg("cold_threshold", int(os.environ.get("USIM_COLD_THRESHOLD", "1") or 1))),
        "n_epochs": int(_cfg("n_epochs", 0) or 0),
        "early_stop_patience": int(_cfg("early_stop_patience", 0) or 0),
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


def _checkpoint_config_matches(resume_state, cfg, split_info=None, script_path=None):
    """Return (ok, reason, current_fp, ckpt_fp)."""
    current_fp, current_payload = _static_train_config_fingerprint(cfg, split_info=split_info, script_path=script_path)
    if not isinstance(resume_state, dict):
        return False, "checkpoint missing or invalid", current_fp, None
    ckpt_fp = resume_state.get("train_config_fingerprint")
    ckpt_payload = resume_state.get("train_config_payload")
    if not ckpt_fp:
        # Legacy checkpoints: fall back to coarse fields when present.
        coarse = {
            "n_epochs": int(resume_state.get("n_epochs_requested", -1)),
            "cold_threshold": int(resume_state.get("cold_threshold", -1)),
            "split_mode": str(resume_state.get("split_mode", "")),
        }
        cur_coarse = {
            "n_epochs": int(getattr(cfg, "n_epochs", -1)),
            "cold_threshold": int(getattr(cfg, "cold_threshold", -1)),
            "split_mode": str((split_info or {}).get("split_mode", "")),
        }
        if coarse != cur_coarse:
            return False, f"legacy coarse config mismatch ckpt={coarse} cur={cur_coarse}", current_fp, None
        return True, "legacy checkpoint without fingerprint (coarse match)", current_fp, None
    if str(ckpt_fp) != str(current_fp):
        diffs = []
        if isinstance(ckpt_payload, dict):
            keys = sorted(set(ckpt_payload) | set(current_payload))
            for key in keys:
                old = ckpt_payload.get(key, "<missing>")
                new = current_payload.get(key, "<missing>")
                if old != new:
                    diffs.append(f"{key}:{old}->{new}")
        reason = "train config fingerprint mismatch"
        if diffs:
            reason = reason + " | " + "; ".join(diffs[:12])
            if len(diffs) > 12:
                reason = reason + f" | ...(+{len(diffs) - 12} more)"
        return False, reason, current_fp, str(ckpt_fp)
    return True, "fingerprint match", current_fp, str(ckpt_fp)


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
