import copy
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
    return os.environ.get("USIM_FB_SAVE_OPT_STATE", "0") == "1"


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
