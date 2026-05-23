import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class CheckpointConfig:
    dir: str
    save: bool
    resume: bool
    force_fresh: bool
    save_opt: bool


def checkpoint_config(prefix: str) -> CheckpointConfig:
    prefix = prefix.strip().upper()
    default_dir = os.environ.get("BASELINE_CKPT_DIR", "").strip()
    ckpt_dir = os.environ.get(f"{prefix}_CKPT_DIR", default_dir).strip()
    return CheckpointConfig(
        dir=ckpt_dir,
        save=env_flag(f"{prefix}_SAVE_CKPT", env_flag("BASELINE_SAVE_CKPT", bool(ckpt_dir))),
        resume=env_flag(f"{prefix}_AUTO_RESUME", env_flag("BASELINE_AUTO_RESUME", bool(ckpt_dir))),
        force_fresh=env_flag(f"{prefix}_FORCE_FRESH", env_flag("BASELINE_FORCE_FRESH", False)),
        save_opt=env_flag(f"{prefix}_SAVE_OPT_STATE", env_flag("BASELINE_SAVE_OPT_STATE", True)),
    )


def _state_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {name: module.state_dict() for name, module in obj.items()}
    return obj.state_dict()


def _load_state(obj: Any, state: Any) -> None:
    if obj is None or state is None:
        return
    if isinstance(obj, dict):
        for name, module in obj.items():
            if name in state:
                module.load_state_dict(state[name])
        return
    obj.load_state_dict(state)


def save_checkpoint(
    cfg: CheckpointConfig,
    filename: str,
    epoch: int,
    model: Any,
    optimizer: Any = None,
    best_state: Optional[Dict[str, torch.Tensor]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not cfg.dir:
        return
    os.makedirs(cfg.dir, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": _state_dict(model),
        "optimizer_state": _state_dict(optimizer) if cfg.save_opt else None,
        "best_state": best_state,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(cfg.dir, filename))


def maybe_resume_checkpoint(
    cfg: CheckpointConfig,
    model: Any,
    optimizer: Any = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, Dict[str, Any]]:
    if not cfg.dir:
        return 0, {}
    print(
        f"Checkpoint: save={cfg.save} resume={cfg.resume} "
        f"force_fresh={cfg.force_fresh} save_opt={cfg.save_opt} dir={cfg.dir}"
    )
    if cfg.force_fresh or not cfg.resume:
        return 0, {}
    latest_path = os.path.join(cfg.dir, "latest.pt")
    if not os.path.exists(latest_path):
        return 0, {}
    try:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(latest_path, map_location=device)
    _load_state(model, ckpt.get("model_state"))
    _load_state(optimizer, ckpt.get("optimizer_state"))
    epoch = int(ckpt.get("epoch", 0))
    print(
        f"Resume checkpoint: latest_epoch={epoch} | "
        f"best_epoch={ckpt.get('best_epoch', -1)} | best_score={float(ckpt.get('best_val', -1.0)):.6f}"
    )
    return epoch, ckpt
