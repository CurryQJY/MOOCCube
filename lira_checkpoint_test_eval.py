"""Read-only test replay for a validation-selected LIRA checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as shared_protocol
from lira.protocol_adapter import LIRAProtocolAdapter


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def load_best_validation_state(checkpoint_dir: str | os.PathLike) -> dict:
    """Load the validation-selected state and make the training range empty."""
    path = Path(checkpoint_dir) / "validation_finished.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    best_state = state.get("es_best_state")
    if not best_state:
        raise RuntimeError(f"validation checkpoint has no es_best_state: {path}")

    replay = dict(state)
    replay["model_state"] = best_state
    replay["optimizer_state"] = None
    replay["next_epoch"] = 2**31 - 1
    replay["status"] = "test_replay"
    return replay


def skip_checkpoint_write(checkpoint_dir, state, snapshot_name=None):
    """Keep the source checkpoint tree read-only during evaluation."""
    target = Path(checkpoint_dir) / (snapshot_name or "latest.pt")
    print(f">> LIRA TEST READ-ONLY: skipped checkpoint write {target}", flush=True)
    return str(Path(checkpoint_dir) / "latest.pt")


def main() -> None:
    shared_protocol.Fast3FeedbackUSIM = LIRAProtocolAdapter
    shared_protocol._load_feedback_checkpoint = load_best_validation_state
    shared_protocol._save_feedback_checkpoint = skip_checkpoint_write
    os.environ["USIM_VALIDATION_ONLY"] = "0"
    shared_protocol.setup_seed(
        int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025")))
    )
    shared_protocol.main()


if __name__ == "__main__":
    main()
