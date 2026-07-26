from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_test_replay_loads_validation_best_state_and_skips_training(tmp_path):
    from lira_checkpoint_test_eval import load_best_validation_state

    best = {"weight": torch.tensor([2.0])}
    torch.save(
        {
            "model_state": {"weight": torch.tensor([1.0])},
            "es_best_state": best,
            "optimizer_state": {"state": {1: {}}},
            "next_epoch": 7,
            "status": "validation_finished",
        },
        tmp_path / "validation_finished.pt",
    )

    replay = load_best_validation_state(tmp_path)

    assert torch.equal(replay["model_state"]["weight"], best["weight"])
    assert replay["optimizer_state"] is None
    assert replay["next_epoch"] > 1_000_000
    assert replay["status"] == "test_replay"


def test_test_replay_rejects_missing_best_validation_state(tmp_path):
    from lira_checkpoint_test_eval import load_best_validation_state

    torch.save({"model_state": {}}, tmp_path / "validation_finished.pt")
    with pytest.raises(RuntimeError, match="es_best_state"):
        load_best_validation_state(tmp_path)


def test_seed2025_test_queue_covers_full_and_all_five_ablations_read_only():
    source = (ROOT / "run_lira_v2_seed2025_test_trend_queue.ps1").read_text(
        encoding="utf-8"
    )

    for name in (
        "dynamic_dualloss",
        "ablation_t0",
        "ablation_t1",
        "ablation_no_stop",
        "ablation_no_refined_loss",
        "ablation_no_stability",
    ):
        assert name in source
    assert "lira_checkpoint_test_eval.py" in source
    assert "validation_finished.pt" in source
    assert "WAIT 3-seed ablation training queue" in source
    assert "USIM_VALIDATION_ONLY = \"0\"" in source
    assert "-ForceFresh" not in source
    assert "-ValidationOnly" not in source
