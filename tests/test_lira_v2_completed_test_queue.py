from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_lira_v2_completed_test_queue.ps1"


def test_completed_test_queue_contains_only_finished_seed_sets():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "lira_v2_dynamic_dualloss_seed2025" in source
    assert "lira_v2_dynamic_dualloss_seed2026" in source
    assert "lira_v2_dynamic_dualloss_seed2027" in source
    for name in (
        "ablation_t0",
        "ablation_t1",
        "ablation_no_stop",
        "ablation_no_refined_loss",
        "ablation_no_stability",
    ):
        assert name in source
    assert "lira_checkpoint_test_eval.py" in source
    assert "validation_finished.pt" in source
    assert "-ForceFresh" not in source


def test_completed_test_queue_is_serial_and_read_only():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "foreach ($job in $jobs)" in source
    assert "test_finished" in source
    assert "locked_config.json" in source
    assert "source_sha256" in source
    assert "ALL COMPLETED TEST REPLAYS DONE" in source
