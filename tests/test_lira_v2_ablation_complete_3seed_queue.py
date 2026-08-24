from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_lira_v2_ablation_complete_3seed_queue.ps1"


def test_queue_completes_all_five_ablations_for_seeds_2026_and_2027():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "2026,2027" in source.replace(" ", "")
    for name in (
        "ablation_t0",
        "ablation_t1",
        "ablation_no_stop",
        "ablation_no_refined_loss",
        "ablation_no_stability",
    ):
        assert name in source

    assert "validation_finished.pt" in source
    assert "Wait-Process" in source
    assert "MaxParallel" in source


def test_queue_is_resumable_and_rejects_silent_fresh_restarts():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "SKIP" in source
    assert "Test-Path -LiteralPath $checkpoint" in source
    assert "-ForceFresh" not in source
    assert "run_learner_guided_full_seed2025.ps1" in source
    assert "[switch]$DryRun" in source


def test_queue_waits_for_seed2025_reference_ablations_before_launching():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "WAIT seed2025 reference ablations" in source
    assert "strict_item_cold_balanced_thr1_seed_2025" in source
    assert "Ablation failed or incomplete" in source
