from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_postformal_queue_waits_for_three_seeds_and_runs_core_ablations():
    source = (ROOT / "run_lira_v2_postformal_ablation_queue.ps1").read_text(encoding="utf-8")
    for seed in (2025, 2026, 2027):
        assert f"lira_v2_dynamic_dualloss_seed$_" in source or "2025,2026,2027" in source
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
