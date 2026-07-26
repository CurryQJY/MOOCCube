from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_master_queue_runs_approved_stages_in_order_and_keeps_cgrc_last():
    source = read("run_overnight_recommended_then_cgrc.ps1")
    stages = [
        "run_cross_dataset_actor_inference_ab.ps1",
        "run_mooccube_test_policy_controls.ps1",
        "run_recovered_missing_ablation_wo_sampler.ps1",
        "run_cgrc_p1_reproduction_resume.ps1",
    ]
    positions = [source.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "[switch]$DryRun" in source
    assert "Invoke-Stage" in source
    assert "stage_failures.json" in source


def test_master_queue_relies_on_powershell_exceptions_not_stale_native_exit_code():
    source = read("run_overnight_recommended_then_cgrc.ps1")
    assert "if ($LASTEXITCODE -ne 0)" not in source


def test_cross_dataset_actor_queue_uses_existing_three_seed_checkpoints_read_only():
    source = read("run_cross_dataset_actor_inference_ab.ps1")
    assert "checkpoints\\junyi\\main_table_3seed\\ours" in source
    assert "checkpoints\\coco\\single_seed_triage\\ours_full" in source
    assert '@("static", "actor")' in source
    assert "main_checkpoint_actor_inference_ab.py" in source
    assert "USIM_ACTOR_EVAL_TARGET" in source
    assert "2025, 2026, 2027" in source
    assert "[switch]$DryRun" in source


def test_mooccube_policy_queue_has_deterministic_controls_and_repeated_random_control():
    source = read("run_mooccube_test_policy_controls.ps1")
    assert '@("static", "ppo", "greedy_similarity", "course_fit")' in source
    assert "7001, 7002, 7003, 7004, 7005" in source
    assert "random_seed_$evalSeed" in source
    assert "USIM_ACTOR_EVAL_TARGET = \"test\"" in source
    assert "[switch]$DryRun" in source


def test_missing_sampler_ablation_changes_only_sampler_and_replays_actor_after_training():
    source = read("run_recovered_missing_ablation_wo_sampler.ps1")
    assert "2025, 2026, 2027" in source
    assert "UseCourseSample = $false" in source
    assert "UseCourseReward = $true" in source
    assert "UsePrereqAux = $true" in source
    assert "PpoLossWeight = 1.0" in source
    assert "UsimSteps = 5" in source
    assert "AutoResume = $true" in source
    assert "ForceFresh = $false" in source
    assert "USIM_ACTOR_INFERENCE_MODE = \"actor\"" in source
    assert '@("validation", "test")' in source


def test_cgrc_resume_skips_completed_seed_and_resumes_2026_before_2027():
    source = read("run_cgrc_p1_reproduction_resume.ps1")
    assert "2026, 2027" in source
    assert "cgrc_paper_static_result.json" in source
    assert "CGRC_PAPER_AUTO_RESUME" in source
    assert "CGRC_PAPER_CKPT_DIR" in source
    assert "2>&1" in source
    assert "aggregate_main_table_static_results.py" in source
    assert "[switch]$DryRun" in source
