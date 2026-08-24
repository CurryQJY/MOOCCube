import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v1_launcher_is_isolated_and_locks_the_semantic_contract():
    launcher = (ROOT / "run_ckg_rl_v1_seed2025.ps1").read_text(encoding="utf-8")

    required = [
        '[switch]$DryRun',
        '[string]$RunName = "seed2025"',
        '[int]$BatchSize = 2048',
        'OutputRoot = $runOutputRelative',
        'CheckpointRoot = $runCheckpointRelative',
        '"USIM_BATCH_SIZE" = [string]$BatchSize',
        'PseudoColdMode = "fixed_item_stratified"',
        'TrainForceCold = $false',
        'AuxHotOnly = $true',
        'PpoLossWeight = 0.0',
        'RolloutPolicy = "course_fit"',
        'SimulatorTargetMode = "initial_state"',
        'DeterministicEvalCandidates = $true',
        'EvalReuseItemBank = $true',
        'UseCourseReward = $false',
        'UsePrereqAux = $false',
        'UseSageLite = $false',
        'UseCgrcRecon = $false',
        'USIM_CKG_RL_V1',
        'USIM_V1_REFERENCE_BATCH_SIZE',
        'USIM_V1_TARGET_HISTORY_EXCLUSION',
        'USIM_FB_COURSE_MATCH_EXCLUDE_TARGET',
        'USIM_PPO_EPOCHS',
        'USIM_FB_REWARD_DUP_W',
        'USIM_BATCH_SIZE',
        'USIM_USE_EPOCH_EARLY_STOP',
        'USIM_FB_ENTRY_SCRIPT',
        'USIM_FB_COURSE_WARM_SEEN',
        'USIM_FB_COURSE_CONCEPT_MIN',
        'USIM_FB_COURSE_REDUNDANT_THR',
        'USIM_FB_COURSE_STRUCT_VIDEO_MIN',
        'USIM_FB_PREREQ_WEIGHTED_EDGES',
        'USIM_FB_PREREQ_SOFT_PENALTY',
        'USIM_PREREQ_CONCEPT_SCORE_THR',
        'USIM_PREREQ_CONCEPT_MIN_HITS',
        'Push-Location -LiteralPath $repoPath',
        'Refusing to overwrite an existing CKG-RL V1 run',
        '$Seed -ne 2025',
    ]
    for fragment in required:
        assert fragment in launcher


def test_static_runner_accepts_the_fixed_item_v1_pseudocold_mode():
    runner = (ROOT / "run_usim_feedback_fast3_content_delta_static.ps1").read_text(
        encoding="utf-8"
    )

    assert '"fixed_item_stratified"' in runner
    assert 'USIM_CKG_RL_V1' in runner
    assert 'USIM_V1_REFERENCE_BATCH_SIZE' in runner
    assert 'USIM_V1_TARGET_HISTORY_EXCLUSION' in runner


def test_v1_launcher_dry_run_accepts_an_isolated_reduced_batch_run_name():
    launcher = ROOT / "run_ckg_rl_v1_seed2025.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-DryRun",
            "-RunName",
            "seed2025_bs1024",
            "-BatchSize",
            "1024",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OutputRoot=outputs\\ckg_rl_v1\\seed2025_bs1024" in result.stdout


def test_v1_launcher_dry_run_accepts_a_zero_step_simulator_control():
    launcher = ROOT / "run_ckg_rl_v1_seed2025.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-DryRun",
            "-RunName",
            "seed2025_nosim_bs1024",
            "-BatchSize",
            "1024",
            "-UsimSteps",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OutputRoot=outputs\\ckg_rl_v1\\seed2025_nosim_bs1024" in result.stdout
    assert "UsimSteps=0" in result.stdout


def test_v1_launcher_dry_run_propagates_the_running_retention_selector_mode():
    launcher = ROOT / "run_ckg_rl_v1_seed2025.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-DryRun",
            "-RunName",
            "seed2025_selector_pareto_bs1024",
            "-BatchSize",
            "1024",
            "-V1SelectorMode",
            "cold_ndcg_running_retention",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SelectorMode=cold_ndcg_running_retention" in result.stdout
