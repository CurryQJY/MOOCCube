"""Launcher and static-runner contracts for the isolated CKG-RL V3 route."""

import json
from pathlib import Path
import subprocess

import ckg_rl_usim_v3 as v3


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v3_module_declares_static_delegate_entrypoints():
    assert v3.USIM_STATIC_DELEGATE_ENTRYPOINT is True
    assert callable(v3.run_static_experiment)
    assert callable(v3.main)


def test_v3_launcher_is_seed2025_isolated_and_enables_current_ckg_components():
    launcher = REPO_ROOT / "run_ckg_rl_usim_v3_seed2025.ps1"
    text = launcher.read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v3.py"' in text
    assert 'outputs\\ckg_rl_usim_v3' in text
    assert 'checkpoints\\ckg_rl_usim_v3' in text
    assert '"USIM_ORIGINAL_V2" = "1"' in text
    assert '"USIM_V3_CORE" = "1"' in text
    assert '"USIM_BATCH_SIZE" = [string]$BatchSize' in text
    assert 'UseCourseReward = $true' in text
    assert 'UsePrereqAux = $true' in text
    assert 'UseCourseSample = $true' in text
    assert 'UsePseudoColdTrain = $true' in text
    assert '-not $DryRun' in text


def test_v3_launcher_dryrun_formats_its_contract_values():
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "run_ckg_rl_usim_v3_seed2025.ps1"),
            "-DryRun",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "teacher={0}" not in result.stdout
    assert "pseudo_ratio=0.3" in result.stdout
    assert "candidates=20" in result.stdout


def test_v31_launcher_is_seed2025_isolated_and_declares_support_repair():
    launcher = REPO_ROOT / "run_ckg_rl_usim_v31_seed2025.ps1"
    text = launcher.read_text(encoding="utf-8")

    assert 'ScriptPath = "ckg_rl_usim_v3.py"' in text
    assert 'outputs\\ckg_rl_usim_v31' in text
    assert 'checkpoints\\ckg_rl_usim_v31' in text
    assert '"USIM_V3_ENGINE_REVISION" = "v3.1"' in text
    assert 'quota=6/6/6/2' in text
    assert 'UseCourseReward = $true' in text
    assert 'UseCourseSample = $true' in text
    assert 'UsePrereqAux = $true' in text


def test_v31_manifest_records_resolved_candidate_support(monkeypatch, tmp_path):
    monkeypatch.setenv("USIM_FB_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("USIM_V3_CANDIDATES", "20")
    monkeypatch.setenv("USIM_V3_ENGINE_REVISION", "v3.1")

    v3._write_v3_engine_manifest()

    payload = json.loads((tmp_path / "v3_engine_manifest.json").read_text(encoding="utf-8"))
    assert payload["engine_revision"] == "v3.1"
    assert payload["train_candidate_quotas"] == {
        "residual": 6,
        "positive": 6,
        "state": 6,
        "random": 2,
    }
    assert payload["course_policy_integration"] == "normalized_observable_course_fit_logit_bias"


def test_v31_static_runner_persists_v3_rollout_diagnostics():
    static_runner = REPO_ROOT / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    text = static_runner.read_text(encoding="utf-8")

    for key in (
        "V3EndRate",
        "V3ActiveSteps",
        "V3EmbeddingReward",
        "V3RecommendationReward",
        "V3CourseReward",
        "V3RolloutDeltaL2",
        "V3CourseLogitBiasAbs",
        "V3TrainResidualShare",
        "V3TrainPositiveShare",
        "V3TrainStateShare",
        "V3TrainRandomShare",
    ):
        assert f'"{key}"' in text
    assert " | V3[end_rate=" in text
