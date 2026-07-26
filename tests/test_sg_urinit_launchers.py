from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_runner_exposes_sg_urinit_environment_switches():
    script = (ROOT / "run_usim_feedback_fast3_content_delta_static.ps1").read_text(encoding="utf-8")

    assert "[bool]$UseSgUrinit" in script
    assert "[int]$SgUrinitClusterK" in script
    assert "[double]$SgUrinitLocalW" in script
    assert "[double]$SgUrinitGlobalW" in script
    assert '"USIM_USE_SG_URINIT"' in script
    assert '"USIM_SG_URINIT_CLUSTER_K"' in script
    assert '"USIM_SG_URINIT_LOCAL_W"' in script
    assert '"USIM_SG_URINIT_GLOBAL_W"' in script
    assert "$env:USIM_SG_URINIT_SEED = [string]$seed" in script


def test_sg_urinit_three_seed_launcher_uses_isolated_formal_output():
    launcher = ROOT / "run_sg_urinit_3seed_static.ps1"
    script = launcher.read_text(encoding="utf-8")

    assert "outputs\\content_delta_pop5\\sg_urinit_v1\\K32_lw0p7_gw0p3" in script
    assert "checkpoints\\content_delta_pop5\\sg_urinit_v1\\K32_lw0p7_gw0p3" in script
    assert "Seeds = @(2025, 2026, 2027)" in script
    assert "UseSgUrinit = $true" in script
    assert "SgUrinitClusterK = 32" in script
    assert "SgUrinitLocalW = 0.70" in script
    assert "SgUrinitGlobalW = 0.30" in script
    assert "Patience = 60" in script
    assert "EarlyStopAverageMode = \"item_macro\"" in script
    assert "EarlyStopScoreMode = \"cold_only\"" in script


def test_sg_urinit_main_table_launcher_matches_main_table_config_shape():
    launcher = ROOT / "run_sg_urinit_main_table_3seed_static.ps1"
    script = launcher.read_text(encoding="utf-8")

    assert "outputs\\significance_per_item_exports\\mooccube\\ckg_rl_full_sg_urinit_K32_lw0p7_gw0p3" in script
    assert "Reference config: outputs\\significance_per_item_exports\\mooccube\\ckg_rl_full" in script
    assert "Seeds = $Seeds" in script
    assert "Epochs = $Epochs" in script
    assert "Patience = $Patience" in script
    assert "EarlyStopAverageMode = \"item_macro\"" in script
    assert "EarlyStopScoreMode = \"cold_only\"" in script
    assert "UseContentDelta = $false" in script
    assert "UseSgUrinit = $true" in script
    assert "SgUrinitClusterK = 32" in script
    assert "SgUrinitLocalW = 0.70" in script
    assert "SgUrinitGlobalW = 0.30" in script
    assert "UseCourseFeedback = $true" in script
    assert "UseCourseReward = $true" in script
    assert "UseCourseSample = $true" in script
    assert "UsePrereqAux = $true" in script
    assert "CourseFeedbackOnlyCold = $false" in script
    assert "CourseSampleOnlyCold = $false" in script
    assert "PrereqAuxOnlyCold = $false" in script
    assert "MaskKnownPosNeg = $true" in script
    assert "MaskSameItemNeg = $true" in script
    assert "SaveCkpt = $true" in script
    assert "SaveOptState = $true" in script
