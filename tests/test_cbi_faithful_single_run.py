from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_cbi_faithful_seed2025.ps1"


def test_launcher_is_isolated_and_uses_cbi_faithful_configuration():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'outputs\\cbi_faithful_single_seed2025' in text
    assert 'checkpoints\\cbi_faithful_single_seed2025' in text
    assert 'background_logs\\cbi_faithful_single_seed2025' in text
    assert 'ContentDeltaPaperStyle = $true' in text
    assert 'ContentDeltaReplaceItem = $true' in text
    assert 'ContentDeltaColdOnly = $false' in text
    assert 'ContentDeltaMaxNorm = 0.5' in text
    assert 'ContentDeltaScale = 1.0' in text
    assert 'ContentDeltaLrMult = 1.0' in text
    assert 'ContentDeltaL2W = 0.0' in text
    assert 'ContentDeltaCapW = 0.0' in text
    assert 'ContentDeltaTrainOnIdDropout = $false' in text
    assert 'Seeds = @(2025)' in text
    assert 'Epochs = 60' in text
    assert 'Patience = 60' in text


def test_launcher_protects_main_table_files_and_output_roots():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'course_maincfg_runs\\maincfg' in text
    assert 'course_ablation_e60_3seed\\full' in text
    assert 'protected_files_before' in text
    assert 'protected_files_after' in text
    assert 'run_manifest.json' in text
    assert 'Set-Content "paper_aaai27' not in text
    assert 'Set-Content "usim_feedback_fast3_content_delta.py' not in text


def test_launcher_supports_dry_run_without_training():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '[switch]$DryRun' in text
    assert 'if ($DryRun)' in text
    assert 'DRY_RUN' in text
