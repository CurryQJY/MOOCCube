from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_missing_runtime_queue_is_resumable_and_targets_only_missing_runs():
    source = (ROOT / "run_cgrc_missing_runtime_queue.ps1").read_text(
        encoding="utf-8"
    )

    assert "[switch]$DryRun" in source
    assert 'Dataset = "MOOCCube"; Seed = 2025' in source
    assert 'Dataset = "Junyi"; Seed = 2025' in source
    assert 'Dataset = "Junyi"; Seed = 2026' in source
    assert 'Dataset = "Junyi"; Seed = 2027' in source
    junyi_2025 = source.index('Dataset = "Junyi"; Seed = 2025')
    junyi_2026 = source.index('Dataset = "Junyi"; Seed = 2026')
    junyi_2027 = source.index('Dataset = "Junyi"; Seed = 2027')
    mooccube_2025 = source.index('Dataset = "MOOCCube"; Seed = 2025')
    assert junyi_2025 < junyi_2026 < junyi_2027 < mooccube_2025
    assert "runtime_cgrc_profile" in source
    assert "cgrc_runtime_profile" in source
    assert "checkpoints\\mooccubex\\runtime_cgrc_profile" in source
    assert "checkpoints\\junyi\\cgrc_runtime_profile" in source
    assert 'CGRC_PAPER_AUTO_RESUME = "1"' in source
    assert 'CGRC_PAPER_FORCE_FRESH = "0"' in source
    assert 'CGRC_PAPER_SAVE_CKPT = "1"' in source
    assert "cgrc_paper_static_result.json" in source
    assert "SKIP completed" in source


def test_missing_runtime_queue_checks_native_failures_and_refreshes_tables():
    source = (ROOT / "run_cgrc_missing_runtime_queue.ps1").read_text(
        encoding="utf-8"
    )

    assert "cgrc_paper_static_hin.py" in source
    assert "if ($LASTEXITCODE -ne 0)" in source
    assert "paper_aaai27\\scripts\\build_revision_tables.py" in source
    assert "paper_aaai27\\scripts\\export_efficiency_table.py" in source
    assert "Tee-Object" not in source
    assert ">> `\"$logPath`\" 2>&1" in source
