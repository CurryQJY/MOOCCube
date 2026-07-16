import csv
from pathlib import Path

import summarize_cbi_faithful_seed2025 as summary_module
from summarize_cbi_faithful_seed2025 import (
    build_comparison,
    parse_delta_stats,
    read_report,
    select_validation_epoch,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_cbi_faithful_seed2025.ps1"


def _write_report(path, r10, n10):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "full_cold",
                "full_hot",
                "full_cold_item_macro",
                "full_hot_item_macro",
            ],
        )
        writer.writeheader()
        values = {
            "R@5": (r10 - 0.04, 0.18),
            "R@10": (r10, 0.20),
            "R@20": (r10 + 0.04, 0.24),
            "N@5": (n10 - 0.02, 0.08),
            "N@10": (n10, 0.10),
            "N@20": (n10 + 0.02, 0.12),
        }
        for metric, (cold, hot) in values.items():
            writer.writerow(
                {
                    "metric": metric,
                    "full_cold_item_macro": cold,
                    "full_hot_item_macro": hot,
                }
            )


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


def test_summary_uses_item_macro_metrics_and_screening_rule(tmp_path):
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write_report(candidate, 0.2540, 0.1870)
    _write_report(baseline, 0.2530, 0.1830)

    result = build_comparison(read_report(candidate), read_report(baseline))

    assert result["metrics"]["N@10"]["cold_delta"] == 0.004
    assert result["metrics"]["R@10"]["cold_delta"] == 0.001
    assert result["screening"]["promising"] is True


def test_validation_epoch_is_selected_by_cold_ndcg10(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "Epoch,Val_full_cold_R@10,Val_full_hot_R@10,Val_full_cold_N@10,Val_full_hot_N@10\n"
        "1,0.20,0.10,0.15,0.08\n"
        "2,0.21,0.11,0.18,0.09\n",
        encoding="utf-8",
    )

    assert select_validation_epoch(history)["epoch"] == 2


def test_delta_stats_parser_reads_last_epoch_diagnostics(tmp_path):
    log = tmp_path / "training.log"
    log.write_text(
        "DeltaNorm[mean=0.1000, max=0.5000, eff_mean=0.1000, eff_max=0.5000, clip=12.50%]\n"
        "DeltaNorm[mean=0.2000, max=0.5000, eff_mean=0.2000, eff_max=0.5000, clip=25.00%]\n",
        encoding="utf-8",
    )

    assert parse_delta_stats(log) == {
        "mean_norm": 0.2,
        "max_norm": 0.5,
        "effective_mean_norm": 0.2,
        "effective_max_norm": 0.5,
        "clipped_ratio": 0.25,
    }


def test_screening_rejects_ndcg_gain_when_recall_drops_too_much(tmp_path):
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write_report(candidate, 0.2500, 0.1900)
    _write_report(baseline, 0.2530, 0.1830)

    result = build_comparison(read_report(candidate), read_report(baseline))

    assert result["screening"]["promising"] is False


def test_summarize_writes_reproducible_json_csv_and_markdown(tmp_path):
    candidate_dir = tmp_path / "candidate"
    baseline_dir = tmp_path / "baseline"
    output_root = tmp_path / "output"
    candidate_dir.mkdir()
    baseline_dir.mkdir()
    _write_report(candidate_dir / summary_module.REPORT_NAME, 0.2540, 0.1870)
    _write_report(baseline_dir / summary_module.REPORT_NAME, 0.2530, 0.1830)
    (candidate_dir / summary_module.HISTORY_NAME).write_text(
        "Epoch,Val_full_cold_R@10,Val_full_hot_R@10,Val_full_cold_N@10,Val_full_hot_N@10\n"
        "1,0.20,0.10,0.15,0.08\n"
        "2,0.21,0.11,0.18,0.09\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "DeltaNorm[mean=0.2000, max=0.5000, eff_mean=0.2000, eff_max=0.5000, clip=25.00%]\n",
        encoding="utf-8",
    )

    payload = summary_module.summarize(
        candidate_dir=candidate_dir,
        baseline_dir=baseline_dir,
        log_path=log_path,
        output_root=output_root,
    )

    assert payload["scope"]["status"] == "one_seed_screening_only"
    assert (output_root / "cbi_comparison.json").exists()
    assert (output_root / "cbi_comparison.csv").exists()
    markdown = (output_root / "cbi_comparison.md").read_text(encoding="utf-8")
    assert "one-seed exploratory screening result" in markdown
    assert "cannot change the AAAI main table" in markdown
