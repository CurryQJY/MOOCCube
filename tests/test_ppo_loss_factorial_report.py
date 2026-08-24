import json

import pandas as pd
import pytest

import ppo_loss_factorial_report as report


def test_summarize_factorial_computes_locked_effects():
    rows = pd.DataFrame(
        [
            {"seed": 2025, "metric": "N@10", "cell": "on_static", "value": 0.20},
            {"seed": 2025, "metric": "N@10", "cell": "on_course_fit", "value": 0.25},
            {"seed": 2025, "metric": "N@10", "cell": "off_static", "value": 0.18},
            {"seed": 2025, "metric": "N@10", "cell": "off_course_fit", "value": 0.22},
            {"seed": 2026, "metric": "N@10", "cell": "on_static", "value": 0.30},
            {"seed": 2026, "metric": "N@10", "cell": "on_course_fit", "value": 0.35},
            {"seed": 2026, "metric": "N@10", "cell": "off_static", "value": 0.28},
            {"seed": 2026, "metric": "N@10", "cell": "off_course_fit", "value": 0.32},
        ]
    )

    by_seed, summary = report.summarize_factorial(rows)

    assert by_seed["training_effect_static"].tolist() == pytest.approx([0.02, 0.02])
    assert by_seed["training_effect_course_fit"].tolist() == pytest.approx([0.03, 0.03])
    assert by_seed["inference_effect_ppo_on"].tolist() == pytest.approx([0.05, 0.05])
    assert by_seed["inference_effect_ppo_off"].tolist() == pytest.approx([0.04, 0.04])
    assert by_seed["interaction"].tolist() == pytest.approx([0.01, 0.01])
    row = summary.iloc[0]
    assert row["on_static_mean"] == pytest.approx(0.25)
    assert row["off_course_fit_mean"] == pytest.approx(0.27)
    assert row["interaction_mean"] == pytest.approx(0.01)


def test_load_factorial_rejects_missing_seed_cell(tmp_path):
    roots = {cell: tmp_path / cell for cell in report.CELLS}
    for cell, root in roots.items():
        if cell == "off_course_fit":
            continue
        directory = root / "strict_item_cold_balanced_thr1_seed_2025"
        directory.mkdir(parents=True)
        pd.DataFrame(
            [{column: 0.2 for column in report.METRIC_COLUMNS.values()}]
        ).to_csv(directory / report.FINAL_CSV, index=False)

    with pytest.raises(FileNotFoundError, match="off_course_fit.*2025"):
        report.load_factorial(roots, seeds=[2025])


def test_validate_off_course_fit_audit_rejects_actor_call(tmp_path):
    path = tmp_path / "actor_inference_audit.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_target": "test",
                "mode": "course_fit",
                "actor_calls": 1,
                "history_all_train_only": True,
                "target_seen_candidate_pairs": 0,
                "refined_item_composition": {"total_unique": 102, "train_present": 0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="actor_calls"):
        report.validate_off_course_fit_audit(path)
