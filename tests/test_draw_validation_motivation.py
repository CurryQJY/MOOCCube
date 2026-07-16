from pathlib import Path

import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_validation_motivation import (
    summarize_validation_course_rows,
)
from paper_aaai27.scripts.draw_validation_motivation import (
    STRUCTURAL_METRICS,
    draw_validation_motivation,
    validate_validation_figure_inputs,
)


def _course_rows() -> pd.DataFrame:
    rows = []
    for model, offset in (("pcgnn", 0.0), ("cgrc", 0.08)):
        for index, seed in enumerate((2025, 2026, 2027)):
            ndcg = 0.03 + offset + 0.02 * index
            rows.append(
                {
                    "analysis_split": "validation",
                    "model": model,
                    "seed": seed,
                    "target_item_id": 10 + index,
                    "list_count": 10,
                    "cold_list_count": 8 - index,
                    "ndcg_at_10": ndcg,
                    "low_ndcg_at_10": float(ndcg <= 0.10),
                    "cold_proportion": 0.30 + offset,
                    "effective_coverage": (8 - index) / 10,
                    "missingness": 1.0 - (8 - index) / 10,
                    "cold_prerequisite_gap": 0.55 - offset,
                    "cold_concept_continuity": 0.15 + offset,
                    "cold_difficulty_gap": 0.10 - offset / 2,
                    "cold_structural_redundancy": 0.06 + offset / 2,
                }
            )
    return pd.DataFrame(rows)


def _summary(course_rows: pd.DataFrame) -> pd.DataFrame:
    return summarize_validation_course_rows(
        course_rows,
        n_bootstrap=100,
        random_seed=2027,
    )


def test_validation_figure_inputs_reject_test_rows_and_ckg_rl():
    course_rows = _course_rows()
    summary = _summary(course_rows)
    validate_validation_figure_inputs(course_rows, summary)

    test_rows = course_rows.copy()
    test_rows.loc[0, "analysis_split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        validate_validation_figure_inputs(test_rows, summary)

    ckg_rows = course_rows.copy()
    ckg_rows.loc[0, "model"] = "ckg_rl"
    with pytest.raises(ValueError, match="PCGNN and CGRC"):
        validate_validation_figure_inputs(ckg_rows, summary)


def test_validation_figure_requires_four_conditional_proxy_intervals():
    course_rows = _course_rows()
    summary = _summary(course_rows)
    incomplete = summary.loc[
        ~(
            summary["model"].eq("pcgnn")
            & summary["metric"].eq("cold_prerequisite_gap")
        )
    ]

    with pytest.raises(ValueError, match="structural summary"):
        validate_validation_figure_inputs(course_rows, incomplete)

    assert tuple(STRUCTURAL_METRICS) == (
        "cold_prerequisite_gap",
        "cold_concept_continuity",
        "cold_difficulty_gap",
        "cold_structural_redundancy",
    )


def test_draw_validation_motivation_exports_bounded_two_panel_figure(tmp_path):
    course_rows = _course_rows()
    summary = _summary(course_rows)
    outputs = draw_validation_motivation(
        course_rows,
        summary,
        output_base=tmp_path / "mooccube_validation_motivation",
    )

    assert {Path(path).suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs)

    svg = (tmp_path / "mooccube_validation_motivation.svg").read_text(encoding="utf-8")
    assert "Validation cold-course exposure" in svg
    assert "conditional on a cold course being recommended" in svg
    assert "Coverage / missingness" in svg
    assert "PCGNN" in svg and "CGRC" in svg
    assert "CKG-RL response" not in svg
    assert "improvement" not in svg.lower()
    assert "significant" not in svg.lower()
    assert "p=" not in svg.lower()
    assert "p &lt;" not in svg.lower()
