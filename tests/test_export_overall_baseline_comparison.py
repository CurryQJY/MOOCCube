from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper_aaai27/scripts/export_overall_baseline_comparison.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_overall_baseline_comparison", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_weighted_overall_uses_course_counts() -> None:
    value = MODULE.weighted_overall(0.2, 2, 0.8, 8)

    assert value == pytest.approx(0.68)
    assert value != pytest.approx(0.5)


def test_validate_direct_overall_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="direct overall mismatch"):
        MODULE.validate_direct_overall(0.2, 0.25, tolerance=1e-8)


def test_unavailable_row_contains_no_numeric_metrics() -> None:
    row = MODULE.unavailable_row(
        "MOOCCube", "PCGNN", 2025, "missing warm targets"
    )

    assert row["status"] == "unavailable_missing_warm_targets"
    assert all(pd.isna(row[metric]) for metric in MODULE.METRICS)


def test_build_seed_row_reconstructs_all_metrics() -> None:
    row = MODULE.build_seed_row(
        dataset="Toy",
        method="Baseline",
        seed=2025,
        cold={metric: 0.2 for metric in MODULE.METRICS},
        hot={metric: 0.8 for metric in MODULE.METRICS},
        cold_count=2,
        hot_count=8,
        cold_source="cold.csv",
        hot_source="hot.csv",
        direct={metric: 0.68 for metric in MODULE.METRICS},
    )

    assert row["status"] == "ready"
    assert row["aggregation_route"] == "cold_hot_course_count_weighted"
    assert row["R@10"] == pytest.approx(0.68)
    assert row["cold_count"] == 2
    assert row["hot_count"] == 8


def test_build_seed_row_rejects_missing_metric() -> None:
    cold = {metric: 0.2 for metric in MODULE.METRICS if metric != "N@20"}

    with pytest.raises(ValueError, match="missing metric N@20"):
        MODULE.build_seed_row(
            dataset="Toy",
            method="Baseline",
            seed=2025,
            cold=cold,
            hot={metric: 0.8 for metric in MODULE.METRICS},
            cold_count=2,
            hot_count=8,
            cold_source="cold.csv",
            hot_source="hot.csv",
            direct=None,
        )


def test_collect_seed_rows_covers_current_main_table_artifacts() -> None:
    detail, coverage = MODULE.collect_seed_rows()

    assert len(coverage) == 108
    assert len(detail) == 99
    assert set(coverage.loc[coverage.status != "ready", "method"]) == {"PCGNN"}
    assert (coverage.status == "unavailable_missing_warm_targets").sum() == 9
    assert set(detail.method) == set(MODULE.METHOD_ORDER) - {"PCGNN"}
    assert set(detail.seed) == {2025, 2026, 2027}
    assert set(detail.dataset) == {"MOOCCube", "Junyi", "COCO"}
    assert (
        detail.loc[detail.method == "KGRec", "aggregation_route"]
        == "direct_all_item_macro_validated"
    ).all()
