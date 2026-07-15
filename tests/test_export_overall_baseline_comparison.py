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
