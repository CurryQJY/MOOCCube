import pandas as pd
import pytest

import actor_inference_ab_report as report


def test_compare_per_item_joins_rows_and_computes_deltas(tmp_path):
    static = tmp_path / "static.csv"
    actor = tmp_path / "actor.csv"
    pd.DataFrame(
        [
            {"item_id": 1, "count": 2, "R@10": 0.25, "N@10": 0.10},
            {"item_id": 2, "count": 3, "R@10": 0.50, "N@10": 0.20},
        ]
    ).to_csv(static, index=False)
    pd.DataFrame(
        [
            {"item_id": 1, "count": 2, "R@10": 0.50, "N@10": 0.15},
            {"item_id": 2, "count": 3, "R@10": 0.25, "N@10": 0.30},
        ]
    ).to_csv(actor, index=False)

    detail, summary = report.compare_per_item(static, actor, seed=2025)

    assert list(detail["item_id"]) == [1, 2]
    assert detail.loc[0, "delta_R@10"] == pytest.approx(0.25)
    assert summary["delta_N@10"] == pytest.approx(0.075)
    assert summary["positive_N@10"] == pytest.approx(1.0)


def test_compare_fullrank_uses_item_macro_columns(tmp_path):
    static = tmp_path / "static_full.csv"
    actor = tmp_path / "actor_full.csv"
    base = {
        "full_cold_item_macro_r5": 0.20,
        "full_cold_item_macro_r10": 0.30,
        "full_cold_item_macro_r20": 0.40,
        "full_cold_item_macro_n5": 0.10,
        "full_cold_item_macro_n10": 0.15,
        "full_cold_item_macro_n20": 0.18,
    }
    changed = dict(base)
    changed["full_cold_item_macro_n10"] = 0.17
    pd.DataFrame([base]).to_csv(static, index=False)
    pd.DataFrame([changed]).to_csv(actor, index=False)

    rows = report.compare_fullrank(static, actor, seed=2025)

    n10 = next(row for row in rows if row["metric"] == "N@10")
    assert n10["static"] == pytest.approx(0.15)
    assert n10["actor"] == pytest.approx(0.17)
    assert n10["delta"] == pytest.approx(0.02)
