from pathlib import Path

import numpy as np
import pandas as pd

from analyze_usim_official_significance import (
    holm_adjust,
    load_per_item_pair,
    paired_ttest_pvalue,
    sig_marker,
)


def test_holm_adjust_is_monotone_in_original_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]


def test_load_per_item_pair_matches_seed_items_and_diffs(tmp_path):
    ours = tmp_path / "ours.csv"
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        {
            "item_id": [2, 1],
            "count": [3, 4],
            "R@10": [0.4, 0.8],
            "N@10": [0.2, 0.6],
        }
    ).to_csv(ours, index=False)
    pd.DataFrame(
        {
            "item_id": [1, 2],
            "count": [4, 3],
            "R@10": [0.5, 0.1],
            "N@10": [0.4, 0.1],
        }
    ).to_csv(baseline, index=False)

    paired = load_per_item_pair("2025", "cold", ours, baseline, ["R@10", "N@10"])

    assert paired["item_id"].tolist() == [1, 2]
    assert paired["R@10_diff"].round(6).tolist() == [0.3, 0.3]
    assert paired["N@10_diff"].round(6).tolist() == [0.2, 0.1]


def test_paired_ttest_pvalue_detects_positive_shift():
    p_value = paired_ttest_pvalue(np.array([0.5, 0.62, 0.75]), np.array([0.1, 0.2, 0.3]))

    assert p_value < 0.05
    assert sig_marker(p_value) in {"*", "**", "***"}
