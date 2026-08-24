import pandas as pd
import pytest

import validation_inference_policy_report as report


def test_rank_policies_uses_n10_then_r10_only():
    rows = pd.DataFrame(
        [
            {"policy": "ppo", "seed": 2025, "N@10": 0.22, "R@10": 0.30},
            {"policy": "ppo", "seed": 2026, "N@10": 0.20, "R@10": 0.28},
            {"policy": "greedy_similarity", "seed": 2025, "N@10": 0.21, "R@10": 0.40},
            {"policy": "greedy_similarity", "seed": 2026, "N@10": 0.20, "R@10": 0.41},
        ]
    )

    ranking = report.rank_policies(rows)

    assert ranking.iloc[0]["policy"] == "ppo"
    assert ranking.iloc[0]["mean_N@10"] == pytest.approx(0.21)
