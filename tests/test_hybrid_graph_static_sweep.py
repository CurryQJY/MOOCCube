import numpy as np
import pytest
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_graph_static_sweep import (
    blend_scores,
    rank_calibrated_hot_scores,
    select_guarded_lambda,
)


def test_blend_scores_changes_hot_columns_only():
    static = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    graph = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
    hot_mask = np.array([False, True, True])

    actual = blend_scores(static, graph, hot_mask, 0.5)

    np.testing.assert_allclose(actual, [[1.0, 3.0, 4.0]])
    np.testing.assert_allclose(actual[:, ~hot_mask], static[:, ~hot_mask])


def test_select_guarded_lambda_maximizes_hot_n10_without_cold_drop():
    rows = [
        {"lambda": 0.0, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.10},
        {"lambda": 0.5, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.13},
        {"lambda": 1.0, "cold_R@10": 0.29, "cold_N@10": 0.21, "hot_N@10": 0.15},
    ]

    selected = select_guarded_lambda(
        rows,
        baseline_cold_r10=0.30,
        baseline_cold_n10=0.20,
    )

    assert selected["lambda"] == pytest.approx(0.5)


def test_select_guarded_lambda_does_not_prefer_largest_lambda_over_hot_metric():
    rows = [
        {"lambda": 0.0, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.10},
        {"lambda": 0.6, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.15},
        {"lambda": 1.0, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.13},
    ]

    selected = select_guarded_lambda(
        rows,
        baseline_cold_r10=0.30,
        baseline_cold_n10=0.20,
    )

    assert selected["lambda"] == pytest.approx(0.6)


def test_select_guarded_lambda_falls_back_to_static_when_all_candidates_drop():
    rows = [
        {"lambda": 0.0, "cold_R@10": 0.30, "cold_N@10": 0.20, "hot_N@10": 0.10},
        {"lambda": 0.5, "cold_R@10": 0.29, "cold_N@10": 0.19, "hot_N@10": 0.13},
    ]

    selected = select_guarded_lambda(
        rows,
        baseline_cold_r10=0.30,
        baseline_cold_n10=0.20,
    )

    assert selected["lambda"] == pytest.approx(0.0)


def test_rank_calibrated_hot_scores_preserve_cold_and_hot_score_distribution():
    static = torch.tensor([[0.9, 0.8, 0.4, 0.1]])
    graph = torch.tensor([[0.2, 0.1, 0.9, 0.8]])
    hot_mask = torch.tensor([False, True, True, True])

    actual = rank_calibrated_hot_scores(static, graph, hot_mask, 1.0)

    assert actual[0, 0].item() == pytest.approx(static[0, 0].item())
    torch.testing.assert_close(
        torch.sort(actual[:, hot_mask], dim=1).values,
        torch.sort(static[:, hot_mask], dim=1).values,
    )
    assert torch.argmax(actual[0, 1:]).item() == 1
