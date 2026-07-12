from __future__ import annotations

import numpy as np

from paper_aaai27.scripts.run_kgrec_strict_seed import evaluate_item_macro_from_scores
from paper_aaai27.scripts.run_kgrec_strict_seed import sample_warm_negatives


def test_sample_warm_negatives_never_uses_cold_or_seen_items() -> None:
    train_pairs = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    user_seen = {0: {0, 1}, 1: {1}}
    warm_items = np.array([0, 1, 3], dtype=np.int64)

    triples = sample_warm_negatives(train_pairs, user_seen, warm_items, seed=7)

    assert triples.shape == (3, 3)
    assert set(triples[:, 2]).issubset({0, 1, 3})
    assert all(int(neg) not in user_seen[int(user)] for user, _pos, neg in triples)
    assert 2 not in set(triples[:, 2])


def test_evaluate_item_macro_masks_train_history_and_macro_averages_by_item() -> None:
    scores = np.array(
        [
            [0.9, 0.8, 0.1],
            [0.7, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    eval_pairs = [(0, 1), (1, 2)]
    train_seen = {0: {0}, 1: set()}

    metrics = evaluate_item_macro_from_scores(
        scores=scores,
        eval_pairs=eval_pairs,
        train_seen_by_user=train_seen,
        cold_item_ids={1, 2},
        k_list=(1, 2),
    )

    assert metrics["full_cold_item_macro"]["R@1"] == 0.5
    assert metrics["full_cold_item_macro"]["R@2"] == 1.0
    assert metrics["full_cold_item_macro"]["N@1"] == 0.5
    assert metrics["counts"]["cold_items"] == 2
