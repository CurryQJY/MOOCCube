import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_representation_probe import (
    bootstrap_mean_ci,
    l2_normalize,
    merge_per_item_pair,
    neighbor_label_purity,
    summarize_signed_delta,
)


class RepresentationProbeTests(unittest.TestCase):
    def test_l2_normalize_leaves_zero_rows_zero(self):
        x = np.array([[3.0, 4.0], [0.0, 0.0]])

        out = l2_normalize(x)

        self.assertTrue(np.allclose(out[0], [0.6, 0.8]))
        self.assertTrue(np.allclose(out[1], [0.0, 0.0]))

    def test_neighbor_label_purity_excludes_self_and_uses_shared_labels(self):
        emb = l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            )
        )
        labels = [{1}, {1, 2}, {3}]

        scores = neighbor_label_purity(emb, labels, query_ids=[0], candidate_ids=[0, 1, 2], k=1)

        self.assertEqual(scores["mean_purity"], 1.0)
        self.assertEqual(scores["n_queries"], 1)
        self.assertEqual(scores["n_empty_label_queries"], 0)

    def test_neighbor_label_purity_reports_empty_label_queries(self):
        emb = l2_normalize(np.eye(3))
        labels = [set(), {1}, {1}]

        scores = neighbor_label_purity(emb, labels, query_ids=[0, 1], candidate_ids=[0, 1, 2], k=1)

        self.assertEqual(scores["n_queries"], 2)
        self.assertEqual(scores["n_empty_label_queries"], 1)

    def test_merge_per_item_pair_matches_on_seed_and_item(self):
        ours = pd.DataFrame(
            {
                "seed": [2025, 2025, 2026],
                "item_id": [1, 2, 1],
                "R@10": [0.4, 0.0, 0.5],
                "N@10": [0.3, 0.0, 0.4],
            }
        )
        baseline = pd.DataFrame(
            {
                "seed": [2025, 2025, 2026],
                "item_id": [1, 2, 1],
                "R@10": [0.1, 0.2, 0.5],
                "N@10": [0.2, 0.1, 0.3],
            }
        )

        merged = merge_per_item_pair(ours, baseline, ours_name="CKG-RL", baseline_name="CGRC")

        self.assertEqual(list(merged["delta_R@10"]), [0.3, -0.2, 0.0])
        self.assertEqual(list(merged["delta_N@10"]), [0.1, -0.1, 0.1])
        self.assertEqual(set(merged["ours"]), {"CKG-RL"})
        self.assertEqual(set(merged["baseline"]), {"CGRC"})

    def test_summarize_signed_delta_counts_win_tie_loss(self):
        values = np.array([0.2, 0.0, -0.1, 0.0, 0.3])

        summary = summarize_signed_delta(values)

        self.assertEqual(summary["n"], 5)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["ties"], 2)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["win_ratio"], 0.4)
        self.assertAlmostEqual(summary["tie_ratio"], 0.4)
        self.assertAlmostEqual(summary["loss_ratio"], 0.2)

    def test_bootstrap_mean_ci_is_deterministic_and_contains_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])

        first = bootstrap_mean_ci(values, n_boot=200, seed=7)
        second = bootstrap_mean_ci(values, n_boot=200, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first["ci_low"], first["mean"])
        self.assertGreaterEqual(first["ci_high"], first["mean"])
        self.assertAlmostEqual(first["mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
