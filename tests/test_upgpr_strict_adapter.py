import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np

from paper_aaai27.scripts.upgpr_strict_adapter import (
    _base_config,
    build_strict_candidates,
    compute_item_macro_metrics,
    reconstruct_cold_item_embeddings,
    rank_target_with_path_priority,
    warm_only_negative_support,
)


class UPGPRStrictAdapterTests(unittest.TestCase):
    def test_formal_profile_preserves_official_capacity_and_batch_sizes(self):
        root = Path(__file__).resolve().parents[1]
        source_config = (
            root
            / "paper_aaai27"
            / "baseline_sources"
            / "UPGPR-courserec"
            / "config"
            / "UPGPR"
            / "mooc.json"
        )
        config = _base_config(
            source_config,
            Path("processed"),
            Path("tmp"),
            seed=2025,
            embedding_epochs=1,
            policy_epochs=1,
            device="cuda",
            profile="formal-throughput",
        )

        self.assertEqual(config["TRAIN_EMBEDS"]["batch_size"], 32)
        self.assertEqual(config["TRAIN_EMBEDS"]["embed_size"], 100)
        self.assertEqual(config["TRAIN_AGENT"]["batch_size"], 32)
        self.assertEqual(config["TRAIN_AGENT"]["max_acts"], 250)
        self.assertEqual(config["TRAIN_AGENT"]["hidden"], [512, 256])

    def test_policy_step_budget_only_stops_at_configured_limit(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "paper_aaai27"
            / "baseline_sources"
            / "UPGPR-courserec"
            / "src"
            / "UPGPR"
        )
        sys.path.insert(0, str(source))
        try:
            from train_agent import policy_step_budget_reached

            self.assertFalse(policy_step_budget_reached(step=999, max_train_steps=1000))
            self.assertTrue(policy_step_budget_reached(step=1000, max_train_steps=1000))
            self.assertFalse(policy_step_budget_reached(step=1000, max_train_steps=-1))
        finally:
            sys.path.remove(str(source))

    def test_official_loader_excludes_cold_items_from_cf_negative_support(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "paper_aaai27"
            / "baseline_sources"
            / "UPGPR-courserec"
            / "src"
            / "UPGPR"
        )
        sys.path.insert(0, str(source))
        try:
            from data_utils import Dataset
            from easydict import EasyDict as edict

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "users.txt").write_text("u0\nu1\n", encoding="utf-8")
                (root / "courses.txt").write_text("i0\ni1\ni2\n", encoding="utf-8")
                (root / "concepts.txt").write_text("c0\nc1\n", encoding="utf-8")
                (root / "course_concepts.txt").write_text("0\n1\n0\n", encoding="utf-8")
                (root / "train.txt").write_text("0 0\n1 2\n", encoding="utf-8")
                args = edict(
                    entity_files={
                        "user": "users.txt",
                        "item": "courses.txt",
                        "concept": "concepts.txt",
                    },
                    item_relation={
                        "item_concept": ["course_concepts.txt", "concept"],
                    },
                    user_relation={},
                    entity_relation={},
                    entities=["user", "item", "concept"],
                )
                dataset = Dataset(str(root), args)

            np.testing.assert_array_equal(
                dataset.interactions.item_uniform_distrib,
                np.array([1.0, 0.0, 1.0]),
            )
            np.testing.assert_array_equal(
                dataset.item_concept.et_distrib,
                np.array([2.0, 0.0]),
            )
        finally:
            sys.path.remove(str(source))

    def test_cf_negative_support_contains_only_training_items(self):
        support = warm_only_negative_support([(0, 0), (0, 2), (1, 2)], n_items=5)

        np.testing.assert_array_equal(support, np.array([1.0, 0.0, 1.0, 0.0, 0.0]))

    def test_cold_embedding_is_mean_tail_minus_relation(self):
        item_embeddings = np.zeros((3, 2), dtype=np.float32)
        relation_embeddings = {
            "item_concept": np.array([1.0, 2.0], dtype=np.float32),
            "item_teacher": np.array([-1.0, 1.0], dtype=np.float32),
        }
        tail_embeddings = {
            "concept": np.array([[5.0, 8.0], [9.0, 10.0]], dtype=np.float32),
            "teacher": np.array([[3.0, 7.0]], dtype=np.float32),
        }
        item_relation_tails = {
            "item_concept": {2: [0, 1]},
            "item_teacher": {2: [0]},
        }
        relation_tail_types = {
            "item_concept": "concept",
            "item_teacher": "teacher",
        }

        reconstructed, audit = reconstruct_cold_item_embeddings(
            item_embeddings,
            relation_embeddings,
            tail_embeddings,
            item_relation_tails,
            relation_tail_types,
            cold_item_ids={2},
        )

        # mean([5,8]-[1,2], [9,10]-[1,2], [3,7]-[-1,1])
        np.testing.assert_allclose(reconstructed[2], np.array([16.0 / 3.0, 20.0 / 3.0]))
        self.assertEqual(audit[2]["static_edge_count"], 3)
        self.assertEqual(audit[2]["relations_used"], ["item_concept", "item_teacher"])

    def test_cold_reconstruction_ignores_tails_without_warm_anchor(self):
        reconstructed, audit = reconstruct_cold_item_embeddings(
            np.zeros((2, 2), dtype=np.float32),
            {"item_concept": np.array([1.0, 1.0], dtype=np.float32)},
            {"concept": np.array([[3.0, 5.0], [20.0, 30.0]], dtype=np.float32)},
            {"item_concept": {0: [0], 1: [0, 1]}},
            {"item_concept": "concept"},
            cold_item_ids={1},
            warm_item_ids={0},
        )

        np.testing.assert_allclose(reconstructed[1], np.array([2.0, 4.0]))
        self.assertEqual(audit[1]["static_edge_count"], 1)
        self.assertEqual(audit[1]["discarded_unanchored_edges"], 1)

    def test_candidates_are_all_warm_plus_current_target_with_history_mask(self):
        candidates = build_strict_candidates(
            warm_item_ids={0, 1, 2, 3},
            cold_target=5,
            train_history={1, 3},
        )

        np.testing.assert_array_equal(candidates, np.array([0, 2, 5]))
        self.assertNotIn(4, candidates)  # A different cold item.

    def test_path_reachability_precedes_transe_score_in_total_ranking(self):
        item_embeddings = np.array([[5.0], [4.0], [1.0]], dtype=np.float32)
        rank = rank_target_with_path_priority(
            user_vector=np.array([1.0], dtype=np.float32),
            item_embeddings=item_embeddings,
            candidate_ids=np.array([0, 1, 2]),
            target=2,
            endpoint_probabilities={2: 0.2},
        )

        self.assertEqual(rank, 1)

    def test_item_macro_metrics_average_per_target(self):
        # Target 5: hits at ranks 1 and 3. Target 6: misses twice.
        metrics = compute_item_macro_metrics(
            [(5, 1), (5, 3), (6, None), (6, None)],
            ks=(1, 3),
        )

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["R@1"], 0.25)
        self.assertAlmostEqual(metrics["R@3"], 0.5)
        self.assertAlmostEqual(metrics["N@1"], 0.25)
        expected_target5_ndcg3 = (1.0 + 1.0 / np.log2(4.0)) / 2.0
        self.assertAlmostEqual(metrics["N@3"], expected_target5_ndcg3 / 2.0)


if __name__ == "__main__":
    unittest.main()
