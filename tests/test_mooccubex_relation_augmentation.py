from build_mooccubex_augmented_relations import (
    build_propagated_concept_pairs,
    build_semantic_cluster_pairs,
    dedupe_pairs_preserve_order,
)
import unittest

import numpy as np


class RelationAugmentationTests(unittest.TestCase):
    def test_build_semantic_cluster_pairs_covers_every_course_at_each_level(self):
        course_ids = ["C_1", "C_2", "C_3"]
        labels_by_level = {
            2: [0, 0, 1],
            3: [0, 1, 2],
        }

        pairs = build_semantic_cluster_pairs(course_ids, labels_by_level)

        self.assertIn(("C_1", "SEM_CLUSTER_L2_00000"), pairs)
        self.assertIn(("C_2", "SEM_CLUSTER_L2_00000"), pairs)
        self.assertIn(("C_3", "SEM_CLUSTER_L2_00001"), pairs)
        self.assertIn(("C_3", "SEM_CLUSTER_L3_00002"), pairs)
        self.assertEqual(len(pairs), len(course_ids) * len(labels_by_level))

    def test_dedupe_pairs_preserve_order_keeps_first_occurrence(self):
        pairs = [
            ("C_1", "K_A"),
            ("C_1", "K_A"),
            ("C_1", "SEM_CLUSTER_L2_00000"),
            ("C_2", "K_A"),
        ]

        self.assertEqual(
            dedupe_pairs_preserve_order(pairs),
            [
                ("C_1", "K_A"),
                ("C_1", "SEM_CLUSTER_L2_00000"),
                ("C_2", "K_A"),
            ],
        )

    def test_build_propagated_concept_pairs_uses_high_confidence_real_concepts(self):
        course_ids = ["src_a", "src_b", "target", "far"]
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.20],
                [0.99, 0.10],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        original_pairs = [
            ("src_a", "K_CORE"),
            ("src_a", "K_SINGLETON"),
            ("src_b", "K_CORE"),
            ("src_b", "K_ADVANCED_SINGLETON"),
            ("far", "K_FAR"),
        ]

        pairs, stats = build_propagated_concept_pairs(
            course_ids,
            embeddings,
            original_pairs,
            top_m=2,
            min_similarity=0.85,
            min_concept_support=0.10,
            max_concepts_per_course=4,
            min_concept_df=2,
            max_concept_course_frac=1.0,
            only_missing=True,
        )

        self.assertEqual(pairs, [("target", "K_CORE")])
        self.assertEqual(stats["source_courses"], 2)
        self.assertEqual(stats["target_courses"], 1)
        self.assertEqual(stats["propagated_courses"], 1)
        self.assertEqual(stats["propagated_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
