import unittest

import pandas as pd

from data_process_xes3g5m import (
    build_item_metadata,
    expand_sequence_frame,
    route_hierarchy_edges,
    route_node_id,
)


class XES3G5MProcessorTest(unittest.TestCase):
    def test_expand_sequence_frame_skips_masked_positions_and_splits_concepts(self):
        frame = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "questions": "q1,q2,q3",
                    "concepts": "c1_c2,c3,c4",
                    "responses": "1,0,1",
                    "timestamps": "1000,2000,3000",
                    "selectmasks": "1,-1,1",
                }
            ]
        )

        interactions, item_concepts, stats = expand_sequence_frame(frame, source="train")

        self.assertEqual(list(interactions["course_id"]), ["q1", "q3"])
        self.assertEqual(list(interactions["timestamp"]), [1, 3])
        self.assertEqual(item_concepts["q1"], {"KC_c1", "KC_c2"})
        self.assertEqual(item_concepts["q3"], {"KC_c4"})
        self.assertEqual(stats["masked_positions"], 1)

    def test_build_item_metadata_adds_route_concepts_and_hierarchy_prereqs(self):
        questions = {
            "q1": {
                "content": "题干",
                "analysis": "解析",
                "kc_routes": ["数学----计数----乘法原理"],
            }
        }
        item_concepts = {"q1": {"KC_c1"}}

        metadata, prereq_edges = build_item_metadata(questions, item_concepts)

        self.assertIn("KC_c1", metadata["q1"].concepts)
        self.assertIn(route_node_id(["数学", "计数", "乘法原理"]), metadata["q1"].concepts)
        self.assertIn(
            (route_node_id(["数学", "计数"]), route_node_id(["数学", "计数", "乘法原理"])),
            prereq_edges,
        )

    def test_route_hierarchy_edges_uses_prefix_order(self):
        edges = route_hierarchy_edges("数学----计数----乘法原理")

        self.assertEqual(
            edges,
            [
                (route_node_id(["数学"]), route_node_id(["数学", "计数"])),
                (route_node_id(["数学", "计数"]), route_node_id(["数学", "计数", "乘法原理"])),
            ],
        )


if __name__ == "__main__":
    unittest.main()
