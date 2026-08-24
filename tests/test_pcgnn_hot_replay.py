import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.pcgnn_hot_replay import (  # noqa: E402
    combine_cold_hot_overall,
    resolve_config_file,
    select_test_frame,
)


class PCGNNHotReplayTests(unittest.TestCase):
    def test_select_test_frame_can_select_hot_targets(self):
        frame = pd.DataFrame(
            {
                "_split_source": [
                    "strict_item_cold_test",
                    "strict_item_cold_warm_test",
                    "strict_item_cold_warm_test",
                ],
                "i_idx": [10, 11, 12],
            }
        )

        selected = select_test_frame(frame, "hot")

        self.assertEqual(selected["i_idx"].tolist(), [11, 12])

    def test_select_test_frame_can_select_all_strict_test_targets(self):
        frame = pd.DataFrame(
            {
                "_split_source": [
                    "strict_item_cold_test",
                    "strict_item_cold_warm_test",
                    "strict_item_cold_warm_val",
                ],
                "i_idx": [10, 11, 12],
            }
        )

        selected = select_test_frame(frame, "all")

        self.assertEqual(selected["i_idx"].tolist(), [10, 11])

    def test_combine_cold_hot_overall_weights_by_item_counts(self):
        replay = combine_cold_hot_overall(
            {
                "full_cold_item_macro": {"R@5": 0.2, "N@10": 0.4},
                "full_hot_item_macro": {"R@5": 0.8, "N@10": 1.0},
                "count_full_cold_item_macro": 1,
                "count_full_hot_item_macro": 3,
            }
        )

        self.assertAlmostEqual(replay["full_all_item_macro"]["R@5"], 0.65)
        self.assertAlmostEqual(replay["full_all_item_macro"]["N@10"], 0.85)
        self.assertEqual(replay["count_full_all_item_macro"], 4)

    def test_resolve_config_file_uses_report_dataset_name(self):
        config = resolve_config_file("junyi_strict_full")

        self.assertEqual(config.name, "recbole_junyi_strict_full.yaml")


if __name__ == "__main__":
    unittest.main()
