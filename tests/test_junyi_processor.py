import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_process_junyi import (
    build_junyi_metadata,
    junyi_self_concept,
    normalize_junyi_timestamp,
    save_junyi_processed_dataset,
)


class JunyiProcessorTest(unittest.TestCase):
    def test_time_done_microseconds_are_normalized_to_seconds(self):
        interactions = pd.DataFrame({"time_done": [1420714810324490]})

        normalized, timestamp_col = normalize_junyi_timestamp(interactions, "time_done")

        self.assertEqual(timestamp_col, "timestamp")
        self.assertEqual(int(normalized["timestamp"].iloc[0]), 1420714810)

    def test_prerequisites_use_exercise_self_concepts(self):
        interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "exercise": ["basic_fraction", "advanced_fraction"],
                "timestamp": [1, 2],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            pd.DataFrame(
                {
                    "name": ["basic_fraction", "advanced_fraction"],
                    "topic": ["fractions", "fractions"],
                    "area": ["arithmetic", "arithmetic"],
                    "prerequisites": ["", "basic_fraction"],
                }
            ).to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

            metadata = build_junyi_metadata(interactions, item_col="exercise", raw_dir=raw_dir)
            stats = {
                "dataset": "Junyi",
                "interactions": 2,
                "n_users": 1,
                "n_items": 2,
                "min_user_interactions": 1,
                "min_item_interactions": 1,
                "positive_only": False,
                "item_id_classes": ["advanced_fraction", "basic_fraction"],
            }
            save_junyi_processed_dataset(
                pd.DataFrame(
                    {
                        "user_id": ["u1", "u1"],
                        "course_id": ["basic_fraction", "advanced_fraction"],
                        "raw_time": ["1", "2"],
                        "timestamp": [1, 2],
                        "u_idx": [0, 0],
                        "i_idx": [1, 0],
                        "popularity": [0, 0],
                    }
                ),
                stats,
                metadata,
                type(
                    "Spec",
                    (),
                    {
                        "dataset": "Junyi",
                        "output_dir": out_dir,
                        "content_dim": 8,
                        "embedding_backend": "stable_hash",
                        "embedding_model": "",
                        "embedding_max_length": 256,
                        "embedding_batch_size": 32,
                        "embedding_local_files_only": False,
                    },
                )(),
            )

            prereq = (out_dir / "relations" / "prerequisite-dependency.json").read_text(
                encoding="utf-8"
            )
            course_concept = (out_dir / "relations" / "course-concept.json").read_text(
                encoding="utf-8"
            )

        self.assertIn(
            f"{junyi_self_concept('basic_fraction')}\t{junyi_self_concept('advanced_fraction')}",
            prereq,
        )
        self.assertNotIn("basic_fraction\tfractions", prereq)
        self.assertIn(f"advanced_fraction\t{junyi_self_concept('advanced_fraction')}", course_concept)


if __name__ == "__main__":
    unittest.main()
