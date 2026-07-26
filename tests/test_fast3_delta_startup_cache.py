import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast3_delta.course_artifacts import build_course_artifacts
from fast3_delta.static_protocol import static_split_df, write_static_split_artifacts


class Fast3DeltaStartupCacheTest(unittest.TestCase):
    def _set_env(self, key, value):
        old_value = os.environ.get(key)
        os.environ[key] = value

        def restore():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

        self.addCleanup(restore)

    def _unset_env(self, key):
        old_value = os.environ.pop(key, None)

        def restore():
            if old_value is not None:
                os.environ[key] = old_value

        self.addCleanup(restore)

    def test_static_split_df_loads_shared_split_before_recomputing(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "shared_split"
            split_dir.mkdir()
            train_df = pd.DataFrame(
                {
                    "u_idx": [0, 1],
                    "i_idx": [10, 11],
                    "popularity": [3, 4],
                    "course_id": ["c10", "c11"],
                    "_row_id": [0, 1],
                    "_split_source": ["cached_train", "cached_train"],
                }
            )
            val_df = train_df.iloc[[0]].copy()
            val_df["_split_source"] = "cached_val"
            test_df = train_df.iloc[[1]].copy()
            test_df["_split_source"] = "cached_test"
            train_df.to_pickle(split_dir / "static_train.pkl")
            val_df.to_pickle(split_dir / "static_val.pkl")
            test_df.to_pickle(split_dir / "static_test.pkl")
            summary = {
                "split_mode": "strict_item_cold_balanced",
                "split_family": "strict_item_cold",
                "train_rows": 2,
                "val_rows": 1,
                "test_rows": 1,
                "marker": "loaded-from-summary",
            }
            (split_dir / "static_split_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            self._set_env("USIM_STATIC_SPLIT_DIR", str(split_dir))
            self._set_env("USIM_STATIC_SPLIT_MODE", "strict_item_cold_balanced")

            empty_source = pd.DataFrame(columns=["u_idx", "i_idx", "popularity", "course_id"])
            loaded_train, loaded_val, loaded_test, split_info = static_split_df(empty_source)

        self.assertEqual(len(loaded_train), 2)
        self.assertEqual(len(loaded_val), 1)
        self.assertEqual(len(loaded_test), 1)
        self.assertEqual(split_info["marker"], "loaded-from-summary")
        self.assertTrue(split_info["static_split_loaded"])
        self.assertEqual(split_info["static_split_dir"], str(split_dir))

    def test_write_static_split_artifacts_skips_large_exports_for_shared_split_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            split_df = pd.DataFrame(
                {
                    "u_idx": [0],
                    "i_idx": [1],
                    "popularity": [1],
                    "_row_id": [0],
                    "_split_source": ["cached_train"],
                }
            )

            class Cfg:
                n_items = 2
                cold_threshold = 1

            self._set_env("USIM_STATIC_EXPORT_SPLIT", "1")
            self._unset_env("USIM_STATIC_EXPORT_SHARED_SPLIT")
            exports = write_static_split_artifacts(
                split_df,
                split_df,
                split_df,
                {"static_split_loaded": True},
                Cfg(),
                lambda name: str(out_dir / name),
            )

            self.assertTrue(Path(exports["split_summary"]).exists())
            self.assertNotIn("train_split", exports)
            self.assertFalse((out_dir / "static_train.pkl").exists())
            self.assertFalse((out_dir / "static_split_assignments.csv").exists())

    def test_build_course_artifacts_reuses_cache_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relation_dir = root / "relations"
            cache_dir = root / "course_cache"
            relation_dir.mkdir()
            (relation_dir / "course-concept.json").write_text(
                "c0\tKC_A\nc1\tKC_B\n",
                encoding="utf-8",
            )
            (relation_dir / "prerequisite-dependency.json").write_text(
                "KC_A\tKC_B\n",
                encoding="utf-8",
            )
            df = pd.DataFrame(
                {
                    "u_idx": [0, 1, 0],
                    "i_idx": [0, 1, 1],
                    "course_id": ["c0", "c1", "c1"],
                    "timestamp": [1, 2, 3],
                }
            )
            self._set_env("USIM_COURSE_ARTIFACT_CACHE_DIR", str(cache_dir))
            self._unset_env("USIM_COURSE_ARTIFACT_CACHE_DISABLE")

            artifacts_1, stats_1 = build_course_artifacts(
                df,
                2,
                relation_dir=str(relation_dir),
                prereq_graph_source="concept",
                prereq_max_per_item=2,
            )
            artifacts_2, stats_2 = build_course_artifacts(
                df,
                2,
                relation_dir=str(relation_dir),
                prereq_graph_source="concept",
                prereq_max_per_item=2,
            )

            self.assertEqual(stats_1["course_artifact_cache_status"], "miss")
            self.assertEqual(stats_2["course_artifact_cache_status"], "hit")
            self.assertTrue(Path(stats_2["course_artifact_cache_path"]).exists())
            self.assertTrue(
                torch.equal(
                    artifacts_1["item_prereq_item_mat"],
                    artifacts_2["item_prereq_item_mat"],
                )
            )


if __name__ == "__main__":
    unittest.main()
