import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class PamOfficialStaticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pam_official_static_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_export_pam_dataset_writes_official_columns_and_manifest(self):
        from pam_official_static import export_pam_dataset_view

        data_dir = self.tmp / "processed"
        relation_dir = data_dir / "relations"
        relation_dir.mkdir(parents=True)
        with (data_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"n_users": 3, "n_items": 4, "content_dim": 8}, f)
        df = pd.DataFrame(
            {
                "user_id": ["u0", "u1", "u0", "u2", "u1"],
                "course_id": ["c0", "c1", "c2", "c3", "c0"],
                "timestamp": [1, 2, 3, 4, 5],
                "u_idx": [0, 1, 0, 2, 1],
                "i_idx": [0, 1, 2, 3, 0],
                "popularity": [0, 0, 0, 0, 1],
            }
        )
        df.to_pickle(data_dir / "stream_data.pkl")
        with (relation_dir / "course-concept.json").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["c0", "math"])
            writer.writerow(["c0", "algebra"])
            writer.writerow(["c2", "geometry"])

        train_df = df.iloc[:3].copy()
        val_df = df.iloc[3:4].copy()
        test_df = df.iloc[4:].copy()
        out_dir = self.tmp / "pam_view"

        manifest = export_pam_dataset_view(
            data_dir=str(data_dir),
            relation_dir=str(relation_dir),
            output_dir=out_dir,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            n_users=3,
            n_items=4,
            seed=2025,
            cold_threshold=1,
            batch_size=4,
            neg_per_pos=1,
            max_train_pos=0,
            max_eval_rows=0,
            max_cates_per_item=3,
        )

        train = pd.read_csv(out_dir / "pam_train.csv")
        train_interactions = pd.read_csv(out_dir / "pam_train_interactions.csv")
        test_targets = pd.read_csv(out_dir / "pam_test_targets.csv")
        content = pd.read_csv(out_dir / "pam_content.csv")
        self.assertEqual(
            list(train.columns),
            ["userId", "itemSeq", "itemId", "userSeq", "label", "vv", "period"],
        )
        self.assertGreaterEqual(set(train["label"].unique()), {0, 1})
        self.assertEqual(len(train) % 4, 0)
        self.assertEqual(list(train_interactions.columns), ["u_idx", "i_idx", "timestamp", "popularity"])
        self.assertEqual(list(test_targets.columns), ["u_idx", "i_idx", "timestamp", "popularity"])
        self.assertEqual(list(content.columns), ["itemId", "cateId"])
        self.assertEqual(len(content), 4)
        self.assertTrue(str(content.loc[content["itemId"] == 0, "cateId"].iloc[0]))
        self.assertEqual(manifest["official_format"], "PAM-F csv")
        self.assertEqual(manifest["trimmed_train_rows_mod_batch"], 2)
        self.assertEqual(manifest["category_source"], str(relation_dir / "course-concept.json"))

        import pam_official_static as pam

        train_loaded, _ = pam._load_pam_csv(out_dir)
        self.assertIsInstance(train_loaded.loc[0, "itemSeq"], list)
        self.assertIsInstance(train_loaded.loc[0, "userSeq"], list)
        self.assertEqual(pam._parse_hash_seq("132.0#5"), [132, 5])

    def test_build_eval_targets_keeps_item_macro_groups(self):
        from pam_official_static import build_eval_targets, limit_eval_rows

        df = pd.DataFrame(
            {
                "u_idx": [0, 0, 1, 1],
                "i_idx": [2, 3, 2, 1],
                "popularity": [0, 5, 0, 2],
            }
        )

        cold, hot = build_eval_targets(df, cold_threshold=1)

        self.assertEqual([row.item_id for row in cold], [2, 2])
        self.assertEqual([row.item_id for row in hot], [3, 1])

        limited = limit_eval_rows(df, max_rows=2, cold_threshold=1)
        self.assertEqual(len(limited), 2)
        self.assertEqual(int((limited["popularity"] < 1).sum()), 1)
        self.assertEqual(int((limited["popularity"] >= 1).sum()), 1)

    def test_main_table_aggregator_knows_pam_result(self):
        import aggregate_main_table_static_results as agg

        self.assertIn("pam_official_static_result.json", agg.RESULT_FILES)
        self.assertIn("PAM", agg.MODEL_ORDER)


if __name__ == "__main__":
    unittest.main()
