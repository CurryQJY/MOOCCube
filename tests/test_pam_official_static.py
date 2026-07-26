import csv
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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

    def test_train_selects_earliest_best_validation_checkpoint_before_one_test_pass(self):
        import pam_official_static as pam

        output_dir = self.tmp / "output"
        view_dir = output_dir / "pam_official_view"
        view_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "userId": [0],
                "itemSeq": [""],
                "itemId": [0],
                "userSeq": [""],
                "label": [1],
                "vv": [0],
                "period": [0],
            }
        ).to_csv(view_dir / "pam_train.csv", index=False)
        pd.DataFrame({"itemId": [0, 1, 2, 3], "cateId": ["1", "1", "1", "1"]}).to_csv(
            view_dir / "pam_content.csv",
            index=False,
        )
        pd.DataFrame({"u_idx": [0], "i_idx": [0], "timestamp": [0], "popularity": [0]}).to_csv(
            view_dir / "pam_train_interactions.csv",
            index=False,
        )
        pd.DataFrame({"u_idx": [0], "i_idx": [1], "timestamp": [1], "popularity": [0]}).to_csv(
            view_dir / "pam_val_targets.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "u_idx": [0, 0],
                "i_idx": [2, 3],
                "timestamp": [2, 3],
                "popularity": [0, 1],
            }
        ).to_csv(view_dir / "pam_test_targets.csv", index=False)

        pam_root = self.tmp / "PAM"
        pam_code_dir = pam_root / "PAM-F"
        pam_code_dir.mkdir(parents=True)
        (pam_code_dir / "model.py").touch()
        cfg = pam.Config(
            data_dir="",
            split_dir="",
            relation_dir="",
            output_dir=output_dir,
            pam_root=pam_root,
            seed=2025,
            static_seed=2025,
            cold_threshold=1,
            train_ratio=0.8,
            val_ratio=0.1,
            epochs=3,
            batch_size=1,
            lr=1e-3,
            emb_dim=8,
            hidden_dim=16,
            cate_dim=8,
            neg_per_pos=1,
            max_train_pos=0,
            max_eval_rows=0,
            max_cates_per_item=8,
            eval_item_batch_size=4,
            use_gpu=False,
            init_checkpoint="",
            start_epoch=0,
        )
        manifest = {"n_users": 1, "n_items": 4, "num_cates": 2}
        events = []
        holder = {}

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def run(self, values, feed_dict=None):
                return None

        class FakeSaver:
            def save(self, sess, path):
                events.append(("save", path))
                return path

            def restore(self, sess, path):
                events.append(("restore", path))

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.epoch = 0
                holder["model"] = self

        class FakeEngine:
            def __init__(self, sess, model, history):
                self.model = model

            def base_train_an_epoch(self, epoch, train_view, train_config):
                self.model.epoch = epoch
                events.append(("train", epoch))
                return float(epoch)

        fake_tf = types.ModuleType("tensorflow.compat.v1")
        fake_tf.Session = FakeSession
        fake_tf.train = types.SimpleNamespace(Saver=lambda: FakeSaver())
        fake_tf.global_variables_initializer = lambda: "global-init"
        fake_tf.local_variables_initializer = lambda: "local-init"
        fake_tf.reset_default_graph = lambda: None
        fake_tf.disable_eager_execution = lambda: None
        fake_tf.set_random_seed = lambda seed: None
        fake_compat = types.ModuleType("tensorflow.compat")
        fake_compat.__path__ = []
        fake_compat.v1 = fake_tf
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.__path__ = []
        fake_tensorflow.compat = fake_compat
        fake_engine = types.ModuleType("engine")
        fake_engine.Engine = FakeEngine
        fake_model = types.ModuleType("model")
        fake_model.EmbMLP = FakeModel

        def fake_evaluate(*, targets, **kwargs):
            target_ids = tuple(target.item_id for target in targets)
            events.append(("evaluate", target_ids))
            if target_ids == (1,):
                ndcg = {1: 0.20, 2: 0.40, 3: 0.40}[holder["model"].epoch]
                return {"N@10": ndcg}, {"N@10": ndcg}, 1, 1
            return {"N@10": 0.10}, {"N@10": 0.10}, 1, 1

        fake_modules = {
            "tensorflow": fake_tensorflow,
            "tensorflow.compat": fake_compat,
            "tensorflow.compat.v1": fake_tf,
            "engine": fake_engine,
            "model": fake_model,
        }
        with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
            pam,
            "evaluate_pam_full_catalog",
            side_effect=fake_evaluate,
        ):
            result = pam.train_and_evaluate(cfg, manifest)

        selection = result.get("checkpoint_selection")
        self.assertIsNotNone(selection)
        self.assertEqual(selection["selected_epoch"], 2)
        self.assertEqual(
            [
                event
                for event in events
                if event[0] == "save" and Path(event[1]).name.startswith("pam_official_epoch_")
            ],
            [
                ("save", str(output_dir / "checkpoints" / "pam_official_epoch_1.ckpt")),
                ("save", str(output_dir / "checkpoints" / "pam_official_epoch_2.ckpt")),
                ("save", str(output_dir / "checkpoints" / "pam_official_epoch_3.ckpt")),
            ],
        )
        self.assertIn(
            ("save", str(output_dir / "checkpoints" / "pam_official_latest.ckpt")),
            events,
        )
        self.assertEqual(
            [event for event in events if event[0] == "restore"],
            [("restore", str(output_dir / "checkpoints" / "pam_official_epoch_2.ckpt"))],
        )
        self.assertEqual(
            [event for event in events if event[0] == "evaluate"],
            [
                ("evaluate", (1,)),
                ("evaluate", (1,)),
                ("evaluate", (1,)),
                ("evaluate", (2,)),
                ("evaluate", (3,)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
