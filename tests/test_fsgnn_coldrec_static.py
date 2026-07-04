import csv
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsgnn_coldrec_static import (
    apply_cold_item_score_bias,
    export_coldrec_dataset,
    restore_original_order_embeddings,
)


def _read_pairs(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FSGNNColdRecStaticTests(unittest.TestCase):
    def test_export_coldrec_dataset_writes_source_format_and_metadata(self):
        coldrec_root = Path(self._testMethodName)
        if coldrec_root.exists():
            import shutil

            shutil.rmtree(coldrec_root)
        self.addCleanup(lambda: __import__("shutil").rmtree(coldrec_root, ignore_errors=True))
        meta = {"n_users": 4, "n_items": 5, "content_dim": 3}
        content = torch.arange(15, dtype=torch.float32).view(5, 3)
        train_df = pd.DataFrame(
            {
                "u_idx": [0, 1, 2, 3],
                "i_idx": [0, 1, 2, 0],
                "popularity": [3, 2, 2, 3],
            }
        )
        val_df = pd.DataFrame(
            {
                "u_idx": [0, 1],
                "i_idx": [3, 1],
                "popularity": [0, 2],
            }
        )
        test_df = pd.DataFrame(
            {
                "u_idx": [2, 3],
                "i_idx": [4, 2],
                "popularity": [0, 3],
            }
        )

        dataset_dir = export_coldrec_dataset(
            coldrec_root=coldrec_root,
            dataset_name="toy_static_seed2025",
            meta=meta,
            content_emb=content,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cold_threshold=1,
            source_data_dir="processed_toy",
            split_dir="split_toy",
        )

        cold_dir = dataset_dir / "cold_item"
        self.assertEqual(
            _read_pairs(cold_dir / "warm_train.csv"),
            [
                {"user": "0", "item": "0"},
                {"user": "1", "item": "1"},
                {"user": "2", "item": "2"},
                {"user": "3", "item": "0"},
            ],
        )
        self.assertEqual(_read_pairs(cold_dir / "warm_val.csv"), [{"user": "1", "item": "1"}])
        self.assertEqual(_read_pairs(cold_dir / "cold_item_val.csv"), [{"user": "0", "item": "3"}])
        self.assertEqual(_read_pairs(cold_dir / "warm_test.csv"), [{"user": "3", "item": "2"}])
        self.assertEqual(_read_pairs(cold_dir / "cold_item_test.csv"), [{"user": "2", "item": "4"}])
        self.assertEqual(
            _read_pairs(cold_dir / "overall_val.csv"),
            [{"user": "0", "item": "3"}, {"user": "1", "item": "1"}],
        )
        self.assertEqual(
            _read_pairs(cold_dir / "overall_test.csv"),
            [{"user": "2", "item": "4"}, {"user": "3", "item": "2"}],
        )

        with (cold_dir / "info_dict.pkl").open("rb") as f:
            info = pickle.load(f)
        self.assertEqual(info["user_num"], 4)
        self.assertEqual(info["item_num"], 5)
        self.assertEqual(info["warm_user"].tolist(), [0, 1, 2, 3])
        self.assertEqual(info["warm_item"].tolist(), [0, 1, 2])
        self.assertEqual(info["cold_user"].tolist(), [])
        self.assertEqual(info["cold_item"].tolist(), [3, 4])

        saved_content = np.load(dataset_dir / "toy_static_seed2025_item_content.npy")
        np.testing.assert_allclose(saved_content, content.numpy())

    def test_restore_original_order_embeddings_uses_coldrec_id_maps(self):
        mapped_user = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        mapped_item = torch.tensor([[10.0, 10.5], [20.0, 20.5]])

        user_out, item_out = restore_original_order_embeddings(
            mapped_user_emb=mapped_user,
            mapped_item_emb=mapped_item,
            id2user={0: 5, 1: 2, 2: 0},
            id2item={0: 4, 1: 1},
            n_users=6,
            n_items=5,
        )

        self.assertTrue(torch.equal(user_out[5], mapped_user[0]))
        self.assertTrue(torch.equal(user_out[2], mapped_user[1]))
        self.assertTrue(torch.equal(user_out[0], mapped_user[2]))
        self.assertTrue(torch.equal(user_out[1], torch.zeros(2)))
        self.assertTrue(torch.equal(item_out[4], mapped_item[0]))
        self.assertTrue(torch.equal(item_out[1], mapped_item[1]))
        self.assertTrue(torch.equal(item_out[0], torch.zeros(2)))

    def test_main_table_aggregator_knows_fsgnn_result(self):
        import aggregate_main_table_static_results as agg

        self.assertIn("fsgnn_coldrec_static_result.json", agg.RESULT_FILES)
        self.assertIn("FS-GNN", agg.MODEL_ORDER)

    def test_apply_cold_item_score_bias_handles_full_and_sampled_candidates(self):
        scores = torch.zeros(2, 5)
        cold_items = torch.tensor([1, 3])

        full = apply_cold_item_score_bias(scores, cold_items, bias=2.5, cand_idx=None)
        self.assertTrue(torch.equal(full[0], torch.tensor([0.0, 2.5, 0.0, 2.5, 0.0])))
        self.assertTrue(torch.equal(full[1], torch.tensor([0.0, 2.5, 0.0, 2.5, 0.0])))

        sampled_scores = torch.zeros(2, 3)
        cand_idx = torch.tensor([[0, 1, 4], [3, 2, 1]])
        sampled = apply_cold_item_score_bias(sampled_scores, cold_items, bias=1.0, cand_idx=cand_idx)
        self.assertTrue(torch.equal(sampled, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])))


if __name__ == "__main__":
    unittest.main()
