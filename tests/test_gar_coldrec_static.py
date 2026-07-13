import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GARColdRecStaticTests(unittest.TestCase):
    def _config(self, root: Path, output: Path):
        from gar_coldrec_static import Config

        return Config(
            data_dir="processed_data_hin_clean_pop5",
            split_dir=(
                "outputs/content_delta_pop5/static_item_cold_balanced/"
                "strict_item_cold_balanced_thr1_seed_2025"
            ),
            cold_threshold=1,
            seed=2025,
            static_seed=2025,
            coldrec_root=root,
            dataset_name="gar_mooccube_seed2025",
            output_dir=output,
            epochs=10,
            emb_size=64,
            batch_size=4096,
            lr=1e-3,
            reg=1e-4,
            topn="5,10,20",
            use_gpu=True,
            gpu_id=0,
            early_stop=5,
            eval_every=1,
            extra_args="",
            eval_batch_size=2048,
            eval_n_neg=200,
            run_sampled_eval=False,
            test_history_policy="train_only",
            backbone="MF",
            alpha=0.5,
            beta=0.5,
        )

    def test_build_coldrec_argv_selects_released_gar_with_mf_backbone(self):
        from gar_coldrec_static import _build_coldrec_argv

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._config(tmp_path / "ColdRec", tmp_path / "out")
            argv = _build_coldrec_argv(cfg.dataset_name, cfg)

        self.assertEqual(argv[argv.index("--model") + 1], "GAR")
        self.assertEqual(argv[argv.index("--backbone") + 1], "MF")
        self.assertEqual(argv[argv.index("--cold_object") + 1], "item")
        self.assertEqual(argv[argv.index("--seed") + 1], "2025")
        self.assertEqual(argv[argv.index("--runs") + 1], "1")
        self.assertEqual(argv[argv.index("--use_gpu") + 1], "true")
        self.assertEqual(argv[argv.index("--alpha") + 1], "0.5")
        self.assertEqual(argv[argv.index("--beta") + 1], "0.5")
        self.assertNotIn("--m2vae_pretrain", argv)
        self.assertNotIn("--fsgnn_ppr_alpha", argv)

    def test_require_mf_embeddings_needs_both_matching_files(self):
        from gar_coldrec_static import require_mf_embeddings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emb_dir = root / "emb"
            emb_dir.mkdir()
            dataset = "gar_mooccube_seed2025"

            with self.assertRaises(FileNotFoundError):
                require_mf_embeddings(root, dataset, "MF")

            torch.save(torch.zeros(2, 3), emb_dir / f"{dataset}_cold_item_MF_user_emb.pt")
            with self.assertRaises(FileNotFoundError):
                require_mf_embeddings(root, dataset, "MF")

            item_path = emb_dir / f"{dataset}_cold_item_MF_item_emb.pt"
            torch.save(torch.zeros(4, 3), item_path)
            user_path, returned_item_path = require_mf_embeddings(root, dataset, "MF")

        self.assertTrue(user_path.name.endswith("_MF_user_emb.pt"))
        self.assertEqual(returned_item_path.name, item_path.name)

    def test_assert_strict_cold_disjoint_rejects_heldout_cold_item_in_train(self):
        from gar_coldrec_static import assert_strict_cold_disjoint

        train_df = pd.DataFrame({"i_idx": [0, 1, 1], "popularity": [3, 2, 2]})
        val_df = pd.DataFrame({"i_idx": [2], "popularity": [0]})
        test_df = pd.DataFrame({"i_idx": [3], "popularity": [0]})
        audit = assert_strict_cold_disjoint(train_df, val_df, test_df, cold_threshold=1)
        self.assertEqual(audit["heldout_cold_item_count"], 2)
        self.assertEqual(audit["train_overlap_count"], 0)

        leaked_train = pd.DataFrame({"i_idx": [0, 2], "popularity": [3, 2]})
        with self.assertRaisesRegex(ValueError, "held-out cold items appear in train"):
            assert_strict_cold_disjoint(leaked_train, val_df, test_df, cold_threshold=1)

    def test_strict_validation_callback_selects_cold_item_macro_n10(self):
        from gar_coldrec_static import bind_strict_validation_callback

        class FakeTrainer:
            def __init__(self):
                self.bestPerformance = []
                self.early_stop_flag = True
                self.early_stop_patience = 2
                self.max_early_stop_patience = 2
                self.saved = 0

            def save(self):
                self.saved += 1

        scores = iter([0.25, 0.20, 0.30])

        def evaluate(_trainer):
            score = next(scores)
            return {"R@10": score + 0.1, "N@10": score}, 2

        trainer = FakeTrainer()
        bind_strict_validation_callback(trainer, evaluate)

        first = trainer.fast_evaluation(0)
        self.assertEqual(first["N@10"], 0.25)
        self.assertEqual(trainer.saved, 1)
        self.assertEqual(trainer.bestPerformance[0], 1)
        self.assertEqual(trainer.bestPerformance[1]["NDCG"], 0.25)
        self.assertEqual(trainer.early_stop_patience, 2)

        trainer.fast_evaluation(1)
        self.assertEqual(trainer.saved, 1)
        self.assertEqual(trainer.early_stop_patience, 1)

        trainer.fast_evaluation(2)
        self.assertEqual(trainer.saved, 2)
        self.assertEqual(trainer.bestPerformance[0], 3)
        self.assertEqual(trainer.bestPerformance[1]["NDCG"], 0.30)
        self.assertEqual(trainer.early_stop_patience, 2)
        self.assertEqual(len(trainer.strict_validation_history), 3)

    def test_strict_validation_evaluator_uses_full_catalog_item_macro(self):
        from gar_coldrec_static import make_strict_validation_evaluator

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp) / "ColdRec", Path(tmp) / "out")
            cfg.eval_batch_size = 2
            val_df = pd.DataFrame(
                {
                    "u_idx": [0],
                    "i_idx": [2],
                    "popularity": [0],
                }
            )
            trainer = SimpleNamespace(
                user_emb=torch.tensor([[1.0, 0.0]]),
                item_emb=torch.tensor([[0.8, 0.0], [0.0, 1.0], [1.0, 0.0]]),
                data=SimpleNamespace(id2user={0: 0}, id2item={0: 0, 1: 1, 2: 2}),
            )
            evaluate = make_strict_validation_evaluator(
                val_df=val_df,
                train_seen={0: {0}},
                cfg=cfg,
                n_users=1,
                n_items=3,
            )
            metrics, item_count = evaluate(trainer)

        self.assertEqual(item_count, 1)
        self.assertEqual(metrics["R@10"], 1.0)
        self.assertEqual(metrics["N@10"], 1.0)

    def test_result_payload_records_strict_protocol_and_source_fidelity(self):
        from gar_coldrec_static import build_result_payload

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp) / "ColdRec", Path(tmp) / "out")
            result = build_result_payload(
                cfg=cfg,
                git_info={
                    "official_repo": "https://github.com/YuanchenBei/ColdRec",
                    "official_commit": "18efd24",
                    "official_tree_clean": True,
                },
                strict_audit={"heldout_cold_item_count": 2, "train_overlap_count": 0},
                best_epoch=3,
                best_val_n10=0.25,
                full_cold={"R@10": 0.2, "N@10": 0.1},
                full_hot={"R@10": 0.3, "N@10": 0.2},
                full_cold_item={"R@10": 0.4, "N@10": 0.3},
                full_hot_item={"R@10": 0.5, "N@10": 0.4},
                counts={
                    "full_cold": 10,
                    "full_hot": 20,
                    "full_cold_item": 2,
                    "full_hot_item": 3,
                },
                device="cuda:0",
                elapsed_seconds=12.5,
                per_item_cold_path=Path("cold.csv"),
                per_item_hot_path=Path("hot.csv"),
            )

        self.assertEqual(result["model"], "GAR-coldrec-source-strict")
        self.assertEqual(result["candidate_mode"], "full_catalog")
        self.assertEqual(result["checkpoint_metric"], "validation_full_cold_item_macro.N@10")
        self.assertTrue(result["item_macro_metrics"])
        self.assertTrue(result["train_history_masking"])
        self.assertEqual(result["test_history_policy"], "train_only")
        self.assertTrue(result["source_model_unchanged"])
        self.assertEqual(result["official_commit"], "18efd24")
        self.assertEqual(result["strict_audit"]["train_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
