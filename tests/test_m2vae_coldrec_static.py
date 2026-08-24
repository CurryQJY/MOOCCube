import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class M2VAEColdRecStaticTests(unittest.TestCase):
    def test_build_coldrec_argv_uses_official_m2vae_model_and_outputs(self):
        from m2vae_coldrec_static import Config, _build_coldrec_argv

        cfg = Config(
            data_dir="processed_data_coco",
            split_dir="outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_2025",
            cold_threshold=1,
            seed=2025,
            static_seed=2025,
            coldrec_root=Path(".runtime_tmp/ColdRec"),
            dataset_name="m2vae_coco_seed2025",
            output_dir=Path(self._testMethodName),
            epochs=1,
            emb_size=64,
            batch_size=128,
            lr=5e-5,
            topn="5,10,20",
            use_gpu=False,
            gpu_id=0,
            early_stop=1,
            eval_every=1,
            extra_args="",
            eval_batch_size=64,
            eval_n_neg=200,
            run_sampled_eval=False,
            test_history_policy="train_only",
            positive_number=2,
            negative_number=3,
            self_neg_number=4,
            attr_present_dim=16,
            implicit_dim=16,
            cat_implicit_dim=16,
            tau=0.2,
            weight_decay=0.01,
            kld_weight=1.0,
            recon_weight=1.0,
            decouple_weight=10.0,
            pretrain=False,
            pretrain_update=False,
            attr_mask_neg1=False,
        )

        argv = _build_coldrec_argv("m2vae_coco_seed2025", cfg)

        self.assertEqual(argv[argv.index("--model") + 1], "M2VAE")
        self.assertEqual(argv[argv.index("--cold_object") + 1], "item")
        self.assertTrue(argv[argv.index("--result_file") + 1].endswith("coldrec_native_m2vae_result.txt"))
        self.assertIn("--m2vae_weight_decay", argv)
        self.assertEqual(argv[argv.index("--m2vae_weight_decay") + 1], "0.01")
        self.assertNotIn("--m2vae_pretrain", argv)

    def test_main_table_aggregator_knows_m2vae_result(self):
        import aggregate_main_table_static_results as agg

        self.assertIn("m2vae_coldrec_static_result.json", agg.RESULT_FILES)
        self.assertIn("M2VAE", agg.MODEL_ORDER)


if __name__ == "__main__":
    unittest.main()
