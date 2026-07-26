import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.pcgnn_strict_adapter import (  # noqa: E402
    BestValidationTracker,
    ItemMacroRankingAccumulator,
    build_kg_training_pool,
    build_strict_eval_examples,
    build_train_item_ids,
    calculate_rs_loss_with_candidates,
    clean_argv_for_recbole,
    format_epoch_progress,
    metric_from_report,
    parse_args,
    resolve_workspace_path,
    sample_kg_batch,
    train_one_epoch,
)


class PCGNNStrictAdapterTests(unittest.TestCase):
    def test_accumulator_masks_seen_items_but_restores_target_score(self):
        accumulator = ItemMacroRankingAccumulator(k_list=[1], cold_threshold=1)
        scores = np.array([[0.0, 0.95, 0.80, 0.70]], dtype=np.float32)
        examples = [
            {
                "user": 7,
                "target": 2,
                "raw_item": 20,
                "popularity": 0,
                "history": [1, 2],
            }
        ]

        accumulator.add_batch(scores, examples, user_seen_items={7: {1, 2}})
        report = accumulator.result()

        self.assertEqual(report["count_full_cold_item_macro"], 1)
        self.assertAlmostEqual(report["full_cold_item_macro"]["R@1"], 1.0)
        self.assertAlmostEqual(report["full_cold_item_macro"]["N@1"], 1.0)

    def test_accumulator_computes_item_macro_not_interaction_macro(self):
        accumulator = ItemMacroRankingAccumulator(k_list=[1], cold_threshold=1)
        scores = np.array(
            [
                [0.0, 0.90, 0.95, 0.10],
                [0.0, 0.90, 0.20, 0.10],
                [0.0, 0.10, 0.20, 0.95],
            ],
            dtype=np.float32,
        )
        examples = [
            {"user": 1, "target": 2, "raw_item": 20, "popularity": 0, "history": []},
            {"user": 2, "target": 2, "raw_item": 20, "popularity": 0, "history": []},
            {"user": 3, "target": 3, "raw_item": 30, "popularity": 0, "history": []},
        ]

        accumulator.add_batch(scores, examples, user_seen_items={})
        report = accumulator.result()

        self.assertEqual(report["count_full_cold_item_macro"], 2)
        self.assertAlmostEqual(report["full_cold_item_macro"]["R@1"], 0.75)
        self.assertAlmostEqual(report["full_cold_item_macro"]["N@1"], 0.75)

    def test_build_strict_eval_examples_keeps_popularity_and_requires_train_history(self):
        train_rows = [
            {"u_idx": 1, "i_idx": 10, "timestamp": 1},
            {"u_idx": 2, "i_idx": 11, "timestamp": 1},
        ]
        eval_rows = [
            {"u_idx": 1, "i_idx": 12, "timestamp": 2, "popularity": 0},
            {"u_idx": 3, "i_idx": 13, "timestamp": 2, "popularity": 7},
        ]
        token_map = {"10": 110, "11": 111, "12": 112, "13": 113}

        examples = build_strict_eval_examples(train_rows, eval_rows, token_map, max_len=5)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["user"], 1)
        self.assertEqual(examples[0]["history"], [110])
        self.assertEqual(examples[0]["target"], 112)
        self.assertEqual(examples[0]["raw_item"], 12)
        self.assertEqual(examples[0]["popularity"], 0)

    def test_script_entrypoint_runs_when_invoked_by_path(self):
        script = ROOT / "paper_aaai27" / "scripts" / "pcgnn_strict_adapter.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-train-examples", result.stdout)

    def test_default_cli_uses_full_dataset_not_smoke(self):
        args = parse_args([])

        self.assertEqual(args.dataset_name, "mooccube_strict_seed2025_full")
        self.assertTrue(str(args.config_file).endswith("recbole_mooccube_strict_seed2025_full.yaml"))

    def test_metric_from_report_reads_nested_validation_metric(self):
        report = {"full_cold_item_macro": {"N@10": 0.123}}

        self.assertAlmostEqual(metric_from_report(report, "full_cold_item_macro.N@10"), 0.123)

    def test_best_validation_tracker_saves_best_and_stops_after_patience(self):
        tracker = BestValidationTracker(metric_path="full_cold_item_macro.N@10", patience=1)

        self.assertTrue(tracker.update(1, {"full_cold_item_macro": {"N@10": 0.10}}, {"w": 1}))
        self.assertEqual(tracker.best_epoch, 1)
        self.assertFalse(tracker.should_stop)

        self.assertFalse(tracker.update(2, {"full_cold_item_macro": {"N@10": 0.05}}, {"w": 2}))
        self.assertTrue(tracker.should_stop)
        self.assertEqual(tracker.best_state, {"w": 1})

    def test_resolve_workspace_path_makes_relative_paths_absolute_before_chdir(self):
        path = resolve_workspace_path(Path("paper_aaai27") / "baseline_sources")

        self.assertEqual(path, ROOT / "paper_aaai27" / "baseline_sources")
        self.assertTrue(path.is_absolute())

    def test_clean_argv_for_recbole_hides_adapter_cli_args(self):
        original = sys.argv[:]
        sys.argv = ["pcgnn_strict_adapter.py", "--epochs", "2"]
        try:
            with clean_argv_for_recbole():
                self.assertEqual(sys.argv, ["pcgnn_strict_adapter.py"])
            self.assertEqual(sys.argv, ["pcgnn_strict_adapter.py", "--epochs", "2"])
        finally:
            sys.argv = original

    def test_pcgnn_slice_uses_numpy_array_before_tensor_conversion(self):
        model_path = (
            ROOT
            / "paper_aaai27"
            / "baseline_sources"
            / "PCGNN_recbole_drive"
            / "RecBole-master"
            / "recbole"
            / "model"
            / "sequential_recommender"
            / "kg_model.py"
        )
        text = model_path.read_text(encoding="utf-8")

        self.assertNotIn("torch.FloatTensor(A).to(self.device)", text)
        self.assertIn("np.asarray(A", text)

    def test_format_epoch_progress_includes_best_marker(self):
        text = format_epoch_progress(
            epoch=3,
            loss=1.23456,
            metric_name="full_cold_item_macro.N@10",
            metric_value=0.04567,
            improved=True,
        )

        self.assertIn("epoch=3", text)
        self.assertIn("loss=1.2346", text)
        self.assertIn("full_cold_item_macro.N@10=0.04567000", text)
        self.assertIn("best", text)

    def test_sample_kg_batch_adds_negative_tail_and_avoids_known_positive_tails(self):
        kg_feat = pd.DataFrame(
            {
                "head_id": [1, 1, 2],
                "relation_id": [3, 4, 3],
                "tail_id": [5, 6, 5],
            }
        )
        pool = build_kg_training_pool(
            kg_feat,
            head_field="head_id",
            relation_field="relation_id",
            tail_field="tail_id",
            entity_count=9,
        )

        batch = sample_kg_batch(
            pool,
            batch_size=24,
            neg_tail_field="neg_tail_id",
            rng=np.random.default_rng(2025),
        )

        self.assertEqual(set(batch), {"head_id", "relation_id", "tail_id", "neg_tail_id"})
        self.assertEqual(batch["head_id"].shape[0], 24)
        for head, neg_tail in zip(batch["head_id"].tolist(), batch["neg_tail_id"].tolist()):
            self.assertNotIn(int(neg_tail), pool.used_tails_by_head[int(head)])
            self.assertGreaterEqual(int(neg_tail), 1)
            self.assertLess(int(neg_tail), pool.entity_count)

    def test_train_one_epoch_combines_rs_and_kg_losses_when_pool_is_provided(self):
        import torch

        class SimpleInteraction(dict):
            pass

        class FakePCGNN(torch.nn.Module):
            ITEM_SEQ = "item_seq"
            ITEM_SEQ_LEN = "item_length"
            ITEM_ID = "item_id"
            HEAD_ENTITY_ID = "head_id"
            RELATION_ID = "relation_id"
            TAIL_ENTITY_ID = "tail_id"
            NEG_TAIL_ENTITY_ID = "neg_tail_id"

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.rs_calls = 0
                self.kg_calls = 0

            def calculate_rs_loss(self, interaction):
                self.rs_calls += 1
                self.assert_has = interaction[self.ITEM_ID].shape[0]
                return self.weight * 0 + 2.0

            def calculate_kg_loss(self, interaction):
                self.kg_calls += 1
                self.assert_has = interaction[self.NEG_TAIL_ENTITY_ID].shape[0]
                return self.weight * 0 + 3.0

        kg_pool = build_kg_training_pool(
            pd.DataFrame({"head_id": [1, 2], "relation_id": [1, 1], "tail_id": [3, 4]}),
            head_field="head_id",
            relation_field="relation_id",
            tail_field="tail_id",
            entity_count=8,
        )
        model = FakePCGNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        stats = train_one_epoch(
            model,
            SimpleInteraction,
            examples=[{"history": [1], "target": 2}],
            max_len=3,
            batch_size=1,
            optimizer=optimizer,
            kg_pool=kg_pool,
            kg_batch_size=2,
            kg_loss_weight=0.5,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(model.rs_calls, 1)
        self.assertEqual(model.kg_calls, 1)
        self.assertAlmostEqual(stats["rs_loss"], 2.0)
        self.assertAlmostEqual(stats["kg_loss"], 3.0)
        self.assertAlmostEqual(stats["loss"], 3.5)

    def test_build_train_item_ids_excludes_unseen_cold_items(self):
        train_rows = [
            {"u_idx": 1, "i_idx": 10, "timestamp": 1},
            {"u_idx": 2, "i_idx": 11, "timestamp": 2},
            {"u_idx": 3, "i_idx": 10, "timestamp": 3},
        ]
        token_map = {"10": 4, "11": 2, "12": 9}

        self.assertEqual(build_train_item_ids(train_rows, token_map), [2, 4])

    def test_candidate_rs_loss_uses_only_train_items_as_ce_classes(self):
        import torch

        class FakePCGNN(torch.nn.Module):
            ITEM_SEQ = "item_seq"
            ITEM_SEQ_LEN = "item_length"
            ITEM_ID = "item_id"

            def __init__(self):
                super().__init__()
                self.n_items = 5
                self.n_cats = 1
                self.aux_weight = 0
                self.loss_type = "CE"
                self.loss_fct = torch.nn.CrossEntropyLoss()
                self.entity_embedding = torch.nn.Embedding(5, 2)
                with torch.no_grad():
                    self.entity_embedding.weight.copy_(
                        torch.tensor(
                            [
                                [0.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 1.0],
                                [5.0, 5.0],
                                [6.0, 6.0],
                            ]
                        )
                    )

            def forward(self, item_seq, item_seq_len):
                output = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
                return output, output[:, :2]

            def _get_cat_seq(self, item_ids):
                return torch.zeros_like(item_ids)

            def calculate_rs_loss(self, interaction):
                raise AssertionError("candidate loss should not call full-catalog RS loss")

        interaction = {
            "item_seq": torch.tensor([[1]]),
            "item_length": torch.tensor([1]),
            "item_id": torch.tensor([2]),
        }
        model = FakePCGNN()

        train_only_loss = calculate_rs_loss_with_candidates(
            model,
            interaction,
            candidate_item_ids=torch.tensor([1, 2]),
        )
        with_cold_negative_loss = calculate_rs_loss_with_candidates(
            model,
            interaction,
            candidate_item_ids=torch.tensor([1, 2, 3, 4]),
        )

        self.assertLess(float(train_only_loss.detach()), float(with_cold_negative_loss.detach()))

    def test_format_epoch_progress_can_show_rs_and_kg_losses(self):
        text = format_epoch_progress(
            epoch=1,
            loss=3.5,
            metric_name="full_cold_item_macro.N@10",
            metric_value=0.0,
            improved=False,
            rs_loss=2.0,
            kg_loss=3.0,
        )

        self.assertIn("rs_loss=2.0000", text)
        self.assertIn("kg_loss=3.0000", text)

    def test_format_epoch_progress_keeps_tiny_validation_metric_visible(self):
        text = format_epoch_progress(
            epoch=1,
            loss=3.5,
            metric_name="full_cold_item_macro.N@10",
            metric_value=0.000006730884,
            improved=True,
        )

        self.assertIn("full_cold_item_macro.N@10=0.00000673", text)


if __name__ == "__main__":
    unittest.main()
