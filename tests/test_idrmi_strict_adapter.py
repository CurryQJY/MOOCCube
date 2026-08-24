import math
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_aaai27.scripts.idrmi_strict_adapter as idrmi_adapter  # noqa: E402

from paper_aaai27.scripts.idrmi_strict_adapter import (  # noqa: E402
    BestValidationTracker,
    DeviceKGCN,
    IDRMIStrictModel,
    ItemMacroRankingAccumulator,
    build_kg_neighbors,
    build_course_match_table,
    build_history_tensor,
    catalog_interest_factors,
    evaluate_split,
    gpu_interest_factors,
    load_strict_dataset,
    build_train_structures,
    configure_reproducibility,
    load_idrmi_source_classes,
    parse_args,
    prepare_labeled_training_rows,
    resolve_source_dir,
    run_idrmi_strict_adapter,
    sample_warm_negatives,
    source_fusion_score,
    source_interest_factors,
)


class IDRMIStrictAdapterTests(unittest.TestCase):
    def test_train_structures_use_positive_train_edges_only(self):
        rows = [(0, 0, 1), (0, 2, 0), (1, 1, 1)]

        structures = build_train_structures(
            rows,
            n_users=2,
            n_items=3,
            forbidden_cold_items={2},
        )

        self.assertEqual(structures.positive_edges, 2)
        self.assertEqual(structures.user_items, {0: {0}, 1: {1}})
        self.assertEqual(structures.item_users, {0: {0}, 1: {1}})
        self.assertEqual(structures.warm_items, (0, 1))
        self.assertEqual(structures.norm_adj.shape, (5, 5))
        self.assertGreater(structures.norm_adj[0, 2], 0.0)
        self.assertGreater(structures.norm_adj[1, 3], 0.0)
        self.assertEqual(float(structures.norm_adj[0, 4]), 0.0)

    def test_train_structures_reject_positive_cold_edge(self):
        with self.assertRaisesRegex(ValueError, "cold course"):
            build_train_structures(
                [(0, 2, 1)],
                n_users=1,
                n_items=3,
                forbidden_cold_items={2},
            )

    def test_warm_negative_sampling_never_uses_cold_courses(self):
        positives = [(0, 0), (1, 1), (0, 0), (1, 1)]

        negatives = sample_warm_negatives(
            positives,
            warm_items=(0, 1),
            user_items={0: {0}, 1: {1}},
            rng=np.random.default_rng(2025),
        )

        self.assertEqual(negatives, [(0, 1), (1, 0), (0, 1), (1, 0)])
        self.assertNotIn(2, {item for _, item in negatives})

    def test_source_fusion_matches_released_minmax_and_tanh_formula(self):
        user_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        item_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        factors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        scores = source_fusion_score(user_vectors, item_vectors, factors)

        learned = torch.sigmoid(torch.tensor([1.0, 1.0, 2.0]))
        expected = torch.tanh(learned * torch.tensor([0.5, 1.0, 1.5]))
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)

    def test_interest_factors_follow_released_batch_local_definitions(self):
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([1, 1])
        user_vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        item_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        factors = source_interest_factors(
            user_ids,
            item_ids,
            user_vectors,
            item_vectors,
            user_items={0: {0}, 1: {1}},
            item_users={0: {0}, 1: {1}},
        )

        inverse_distance = 1.0 / (math.sqrt(2.0) + 1.0)
        expected = torch.tensor(
            [
                [0.0, 1.0 / 6.0, 0.0],
                [0.0, 1.0 / 6.0, (1.0 + inverse_distance) / 3.0],
            ]
        )
        torch.testing.assert_close(factors.cpu(), expected, atol=1e-6, rtol=1e-6)

    def test_gpu_factor_structures_encode_train_history_and_course_overlap(self):
        user_items = {0: {0}, 1: {0, 1}}
        item_users = {0: {0, 1}, 1: {1}}

        history = build_history_tensor(
            user_items,
            n_users=2,
            n_items=3,
            device=torch.device("cpu"),
        )
        course_match = build_course_match_table(
            item_users,
            n_users=2,
            n_items=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(history.dtype, torch.bool)
        torch.testing.assert_close(
            history,
            torch.tensor([[True, False, False], [True, True, False]]),
        )
        expected_1_to_0 = (1.0 / (2.0 + 1e-4) + 1.0 / (1.0 + 1e-4)) / 2.0
        self.assertAlmostEqual(float(course_match[1, 0]), expected_1_to_0, places=6)
        self.assertEqual(float(course_match[2].sum()), 0.0)

    def test_vectorized_interest_factors_match_author_reference(self):
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([1, 1])
        user_vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        item_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        user_items = {0: {0}, 1: {1}}
        item_users = {0: {0}, 1: {1}}
        history = build_history_tensor(user_items, 2, 2, torch.device("cpu"))
        course_match = build_course_match_table(item_users, 2, 2, torch.device("cpu"))

        reference = source_interest_factors(
            user_ids,
            item_ids,
            user_vectors,
            item_vectors,
            user_items,
            item_users,
        )
        vectorized = gpu_interest_factors(
            user_ids,
            item_ids,
            user_vectors,
            item_vectors,
            history,
            course_match,
        )

        torch.testing.assert_close(vectorized, reference, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the GPU factor check")
    def test_vectorized_interest_factors_stay_on_gpu(self):
        device = torch.device("cuda")
        user_items = {0: {0}, 1: {1}}
        item_users = {0: {0}, 1: {1}}
        history = build_history_tensor(user_items, 2, 2, device)
        course_match = build_course_match_table(item_users, 2, 2, device)

        factors = gpu_interest_factors(
            torch.tensor([0, 1], device=device),
            torch.tensor([1, 1], device=device),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device),
            history,
            course_match,
        )

        self.assertEqual(factors.device.type, "cuda")

    def test_catalog_interest_factors_match_full_catalog_batch_semantics(self):
        item_ids = torch.tensor([0, 1, 2, 3])
        user_ids = torch.zeros(4, dtype=torch.long)
        user_vectors = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
        item_vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
        user_items = {0: {0, 2}, 1: {1}}
        item_users = {0: {0}, 1: {1}, 2: {0}}

        reference = source_interest_factors(
            user_ids,
            item_ids,
            user_vectors,
            item_vectors,
            user_items,
            item_users,
        )
        optimized = catalog_interest_factors(
            user=0,
            item_ids=item_ids,
            item_vectors=item_vectors,
            user_items=user_items,
            item_users=item_users,
        )

        torch.testing.assert_close(optimized, reference, atol=1e-6, rtol=1e-6)

    def test_device_kgcn_reuses_author_class_and_runs_on_cpu(self):
        source_ngcf, source_kgcn = load_idrmi_source_classes(
            ROOT / "paper_aaai27" / "baseline_sources" / "IDRMI"
        )
        self.assertEqual(source_ngcf.__name__, "NGCF")
        self.assertTrue(issubclass(DeviceKGCN, source_kgcn))

        model = DeviceKGCN(
            n_users=2,
            n_entitys=3,
            n_relations=1,
            adj_entity=np.array([[1], [0], [2]], dtype=np.int64),
            adj_relation=np.zeros((3, 1), dtype=np.int64),
            n_neighbors=1,
            e_dim=2,
            drop_rate=0.0,
        )
        output = model(torch.tensor([0, 1]), torch.tensor([0, 1]), is_evaluate=True)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())

    def test_kg_neighbors_are_deterministic_and_cover_isolated_entities(self):
        triples = [(0, 0, 1), (1, 1, 2)]

        first_entity, first_relation = build_kg_neighbors(
            triples,
            n_entities=4,
            n_neighbors=3,
            rng=np.random.default_rng(9),
        )
        second_entity, second_relation = build_kg_neighbors(
            triples,
            n_entities=4,
            n_neighbors=3,
            rng=np.random.default_rng(9),
        )

        np.testing.assert_array_equal(first_entity, second_entity)
        np.testing.assert_array_equal(first_relation, second_relation)
        self.assertEqual(first_entity.shape, (4, 3))
        self.assertTrue(np.all(first_entity[3] == 3))
        self.assertTrue(np.all(first_relation[3] == 0))

    def test_strict_dataset_loader_uses_only_declared_cold_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_root = root / "split"
            split_root.mkdir()
            pd.DataFrame(
                {"u_idx": [0, 1], "i_idx": [0, 1], "_split_source": ["train", "train"]}
            ).to_pickle(split_root / "static_train.pkl")
            pd.DataFrame(
                {
                    "u_idx": [0, 1],
                    "i_idx": [2, 1],
                    "_split_source": ["strict_item_cold_val", "warm_val"],
                }
            ).to_pickle(split_root / "static_val.pkl")
            pd.DataFrame(
                {"u_idx": [1], "i_idx": [3], "_split_source": ["strict_item_cold_test"]}
            ).to_pickle(split_root / "static_test.pkl")
            kg_path = root / "kg.tsv"
            kg_path.write_text("0\t0\t4\n1\t0\t4\n2\t0\t4\n3\t0\t4\n", encoding="utf-8")

            dataset = load_strict_dataset(split_root, kg_path)

        self.assertEqual(dataset.train_positives, ((0, 0), (1, 1)))
        self.assertEqual(dataset.validation_rows, ((0, 2),))
        self.assertEqual(dataset.test_rows, ((1, 3),))
        self.assertEqual(dataset.cold_items, frozenset({2, 3}))
        self.assertEqual(dataset.n_users, 2)
        self.assertEqual(dataset.n_items, 4)
        self.assertEqual(dataset.n_entities, 5)
        self.assertEqual(dataset.n_relations, 1)

    def test_dataset_loader_rejects_incomplete_expected_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_root = root / "split"
            split_root.mkdir()
            pd.DataFrame({"u_idx": [0], "i_idx": [0]}).to_pickle(split_root / "static_train.pkl")
            pd.DataFrame(
                {"u_idx": [0], "i_idx": [1], "_split_source": ["strict_item_cold_val"]}
            ).to_pickle(split_root / "static_val.pkl")
            pd.DataFrame(
                {"u_idx": [0], "i_idx": [2], "_split_source": ["strict_item_cold_test"]}
            ).to_pickle(split_root / "static_test.pkl")
            kg_path = root / "kg.tsv"
            kg_path.write_text("0\t0\t3\n1\t0\t3\n2\t0\t3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected course catalog"):
                load_strict_dataset(split_root, kg_path, expected_n_items=4)

    def test_labeled_training_rows_are_balanced_and_warm_only(self):
        positives = [(0, 0), (1, 1), (2, 0), (3, 1)]
        structures = build_train_structures(
            [(user, item, 1) for user, item in positives],
            n_users=4,
            n_items=3,
            forbidden_cold_items={2},
        )

        rows = prepare_labeled_training_rows(
            positives,
            structures,
            max_examples=6,
            rng=np.random.default_rng(2025),
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(label for _, _, label in rows), 3)
        self.assertNotIn(2, {item for _, item, _ in rows})
        for user, item, label in rows:
            if label == 0:
                self.assertNotIn(item, structures.user_items[user])

    def test_evaluate_split_scores_full_catalog_once_per_unique_user(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def eval(self):
                return self

            def score_catalog(self, user, item_ids):
                self.calls.append(int(user))
                rows = {
                    0: torch.tensor([0.9, 0.8, 0.7]),
                    1: torch.tensor([0.9, 0.8, 0.95]),
                }
                return rows[int(user)].to(item_ids.device)

        structures = build_train_structures(
            [(0, 0, 1), (1, 0, 1)],
            n_users=2,
            n_items=3,
            forbidden_cold_items={1, 2},
        )
        model = FakeModel()

        report = evaluate_split(
            model,
            rows=((0, 1), (0, 2), (1, 2)),
            structures=structures,
            n_items=3,
            max_users=-1,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.calls, [0, 1])
        self.assertEqual(report["evaluated_users"], 2)
        self.assertEqual(report["candidate_courses"], 3)
        self.assertEqual(report["rows_full_cold"], 3)
        self.assertEqual(report["count_full_cold_item_macro"], 2)
        self.assertGreater(report["score_std"], 0.0)

    def test_strict_model_reuses_source_backbones_and_backpropagates(self):
        structures = build_train_structures(
            [(0, 0, 1), (1, 1, 1)],
            n_users=2,
            n_items=2,
            forbidden_cold_items=set(),
        )
        adjacency_entity = np.array([[1], [0]], dtype=np.int64)
        adjacency_relation = np.zeros((2, 1), dtype=np.int64)
        model = IDRMIStrictModel(
            n_users=2,
            n_items=2,
            n_entities=2,
            n_relations=1,
            norm_adj=structures.norm_adj,
            adj_entity=adjacency_entity,
            adj_relation=adjacency_relation,
            user_items=structures.user_items,
            item_users=structures.item_users,
            embed_dim=2,
            n_neighbors=1,
            batch_size=2,
            device=torch.device("cpu"),
        )

        with mock.patch.object(
            idrmi_adapter,
            "source_interest_factors",
            side_effect=AssertionError("CPU reference factors must not run in the model path"),
        ):
            scores = model(torch.tensor([0, 1]), torch.tensor([0, 1]), training_factors=True)
        loss = torch.nn.functional.binary_cross_entropy(scores, torch.tensor([1.0, 1.0]))
        loss.backward()

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_script_help_exposes_smoke_and_full_run_controls(self):
        script = ROOT / "paper_aaai27" / "scripts" / "idrmi_strict_adapter.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-train-examples", result.stdout)
        self.assertIn("--max-eval-users", result.stdout)

    def test_tiny_end_to_end_run_writes_protocol_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_root = root / "split"
            split_root.mkdir()
            pd.DataFrame(
                {
                    "u_idx": [0, 1, 2, 3],
                    "i_idx": [0, 1, 0, 1],
                    "_split_source": ["train"] * 4,
                }
            ).to_pickle(split_root / "static_train.pkl")
            pd.DataFrame(
                {
                    "u_idx": [0, 1],
                    "i_idx": [2, 2],
                    "_split_source": ["strict_item_cold_val"] * 2,
                }
            ).to_pickle(split_root / "static_val.pkl")
            pd.DataFrame(
                {
                    "u_idx": [2, 3],
                    "i_idx": [3, 3],
                    "_split_source": ["strict_item_cold_test"] * 2,
                }
            ).to_pickle(split_root / "static_test.pkl")
            kg_path = root / "kg.tsv"
            kg_path.write_text(
                "0\t0\t4\n1\t0\t4\n2\t0\t4\n3\t0\t4\n",
                encoding="utf-8",
            )
            out_dir = root / "out"
            args = parse_args(
                [
                    "--split-root",
                    str(split_root),
                    "--kg-path",
                    str(kg_path),
                    "--out-dir",
                    str(out_dir),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--max-train-examples",
                    "8",
                    "--max-eval-users",
                    "-1",
                    "--batch-size",
                    "4",
                    "--embed-dim",
                    "4",
                    "--n-neighbors",
                    "2",
                    "--expected-courses",
                    "4",
                ]
            )

            report = run_idrmi_strict_adapter(args)

            self.assertTrue((out_dir / "idrmi_strict_adapter_report.json").exists())
            self.assertTrue((out_dir / "idrmi_strict_adapter_report.md").exists())
            self.assertEqual(report["protocol"]["candidate_courses"], 4)
            self.assertEqual(report["protocol"]["positive_train_edges"], 4)
            self.assertFalse(report["protocol"]["bitwise_cuda_reproducibility_verified"])
            self.assertEqual(report["validation"]["evaluated_users"], 2)
            self.assertEqual(report["test"]["evaluated_users"], 2)
            self.assertIn("full_cold_item_macro", report["validation"])
            self.assertTrue(report["gates"]["nonempty_adjacency"])
            self.assertTrue(report["gates"]["warm_only_negatives"])
            self.assertTrue(report["gates"]["finite_loss"])

    def test_item_macro_evaluation_masks_history_and_restores_target(self):
        accumulator = ItemMacroRankingAccumulator(k_list=(1, 2), cold_threshold=1)
        scores = np.array(
            [
                [0.95, 0.80, 0.70],
                [0.90, 0.85, 0.95],
                [0.90, 0.80, 0.70],
            ],
            dtype=np.float32,
        )
        examples = [
            {"user": 0, "target": 1, "raw_item": 1, "popularity": 0},
            {"user": 1, "target": 2, "raw_item": 2, "popularity": 0},
            {"user": 2, "target": 1, "raw_item": 1, "popularity": 0},
        ]

        accumulator.add_batch(scores, examples, user_seen_items={0: {0, 1}, 1: {0}, 2: set()})
        report = accumulator.result()

        self.assertEqual(report["count_full_cold_item_macro"], 2)
        self.assertAlmostEqual(report["full_cold_item_macro"]["R@1"], 0.75)
        self.assertAlmostEqual(
            report["full_cold_item_macro"]["N@2"],
            (1.0 + (1.0 + 1.0 / math.log2(3.0)) / 2.0) / 2.0,
        )

    def test_validation_tracker_restores_best_state(self):
        tracker = BestValidationTracker(metric_path="full_cold_item_macro.N@10", patience=1)

        self.assertTrue(tracker.update(1, {"full_cold_item_macro": {"N@10": 0.2}}, {"w": 1}))
        self.assertFalse(tracker.update(2, {"full_cold_item_macro": {"N@10": 0.1}}, {"w": 2}))

        self.assertTrue(tracker.should_stop)
        self.assertEqual(tracker.best_state, {"w": 1})

    def test_cli_defaults_to_seed2025_gpu_smoke(self):
        args = parse_args([])

        self.assertEqual(args.seed, 2025)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_train_examples, 2048)
        self.assertEqual(args.max_eval_users, 64)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.expected_courses, 698)
        self.assertEqual(args.ngcf_device, "same")

    def test_nondefault_source_directory_is_rejected_to_keep_provenance_truthful(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "author source snapshot"):
                resolve_source_dir(Path(tmp))

    def test_reproducibility_configuration_enables_deterministic_algorithms(self):
        configure_reproducibility(2025)

        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
