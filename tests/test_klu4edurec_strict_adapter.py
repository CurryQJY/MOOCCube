import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.klu4edurec_strict_adapter import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    ItemMacroRankingAccumulator,
    KLU4EduRecStrictModel,
    build_train_structures,
    evaluate_split,
    load_author_model_class,
    load_strict_dataset,
    parse_args,
    prepare_epoch_triples,
    resolve_source_dir,
    run_klu4edurec_strict_adapter,
    sample_warm_negatives,
)


class KLU4EduRecStrictAdapterTests(unittest.TestCase):
    @staticmethod
    def _write_tiny_split(root: Path, leak_cold: bool = False) -> None:
        root.mkdir(parents=True, exist_ok=True)
        train_items = [0, 2] if leak_cold else [0, 1]
        pd.DataFrame(
            {
                "u_idx": [0, 1],
                "i_idx": train_items,
                "_split_source": ["strict_item_cold_train"] * 2,
            }
        ).to_pickle(root / "static_train.pkl")
        pd.DataFrame(
            {
                "u_idx": [0, 1],
                "i_idx": [2, 2],
                "_split_source": ["strict_item_cold_val"] * 2,
            }
        ).to_pickle(root / "static_val.pkl")
        pd.DataFrame(
            {
                "u_idx": [0, 1],
                "i_idx": [3, 3],
                "_split_source": ["strict_item_cold_test"] * 2,
            }
        ).to_pickle(root / "static_test.pkl")

    @staticmethod
    def _write_content(path: Path) -> torch.Tensor:
        content = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        torch.save(content, path)
        return content

    def test_load_strict_dataset_uses_only_cold_validation_and_test_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_tiny_split(root)
            content_path = root / "content.pt"
            content = self._write_content(content_path)

            dataset = load_strict_dataset(root, content_path, expected_n_items=4)

        self.assertEqual(dataset.train_positives, ((0, 0), (1, 1)))
        self.assertEqual(dataset.validation_rows, ((0, 2), (1, 2)))
        self.assertEqual(dataset.test_rows, ((0, 3), (1, 3)))
        self.assertEqual(dataset.cold_items, frozenset({2, 3}))
        self.assertEqual((dataset.n_users, dataset.n_items), (2, 4))
        torch.testing.assert_close(dataset.content_embeddings, content)

    def test_load_strict_dataset_rejects_positive_cold_train_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_tiny_split(root, leak_cold=True)
            content_path = root / "content.pt"
            self._write_content(content_path)

            with self.assertRaisesRegex(ValueError, "cold course"):
                load_strict_dataset(root, content_path, expected_n_items=4)

    def test_train_graph_is_bidirectional_and_train_only(self):
        structures = build_train_structures(
            positives=[(0, 0), (1, 1)],
            n_users=2,
            n_items=4,
            forbidden_cold_items={2, 3},
        )

        self.assertEqual(structures.positive_edges, 2)
        self.assertEqual(structures.warm_items, (0, 1))
        self.assertEqual(structures.user_items, {0: {0}, 1: {1}})
        edges = {tuple(edge) for edge in structures.edge_index.t().tolist()}
        self.assertEqual(edges, {(0, 2), (2, 0), (1, 3), (3, 1)})
        self.assertFalse(any(node >= 4 for edge in edges for node in edge))

    def test_warm_negative_sampling_excludes_seen_and_cold_courses(self):
        negatives = sample_warm_negatives(
            positives=[(0, 0), (1, 1), (0, 0), (1, 1)],
            warm_items=(0, 1),
            user_items={0: {0}, 1: {1}},
            rng=np.random.default_rng(2025),
        )

        self.assertEqual(negatives, [(0, 1), (1, 0), (0, 1), (1, 0)])
        self.assertNotIn(2, {item for _, item in negatives})

    def test_epoch_triples_are_capped_and_use_warm_negatives(self):
        structures = build_train_structures(
            positives=[(0, 0), (1, 1), (2, 0)],
            n_users=3,
            n_items=4,
            forbidden_cold_items={2, 3},
        )

        triples = prepare_epoch_triples(
            positives=[(0, 0), (1, 1), (2, 0)],
            structures=structures,
            max_examples=2,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(len(triples), 2)
        for user, positive, negative in triples:
            self.assertIn((user, positive), {(0, 0), (1, 1), (2, 0)})
            self.assertIn(negative, structures.warm_items)
            self.assertNotIn(negative, structures.user_items[user])

    def test_source_directory_is_locked_to_audited_snapshot(self):
        self.assertEqual(resolve_source_dir(DEFAULT_SOURCE_DIR), DEFAULT_SOURCE_DIR.resolve())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "locked"):
                resolve_source_dir(Path(tmp))

    def test_author_model_class_loads_from_source_snapshot(self):
        author_class = load_author_model_class(DEFAULT_SOURCE_DIR)

        self.assertEqual(author_class.__name__, "LLM4EduRec")
        self.assertIn("klu_author", author_class.__module__)

    def test_cold_course_score_changes_when_only_its_semantics_change(self):
        torch.manual_seed(11)
        structures = build_train_structures(
            positives=[(0, 0), (1, 1)],
            n_users=2,
            n_items=4,
            forbidden_cold_items={2, 3},
        )
        content = torch.eye(4, dtype=torch.float32)
        model = KLU4EduRecStrictModel(
            n_users=2,
            n_items=4,
            edge_index=structures.edge_index,
            content_embeddings=content,
            embed_dim=4,
            n_layers=1,
            edge_drop=0.0,
            item_temperature=0.1,
            item_loss_reg=1e-4,
            weight_decay=5e-5,
            device=torch.device("cpu"),
        )
        model.eval()
        model.prepare_catalog_scoring()
        before = float(model.score_catalog(0, torch.tensor([2]))[0])
        model.clear_catalog_scoring()

        with torch.no_grad():
            model.author_model.item_LLM_embedding.weight[2].copy_(torch.tensor([3.0, -2.0, 1.0, 4.0]))
        model.prepare_catalog_scoring()
        after = float(model.score_catalog(0, torch.tensor([2]))[0])
        model.clear_catalog_scoring()

        self.assertNotAlmostEqual(before, after, places=6)

    def test_wrapper_restores_pretrained_semantics_after_author_initialization(self):
        structures = build_train_structures(
            positives=[(0, 0), (1, 1)],
            n_users=2,
            n_items=4,
            forbidden_cold_items={2, 3},
        )
        content = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        model = KLU4EduRecStrictModel(
            n_users=2,
            n_items=4,
            edge_index=structures.edge_index,
            content_embeddings=content,
            embed_dim=4,
            n_layers=1,
            edge_drop=0.0,
            item_temperature=0.1,
            item_loss_reg=1e-4,
            weight_decay=5e-5,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(model.author_model.item_LLM_embedding.weight.cpu(), content)
        self.assertFalse(model.author_model.item_LLM_embedding.weight.requires_grad)

    def test_item_macro_evaluation_masks_train_history_and_restores_target(self):
        accumulator = ItemMacroRankingAccumulator(k_list=(1, 2), cold_threshold=1)
        scores = np.asarray([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]], dtype=np.float32)
        examples = [
            {"user": 0, "target": 1, "raw_item": 1, "popularity": 0},
            {"user": 1, "target": 0, "raw_item": 0, "popularity": 0},
        ]

        accumulator.add_batch(scores, examples, user_seen_items={0: {0}, 1: {0}})
        report = accumulator.result()

        self.assertEqual(report["count_full_cold_item_macro"], 2)
        self.assertAlmostEqual(report["full_cold_item_macro"]["R@1"], 1.0)
        self.assertAlmostEqual(report["full_cold_item_macro"]["N@2"], 1.0)

    def test_tiny_end_to_end_run_records_item_semantic_scope_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_root = root / "split"
            self._write_tiny_split(split_root)
            content_path = root / "content.pt"
            self._write_content(content_path)
            out_dir = root / "out"
            args = parse_args(
                [
                    "--device",
                    "cpu",
                    "--split-root",
                    str(split_root),
                    "--content-path",
                    str(content_path),
                    "--expected-courses",
                    "4",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--batch-size",
                    "2",
                    "--embed-dim",
                    "4",
                    "--n-layers",
                    "1",
                    "--edge-drop",
                    "0",
                    "--max-train-examples",
                    "2",
                    "--max-eval-users",
                    "-1",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            report = run_klu4edurec_strict_adapter(args)
            saved = json.loads((out_dir / "klu4edurec_strict_adapter_report.json").read_text())

        self.assertEqual(report["model"], "KLU4EduRec-item_se (author source, strict adapter)")
        self.assertEqual(report["source_commit"], "57686b10c7a1d179ec9f6831a306b6d80b9f7b02")
        self.assertEqual(report["status"], "smoke_passed")
        self.assertEqual(report["protocol"]["source_mode"], "item_se")
        self.assertFalse(report["protocol"]["full_model_claimed"])
        self.assertTrue(report["protocol"]["pretrained_semantics_restored_after_author_init"])
        self.assertTrue(report["protocol"]["full_catalog_ranking"])
        self.assertTrue(report["protocol"]["item_macro"])
        self.assertEqual(report["protocol"]["candidate_courses"], 4)
        self.assertEqual(saved["best_epoch"], 1)
        self.assertTrue(all(report["gates"].values()))


if __name__ == "__main__":
    unittest.main()
