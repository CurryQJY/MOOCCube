import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")


class AggregateGARColdRecThreeSeedTests(unittest.TestCase):
    def _write_result(self, root: Path, seed: int, cold_n10: float) -> Path:
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        cold_csv = seed_dir / "per_item_full_cold_gar_coldrec.csv"
        hot_csv = seed_dir / "per_item_full_hot_gar_coldrec.csv"
        pd.DataFrame(
            {"item_id": [1, 2], "count": [3, 4], "N@10": [cold_n10, cold_n10]}
        ).to_csv(cold_csv, index=False)
        pd.DataFrame({"item_id": [3], "count": [5], "N@10": [0.2]}).to_csv(
            hot_csv, index=False
        )

        cold_metrics = {metric: cold_n10 for metric in METRICS}
        hot_metrics = {metric: 0.2 for metric in METRICS}
        payload = [
            {
                "model": "GAR-coldrec-source-strict",
                "model_display": "GAR (ColdRec source, strict adapter)",
                "seed": seed,
                "static_seed": seed,
                "official_commit": "18efd24",
                "source_model_unchanged": True,
                "protocol": "static_item_cold_balanced",
                "candidate_mode": "full_catalog",
                "checkpoint_metric": "validation_full_cold_item_macro.N@10",
                "item_macro_metrics": True,
                "train_history_masking": True,
                "train_only_interaction_evidence": True,
                "test_history_policy": "train_only",
                "cuda_used": True,
                "device": "cuda:0",
                "epochs": 500,
                "epochs_ran": 7,
                "best_epoch": 7,
                "best_val_full_cold_item_macro_n10": cold_n10,
                "strict_audit": {
                    "heldout_cold_item_count": 3,
                    "train_overlap_count": 0,
                },
                "counts": {
                    "full_cold": 10,
                    "full_hot": 5,
                    "full_cold_item": 2,
                    "full_hot_item": 1,
                },
                "strict_validation_history": [
                    {
                        "epoch": 7,
                        "item_count": 1,
                        "N@10": cold_n10,
                        "improved": True,
                    }
                ],
                "full_cold": dict(cold_metrics),
                "full_hot": dict(hot_metrics),
                "full_cold_item_macro": dict(cold_metrics),
                "full_hot_item_macro": dict(hot_metrics),
                "per_item_full_cold_path": str(cold_csv.resolve()),
                "per_item_full_hot_path": str(hot_csv.resolve()),
                "coldrec_args": {
                    "model": "GAR",
                    "backbone": "MF",
                    "epochs": 500,
                    "seed": seed,
                    "use_gpu": True,
                },
            }
        ]
        result_path = seed_dir / "gar_coldrec_strict_result.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return result_path

    @staticmethod
    def _load_result(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))[0]

    @staticmethod
    def _save_result(path: Path, result: dict) -> None:
        path.write_text(json.dumps([result], indent=2), encoding="utf-8")

    def test_aggregate_writes_three_seed_mean_and_sample_std(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            out_dir = root / "aggregate"
            for seed, value in zip((2025, 2026, 2027), (0.1, 0.2, 0.3)):
                self._write_result(root, seed, value)

            summary = aggregate(root, (2025, 2026, 2027), out_dir)
            detail = pd.read_csv(out_dir / "gar_coldrec_3seed_detail.csv")

            self.assertEqual(summary["runs"], 3)
            self.assertEqual(summary["seeds"], [2025, 2026, 2027])
            stats = summary["full_cold_item_macro"]["N@10"]
            self.assertTrue(math.isclose(stats["mean"], 0.2))
            self.assertTrue(math.isclose(stats["std"], 0.1))
            self.assertEqual(detail["seed"].tolist(), [2025, 2026, 2027])
            for filename in (
                "gar_coldrec_3seed_detail.csv",
                "gar_coldrec_3seed_summary.csv",
                "gar_coldrec_3seed_summary.json",
                "gar_coldrec_3seed_report.md",
            ):
                self.assertTrue((out_dir / filename).is_file(), filename)

    def test_aggregate_rejects_missing_seed(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, 2025, 0.1)
            self._write_result(root, 2026, 0.2)
            with self.assertRaisesRegex(FileNotFoundError, "seed 2027"):
                aggregate(root, (2025, 2026, 2027), root / "aggregate")

    def test_aggregate_rejects_nonzero_train_overlap(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._write_result(root, seed, 0.1) for seed in (2025, 2026, 2027)]
            result = self._load_result(paths[1])
            result["strict_audit"]["train_overlap_count"] = 1
            self._save_result(paths[1], result)
            with self.assertRaisesRegex(ValueError, "overlap"):
                aggregate(root, (2025, 2026, 2027), root / "aggregate")

    def test_aggregate_rejects_missing_protocol_gate(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._write_result(root, seed, 0.1) for seed in (2025, 2026, 2027)]
            result = self._load_result(paths[0])
            del result["protocol"]
            self._save_result(paths[0], result)
            with self.assertRaisesRegex(ValueError, "protocol"):
                aggregate(root, (2025, 2026, 2027), root / "aggregate")

    def test_aggregate_rejects_nonfinite_metric(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._write_result(root, seed, 0.1) for seed in (2025, 2026, 2027)]
            result = self._load_result(paths[2])
            result["full_cold_item_macro"]["N@10"] = float("nan")
            self._save_result(paths[2], result)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                aggregate(root, (2025, 2026, 2027), root / "aggregate")

    def test_aggregate_rejects_per_course_row_count_mismatch(self):
        from aggregate_gar_coldrec_3seed import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._write_result(root, seed, 0.1) for seed in (2025, 2026, 2027)]
            result = self._load_result(paths[0])
            result["counts"]["full_cold_item"] = 3
            self._save_result(paths[0], result)
            with self.assertRaisesRegex(ValueError, "row-count mismatch"):
                aggregate(root, (2025, 2026, 2027), root / "aggregate")


if __name__ == "__main__":
    unittest.main()
