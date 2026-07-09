import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.priority_baseline_experiments import (  # noqa: E402
    build_strict_sequence_examples,
    compute_relation_reachability,
    pcgnn_smoke_config_overrides,
)


class PriorityBaselineExperimentTests(unittest.TestCase):
    def test_build_strict_sequence_examples_uses_train_history_for_eval(self):
        train_rows = [
            {"u_idx": 10, "i_idx": 1, "timestamp": 1},
            {"u_idx": 10, "i_idx": 2, "timestamp": 2},
            {"u_idx": 20, "i_idx": 3, "timestamp": 1},
        ]
        eval_rows = [
            {"u_idx": 10, "i_idx": 4, "timestamp": 3},
            {"u_idx": 20, "i_idx": 5, "timestamp": 4},
            {"u_idx": 30, "i_idx": 6, "timestamp": 5},
        ]
        token_map = {"1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16}

        examples = build_strict_sequence_examples(train_rows, eval_rows, token_map, max_len=3)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["user"], 10)
        self.assertEqual(examples[0]["history"], [11, 12])
        self.assertEqual(examples[0]["target"], 14)
        self.assertEqual(examples[1]["history"], [13])
        self.assertEqual(examples[1]["target"], 15)

    def test_compute_relation_reachability_counts_targets_reached_from_history(self):
        user_history = {0: {1}, 1: {2}, 2: set()}
        eval_pairs = [(0, 3), (1, 4), (2, 5)]
        item_relations = {
            1: {"concept": {"a"}, "teacher": {"t1"}},
            2: {"concept": {"b"}, "teacher": set()},
            3: {"concept": {"a"}, "teacher": set()},
            4: {"concept": {"c"}, "teacher": {"t2"}},
            5: {"concept": {"a"}, "teacher": {"t1"}},
        }

        report = compute_relation_reachability(user_history, eval_pairs, item_relations)

        self.assertEqual(report["eval_pairs"], 3)
        self.assertEqual(report["with_train_history"], 2)
        self.assertEqual(report["target_reachable"], 1)
        self.assertAlmostEqual(report["target_reachable_rate"], 0.5)

    def test_pcgnn_smoke_config_forces_cpu_execution(self):
        overrides = pcgnn_smoke_config_overrides(train_batch_size=16, eval_batch_size=32)

        self.assertEqual(overrides["device"], "cpu")
        self.assertFalse(overrides["use_gpu"])
        self.assertEqual(overrides["gpu_id"], -1)
        self.assertEqual(overrides["train_batch_size"], 16)
        self.assertEqual(overrides["eval_batch_size"], 32)


if __name__ == "__main__":
    unittest.main()
