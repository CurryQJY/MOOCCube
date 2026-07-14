import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.course_baseline_adaptability import (
    TinyAdapterInput,
    dataframe_pairs,
    parse_targets,
    safe_token,
    write_msec_smoke_dataset,
    write_pcgnn_atomic_dataset,
    write_upgpr_processed_dataset,
)


def tiny_input() -> TinyAdapterInput:
    return TinyAdapterInput(
        n_users=3,
        n_items=4,
        train_pairs=[(0, 0), (0, 1), (1, 1)],
        val_pairs=[(1, 2)],
        test_pairs=[(2, 3)],
        course_tokens=["course0", "course1", "course2", "course3"],
        course_concepts={
            0: ["math", "logic"],
            1: ["logic"],
            2: [],
            3: ["ai"],
        },
        course_teachers={0: ["teacher0"], 2: ["teacher1"]},
        course_schools={0: "school0", 1: "school0", 2: "school1", 3: "school1"},
        course_videos={0: ["video0"], 1: ["video1"], 3: ["video2"]},
        user_videos={0: ["video0", "video1"], 1: ["video1"], 2: ["video2"]},
        video_concepts={"video0": ["math"], "video1": ["logic"], "video2": ["ai"]},
        kg_triples=[("course0", "after", "course1"), ("course3", "concept", "ai")],
    )


class CourseBaselineAdaptabilityTests(unittest.TestCase):
    def test_dataframe_pairs_negative_limit_means_uncapped(self):
        import pandas as pd

        df = pd.DataFrame({"u_idx": [1, 2, 3], "i_idx": [10, 20, 30]})

        self.assertEqual(dataframe_pairs(df, -1), [(1, 10), (2, 20), (3, 30)])

    def test_parse_targets_supports_pcgnn_only_and_all(self):
        self.assertEqual(parse_targets("pcgnn"), {"pcgnn"})
        self.assertEqual(parse_targets("all"), {"pcgnn", "upgpr", "msec"})
        self.assertEqual(parse_targets("pcgnn,upgpr"), {"pcgnn", "upgpr"})

    def test_safe_token_makes_non_ascii_loader_safe(self):
        token = safe_token("机器 学习")
        self.assertTrue(token.startswith("tok_"))
        token.encode("ascii")
        self.assertNotIn(" ", token)

    def test_write_pcgnn_atomic_dataset_uses_benchmark_split_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = write_pcgnn_atomic_dataset(tmp_path, "tiny_pcgnn", tiny_input())

            dataset_dir = tmp_path / "dataset" / "tiny_pcgnn"
            config_path = tmp_path / "recbole_tiny_pcgnn.yaml"

            self.assertEqual(report["dataset_name"], "tiny_pcgnn")
            self.assertEqual(
                (dataset_dir / "tiny_pcgnn.train.inter").read_text().splitlines(),
                [
                    "user_id:token\titem_id:token\trating:float\ttimestamp:float",
                    "0\t0\t1\t1",
                    "0\t1\t1\t2",
                    "1\t1\t1\t3",
                ],
            )
            self.assertEqual(
                (dataset_dir / "tiny_pcgnn.valid.inter").read_text().splitlines()[1:],
                ["1\t2\t1\t4"],
            )
            self.assertEqual(
                (dataset_dir / "tiny_pcgnn.test.inter").read_text().splitlines()[1:],
                ["2\t3\t1\t5"],
            )
            self.assertEqual(len((dataset_dir / "tiny_pcgnn.item").read_text().splitlines()), 5)
            self.assertEqual(len((dataset_dir / "tiny_pcgnn.link").read_text().splitlines()), 5)
            kg_text = (dataset_dir / "tiny_pcgnn.kg").read_text()
            self.assertIn("\titem_category\tmath", kg_text)
            self.assertIn("\titem_category\tai", kg_text)
            config_text = config_path.read_text()
            self.assertIn("benchmark_filename: ['train', 'valid', 'test']", config_text)
            self.assertIn("eval_setting: TO_LS,full", config_text)
            self.assertIn("inter: [user_id, item_id, timestamp]", config_text)
            self.assertNotIn("inter: [user_id, item_id, rating, timestamp]", config_text)

    def test_write_upgpr_processed_dataset_preserves_strict_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = write_upgpr_processed_dataset(tmp_path, tiny_input())

            self.assertEqual(report["train_rows"], 3)
            self.assertEqual(report["validation_rows"], 1)
            self.assertEqual(report["test_rows"], 1)
            self.assertEqual((tmp_path / "train.txt").read_text().splitlines(), ["0 0", "0 1", "1 1"])
            self.assertEqual((tmp_path / "validation.txt").read_text().splitlines(), ["1 2"])
            self.assertEqual((tmp_path / "test.txt").read_text().splitlines(), ["2 3"])
            self.assertEqual(
                (tmp_path / "enrolments.txt").read_text().splitlines(),
                ["0 0", "0 1", "1 1"],
            )
            self.assertEqual(len((tmp_path / "courses.txt").read_text().splitlines()), 4)
            self.assertEqual(len((tmp_path / "course_concepts.txt").read_text().splitlines()), 4)
            self.assertEqual(len((tmp_path / "course_teachers.txt").read_text().splitlines()), 4)
            self.assertEqual(len((tmp_path / "course_school.txt").read_text().splitlines()), 4)

    def test_write_msec_smoke_dataset_exports_expected_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = write_msec_smoke_dataset(tmp_path, tiny_input())

            self.assertEqual(report["n_users"], 3)
            self.assertEqual(report["n_items"], 4)
            self.assertEqual(np.load(tmp_path / "train_uc.npy").shape, (3, 4))
            self.assertEqual(int(np.load(tmp_path / "val_uc.npy").sum()), 1)
            self.assertEqual(np.load(tmp_path / "ck.npy").shape[0], 4)
            self.assertEqual(np.load(tmp_path / "train_uv.npy").shape[0], 3)
            self.assertEqual(np.load(tmp_path / "course_video.npy").shape[0], 4)
            self.assertEqual(np.load(tmp_path / "video_concept.npy").shape[1], report["n_concepts"])


if __name__ == "__main__":
    unittest.main()
