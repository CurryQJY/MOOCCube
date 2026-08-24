import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.export_stable_epoch_timing import (
    parse_indexed_epoch_times,
    render_fragment,
    summarize_seed_windows,
)


class StableEpochTimingExportTests(unittest.TestCase):
    def test_summary_timer_overrides_progress_timer_for_the_same_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "[CGRC-TRAIN-PROGRESS] Epoch 5/50 | 10/10 (100%) | "
                "avg_loss=1.0 | elapsed=1m10s | eta=0s\n"
                "[CGRC-TRAIN] Epoch 5/50 Time: 70.50s\n",
                encoding="utf-8",
            )

            self.assertEqual(parse_indexed_epoch_times([log_path]), {5: 70.50})

    def test_window_summary_merges_resume_logs_and_aggregates_seed_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "seed_2025_first.log"
            resumed = tmp_path / "seed_2025_resumed.log"
            second = tmp_path / "seed_2026.log"

            first.write_text(
                "\n".join(
                    f"[CGRC-TRAIN] Epoch {epoch}/50 Time: {float(epoch):.1f}s"
                    for epoch in range(5, 10)
                ),
                encoding="utf-8",
            )
            resumed.write_text(
                "\n".join(
                    f"[CGRC-TRAIN] Epoch {epoch}/50 Time: {float(epoch):.1f}s"
                    for epoch in range(10, 16)
                ),
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(
                    f"[CGRC-TRAIN] Epoch {epoch}/50 Time: {float(epoch + 10):.1f}s"
                    for epoch in range(5, 16)
                ),
                encoding="utf-8",
            )

            details, summary = summarize_seed_windows(
                {2025: [first, resumed], 2026: [second]},
                window_start=5,
                window_end=15,
            )

            self.assertEqual([row["seed"] for row in details], [2025, 2026])
            self.assertAlmostEqual(details[0]["mean_train_time_s_per_epoch"], 10.0)
            self.assertAlmostEqual(details[1]["mean_train_time_s_per_epoch"], 20.0)
            self.assertEqual(summary["seed_count"], 2)
            self.assertAlmostEqual(summary["mean_train_time_s_per_epoch"], 15.0)
            self.assertAlmostEqual(
                summary["std_train_time_s_per_epoch_across_seed_means"],
                7.0710678118654755,
            )

    def test_table_note_keeps_a_space_after_the_standard_deviation_symbol(self):
        fragment = render_fragment([])

        self.assertIn(r"mean\(\pm\) sample std", fragment)


if __name__ == "__main__":
    unittest.main()
