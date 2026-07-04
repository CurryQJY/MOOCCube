from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class PlotMooccubePaperHparamSensitivityTest(unittest.TestCase):
    def test_completed_hparam_figure_includes_wide_multiseed_points(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "figures"

            cmd = [
                sys.executable,
                str(repo / "plot_mooccube_paper_hparam_sensitivity.py"),
                "--repo",
                str(repo),
                "--wide-root",
                "outputs/content_delta_pop5/course_hparam_wide_seed2025",
                "--out-dir",
                str(out_dir),
                "--style",
                "line-grid",
            ]
            result = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            summary_path = out_dir / "mooccube_hparam_sensitivity_paper_complete_summary.csv"
            self.assertTrue(summary_path.exists())
            summary = pd.read_csv(summary_path)

            expected = {
                "reward_scale_0p75",
                "reward_scale_1p25",
                "prereq_gate_0p70",
            }
            self.assertTrue(expected.issubset(set(summary["variant"])))
            self.assertNotIn("term_norm_batch", set(summary["variant"]))

            for variant in expected:
                rows = summary.loc[summary["variant"] == variant]
                self.assertFalse(rows.empty)
                self.assertEqual(set(rows["n"].astype(int)), {3})
                self.assertEqual(set(rows["seeds"].astype(str)), {"2025,2026,2027"})

            self.assertTrue((out_dir / "mooccube_hparam_sensitivity_paper_complete_linegrid.svg").exists())
            self.assertTrue((out_dir / "mooccube_hparam_sensitivity_paper_complete_linegrid.pdf").exists())

            svg_text = (out_dir / "mooccube_hparam_sensitivity_paper_complete_linegrid.svg").read_text(
                encoding="utf-8"
            )
            self.assertIn("Recall@10", svg_text)
            self.assertIn("NDCG@10", svg_text)
            self.assertNotIn("Term Normalization", svg_text)
            self.assertNotIn("broken-y-axis", svg_text)


if __name__ == "__main__":
    unittest.main()
