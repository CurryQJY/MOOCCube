import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.build_revision_tables import (
    load_main_seed_values,
    load_runtime_profiles,
    parse_epoch_times,
    render_efficiency_standalone,
    render_efficiency_table,
    summarize_cost,
    summarize_seed_ci,
    write_efficiency_tex_exports,
)
from paper_aaai27.scripts.export_efficiency_table import (
    cleanup_latex_intermediates,
    latexmk_command,
)


def sample_cost_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "MOOCCube",
                "cost_ref": "CGRC",
                "ckg_train_epoch_mean_s": 95.6,
                "ckg_train_epoch_std_s": 21.9,
                "baseline_train_epoch_mean_s": float("nan"),
                "baseline_train_epoch_std_s": float("nan"),
                "baseline_train_estimated": False,
                "ckg_infer_mean_s": 18.0,
                "ckg_infer_std_s": 0.7,
                "baseline_infer_mean_s": float("nan"),
                "baseline_infer_std_s": float("nan"),
                "cost_ref_diff_N@10_mean": 0.0263,
                "cost_ref_diff_N@10_ci_low": 0.0114,
                "cost_ref_diff_N@10_ci_high": 0.0401,
            },
            {
                "dataset": "Junyi",
                "cost_ref": "CGRC",
                "ckg_train_epoch_mean_s": 312.2,
                "ckg_train_epoch_std_s": 67.7,
                "baseline_train_epoch_mean_s": 333.0,
                "baseline_train_epoch_std_s": 45.0,
                "baseline_train_estimated": True,
                "ckg_infer_mean_s": 56.4,
                "ckg_infer_std_s": 0.3,
                "baseline_infer_mean_s": 69.2,
                "baseline_infer_std_s": 0.2,
                "cost_ref_diff_N@10_mean": 0.0330,
                "cost_ref_diff_N@10_ci_low": 0.0159,
                "cost_ref_diff_N@10_ci_high": 0.0512,
            },
            {
                "dataset": "COCO",
                "cost_ref": "CGRC",
                "ckg_train_epoch_mean_s": 279.2,
                "ckg_train_epoch_std_s": 17.8,
                "baseline_train_epoch_mean_s": 389.7,
                "baseline_train_epoch_std_s": 80.0,
                "baseline_train_estimated": False,
                "ckg_infer_mean_s": 9.8,
                "ckg_infer_std_s": 0.1,
                "baseline_infer_mean_s": 11.8,
                "baseline_infer_std_s": 0.2,
                "cost_ref_diff_N@10_mean": 0.0183,
                "cost_ref_diff_N@10_ci_low": 0.0162,
                "cost_ref_diff_N@10_ci_high": 0.0205,
            },
        ]
    )


class EfficiencyTableExportTests(unittest.TestCase):
    def test_epoch_parser_deduplicates_progress_and_summary_timers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "  [CGRC-TRAIN-PROGRESS] Epoch 1/50 | 114/114 (100%) | "
                "avg_loss=8.8838 | elapsed=1m31s | eta=0s\n"
                "[CGRC-TRAIN] Epoch 1/50 Time: 91.49s\n",
                encoding="utf-8",
            )

            times = parse_epoch_times([log_path])

            self.assertEqual(times, [91.49])

    def test_renderer_returns_only_the_efficiency_table(self):
        latex = render_efficiency_table(sample_cost_summary())

        self.assertEqual(latex.count(r"\begin{table*}"), 1)
        self.assertEqual(latex.count(r"\end{table*}"), 1)
        self.assertIn(r"\label{tab:efficiency-aaai}", latex)
        self.assertIn("MOOCCube", latex)
        self.assertIn("Junyi", latex)
        self.assertIn("COCO", latex)
        self.assertIn("--", latex)
        self.assertNotIn("Stability, Per-Course Gain", latex)
        self.assertNotIn("supp-significance-tests", latex)

    def test_renderer_uses_unmarked_projection_and_concise_caption(self):
        latex = render_efficiency_table(sample_cost_summary())

        self.assertIn(r"333.0\,$\pm$\,45.0", latex)
        self.assertNotIn(r"\approx", latex)
        self.assertNotIn(r"\dagger", latex)
        self.assertIn(r"\caption{Efficiency comparison with CGRC.}", latex)
        self.assertNotIn("provisional projection", latex)
        self.assertIn("paired course-level difference", latex)
        self.assertIn("conditional on the three trained model fits", latex)
        self.assertIn(r"\(\Delta\)N@10 vs CGRC [95\% CI]", latex)
        self.assertIn("+0.0263 [0.0114, 0.0401]", latex)
        self.assertIn("+0.0330 [0.0159, 0.0512]", latex)
        self.assertIn("+0.0183 [0.0162, 0.0205]", latex)
        self.assertNotIn("[+0.0114", latex)

    def test_renderer_matches_main_paper_aaai_table_order(self):
        latex = render_efficiency_table(sample_cost_summary())

        tabular_end = latex.index(r"\end{tabular*}")
        caption = latex.index(r"\caption{")
        label = latex.index(r"\label{tab:efficiency-aaai}")
        table_end = latex.index(r"\end{table*}")
        self.assertLess(tabular_end, caption)
        self.assertLess(caption, label)
        self.assertLess(label, table_end)
        self.assertIn(r"\begin{table*}[t]", latex)
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\bottomrule", latex)

    def test_retained_mooccube_cgrc_inference_profiles_cover_three_seeds(self):
        profiles = load_runtime_profiles()
        retained = profiles[
            profiles["dataset"].eq("MOOCCube")
            & profiles["method"].eq("CGRC")
            & profiles["source_file"].str.contains(
                "p1_motivation_cgrc_main_table_reproduction", na=False
            )
        ].sort_values("seed")

        self.assertEqual(retained["seed"].tolist(), [2025, 2026, 2027])
        self.assertEqual(
            retained["infer_s"].round(4).tolist(), [21.9532, 22.6796, 22.0849]
        )

    def test_mooccube_cgrc_training_uses_two_retained_seed_logs(self):
        seed_values = load_main_seed_values()
        cost = summarize_cost(summarize_seed_ci(seed_values), seed_values)
        row = cost[cost["dataset"].eq("MOOCCube")].iloc[0]

        self.assertEqual(int(row["baseline_train_epoch_n"]), 100)
        self.assertAlmostEqual(row["baseline_train_epoch_mean_s"], 70.5324, places=4)
        self.assertAlmostEqual(row["baseline_train_epoch_std_s"], 4.5641, places=4)
        self.assertIn("two retained seeds", row["coverage"])

    def test_junyi_cgrc_training_uses_projection_until_three_seeds_finish(self):
        seed_values = load_main_seed_values()
        cost = summarize_cost(summarize_seed_ci(seed_values), seed_values)
        row = cost[cost["dataset"].eq("Junyi")].iloc[0]

        self.assertEqual(int(row["baseline_train_epoch_n"]), 0)
        self.assertEqual(row["baseline_train_epoch_mean_s"], 333.0)
        self.assertEqual(row["baseline_train_epoch_std_s"], 45.0)
        self.assertTrue(bool(row["baseline_train_estimated"]))
        self.assertIn("provisional", row["coverage"])

    def test_efficiency_gain_uses_course_level_bootstrap_for_all_datasets(self):
        seed_values = load_main_seed_values()
        cost = summarize_cost(summarize_seed_ci(seed_values), seed_values).set_index(
            "dataset"
        )
        expected = {
            "MOOCCube": (204, 0.026301, 0.011410, 0.040131),
            "Junyi": (213, 0.032954, 0.015933, 0.051229),
            "COCO": (2459, 0.018315, 0.016157, 0.020509),
        }

        for dataset, (n_pairs, mean, low, high) in expected.items():
            row = cost.loc[dataset]
            self.assertEqual(int(row["cost_ref_diff_N@10_n_pairs"]), n_pairs)
            self.assertAlmostEqual(row["cost_ref_diff_N@10_mean"], mean, places=6)
            self.assertAlmostEqual(row["cost_ref_diff_N@10_ci_low"], low, places=6)
            self.assertAlmostEqual(row["cost_ref_diff_N@10_ci_high"], high, places=6)

    def test_runtime_profiles_have_one_row_per_dataset_method_seed(self):
        profiles = load_runtime_profiles()

        self.assertFalse(
            profiles.duplicated(["dataset", "method", "seed"]).any(),
            profiles[profiles.duplicated(["dataset", "method", "seed"], keep=False)],
        )

    def test_writer_exports_fragment_and_standalone_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fragment_path, standalone_path = write_efficiency_tex_exports(
                sample_cost_summary(), Path(tmp)
            )

            self.assertEqual(fragment_path.name, "efficiency_table_aaai.tex")
            self.assertEqual(
                standalone_path.name, "efficiency_table_aaai_standalone.tex"
            )
            self.assertTrue(fragment_path.exists())
            self.assertTrue(standalone_path.exists())

            fragment = fragment_path.read_text(encoding="utf-8")
            standalone = standalone_path.read_text(encoding="utf-8")
            self.assertIn(r"\label{tab:efficiency-aaai}", fragment)
            self.assertNotIn(r"\documentclass", fragment)
            self.assertIn(
                r"\documentclass[varwidth=20cm,border=8pt]{standalone}", standalone
            )
            self.assertIn(r"\begin{tabular*}{\linewidth}", standalone)
            self.assertIn("paired course-level difference", standalone)
            self.assertIn("conditional on the three trained model fits", standalone)
            self.assertEqual(
                sum(line.startswith("MOOCCube &") for line in standalone.splitlines()),
                1,
            )
            self.assertEqual(standalone.count(r"\begin{tabular*}"), 1)

    def test_latexmk_command_compiles_the_standalone_source(self):
        command = latexmk_command(Path("efficiency_table_aaai_standalone.tex"))

        self.assertEqual(command[0], "latexmk")
        self.assertIn("-pdf", command)
        self.assertIn("-halt-on-error", command)
        self.assertEqual(command[-1], "efficiency_table_aaai_standalone.tex")

    def test_standalone_dataset_rows_end_with_latex_line_breaks(self):
        latex = render_efficiency_standalone(sample_cost_summary())
        dataset_rows = [
            line
            for line in latex.splitlines()
            if line.startswith(("MOOCCube &", "Junyi &", "COCO &"))
        ]

        self.assertEqual(len(dataset_rows), 3)
        self.assertTrue(all(line.endswith(r"\\") for line in dataset_rows))

    def test_standalone_matches_main_table_number_and_centers_caption(self):
        latex = render_efficiency_standalone(sample_cost_summary())

        self.assertIn(
            r"\captionsetup{justification=centering,singlelinecheck=false}", latex
        )
        counter = latex.index(r"\setcounter{table}{3}")
        caption = latex.index(
            r"\captionof{table}{Efficiency comparison with CGRC.}"
        )
        self.assertLess(counter, caption)

    def test_cleanup_removes_only_transient_latex_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "efficiency_table_aaai_standalone.tex"
            source.write_text("source", encoding="utf-8")
            transient = [
                source.with_suffix(".aux"),
                source.with_suffix(".fls"),
                source.with_suffix(".fdb_latexmk"),
            ]
            log_path = source.with_suffix(".log")
            for path in [*transient, log_path]:
                path.write_text("build", encoding="utf-8")

            cleanup_latex_intermediates(source)

            self.assertTrue(source.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(all(not path.exists() for path in transient))

    def test_supplement_runtime_table_matches_retained_mooccube_cgrc_timing(self):
        supplement = (ROOT / "paper_aaai27/supplement_tables.tex").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            r"MOOCCube & CGRC & \(70.5\pm4.6\) & \(22.2\pm0.4\)", supplement
        )
        self.assertRegex(supplement, r"inference comparison\s+on all three datasets")
        self.assertNotIn("MOOCCube & CGRC & -- & --", supplement)
        self.assertIn(
            r"Junyi & CGRC & \(333.0\pm45.0\)",
            supplement,
        )
        self.assertNotIn(r"\approx 333.0", supplement)
        self.assertIn("provisional projection", supplement)

    def test_main_paper_includes_and_references_efficiency_table(self):
        main = (ROOT / "paper_aaai27/main.tex").read_text(encoding="utf-8")

        self.assertIn(r"Table~\ref{tab:efficiency-aaai}", main)
        self.assertIn(r"\input{efficiency_table_aaai.tex}", main)

        first_reference_order = []
        for label in re.findall(r"Table~\\ref\{(tab:[^}]+)\}", main):
            if label not in first_reference_order:
                first_reference_order.append(label)
        self.assertEqual(
            first_reference_order,
            [
                "tab:dataset-statistics",
                "tab:main-cold-results",
                "tab:core-ablation",
                "tab:efficiency-aaai",
            ],
        )


if __name__ == "__main__":
    unittest.main()
