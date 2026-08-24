import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_mooccube_exact_significance_queue.ps1"


class MooccubeExactSignificanceQueueTests(unittest.TestCase):
    def test_queue_script_uses_robust_child_process_logging(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("function Start-LoggedCommand", text)
        self.assertIn("Start-Process", text)
        self.assertIn("-EncodedCommand", text)
        self.assertIn("-RedirectStandardOutput", text)
        self.assertIn("-RedirectStandardError", text)
        self.assertNotIn("*>>", text)

    def test_queue_runs_only_main_significance_reference_jobs(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("run_cgrc_paper_static.ps1", text)
        self.assertIn("run_usim_feedback_fast3_content_delta_static.ps1", text)
        self.assertIn("significance_cgrc_exact_reexport", text)
        self.assertIn("significance_per_item_exports\\mooccube\\ckg_rl_full", text)
        self.assertNotIn("bpr_static_fair.py", text)
        self.assertNotIn("lightgcn_static_hin_fair.py", text)

    def test_queue_validates_metrics_before_copying_per_item_files(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("Compare-And-CopyPerItem", text)
        self.assertIn('$metrics = @("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")', text)
        self.assertIn("foreach ($metric in $metrics)", text)
        self.assertIn("METRIC_MISMATCH", text)
        self.assertIn("METRIC_MATCH copied", text)
        self.assertIn("audit_significance_inputs.py", text)

    def test_dry_run_skips_waiting(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("DRY_RUN skip PID wait", text)
        self.assertIn("DRY_RUN skip GPU wait", text)

    def test_dry_run_does_not_remove_child_logs(self):
        text = SCRIPT.read_text(encoding="utf-8")

        dry_run_pos = text.index('if ($DryRun) {\n        Write-QueueLine ("DRY_RUN skip {0}" -f $Name)')
        remove_pos = text.index("Remove-Item -LiteralPath $stdout -Force")
        self.assertLess(dry_run_pos, remove_pos)

    def test_pid_wait_logs_real_process_ids(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("foreach ($processId in $Pids)", text)
        self.assertIn("Get-Process -Id $processId", text)

    def test_seeds_are_passed_as_powershell_array_literal(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("function ConvertTo-PsIntArrayLiteral", text)
        self.assertIn("$seedLiteral = ConvertTo-PsIntArrayLiteral -Values $Seeds", text)
        self.assertIn("-Seeds $seedLiteral", text)
        self.assertNotIn("$seedCsv", text)
        self.assertNotIn("$seedArgs", text)

    def test_bool_arguments_are_passed_as_numeric_literals(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("-UseContentDelta 0", text)
        self.assertIn("-TrainForceCold 1", text)
        self.assertIn("-SaveCkpt 1", text)
        self.assertNotIn("'-UseContentDelta', '$false'", text)


if __name__ == "__main__":
    unittest.main()
