import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_pcgnn_other_2datasets_3seed_serial.ps1"


class PCGNNOtherDatasetsSerialScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_declares_two_datasets_and_three_seeds(self):
        self.assertIn('[string[]]$Datasets = @("junyi", "coco")', self.text)
        self.assertIn('[int[]]$Seeds = @(2025, 2026, 2027)', self.text)

    def test_uses_current_strict_split_roots(self):
        self.assertIn('outputs\\junyi\\mask_ablation\\mask_tt', self.text)
        self.assertIn('outputs\\junyi\\main_table_3seed', self.text)
        self.assertIn('outputs\\coco\\single_seed_triage\\ours_full', self.text)

    def test_prepares_full_pcgnn_atomic_data(self):
        self.assertIn('course_baseline_adaptability.py', self.text)
        self.assertIn('--pcgnn-dataset-name', self.text)
        self.assertEqual(self.text.count('"--max-train-pos", "-1"'), 1)
        self.assertEqual(self.text.count('"--max-val-pos", "-1"'), 1)
        self.assertEqual(self.text.count('"--max-test-pos", "-1"'), 1)

    def test_native_stderr_does_not_abort_completed_tasks(self):
        self.assertIn('$ErrorActionPreference = "Continue"', self.text)
        self.assertIn('$ErrorActionPreference = $previousErrorAction', self.text)

    def test_runs_cuda_tasks_serially_and_validates_reports(self):
        self.assertIn('foreach ($task in $tasks) { Invoke-PCGNNTask $task }', self.text)
        self.assertIn('"--device", "cuda"', self.text)
        self.assertIn('session_graph_backend -eq "torch_batch_scatter"', self.text)
        self.assertIn('full_cold_item_macro', self.text)
        self.assertNotIn('Start-Job', self.text)


if __name__ == "__main__":
    unittest.main()
