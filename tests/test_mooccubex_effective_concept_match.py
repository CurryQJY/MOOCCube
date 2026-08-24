import unittest

from diagnose_mooccubex_effective_concept_match import (
    _parse_relation_cases,
    _threshold_activation_stats,
)


class EffectiveConceptMatchTests(unittest.TestCase):
    def test_parse_relation_cases_accepts_label_equals_path(self):
        cases = _parse_relation_cases(["Raw=MOOCCubeX/relations", "Aug=MOOCCubeX/relations_aug"])

        self.assertEqual(
            cases,
            [
                ("Raw", "MOOCCubeX/relations"),
                ("Aug", "MOOCCubeX/relations_aug"),
            ],
        )

    def test_threshold_activation_stats_reports_row_and_item_ratios(self):
        values = [0.0, 0.01, 0.03, 0.12]
        by_item = {
            1: [0.0, 0.01],
            2: [0.03, 0.12],
        }

        rows = _threshold_activation_stats(values, by_item, thresholds=[0.01, 0.05])

        self.assertEqual(rows[0]["threshold"], 0.01)
        self.assertEqual(rows[0]["row_active"], 3)
        self.assertAlmostEqual(rows[0]["row_active_ratio_ge_threshold"], 0.75)
        self.assertEqual(rows[0]["item_active"], 1)
        self.assertAlmostEqual(rows[0]["item_active_ratio_ge_threshold"], 0.5)
        self.assertEqual(rows[1]["threshold"], 0.05)
        self.assertEqual(rows[1]["row_active"], 1)
        self.assertEqual(rows[1]["item_active"], 1)


if __name__ == "__main__":
    unittest.main()
