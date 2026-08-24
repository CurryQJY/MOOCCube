import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.audit_significance_inputs import (
    ArtifactSpec,
    MAIN_SIGNIFICANCE_REFERENCES,
    main_significance_missing,
    main_significance_required,
)
from paper_aaai27.scripts.export_missing_per_item import AUDIT_CSV
from paper_aaai27.scripts.build_revision_tables import load_main_seed_values, load_per_course_pairs


def spec(dataset: str, method: str, seed: int = 2025, status: str = "ready") -> dict:
    artifact = ArtifactSpec(dataset, method, seed, (Path("result.json"),), (Path("per_item.csv"),))
    return {
        "dataset": artifact.dataset,
        "method": artifact.method,
        "seed": artifact.seed,
        "status": status,
    }


class SignificanceInputScopeTests(unittest.TestCase):
    def test_only_ckg_and_strongest_baseline_are_main_significance_required(self):
        self.assertEqual(
            MAIN_SIGNIFICANCE_REFERENCES,
            {"MOOCCube": "CGRC", "Junyi": "ALDI", "COCO": "CCFCRec"},
        )
        self.assertTrue(main_significance_required("MOOCCube", "CKG-RL"))
        self.assertTrue(main_significance_required("MOOCCube", "CGRC"))
        self.assertFalse(main_significance_required("MOOCCube", "BPR"))
        self.assertTrue(main_significance_required("Junyi", "ALDI"))
        self.assertFalse(main_significance_required("Junyi", "CGRC"))
        self.assertTrue(main_significance_required("COCO", "CCFCRec"))
        self.assertFalse(main_significance_required("COCO", "CGRC"))

    def test_main_significance_missing_filters_non_reference_baselines(self):
        frame = pd.DataFrame(
            [
                spec("MOOCCube", "BPR", status="missing_per_item"),
                spec("MOOCCube", "CGRC", status="ready_alt"),
                spec("MOOCCube", "CKG-RL", status="ready_alt"),
                spec("Junyi", "CGRC", status="missing_per_item"),
                spec("Junyi", "ALDI", status="ready"),
                spec("COCO", "CCFCRec", status="ready"),
            ]
        )

        missing = main_significance_missing(frame)

        self.assertEqual(missing["method"].tolist(), ["CGRC", "CKG-RL"])
        self.assertTrue((missing["dataset"] == "MOOCCube").all())

    def test_export_queue_defaults_to_main_significance_missing_csv(self):
        self.assertEqual(AUDIT_CSV.name, "significance_main_missing_inputs.csv")

    def test_revision_seed_values_use_junyi_aldi_as_main_reference(self):
        values = load_main_seed_values()
        junyi = values[values["dataset"].eq("Junyi")]

        self.assertEqual(set(junyi["baseline"]), {"ALDI"})
        self.assertAlmostEqual(junyi["baseline_N@10"].mean(), 0.1106097057, places=6)

    def test_revision_per_course_pairs_use_junyi_aldi_as_main_reference(self):
        pairs = load_per_course_pairs()
        junyi = pairs[pairs["dataset"].eq("Junyi")]

        self.assertEqual(set(junyi["baseline"]), {"ALDI"})
        self.assertEqual(len(junyi), 213)


if __name__ == "__main__":
    unittest.main()
