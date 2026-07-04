import json
import tempfile
import unittest
from pathlib import Path

from edu_dataset_common import DatasetSpec
from data_process_coco import build_coco_metadata, load_coco_tables, process_coco


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CocoProcessorTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        pre = root / "preprocessed"
        write_text(
            pre / "ratings.txt",
            "\n".join(
                [
                    "uid\tpid\trating\ttimestamp",
                    "u1\tc1\t1\t100",
                    "u1\tc2\t1\t200",
                    "u2\tc1\t1\t300",
                ]
            )
            + "\n",
        )
        write_text(
            pre / "i2kg_map.txt",
            "\n".join(
                [
                    "eid\tpid\tname\tentity",
                    "0\tc1\tintro-python\tc1",
                    "1\tc2\tadvanced-python\tc2",
                ]
            )
            + "\n",
        )
        write_text(
            pre / "e_map.txt",
            "\n".join(
                [
                    "eid\tname\tentity",
                    "0\tc1\tc1",
                    "1\tc2\tc2",
                    "2\tprogramming\tprogramming",
                    "3\tPython\tPython",
                    "4\tbeginner\tbeginner",
                    "5\tenglish\tenglish",
                    "6\tSoftware developers\tSoftware developers",
                ]
            )
            + "\n",
        )
        write_text(
            pre / "r_map.txt",
            "\n".join(
                [
                    "id\tkb_relation\tname",
                    "0\tbelong_to_category\tbelong_to_category",
                    "1\trelated_to_concept\trelated_to_concept",
                    "2\ttaught_in_level\ttaught_in_level",
                    "3\ttaught_in_language\ttaught_in_language",
                    "4\thas_target_audience\thas_target_audience",
                ]
            )
            + "\n",
        )
        write_text(
            pre / "kg_final.txt",
            "\n".join(
                [
                    "entity_head\trelation\tentity_tail",
                    "0\t0\t2",
                    "0\t1\t3",
                    "0\t2\t4",
                    "0\t3\t5",
                    "0\t4\t6",
                    "1\t0\t2",
                ]
            )
            + "\n",
        )

    def test_build_metadata_keeps_conservative_concepts_and_text_uses_all_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)

            tables = load_coco_tables(root)
            metadata, audit = build_coco_metadata(tables, concept_scope="conservative")

        self.assertEqual(metadata["c1"].concepts, {"COCO_CATEGORY:programming", "COCO_CONCEPT:Python"})
        self.assertEqual(metadata["c2"].concepts, {"COCO_CATEGORY:programming"})
        self.assertNotIn("COCO_LEVEL:beginner", metadata["c1"].concepts)
        self.assertIn("intro python", metadata["c1"].text)
        self.assertIn("beginner", metadata["c1"].text)
        self.assertIn("Software developers", metadata["c1"].text)
        self.assertEqual(audit["relation_coverage"]["related_to_concept"]["items_covered"], 1)

    def test_process_coco_writes_fast3_compatible_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            self._write_fixture(root)
            spec = DatasetSpec(
                dataset="COCO",
                raw_dir=root,
                output_dir=out,
                min_user_interactions=1,
                min_item_interactions=1,
                embedding_backend="stable_hash",
            )

            process_coco(spec, concept_scope="conservative")

            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            course_concept = (out / "relations" / "course-concept.json").read_text(encoding="utf-8")
            prereq = (out / "relations" / "prerequisite-dependency.json").read_text(encoding="utf-8")
            source_audit = json.loads((out / "source_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["n_users"], 2)
        self.assertEqual(meta["n_items"], 2)
        self.assertEqual(meta["relations"]["items_with_concept"], 2)
        self.assertIn("c1\tCOCO_CONCEPT:Python", course_concept)
        self.assertEqual(prereq, "")
        self.assertEqual(source_audit["kg"]["relations"]["related_to_concept"]["edges"], 1)


if __name__ == "__main__":
    unittest.main()
