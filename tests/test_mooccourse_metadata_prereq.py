import tempfile
import unittest
from pathlib import Path

from build_mooccourse_metadata_prereq import (
    CourseRecord,
    build_metadata_prereq_edges,
    course_self_concept,
    write_relation_bundle,
)


class MoocCourseMetadataPrereqTest(unittest.TestCase):
    def test_sequence_edges_use_course_self_concepts(self):
        records = [
            CourseRecord("0", "Chinese Architecture History (I)", "20", "history"),
            CourseRecord("1", "Chinese Architecture History (II)", "20", "history"),
        ]

        edges = build_metadata_prereq_edges(records, max_per_target=3)

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, "0")
        self.assertEqual(edges[0].target_id, "1")
        self.assertEqual(edges[0].rule, "sequence")
        self.assertEqual(edges[0].source_concept, course_self_concept("0"))
        self.assertEqual(edges[0].target_concept, course_self_concept("1"))

    def test_foundation_edges_stay_within_type(self):
        records = [
            CourseRecord("10", "Introduction to Computer Science", "7", "computer"),
            CourseRecord("11", "Python Programming Practice", "7", "computer"),
            CourseRecord("12", "Art Design Practice", "20", "art"),
        ]

        edges = build_metadata_prereq_edges(records, max_per_target=3)
        edge_pairs = {(edge.source_id, edge.target_id) for edge in edges}

        self.assertIn(("10", "11"), edge_pairs)
        self.assertNotIn(("10", "12"), edge_pairs)

    def test_foundation_edges_require_specific_target_signal(self):
        records = [
            CourseRecord("20", "Piano Basics", "20", "art"),
            CourseRecord("21", "Chinese Architecture History", "20", "art"),
        ]

        edges = build_metadata_prereq_edges(records, max_per_target=3)
        edge_pairs = {(edge.source_id, edge.target_id) for edge in edges}

        self.assertNotIn(("20", "21"), edge_pairs)

    def test_write_relation_bundle_preserves_concepts_and_adds_self_concepts(self):
        records = [
            CourseRecord("10", "Introduction to Computer Science", "7", "computer"),
            CourseRecord("11", "Python Programming Practice", "7", "computer"),
        ]
        edges = build_metadata_prereq_edges(records, max_per_target=3)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "output"
            source.mkdir()
            (source / "course-concept.json").write_text("10\tTYPE_7\n11\tTYPE_7\n", encoding="utf-8")

            write_relation_bundle(records, edges, source, output)

            course_concept = (output / "course-concept.json").read_text(encoding="utf-8")
            prereq = (output / "prerequisite-dependency.json").read_text(encoding="utf-8")

        self.assertIn("10\tTYPE_7", course_concept)
        self.assertIn(f"10\t{course_self_concept('10')}", course_concept)
        self.assertIn(f"{course_self_concept('10')}\t{course_self_concept('11')}", prereq)


if __name__ == "__main__":
    unittest.main()
