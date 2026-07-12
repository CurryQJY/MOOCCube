from __future__ import annotations

import json

import pandas as pd

from paper_aaai27.scripts.kgrec_strict_adapter import build_atomic_data
from paper_aaai27.scripts.kgrec_strict_adapter import export_mooccube_kgrec_dataset
from paper_aaai27.scripts.kgrec_strict_adapter import normalize_mooccube_course_side_edges
from paper_aaai27.scripts.kgrec_strict_adapter import write_kgrec_atomic_dataset


def test_atomic_data_keeps_cold_courses_in_kg_but_out_of_cf_training() -> None:
    atomic = build_atomic_data(
        train_pairs=[("u0", "c0"), ("u1", "c1")],
        validation_pairs=[("u0", "c2")],
        test_pairs=[("u1", "c2")],
        course_side_edges={
            "course_video": [("c0", "v0"), ("c2", "v2")],
            "course_concept": [("c1", "k1"), ("c2", "k2")],
        },
    )

    cold_item = atomic.course_to_item["c2"]
    assert cold_item not in atomic.warm_item_ids
    assert cold_item in atomic.cold_item_ids
    assert all(item != cold_item for _, item in atomic.train_pairs)
    assert atomic.course_kg_degree[cold_item] == 2
    assert any(head == cold_item or tail == cold_item for head, _, tail in atomic.kg_triples)


def test_write_kgrec_dataset_preserves_strict_boundaries(tmp_path) -> None:
    atomic = build_atomic_data(
        train_pairs=[("u0", "c0"), ("u0", "c1")],
        validation_pairs=[("u0", "c2")],
        test_pairs=[("u1", "c2")],
        course_side_edges={
            "course_video": [("c0", "v0"), ("c2", "v2")],
            "course_concept": [("c2", "k2")],
        },
    )

    manifest = write_kgrec_atomic_dataset(tmp_path, atomic)
    cold_item = atomic.course_to_item["c2"]

    train_lines = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    assert train_lines
    train_tokens = [int(token) for line in train_lines for token in line.split()[1:]]
    assert cold_item not in train_tokens

    test_tokens = [int(token) for line in (tmp_path / "test.txt").read_text(encoding="utf-8").splitlines() for token in line.split()[1:]]
    assert cold_item in test_tokens

    triples = [
        tuple(int(token) for token in line.split())
        for line in (tmp_path / "kg_final.txt").read_text(encoding="utf-8").splitlines()
    ]
    assert triples
    assert all(0 <= relation < atomic.n_relations for _head, relation, _tail in triples)
    assert all(0 <= head < atomic.n_entities and 0 <= tail < atomic.n_entities for head, _relation, tail in triples)

    disk_manifest = json.loads((tmp_path / "strict_manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest == manifest
    assert disk_manifest["strict_checks"]["cold_items_absent_from_train"]
    assert disk_manifest["strict_checks"]["kg_entity_ids_contiguous"]


def test_normalize_mooccube_edges_filters_to_courses_and_flips_external_course_relations() -> None:
    edges = normalize_mooccube_course_side_edges(
        {
            "course-video": [("c0", "v0"), ("c9", "v9")],
            "course-concept": [("c1", "k1")],
            "teacher-course": [("t0", "c0"), ("t9", "c9")],
            "school-course": [("s0", "c1")],
        },
        course_ids={"c0", "c1"},
    )

    assert edges == {
        "course_video": [("c0", "video:v0")],
        "course_concept": [("c1", "concept:k1")],
        "course_teacher": [("c0", "teacher:t0")],
        "course_school": [("c1", "school:s0")],
    }


def test_export_mooccube_kgrec_dataset_from_strict_split(tmp_path) -> None:
    split_root = tmp_path / "strict_item_cold_balanced_thr1_seed_2025"
    split_root.mkdir()
    pd.DataFrame([{"user_id": "u0", "course_id": "c0"}, {"user_id": "u1", "course_id": "c1"}]).to_pickle(
        split_root / "static_train.pkl"
    )
    pd.DataFrame([{"user_id": "u0", "course_id": "c2"}]).to_pickle(split_root / "static_val.pkl")
    pd.DataFrame([{"user_id": "u1", "course_id": "c2"}]).to_pickle(split_root / "static_test.pkl")

    relations_dir = tmp_path / "relations"
    relations_dir.mkdir()
    (relations_dir / "course-video.json").write_text("c0\tv0\nc2\tv2\n", encoding="utf-8")
    (relations_dir / "course-concept.json").write_text("c2\tk2\n", encoding="utf-8")
    (relations_dir / "teacher-course.json").write_text("t0\tc2\n", encoding="utf-8")
    (relations_dir / "school-course.json").write_text("s0\tc9\n", encoding="utf-8")

    manifest = export_mooccube_kgrec_dataset(
        split_root=split_root,
        relations_dir=relations_dir,
        output_dir=tmp_path / "kgrec_out",
    )

    assert manifest["source"]["split_root"].endswith("strict_item_cold_balanced_thr1_seed_2025")
    assert manifest["strict_checks"]["cold_items_absent_from_train"]
    assert manifest["strict_checks"]["all_cold_items_have_kg_edges"]
    assert manifest["relation_edge_counts"] == {
        "course_concept": 1,
        "course_teacher": 1,
        "course_video": 2,
    }

