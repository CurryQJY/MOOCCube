"""Preprocess Junyi educational logs into the FAST3-compatible stream format.

The script supports common Junyi layouts, including:

  data_raw/junyi/Log_Problem.csv
  data_raw/junyi/Info_Content.csv
  data_raw/junyi/junyi_ProblemLog_original.csv
  data_raw/junyi/junyi_Exercise_table.csv

The output treats each exercise/problem/content id as the recommendation item.
Concepts are read from skill/topic/tag columns when available. Explicit
exercise prerequisites are exported as concept-level prerequisite edges when a
metadata file provides them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from edu_dataset_common import (
    ItemMetadata,
    add_common_args,
    find_first_existing,
    first_present,
    ensure_dir,
    interaction_to_stream,
    make_spec,
    parse_concepts,
    read_table,
    save_processed_dataset,
)


DATASET = "Junyi"
JUNYI_SELF_CONCEPT_PREFIX = "JUNYI_EXERCISE_SELF:"


INTERACTION_FILES = [
    "Log_Problem.csv",
    "log_problem.csv",
    "junyi_ProblemLog_original.csv",
    "junyi_problem_log.csv",
    "ProblemLog.csv",
    "problem_log.csv",
    "interactions.csv",
    "logs.csv",
]

METADATA_FILES = [
    "Info_Content.csv",
    "info_content.csv",
    "junyi_Exercise_table.csv",
    "junyi_exercise_table.csv",
    "Exercise_table.csv",
    "exercise_table.csv",
    "contents.csv",
    "metadata.csv",
]


def junyi_self_concept(item_id: str) -> str:
    return f"{JUNYI_SELF_CONCEPT_PREFIX}{str(item_id).strip()}"


def junyi_metadata_concept(column: str, value: str) -> str:
    return f"JUNYI_{column.upper()}:{str(value).strip()}"


def load_junyi_interactions(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    path = find_first_existing(raw_dir, INTERACTION_FILES)
    if path is None:
        raise FileNotFoundError(
            "Could not find Junyi interaction file. Expected one of: "
            + ", ".join(INTERACTION_FILES)
        )
    kwargs = {"nrows": max_rows} if max_rows is not None else {}
    header = read_table(path, nrows=0)
    required_cols = [
        first_present(header.columns, ["user_id", "anon_student_id", "student_id", "uid", "user"]),
        first_present(header.columns, ["exercise", "exercise_id", "problem_id", "content_id", "item_id", "problem_name"]),
        first_present(header.columns, ["timestamp", "time_done", "submit_time", "time", "date"]),
        first_present(header.columns, ["correct", "is_correct", "first_response", "answer_correct", "outcome"]),
    ]
    concept_cols = [
        col
        for col in header.columns
        if str(col).lower() in {
            "skill",
            "skill_id",
            "topic",
            "area",
            "tags",
            "kc",
            "knowledge_component",
            "concept_id",
        }
    ]
    usecols = []
    for col in required_cols + concept_cols:
        if col is not None and col not in usecols:
            usecols.append(col)
    if usecols:
        kwargs["usecols"] = usecols
    return read_table(path, **kwargs)


def normalize_junyi_timestamp(interactions: pd.DataFrame, timestamp_col: str | None) -> tuple[pd.DataFrame, str | None]:
    if timestamp_col is None or timestamp_col not in interactions.columns:
        return interactions, timestamp_col
    if timestamp_col.lower() != "time_done":
        return interactions, timestamp_col

    numeric = pd.to_numeric(interactions[timestamp_col], errors="coerce")
    if numeric.notna().any() and numeric.max() > 10_000_000_000_000:
        normalized = interactions.copy()
        normalized["timestamp"] = (numeric // 1_000_000).astype("Int64")
        return normalized, "timestamp"
    return interactions, timestamp_col


def load_junyi_metadata(raw_dir: Path) -> pd.DataFrame | None:
    path = find_first_existing(raw_dir, METADATA_FILES)
    if path is None:
        return None
    return read_table(path)


def build_metadata_from_interactions(
    interactions: pd.DataFrame,
    *,
    item_col: str,
    concept_cols: list[str],
) -> dict[str, ItemMetadata]:
    metadata: dict[str, ItemMetadata] = {}
    cols = [item_col] + concept_cols
    for _, row in interactions[cols].drop_duplicates().iterrows():
        row_map = row.to_dict()
        item_id = str(row_map[item_col]).strip()
        if not item_id or item_id.lower() == "nan":
            continue
        meta = metadata.setdefault(item_id, ItemMetadata(item_id=item_id, text=f"exercise {item_id}"))
        meta.concepts.add(junyi_self_concept(item_id))
        for col in concept_cols:
            meta.concepts.update(junyi_metadata_concept(col, value) for value in parse_concepts(row_map.get(col)))
    return metadata


def _build_metadata_from_exercise_table(
    table: pd.DataFrame,
    *,
    item_col: str,
    text_cols: list[str],
    concept_cols: list[str],
    prereq_cols: list[str],
    family_col: str | None,
) -> dict[str, ItemMetadata]:
    metadata: dict[str, ItemMetadata] = {}
    for _, row in table.iterrows():
        row_map = row.to_dict()
        item_id = str(row_map[item_col]).strip()
        if not item_id or item_id.lower() == "nan":
            continue

        text_parts = []
        for col in text_cols:
            value = row_map.get(col)
            if pd.notna(value):
                value_s = str(value).strip()
                if value_s and value_s.lower() != "nan":
                    text_parts.append(value_s)

        concepts = {junyi_self_concept(item_id)}
        for col in concept_cols:
            concepts.update(junyi_metadata_concept(col, value) for value in parse_concepts(row_map.get(col)))

        prereqs = set()
        for col in prereq_cols:
            prereqs.update(junyi_self_concept(value) for value in parse_concepts(row_map.get(col)))

        family = None
        if family_col is not None:
            raw_family = row_map.get(family_col)
            if pd.notna(raw_family):
                family = str(raw_family).strip() or None

        metadata[item_id] = ItemMetadata(
            item_id=item_id,
            text=" ".join(text_parts),
            concepts=concepts,
            prerequisites=prereqs,
            family=family,
        )
    return metadata


def build_junyi_metadata(interactions: pd.DataFrame, item_col: str, raw_dir: Path) -> dict[str, ItemMetadata]:
    table = load_junyi_metadata(raw_dir)
    interaction_concept_cols = [
        col
        for col in interactions.columns
        if str(col).lower() in {
            "skill",
            "skill_id",
            "topic",
            "area",
            "tags",
            "kc",
            "knowledge_component",
            "concept_id",
        }
    ]

    if table is None:
        metadata = build_metadata_from_interactions(
            interactions,
            item_col=item_col,
            concept_cols=interaction_concept_cols,
        )
    else:
        meta_item_col = first_present(
            table.columns,
            [
                item_col,
                "exercise",
                "exercise_id",
                "problem_id",
                "content_id",
                "item_id",
                "name",
            ],
        )
        if meta_item_col is None:
            metadata = {}
        else:
            text_cols = [
                col
                for col in ["name", "exercise", "exercise_id", "problem_id", "topic", "area", "problem_type"]
                if col in table.columns
            ]
            concept_cols = [
                col
                for col in ["skill", "skill_id", "topic", "area", "tags", "kc", "knowledge_component", "concept_id"]
                if col in table.columns
            ]
            prereq_cols = [
                col
                for col in ["prerequisite", "prerequisites", "prereq", "required_skill", "required_concept"]
                if col in table.columns
            ]
            family_col = first_present(table.columns, ["topic", "area", "subject", "strand"])
            metadata = _build_metadata_from_exercise_table(
                table,
                item_col=meta_item_col,
                text_cols=text_cols,
                concept_cols=concept_cols,
                prereq_cols=prereq_cols,
                family_col=family_col,
            )

    for item_id in sorted(interactions[item_col].astype(str).unique()):
        meta = metadata.setdefault(item_id, ItemMetadata(item_id=item_id, text=f"exercise {item_id}"))
        meta.concepts.add(junyi_self_concept(item_id))
    return metadata


def write_junyi_compat_relations(
    output_dir: Path,
    item_ids: list[str],
    item_metadata: dict[str, ItemMetadata],
) -> dict:
    relation_dir = output_dir / "relations"
    entity_dir = output_dir / "entities"
    ensure_dir(relation_dir)
    ensure_dir(entity_dir)

    concept_edges = 0
    prereq_edges = 0
    with (relation_dir / "course-concept.json").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            concepts = set(meta.concepts)
            concepts.add(junyi_self_concept(str(item_id)))
            for concept in sorted(concepts):
                writer.writerow([str(item_id), concept])
                concept_edges += 1

    with (relation_dir / "prerequisite-dependency.json").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            target_self = junyi_self_concept(str(item_id))
            for prereq in sorted(meta.prerequisites):
                if prereq != target_self:
                    writer.writerow([prereq, target_self])
                    prereq_edges += 1

    with (entity_dir / "course.json").open("w", encoding="utf-8") as f:
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            obj = {
                "id": str(item_id),
                "name": meta.text[:80] if meta.text else str(item_id),
                "about": meta.text,
                "core_id": meta.family or str(item_id),
                "resource": [],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "items_with_concept": int(
            sum(
                1
                for item_id in item_ids
                if item_metadata.get(str(item_id), ItemMetadata(str(item_id))).concepts
            )
        ),
        "concept_edges": int(concept_edges),
        "concept_prereq_edges": int(prereq_edges),
        "relation_writer": "junyi_exercise_self_concept",
    }


def save_junyi_processed_dataset(
    df: pd.DataFrame,
    stats: dict,
    item_metadata: dict[str, ItemMetadata],
    spec,
) -> None:
    save_processed_dataset(df, stats, item_metadata, spec)
    relation_stats = write_junyi_compat_relations(spec.output_dir, stats["item_id_classes"], item_metadata)

    meta_path = spec.output_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["relations"] = relation_stats
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, dataset=DATASET, default_raw="data_raw/junyi", default_out="processed_data_junyi")
    args = parser.parse_args()
    spec = make_spec(args, DATASET)

    interactions = load_junyi_interactions(spec.raw_dir, max_rows=spec.max_rows)
    user_col = first_present(interactions.columns, ["user_id", "anon_student_id", "student_id", "uid", "user"])
    item_col = first_present(
        interactions.columns,
        ["exercise", "exercise_id", "problem_id", "content_id", "item_id", "problem_name"],
    )
    timestamp_col = first_present(interactions.columns, ["timestamp", "time_done", "submit_time", "time", "date"])
    correct_col = first_present(interactions.columns, ["correct", "is_correct", "first_response", "answer_correct", "outcome"])
    interactions, timestamp_col = normalize_junyi_timestamp(interactions, timestamp_col)

    missing = []
    if user_col is None:
        missing.append("user id")
    if item_col is None:
        missing.append("item/exercise id")
    if missing:
        raise ValueError(f"Could not infer Junyi columns for: {', '.join(missing)}")

    stream_df, stats = interaction_to_stream(
        interactions,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        correct_col=correct_col,
        spec=spec,
    )
    metadata = build_junyi_metadata(interactions, item_col=item_col, raw_dir=spec.raw_dir)
    save_junyi_processed_dataset(stream_df, stats, metadata, spec)

    print(f"[Done] {DATASET} processed to {spec.output_dir}")
    print(f"       users={stats['n_users']:,}, items={stats['n_items']:,}, interactions={stats['interactions']:,}")
    print(f"       set USIM_DATA_DIR={spec.output_dir}")
    print(f"       set USIM_RELATION_DIR={spec.output_dir / 'relations'}")


if __name__ == "__main__":
    main()
