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
from pathlib import Path

import pandas as pd

from edu_dataset_common import (
    ItemMetadata,
    add_common_args,
    build_item_metadata_from_table,
    find_first_existing,
    first_present,
    interaction_to_stream,
    make_spec,
    parse_concepts,
    read_table,
    save_processed_dataset,
)


DATASET = "Junyi"


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


def load_junyi_interactions(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    path = find_first_existing(raw_dir, INTERACTION_FILES)
    if path is None:
        raise FileNotFoundError(
            "Could not find Junyi interaction file. Expected one of: "
            + ", ".join(INTERACTION_FILES)
        )
    kwargs = {"nrows": max_rows} if max_rows is not None else {}
    return read_table(path, **kwargs)


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
        for col in concept_cols:
            meta.concepts.update(parse_concepts(row_map.get(col)))
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
            metadata = build_item_metadata_from_table(
                table,
                item_col=meta_item_col,
                text_cols=text_cols,
                concept_cols=concept_cols,
                prereq_cols=prereq_cols,
                family_col=family_col,
            )

    for item_id in sorted(interactions[item_col].astype(str).unique()):
        metadata.setdefault(item_id, ItemMetadata(item_id=item_id, text=f"exercise {item_id}"))
    return metadata


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
    save_processed_dataset(stream_df, stats, metadata, spec)

    print(f"[Done] {DATASET} processed to {spec.output_dir}")
    print(f"       users={stats['n_users']:,}, items={stats['n_items']:,}, interactions={stats['interactions']:,}")
    print(f"       set USIM_DATA_DIR={spec.output_dir}")
    print(f"       set USIM_RELATION_DIR={spec.output_dir / 'relations'}")


if __name__ == "__main__":
    main()
