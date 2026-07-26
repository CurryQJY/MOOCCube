"""Preprocess MOOCCourse into the FAST3-compatible stream format.

MOOCCourse is the XuetangX course recommendation dataset used by
"Hierarchical Reinforcement Learning for Course Recommendation in MOOCs"
(AAAI 2019). The official MoocData page calls it "Course Recommendation".

Expected raw layout:

  data_raw/MOOCCourse/mooc_data/data.csv

The CSV is encoded as GB18030 and contains:

  stu_id,time,course_index,name,type,type_id

MOOCCourse has no official concept-prerequisite graph. We export coarse course
categories as concepts and leave prerequisite-dependency.json empty; downstream
experiments should use behavior-derived prerequisites for this dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from edu_dataset_common import (
    ItemMetadata,
    add_common_args,
    find_first_existing,
    interaction_to_stream,
    make_spec,
    save_processed_dataset,
)


DATASET = "MOOCCourse"


def load_mooccourse_table(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    path = find_first_existing(
        raw_dir,
        [
            "data.csv",
            "mooc_data/data.csv",
            "Course Recommendation/data.csv",
        ],
    )
    if path is None:
        raise FileNotFoundError(
            "Could not find MOOCCourse data.csv. Expected data.csv or mooc_data/data.csv "
            f"under {raw_dir}"
        )
    kwargs = {"nrows": max_rows} if max_rows is not None else {}
    try:
        table = pd.read_csv(path, encoding="gb18030", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    table.columns = [str(col).strip() for col in table.columns]
    required = {"stu_id", "time", "course_index", "name", "type", "type_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"MOOCCourse data.csv missing columns: {sorted(missing)}")
    return table


def _clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _concept_safe(value: str) -> str:
    return (
        value.strip()
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def split_course_type(type_text: str) -> list[str]:
    """Split coarse MoocData course labels without over-fragmenting names."""
    type_text = _concept_safe(type_text)
    if not type_text:
        return []
    # The source uses labels like "艺术·设计 历史"; keep the original label and
    # add whitespace-separated sublabels for overlap signals.
    parts = [part.strip() for part in type_text.split() if part.strip()]
    return [type_text] + [part for part in parts if part != type_text]


def build_mooccourse_metadata(table: pd.DataFrame) -> dict[str, ItemMetadata]:
    metadata: dict[str, ItemMetadata] = {}
    course_rows = table[["course_index", "name", "type", "type_id"]].drop_duplicates("course_index")
    for row in course_rows.itertuples(index=False):
        course_id = _clean_text(row.course_index)
        name = _clean_text(row.name)
        type_text = _clean_text(row.type)
        type_id = _clean_text(row.type_id)
        concepts = set()
        if type_id:
            concepts.add(f"MOOCCOURSE_TYPE_ID_{type_id}")
        for label in split_course_type(type_text):
            concepts.add(f"MOOCCOURSE_TYPE_{label}")
        text_parts = []
        if name:
            text_parts.append(name)
        if type_text:
            text_parts.append(f"category {type_text}")
        if type_id:
            text_parts.append(f"type_id {type_id}")
        metadata[course_id] = ItemMetadata(
            item_id=course_id,
            text=" ".join(text_parts) if text_parts else f"course {course_id}",
            concepts=concepts,
            prerequisites=set(),
            family=type_id or type_text or None,
        )
    return metadata


def write_source_audit(table: pd.DataFrame, output_dir: Path, raw_dir: Path) -> None:
    user_len = table.groupby("stu_id").size()
    item_len = table.groupby("course_index").size()
    parsed_time = pd.to_datetime(table["time"], errors="coerce")
    train_path = find_first_existing(raw_dir, ["Data/mooc.train.rating", "mooc_data/Data/mooc.train.rating"])
    test_path = find_first_existing(raw_dir, ["Data/mooc.test.rating", "mooc_data/Data/mooc.test.rating"])
    all_path = find_first_existing(raw_dir, ["Data/mooc.all.rating", "mooc_data/Data/mooc.all.rating"])
    negative_path = find_first_existing(raw_dir, ["Data/mooc.test.negative", "mooc_data/Data/mooc.test.negative"])
    audit = {
        "dataset": DATASET,
        "source_rows": int(len(table)),
        "source_users": int(table["stu_id"].nunique()),
        "source_courses": int(table["course_index"].nunique()),
        "source_course_types": int(table["type_id"].nunique(dropna=True)),
        "duplicate_user_course_pairs": int(table.duplicated(["stu_id", "course_index"]).sum()),
        "time_min": str(parsed_time.min()) if parsed_time.notna().any() else None,
        "time_max": str(parsed_time.max()) if parsed_time.notna().any() else None,
        "missing_time": int(parsed_time.isna().sum()),
        "user_interactions": {
            "min": int(user_len.min()),
            "median": float(user_len.median()),
            "mean": float(user_len.mean()),
            "max": int(user_len.max()),
            "users_ge_5": int((user_len >= 5).sum()),
        },
        "item_interactions": {
            "min": int(item_len.min()),
            "median": float(item_len.median()),
            "mean": float(item_len.mean()),
            "max": int(item_len.max()),
            "items_ge_10": int((item_len >= 10).sum()),
        },
        "provided_split_files": {
            "train_rating": str(train_path) if train_path else None,
            "test_rating": str(test_path) if test_path else None,
            "all_rating": str(all_path) if all_path else None,
            "test_negative": str(negative_path) if negative_path else None,
        },
    }
    with (output_dir / "source_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        dataset=DATASET,
        default_raw="data_raw/MOOCCourse",
        default_out="processed_data_mooccourse",
    )
    parser.set_defaults(
        embedding_backend="bert_cls",
        embedding_model="bert-base-chinese",
        embedding_max_length=256,
        embedding_batch_size=32,
    )
    args = parser.parse_args()
    spec = make_spec(args, DATASET)

    table = load_mooccourse_table(spec.raw_dir, max_rows=spec.max_rows)
    stream_df, stats = interaction_to_stream(
        table,
        user_col="stu_id",
        item_col="course_index",
        timestamp_col="time",
        correct_col=None,
        spec=spec,
    )
    metadata = build_mooccourse_metadata(table)
    save_processed_dataset(stream_df, stats, metadata, spec)
    write_source_audit(table, spec.output_dir, spec.raw_dir)

    print(f"[Done] {DATASET} processed to {spec.output_dir}")
    print(f"       users={stats['n_users']:,}, items={stats['n_items']:,}, interactions={stats['interactions']:,}")
    print(f"       set USIM_DATA_DIR={spec.output_dir}")
    print(f"       set USIM_RELATION_DIR={spec.output_dir / 'relations'}")
    print("       set USIM_PREREQ_GRAPH_SOURCE=behavior")


if __name__ == "__main__":
    main()
