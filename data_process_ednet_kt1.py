"""Preprocess EdNet-KT1 into the FAST3-compatible stream format.

Expected raw layout is flexible. Common examples:

  data_raw/ednet_kt1/KT1/*.csv
  data_raw/ednet_kt1/KT1/train_data/*.csv
  data_raw/ednet_kt1/contents/questions.csv

KT1 interaction files are usually one CSV per learner and include
timestamp, question_id, user_answer, and elapsed_time. The file stem is used
as user_id when no user_id column is present.
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
    read_table,
    save_processed_dataset,
)


DATASET = "EdNet-KT1"


def _find_kt1_dir(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "KT1" / "train_data",
        raw_dir / "KT1",
        raw_dir / "train_data",
        raw_dir,
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.csv")):
            return path
    raise FileNotFoundError(
        "Could not find EdNet-KT1 learner CSV files. Expected one of: "
        "KT1/train_data/*.csv, KT1/*.csv, train_data/*.csv, or raw-dir/*.csv"
    )


def _is_metadata_file(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ["question", "lecture", "metadata", "content"])


def load_kt1_interactions(raw_dir: Path, max_rows: int | None = None, max_users: int | None = None) -> pd.DataFrame:
    kt1_dir = _find_kt1_dir(raw_dir)
    frames = []
    total = 0
    user_files = [path for path in sorted(kt1_dir.glob("*.csv")) if not _is_metadata_file(path)]
    if max_users is not None:
        user_files = user_files[: max(0, int(max_users))]
    if not user_files:
        raise FileNotFoundError(f"No learner CSV files found in {kt1_dir}")

    for path in user_files:
        usecols = None
        head = pd.read_csv(path, nrows=0)
        cols = set(head.columns)
        keep = [col for col in ["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time", "user_id"] if col in cols]
        if "question_id" not in keep:
            continue
        usecols = keep
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        if "user_id" not in df.columns:
            df["user_id"] = path.stem
        frames.append(df)
        total += len(df)
        if max_rows is not None and total >= max_rows:
            break

    if not frames:
        raise ValueError("No EdNet-KT1 interaction rows with question_id were loaded.")
    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.head(max_rows).copy()
    return data


def load_question_metadata(raw_dir: Path) -> dict[str, ItemMetadata]:
    questions = load_question_table(raw_dir)
    if questions is None:
        return {}

    item_col = first_present(questions.columns, ["question_id", "item_id", "problem_id"])
    if item_col is None:
        return {}
    text_cols = [col for col in ["question_id", "bundle_id", "part", "correct_answer"] if col in questions.columns]
    concept_cols = [col for col in ["tags", "skill_id", "knowledge_tag", "concept_id"] if col in questions.columns]
    family_col = first_present(questions.columns, ["bundle_id", "part"])
    return build_item_metadata_from_table(
        questions,
        item_col=item_col,
        text_cols=text_cols,
        concept_cols=concept_cols,
        family_col=family_col,
    )


def load_question_table(raw_dir: Path) -> pd.DataFrame | None:
    questions_path = find_first_existing(
        raw_dir,
        [
            "contents/questions.csv",
            "KT1/contents/questions.csv",
            "questions.csv",
            "metadata/questions.csv",
        ],
    )
    if questions_path is None:
        return None
    return read_table(questions_path)


def add_correctness_from_questions(interactions: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if "correct" in interactions.columns or "user_answer" not in interactions.columns:
        return interactions
    questions = load_question_table(raw_dir)
    if questions is None:
        return interactions
    item_col = first_present(questions.columns, ["question_id", "item_id", "problem_id"])
    correct_col = first_present(questions.columns, ["correct_answer", "answer"])
    if item_col is None or correct_col is None:
        return interactions
    answer_map = questions[[item_col, correct_col]].dropna().copy()
    answer_map[item_col] = answer_map[item_col].astype(str)
    merged = interactions.copy()
    merged["question_id"] = merged["question_id"].astype(str)
    merged = merged.merge(answer_map, left_on="question_id", right_on=item_col, how="left")
    has_answer = merged[correct_col].notna()
    merged["correct"] = (
        has_answer
        & (
            merged["user_answer"].astype(str).str.strip()
            == merged[correct_col].astype(str).str.strip()
        )
    ).astype("int64")
    drop_cols = [col for col in [item_col, correct_col] if col in merged.columns and col != "question_id"]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def fill_missing_question_metadata(interactions: pd.DataFrame, metadata: dict[str, ItemMetadata]) -> dict[str, ItemMetadata]:
    for question_id in sorted(interactions["question_id"].astype(str).unique()):
        metadata.setdefault(
            question_id,
            ItemMetadata(item_id=question_id, text=f"question {question_id}", concepts=set()),
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, dataset=DATASET, default_raw="data_raw/ednet_kt1", default_out="processed_data_ednet_kt1")
    parser.add_argument("--max-users", type=int, default=None, help="Optional cap on learner files for smoke tests.")
    args = parser.parse_args()
    spec = make_spec(args, DATASET)

    interactions = load_kt1_interactions(spec.raw_dir, max_rows=spec.max_rows, max_users=args.max_users)
    interactions = add_correctness_from_questions(interactions, spec.raw_dir)
    timestamp_col = first_present(interactions.columns, ["timestamp"])
    correct_col = first_present(interactions.columns, ["correct", "is_correct"])

    stream_df, stats = interaction_to_stream(
        interactions,
        user_col="user_id",
        item_col="question_id",
        timestamp_col=timestamp_col,
        correct_col=correct_col,
        spec=spec,
    )
    metadata = fill_missing_question_metadata(interactions, load_question_metadata(spec.raw_dir))
    save_processed_dataset(stream_df, stats, metadata, spec)

    print(f"[Done] {DATASET} processed to {spec.output_dir}")
    print(f"       users={stats['n_users']:,}, items={stats['n_items']:,}, interactions={stats['interactions']:,}")
    print(f"       set USIM_DATA_DIR={spec.output_dir}")
    print(f"       set USIM_RELATION_DIR={spec.output_dir / 'relations'}")


if __name__ == "__main__":
    main()
