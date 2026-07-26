"""Preprocess XES3G5M into the FAST3-compatible stream format.

XES3G5M is released as pyKT-style sequence files plus rich question metadata.
This converter treats each question as the recommendation item and expands the
question-level sequences into user-question interactions.

Default inputs:

  data_raw/XES3G5M/question_level/train_valid_sequences_quelevel.csv
  data_raw/XES3G5M/question_level/test_quelevel.csv
  data_raw/XES3G5M/metadata/questions.json
  data_raw/XES3G5M/metadata/embeddings/qid2content_emb.json

The prerequisite graph is hierarchy-derived from textual KC routes. It is not
an explicit official prerequisite graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import torch
from tqdm import tqdm

from edu_dataset_common import DatasetSpec, ItemMetadata, ensure_dir, interaction_to_stream


DATASET = "XES3G5M"
DEFAULT_RAW = "data_raw/XES3G5M"
DEFAULT_OUT = "processed_data_xes3g5m"


def _clean_token(value) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _split_csv_cell(value) -> list[str]:
    text = _clean_token(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def concept_ids_from_cell(value) -> set[str]:
    concepts: set[str] = set()
    for token in str(value).split("_"):
        token = _clean_token(token)
        if token:
            concepts.add(f"KC_{token}")
    return concepts


def _timestamp_to_seconds(value: str) -> int:
    try:
        timestamp = int(float(value))
    except (TypeError, ValueError):
        return 0
    # XES3G5M README defines timestamps as millisecond-level values.
    return int(timestamp // 1000)


def _mask_is_kept(value: str) -> bool:
    try:
        return int(float(value)) > 0
    except (TypeError, ValueError):
        return False


def expand_sequence_frame(
    frame: pd.DataFrame,
    *,
    source: str,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, int]]:
    """Expand pyKT question-level sequence rows into interaction rows."""
    required = ["uid", "questions", "concepts", "responses", "timestamps"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing XES3G5M sequence columns: {missing}")

    rows: list[dict] = []
    item_concepts: dict[str, set[str]] = defaultdict(set)
    stats = {
        "sequence_rows": int(len(frame)),
        "expanded_positions": 0,
        "kept_positions": 0,
        "masked_positions": 0,
    }

    has_masks = "selectmasks" in frame.columns
    for rec in frame.itertuples(index=False):
        rec_map = rec._asdict()
        uid = _clean_token(rec_map["uid"])
        questions = _split_csv_cell(rec_map["questions"])
        concepts = _split_csv_cell(rec_map["concepts"])
        responses = _split_csv_cell(rec_map["responses"])
        timestamps = _split_csv_cell(rec_map["timestamps"])
        masks = _split_csv_cell(rec_map["selectmasks"]) if has_masks else ["1"] * len(questions)

        lengths = {len(questions), len(concepts), len(responses), len(timestamps), len(masks)}
        if len(lengths) != 1:
            raise ValueError(
                f"Sequence length mismatch in source={source}, uid={uid}: "
                f"questions={len(questions)} concepts={len(concepts)} "
                f"responses={len(responses)} timestamps={len(timestamps)} masks={len(masks)}"
            )

        for qid, concept_cell, response, timestamp, mask in zip(
            questions, concepts, responses, timestamps, masks
        ):
            stats["expanded_positions"] += 1
            qid = _clean_token(qid)
            if not qid or not _mask_is_kept(mask):
                stats["masked_positions"] += 1
                continue
            concept_ids = concept_ids_from_cell(concept_cell)
            item_concepts[qid].update(concept_ids)
            rows.append(
                {
                    "user_id": uid,
                    "course_id": qid,
                    "timestamp": _timestamp_to_seconds(timestamp),
                    "raw_time": _clean_token(timestamp),
                    "correct": int(float(response)) if _clean_token(response) else 0,
                    "source": source,
                }
            )
            stats["kept_positions"] += 1

    return pd.DataFrame(rows), dict(item_concepts), stats


def route_parts(route: str) -> list[str]:
    return [part.strip() for part in str(route).split("----") if part.strip()]


def route_node_id(parts: Iterable[str]) -> str:
    cleaned = [str(part).replace("\t", " ").replace("\n", " ").strip() for part in parts]
    return "ROUTE::" + "----".join(part for part in cleaned if part)


def route_hierarchy_edges(route: str) -> list[tuple[str, str]]:
    parts = route_parts(route)
    edges = []
    for idx in range(1, len(parts)):
        edges.append((route_node_id(parts[:idx]), route_node_id(parts[: idx + 1])))
    return edges


def route_prefix_concepts(route: str) -> set[str]:
    parts = route_parts(route)
    return {route_node_id(parts[:idx]) for idx in range(1, len(parts) + 1)}


def build_item_metadata(
    questions: Mapping[str, Mapping],
    item_concepts: Mapping[str, set[str]],
) -> tuple[dict[str, ItemMetadata], set[tuple[str, str]]]:
    metadata: dict[str, ItemMetadata] = {}
    prereq_edges: set[tuple[str, str]] = set()

    all_item_ids = set(map(str, questions.keys())) | set(map(str, item_concepts.keys()))
    for item_id in sorted(all_item_ids, key=lambda x: (len(x), x)):
        question = questions.get(str(item_id), {})
        concepts = set(item_concepts.get(str(item_id), set()))
        routes = question.get("kc_routes") or []
        if isinstance(routes, str):
            routes = [routes]
        for route in routes:
            concepts.update(route_prefix_concepts(route))
            prereq_edges.update(route_hierarchy_edges(route))

        content = _clean_token(question.get("content", ""))
        analysis = _clean_token(question.get("analysis", ""))
        text = " ".join(part for part in [content, analysis] if part)
        metadata[str(item_id)] = ItemMetadata(
            item_id=str(item_id),
            text=text or f"question {item_id}",
            concepts=concepts,
            family=None,
        )

    return metadata, prereq_edges


def load_questions(raw_dir: Path) -> dict[str, dict]:
    path = raw_dir / "metadata" / "questions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing XES3G5M questions metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_official_question_embeddings(raw_dir: Path, item_ids: list[str]) -> torch.Tensor:
    path = raw_dir / "metadata" / "embeddings" / "qid2content_emb.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing XES3G5M official question embeddings: {path}")
    qid2emb = json.loads(path.read_text(encoding="utf-8"))
    missing = [item_id for item_id in item_ids if str(item_id) not in qid2emb]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing official content embeddings for {len(missing)} items: {preview}")
    matrix = [qid2emb[str(item_id)] for item_id in item_ids]
    return torch.tensor(matrix, dtype=torch.float32)


def write_xes_relations(
    output_dir: Path,
    item_ids: list[str],
    item_metadata: Mapping[str, ItemMetadata],
    prereq_edges: set[tuple[str, str]],
) -> dict:
    relation_dir = output_dir / "relations"
    entity_dir = output_dir / "entities"
    ensure_dir(relation_dir)
    ensure_dir(entity_dir)

    concept_edges = 0
    with (relation_dir / "course-concept.json").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            for concept in sorted(meta.concepts):
                writer.writerow([str(item_id), concept])
                concept_edges += 1

    with (relation_dir / "prerequisite-dependency.json").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for source, target in sorted(prereq_edges):
            if source != target:
                writer.writerow([source, target])

    with (entity_dir / "course.json").open("w", encoding="utf-8") as f:
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            obj = {
                "id": str(item_id),
                "name": meta.text[:80] if meta.text else f"question {item_id}",
                "about": meta.text,
                "core_id": str(item_id),
                "resource": [],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "items_with_concept": int(
            sum(1 for item_id in item_ids if item_metadata.get(str(item_id), ItemMetadata(str(item_id))).concepts)
        ),
        "concept_edges": int(concept_edges),
        "concept_prereq_edges": int(len({edge for edge in prereq_edges if edge[0] != edge[1]})),
        "prereq_source": "kc_route_hierarchy",
    }


def _merge_item_concepts(
    target: dict[str, set[str]],
    source: Mapping[str, set[str]],
) -> None:
    for item_id, concepts in source.items():
        target.setdefault(str(item_id), set()).update(concepts)


def load_question_level_interactions(
    raw_dir: Path,
    *,
    include_official_test: bool = True,
    max_sequence_rows: int | None = None,
    chunksize: int = 2000,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict]:
    files = [("train_valid", raw_dir / "question_level" / "train_valid_sequences_quelevel.csv")]
    if include_official_test:
        files.append(("test", raw_dir / "question_level" / "test_quelevel.csv"))

    all_frames: list[pd.DataFrame] = []
    merged_item_concepts: dict[str, set[str]] = {}
    source_stats: dict[str, dict] = {}
    remaining = max_sequence_rows

    for source, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing XES3G5M sequence file: {path}")
        source_stats[source] = {
            "sequence_rows": 0,
            "expanded_positions": 0,
            "kept_positions": 0,
            "masked_positions": 0,
        }
        usecols = ["uid", "questions", "concepts", "responses", "timestamps"]
        header = pd.read_csv(path, nrows=0)
        if "selectmasks" in header.columns:
            usecols.append("selectmasks")

        iterator = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
        for chunk in tqdm(iterator, desc=f"expand {source}"):
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)
                remaining -= len(chunk)
            expanded, item_concepts, stats = expand_sequence_frame(chunk, source=source)
            all_frames.append(expanded)
            _merge_item_concepts(merged_item_concepts, item_concepts)
            for key, value in stats.items():
                source_stats[source][key] += int(value)
        if remaining is not None and remaining <= 0:
            break

    if not all_frames:
        raise ValueError("No XES3G5M sequence rows were loaded.")
    interactions = pd.concat(all_frames, ignore_index=True)
    return interactions, merged_item_concepts, source_stats


def save_xes_processed(
    stream_df: pd.DataFrame,
    stream_stats: dict,
    item_metadata: Mapping[str, ItemMetadata],
    prereq_edges: set[tuple[str, str]],
    *,
    raw_dir: Path,
    output_dir: Path,
    source_stats: Mapping[str, Mapping],
) -> None:
    ensure_dir(output_dir)
    item_ids = [str(x) for x in stream_stats["item_id_classes"]]
    content_emb = load_official_question_embeddings(raw_dir, item_ids)
    relation_stats = write_xes_relations(output_dir, item_ids, item_metadata, prereq_edges)

    stream_df.to_pickle(output_dir / "stream_data.pkl")
    torch.save(content_emb, output_dir / "content_emb.pt")

    meta = {
        "dataset": DATASET,
        "n_users": int(stream_stats["n_users"]),
        "n_items": int(stream_stats["n_items"]),
        "content_dim": int(content_emb.shape[1]),
        "n_interactions": int(stream_stats["interactions"]),
        "min_user_interactions": int(stream_stats["min_user_interactions"]),
        "min_item_interactions": int(stream_stats["min_item_interactions"]),
        "positive_only": bool(stream_stats["positive_only"]),
        "embedding_backend": "xes3g5m_official_roberta_content",
        "embedding_model": "XES3G5M released qid2content_emb.json",
        "relations": relation_stats,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"i_idx": range(len(item_ids)), "course_id": item_ids}).to_csv(
        output_dir / "_item_id_map.csv",
        index=False,
    )

    source_audit = {
        "dataset": DATASET,
        "raw_dir": str(raw_dir),
        "source_stats": source_stats,
        "processed": {
            "users": int(stream_stats["n_users"]),
            "items": int(stream_stats["n_items"]),
            "interactions": int(stream_stats["interactions"]),
            "content_dim": int(content_emb.shape[1]),
            "items_with_concept": int(relation_stats["items_with_concept"]),
            "concept_edges": int(relation_stats["concept_edges"]),
            "prereq_edges": int(relation_stats["concept_prereq_edges"]),
        },
        "notes": [
            "Interactions are expanded from released pyKT question-level sequence files.",
            "Positions with selectmasks <= 0 are excluded.",
            "Prerequisite edges are hierarchy-derived from question kc_routes, not explicit prerequisite annotations.",
            "content_emb.pt uses the official released qid2content_emb.json representations.",
        ],
    }
    (output_dir / "source_audit.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README_processed.txt").write_text(
        f"{DATASET} processed for FAST3-compatible educational recommendation.\n"
        f"Use USIM_DATA_DIR={output_dir}\n"
        f"Use USIM_RELATION_DIR={output_dir / 'relations'} for course-aware artifacts.\n"
        "Content embedding backend=xes3g5m_official_roberta_content\n"
        "Prerequisite graph source=kc_route_hierarchy\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--min-item-interactions", type=int, default=1)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap after expansion/filtering.")
    parser.add_argument("--max-sequence-rows", type=int, default=None, help="Optional sequence-row cap for smoke tests.")
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--train-only", action="store_true", help="Do not include official test_quelevel.csv.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    interactions, item_concepts, source_stats = load_question_level_interactions(
        raw_dir,
        include_official_test=not args.train_only,
        max_sequence_rows=args.max_sequence_rows,
        chunksize=int(args.chunksize),
    )
    questions = load_questions(raw_dir)
    item_metadata, prereq_edges = build_item_metadata(questions, item_concepts)

    spec = DatasetSpec(
        dataset=DATASET,
        raw_dir=raw_dir,
        output_dir=output_dir,
        min_user_interactions=int(args.min_user_interactions),
        min_item_interactions=int(args.min_item_interactions),
        positive_only=bool(args.positive_only),
        max_rows=args.max_rows,
    )
    stream_df, stream_stats = interaction_to_stream(
        interactions,
        user_col="user_id",
        item_col="course_id",
        timestamp_col="timestamp",
        correct_col="correct",
        spec=spec,
    )
    save_xes_processed(
        stream_df,
        stream_stats,
        item_metadata,
        prereq_edges,
        raw_dir=raw_dir,
        output_dir=output_dir,
        source_stats=source_stats,
    )

    print(f"[Done] {DATASET} processed to {output_dir}")
    print(
        f"       users={stream_stats['n_users']:,}, "
        f"items={stream_stats['n_items']:,}, interactions={stream_stats['interactions']:,}"
    )
    print(f"       set USIM_DATA_DIR={output_dir}")
    print(f"       set USIM_RELATION_DIR={output_dir / 'relations'}")


if __name__ == "__main__":
    main()
