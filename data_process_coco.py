"""Preprocess public COCO educational recommendation data for FAST3.

Expected public layout:

  data_raw/COCO_Educational_Recommendation_Dataset/preprocessed/ratings.txt
  data_raw/COCO_Educational_Recommendation_Dataset/preprocessed/i2kg_map.txt
  data_raw/COCO_Educational_Recommendation_Dataset/preprocessed/e_map.txt
  data_raw/COCO_Educational_Recommendation_Dataset/preprocessed/r_map.txt
  data_raw/COCO_Educational_Recommendation_Dataset/preprocessed/kg_final.txt

The public repository exposes a metadata KG but no official prerequisite
relation. This processor exports course-concept relations from metadata and
leaves prerequisite-dependency.json empty. For experiments, use
USIM_PREREQ_GRAPH_SOURCE=behavior unless a separate constructed prerequisite
graph is explicitly documented.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from edu_dataset_common import (
    DatasetSpec,
    ItemMetadata,
    add_common_args,
    find_first_existing,
    interaction_to_stream,
    make_spec,
    save_processed_dataset,
)


DATASET = "COCO"
DEFAULT_RAW = "data_raw/COCO_Educational_Recommendation_Dataset"
DEFAULT_OUT = "processed_data_coco"

RELATION_PREFIX = {
    "belong_to_category": "COCO_CATEGORY",
    "related_to_concept": "COCO_CONCEPT",
    "taught_in_level": "COCO_LEVEL",
    "taught_in_language": "COCO_LANGUAGE",
    "has_target_audience": "COCO_AUDIENCE",
}
CONSERVATIVE_CONCEPT_RELATIONS = {"belong_to_category", "related_to_concept"}
FULL_CONCEPT_RELATIONS = set(RELATION_PREFIX)
TEXT_RELATION_ORDER = [
    "belong_to_category",
    "related_to_concept",
    "taught_in_level",
    "taught_in_language",
    "has_target_audience",
]
TEXT_RELATION_LABEL = {
    "belong_to_category": "category",
    "related_to_concept": "concept",
    "taught_in_level": "level",
    "taught_in_language": "language",
    "has_target_audience": "audience",
}


@dataclass
class CocoTables:
    ratings: pd.DataFrame
    i2kg: pd.DataFrame
    entities: pd.DataFrame
    relations: pd.DataFrame
    kg: pd.DataFrame
    train: pd.DataFrame | None = None
    valid: pd.DataFrame | None = None
    test: pd.DataFrame | None = None


def _clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _slug_to_text(value) -> str:
    text = _clean_text(value)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(part for part in text.split() if part)


def _concept_token(prefix: str, label: str) -> str:
    label = _clean_text(label)
    return f"{prefix}:{label}" if label else ""


def _preprocessed_dir(raw_dir: Path) -> Path:
    if (raw_dir / "preprocessed").exists():
        return raw_dir / "preprocessed"
    return raw_dir


def _read_required(preprocessed_dir: Path, filename: str, **kwargs) -> pd.DataFrame:
    path = preprocessed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing COCO public preprocessed file: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def _read_optional_split(preprocessed_dir: Path, filename: str) -> pd.DataFrame | None:
    path = find_first_existing(preprocessed_dir, [filename])
    if path is None:
        return None
    return pd.read_csv(path, sep="\t", names=["uid", "pid", "rating", "timestamp"], low_memory=False)


def load_coco_tables(raw_dir: Path) -> CocoTables:
    pre = _preprocessed_dir(raw_dir)
    tables = CocoTables(
        ratings=_read_required(pre, "ratings.txt"),
        i2kg=_read_required(pre, "i2kg_map.txt"),
        entities=_read_required(pre, "e_map.txt"),
        relations=_read_required(pre, "r_map.txt"),
        kg=_read_required(pre, "kg_final.txt"),
        train=_read_optional_split(pre, "train.txt"),
        valid=_read_optional_split(pre, "valid.txt"),
        test=_read_optional_split(pre, "test.txt"),
    )
    required = {
        "ratings": {"uid", "pid", "rating", "timestamp"},
        "i2kg": {"eid", "pid", "name", "entity"},
        "entities": {"eid", "name", "entity"},
        "relations": {"id", "name"},
        "kg": {"entity_head", "relation", "entity_tail"},
    }
    for attr, cols in required.items():
        missing = cols - set(getattr(tables, attr).columns)
        if missing:
            raise ValueError(f"COCO {attr} missing columns: {sorted(missing)}")
    return tables


def _concept_relations_for_scope(concept_scope: str) -> set[str]:
    scope = concept_scope.strip().lower()
    if scope == "conservative":
        return set(CONSERVATIVE_CONCEPT_RELATIONS)
    if scope == "full":
        return set(FULL_CONCEPT_RELATIONS)
    if scope == "category_only":
        return {"belong_to_category"}
    raise ValueError(f"Unsupported COCO concept scope: {concept_scope}")


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_coco_metadata(
    tables: CocoTables,
    *,
    concept_scope: str = "conservative",
) -> tuple[dict[str, ItemMetadata], dict]:
    concept_relations = _concept_relations_for_scope(concept_scope)
    relation_id_to_name = {
        int(row.id): _clean_text(row.name)
        for row in tables.relations.itertuples(index=False)
        if _int_or_none(row.id) is not None
    }
    entity_id_to_name = {
        int(row.eid): _clean_text(row.name)
        for row in tables.entities.itertuples(index=False)
        if _int_or_none(row.eid) is not None
    }
    eid_to_pid: dict[int, str] = {}
    pid_to_slug: dict[str, str] = {}
    for row in tables.i2kg.itertuples(index=False):
        eid = _int_or_none(row.eid)
        pid = _clean_text(row.pid)
        if eid is None or not pid:
            continue
        eid_to_pid[eid] = pid
        pid_to_slug[pid] = _clean_text(row.name) or pid

    labels_by_item: dict[str, dict[str, set[str]]] = {
        pid: {rel: set() for rel in TEXT_RELATION_ORDER}
        for pid in pid_to_slug
    }
    concepts_by_item: dict[str, set[str]] = {pid: set() for pid in pid_to_slug}
    relation_coverage: dict[str, dict[str, int]] = {
        name: {"edges": 0, "items_covered": 0, "tail_unique": 0}
        for name in RELATION_PREFIX
    }
    relation_items: dict[str, set[str]] = {name: set() for name in RELATION_PREFIX}
    relation_tails: dict[str, set[str]] = {name: set() for name in RELATION_PREFIX}

    for row in tables.kg.itertuples(index=False):
        head = _int_or_none(row.entity_head)
        rel_id = _int_or_none(row.relation)
        tail = _int_or_none(row.entity_tail)
        if head is None or rel_id is None or tail is None:
            continue
        pid = eid_to_pid.get(head)
        relation_name = relation_id_to_name.get(rel_id, "")
        if pid is None or relation_name not in RELATION_PREFIX:
            continue
        label = entity_id_to_name.get(tail, "")
        if not label:
            continue

        labels_by_item.setdefault(pid, {rel: set() for rel in TEXT_RELATION_ORDER})
        labels_by_item[pid].setdefault(relation_name, set()).add(label)
        relation_items[relation_name].add(pid)
        relation_tails[relation_name].add(label)
        relation_coverage[relation_name]["edges"] += 1

        if relation_name in concept_relations:
            token = _concept_token(RELATION_PREFIX[relation_name], label)
            if token:
                concepts_by_item.setdefault(pid, set()).add(token)

    for relation_name in RELATION_PREFIX:
        relation_coverage[relation_name]["items_covered"] = len(relation_items[relation_name])
        relation_coverage[relation_name]["tail_unique"] = len(relation_tails[relation_name])

    metadata: dict[str, ItemMetadata] = {}
    for pid, slug in pid_to_slug.items():
        text_parts = [_slug_to_text(slug) or f"course {pid}"]
        item_labels = labels_by_item.get(pid, {})
        for relation_name in TEXT_RELATION_ORDER:
            labels = sorted(item_labels.get(relation_name, set()))
            if labels:
                text_parts.append(f"{TEXT_RELATION_LABEL[relation_name]} {' ; '.join(labels)}")

        categories = sorted(item_labels.get("belong_to_category", set()))
        metadata[pid] = ItemMetadata(
            item_id=pid,
            text=" ".join(part for part in text_parts if part),
            concepts=concepts_by_item.get(pid, set()),
            prerequisites=set(),
            family=categories[0] if categories else None,
        )

    audit = {
        "concept_scope": concept_scope,
        "items_in_i2kg": int(len(pid_to_slug)),
        "items_with_text": int(sum(1 for meta in metadata.values() if meta.text)),
        "items_with_concept": int(sum(1 for meta in metadata.values() if meta.concepts)),
        "concept_edges": int(sum(len(meta.concepts) for meta in metadata.values())),
        "relation_coverage": relation_coverage,
    }
    return metadata, audit


def _frame_stats(frame: pd.DataFrame | None) -> dict | None:
    if frame is None:
        return None
    return {
        "rows": int(len(frame)),
        "users": int(frame["uid"].nunique()),
        "items": int(frame["pid"].nunique()),
    }


def write_source_audit(
    tables: CocoTables,
    output_dir: Path,
    *,
    metadata_audit: Mapping,
    stream_stats: Mapping,
) -> None:
    ratings = tables.ratings
    user_len = ratings.groupby("uid").size()
    item_len = ratings.groupby("pid").size()
    timestamp = pd.to_numeric(ratings["timestamp"], errors="coerce")
    if timestamp.notna().any():
        time_min = str(pd.to_datetime(int(timestamp.min()), unit="s"))
        time_max = str(pd.to_datetime(int(timestamp.max()), unit="s"))
    else:
        time_min = None
        time_max = None

    audit = {
        "dataset": DATASET,
        "source": "public_preprocessed_coco_kg",
        "ratings": {
            "rows": int(len(ratings)),
            "users": int(ratings["uid"].nunique()),
            "items": int(ratings["pid"].nunique()),
            "duplicate_user_item": int(ratings.duplicated(["uid", "pid"]).sum()),
            "time_min": time_min,
            "time_max": time_max,
        },
        "provided_splits": {
            "train": _frame_stats(tables.train),
            "valid": _frame_stats(tables.valid),
            "test": _frame_stats(tables.test),
        },
        "interactions_per_user": {
            "min": int(user_len.min()),
            "median": float(user_len.median()),
            "mean": float(user_len.mean()),
            "max": int(user_len.max()),
        },
        "interactions_per_item": {
            "min": int(item_len.min()),
            "median": float(item_len.median()),
            "mean": float(item_len.mean()),
            "max": int(item_len.max()),
        },
        "kg": {
            "entities": int(len(tables.entities)),
            "item_entities": int(len(tables.i2kg)),
            "edges": int(len(tables.kg)),
            "relations": dict(metadata_audit["relation_coverage"]),
        },
        "processed": {
            "users": int(stream_stats["n_users"]),
            "items": int(stream_stats["n_items"]),
            "interactions": int(stream_stats["interactions"]),
            "concept_scope": metadata_audit["concept_scope"],
            "items_with_concept": int(metadata_audit["items_with_concept"]),
            "concept_edges": int(metadata_audit["concept_edges"]),
            "prerequisite_policy": "no_official_prerequisite_edges_use_behavior_source_for_experiments",
        },
    }
    with (output_dir / "source_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


def process_coco(spec: DatasetSpec, *, concept_scope: str = "conservative") -> None:
    tables = load_coco_tables(spec.raw_dir)
    stream_df, stream_stats = interaction_to_stream(
        tables.ratings,
        user_col="uid",
        item_col="pid",
        timestamp_col="timestamp",
        correct_col=None,
        spec=spec,
    )
    metadata, metadata_audit = build_coco_metadata(tables, concept_scope=concept_scope)
    save_processed_dataset(stream_df, stream_stats, metadata, spec)
    write_source_audit(
        tables,
        spec.output_dir,
        metadata_audit=metadata_audit,
        stream_stats=stream_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        dataset=DATASET,
        default_raw=DEFAULT_RAW,
        default_out=DEFAULT_OUT,
    )
    parser.add_argument(
        "--concept-scope",
        choices=["conservative", "full", "category_only"],
        default="conservative",
        help="Which COCO KG relations are exported as FAST3 course-concept edges.",
    )
    parser.set_defaults(
        embedding_backend="bert_cls",
        embedding_model="bert-base-uncased",
        embedding_max_length=256,
        embedding_batch_size=32,
    )
    args = parser.parse_args()
    spec = make_spec(args, DATASET)
    process_coco(spec, concept_scope=args.concept_scope)

    meta = json.loads((spec.output_dir / "meta.json").read_text(encoding="utf-8"))
    audit = json.loads((spec.output_dir / "source_audit.json").read_text(encoding="utf-8"))
    print(f"[Done] {DATASET} processed to {spec.output_dir}")
    print(f"       users={meta['n_users']:,}, items={meta['n_items']:,}, interactions={meta['n_interactions']:,}")
    print(
        "       concepts={0:,} items={1:,}/{2:,} scope={3}".format(
            meta["relations"]["concept_edges"],
            meta["relations"]["items_with_concept"],
            meta["n_items"],
            audit["processed"]["concept_scope"],
        )
    )
    print(f"       embedding={meta['embedding_backend']} model={meta['embedding_model']}")
    print(f"       set USIM_DATA_DIR={spec.output_dir}")
    print(f"       set USIM_RELATION_DIR={spec.output_dir / 'relations'}")
    print("       set USIM_PREREQ_GRAPH_SOURCE=behavior")


if __name__ == "__main__":
    main()
