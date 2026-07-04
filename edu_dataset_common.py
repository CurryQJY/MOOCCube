"""Shared preprocessing helpers for educational recommendation datasets.

The existing FAST3 training code expects a MOOCCube-like processed directory:

  - stream_data.pkl with user_id, course_id, timestamp, u_idx, i_idx, popularity
  - content_emb.pt with one row per encoded item
  - meta.json with n_users, n_items, content_dim

For exercise/knowledge-tracing datasets, `course_id` is kept as the generic
item id column for compatibility with the downstream code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


DEFAULT_CONTENT_DIM = 768


@dataclass
class ItemMetadata:
    item_id: str
    text: str = ""
    concepts: set[str] = field(default_factory=set)
    prerequisites: set[str] = field(default_factory=set)
    family: str | None = None


@dataclass
class DatasetSpec:
    dataset: str
    raw_dir: Path
    output_dir: Path
    min_user_interactions: int = 2
    min_item_interactions: int = 1
    positive_only: bool = False
    max_rows: int | None = None
    content_dim: int = DEFAULT_CONTENT_DIM
    embedding_backend: str = "stable_hash"
    embedding_model: str = ""
    embedding_max_length: int = 256
    embedding_batch_size: int = 32
    embedding_local_files_only: bool = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    if suffixes.endswith(".jsonl"):
        return pd.read_json(path, lines=True, **kwargs)
    if suffixes.endswith(".json"):
        return pd.read_json(path, **kwargs)
    sep = kwargs.pop("sep", None)
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False, **kwargs)


def find_first_existing(root: Path, candidates: Sequence[str]) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    lower_to_path = {str(p.relative_to(root)).replace("\\", "/").lower(): p for p in root.rglob("*") if p.is_file()}
    for rel in candidates:
        hit = lower_to_path.get(rel.replace("\\", "/").lower())
        if hit is not None:
            return hit
    return None


def first_present(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in columns}
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


def normalize_timestamp(series: pd.Series | None, n_rows: int) -> pd.Series:
    if series is None:
        return pd.Series(np.arange(n_rows, dtype=np.int64), dtype="int64")
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
        # EdNet timestamps are millisecond offsets; most model code expects seconds.
        if values.max() > 10_000_000_000:
            values = (values // 1000).astype("int64")
        return values
    parsed = pd.to_datetime(series, errors="coerce")
    fallback = pd.Series(np.arange(n_rows, dtype=np.int64), dtype="int64")
    values = (parsed.astype("int64") // 1_000_000_000).where(parsed.notna(), fallback)
    return values.astype("int64")


def parse_concepts(value) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    for sep in [";", "|", ",", " "]:
        if sep in text:
            return {part.strip() for part in text.split(sep) if part.strip()}
    return {text}


def stable_hash_embedding(text: str, dim: int = DEFAULT_CONTENT_DIM) -> np.ndarray:
    text = text or "unknown educational item"
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=16).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_hash_content_embeddings(texts: Sequence[str], dim: int = DEFAULT_CONTENT_DIM) -> torch.Tensor:
    matrix = np.vstack([stable_hash_embedding(text, dim=dim) for text in texts]).astype(np.float32)
    return torch.from_numpy(matrix)


def build_bert_cls_content_embeddings(
    texts: Sequence[str],
    *,
    model_name: str = "bert-base-chinese",
    max_length: int = 256,
    batch_size: int = 32,
    local_files_only: bool = False,
) -> torch.Tensor:
    """Encode item text with a fixed BERT [CLS] representation.

    This intentionally does not fall back to another model. Silent model
    fallback makes cross-dataset comparisons hard to audit.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"{model_name} CLS"):
            batch = list(texts[start : start + batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            all_embs.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(all_embs, dim=0)


def build_content_embeddings(
    texts: Sequence[str],
    *,
    spec: DatasetSpec,
) -> tuple[torch.Tensor, dict]:
    backend = spec.embedding_backend.strip().lower()
    if backend in {"stable_hash", "hash"}:
        content_emb = build_hash_content_embeddings(texts, dim=spec.content_dim)
        meta = {
            "embedding_backend": "stable_hash",
            "embedding_model": "",
            "embedding_max_length": None,
            "embedding_batch_size": None,
            "embedding_local_files_only": None,
        }
        return content_emb, meta
    if backend in {"bert_cls", "bert"}:
        model_name = spec.embedding_model or "bert-base-chinese"
        content_emb = build_bert_cls_content_embeddings(
            texts,
            model_name=model_name,
            max_length=spec.embedding_max_length,
            batch_size=spec.embedding_batch_size,
            local_files_only=spec.embedding_local_files_only,
        )
        meta = {
            "embedding_backend": "bert_cls",
            "embedding_model": model_name,
            "embedding_max_length": int(spec.embedding_max_length),
            "embedding_batch_size": int(spec.embedding_batch_size),
            "embedding_local_files_only": bool(spec.embedding_local_files_only),
        }
        return content_emb, meta
    raise ValueError(f"Unsupported embedding backend: {spec.embedding_backend}")


def build_item_metadata_from_table(
    table: pd.DataFrame,
    *,
    item_col: str,
    text_cols: Sequence[str] = (),
    concept_cols: Sequence[str] = (),
    prereq_cols: Sequence[str] = (),
    family_col: str | None = None,
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
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                value_s = str(value).strip()
                if value_s and value_s.lower() != "nan":
                    text_parts.append(value_s)
        concepts: set[str] = set()
        for col in concept_cols:
            concepts.update(parse_concepts(row_map.get(col)))
        prereqs: set[str] = set()
        for col in prereq_cols:
            prereqs.update(parse_concepts(row_map.get(col)))
        family = None
        if family_col is not None and family_col in row_map:
            raw_family = row_map.get(family_col)
            if raw_family is not None and not (isinstance(raw_family, float) and math.isnan(raw_family)):
                family = str(raw_family).strip() or None
        metadata[item_id] = ItemMetadata(
            item_id=item_id,
            text=" ".join(text_parts),
            concepts=concepts,
            prerequisites=prereqs,
            family=family,
        )
    return metadata


def coerce_correctness(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    text = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "correct", "right", "success"}
    falsy = {"0", "false", "f", "no", "n", "incorrect", "wrong", "fail", "failed"}
    values = pd.Series(np.nan, index=series.index, dtype="float64")
    values[text.isin(truthy)] = 1.0
    values[text.isin(falsy)] = 0.0
    return values


def interaction_to_stream(
    interactions: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    timestamp_col: str | None,
    correct_col: str | None,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, dict]:
    cols = [user_col, item_col]
    if timestamp_col:
        cols.append(timestamp_col)
    if correct_col:
        cols.append(correct_col)
    df = interactions[cols].copy()
    df = df.rename(columns={user_col: "user_id", item_col: "course_id"})
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["course_id"] = df["course_id"].astype(str).str.strip()
    df = df[(df["user_id"] != "") & (df["course_id"] != "")]
    df = df[(df["user_id"].str.lower() != "nan") & (df["course_id"].str.lower() != "nan")]
    if correct_col:
        correct = coerce_correctness(df[correct_col])
        if spec.positive_only:
            df = df[correct.fillna(0) > 0].copy()
    df["raw_time"] = df[timestamp_col].astype(str) if timestamp_col else ""
    df["timestamp"] = normalize_timestamp(df[timestamp_col] if timestamp_col else None, len(df))
    df = df[["user_id", "course_id", "raw_time", "timestamp"]].copy()
    df = df.sort_values(["timestamp", "user_id", "course_id"]).reset_index(drop=True)
    df = df.drop_duplicates(["user_id", "course_id"], keep="first")

    for _ in range(4):
        before = len(df)
        if spec.min_user_interactions > 1:
            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].map(user_counts) >= spec.min_user_interactions]
        if spec.min_item_interactions > 1:
            item_counts = df["course_id"].value_counts()
            df = df[df["course_id"].map(item_counts) >= spec.min_item_interactions]
        if len(df) == before:
            break

    if spec.max_rows is not None and len(df) > spec.max_rows:
        df = df.head(spec.max_rows).copy()

    if df.empty:
        raise ValueError("No interactions left after filtering.")

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["u_idx"] = user_enc.fit_transform(df["user_id"])
    df["i_idx"] = item_enc.fit_transform(df["course_id"])
    df = df.sort_values(["timestamp", "u_idx", "i_idx"]).reset_index(drop=True)
    df["popularity"] = df.groupby("i_idx").cumcount().astype("int64")

    stats = {
        "dataset": spec.dataset,
        "interactions": int(len(df)),
        "n_users": int(len(user_enc.classes_)),
        "n_items": int(len(item_enc.classes_)),
        "min_user_interactions": int(spec.min_user_interactions),
        "min_item_interactions": int(spec.min_item_interactions),
        "positive_only": bool(spec.positive_only),
        "user_id_classes": [str(x) for x in user_enc.classes_],
        "item_id_classes": [str(x) for x in item_enc.classes_],
    }
    return df, stats


def write_compat_relations(
    output_dir: Path,
    item_ids: Sequence[str],
    item_metadata: Mapping[str, ItemMetadata],
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
            for concept in sorted(meta.concepts):
                writer.writerow([str(item_id), concept])
                concept_edges += 1

    with (relation_dir / "prerequisite-dependency.json").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for item_id in item_ids:
            meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
            for concept in sorted(meta.concepts):
                for prereq in sorted(meta.prerequisites):
                    if concept != prereq:
                        writer.writerow([prereq, concept])
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
        "items_with_concept": int(sum(1 for item_id in item_ids if item_metadata.get(str(item_id), ItemMetadata(str(item_id))).concepts)),
        "concept_edges": int(concept_edges),
        "concept_prereq_edges": int(prereq_edges),
    }


def save_processed_dataset(
    df: pd.DataFrame,
    stats: dict,
    item_metadata: Mapping[str, ItemMetadata],
    spec: DatasetSpec,
) -> None:
    ensure_dir(spec.output_dir)
    item_ids = stats["item_id_classes"]
    texts = []
    for item_id in item_ids:
        meta = item_metadata.get(str(item_id), ItemMetadata(str(item_id)))
        concepts = " ".join(sorted(meta.concepts))
        texts.append(" ".join(part for part in [meta.text, concepts, str(item_id)] if part))

    content_emb, embedding_meta = build_content_embeddings(texts, spec=spec)
    relation_stats = write_compat_relations(spec.output_dir, item_ids, item_metadata)

    df.to_pickle(spec.output_dir / "stream_data.pkl")
    torch.save(content_emb, spec.output_dir / "content_emb.pt")
    with (spec.output_dir / "meta.json").open("w", encoding="utf-8") as f:
        meta = {
            "dataset": spec.dataset,
            "n_users": stats["n_users"],
            "n_items": stats["n_items"],
            "content_dim": int(content_emb.shape[1]),
            "n_interactions": stats["interactions"],
            "min_user_interactions": stats["min_user_interactions"],
            "min_item_interactions": stats["min_item_interactions"],
            "positive_only": stats["positive_only"],
            **embedding_meta,
            "relations": relation_stats,
        }
        json.dump(meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"i_idx": range(len(item_ids)), "course_id": item_ids}).to_csv(
        spec.output_dir / "_item_id_map.csv",
        index=False,
    )
    with (spec.output_dir / "README_processed.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"{spec.dataset} processed for FAST3-compatible educational recommendation.\n"
            f"Use USIM_DATA_DIR={spec.output_dir}\n"
            f"Use USIM_RELATION_DIR={spec.output_dir / 'relations'} for course-aware artifacts.\n"
            f"Content embedding backend={embedding_meta['embedding_backend']}"
            + (f", model={embedding_meta['embedding_model']}\n" if embedding_meta["embedding_model"] else "\n")
        )


def add_common_args(parser: argparse.ArgumentParser, *, dataset: str, default_raw: str, default_out: str) -> None:
    parser.add_argument("--raw-dir", default=default_raw, help=f"Raw {dataset} directory.")
    parser.add_argument("--output-dir", default=default_out, help="Processed output directory.")
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--min-item-interactions", type=int, default=1)
    parser.add_argument("--positive-only", action="store_true", help="Keep only correct/positive attempts when a correctness column exists.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for a quick smoke-test subset.")
    parser.add_argument("--content-dim", type=int, default=DEFAULT_CONTENT_DIM)
    parser.add_argument("--embedding-backend", choices=["stable_hash", "bert_cls"], default="stable_hash")
    parser.add_argument("--embedding-model", default="", help="Transformer model for --embedding-backend bert_cls.")
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-local-files-only", action="store_true")


def make_spec(args: argparse.Namespace, dataset: str) -> DatasetSpec:
    return DatasetSpec(
        dataset=dataset,
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        min_user_interactions=int(args.min_user_interactions),
        min_item_interactions=int(args.min_item_interactions),
        positive_only=bool(args.positive_only),
        max_rows=args.max_rows,
        content_dim=int(args.content_dim),
        embedding_backend=str(args.embedding_backend),
        embedding_model=str(args.embedding_model or ""),
        embedding_max_length=int(args.embedding_max_length),
        embedding_batch_size=int(args.embedding_batch_size),
        embedding_local_files_only=bool(args.embedding_local_files_only),
    )
