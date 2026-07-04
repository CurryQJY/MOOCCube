"""Official-source PAM adapter for the shared static item-cold protocol.

The upstream PAM implementation remains under ``.runtime_tmp/PAM`` unchanged.
This adapter:

1. Converts a project static split to the official PAM-F CSV layout.
2. Imports the official ``EmbMLP``/``Engine``/``BatchLoader`` code.
3. Trains the official TensorFlow 1.x graph.
4. Recomputes full-catalog static item-cold metrics with item-macro output.

Run with the TensorFlow 1.15 Python environment, not the project Python 3.12
runner. The bundled runner prepends the TF1 conda DLL directories automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PAM_ROOT = REPO_ROOT / ".runtime_tmp" / "PAM"
METRICS = [f"{m}@{k}" for m in ("R", "N") for k in (5, 10, 20)]
K_LIST = (5, 10, 20)
MAX_SEQ_LEN = 30


@dataclass(frozen=True)
class EvalTarget:
    user_id: int
    item_id: int
    popularity: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_first(names: Sequence[str], default: str = "") -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _result_path(cfg: "Config", filename: str) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir / filename


def _seq_to_str(seq: Sequence[int]) -> str:
    return "#".join(str(int(x)) for x in seq[-MAX_SEQ_LEN:] if int(x) >= 0)


def _seq_matrix(seq: Sequence[int]) -> Tuple[List[int], int]:
    clipped = [int(x) for x in seq[-MAX_SEQ_LEN:] if int(x) >= 0]
    out = clipped + [0] * (MAX_SEQ_LEN - len(clipped))
    return out[:MAX_SEQ_LEN], min(len(clipped), MAX_SEQ_LEN)


def _safe_mod(v: int, modulo: int) -> int:
    return int(v) % int(modulo) if modulo > 0 else 0


def _git_value(cwd: Path, args: Sequence[str]) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return res.stdout.strip()


def pam_git_info(pam_root: Path) -> Dict[str, object]:
    tracked = _git_value(pam_root, ["status", "--short", "--untracked-files=no"])
    runtime = _git_value(pam_root, ["status", "--short"])
    return {
        "official_repo": _git_value(pam_root, ["remote", "get-url", "origin"])
        or "https://github.com/Sycamoretail/PAM",
        "official_commit": _git_value(pam_root, ["rev-parse", "--short", "HEAD"]),
        "official_tree_clean": tracked == "",
        "official_status_short": tracked or "",
        "official_runtime_status_short": runtime or "",
    }


def load_meta_and_stream(data_dir: str) -> Tuple[dict, pd.DataFrame]:
    root = Path(data_dir)
    meta_path = root / "meta.json"
    stream_path = root / "stream_data.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    if not stream_path.exists():
        raise FileNotFoundError(f"Missing stream_data.pkl: {stream_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(stream_path)
    required = {"u_idx", "i_idx", "timestamp", "popularity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stream_data.pkl missing required columns: {sorted(missing)}")
    return meta, df


def load_split_cold_threshold(split_dir: str, fallback: int) -> int:
    if split_dir:
        summary_path = Path(split_dir) / "static_split_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if "cold_threshold" in summary:
                return int(summary["cold_threshold"])
        counts_path = Path(split_dir) / "static_split_counts.csv"
        if counts_path.exists():
            counts = pd.read_csv(counts_path)
            if "cold_threshold" in counts.columns and len(counts) > 0:
                values = counts["cold_threshold"].dropna().unique()
                if len(values):
                    return int(values[0])
    return int(fallback)


def load_static_split(
    df: pd.DataFrame,
    split_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_dir:
        root = Path(split_dir)
        paths = [root / "static_train.pkl", root / "static_val.pkl", root / "static_test.pkl"]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Static split files missing: " + ", ".join(missing))
        train_df = pd.read_pickle(paths[0]).copy()
        val_df = pd.read_pickle(paths[1]).copy()
        test_df = pd.read_pickle(paths[2]).copy()
        print(
            f"Loaded shared static split from {root}: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
            flush=True,
        )
        return train_df, val_df, test_df

    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    if n_train < 1 or n_val < 1 or n_train + n_val >= n_total:
        raise ValueError("Invalid split sizes for fallback static split")
    return (
        df_shuffled.iloc[:n_train].copy(),
        df_shuffled.iloc[n_train : n_train + n_val].copy(),
        df_shuffled.iloc[n_train + n_val :].copy(),
    )


def infer_relation_dir(data_dir: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    data_path = Path(data_dir)
    local = data_path / "relations"
    if local.exists():
        return str(local)
    name = data_path.name.lower()
    if "hin" in name and (REPO_ROOT / "MOOCCube" / "relations").exists():
        return str(REPO_ROOT / "MOOCCube" / "relations")
    if "mooccubex" in name and (REPO_ROOT / "MOOCCubeX" / "relations").exists():
        return str(REPO_ROOT / "MOOCCubeX" / "relations")
    return ""


def _item_lookup(df: pd.DataFrame, n_items: int) -> Dict[int, str]:
    if "course_id" not in df.columns:
        return {idx: str(idx) for idx in range(n_items)}
    lookup: Dict[int, str] = {}
    subset = df[["i_idx", "course_id"]].drop_duplicates("i_idx")
    for row in subset.itertuples(index=False):
        lookup[int(row.i_idx)] = str(row.course_id)
    for idx in range(n_items):
        lookup.setdefault(idx, str(idx))
    return lookup


def read_course_concepts(relation_dir: str) -> Tuple[Dict[str, List[str]], str]:
    if not relation_dir:
        return {}, ""
    path = Path(relation_dir) / "course-concept.json"
    if not path.exists() or path.stat().st_size == 0:
        return {}, ""

    out: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            item_id = str(row[0]).strip()
            concept = str(row[1]).strip()
            if item_id and concept:
                out[item_id].append(concept)
    return dict(out), str(path)


def build_pam_content_frame(
    *,
    df: pd.DataFrame,
    n_items: int,
    relation_dir: str,
    max_cates_per_item: int,
) -> Tuple[pd.DataFrame, Dict[str, object], int]:
    lookup = _item_lookup(df, n_items)
    concepts_by_item, source = read_course_concepts(relation_dir)
    concept_to_id: Dict[str, int] = {"__NO_CONCEPT__": 1}
    rows = []
    matched_items = 0

    for item_idx in range(n_items):
        raw_id = lookup.get(item_idx, str(item_idx))
        concepts = sorted(set(concepts_by_item.get(raw_id, [])))
        if concepts:
            matched_items += 1
        else:
            concepts = ["__NO_CONCEPT__"]
        cate_ids = []
        for concept in concepts[: max(1, max_cates_per_item)]:
            if concept not in concept_to_id:
                concept_to_id[concept] = len(concept_to_id) + 1
            cate_ids.append(concept_to_id[concept])
        rows.append({"itemId": int(item_idx), "cateId": "#".join(str(x) for x in cate_ids)})

    info = {
        "category_source": source or "fallback:__NO_CONCEPT__",
        "items_with_relation_concepts": int(matched_items),
        "n_concept_tokens": int(len(concept_to_id)),
        "max_cates_per_item": int(max_cates_per_item),
    }
    # +1 because category id 0 is used as padding in the official model.
    return pd.DataFrame(rows), info, int(max(concept_to_id.values()) + 1)


def _current_vv(row_popularity: int, count_so_far: int) -> int:
    return max(int(row_popularity), int(count_so_far))


def build_pam_training_frame(
    *,
    train_df: pd.DataFrame,
    n_items: int,
    seed: int,
    neg_per_pos: int,
    max_train_pos: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, List[int]], Dict[int, set], Dict[int, int], int]:
    rng = np.random.default_rng(seed)
    sorted_df = train_df.sort_values(["timestamp", "u_idx", "i_idx"]).reset_index(drop=True)
    if max_train_pos > 0:
        sorted_df = sorted_df.head(max_train_pos).copy()

    user_hist: Dict[int, List[int]] = defaultdict(list)
    item_user_hist: Dict[int, List[int]] = defaultdict(list)
    user_seen: Dict[int, set] = defaultdict(set)
    item_counts: Dict[int, int] = defaultdict(int)
    rows = []

    for row in sorted_df.itertuples(index=False):
        u = int(row.u_idx)
        i = int(row.i_idx)
        item_seq = list(user_hist[u])
        user_seq = list(item_user_hist[i])
        vv = _current_vv(int(row.popularity), item_counts[i])
        rows.append(
            {
                "userId": u,
                "itemSeq": _seq_to_str(item_seq),
                "itemId": i,
                "userSeq": _seq_to_str(user_seq),
                "label": 1,
                "vv": vv,
                "period": 0,
            }
        )

        for _ in range(max(0, neg_per_pos)):
            neg = int(rng.integers(0, n_items))
            for _attempt in range(64):
                if neg != i and neg not in user_seen[u]:
                    break
                neg = int(rng.integers(0, n_items))
            rows.append(
                {
                    "userId": u,
                    "itemSeq": _seq_to_str(item_seq),
                    "itemId": neg,
                    "userSeq": _seq_to_str(item_user_hist[neg]),
                    "label": 0,
                    "vv": int(item_counts.get(neg, 0)),
                    "period": 0,
                }
            )

        user_hist[u].append(i)
        if len(user_hist[u]) > MAX_SEQ_LEN:
            del user_hist[u][:-MAX_SEQ_LEN]
        item_user_hist[i].append(u)
        if len(item_user_hist[i]) > MAX_SEQ_LEN:
            del item_user_hist[i][:-MAX_SEQ_LEN]
        user_seen[u].add(i)
        item_counts[i] += 1

    train_frame = pd.DataFrame(rows, columns=["userId", "itemSeq", "itemId", "userSeq", "label", "vv", "period"])
    if batch_size < 1:
        raise ValueError("PAM batch_size must be >= 1")
    trim = len(train_frame) % batch_size
    if trim:
        train_frame = train_frame.iloc[:-trim].copy()
    if train_frame.empty:
        raise ValueError(
            f"PAM train view is empty after trimming to batch_size={batch_size}. "
            "Increase max_train_pos or reduce batch_size."
        )
    return train_frame, dict(user_hist), dict(item_user_hist), dict(user_seen), dict(item_counts), int(trim)


def build_eval_targets(df: pd.DataFrame, cold_threshold: int) -> Tuple[List[EvalTarget], List[EvalTarget]]:
    cold: List[EvalTarget] = []
    hot: List[EvalTarget] = []
    for row in df.itertuples(index=False):
        target = EvalTarget(int(row.u_idx), int(row.i_idx), int(row.popularity))
        if int(row.popularity) < int(cold_threshold):
            cold.append(target)
        else:
            hot.append(target)
    return cold, hot


def limit_eval_rows(df: pd.DataFrame, max_rows: int, cold_threshold: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    cold = df.loc[df["popularity"].astype(int) < int(cold_threshold)]
    hot = df.loc[df["popularity"].astype(int) >= int(cold_threshold)]
    if cold.empty or hot.empty:
        return df.head(max_rows).copy()

    cold_quota = min(len(cold), max(1, max_rows // 2))
    hot_quota = min(len(hot), max_rows - cold_quota)
    remaining = max_rows - cold_quota - hot_quota
    if remaining > 0:
        cold_extra = min(len(cold) - cold_quota, remaining)
        cold_quota += cold_extra
        remaining -= cold_extra
    if remaining > 0:
        hot_quota += min(len(hot) - hot_quota, remaining)
    limited = pd.concat([cold.head(cold_quota), hot.head(hot_quota)], axis=0)
    return limited.sort_index().copy()


def export_pam_dataset_view(
    *,
    data_dir: str,
    relation_dir: str,
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    seed: int,
    cold_threshold: int,
    batch_size: int,
    neg_per_pos: int,
    max_train_pos: int,
    max_eval_rows: int,
    max_cates_per_item: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    content_df, category_info, num_cates = build_pam_content_frame(
        df=all_df,
        n_items=n_items,
        relation_dir=relation_dir,
        max_cates_per_item=max_cates_per_item,
    )
    train_frame, user_hist, item_user_hist, user_seen, item_counts, trim = build_pam_training_frame(
        train_df=train_df,
        n_items=n_items,
        seed=seed,
        neg_per_pos=neg_per_pos,
        max_train_pos=max_train_pos,
        batch_size=batch_size,
    )

    val_export = limit_eval_rows(val_df, max_eval_rows, cold_threshold)
    test_export = limit_eval_rows(test_df, max_eval_rows, cold_threshold)

    train_frame.to_csv(output_dir / "pam_train.csv", index=False)
    content_df.to_csv(output_dir / "pam_content.csv", index=False)
    train_df[["u_idx", "i_idx", "timestamp", "popularity"]].to_csv(
        output_dir / "pam_train_interactions.csv",
        index=False,
    )
    val_export[["u_idx", "i_idx", "timestamp", "popularity"]].to_csv(
        output_dir / "pam_val_targets.csv",
        index=False,
    )
    test_export[["u_idx", "i_idx", "timestamp", "popularity"]].to_csv(
        output_dir / "pam_test_targets.csv",
        index=False,
    )
    np.save(output_dir / "his_dict.npy", {int(k): set(v) for k, v in item_user_hist.items()}, allow_pickle=True)

    manifest = {
        "official_format": "PAM-F csv",
        "data_dir": str(data_dir),
        "relation_dir": str(relation_dir),
        "seed": int(seed),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "num_cates": int(num_cates),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_export)),
        "test_rows": int(len(test_export)),
        "batch_size": int(batch_size),
        "neg_per_pos": int(neg_per_pos),
        "max_train_pos": int(max_train_pos),
        "max_eval_rows": int(max_eval_rows),
        "trimmed_train_rows_mod_batch": int(trim),
        "cold_threshold": int(cold_threshold),
        "user_history_count": int(len(user_hist)),
        "item_user_history_count": int(len(item_user_hist)),
        "item_count_nonzero": int(len(item_counts)),
        **category_info,
    }
    with (output_dir / "pam_static_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _load_pam_csv(view_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(view_dir / "pam_train.csv")
    content = pd.read_csv(view_dir / "pam_content.csv")
    for col in ["itemSeq", "userSeq"]:
        if col in train.columns:
            train[col] = train[col].fillna("").apply(_parse_hash_seq)
    if "cateId" in content.columns:
        content["cateId"] = content["cateId"].fillna("")
    return train, content


def load_manifest(view_dir: Path) -> Dict[str, object]:
    manifest_path = view_dir / "pam_static_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing PAM manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_required_interaction_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing PAM interaction CSV: {path}")
    df = pd.read_csv(path)
    required = {"u_idx", "i_idx", "timestamp", "popularity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _parse_hash_seq(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text:
        return []
    out = []
    for part in text.split("#"):
        part = str(part).strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def _process_cates(content_df: pd.DataFrame):
    cate_ls = [_parse_hash_seq(x) for x in content_df.sort_values("itemId")["cateId"].tolist()]
    max_len = max(1, max((len(x) for x in cate_ls), default=1))
    cate_matrix = np.zeros((len(cate_ls), max_len), dtype=np.int32)
    cate_lens = []
    for row, seq in enumerate(cate_ls):
        clipped = seq[:max_len]
        cate_lens.append(len(clipped))
        if clipped:
            cate_matrix[row, : len(clipped)] = np.asarray(clipped, dtype=np.int32)
    return cate_matrix, cate_lens


def _build_eval_histories(train_df: pd.DataFrame, n_items: int):
    user_hist: Dict[int, List[int]] = defaultdict(list)
    item_user_hist: Dict[int, List[int]] = defaultdict(list)
    user_seen: Dict[int, set] = defaultdict(set)
    item_counts = np.zeros(n_items, dtype=np.int64)
    for row in train_df.sort_values(["timestamp", "u_idx", "i_idx"]).itertuples(index=False):
        u = int(row.u_idx)
        i = int(row.i_idx)
        user_hist[u].append(i)
        if len(user_hist[u]) > MAX_SEQ_LEN:
            del user_hist[u][:-MAX_SEQ_LEN]
        item_user_hist[i].append(u)
        if len(item_user_hist[i]) > MAX_SEQ_LEN:
            del item_user_hist[i][:-MAX_SEQ_LEN]
        user_seen[u].add(i)
        item_counts[i] += 1
    return dict(user_hist), dict(item_user_hist), dict(user_seen), item_counts


def _batch_score_items(
    *,
    sess,
    model,
    user_id: int,
    item_ids: np.ndarray,
    user_hist: Dict[int, List[int]],
    item_user_hist: Dict[int, List[int]],
    item_counts: np.ndarray,
):
    user_seq, user_seq_len = _seq_matrix(user_hist.get(int(user_id), []))
    hist_i = np.asarray([user_seq] * len(item_ids), dtype=np.int32)
    hist_i_len = np.asarray([user_seq_len] * len(item_ids), dtype=np.int32)
    hist_u_rows = []
    hist_u_lens = []
    for item_id in item_ids:
        row, row_len = _seq_matrix(item_user_hist.get(int(item_id), []))
        hist_u_rows.append(row)
        hist_u_lens.append(row_len)
    feed = {
        model.u: np.full(len(item_ids), int(user_id), dtype=np.int32),
        model.i: item_ids.astype(np.int32),
        model.hist_i: np.asarray(hist_i, dtype=np.int32),
        model.hist_i_len: hist_i_len.astype(np.int32),
        model.hist_u: np.asarray(hist_u_rows, dtype=np.int32),
        model.hist_u_len: np.asarray(hist_u_lens, dtype=np.int32),
        model.y: np.zeros(len(item_ids), dtype=np.float32),
        model.vv_group: np.asarray([_vv_group(int(item_counts[int(i)])) for i in item_ids], dtype=np.int32),
        model.store: np.zeros(len(item_ids), dtype=np.bool_),
    }
    return sess.run(model.scores, feed_dict=feed)


def _vv_group(vv: int) -> int:
    thresholds = [-1, 5, 50, 200, 1000, 5000, float("inf")]
    group = 0
    while not (vv > thresholds[group] and vv <= thresholds[group + 1]):
        group += 1
    return group


def _score_all_items_for_user(
    *,
    sess,
    model,
    user_id: int,
    n_items: int,
    user_hist: Dict[int, List[int]],
    item_user_hist: Dict[int, List[int]],
    item_counts: np.ndarray,
    item_batch_size: int,
) -> np.ndarray:
    scores = np.empty(n_items, dtype=np.float32)
    for start in range(0, n_items, item_batch_size):
        end = min(n_items, start + item_batch_size)
        item_ids = np.arange(start, end, dtype=np.int32)
        scores[start:end] = _batch_score_items(
            sess=sess,
            model=model,
            user_id=user_id,
            item_ids=item_ids,
            user_hist=user_hist,
            item_user_hist=item_user_hist,
            item_counts=item_counts,
        )
    return scores


def _metric_values(scores: np.ndarray, target: int) -> Dict[str, float]:
    max_k = min(max(K_LIST), scores.shape[0])
    if max_k < scores.shape[0]:
        idx = np.argpartition(-scores, max_k - 1)[:max_k]
        idx = idx[np.argsort(-scores[idx])]
    else:
        idx = np.argsort(-scores)
    out: Dict[str, float] = {}
    for k in K_LIST:
        top = idx[: min(k, len(idx))]
        hit_pos = np.where(top == int(target))[0]
        hit = float(hit_pos.size > 0)
        out[f"R@{k}"] = hit
        out[f"N@{k}"] = float(1.0 / np.log2(hit_pos[0] + 2.0)) if hit else 0.0
    return out


def _aggregate_metrics(rows: List[Tuple[int, Dict[str, float]]]) -> Tuple[Dict[str, float], Dict[str, float], int, int]:
    if not rows:
        return {}, {}, 0, 0
    interaction = {key: float(np.mean([m[key] for _, m in rows])) for key in METRICS}
    per_item: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for item_id, metric in rows:
        per_item[int(item_id)].append(metric)
    item_macro = {}
    for key in METRICS:
        item_values = [float(np.mean([m[key] for m in vals])) for vals in per_item.values()]
        item_macro[key] = float(np.mean(item_values)) if item_values else 0.0
    return interaction, item_macro, len(rows), len(per_item)


def evaluate_pam_full_catalog(
    *,
    sess,
    model,
    targets: Sequence[EvalTarget],
    n_items: int,
    user_hist: Dict[int, List[int]],
    item_user_hist: Dict[int, List[int]],
    user_seen: Dict[int, set],
    item_counts: np.ndarray,
    item_batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], int, int]:
    by_user: Dict[int, List[EvalTarget]] = defaultdict(list)
    for target in targets:
        by_user[int(target.user_id)].append(target)

    rows: List[Tuple[int, Dict[str, float]]] = []
    for pos, (user_id, user_targets) in enumerate(by_user.items(), start=1):
        if pos == 1 or pos % 1000 == 0:
            print(f"  eval users {pos}/{len(by_user)}", flush=True)
        base_scores = _score_all_items_for_user(
            sess=sess,
            model=model,
            user_id=user_id,
            n_items=n_items,
            user_hist=user_hist,
            item_user_hist=item_user_hist,
            item_counts=item_counts,
            item_batch_size=item_batch_size,
        )
        seen = [item for item in user_seen.get(int(user_id), set()) if 0 <= int(item) < n_items]
        for target in user_targets:
            scores = base_scores.copy()
            if seen:
                scores[np.asarray(seen, dtype=np.int64)] = -1e9
            if 0 <= target.item_id < n_items:
                scores[target.item_id] = base_scores[target.item_id]
            rows.append((int(target.item_id), _metric_values(scores, int(target.item_id))))
    return _aggregate_metrics(rows)


@dataclass
class Config:
    data_dir: str
    split_dir: str
    relation_dir: str
    output_dir: Path
    pam_root: Path
    seed: int
    static_seed: int
    cold_threshold: int
    train_ratio: float
    val_ratio: float
    epochs: int
    batch_size: int
    lr: float
    emb_dim: int
    hidden_dim: int
    cate_dim: int
    neg_per_pos: int
    max_train_pos: int
    max_eval_rows: int
    max_cates_per_item: int
    eval_item_batch_size: int
    use_gpu: bool
    init_checkpoint: str
    start_epoch: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        data_dir = _env_first(["PAM_DATA_DIR", "USIM_DATA_DIR"], args.data_dir)
        split_dir = _env_first(["PAM_STATIC_SPLIT_DIR", "USIM_STATIC_SPLIT_DIR"], args.split_dir)
        relation_dir = infer_relation_dir(data_dir, _env_first(["PAM_RELATION_DIR", "USIM_RELATION_DIR"], args.relation_dir))
        output_raw = _env_first(["PAM_BASELINE_OUTPUT_DIR", "USIM_BASELINE_OUTPUT_DIR"], args.output_dir)
        output_dir = Path(output_raw or "outputs/pam_official_static").resolve()
        seed = _env_int("PAM_SEED", _env_int("USIM_STATIC_SEED", args.seed))
        static_seed = _env_int("PAM_STATIC_SEED", seed)
        fallback_threshold = _env_int("PAM_COLD_THRESHOLD", _env_int("USIM_COLD_THRESHOLD", args.cold_threshold))
        return cls(
            data_dir=data_dir,
            split_dir=split_dir,
            relation_dir=relation_dir,
            output_dir=output_dir,
            pam_root=Path(_env_first(["PAM_ROOT"], args.pam_root)).resolve(),
            seed=seed,
            static_seed=static_seed,
            cold_threshold=load_split_cold_threshold(split_dir, fallback_threshold),
            train_ratio=_env_float("PAM_STATIC_TRAIN_RATIO", args.train_ratio),
            val_ratio=_env_float("PAM_STATIC_VAL_RATIO", args.val_ratio),
            epochs=_env_int("PAM_EPOCHS", args.epochs),
            batch_size=_env_int("PAM_BATCH_SIZE", args.batch_size),
            lr=_env_float("PAM_LR", args.lr),
            emb_dim=_env_int("PAM_EMB_DIM", args.emb_dim),
            hidden_dim=_env_int("PAM_HIDDEN_DIM", args.hidden_dim),
            cate_dim=_env_int("PAM_CATE_DIM", args.cate_dim),
            neg_per_pos=_env_int("PAM_NEG_PER_POS", args.neg_per_pos),
            max_train_pos=_env_int("PAM_MAX_TRAIN_POS", args.max_train_pos),
            max_eval_rows=_env_int("PAM_MAX_EVAL_ROWS", args.max_eval_rows),
            max_cates_per_item=_env_int("PAM_MAX_CATES_PER_ITEM", args.max_cates_per_item),
            eval_item_batch_size=_env_int("PAM_EVAL_ITEM_BATCH_SIZE", args.eval_item_batch_size),
            use_gpu=_env_bool("PAM_USE_GPU", args.use_gpu),
            init_checkpoint=_env_first(["PAM_INIT_CKPT"], args.init_checkpoint),
            start_epoch=_env_int("PAM_START_EPOCH", args.start_epoch),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "export", "train_eval"], default=os.environ.get("PAM_MODE", "full"))
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--relation-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pam-root", default=str(DEFAULT_PAM_ROOT))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--cate-dim", type=int, default=8)
    parser.add_argument("--neg-per-pos", type=int, default=1)
    parser.add_argument("--max-train-pos", type=int, default=4096)
    parser.add_argument("--max-eval-rows", type=int, default=2048)
    parser.add_argument("--max-cates-per-item", type=int, default=8)
    parser.add_argument("--eval-item-batch-size", type=int, default=1024)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    return parser.parse_args()


def train_and_evaluate(cfg: Config, manifest: Dict[str, object]):
    if not cfg.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    pam_code_dir = cfg.pam_root / "PAM-F"
    if not (pam_code_dir / "model.py").exists():
        raise FileNotFoundError(f"Missing official PAM-F model.py: {pam_code_dir / 'model.py'}")
    sys.path.insert(0, str(pam_code_dir))
    try:
        import tensorflow.compat.v1 as tf  # type: ignore
        from engine import Engine  # type: ignore
        from model import EmbMLP  # type: ignore
    finally:
        try:
            sys.path.remove(str(pam_code_dir))
        except ValueError:
            pass

    if not hasattr(tf, "Session"):
        raise RuntimeError("PAM official adapter requires TensorFlow 1.x compatibility mode")
    tf.disable_eager_execution()
    tf.set_random_seed(cfg.seed)

    view_dir = cfg.output_dir / "pam_official_view"
    train_view, content_view = _load_pam_csv(view_dir)
    train_df = _load_required_interaction_csv(view_dir / "pam_train_interactions.csv")
    test_df = _load_required_interaction_csv(view_dir / "pam_test_targets.csv")
    cates, cate_lens = _process_cates(content_view)
    n_users = int(manifest["n_users"])
    n_items = int(manifest["n_items"])
    num_cates = int(manifest["num_cates"])
    hyperparams = {
        "num_users": n_users,
        "num_items": n_items,
        "num_cates": num_cates,
        "user_embed_dim": cfg.emb_dim,
        "item_embed_dim": cfg.emb_dim,
        "cate_embed_dim": cfg.cate_dim,
        "layers": [cfg.emb_dim + cfg.emb_dim + cfg.cate_dim, cfg.hidden_dim, cfg.emb_dim, 1],
    }
    train_config = {
        "base_optimizer": "adam",
        "base_lr": cfg.lr,
        "base_bs": cfg.batch_size,
        "base_num_epochs": cfg.epochs,
        "shuffle": False,
    }

    user_hist, item_user_hist, user_seen, item_counts = _build_eval_histories(train_df, n_items)
    cold_targets, hot_targets = build_eval_targets(test_df, cfg.cold_threshold)

    tf.reset_default_graph()
    model = EmbMLP(cates, cate_lens, hyperparams, train_config=train_config)
    ckpt_dir = cfg.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "pam_official_latest.ckpt"

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if cfg.init_checkpoint:
            ckpt_index = Path(cfg.init_checkpoint + ".index")
            if not ckpt_index.exists():
                raise FileNotFoundError(f"PAM init checkpoint is missing index file: {ckpt_index}")
            if cfg.start_epoch < 1:
                raise ValueError("PAM_START_EPOCH must be positive when PAM_INIT_CKPT is set.")
            if cfg.start_epoch >= cfg.epochs:
                print(
                    f"Restoring PAM checkpoint from {cfg.init_checkpoint}; "
                    f"start_epoch={cfg.start_epoch} >= epochs={cfg.epochs}, evaluation only.",
                    flush=True,
                )
            else:
                print(
                    f"Restoring PAM checkpoint from {cfg.init_checkpoint}; "
                    f"continuing epochs {cfg.start_epoch + 1}/{cfg.epochs}.",
                    flush=True,
                )
            saver.restore(sess, cfg.init_checkpoint)
        engine = Engine(sess, model, {})
        start = time.time()
        last_loss = None
        for epoch in range(cfg.start_epoch + 1, cfg.epochs + 1):
            print(f"Training official PAM epoch {epoch}/{cfg.epochs} ...", flush=True)
            last_loss = engine.base_train_an_epoch(epoch, train_view, train_config)
            print(f"Epoch {epoch} loss={last_loss:.6f}", flush=True)
            saver.save(sess, str(latest_ckpt))

        print("Evaluating full-catalog cold targets ...", flush=True)
        full_cold, full_cold_item, n_fc, n_fc_item = evaluate_pam_full_catalog(
            sess=sess,
            model=model,
            targets=cold_targets,
            n_items=n_items,
            user_hist=user_hist,
            item_user_hist=item_user_hist,
            user_seen=user_seen,
            item_counts=item_counts,
            item_batch_size=cfg.eval_item_batch_size,
        )
        print("Evaluating full-catalog hot targets ...", flush=True)
        full_hot, full_hot_item, n_fh, n_fh_item = evaluate_pam_full_catalog(
            sess=sess,
            model=model,
            targets=hot_targets,
            n_items=n_items,
            user_hist=user_hist,
            item_user_hist=item_user_hist,
            user_seen=user_seen,
            item_counts=item_counts,
            item_batch_size=cfg.eval_item_batch_size,
        )

    return {
        "last_train_loss": float(last_loss) if last_loss is not None else None,
        "training_seconds": float(time.time() - start),
        "checkpoint_prefix": str(latest_ckpt),
        "init_checkpoint": cfg.init_checkpoint,
        "start_epoch": int(cfg.start_epoch),
        "trained_epoch_count": int(max(0, cfg.epochs - cfg.start_epoch)),
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item,
        "full_hot_item_macro": full_hot_item,
        "count_full_cold": int(n_fc),
        "count_full_hot": int(n_fh),
        "count_full_cold_item_macro": int(n_fc_item),
        "count_full_hot_item_macro": int(n_fh_item),
    }


def print_report(result: Dict[str, object]) -> None:
    print("\n" + "=" * 76)
    print("         FINAL REPORT: full ranking only (PAM official-source adapted)")
    print("=" * 76)
    print(f"{'Metric':<10} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 76)
    cold = result.get("full_cold") or {}
    hot = result.get("full_hot") or {}
    for metric in METRICS:
        print(f"{metric:<10} | {cold.get(metric, 0.0):<12.4f} | {hot.get(metric, 0.0):<12.4f}")
    print("-" * 76)
    print(f"Full samples: Cold={result.get('count_full_cold', 0)}, Hot={result.get('count_full_hot', 0)}")
    print("=" * 76)


def main() -> None:
    args = parse_args()
    cfg = Config.from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    view_dir = cfg.output_dir / "pam_official_view"

    if args.mode in {"full", "export"}:
        meta, df = load_meta_and_stream(cfg.data_dir)
        n_users = int(meta["n_users"])
        n_items = int(meta["n_items"])
        train_df, val_df, test_df = load_static_split(
            df,
            cfg.split_dir,
            seed=cfg.static_seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
        )
        manifest = export_pam_dataset_view(
            data_dir=cfg.data_dir,
            relation_dir=cfg.relation_dir,
            output_dir=view_dir,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            n_users=n_users,
            n_items=n_items,
            seed=cfg.seed,
            cold_threshold=cfg.cold_threshold,
            batch_size=cfg.batch_size,
            neg_per_pos=cfg.neg_per_pos,
            max_train_pos=cfg.max_train_pos,
            max_eval_rows=cfg.max_eval_rows,
            max_cates_per_item=cfg.max_cates_per_item,
        )
        print(f"Wrote PAM official-format view to {view_dir}", flush=True)
        if args.mode == "export":
            return
    else:
        manifest = load_manifest(view_dir)

    print(
        f"Official PAM adapter: repo={cfg.pam_root} data={cfg.data_dir} "
        f"epochs={cfg.epochs} batch={cfg.batch_size} max_train_pos={cfg.max_train_pos} "
        f"max_eval_rows={cfg.max_eval_rows}",
        flush=True,
    )

    eval_result = train_and_evaluate(cfg, manifest)
    print_report(eval_result)
    git_info = pam_git_info(cfg.pam_root)
    result = {
        "model": "PAM-official-source-adapted",
        "model_display": "PAM",
        "source": (
            "Official Sycamoretail/PAM PAM-F EmbMLP and Engine imported unchanged; "
            "adapter converts the static split and recomputes full-catalog metrics."
        ),
        "official_source_dir": str(cfg.pam_root),
        "official_code": str(cfg.pam_root),
        **git_info,
        "paper": "Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning",
        "paper_venue": "KDD 2025",
        "protocol": "static_item_cold",
        "score_function": "official PAM pairwise sigmoid score over full catalog",
        "sample_cold": {},
        "sample_hot": {},
        **eval_result,
        "count_sample_cold": 0,
        "count_sample_hot": 0,
        "eval_n_neg": 0,
        "static_seed": cfg.static_seed,
        "seed": cfg.seed,
        "cold_threshold": cfg.cold_threshold,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "emb_dim": cfg.emb_dim,
        "hidden_dim": cfg.hidden_dim,
        "cate_dim": cfg.cate_dim,
        "neg_per_pos": cfg.neg_per_pos,
        "max_train_pos": cfg.max_train_pos,
        "max_eval_rows": cfg.max_eval_rows,
        "pam_view_dir": str(view_dir),
        "manifest": manifest,
        "note": (
            "PAM is an online item-cold method with an official vv-window cold protocol. "
            "The reported main-table fields are recomputed under this paper's static "
            "full-catalog item-macro protocol; smoke runs may be capped by max_train_pos/max_eval_rows."
        ),
    }
    result_path = _result_path(cfg, "pam_official_static_result.json")
    with result_path.open("w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
