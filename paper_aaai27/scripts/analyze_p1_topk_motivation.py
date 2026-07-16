from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RISK_COLUMNS = (
    "prerequisite_gap",
    "concept_continuity",
    "difficulty_gap",
    "structural_redundancy",
)
COLD_RISK_COLUMNS = tuple(f"cold_{column}" for column in RISK_COLUMNS)
LIST_METRIC_COLUMNS = RISK_COLUMNS + ("cold_proportion",) + COLD_RISK_COLUMNS
METRIC_DIRECTIONS = {
    "prerequisite_gap": "lower",
    "concept_continuity": "higher",
    "difficulty_gap": "lower",
    "structural_redundancy": "lower",
    "cold_proportion": "descriptive",
    "cold_prerequisite_gap": "lower",
    "cold_concept_continuity": "higher",
    "cold_difficulty_gap": "lower",
    "cold_structural_redundancy": "lower",
}
MOTIVATION_COMPARISONS = (
    ("ckg_rl", "pcgnn", "primary"),
    ("ckg_rl", "cgrc", "secondary"),
)


@dataclass(frozen=True)
class RiskArtifacts:
    prerequisite_matrix: np.ndarray
    concept_overlap: np.ndarray
    video_containment: np.ndarray
    same_family: np.ndarray
    structural_complexity: np.ndarray


class CourseMacroAccumulator:
    def __init__(self) -> None:
        self._groups = {}

    def update(self, row: dict) -> None:
        key = (
            str(row["model"]),
            int(row["seed"]),
            int(row["target_item_id"]),
            int(row["cutoff"]),
        )
        group = self._groups.setdefault(
            key,
            {
                "list_count": 0,
                "cold_list_count": 0,
                "sums": defaultdict(float),
                "counts": defaultdict(int),
            },
        )
        group["list_count"] += 1
        if np.isfinite(float(row["cold_prerequisite_gap"])):
            group["cold_list_count"] += 1
        for column in LIST_METRIC_COLUMNS:
            value = float(row[column])
            if np.isfinite(value):
                group["sums"][column] += value
                group["counts"][column] += 1

    def to_frame(self) -> pd.DataFrame:
        rows = []
        keys = ["model", "seed", "target_item_id", "cutoff"]
        for key in sorted(self._groups):
            group = self._groups[key]
            row = dict(zip(keys, key))
            row["list_count"] = int(group["list_count"])
            row["cold_list_count"] = int(group["cold_list_count"])
            for column in LIST_METRIC_COLUMNS:
                count = int(group["counts"][column])
                row[column] = (
                    float(group["sums"][column] / count)
                    if count
                    else float("nan")
                )
            rows.append(row)
        return pd.DataFrame(rows)


def robust_normalize_nonnegative(
    values: np.ndarray,
    upper_quantile: float = 0.95,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not 0.0 < float(upper_quantile) <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1]")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("values must be finite and nonnegative")
    if values.size == 0:
        return values.copy()

    scale = float(np.quantile(values, upper_quantile))
    if scale <= 0.0:
        scale = float(values.max(initial=0.0))
    if scale <= 0.0:
        return np.zeros_like(values)
    return np.clip(values / scale, 0.0, 1.0)


def build_structural_complexity(
    prerequisite_counts,
    concept_counts,
    upper_quantile: float = 0.95,
) -> np.ndarray:
    prerequisite_counts = np.asarray(prerequisite_counts, dtype=np.float64)
    concept_counts = np.asarray(concept_counts, dtype=np.float64)
    if prerequisite_counts.shape != concept_counts.shape:
        raise ValueError("prerequisite and concept counts must have matching shapes")
    prerequisite_scale = robust_normalize_nonnegative(
        prerequisite_counts,
        upper_quantile=upper_quantile,
    )
    concept_scale = robust_normalize_nonnegative(
        concept_counts,
        upper_quantile=upper_quantile,
    )
    return 0.5 * (prerequisite_scale + concept_scale)


def compute_item_risks(
    recommended_item_ids,
    history_item_ids,
    artifacts: RiskArtifacts,
    readiness_k: int = 5,
    precomputed_readiness: float | None = None,
) -> dict[str, np.ndarray]:
    recommended = np.asarray(recommended_item_ids, dtype=np.int64).reshape(-1)
    history = np.asarray(history_item_ids, dtype=np.int64).reshape(-1)
    n_items = int(np.asarray(artifacts.structural_complexity).size)
    if np.any((recommended < 0) | (recommended >= n_items)):
        raise ValueError("recommended item id outside the artifact catalog")
    if np.any((history < 0) | (history >= n_items)):
        raise ValueError("history item id outside the artifact catalog")

    if int(readiness_k) < 1:
        raise ValueError("readiness_k must be positive")
    prerequisite = np.asarray(artifacts.prerequisite_matrix, dtype=bool)
    concept = np.asarray(artifacts.concept_overlap, dtype=np.float64)
    video = np.asarray(artifacts.video_containment, dtype=np.float64)
    family = np.asarray(artifacts.same_family, dtype=bool)
    complexity = np.asarray(artifacts.structural_complexity, dtype=np.float64)

    prerequisite_rows = prerequisite[recommended]
    prerequisite_count = prerequisite_rows.sum(axis=1)
    if history.size:
        prerequisite_seen = prerequisite_rows[:, history].sum(axis=1)
        continuity = concept[np.ix_(recommended, history)].mean(axis=1)
        if precomputed_readiness is None:
            readiness_items = complexity[history]
            keep = min(int(readiness_k), int(readiness_items.size))
            readiness = float(np.partition(readiness_items, -keep)[-keep:].mean())
        else:
            readiness = float(precomputed_readiness)

        family_duplication = family[np.ix_(recommended, history)].max(axis=1)
        video_forward = video[np.ix_(recommended, history)]
        video_reverse = video[np.ix_(history, recommended)].T
        video_duplication = np.maximum(video_forward, video_reverse).max(axis=1)
        redundancy = np.maximum(family_duplication.astype(np.float64), video_duplication)
    else:
        prerequisite_seen = np.zeros(recommended.size, dtype=np.float64)
        continuity = np.zeros(recommended.size, dtype=np.float64)
        readiness = 0.0 if precomputed_readiness is None else float(precomputed_readiness)
        redundancy = np.zeros(recommended.size, dtype=np.float64)

    prerequisite_gap = np.zeros(recommended.size, dtype=np.float64)
    has_prerequisite = prerequisite_count > 0
    prerequisite_gap[has_prerequisite] = (
        1.0
        - prerequisite_seen[has_prerequisite]
        / prerequisite_count[has_prerequisite]
    )
    difficulty_gap = np.maximum(0.0, complexity[recommended] - readiness)
    return {
        "prerequisite_gap": np.clip(prerequisite_gap, 0.0, 1.0),
        "concept_continuity": np.clip(continuity, 0.0, 1.0),
        "difficulty_gap": np.clip(difficulty_gap, 0.0, 1.0),
        "structural_redundancy": np.clip(redundancy, 0.0, 1.0),
    }


def analyze_recommendation_list(
    recommended_item_ids,
    history_item_ids,
    train_popularity,
    artifacts: RiskArtifacts,
) -> dict[str, float | int]:
    recommended = np.asarray(recommended_item_ids, dtype=np.int64).reshape(-1)
    popularity = np.asarray(train_popularity, dtype=np.int64).reshape(-1)
    risks = compute_item_risks(recommended, history_item_ids, artifacts)
    cold_mask = popularity[recommended] == 0

    result: dict[str, float | int] = {
        "recommended_count": int(recommended.size),
        "cold_recommendation_count": int(cold_mask.sum()),
        "cold_proportion": float(cold_mask.mean()) if recommended.size else float("nan"),
    }
    for column in RISK_COLUMNS:
        values = risks[column]
        result[column] = float(values.mean()) if values.size else float("nan")
        result[f"cold_{column}"] = (
            float(values[cold_mask].mean()) if cold_mask.any() else float("nan")
        )
    return result


def validate_export_record(
    record: dict,
    *,
    expected_model: str,
    expected_seed: int,
    expected_sample_index: int,
    expected_pair: tuple[int, int],
    expected_target_popularity: int,
    history_item_ids,
    expected_top_k: int = 20,
) -> None:
    required = {
        "model",
        "seed",
        "sample_index",
        "user_id",
        "target_item_id",
        "target_popularity",
        "recommended_item_ids",
        "recommended_scores",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"export record missing fields: {missing}")
    if str(record["model"]) != str(expected_model):
        raise ValueError("export model metadata mismatch")
    if int(record["seed"]) != int(expected_seed):
        raise ValueError("export seed metadata mismatch")
    if int(record["sample_index"]) != int(expected_sample_index):
        raise ValueError("export sample index is not sequential")
    pair = (int(record["user_id"]), int(record["target_item_id"]))
    if pair != tuple(map(int, expected_pair)):
        raise ValueError("export user/target pair does not match the static split")
    if int(record["target_popularity"]) != int(expected_target_popularity):
        raise ValueError("export target popularity does not match training history")

    items = [int(item) for item in record["recommended_item_ids"]]
    scores = [float(score) for score in record["recommended_scores"]]
    if len(items) != int(expected_top_k) or len(scores) != int(expected_top_k):
        raise ValueError(f"export does not contain a complete Top-{expected_top_k} list")
    if len(set(items)) != len(items):
        raise ValueError("export contains duplicate recommended items")
    if any(item < 0 for item in items):
        raise ValueError("export contains a negative recommended item id")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("export contains a non-finite score")
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise ValueError("export scores are not in descending order")
    leaked = set(map(int, history_item_ids)).intersection(items)
    if leaked:
        raise ValueError(f"seen-item leakage in recommendations: {sorted(leaked)[:5]}")


def _summarize_precomputed_risks(
    recommended: np.ndarray,
    risks: dict[str, np.ndarray],
    train_popularity: np.ndarray,
) -> dict[str, float | int]:
    cold_mask = train_popularity[recommended] == 0
    result: dict[str, float | int] = {
        "recommended_count": int(recommended.size),
        "cold_recommendation_count": int(cold_mask.sum()),
        "cold_proportion": float(cold_mask.mean()) if recommended.size else float("nan"),
    }
    for column in RISK_COLUMNS:
        values = risks[column]
        result[column] = float(values.mean()) if values.size else float("nan")
        result[f"cold_{column}"] = (
            float(values[cold_mask].mean()) if cold_mask.any() else float("nan")
        )
    return result


def analyze_export_record(
    record: dict,
    history_item_ids,
    train_popularity,
    artifacts: RiskArtifacts,
    cutoffs=(10, 20),
    readiness_k: int = 5,
    precomputed_readiness: float | None = None,
) -> tuple[list[dict], list[dict]]:
    recommended = np.asarray(record["recommended_item_ids"], dtype=np.int64)
    scores = np.asarray(record["recommended_scores"], dtype=np.float64)
    popularity = np.asarray(train_popularity, dtype=np.int64)
    risks = compute_item_risks(
        recommended,
        history_item_ids,
        artifacts,
        readiness_k=readiness_k,
        precomputed_readiness=precomputed_readiness,
    )
    base = {
        "model": str(record["model"]),
        "seed": int(record["seed"]),
        "sample_index": int(record["sample_index"]),
        "user_id": int(record["user_id"]),
        "target_item_id": int(record["target_item_id"]),
    }

    recommendation_rows = []
    for index, item_id in enumerate(recommended):
        row = dict(base)
        row.update(
            {
                "rank": index + 1,
                "recommended_item_id": int(item_id),
                "score": float(scores[index]),
                "is_cold": int(popularity[item_id] == 0),
            }
        )
        for column in RISK_COLUMNS:
            row[column] = float(risks[column][index])
        recommendation_rows.append(row)

    list_rows = []
    for cutoff in cutoffs:
        cutoff = int(cutoff)
        if cutoff < 1 or cutoff > recommended.size:
            raise ValueError("cutoff must be covered by every exported list")
        row = dict(base)
        row["cutoff"] = cutoff
        summarized = _summarize_precomputed_risks(
            recommended[:cutoff],
            {column: risks[column][:cutoff] for column in RISK_COLUMNS},
            popularity,
        )
        row.update(summarized)
        list_rows.append(row)
    return recommendation_rows, list_rows


def analyze_seed_export_pair(
    *,
    seed: int,
    expected_pairs,
    histories,
    train_popularity,
    artifacts: RiskArtifacts,
    recommendation_sink,
    list_sink,
    course_accumulator: CourseMacroAccumulator,
    model_paths: Mapping[str, Path] | None = None,
    pair_keyed_models=(),
    ckg_rl_path=None,
    cgrc_path=None,
    expected_top_k: int = 20,
    cutoffs=(10, 20),
    metric_k: int = 10,
) -> dict[str, dict[str, float | int]]:
    popularity = np.asarray(train_popularity, dtype=np.int64)
    complexity = np.asarray(artifacts.structural_complexity, dtype=np.float64)
    readiness_by_user = {}
    for user_id, history in histories.items():
        history = np.asarray(history, dtype=np.int64)
        if history.size:
            keep = min(5, int(history.size))
            values = complexity[history]
            readiness_by_user[int(user_id)] = float(
                np.partition(values, -keep)[-keep:].mean()
            )
        else:
            readiness_by_user[int(user_id)] = 0.0
    if model_paths is None:
        if ckg_rl_path is None or cgrc_path is None:
            raise ValueError("provide model_paths or both legacy model paths")
        paths = {"ckg_rl": Path(ckg_rl_path), "cgrc": Path(cgrc_path)}
    else:
        if ckg_rl_path is not None or cgrc_path is not None:
            raise ValueError("model_paths cannot be mixed with legacy model paths")
        paths = {str(model): Path(path) for model, path in model_paths.items()}
        if not paths:
            raise ValueError("model_paths must contain at least one model")
    expected_pairs = [tuple(map(int, pair)) for pair in expected_pairs]
    pair_keyed_models = frozenset(map(str, pair_keyed_models))
    unknown_keyed_models = pair_keyed_models.difference(paths)
    if unknown_keyed_models:
        raise ValueError(f"pair-keyed models are not configured: {unknown_keyed_models}")
    if pair_keyed_models and len(set(expected_pairs)) != len(expected_pairs):
        raise ValueError("pair-keyed alignment requires unique split user/target pairs")

    keyed_lines = {}
    expected_pair_set = set(expected_pairs)
    for model in pair_keyed_models:
        indexed = {}
        with paths[model].open("r", encoding="utf-8") as handle:
            for native_index, line in enumerate(handle):
                record = json.loads(line)
                if record.get("model") != model or int(record.get("seed", -1)) != int(seed):
                    raise ValueError(f"{model} seed {seed} metadata mismatch while indexing")
                if int(record.get("sample_index", -1)) != native_index:
                    raise ValueError(f"{model} seed {seed} has non-sequential sample indices")
                pair = (int(record["user_id"]), int(record["target_item_id"]))
                if pair in indexed:
                    raise ValueError(f"{model} seed {seed} has duplicate user/target pairs")
                indexed[pair] = line
        actual_pair_set = set(indexed)
        if actual_pair_set != expected_pair_set:
            missing_count = len(expected_pair_set - actual_pair_set)
            extra_count = len(actual_pair_set - expected_pair_set)
            raise ValueError(
                f"{model} seed {seed} pair coverage mismatch: "
                f"missing={missing_count}, extra={extra_count}"
            )
        keyed_lines[model] = indexed

    handles = {
        model: path.open("r", encoding="utf-8")
        for model, path in paths.items()
        if model not in pair_keyed_models
    }
    hit_sums = defaultdict(float)
    ndcg_sums = defaultdict(float)
    item_hit_sums = {model: defaultdict(float) for model in paths}
    item_ndcg_sums = {model: defaultdict(float) for model in paths}
    item_counts = {model: defaultdict(int) for model in paths}
    record_counts = defaultdict(int)
    try:
        for sample_index, expected_pair in enumerate(expected_pairs):
            user_id, target_item_id = map(int, expected_pair)
            history = histories.get(user_id, ())
            target_popularity = int(popularity[target_item_id])
            records = {}
            for model in paths:
                if model in keyed_lines:
                    line = keyed_lines[model][(user_id, target_item_id)]
                else:
                    line = handles[model].readline()
                    if not line:
                        raise ValueError(
                            f"{model} seed {seed} ended before split row {sample_index}"
                        )
                record = json.loads(line)
                validate_export_record(
                    record,
                    expected_model=model,
                    expected_seed=seed,
                    expected_sample_index=(
                        int(record["sample_index"])
                        if model in keyed_lines
                        else sample_index
                    ),
                    expected_pair=(user_id, target_item_id),
                    expected_target_popularity=target_popularity,
                    history_item_ids=history,
                    expected_top_k=expected_top_k,
                )
                records[model] = record
                recommendation_rows, list_rows = analyze_export_record(
                    record,
                    history_item_ids=history,
                    train_popularity=popularity,
                    artifacts=artifacts,
                    cutoffs=cutoffs,
                    precomputed_readiness=readiness_by_user.get(user_id, 0.0),
                )
                for row in recommendation_rows:
                    recommendation_sink(row)
                for row in list_rows:
                    list_sink(row)
                    course_accumulator.update(row)

                top_items = [int(item) for item in record["recommended_item_ids"][:metric_k]]
                rank = top_items.index(target_item_id) + 1 if target_item_id in top_items else None
                hit = 1.0 if rank is not None else 0.0
                ndcg = 1.0 / math.log2(rank + 1.0) if rank is not None else 0.0
                hit_sums[model] += hit
                ndcg_sums[model] += ndcg
                item_hit_sums[model][target_item_id] += hit
                item_ndcg_sums[model][target_item_id] += ndcg
                item_counts[model][target_item_id] += 1
                record_counts[model] += 1

            shared_fields = (
                "seed",
                "user_id",
                "target_item_id",
                "target_popularity",
            )
            reference_model = next(iter(records))
            reference = records[reference_model]
            for model, record in records.items():
                if any(record[field] != reference[field] for field in shared_fields):
                    raise ValueError(
                        f"seed {seed} cross-model sample mismatch at row "
                        f"{sample_index}: {reference_model} != {model}"
                    )

        for model, handle in handles.items():
            if handle.readline():
                raise ValueError(f"{model} seed {seed} has records beyond split coverage")
    finally:
        for handle in handles.values():
            handle.close()

    audit = {}
    for model in paths:
        count = int(record_counts[model])
        target_items = sorted(item_counts[model])
        macro_hit = np.mean(
            [item_hit_sums[model][item] / item_counts[model][item] for item in target_items]
        )
        macro_ndcg = np.mean(
            [item_ndcg_sums[model][item] / item_counts[model][item] for item in target_items]
        )
        audit[model] = {
            "record_count": count,
            "target_course_count": len(target_items),
            f"R@{metric_k}": float(hit_sums[model] / count),
            f"N@{metric_k}": float(ndcg_sums[model] / count),
            f"course_macro_R@{metric_k}": float(macro_hit),
            f"course_macro_N@{metric_k}": float(macro_ndcg),
            "seen_item_leak_count": 0,
            "invalid_topk_count": 0,
        }
    return audit


def aggregate_course_macro(list_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "target_item_id", "cutoff"]
    missing = [column for column in keys + list(LIST_METRIC_COLUMNS) if column not in list_rows]
    if missing:
        raise ValueError(f"missing list-level columns: {missing}")

    rows = []
    for key, group in list_rows.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row["list_count"] = int(len(group))
        row["cold_list_count"] = int(group["cold_prerequisite_gap"].notna().sum())
        for column in LIST_METRIC_COLUMNS:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_course_macro(
    course_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "seed", "cutoff"]
    missing = [column for column in keys + list(LIST_METRIC_COLUMNS) if column not in course_rows]
    if missing:
        raise ValueError(f"missing course-macro columns: {missing}")

    seed_summary = (
        course_rows.groupby(keys, sort=True, as_index=False)[list(LIST_METRIC_COLUMNS)]
        .mean()
    )
    model_rows = []
    for (model, cutoff), group in seed_summary.groupby(["model", "cutoff"], sort=True):
        row = {
            "model": model,
            "cutoff": int(cutoff),
            "seed_count": int(group["seed"].nunique()),
        }
        for column in LIST_METRIC_COLUMNS:
            values = group[column].dropna().to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{column}_sd"] = (
                float(values.std(ddof=1)) if values.size > 1 else float("nan")
            )
        model_rows.append(row)
    return seed_summary, pd.DataFrame(model_rows)


def _paired_bootstrap_interval(
    differences: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if differences.size == 0:
        return float("nan"), float("nan")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    indices = rng.integers(
        0,
        differences.size,
        size=(int(n_bootstrap), differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _paired_permutation_pvalue(
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    if differences.size == 0:
        return float("nan")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    observed = abs(float(differences.mean()))
    exceedances = 0
    remaining = int(n_permutations)
    while remaining:
        batch_size = min(10_000, remaining)
        signs = rng.integers(
            0,
            2,
            size=(batch_size, differences.size),
            dtype=np.int8,
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
        remaining -= batch_size
    return float((exceedances + 1) / (int(n_permutations) + 1))


def _interpret_interval(direction: str, low: float, high: float) -> str:
    if direction == "descriptive" or not np.isfinite(low) or not np.isfinite(high):
        return "descriptive"
    if direction == "lower":
        if high < 0.0:
            return "supports"
        if low > 0.0:
            return "falsifies"
    elif direction == "higher":
        if low > 0.0:
            return "supports"
        if high < 0.0:
            return "falsifies"
    return "inconclusive"


def paired_course_statistics(
    course_rows: pd.DataFrame,
    comparisons=(("ckg_rl", "cgrc", "secondary"),),
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
) -> pd.DataFrame:
    required = ["model", "seed", "target_item_id", "cutoff"] + list(LIST_METRIC_COLUMNS)
    missing = [column for column in required if column not in course_rows]
    if missing:
        raise ValueError(f"missing paired-statistics columns: {missing}")

    rng = np.random.default_rng(random_seed)
    rows = []
    for treatment, baseline, role in comparisons:
        treatment = str(treatment)
        baseline = str(baseline)
        comparison = f"{treatment}_vs_{baseline}"
        for cutoff in sorted(course_rows["cutoff"].unique()):
            cutoff_rows = course_rows[course_rows["cutoff"].eq(cutoff)]
            for metric in LIST_METRIC_COLUMNS:
                pivot = cutoff_rows.pivot_table(
                    index=["seed", "target_item_id"],
                    columns="model",
                    values=metric,
                    aggfunc="first",
                )
                if treatment not in pivot or baseline not in pivot:
                    matched = np.empty(0, dtype=np.float64)
                else:
                    matched_rows = pivot[[treatment, baseline]].dropna()
                    matched = (
                        matched_rows[treatment].to_numpy(dtype=np.float64)
                        - matched_rows[baseline].to_numpy(dtype=np.float64)
                    )
                low, high = _paired_bootstrap_interval(matched, n_bootstrap, rng)
                p_value = _paired_permutation_pvalue(matched, n_permutations, rng)
                direction = METRIC_DIRECTIONS[metric]
                mean_difference = (
                    float(matched.mean()) if matched.size else float("nan")
                )
                rows.append(
                    {
                        "comparison": comparison,
                        "comparison_role": str(role),
                        "treatment": treatment,
                        "baseline": baseline,
                        "cutoff": int(cutoff),
                        "metric": metric,
                        "direction": direction,
                        "pair_count": int(matched.size),
                        "mean_difference": mean_difference,
                        "mean_difference_ckg_rl_minus_cgrc": (
                            mean_difference
                            if treatment == "ckg_rl" and baseline == "cgrc"
                            else float("nan")
                        ),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "permutation_p_value": p_value,
                        "interpretation": _interpret_interval(direction, low, high),
                    }
                )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_concept_counts(full_df: pd.DataFrame, relation_dir: Path, n_items: int) -> np.ndarray:
    from fast3_delta.course_artifacts import _read_relation_pairs

    item_to_course = (
        full_df[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .set_index("i_idx")["course_id"]
        .to_dict()
    )
    course_to_item = {str(course): int(item) for item, course in item_to_course.items()}
    concept_sets = [set() for _ in range(n_items)]
    for course_id, concept_id in _read_relation_pairs(
        str(relation_dir / "course-concept.json")
    ):
        item_id = course_to_item.get(str(course_id))
        if item_id is not None and concept_id:
            concept_sets[item_id].add(str(concept_id))
    return np.asarray([len(values) for values in concept_sets], dtype=np.float64)


def build_real_risk_artifacts(
    root: Path,
    upper_quantile: float = 0.95,
) -> tuple[RiskArtifacts, dict]:
    from fast3_delta.course_artifacts import build_course_artifacts

    data_path = root / "processed_data_hin_clean_pop5" / "stream_data.pkl"
    relation_dir = root / "MOOCCube" / "relations"
    full_df = pd.read_pickle(data_path)
    n_items = int(full_df["i_idx"].max()) + 1
    raw_artifacts, artifact_stats = build_course_artifacts(
        full_df,
        n_items=n_items,
        relation_dir=str(relation_dir),
        prereq_min_support=30,
        prereq_max_per_item=5,
        prereq_min_items=1,
        prereq_max_forward=20,
        concept_overlap_mode="plain",
        prereq_graph_source="concept",
        prereq_concept_score_thr=0.10,
        prereq_concept_min_hits=1,
        prereq_concept_file="prerequisite-dependency.json",
    )
    prerequisite = raw_artifacts["item_prereq_item_mat"].cpu().numpy() > 0.0
    prerequisite_counts = prerequisite.sum(axis=1).astype(np.float64)
    concept_counts = _item_concept_counts(full_df, relation_dir, n_items)
    complexity = build_structural_complexity(
        prerequisite_counts,
        concept_counts,
        upper_quantile=upper_quantile,
    )
    artifacts = RiskArtifacts(
        prerequisite_matrix=prerequisite,
        concept_overlap=raw_artifacts["item_concept_overlap"].cpu().numpy(),
        video_containment=raw_artifacts["item_video_contain"].cpu().numpy(),
        same_family=raw_artifacts["item_same_family"].cpu().numpy(),
        structural_complexity=complexity,
    )
    stats = dict(artifact_stats)
    stats.update(
        {
            "n_items": n_items,
            "complexity_upper_quantile": float(upper_quantile),
            "prerequisite_count_scale": float(
                np.quantile(prerequisite_counts, upper_quantile)
            ),
            "concept_count_scale": float(np.quantile(concept_counts, upper_quantile)),
            "items_with_nonzero_complexity": int(np.count_nonzero(complexity)),
            "data_path": str(data_path),
            "relation_dir": str(relation_dir),
        }
    )
    return artifacts, stats


def _seed_inputs(
    split_root: Path,
    seed: int,
    n_items: int,
) -> tuple[list[tuple[int, int]], dict[int, np.ndarray], np.ndarray]:
    seed_root = split_root / f"strict_item_cold_balanced_thr1_seed_{seed}"
    train = pd.read_pickle(seed_root / "static_train.pkl")
    test = pd.read_pickle(seed_root / "static_test.pkl")
    cold = test.loc[
        test["_split_source"].eq("strict_item_cold_test"),
        ["u_idx", "i_idx"],
    ]
    expected_pairs = [
        (int(user_id), int(item_id))
        for user_id, item_id in cold.itertuples(index=False, name=None)
    ]
    histories = train.groupby("u_idx")["i_idx"].apply(
        lambda values: np.asarray(sorted(set(map(int, values))), dtype=np.int64)
    ).to_dict()
    popularity = (
        train.groupby("i_idx")
        .size()
        .reindex(range(n_items), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    return expected_pairs, histories, popularity


def _model_export_paths(root: Path, seed: int) -> dict[str, Path]:
    root = Path(root)
    split_id = f"strict_item_cold_balanced_thr1_seed_{seed}"
    shared_root = root / "outputs" / "p1_motivation_topk"
    return {
        "ckg_rl": shared_root / "ckg_rl" / split_id / "top20_cold_test.jsonl",
        "pcgnn": (
            root
            / "paper_aaai27"
            / "baseline_sources"
            / "_pcgnn_strict"
            / f"mooccube_seed{seed}_full_formal_kg_warm"
            / "p1_top20_export"
            / "pcgnn_top20.jsonl"
        ),
        "cgrc": shared_root / "cgrc" / split_id / "top20_cold_test.jsonl",
    }


def _native_export_metrics(root: Path, seed: int, model: str) -> dict[str, float | int]:
    if model.startswith("ckg_rl"):
        path = (
            root
            / "outputs"
            / "p1_motivation_topk"
            / model
            / f"strict_item_cold_balanced_thr1_seed_{seed}"
            / "eval"
            / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        )
        row = pd.read_csv(path).iloc[0]
        return {
            "count_full_cold": int(row["full_cold_count"]),
            "R@10": float(row["full_cold_r10"]),
            "N@10": float(row["full_cold_n10"]),
            "course_macro_R@10": float(row["full_cold_item_macro_r10"]),
            "course_macro_N@10": float(row["full_cold_item_macro_n10"]),
        }

    if model == "pcgnn":
        path = (
            root
            / "paper_aaai27"
            / "baseline_sources"
            / "_pcgnn_strict"
            / f"mooccube_seed{seed}_full_formal_kg_warm"
            / "p1_top20_export"
            / "pcgnn_replay_result.json"
        )
        replay = json.loads(path.read_text(encoding="utf-8"))
        metrics = replay["metrics"]
        item_macro = metrics["full_cold_item_macro"]
        return {
            "count_full_cold": int(metrics["rows_full_cold"]),
            "count_full_cold_item_macro": int(
                metrics["count_full_cold_item_macro"]
            ),
            "course_macro_R@10": float(item_macro["R@10"]),
            "course_macro_N@10": float(item_macro["N@10"]),
        }

    path = (
        root
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
        / "p1_topk_export_cgrc"
        / "cgrc_paper_static_result.json"
    )
    row = json.loads(path.read_text(encoding="utf-8"))[0]
    return {
        "count_full_cold": int(row["count_full_cold"]),
        "R@10": float(row["full_cold"]["R@10"]),
        "N@10": float(row["full_cold"]["N@10"]),
        "course_macro_R@10": float(row["full_cold_item_macro"]["R@10"]),
        "course_macro_N@10": float(row["full_cold_item_macro"]["N@10"]),
    }


def validate_native_export_audit(
    audit: dict,
    native: dict,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    if int(audit["record_count"]) != int(native["count_full_cold"]):
        raise ValueError(
            "export/native coverage mismatch: "
            f"{audit['record_count']} != {native['count_full_cold']}"
        )
    metric_keys = (
        "R@10",
        "N@10",
        "course_macro_R@10",
        "course_macro_N@10",
    )
    deltas = {
        metric: abs(float(audit[metric]) - float(native[metric]))
        for metric in metric_keys
        if metric in native
    }
    if max(deltas.values(), default=0.0) > float(tolerance):
        raise ValueError(f"export/native metric mismatch: {deltas}")
    return deltas


def _validate_file_binding(binding: dict) -> Path:
    path = Path(binding["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(binding.get("size", path.stat().st_size)) != int(path.stat().st_size):
        raise ValueError(f"provenance size mismatch: {path}")
    if str(binding["sha256"]) != _sha256(path):
        raise ValueError(f"provenance hash mismatch: {path}")
    return path


def validate_export_provenance(
    root: Path,
    *,
    model: str,
    seed: int,
    export_path: Path,
    expected_count: int,
) -> dict:
    export_path = Path(export_path).resolve()
    if model.startswith("ckg_rl"):
        manifest_path = export_path.parent / "eval" / "p1_topk_export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != model or int(manifest.get("seed", -1)) != int(seed):
            raise ValueError("CKG-RL export manifest model/seed mismatch")
        if int(manifest.get("top_k", -1)) != 20:
            raise ValueError("CKG-RL export manifest Top-K mismatch")
        if Path(manifest["topk_output"]).resolve() != export_path:
            raise ValueError("CKG-RL export manifest path mismatch")
        if int(manifest.get("record_count", -1)) != int(expected_count):
            raise ValueError("CKG-RL export manifest coverage mismatch")
        for raw_path, expected_hash in manifest.get("checkpoint_hashes", {}).items():
            if _sha256(Path(raw_path)) != str(expected_hash):
                raise ValueError(f"CKG-RL checkpoint provenance mismatch: {raw_path}")
        source_manifest = Path(manifest["source_manifest"])
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_seed = int((source.get("split") or {}).get("seed", -1))
        if source_seed != int(seed):
            raise ValueError("CKG-RL source split seed mismatch")
        return manifest

    manifest_path = export_path.parent / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model == "pcgnn":
        if manifest.get("model") != model or int(manifest.get("seed", -1)) != int(seed):
            raise ValueError("PCGNN export manifest model/seed mismatch")
        if manifest.get("status") != "checkpoint_replay_valid":
            raise ValueError("PCGNN export manifest status mismatch")
        if manifest.get("native_report_test_reproduced") is not True:
            raise ValueError("PCGNN native report reproduction is not validated")
        if int(manifest.get("top_k", -1)) != 20:
            raise ValueError("PCGNN export manifest Top-K mismatch")
        if int(manifest.get("record_count", -1)) != int(expected_count):
            raise ValueError("PCGNN export manifest coverage mismatch")
        if _validate_file_binding(manifest["topk_output"]) != export_path:
            raise ValueError("PCGNN export manifest path mismatch")
        _validate_file_binding(manifest["replay_result"])
        _validate_file_binding(manifest["report"])
        _validate_file_binding(manifest["config"])
        for binding in manifest.get("split_files", []):
            _validate_file_binding(binding)
        for binding in manifest.get("script_files", []):
            _validate_file_binding(binding)
        checkpoint = manifest["checkpoint"]
        checkpoint_path = Path(checkpoint["path"]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if int(checkpoint.get("size", checkpoint_path.stat().st_size)) != int(
            checkpoint_path.stat().st_size
        ):
            raise ValueError(
                f"PCGNN checkpoint provenance mismatch: {checkpoint_path}"
            )
        current = _sha256(checkpoint_path)
        if not (
            current
            == str(checkpoint["sha256_before"])
            == str(checkpoint["sha256_after"])
        ):
            raise ValueError(
                f"PCGNN checkpoint provenance mismatch: {checkpoint_path}"
            )
        return manifest

    if manifest.get("model") != model or int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("CGRC export manifest model/seed mismatch")
    if int(manifest.get("top_k", -1)) != 20:
        raise ValueError("CGRC export manifest Top-K mismatch")
    if int(manifest.get("record_count", -1)) != int(expected_count):
        raise ValueError("CGRC export manifest coverage mismatch")
    if _validate_file_binding(manifest["topk_output"]) != export_path:
        raise ValueError("CGRC export manifest path mismatch")
    _validate_file_binding(manifest["native_result"])
    for binding in manifest.get("split_files", []):
        _validate_file_binding(binding)
    for binding in manifest.get("script_files", []):
        _validate_file_binding(binding)
    for binding in manifest.get("checkpoints", []):
        path = Path(binding["path"]).resolve()
        current = _sha256(path)
        if not (
            current == str(binding["sha256_before"])
            == str(binding["sha256_after"])
        ):
            raise ValueError(f"CGRC checkpoint provenance mismatch: {path}")
    return manifest


def _cgrc_replay_drift(root: Path, seed: int) -> dict[str, float]:
    base = (
        root
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
    )
    original = json.loads(
        (base / "p1_motivation_cgrc_main_table_reproduction" / "cgrc_paper_static_result.json")
        .read_text(encoding="utf-8")
    )[0]
    replay = json.loads(
        (base / "p1_topk_export_cgrc" / "cgrc_paper_static_result.json")
        .read_text(encoding="utf-8")
    )[0]
    original_values = {
        "R@10": float(original["full_cold"]["R@10"]),
        "N@10": float(original["full_cold"]["N@10"]),
        "course_macro_R@10": float(original["full_cold_item_macro"]["R@10"]),
        "course_macro_N@10": float(original["full_cold_item_macro"]["N@10"]),
    }
    replay_values = {
        "R@10": float(replay["full_cold"]["R@10"]),
        "N@10": float(replay["full_cold"]["N@10"]),
        "course_macro_R@10": float(replay["full_cold_item_macro"]["R@10"]),
        "course_macro_N@10": float(replay["full_cold_item_macro"]["N@10"]),
    }
    return {
        metric: replay_values[metric] - original_values[metric]
        for metric in original_values
    }


def _open_gzip_writer(path: Path, fieldnames: list[str]):
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.unlink(missing_ok=True)
    handle = gzip.open(
        tmp_path,
        "wt",
        encoding="utf-8",
        newline="",
        compresslevel=1,
    )
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return tmp_path, handle, writer


def run_analysis(
    root: Path,
    output_dir: Path,
    seeds=(2025, 2026, 2027),
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
) -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"

    print("[P1] Building model-neutral course artifacts", flush=True)
    artifacts, artifact_stats = build_real_risk_artifacts(root)
    n_items = int(artifacts.structural_complexity.size)
    course_accumulator = CourseMacroAccumulator()
    recommendation_path = output_dir / "recommendation_level.csv.gz"
    list_path = output_dir / "list_level.csv.gz"
    recommendation_fields = [
        "model",
        "seed",
        "sample_index",
        "user_id",
        "target_item_id",
        "rank",
        "recommended_item_id",
        "score",
        "is_cold",
        *RISK_COLUMNS,
    ]
    list_fields = [
        "model",
        "seed",
        "sample_index",
        "user_id",
        "target_item_id",
        "cutoff",
        "recommended_count",
        "cold_recommendation_count",
        *LIST_METRIC_COLUMNS,
    ]
    rec_tmp, rec_handle, rec_writer = _open_gzip_writer(
        recommendation_path,
        recommendation_fields,
    )
    list_tmp, list_handle, list_writer = _open_gzip_writer(list_path, list_fields)
    audit_rows = []
    export_paths = []
    try:
        for seed in seeds:
            expected_pairs, histories, popularity = _seed_inputs(split_root, seed, n_items)
            paths = _model_export_paths(root, seed)
            export_paths.extend(paths.values())
            for model, path in paths.items():
                validate_export_provenance(
                    root,
                    model=model,
                    seed=seed,
                    export_path=path,
                    expected_count=len(expected_pairs),
                )
            seed_audit = analyze_seed_export_pair(
                seed=seed,
                model_paths=paths,
                pair_keyed_models=("pcgnn",),
                expected_pairs=expected_pairs,
                histories=histories,
                train_popularity=popularity,
                artifacts=artifacts,
                recommendation_sink=rec_writer.writerow,
                list_sink=list_writer.writerow,
                course_accumulator=course_accumulator,
                expected_top_k=20,
                cutoffs=(10, 20),
                metric_k=10,
            )
            for model, audit in seed_audit.items():
                native = _native_export_metrics(root, seed, model)
                deltas = validate_native_export_audit(audit, native)
                row = {"model": model, "seed": seed, **audit}
                row.update({f"native_delta_{key}": value for key, value in deltas.items()})
                if model == "cgrc":
                    row.update(
                        {
                            f"replay_minus_original_{key}": value
                            for key, value in _cgrc_replay_drift(root, seed).items()
                        }
                    )
                audit_rows.append(row)
            print(
                f"[P1] seed={seed} validated and analyzed "
                f"records={len(expected_pairs)}x{len(paths)}",
                flush=True,
            )
        rec_handle.close()
        list_handle.close()
        os.replace(rec_tmp, recommendation_path)
        os.replace(list_tmp, list_path)
    except Exception:
        rec_handle.close()
        list_handle.close()
        rec_tmp.unlink(missing_ok=True)
        list_tmp.unlink(missing_ok=True)
        raise

    course_rows = course_accumulator.to_frame()
    seed_summary, model_summary = summarize_course_macro(course_rows)
    paired = paired_course_statistics(
        course_rows,
        comparisons=MOTIVATION_COMPARISONS,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    audit = pd.DataFrame(audit_rows)
    if not np.isfinite(course_rows[list(RISK_COLUMNS) + ["cold_proportion"]].to_numpy()).all():
        raise ValueError("primary course-macro output contains non-finite values")

    course_path = output_dir / "course_macro.csv"
    seed_summary_path = output_dir / "seed_summary.csv"
    model_summary_path = output_dir / "model_summary.csv"
    paired_path = output_dir / "paired_statistics.csv"
    audit_path = output_dir / "seed_export_audit.csv"
    course_rows.to_csv(course_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    audit.to_csv(audit_path, index=False)

    figure_dir = root / "paper_aaai27" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_summary.to_csv(
        figure_dir / "mooccube_p1_topk_motivation_summary.csv",
        index=False,
    )
    paired.to_csv(
        figure_dir / "mooccube_p1_topk_motivation_paired.csv",
        index=False,
    )
    course_rows.to_csv(
        figure_dir / "mooccube_p1_topk_motivation_course_macro.csv",
        index=False,
    )

    checkpoint_hashes = {}
    for seed in seeds:
        cgrc_checkpoint = (
            root
            / "checkpoints"
            / "content_delta_pop5"
            / "p1_motivation_cgrc_main_table_reproduction"
            / f"strict_item_cold_balanced_thr1_seed_{seed}"
            / "best.pt"
        )
        checkpoint_hashes[str(cgrc_checkpoint)] = _sha256(cgrc_checkpoint)
    manifest = {
        "seeds": list(map(int, seeds)),
        "cutoffs": [10, 20],
        "top_k": 20,
        "random_seed": int(random_seed),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "comparisons": [
            {
                "treatment": treatment,
                "baseline": baseline,
                "role": role,
            }
            for treatment, baseline, role in MOTIVATION_COMPARISONS
        ],
        "artifact_stats": artifact_stats,
        "export_sha256": {str(path): _sha256(path) for path in export_paths},
        "cgrc_best_checkpoint_sha256": checkpoint_hashes,
        "row_counts": {
            "recommendation": int(audit["record_count"].sum() * 20),
            "list": int(audit["record_count"].sum() * 2),
            "course_macro": int(len(course_rows)),
            "seed_summary": int(len(seed_summary)),
            "paired_statistics": int(len(paired)),
        },
        "outputs": {
            "recommendation_level": str(recommendation_path),
            "list_level": str(list_path),
            "course_macro": str(course_path),
            "seed_summary": str(seed_summary_path),
            "model_summary": str(model_summary_path),
            "paired_statistics": str(paired_path),
            "seed_export_audit": str(audit_path),
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[P1] wrote analysis to {output_dir}", flush=True)
    return manifest


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "paper_aaai27" / "figures" / "p1_topk_motivation_analysis",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--analysis-seed", type=int, default=2027)
    args = parser.parse_args()
    run_analysis(
        root=args.root,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        random_seed=args.analysis_seed,
    )


if __name__ == "__main__":
    main()
