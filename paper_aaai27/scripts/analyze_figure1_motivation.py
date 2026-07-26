"""Build the three evidence tables used by Figure 1.

The analysis is deliberately read-only.  It reuses the frozen validation
replay for the baseline contrast and derives two model-independent diagnostics
from the strict validation split: course-signal availability and variation in
learner-conditioned fit for the same cold target course.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper_aaai27" / "figures"
DEFAULT_COURSE_ROWS = FIGURE_DIR / "validation_motivation_analysis" / "course_macro.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "validation_motivation_analysis"
SPLIT_ROOT = ROOT / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
SEEDS = (2025, 2026, 2027)
N_ITEMS = 698


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            elif "," in line:
                left, right = line.split(",", 1)
            else:
                continue
            rows.append((left.strip(), right.strip()))
    return rows


def _read_json_lines(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            return payload if isinstance(payload, list) else []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validation_cold_units(course_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"analysis_split", "model", "seed", "target_item_id"}
    missing = sorted(required.difference(course_rows.columns))
    if missing:
        raise ValueError(f"course rows missing columns: {missing}")
    if set(course_rows["analysis_split"].astype(str)) != {"validation"}:
        raise ValueError("Figure 1 motivation inputs must be validation-only")
    units = course_rows[["seed", "target_item_id"]].drop_duplicates().copy()
    units["seed"] = units["seed"].astype(int)
    units["target_item_id"] = units["target_item_id"].astype(int)
    expected = len(SEEDS) * 34
    if len(units) != expected:
        raise ValueError(f"expected {expected} validation seed-course units, found {len(units)}")
    if units.duplicated(["seed", "target_item_id"]).any():
        raise ValueError("validation seed-course units are duplicated")
    return units.sort_values(["seed", "target_item_id"]).reset_index(drop=True)


def build_baseline_seed_summary(course_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize each frozen baseline within seed before pooling seeds."""
    required = {"model", "seed", "target_item_id", "ndcg_at_10", "cold_proportion"}
    if missing := sorted(required.difference(course_rows.columns)):
        raise ValueError(f"baseline rows missing columns: {missing}")
    selected = course_rows.loc[course_rows["model"].isin(["pcgnn", "cgrc"])].copy()
    grouped = (
        selected.groupby(["model", "seed"], as_index=False)
        .agg(
            ndcg_at_10=("ndcg_at_10", "mean"),
            cold_proportion=("cold_proportion", "mean"),
            target_course_count=("target_item_id", "nunique"),
        )
        .sort_values(["model", "seed"])
    )
    if set(grouped["model"]) != {"pcgnn", "cgrc"} or len(grouped) != 6:
        raise ValueError("baseline summary must contain both models for all three seeds")
    grouped["analysis_split"] = "validation"
    grouped["protocol"] = "strict course-cold full-catalog ranking"
    return grouped[
        [
            "analysis_split",
            "protocol",
            "model",
            "seed",
            "target_course_count",
            "ndcg_at_10",
            "cold_proportion",
        ]
    ]


def _course_index(full: pd.DataFrame) -> dict[int, str]:
    mapping = full[["i_idx", "course_id"]].drop_duplicates("i_idx")
    return {int(row.i_idx): str(row.course_id) for row in mapping.itertuples()}


def _concept_sets(full: pd.DataFrame, relation_dir: Path) -> list[set[str]]:
    course_to_item = {course: item for item, course in _course_index(full).items()}
    sets = [set() for _ in range(N_ITEMS)]
    for course_id, concept_id in _read_pairs(relation_dir / "course-concept.json"):
        item_id = course_to_item.get(course_id)
        if item_id is not None and concept_id:
            sets[item_id].add(concept_id)
    return sets


def _prerequisite_counts(
    concept_sets: list[set[str]],
    relation_dir: Path,
    *,
    score_threshold: float = 0.10,
    min_hits: int = 1,
    max_per_item: int = 5,
) -> np.ndarray:
    """Match the concept-derived prerequisite construction used by CKG-RL."""
    incoming_concepts: dict[str, set[str]] = defaultdict(set)
    for prerequisite, target in _read_pairs(relation_dir / "prerequisite-dependency.json"):
        if prerequisite and target and prerequisite != target:
            incoming_concepts[target].add(prerequisite)

    required_sets = []
    for concepts in concept_sets:
        required: set[str] = set()
        for concept in concepts:
            required.update(incoming_concepts.get(concept, ()))
        required.difference_update(concepts)
        required_sets.append(required)

    counts = np.zeros(len(concept_sets), dtype=np.float64)
    for target, required in enumerate(required_sets):
        if not required:
            continue
        candidates = []
        denominator = float(len(required))
        for source, concepts in enumerate(concept_sets):
            if source == target or not concepts:
                continue
            hits = len(concepts.intersection(required))
            score = hits / denominator
            if hits >= int(min_hits) and score >= float(score_threshold):
                candidates.append((score, hits, source))
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        counts[target] = min(len(candidates), int(max_per_item))
    return counts


def _robust_normalize(values: np.ndarray, upper_quantile: float = 0.95) -> np.ndarray:
    scale = float(np.quantile(values, upper_quantile))
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / scale, 0.0, 1.0)


def _availability_summary(
    units: pd.DataFrame,
    full: pd.DataFrame,
    concept_sets: list[set[str]],
    prerequisite_counts: np.ndarray,
    entity_by_course: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_to_course = _course_index(full)
    rows = []
    for row in units.itertuples(index=False):
        item_id = int(row.target_item_id)
        course_id = item_to_course[item_id]
        entity = entity_by_course.get(course_id, {})
        name = _clean_text(entity.get("name"))
        about = _clean_text(entity.get("about"))
        rows.append(
            {
                "analysis_split": "validation",
                "seed": int(row.seed),
                "target_item_id": item_id,
                "content_text": float(bool(name and about)),
                "concepts": float(bool(concept_sets[item_id])),
                "prerequisites": float(prerequisite_counts[item_id] > 0),
                "difficulty_proxy": float(
                    prerequisite_counts[item_id] > 0 or bool(concept_sets[item_id])
                ),
                "video_metadata": float(bool(entity.get("video_order"))),
            }
        )
    detail = pd.DataFrame(rows)
    definitions = {
        "content_text": "course name and non-empty about text",
        "concepts": "at least one course-concept relation",
        "prerequisites": "at least one retained concept-derived prerequisite edge",
        "difficulty_proxy": "non-zero robust structural complexity from concept/prerequisite counts",
        "video_metadata": "non-empty video_order course metadata",
    }
    labels = {
        "content_text": "Content text",
        "concepts": "Concepts",
        "prerequisites": "Prerequisites",
        "difficulty_proxy": "Difficulty proxy",
        "video_metadata": "Video metadata",
    }
    summary_rows = []
    for signal in definitions:
        values = detail[signal].to_numpy(dtype=float)
        summary_rows.append(
            {
                "signal": signal,
                "label": labels[signal],
                "available_units": int(values.sum()),
                "total_units": int(values.size),
                "fraction": float(values.mean()),
                "definition": definitions[signal],
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _concept_overlap_matrix(concept_sets: list[set[str]]) -> np.ndarray:
    matrix = np.zeros((len(concept_sets), len(concept_sets)), dtype=np.float32)
    for target, target_concepts in enumerate(concept_sets):
        if not target_concepts:
            continue
        denominator = float(len(target_concepts))
        for history_item, history_concepts in enumerate(concept_sets):
            if history_concepts:
                matrix[target, history_item] = len(
                    target_concepts.intersection(history_concepts)
                ) / denominator
    return matrix


def build_learner_heterogeneity(
    full: pd.DataFrame,
    concept_sets: list[set[str]],
    prerequisite_counts: np.ndarray,
    prerequisite_sources: list[set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure within-target-course variation across validation learner histories."""
    overlap = _concept_overlap_matrix(concept_sets)
    complexity = 0.5 * (
        _robust_normalize(prerequisite_counts)
        + _robust_normalize(np.asarray([len(values) for values in concept_sets], dtype=float))
    )
    rows = []
    pair_rows = []
    for seed in SEEDS:
        seed_root = SPLIT_ROOT / f"strict_item_cold_balanced_thr1_seed_{seed}"
        train = pd.read_pickle(seed_root / "static_train.pkl")
        validation = pd.read_pickle(seed_root / "static_val.pkl")
        histories = train.groupby("u_idx")["i_idx"].apply(
            lambda values: np.asarray(sorted(set(map(int, values))), dtype=np.int64)
        ).to_dict()
        readiness = {}
        for user_id, history in histories.items():
            if history.size:
                keep = min(5, int(history.size))
                readiness[int(user_id)] = float(
                    np.partition(complexity[history], -keep)[-keep:].mean()
                )
            else:
                readiness[int(user_id)] = 0.0

        cold = validation.loc[
            validation["_split_source"].eq("strict_item_cold_val"),
            ["u_idx", "i_idx"],
        ]
        by_course: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for user_id, target_item in cold.itertuples(index=False):
            history = histories.get(int(user_id), np.empty(0, dtype=np.int64))
            target_item = int(target_item)
            if history.size == 0:
                continue
            prerequisite_count = float(prerequisite_counts[target_item])
            if prerequisite_count > 0.0:
                prerequisite_gap = 1.0 - len(
                    prerequisite_sources[target_item].intersection(map(int, history))
                ) / prerequisite_count
            else:
                prerequisite_gap = 0.0
            concept_continuity = float(overlap[target_item, history].mean())
            difficulty_gap = max(
                0.0, complexity[target_item] - readiness.get(int(user_id), 0.0)
            )
            values = (
                float(np.clip(prerequisite_gap, 0.0, 1.0)),
                float(np.clip(concept_continuity, 0.0, 1.0)),
                float(np.clip(difficulty_gap, 0.0, 1.0)),
            )
            by_course[target_item].append(values)
            pair_rows.append(
                {
                    "analysis_split": "validation",
                    "seed": int(seed),
                    "target_item_id": target_item,
                    "user_id": int(user_id),
                    "prerequisite_gap": values[0],
                    "concept_continuity": values[1],
                    "difficulty_gap": values[2],
                }
            )

        for target_item, values in sorted(by_course.items()):
            array = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "analysis_split": "validation",
                    "seed": int(seed),
                    "target_item_id": int(target_item),
                    "pair_count": int(len(array)),
                    "prerequisite_gap_sd": float(np.std(array[:, 0], ddof=1))
                    if len(array) > 1
                    else 0.0,
                    "concept_continuity_sd": float(np.std(array[:, 1], ddof=1))
                    if len(array) > 1
                    else 0.0,
                    "difficulty_gap_sd": float(np.std(array[:, 2], ddof=1))
                    if len(array) > 1
                    else 0.0,
                }
            )
    detail = pd.DataFrame(rows).sort_values(["seed", "target_item_id"])
    metric_labels = {
        "prerequisite_gap_sd": "Prerequisite gap",
        "concept_continuity_sd": "Concept continuity",
        "difficulty_gap_sd": "Difficulty gap",
    }
    summary_rows = []
    for metric, label in metric_labels.items():
        values = detail[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "label": label,
                "unit_count": int(values.size),
                "median_sd": float(np.median(values)),
                "mean_sd": float(values.mean()),
                "q25_sd": float(np.quantile(values, 0.25)),
                "q75_sd": float(np.quantile(values, 0.75)),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def build_prerequisite_sources(
    concept_sets: list[set[str]],
    relation_dir: Path,
) -> tuple[np.ndarray, list[set[int]]]:
    """Return retained prerequisite counts and source-item sets."""
    incoming_concepts: dict[str, set[str]] = defaultdict(set)
    for prerequisite, target in _read_pairs(relation_dir / "prerequisite-dependency.json"):
        if prerequisite and target and prerequisite != target:
            incoming_concepts[target].add(prerequisite)
    sources: list[set[int]] = [set() for _ in concept_sets]
    for target, target_concepts in enumerate(concept_sets):
        required = set()
        for concept in target_concepts:
            required.update(incoming_concepts.get(concept, ()))
        required.difference_update(target_concepts)
        candidates = []
        for source, source_concepts in enumerate(concept_sets):
            if source == target or not source_concepts or not required:
                continue
            hits = len(source_concepts.intersection(required))
            score = hits / float(len(required))
            if hits >= 1 and score >= 0.10:
                candidates.append((score, hits, source))
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        sources[target] = {int(value[2]) for value in candidates[:5]}
    counts = np.asarray([len(values) for values in sources], dtype=np.float64)
    return counts, sources


def run_analysis(
    *,
    course_rows_path: Path = DEFAULT_COURSE_ROWS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    course_rows = pd.read_csv(course_rows_path)
    units = validation_cold_units(course_rows)
    baseline = build_baseline_seed_summary(course_rows)

    data_path = ROOT / "processed_data_hin_clean_pop5" / "stream_data.pkl"
    relation_dir = ROOT / "MOOCCube" / "relations"
    entity_path = ROOT / "MOOCCube" / "entities" / "course.json"
    full = pd.read_pickle(data_path)
    concept_sets = _concept_sets(full, relation_dir)
    prerequisite_counts, prerequisite_sources = build_prerequisite_sources(
        concept_sets, relation_dir
    )
    entity_by_course = {
        str(row.get("id")): row for row in _read_json_lines(entity_path) if row.get("id")
    }
    availability, availability_summary = _availability_summary(
        units,
        full,
        concept_sets,
        prerequisite_counts,
        entity_by_course,
    )
    heterogeneity, heterogeneity_summary = build_learner_heterogeneity(
        full,
        concept_sets,
        prerequisite_counts,
        prerequisite_sources,
    )

    paths = {
        "baseline_seed": output_dir / "baseline_seed.csv",
        "signal_availability": output_dir / "signal_availability.csv",
        "signal_availability_summary": output_dir / "signal_availability_summary.csv",
        "learner_heterogeneity": output_dir / "learner_heterogeneity.csv",
        "learner_heterogeneity_summary": output_dir / "learner_heterogeneity_summary.csv",
        "manifest": output_dir / "figure1_motivation_manifest.json",
    }
    baseline.to_csv(paths["baseline_seed"], index=False)
    availability.to_csv(paths["signal_availability"], index=False)
    availability_summary.to_csv(paths["signal_availability_summary"], index=False)
    heterogeneity.to_csv(paths["learner_heterogeneity"], index=False)
    heterogeneity_summary.to_csv(paths["learner_heterogeneity_summary"], index=False)

    manifest = {
        "analysis": "figure1_motivation_evidence_pack",
        "dataset": "MOOCCube",
        "analysis_split": "validation",
        "protocol": "strict course-cold full-catalog ranking",
        "seeds": list(SEEDS),
        "validation_cold_courses_per_seed": 34,
        "seed_course_units": int(len(units)),
        "baseline_models": ["PCGNN", "CGRC"],
        "baseline_unit": "seed-level mean over validation cold target courses",
        "baseline_bootstrap": "not used in the figure; seed means and range are shown",
        "availability_unit": "validation cold seed-course unit",
        "heterogeneity_unit": "validation cold target course; SD across its learner histories",
        "checkpoint_selection_caveat": "baseline checkpoints were selected on validation and are descriptive",
        "inputs": {
            "course_rows": {"path": str(course_rows_path), "sha256": _sha256(course_rows_path)},
            "stream_data": {"path": str(data_path), "sha256": _sha256(data_path)},
            "course_concept": {"path": str(relation_dir / "course-concept.json"), "sha256": _sha256(relation_dir / "course-concept.json")},
            "prerequisite_dependency": {"path": str(relation_dir / "prerequisite-dependency.json"), "sha256": _sha256(relation_dir / "prerequisite-dependency.json")},
            "course_entities": {"path": str(entity_path), "sha256": _sha256(entity_path)},
        },
        "outputs": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 1 motivation evidence tables")
    parser.add_argument("--course-rows", type=Path, default=DEFAULT_COURSE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name, path in run_analysis(
        course_rows_path=args.course_rows,
        output_dir=args.output_dir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
