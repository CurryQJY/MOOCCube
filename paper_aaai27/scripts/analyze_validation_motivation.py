from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_topk_motivation import (  # noqa: E402
    CourseMacroAccumulator,
    RiskArtifacts,
    analyze_export_record,
    build_real_risk_artifacts,
    validate_export_record,
)


MODELS = ("pcgnn", "cgrc")
STRUCTURAL_METRICS = (
    "cold_prerequisite_gap",
    "cold_concept_continuity",
    "cold_difficulty_gap",
    "cold_structural_redundancy",
)
SUMMARY_METRICS = (
    ("ndcg_at_10", "exposure", "higher"),
    ("low_ndcg_at_10", "exposure", "lower"),
    ("cold_proportion", "exposure", "descriptive"),
    ("effective_coverage", "exposure", "descriptive"),
    ("missingness", "exposure", "descriptive"),
    ("cold_prerequisite_gap", "structure", "lower"),
    ("cold_concept_continuity", "structure", "higher"),
    ("cold_difficulty_gap", "structure", "lower"),
    ("cold_structural_redundancy", "structure", "lower"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_seed_inputs(
    split_root: Path,
    seed: int,
    n_items: int,
) -> tuple[list[tuple[int, int]], dict[int, np.ndarray], np.ndarray]:
    seed_root = Path(split_root) / f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    train = pd.read_pickle(seed_root / "static_train.pkl")
    validation = pd.read_pickle(seed_root / "static_val.pkl")
    test = pd.read_pickle(seed_root / "static_test.pkl")

    required = {"u_idx", "i_idx", "_split_source"}
    if missing := required.difference(validation.columns):
        raise ValueError(f"validation split is missing columns: {sorted(missing)}")
    if missing := required.difference(test.columns):
        raise ValueError(f"test split is missing columns: {sorted(missing)}")

    validation_cold = validation.loc[
        validation["_split_source"].eq("strict_item_cold_val"),
        ["u_idx", "i_idx"],
    ]
    expected_pairs = [
        (int(user_id), int(item_id))
        for user_id, item_id in validation_cold.itertuples(index=False, name=None)
    ]
    if not expected_pairs:
        raise ValueError(f"seed {seed} has no strict validation-cold rows")

    validation_targets = {item_id for _, item_id in expected_pairs}
    test_targets = set(
        map(
            int,
            test.loc[
                test["_split_source"].eq("strict_item_cold_test"),
                "i_idx",
            ].unique(),
        )
    )
    overlap = validation_targets.intersection(test_targets)
    if overlap:
        raise ValueError(f"validation and test cold targets overlap: {sorted(overlap)[:5]}")

    histories = train.groupby("u_idx")["i_idx"].apply(
        lambda values: np.asarray(sorted(set(map(int, values))), dtype=np.int64)
    ).to_dict()
    popularity = (
        train.groupby("i_idx")
        .size()
        .reindex(range(int(n_items)), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    noncold_targets = sorted(item for item in validation_targets if popularity[item] != 0)
    if noncold_targets:
        raise ValueError(
            f"validation targets are not course-cold in training: {noncold_targets[:5]}"
        )
    return expected_pairs, histories, popularity


def validate_validation_export_record(record: dict, **kwargs) -> None:
    if record.get("analysis_split") != "validation":
        raise ValueError("Figure 1 accepts validation-only export records")
    validate_export_record(record, **kwargs)


def _readiness_by_user(
    histories: Mapping[int, np.ndarray],
    artifacts: RiskArtifacts,
) -> dict[int, float]:
    complexity = np.asarray(artifacts.structural_complexity, dtype=np.float64)
    readiness = {}
    for user_id, raw_history in histories.items():
        history = np.asarray(raw_history, dtype=np.int64)
        if history.size:
            keep = min(5, int(history.size))
            values = complexity[history]
            readiness[int(user_id)] = float(np.partition(values, -keep)[-keep:].mean())
        else:
            readiness[int(user_id)] = 0.0
    return readiness


def analyze_validation_seed(
    *,
    seed: int,
    expected_pairs,
    histories: Mapping[int, np.ndarray],
    train_popularity,
    artifacts: RiskArtifacts,
    model_paths: Mapping[str, Path],
    pair_keyed_models=("pcgnn",),
    expected_top_k: int = 20,
    metric_k: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    paths = {str(model): Path(path) for model, path in model_paths.items()}
    if not paths or not set(paths).issubset(MODELS):
        raise ValueError(f"validation motivation models must be a subset of {MODELS}")
    if int(metric_k) < 1 or int(metric_k) > int(expected_top_k):
        raise ValueError("metric_k must be covered by the exported Top-K lists")

    pairs = [tuple(map(int, pair)) for pair in expected_pairs]
    pair_keyed_models = frozenset(map(str, pair_keyed_models)).intersection(paths)
    if len(set(pairs)) != len(pairs):
        raise ValueError("pair-keyed validation alignment requires unique user/target pairs")
    popularity = np.asarray(train_popularity, dtype=np.int64)
    readiness = _readiness_by_user(histories, artifacts)
    accumulator = CourseMacroAccumulator()
    keyed_records = {}
    expected_pair_set = set(pairs)
    for model in pair_keyed_models:
        indexed = {}
        with paths[model].open("r", encoding="utf-8") as handle:
            for native_index, line in enumerate(handle):
                record = json.loads(line)
                if record.get("analysis_split") != "validation":
                    raise ValueError(f"{model} seed {seed} contains a non-validation row")
                if record.get("model") != model or int(record.get("seed", -1)) != int(seed):
                    raise ValueError(f"{model} seed {seed} metadata mismatch while indexing")
                if int(record.get("sample_index", -1)) != native_index:
                    raise ValueError(f"{model} seed {seed} has non-sequential sample indices")
                pair = (int(record["user_id"]), int(record["target_item_id"]))
                if pair in indexed:
                    raise ValueError(f"{model} seed {seed} has duplicate user/target pairs")
                indexed[pair] = record
        actual_pair_set = set(indexed)
        if actual_pair_set != expected_pair_set:
            raise ValueError(
                f"{model} seed {seed} validation pair coverage mismatch: "
                f"missing={len(expected_pair_set - actual_pair_set)}, "
                f"extra={len(actual_pair_set - expected_pair_set)}"
            )
        keyed_records[model] = indexed
    handles = {
        model: path.open("r", encoding="utf-8")
        for model, path in paths.items()
        if model not in pair_keyed_models
    }
    performance = {
        model: defaultdict(lambda: {"count": 0, "hit": 0.0, "ndcg": 0.0})
        for model in paths
    }
    record_counts = defaultdict(int)

    try:
        for sample_index, (user_id, target_item_id) in enumerate(pairs):
            history = histories.get(user_id, np.empty(0, dtype=np.int64))
            records = {}
            for model in paths:
                if model in keyed_records:
                    record = keyed_records[model][(user_id, target_item_id)]
                    expected_sample_index = int(record["sample_index"])
                else:
                    line = handles[model].readline()
                    if not line:
                        raise ValueError(
                            f"{model} seed {seed} ended before validation row {sample_index}"
                        )
                    record = json.loads(line)
                    expected_sample_index = sample_index
                validate_validation_export_record(
                    record,
                    expected_model=model,
                    expected_seed=seed,
                    expected_sample_index=expected_sample_index,
                    expected_pair=(user_id, target_item_id),
                    expected_target_popularity=int(popularity[target_item_id]),
                    history_item_ids=history,
                    expected_top_k=expected_top_k,
                )
                _, list_rows = analyze_export_record(
                    record,
                    history_item_ids=history,
                    train_popularity=popularity,
                    artifacts=artifacts,
                    cutoffs=(metric_k,),
                    precomputed_readiness=readiness.get(user_id, 0.0),
                )
                accumulator.update(list_rows[0])

                prefix = [int(item) for item in record["recommended_item_ids"][:metric_k]]
                rank = prefix.index(target_item_id) + 1 if target_item_id in prefix else None
                stats = performance[model][target_item_id]
                stats["count"] += 1
                stats["hit"] += float(rank is not None)
                stats["ndcg"] += 1.0 / math.log2(rank + 1.0) if rank is not None else 0.0
                record_counts[model] += 1
                records[model] = record

            reference = next(iter(records.values()))
            for model, record in records.items():
                shared = ("seed", "user_id", "target_item_id", "target_popularity")
                if any(record[field] != reference[field] for field in shared):
                    raise ValueError(
                        f"cross-model validation row mismatch at seed {seed}, row {sample_index}: {model}"
                    )

        for model, handle in handles.items():
            if handle.readline():
                raise ValueError(f"{model} seed {seed} has records beyond validation coverage")
    finally:
        for handle in handles.values():
            handle.close()

    course_rows = accumulator.to_frame()
    performance_rows = []
    audit = {}
    for model in paths:
        target_stats = performance[model]
        for target_item_id, stats in target_stats.items():
            count = int(stats["count"])
            performance_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "target_item_id": int(target_item_id),
                    "recall_at_10": float(stats["hit"] / count),
                    "ndcg_at_10": float(stats["ndcg"] / count),
                }
            )
        target_count = len(target_stats)
        audit[model] = {
            "record_count": int(record_counts[model]),
            "target_course_count": int(target_count),
            f"R@{metric_k}": float(
                sum(stats["hit"] for stats in target_stats.values()) / record_counts[model]
            ),
            f"N@{metric_k}": float(
                sum(stats["ndcg"] for stats in target_stats.values()) / record_counts[model]
            ),
            f"course_macro_R@{metric_k}": float(
                np.mean([stats["hit"] / stats["count"] for stats in target_stats.values()])
            ),
            f"course_macro_N@{metric_k}": float(
                np.mean([stats["ndcg"] / stats["count"] for stats in target_stats.values()])
            ),
            "seen_item_leak_count": 0,
            "invalid_topk_count": 0,
        }

    performance_frame = pd.DataFrame(performance_rows)
    course_rows = course_rows.merge(
        performance_frame,
        on=["model", "seed", "target_item_id"],
        how="left",
        validate="one_to_one",
    )
    course_rows.insert(0, "analysis_split", "validation")
    course_rows["low_ndcg_at_10"] = (course_rows["ndcg_at_10"] <= 0.10).astype(float)
    course_rows["effective_coverage"] = (
        course_rows["cold_list_count"] / course_rows["list_count"]
    )
    course_rows["missingness"] = 1.0 - course_rows["effective_coverage"]
    return course_rows.sort_values(
        ["model", "seed", "target_item_id"], ignore_index=True
    ), audit


def seed_stratified_interval(
    course_rows: pd.DataFrame,
    *,
    value_column: str,
    n_bootstrap: int = 10_000,
    random_seed: int = 2027,
) -> tuple[float, float, float]:
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    if "seed" not in course_rows or value_column not in course_rows:
        raise ValueError(f"course rows must contain seed and {value_column}")

    seed_values = []
    for _, group in course_rows.groupby("seed", sort=True):
        values = group[value_column].to_numpy(dtype=np.float64)
        if values.size:
            seed_values.append(values)
    if not seed_values or not any(np.isfinite(values).any() for values in seed_values):
        raise ValueError(f"{value_column} has no finite course observations")

    point_seed_means = [
        float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        for values in seed_values
    ]
    point = float(np.nanmean(point_seed_means))
    rng = np.random.default_rng(int(random_seed))
    bootstrap_seed_means = []
    for values in seed_values:
        indices = rng.integers(
            0,
            values.size,
            size=(int(n_bootstrap), values.size),
        )
        sampled = values[indices]
        finite = np.isfinite(sampled)
        counts = finite.sum(axis=1)
        sums = np.where(finite, sampled, 0.0).sum(axis=1)
        means = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
        np.divide(sums, counts, out=means, where=counts > 0)
        bootstrap_seed_means.append(means)
    stacked = np.stack(bootstrap_seed_means, axis=1)
    finite_counts = np.isfinite(stacked).sum(axis=1)
    replicate_means = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    np.divide(
        np.nansum(stacked, axis=1),
        finite_counts,
        out=replicate_means,
        where=finite_counts > 0,
    )
    replicate_means = replicate_means[np.isfinite(replicate_means)]
    if not replicate_means.size:
        raise ValueError(f"{value_column} bootstrap produced no finite replicates")
    low, high = np.quantile(replicate_means, [0.025, 0.975])
    return point, float(low), float(high)


def summarize_validation_course_rows(
    course_rows: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    random_seed: int = 2027,
) -> pd.DataFrame:
    if set(course_rows.get("analysis_split", ())) != {"validation"}:
        raise ValueError("summary accepts validation-only course rows")
    models = set(map(str, course_rows.get("model", ())))
    if not models or not models.issubset(MODELS):
        raise ValueError(f"summary models must be a subset of {MODELS}")

    rows = []
    for model, group in course_rows.groupby("model", sort=True):
        coverage, _, _ = seed_stratified_interval(
            group,
            value_column="effective_coverage",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        missingness, _, _ = seed_stratified_interval(
            group,
            value_column="missingness",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        for metric, panel, direction in SUMMARY_METRICS:
            mean, low, high = seed_stratified_interval(
                group,
                value_column=metric,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
            values = group[metric].to_numpy(dtype=np.float64)
            observed_seed_count = sum(
                seed_group[metric].notna().any()
                for _, seed_group in group.groupby("seed", sort=True)
            )
            rows.append(
                {
                    "analysis_split": "validation",
                    "panel": panel,
                    "model": str(model),
                    "metric": metric,
                    "direction": direction,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "unit_count": int(len(group)),
                    "observed_unit_count": int(np.isfinite(values).sum()),
                    "seed_count": int(group["seed"].nunique()),
                    "observed_seed_count": int(observed_seed_count),
                    "list_count": int(group["list_count"].sum()),
                    "cold_list_count": int(group["cold_list_count"].sum()),
                    "effective_coverage": coverage,
                    "missingness": missingness,
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_random_seed": int(random_seed),
                }
            )
        ndcg = group["ndcg_at_10"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "analysis_split": "validation",
                "panel": "exposure",
                "model": str(model),
                "metric": "median_ndcg_at_10",
                "direction": "higher",
                "mean": float(np.median(ndcg)),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "unit_count": int(len(group)),
                "observed_unit_count": int(np.isfinite(ndcg).sum()),
                "seed_count": int(group["seed"].nunique()),
                "observed_seed_count": int(group["seed"].nunique()),
                "list_count": int(group["list_count"].sum()),
                "cold_list_count": int(group["cold_list_count"].sum()),
                "effective_coverage": coverage,
                "missingness": missingness,
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_random_seed": int(random_seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["panel", "metric", "model"], ignore_index=True)


def validation_export_paths(root: Path, seed: int) -> dict[str, Path]:
    base = Path(root) / "outputs" / "validation_motivation"
    split_id = f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    return {
        "pcgnn": base / "pcgnn" / split_id / "pcgnn_top20_validation.jsonl",
        "cgrc": base / "cgrc" / split_id / "top20_validation.jsonl",
    }


def _export_manifest_path(export_path: Path, model: str) -> Path:
    filename = "validation_export_manifest.json" if model == "pcgnn" else "export_manifest.json"
    return Path(export_path).parent / filename


def validate_validation_export_provenance(
    *,
    model: str,
    seed: int,
    export_path: Path,
    expected_record_count: int,
    expected_target_count: int,
) -> dict:
    if model not in MODELS:
        raise ValueError(f"unsupported validation model: {model}")
    manifest_path = _export_manifest_path(export_path, model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "seed": int(seed),
        "analysis_split": "validation",
        "top_k": 20,
        "record_count": int(expected_record_count),
        "target_course_count": int(expected_target_count),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{model} seed {seed} manifest mismatch for {key}")
    binding = manifest.get("topk_output", {})
    if Path(str(binding.get("path", ""))).resolve() != Path(export_path).resolve():
        raise ValueError(f"{model} seed {seed} manifest binds a different Top-K file")
    if binding.get("sha256") != _sha256(export_path):
        raise ValueError(f"{model} seed {seed} Top-K hash mismatch")
    checkpoints = [manifest["checkpoint"]] if model == "pcgnn" else manifest["checkpoints"]
    if any(checkpoint["sha256_before"] != checkpoint["sha256_after"] for checkpoint in checkpoints):
        raise ValueError(f"{model} seed {seed} checkpoint changed during replay")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "export_path": str(Path(export_path).resolve()),
        "export_sha256": _sha256(export_path),
        "checkpoints": checkpoints,
    }


def run_analysis(
    *,
    root: Path = ROOT,
    output_dir: Path | None = None,
    seeds=(2025, 2026, 2027),
    n_bootstrap: int = 10_000,
    random_seed: int = 2027,
) -> dict:
    root = Path(root).resolve()
    figure_dir = root / "paper_aaai27" / "figures"
    output_dir = Path(output_dir or figure_dir / "validation_motivation_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("[Validation motivation] Building course-structure artifacts", flush=True)
    artifacts, artifact_stats = build_real_risk_artifacts(root)
    n_items = int(artifacts.structural_complexity.size)
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    all_course_rows = []
    audit_rows = []
    provenance = []
    for seed in seeds:
        expected_pairs, histories, popularity = validation_seed_inputs(
            split_root,
            seed,
            n_items=n_items,
        )
        target_count = len({target for _, target in expected_pairs})
        if target_count != 34:
            raise ValueError(f"seed {seed} validation cold-course count is {target_count}, expected 34")
        paths = validation_export_paths(root, seed)
        for model, path in paths.items():
            provenance.append(
                {
                    "model": model,
                    "seed": int(seed),
                    **validate_validation_export_provenance(
                        model=model,
                        seed=seed,
                        export_path=path,
                        expected_record_count=len(expected_pairs),
                        expected_target_count=target_count,
                    ),
                }
            )
        course_rows, audit = analyze_validation_seed(
            seed=seed,
            expected_pairs=expected_pairs,
            histories=histories,
            train_popularity=popularity,
            artifacts=artifacts,
            model_paths=paths,
            expected_top_k=20,
            metric_k=10,
        )
        all_course_rows.append(course_rows)
        for model, values in audit.items():
            audit_rows.append({"model": model, "seed": int(seed), **values})
        print(
            f"[Validation motivation] seed={seed} rows={len(expected_pairs)} "
            f"courses={target_count} models={len(paths)}",
            flush=True,
        )

    course_rows = pd.concat(all_course_rows, ignore_index=True)
    counts = course_rows.groupby("model").size().to_dict()
    if counts != {"cgrc": 102, "pcgnn": 102}:
        raise ValueError(f"validation course-unit coverage mismatch: {counts}")
    if set(course_rows["analysis_split"]) != {"validation"}:
        raise ValueError("non-validation rows reached Figure 1 analysis")
    summary = summarize_validation_course_rows(
        course_rows,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    audit = pd.DataFrame(audit_rows).sort_values(["model", "seed"], ignore_index=True)

    course_path = output_dir / "course_macro.csv"
    audit_path = output_dir / "seed_export_audit.csv"
    summary_path = figure_dir / "mooccube_validation_motivation_summary.csv"
    manifest_path = figure_dir / "mooccube_validation_motivation_manifest.json"
    course_rows.to_csv(course_path, index=False)
    audit.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "analysis_split": "validation",
        "dataset": "MOOCCube",
        "protocol": "strict course-cold full-catalog ranking",
        "models": list(MODELS),
        "seeds": list(map(int, seeds)),
        "top_k": 20,
        "analysis_cutoff": 10,
        "cold_course_count_per_seed": 34,
        "course_units_per_model": 102,
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_random_seed": int(random_seed),
        "bootstrap_unit": "target course resampled within seed; seed means weighted equally",
        "null_hypothesis_tests": 0,
        "multiplicity_correction": "not applicable; Figure 1 reports no hypothesis tests",
        "cold_only_missingness_policy": "preserve missing lists; never zero-fill",
        "artifact_stats": artifact_stats,
        "exports": provenance,
        "row_counts": {
            "course_macro": int(len(course_rows)),
            "summary": int(len(summary)),
            "seed_export_audit": int(len(audit)),
        },
        "outputs": {
            "course_macro": str(course_path),
            "summary": str(summary_path),
            "seed_export_audit": str(audit_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[Validation motivation] wrote {summary_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-only Figure 1 motivation evidence")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=2027)
    args = parser.parse_args()
    run_analysis(
        root=args.root,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
