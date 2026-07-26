from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_topk_motivation import (
    COLD_RISK_COLUMNS,
    METRIC_DIRECTIONS,
    RISK_COLUMNS,
    _interpret_interval,
    _item_concept_counts,
    _paired_bootstrap_interval,
    _paired_permutation_pvalue,
    _seed_inputs,
    build_real_risk_artifacts,
    build_structural_complexity,
)


SCALES = ("p90", "p95", "max")
READINESS_KS = (3, 5, 10)
LIST_KEYS = (
    "model",
    "seed",
    "sample_index",
    "user_id",
    "target_item_id",
    "history_count",
    "history_bin",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_size_bin(size: int) -> str:
    size = int(size)
    if size < 1:
        return "0"
    if size <= 2:
        return "1-2"
    if size <= 4:
        return "3-4"
    if size <= 9:
        return "5-9"
    return "10+"


def build_readiness_caches(
    histories: dict[int, np.ndarray],
    complexities: dict[str, np.ndarray],
    *,
    readiness_ks=READINESS_KS,
    n_users: int,
) -> dict[tuple[str, int], np.ndarray]:
    caches = {
        (scale, int(k)): np.zeros(int(n_users), dtype=np.float64)
        for scale in complexities
        for k in readiness_ks
    }
    for user_id, raw_history in histories.items():
        user_id = int(user_id)
        history = np.asarray(raw_history, dtype=np.int64)
        if history.size == 0:
            continue
        for scale, complexity in complexities.items():
            values = np.asarray(complexity, dtype=np.float64)[history]
            for k in readiness_ks:
                keep = min(int(k), int(values.size))
                readiness = float(np.partition(values, -keep)[-keep:].mean())
                caches[(scale, int(k))][user_id] = readiness
    return caches


class PartialMeanAccumulator:
    def __init__(self, *, key_columns, value_columns) -> None:
        self.key_columns = tuple(key_columns)
        self.value_columns = tuple(value_columns)
        self._partials: list[pd.DataFrame] = []

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        missing = [
            column
            for column in self.key_columns + self.value_columns
            if column not in frame
        ]
        if missing:
            raise ValueError(f"accumulator input missing columns: {missing}")
        grouped = frame.groupby(
            list(self.key_columns),
            sort=False,
            observed=True,
            dropna=False,
        )[list(self.value_columns)].agg(["sum", "count"])
        grouped.columns = [f"{value}__{stat}" for value, stat in grouped.columns]
        self._partials.append(grouped.reset_index())

    def to_frame(self) -> pd.DataFrame:
        if not self._partials:
            return pd.DataFrame(columns=self.key_columns + self.value_columns)
        partial = pd.concat(self._partials, ignore_index=True)
        aggregate_columns = [
            column for column in partial.columns if column not in self.key_columns
        ]
        combined = (
            partial.groupby(
                list(self.key_columns),
                sort=True,
                observed=True,
                dropna=False,
                as_index=False,
            )[aggregate_columns]
            .sum()
        )
        output = combined[list(self.key_columns)].copy()
        first_count = f"{self.value_columns[0]}__count"
        output["list_count"] = combined[first_count].astype(np.int64)
        for value in self.value_columns:
            count = combined[f"{value}__count"].to_numpy(dtype=np.int64)
            total = combined[f"{value}__sum"].to_numpy(dtype=np.float64)
            output[value] = np.divide(
                total,
                count,
                out=np.full(total.shape, np.nan, dtype=np.float64),
                where=count > 0,
            )
            output[f"{value}_count"] = count
        return output


def _difficulty_name(prefix: str, scale: str, readiness_k: int) -> str:
    return f"{prefix}difficulty_gap__{scale}__k{int(readiness_k)}"


def summarize_recommendation_chunk(
    frame: pd.DataFrame,
    *,
    complexities: dict[str, np.ndarray],
    readiness_caches: dict[int, dict[tuple[str, int], np.ndarray]],
    history_counts: dict[int, np.ndarray],
    cutoff: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "model",
        "seed",
        "sample_index",
        "user_id",
        "target_item_id",
        "rank",
        "recommended_item_id",
        "is_cold",
        *RISK_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"recommendation frame missing columns: {missing}")
    top = frame.loc[frame["rank"].le(int(cutoff))].copy()
    if top.empty:
        return pd.DataFrame(), pd.DataFrame()

    top["history_count"] = 0
    for seed, indices in top.groupby("seed", sort=False).groups.items():
        seed = int(seed)
        users = top.loc[indices, "user_id"].to_numpy(dtype=np.int64)
        counts = np.asarray(history_counts[seed], dtype=np.int64)
        top.loc[indices, "history_count"] = counts[users]
    top["history_count"] = top["history_count"].astype(np.int64)
    top["history_bin"] = top["history_count"].map(history_size_bin)
    top["cold_proportion"] = top["is_cold"].astype(np.float64)

    for metric in RISK_COLUMNS:
        top[f"cold_{metric}"] = top[metric].where(top["is_cold"].eq(1))

    sensitivity_columns = []
    for scale, complexity in complexities.items():
        complexity = np.asarray(complexity, dtype=np.float64)
        for readiness_k in sorted(
            {key[1] for caches in readiness_caches.values() for key in caches if key[0] == scale}
        ):
            name = _difficulty_name("", scale, readiness_k)
            cold_name = _difficulty_name("cold_", scale, readiness_k)
            top[name] = 0.0
            for seed, indices in top.groupby("seed", sort=False).groups.items():
                seed = int(seed)
                users = top.loc[indices, "user_id"].to_numpy(dtype=np.int64)
                items = top.loc[indices, "recommended_item_id"].to_numpy(dtype=np.int64)
                readiness = readiness_caches[seed][(scale, int(readiness_k))][users]
                top.loc[indices, name] = np.maximum(0.0, complexity[items] - readiness)
            top[cold_name] = top[name].where(top["is_cold"].eq(1))
            sensitivity_columns.extend([name, cold_name])

    mean_columns = [
        *RISK_COLUMNS,
        "cold_proportion",
        *COLD_RISK_COLUMNS,
        *sensitivity_columns,
    ]
    lists = (
        top.groupby(
            list(LIST_KEYS),
            sort=False,
            observed=True,
            as_index=False,
        )[mean_columns]
        .mean()
    )
    return lists, top


def paired_group_statistics(
    course_rows: pd.DataFrame,
    *,
    group_columns,
    metrics: dict[str, str],
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
) -> pd.DataFrame:
    group_columns = tuple(group_columns)
    rng = np.random.default_rng(int(random_seed))
    rows = []
    grouped = course_rows.groupby(
        list(group_columns),
        sort=True,
        observed=True,
        dropna=False,
    )
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_columns, group_key))
        for metric, direction in metrics.items():
            pivot = group.pivot_table(
                index=["seed", "target_item_id"],
                columns="model",
                values=metric,
                aggfunc="first",
            )
            if "ckg_rl" not in pivot or "cgrc" not in pivot:
                differences = np.empty(0, dtype=np.float64)
            else:
                matched = pivot[["ckg_rl", "cgrc"]].dropna()
                differences = (
                    matched["ckg_rl"] - matched["cgrc"]
                ).to_numpy(dtype=np.float64)
            low, high = _paired_bootstrap_interval(differences, n_bootstrap, rng)
            p_value = _paired_permutation_pvalue(differences, n_permutations, rng)
            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "direction": direction,
                    "pair_count": int(differences.size),
                    "mean_difference_ckg_rl_minus_cgrc": (
                        float(differences.mean()) if differences.size else float("nan")
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "permutation_p_value": p_value,
                    "interpretation": _interpret_interval(direction, low, high),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def validate_primary_reproduction(
    observed: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> float:
    keys = ["model", "seed", "target_item_id"]
    primary = _difficulty_name("", "p95", 5)
    reference = frozen.loc[frozen["cutoff"].eq(10), keys + ["difficulty_gap"]]
    merged = observed[keys + [primary]].merge(
        reference,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("primary difficulty reproduction has unmatched course rows")
    delta = float(np.max(np.abs(merged[primary] - merged["difficulty_gap"])))
    if delta > float(tolerance):
        raise ValueError(
            f"primary difficulty reproduction drift {delta:.3e} exceeds {tolerance:.3e}"
        )
    return delta


def _long_sensitivity(course_wide: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "target_item_id", "list_count"]
    rows = []
    for scale in SCALES:
        for readiness_k in READINESS_KS:
            overall = _difficulty_name("", scale, readiness_k)
            cold = _difficulty_name("cold_", scale, readiness_k)
            frame = course_wide[keys + [overall, cold]].copy()
            frame["scale"] = scale
            frame["readiness_k"] = int(readiness_k)
            frame = frame.rename(
                columns={
                    overall: "difficulty_gap",
                    cold: "cold_difficulty_gap",
                }
            )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _structural_complexities(root: Path):
    artifacts, stats = build_real_risk_artifacts(root)
    data_path = root / "processed_data_hin_clean_pop5" / "stream_data.pkl"
    relation_dir = root / "MOOCCube" / "relations"
    full_df = pd.read_pickle(data_path)
    n_items = int(artifacts.structural_complexity.size)
    prerequisite_counts = np.asarray(
        artifacts.prerequisite_matrix,
        dtype=bool,
    ).sum(axis=1).astype(np.float64)
    concept_counts = _item_concept_counts(full_df, relation_dir, n_items)
    complexities = {
        "p90": build_structural_complexity(prerequisite_counts, concept_counts, 0.90),
        "p95": build_structural_complexity(prerequisite_counts, concept_counts, 0.95),
        "max": build_structural_complexity(prerequisite_counts, concept_counts, 1.00),
    }
    return full_df, complexities, stats


def run_robustness(
    *,
    root: Path,
    output_dir: Path,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
    chunksize: int = 200_000,
) -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(chunksize) % 20:
        raise ValueError("chunksize must be divisible by 20")

    full_df, complexities, artifact_stats = _structural_complexities(root)
    n_items = len(next(iter(complexities.values())))
    n_users = int(full_df["u_idx"].max()) + 1
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    readiness_caches = {}
    history_counts = {}
    for seed in (2025, 2026, 2027):
        _, histories, _ = _seed_inputs(split_root, seed, n_items)
        readiness_caches[seed] = build_readiness_caches(
            histories,
            complexities,
            readiness_ks=READINESS_KS,
            n_users=n_users,
        )
        counts = np.zeros(n_users, dtype=np.int32)
        for user_id, history in histories.items():
            counts[int(user_id)] = int(len(history))
        history_counts[seed] = counts

    sensitivity_columns = tuple(
        name
        for scale in SCALES
        for readiness_k in READINESS_KS
        for name in (
            _difficulty_name("", scale, readiness_k),
            _difficulty_name("cold_", scale, readiness_k),
        )
    )
    sensitivity_acc = PartialMeanAccumulator(
        key_columns=("model", "seed", "target_item_id"),
        value_columns=sensitivity_columns,
    )
    rank_values = (*RISK_COLUMNS, "cold_proportion")
    rank_acc = PartialMeanAccumulator(
        key_columns=("model", "seed", "target_item_id", "rank"),
        value_columns=rank_values,
    )
    history_values = (*RISK_COLUMNS, "cold_proportion", *COLD_RISK_COLUMNS)
    history_acc = PartialMeanAccumulator(
        key_columns=("model", "seed", "target_item_id", "history_bin"),
        value_columns=history_values,
    )

    recommendation_path = (
        root
        / "paper_aaai27"
        / "figures"
        / "p1_topk_motivation_analysis"
        / "recommendation_level.csv.gz"
    )
    processed_rows = 0
    for chunk_index, frame in enumerate(
        pd.read_csv(recommendation_path, compression="gzip", chunksize=int(chunksize)),
        start=1,
    ):
        if len(frame) % 20:
            raise ValueError("recommendation chunk breaks a Top-20 record boundary")
        lists, rank_rows = summarize_recommendation_chunk(
            frame,
            complexities=complexities,
            readiness_caches=readiness_caches,
            history_counts=history_counts,
            cutoff=10,
        )
        sensitivity_acc.update(lists)
        rank_acc.update(rank_rows)
        history_acc.update(lists)
        processed_rows += len(frame)
        print(
            f"[P1-ROBUST] chunk={chunk_index} rows={processed_rows}",
            flush=True,
        )

    sensitivity_wide = sensitivity_acc.to_frame()
    frozen_course = pd.read_csv(
        root
        / "paper_aaai27"
        / "figures"
        / "p1_topk_motivation_analysis"
        / "course_macro.csv"
    )
    primary_delta = validate_primary_reproduction(sensitivity_wide, frozen_course)
    sensitivity = _long_sensitivity(sensitivity_wide)
    sensitivity_paired = paired_group_statistics(
        sensitivity,
        group_columns=("scale", "readiness_k"),
        metrics={
            "difficulty_gap": "lower",
            "cold_difficulty_gap": "lower",
        },
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )

    rank_course = rank_acc.to_frame()
    rank_paired = paired_group_statistics(
        rank_course,
        group_columns=("rank",),
        metrics={metric: METRIC_DIRECTIONS[metric] for metric in rank_values},
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    history_course = history_acc.to_frame()
    history_paired = paired_group_statistics(
        history_course,
        group_columns=("history_bin",),
        metrics={metric: METRIC_DIRECTIONS[metric] for metric in history_values},
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )

    outputs = {
        "difficulty_sensitivity_course_macro.csv": sensitivity,
        "difficulty_sensitivity_paired.csv": sensitivity_paired,
        "rank_profile_course_macro.csv": rank_course,
        "rank_profile_paired.csv": rank_paired,
        "history_strata_course_macro.csv": history_course,
        "history_strata_paired.csv": history_paired,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    manifest = {
        "scales": list(SCALES),
        "readiness_k": list(READINESS_KS),
        "cutoff": 10,
        "random_seed": int(random_seed),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "input": {
            "path": str(recommendation_path),
            "sha256": _sha256(recommendation_path),
            "rows": int(processed_rows),
        },
        "primary_reproduction_max_abs_delta": float(primary_delta),
        "artifact_stats": artifact_stats,
        "outputs": {},
    }
    for name, frame in outputs.items():
        path = output_dir / name
        manifest["outputs"][name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": int(len(frame)),
        }
    manifest_path = output_dir / "robustness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[P1-ROBUST] wrote {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures" / "p1_risk_robustness",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--analysis-seed", type=int, default=2027)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    run_robustness(
        root=args.root,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        random_seed=args.analysis_seed,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
