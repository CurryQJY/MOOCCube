from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_risk_robustness import PartialMeanAccumulator
from paper_aaai27.scripts.analyze_p1_topk_motivation import (
    _paired_bootstrap_interval,
    _paired_permutation_pvalue,
)


DEFAULT_INPUT = (
    ROOT
    / "paper_aaai27"
    / "figures"
    / "p1_topk_motivation_analysis"
    / "recommendation_level.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "paper_aaai27" / "figures" / "method_motivation_analysis"
)

METRICS = (
    "prerequisite_gap",
    "concept_continuity",
    "difficulty_gap",
    "structural_redundancy",
)
METRIC_DIRECTIONS = {
    "prerequisite_gap": "lower",
    "concept_continuity": "higher",
    "difficulty_gap": "lower",
    "structural_redundancy": "lower",
}
VALUE_COLUMNS = (*METRICS, "cold_proportion")
RANK_KEYS = ("seed", "target_item_id", "rank")


def _prepare_recommendation_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    top_k: int,
) -> pd.DataFrame:
    required = {"model", "seed", "target_item_id", "rank", "is_cold", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"recommendation frame missing columns: {missing}")
    selected = frame.loc[
        frame["model"].eq(str(model))
        & frame["rank"].between(1, int(top_k), inclusive="both")
    ].copy()
    if selected.empty:
        return selected
    selected["cold_proportion"] = selected["is_cold"].astype(np.float64)
    return selected


def aggregate_rank_frame(
    frame: pd.DataFrame,
    *,
    model: str = "cgrc",
    top_k: int = 20,
) -> pd.DataFrame:
    accumulator = PartialMeanAccumulator(
        key_columns=RANK_KEYS,
        value_columns=VALUE_COLUMNS,
    )
    accumulator.update(_prepare_recommendation_frame(frame, model=model, top_k=top_k))
    return accumulator.to_frame()


def _validate_complete_ranks(rank_rows: pd.DataFrame, *, top_k: int) -> None:
    expected = tuple(range(1, int(top_k) + 1))
    observed = rank_rows.groupby(
        ["seed", "target_item_id"], sort=True, observed=True
    )["rank"].agg(lambda values: tuple(sorted(map(int, values))))
    if observed.empty or not observed.map(lambda ranks: ranks == expected).all():
        raise ValueError(f"each seed-course unit must contain complete ranks 1..{int(top_k)}")

    counts = rank_rows.groupby(
        ["seed", "target_item_id"], sort=True, observed=True
    )["list_count"].nunique()
    if not counts.eq(1).all():
        raise ValueError("rank rows within a seed-course unit must share list_count")


def build_bucket_course_rows(
    rank_rows: pd.DataFrame,
    *,
    cutoff: int = 10,
    top_k: int = 20,
) -> pd.DataFrame:
    if int(cutoff) < 1 or int(cutoff) >= int(top_k):
        raise ValueError("cutoff must split the requested rank range")
    required = {"seed", "target_item_id", "rank", "list_count", *VALUE_COLUMNS}
    missing = sorted(required.difference(rank_rows.columns))
    if missing:
        raise ValueError(f"rank-course frame missing columns: {missing}")
    _validate_complete_ranks(rank_rows, top_k=top_k)

    work = rank_rows.copy()
    work["bucket"] = np.where(work["rank"].le(int(cutoff)), "top10", "bottom10")
    bucket_rows = (
        work.groupby(
            ["seed", "target_item_id", "bucket"],
            sort=True,
            observed=True,
            as_index=False,
        )
        .agg(
            **{metric: (metric, "mean") for metric in VALUE_COLUMNS},
            list_count=("list_count", "first"),
            rank_count=("rank", "size"),
        )
    )
    expected_bucket_sizes = {"top10": int(cutoff), "bottom10": int(top_k) - int(cutoff)}
    valid_sizes = bucket_rows.apply(
        lambda row: int(row["rank_count"]) == expected_bucket_sizes[str(row["bucket"])],
        axis=1,
    )
    if not valid_sizes.all():
        raise ValueError("rank buckets do not contain the expected number of positions")
    return bucket_rows


def paired_rank_alignment(
    bucket_rows: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
) -> pd.DataFrame:
    required = {"seed", "target_item_id", "bucket", *METRICS}
    missing = sorted(required.difference(bucket_rows.columns))
    if missing:
        raise ValueError(f"bucket-course frame missing columns: {missing}")

    rng = np.random.default_rng(int(random_seed))
    rows = []
    for metric in METRICS:
        pivot = bucket_rows.pivot_table(
            index=["seed", "target_item_id"],
            columns="bucket",
            values=metric,
            aggfunc="first",
        )
        if "top10" not in pivot or "bottom10" not in pivot:
            raise ValueError("paired alignment requires top10 and bottom10 buckets")
        matched = pivot[["top10", "bottom10"]].dropna()
        raw = (
            matched["top10"].to_numpy(dtype=np.float64)
            - matched["bottom10"].to_numpy(dtype=np.float64)
        )
        raw_low, raw_high = _paired_bootstrap_interval(raw, n_bootstrap, rng)
        p_value = _paired_permutation_pvalue(raw, n_permutations, rng)
        direction = METRIC_DIRECTIONS[metric]
        if direction == "higher":
            favorable = raw
            favorable_low, favorable_high = raw_low, raw_high
        else:
            favorable = -raw
            favorable_low, favorable_high = -raw_high, -raw_low
        if favorable_low > 0.0:
            interpretation = "aligned"
        elif favorable_high < 0.0:
            interpretation = "counter_aligned"
        else:
            interpretation = "not_aligned"
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "pair_count": int(raw.size),
                "top10_mean": float(matched["top10"].mean()),
                "bottom10_mean": float(matched["bottom10"].mean()),
                "raw_difference_top10_minus_bottom10": float(raw.mean()),
                "raw_ci_low": raw_low,
                "raw_ci_high": raw_high,
                "favorable_alignment_effect": float(favorable.mean()),
                "favorable_ci_low": favorable_low,
                "favorable_ci_high": favorable_high,
                "permutation_p_value": p_value,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def stream_rank_course_macro(
    input_path: Path,
    *,
    model: str = "cgrc",
    top_k: int = 20,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    input_path = Path(input_path)
    usecols = ["model", "seed", "target_item_id", "rank", "is_cold", *METRICS]
    accumulator = PartialMeanAccumulator(
        key_columns=RANK_KEYS,
        value_columns=VALUE_COLUMNS,
    )
    rows_read = 0
    model_rows = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=int(chunksize)):
        rows_read += int(len(chunk))
        selected = _prepare_recommendation_frame(chunk, model=model, top_k=top_k)
        model_rows += int(len(selected))
        accumulator.update(selected)
    rank_rows = accumulator.to_frame()
    return rank_rows, {"rows_read": rows_read, "model_rows": model_rows}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str = "cgrc",
    cutoff: int = 10,
    top_k: int = 20,
    chunksize: int = 500_000,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
    expected_pairs: int | None = 204,
) -> dict[str, Path]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_rows, counts = stream_rank_course_macro(
        input_path,
        model=model,
        top_k=top_k,
        chunksize=chunksize,
    )
    bucket_rows = build_bucket_course_rows(rank_rows, cutoff=cutoff, top_k=top_k)
    paired = paired_rank_alignment(
        bucket_rows,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    pair_count = int(paired["pair_count"].min())
    if expected_pairs is not None and pair_count != int(expected_pairs):
        raise ValueError(f"expected {int(expected_pairs)} paired units, found {pair_count}")

    paths = {
        "rank_course_macro": output_dir / "rank_course_macro.csv",
        "bucket_course_macro": output_dir / "bucket_course_macro.csv",
        "rank_alignment_paired": output_dir / "rank_alignment_paired.csv",
        "manifest": output_dir / "manifest.json",
    }
    rank_rows.to_csv(paths["rank_course_macro"], index=False)
    bucket_rows.to_csv(paths["bucket_course_macro"], index=False)
    paired.to_csv(paths["rank_alignment_paired"], index=False)

    top10 = bucket_rows.loc[bucket_rows["bucket"].eq("top10")]
    manifest = {
        "analysis": "cgrc_baseline_method_motivation",
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            **counts,
        },
        "model": str(model),
        "cutoff": int(cutoff),
        "top_k": int(top_k),
        "random_seed": int(random_seed),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "rank_course_rows": int(len(rank_rows)),
        "bucket_course_rows": int(len(bucket_rows)),
        "paired_units": pair_count,
        "top10_cold_course_share": float(top10["cold_proportion"].mean()),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CGRC-only method motivation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--expected-pairs", type=int, default=204)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        expected_pairs=args.expected_pairs,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
