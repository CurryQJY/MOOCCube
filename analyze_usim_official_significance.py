"""CKG-RL vs official USIM significance tests for the paper main tables.

The preferred unit is a matched seed-item pair from per-item item-macro exports.
For the MOOCCube main run, the historical CKG-RL table artifacts did not include
per-item exports, so the script falls back to paired-seed tests and records that
analysis unit explicitly in the output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
SEEDS = [2025, 2026, 2027]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display: str
    seed_roots: dict[int, Path]
    ours_result_root: Path | None = None


DATASETS = {
    "mooccube": DatasetConfig(
        name="mooccube",
        display="MOOCCube",
        seed_roots={
            2025: Path("outputs/content_delta_pop5/static_item_cold_balanced"),
            2026: Path("outputs/content_delta_pop5/static_item_cold_balanced"),
            2027: Path("outputs/content_delta_pop5/static_item_cold_balanced"),
        },
        ours_result_root=Path("outputs/content_delta_pop5/course_ablation_e60_3seed/full"),
    ),
    "junyi": DatasetConfig(
        name="junyi",
        display="Junyi",
        seed_roots={
            2025: Path("outputs/junyi/mask_ablation/mask_tt"),
            2026: Path("outputs/junyi/main_table_3seed"),
            2027: Path("outputs/junyi/main_table_3seed"),
        },
    ),
    "coco": DatasetConfig(
        name="coco",
        display="COCO",
        seed_roots={
            2025: Path("outputs/coco/single_seed_triage/ours_full"),
            2026: Path("outputs/coco/single_seed_triage/ours_full"),
            2027: Path("outputs/coco/single_seed_triage/ours_full"),
        },
    ),
}


def split_name(seed: int, cold_threshold: int) -> str:
    return f"strict_item_cold_balanced_thr{cold_threshold}_seed_{seed}"


def split_dir(root: Path, seed: int, cold_threshold: int) -> Path:
    return root / split_name(seed, cold_threshold)


def ours_per_item_path(split: Path, scope: str) -> Path:
    return split / f"per_item_full_{scope}_usim_feedback_fast3_content_delta_static.csv"


def official_per_item_path(split: Path, result_subdir: str, scope: str) -> Path:
    return split / result_subdir / f"per_item_full_{scope}_usim_official_static.csv"


def ours_seed_metric_path(cfg: DatasetConfig, seed: int, cold_threshold: int) -> Path:
    if cfg.ours_result_root is None:
        root = cfg.seed_roots[seed]
    else:
        root = cfg.ours_result_root
    return split_dir(root, seed, cold_threshold) / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"


def official_seed_metric_path(cfg: DatasetConfig, seed: int, cold_threshold: int, result_subdir: str) -> Path:
    return split_dir(cfg.seed_roots[seed], seed, cold_threshold) / result_subdir / "usim_official_static_result.json"


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def sig_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    max_elems: int = 8_000_000,
) -> tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    chunk = max(1, min(n_boot, max_elems // max(1, diffs.size)))
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        end = min(n_boot, start + chunk)
        idx = rng.integers(0, diffs.size, size=(end - start, diffs.size))
        means[start:end] = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_randomization_pvalue(
    diffs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    max_elems: int = 8_000_000,
) -> float:
    if diffs.size == 0:
        return float("nan")
    observed = abs(float(diffs.mean()))
    if observed <= 1e-15:
        return 1.0
    chunk = max(1, min(n_perm, max_elems // max(1, diffs.size)))
    extreme = 0
    done = 0
    for start in range(0, n_perm, chunk):
        end = min(n_perm, start + chunk)
        signs = rng.integers(0, 2, size=(end - start, diffs.size), dtype=np.int8)
        signed = np.where(signs == 1, diffs, -diffs)
        perm_means = np.abs(signed.mean(axis=1))
        extreme += int(np.count_nonzero(perm_means >= observed - 1e-15))
        done += end - start
    return float((extreme + 1) / (done + 1))


def paired_ttest_pvalue(ours: np.ndarray, baseline: np.ndarray) -> float:
    if ours.size != baseline.size:
        raise ValueError("paired_ttest_pvalue requires equal-length arrays")
    if ours.size < 2:
        return float("nan")
    result = stats.ttest_rel(ours, baseline)
    return float(result.pvalue)


def load_per_item_pair(
    seed: str,
    scope: str,
    ours_file: Path,
    baseline_file: Path,
    metrics: list[str],
) -> pd.DataFrame:
    left = pd.read_csv(ours_file)
    right = pd.read_csv(baseline_file)
    keep_cols = ["item_id", "count", *metrics]
    missing_left = sorted(set(keep_cols) - set(left.columns))
    missing_right = sorted(set(keep_cols) - set(right.columns))
    if missing_left or missing_right:
        raise ValueError(
            f"Missing per-item columns seed={seed} scope={scope}: "
            f"ours={missing_left} baseline={missing_right}"
        )
    left = left[keep_cols].copy()
    right = right[keep_cols].copy()
    left["item_id"] = left["item_id"].astype(int)
    right["item_id"] = right["item_id"].astype(int)
    merged = left.merge(right, on="item_id", suffixes=("_ours", "_usim"), how="inner")
    if len(merged) != len(left) or len(merged) != len(right):
        only_left = sorted(set(left["item_id"]) - set(right["item_id"]))[:10]
        only_right = sorted(set(right["item_id"]) - set(left["item_id"]))[:10]
        raise ValueError(
            f"Unpaired item ids seed={seed} scope={scope}: "
            f"ours_only={only_left} usim_only={only_right}"
        )
    merged = merged.sort_values("item_id").reset_index(drop=True)
    if not np.array_equal(merged["count_ours"].to_numpy(), merged["count_usim"].to_numpy()):
        mismatch = merged.loc[merged["count_ours"] != merged["count_usim"], "item_id"].head(10).tolist()
        raise ValueError(f"Count mismatch seed={seed} scope={scope}: {mismatch}")
    merged.insert(0, "scope", scope)
    merged.insert(0, "seed", seed)
    merged["count"] = merged["count_ours"]
    for metric in metrics:
        merged[f"{metric}_diff"] = merged[f"{metric}_ours"] - merged[f"{metric}_usim"]
    return merged


def metric_key(scope: str, metric: str) -> str:
    prefix = "full_cold_item_macro" if scope == "cold" else "full_hot_item_macro"
    kind = metric[0].lower()
    k = metric.split("@", 1)[1]
    return f"{prefix}_{kind}{k}"


def load_seed_pair(
    cfg: DatasetConfig,
    scope: str,
    metric: str,
    cold_threshold: int,
    result_subdir: str,
) -> tuple[np.ndarray, np.ndarray]:
    ours_values = []
    usim_values = []
    key = metric_key(scope, metric)
    official_section = "full_cold_item_macro" if scope == "cold" else "full_hot_item_macro"
    for seed in SEEDS:
        ours_path = ours_seed_metric_path(cfg, seed, cold_threshold)
        official_path = official_seed_metric_path(cfg, seed, cold_threshold, result_subdir)
        if not ours_path.exists():
            raise FileNotFoundError(f"Missing CKG-RL seed metric file: {ours_path}")
        if not official_path.exists():
            raise FileNotFoundError(f"Missing official USIM seed metric file: {official_path}")
        ours = pd.read_csv(ours_path)
        official = pd.read_json(official_path).iloc[0].to_dict()
        ours_values.append(float(ours.loc[0, key]))
        usim_values.append(float(official[official_section][metric]))
    return np.asarray(ours_values, dtype=np.float64), np.asarray(usim_values, dtype=np.float64)


def summarize_diffs(
    dataset: str,
    scope: str,
    analysis_unit: str,
    metric: str,
    ours_values: np.ndarray,
    baseline_values: np.ndarray,
    bootstrap: int,
    randomization: int,
    rng: np.random.Generator,
    test_name: str,
) -> dict[str, object]:
    diffs = ours_values - baseline_values
    ci_low, ci_high = paired_bootstrap_ci(diffs, bootstrap, rng)
    if test_name == "sign_randomization":
        p_value = sign_randomization_pvalue(diffs, randomization, rng)
    elif test_name == "paired_ttest":
        p_value = paired_ttest_pvalue(ours_values, baseline_values)
    else:
        raise ValueError(f"Unsupported test: {test_name}")
    return {
        "dataset": dataset,
        "scope": scope,
        "comparison": "ckg_rl_vs_usim_official",
        "analysis_unit": analysis_unit,
        "test": test_name,
        "metric": metric,
        "paired_units": int(diffs.size),
        "ours_mean": float(ours_values.mean()),
        "usim_mean": float(baseline_values.mean()),
        "mean_diff": float(diffs.mean()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "p_value": p_value,
        "wins": int(np.count_nonzero(diffs > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(diffs) <= 1e-12)),
        "losses": int(np.count_nonzero(diffs < -1e-12)),
    }


def analyze_dataset(
    cfg: DatasetConfig,
    scopes: list[str],
    metrics: list[str],
    cold_threshold: int,
    result_subdir: str,
    bootstrap: int,
    randomization: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    summary_rows = []
    for scope in scopes:
        scope_frames = []
        per_item_available = True
        for seed in SEEDS:
            split = split_dir(cfg.seed_roots[seed], seed, cold_threshold)
            ours_file = ours_per_item_path(split, scope)
            usim_file = official_per_item_path(split, result_subdir, scope)
            if not ours_file.exists() or not usim_file.exists():
                per_item_available = False
                break
            frame = load_per_item_pair(str(seed), scope, ours_file, usim_file, metrics)
            frame.insert(0, "dataset", cfg.name)
            scope_frames.append(frame)
            detail_frames.append(frame)

        if per_item_available:
            pooled = pd.concat(scope_frames, ignore_index=True)
            for metric in metrics:
                summary_rows.append(
                    summarize_diffs(
                        cfg.name,
                        scope,
                        "pooled_seed_item",
                        metric,
                        pooled[f"{metric}_ours"].to_numpy(dtype=np.float64),
                        pooled[f"{metric}_usim"].to_numpy(dtype=np.float64),
                        bootstrap,
                        randomization,
                        rng,
                        "sign_randomization",
                    )
                )
        else:
            for metric in metrics:
                ours_values, usim_values = load_seed_pair(cfg, scope, metric, cold_threshold, result_subdir)
                summary_rows.append(
                    summarize_diffs(
                        cfg.name,
                        scope,
                        "paired_seed",
                        metric,
                        ours_values,
                        usim_values,
                        bootstrap,
                        randomization,
                        rng,
                        "paired_ttest",
                    )
                )

    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    summary["p_holm_by_dataset_scope"] = np.nan
    for _, idx in summary.groupby(["dataset", "scope"]).groups.items():
        idx_list = list(idx)
        summary.loc[idx_list, "p_holm_by_dataset_scope"] = holm_adjust(
            summary.loc[idx_list, "p_value"].astype(float).tolist()
        )
    summary["sig"] = summary["p_value"].map(sig_marker)
    summary["sig_holm"] = summary["p_holm_by_dataset_scope"].map(sig_marker)
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mooccube", "junyi", "coco"])
    parser.add_argument("--scopes", nargs="+", choices=["cold", "hot"], default=["cold"])
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--result-subdir", default="main_table_balanced_itemmacro_v1")
    parser.add_argument("--out-dir", default="outputs/usim_official_3datasets_3seed/significance")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--randomization", type=int, default=50000)
    parser.add_argument("--rng-seed", type=int, default=20260701)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    details = []
    summaries = []
    for name in args.datasets:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        detail, summary = analyze_dataset(
            DATASETS[name],
            args.scopes,
            args.metrics,
            args.cold_threshold,
            args.result_subdir,
            args.bootstrap,
            args.randomization,
            rng,
        )
        if not detail.empty:
            details.append(detail)
        summaries.append(summary)

    detail_all = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_all = pd.concat(summaries, ignore_index=True)
    detail_path = out_dir / "ckg_rl_vs_usim_official_detail.csv"
    summary_path = out_dir / "ckg_rl_vs_usim_official_summary.csv"
    if not detail_all.empty:
        detail_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)

    print(f"Saved {summary_path}")
    if not detail_all.empty:
        print(f"Saved {detail_path}")
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
