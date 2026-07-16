from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_topk_motivation import (  # noqa: E402
    LIST_METRIC_COLUMNS,
    RISK_COLUMNS,
    CourseMacroAccumulator,
    _native_export_metrics,
    _seed_inputs,
    _sha256,
    analyze_seed_export_pair,
    build_real_risk_artifacts,
    paired_course_statistics,
    summarize_course_macro,
    validate_export_provenance,
    validate_native_export_audit,
)


MECHANISM_COMPARISONS = (
    ("ckg_rl", "ckg_rl_wo_course_reward", "course_reward"),
    ("ckg_rl", "ckg_rl_wo_simulator", "simulator"),
)

MECHANISM_MODELS = tuple(
    dict.fromkeys(
        model
        for treatment, control, _ in MECHANISM_COMPARISONS
        for model in (treatment, control)
    )
)

COURSE_ROW_KEY = ("model", "seed", "target_item_id", "cutoff")


def _ablation_export_paths(root: Path, seed: int) -> dict[str, Path]:
    split_name = f"strict_item_cold_balanced_thr1_seed_{int(seed)}"
    export_root = Path(root) / "outputs" / "p1_motivation_topk"
    return {
        model: export_root / model / split_name / "top20_cold_test.jsonl"
        for model in MECHANISM_MODELS
        if model != "ckg_rl"
    }


def combine_full_and_ablation_course_rows(
    full_rows: pd.DataFrame,
    ablation_rows: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in COURSE_ROW_KEY
        if column not in full_rows.columns or column not in ablation_rows.columns
    ]
    if missing_columns:
        raise ValueError(
            "mechanism course rows missing key columns: "
            + ", ".join(sorted(set(missing_columns)))
        )

    full_model = full_rows.loc[full_rows["model"] == "ckg_rl"]
    ablation_models = ablation_rows.loc[
        ablation_rows["model"].isin(MECHANISM_MODELS[1:])
    ]
    combined = pd.concat([full_model, ablation_models], ignore_index=True)

    duplicate_mask = combined.duplicated(list(COURSE_ROW_KEY), keep=False)
    if duplicate_mask.any():
        duplicate_keys = combined.loc[duplicate_mask, list(COURSE_ROW_KEY)]
        raise ValueError(
            "duplicate mechanism course rows: "
            + duplicate_keys.drop_duplicates().to_dict(orient="records").__repr__()
        )

    pairing_key = list(COURSE_ROW_KEY[1:])
    reference_keys = set(
        map(
            tuple,
            combined.loc[combined["model"].eq("ckg_rl"), pairing_key]
            .astype(int)
            .itertuples(index=False, name=None),
        )
    )
    if not reference_keys:
        raise ValueError("mechanism course coverage mismatch: full CKG-RL is empty")
    for model in MECHANISM_MODELS[1:]:
        model_keys = set(
            map(
                tuple,
                combined.loc[combined["model"].eq(model), pairing_key]
                .astype(int)
                .itertuples(index=False, name=None),
            )
        )
        if model_keys != reference_keys:
            raise ValueError(
                f"mechanism course coverage mismatch for {model}: "
                f"missing={len(reference_keys - model_keys)}, "
                f"extra={len(model_keys - reference_keys)}"
            )

    model_order = {model: index for index, model in enumerate(MECHANISM_MODELS)}
    combined = combined.assign(
        _model_order=combined["model"].map(model_order).astype(int)
    )
    return (
        combined.sort_values(
            ["_model_order", "seed", "target_item_id", "cutoff"],
            kind="stable",
        )
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def analyze_ablation_course_rows(
    *,
    root: Path,
    seeds=(2025, 2026, 2027),
    artifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    root = Path(root).resolve()
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    n_items = int(artifacts.structural_complexity.size)
    course_accumulator = CourseMacroAccumulator()
    audit_rows = []
    export_paths = []

    for seed in map(int, seeds):
        expected_pairs, histories, popularity = _seed_inputs(
            split_root,
            seed,
            n_items,
        )
        paths = _ablation_export_paths(root, seed)
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
            expected_pairs=expected_pairs,
            histories=histories,
            train_popularity=popularity,
            artifacts=artifacts,
            recommendation_sink=lambda row: None,
            list_sink=lambda row: None,
            course_accumulator=course_accumulator,
            expected_top_k=20,
            cutoffs=(10, 20),
            metric_k=10,
        )
        for model, model_audit in seed_audit.items():
            native = _native_export_metrics(root, seed, model)
            deltas = validate_native_export_audit(model_audit, native)
            audit_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    **model_audit,
                    **{
                        f"native_delta_{metric}": delta
                        for metric, delta in deltas.items()
                    },
                }
            )

    return (
        course_accumulator.to_frame(),
        pd.DataFrame(audit_rows),
        export_paths,
    )


def load_full_ckgrl_course_rows(
    full_analysis_dir: Path,
    *,
    seeds=(2025, 2026, 2027),
) -> pd.DataFrame:
    full_analysis_dir = Path(full_analysis_dir).resolve()
    manifest_path = full_analysis_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_seeds = tuple(map(int, seeds))
    if tuple(map(int, manifest.get("seeds", ()))) != expected_seeds:
        raise ValueError("full analysis seed mismatch")
    if tuple(map(int, manifest.get("cutoffs", ()))) != (10, 20):
        raise ValueError("full analysis cutoff mismatch")
    if int(manifest.get("top_k", -1)) != 20:
        raise ValueError("full analysis Top-K mismatch")

    course_path = full_analysis_dir / "course_macro.csv"
    bound_path = Path(manifest.get("outputs", {}).get("course_macro", ""))
    if not bound_path.is_absolute():
        bound_path = full_analysis_dir / bound_path
    if bound_path.resolve() != course_path.resolve():
        raise ValueError("full analysis course-macro path mismatch")

    rows = pd.read_csv(course_path)
    required = list(COURSE_ROW_KEY) + [
        "list_count",
        "cold_list_count",
        *LIST_METRIC_COLUMNS,
    ]
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"full analysis course rows missing columns: {missing}")
    full_rows = rows.loc[rows["model"].eq("ckg_rl"), required].copy()
    if full_rows.empty:
        raise ValueError("full analysis contains no CKG-RL course rows")
    if full_rows.duplicated(list(COURSE_ROW_KEY)).any():
        raise ValueError("duplicate full CKG-RL course rows")
    if set(map(int, full_rows["seed"].unique())) != set(expected_seeds):
        raise ValueError("full analysis CKG-RL seed coverage mismatch")
    if set(map(int, full_rows["cutoff"].unique())) != {10, 20}:
        raise ValueError("full analysis CKG-RL cutoff coverage mismatch")

    for seed in expected_seeds:
        seed_rows = full_rows.loc[full_rows["seed"].eq(seed)]
        target_sets = {
            cutoff: set(
                map(
                    int,
                    seed_rows.loc[
                        seed_rows["cutoff"].eq(cutoff),
                        "target_item_id",
                    ],
                )
            )
            for cutoff in (10, 20)
        }
        if not target_sets[10] or target_sets[10] != target_sets[20]:
            raise ValueError(
                f"full analysis CKG-RL target-course coverage mismatch for seed {seed}"
            )

    return full_rows.sort_values(
        ["model", "seed", "target_item_id", "cutoff"],
        kind="stable",
    ).reset_index(drop=True)


def build_mechanism_claim_audit(
    paired: pd.DataFrame,
    *,
    cutoff: int = 10,
) -> pd.DataFrame:
    selected = paired.loc[paired["cutoff"].eq(int(cutoff))].copy()
    lower_better = selected["direction"].eq("lower")
    selected["favorable_effect"] = np.where(
        lower_better,
        -selected["mean_difference"].astype(float),
        selected["mean_difference"].astype(float),
    )
    selected["favorable_ci_low"] = np.where(
        lower_better,
        -selected["bootstrap_ci_high"].astype(float),
        selected["bootstrap_ci_low"].astype(float),
    )
    selected["favorable_ci_high"] = np.where(
        lower_better,
        -selected["bootstrap_ci_low"].astype(float),
        selected["bootstrap_ci_high"].astype(float),
    )
    selected["claim_status"] = np.select(
        (
            selected["direction"].eq("descriptive"),
            selected["favorable_ci_low"].gt(0.0),
            selected["favorable_ci_high"].lt(0.0),
        ),
        ("descriptive_exposure", "supports_mechanism", "adverse_tradeoff"),
        default="inconclusive",
    )
    columns = [
        "comparison_role",
        "cutoff",
        "metric",
        "direction",
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "permutation_p_value",
        "favorable_effect",
        "favorable_ci_low",
        "favorable_ci_high",
        "claim_status",
    ]
    return selected[columns].reset_index(drop=True)


def run_analysis(
    *,
    root: Path,
    full_analysis_dir: Path,
    output_dir: Path,
    seeds=(2025, 2026, 2027),
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    random_seed: int = 2027,
) -> dict:
    root = Path(root).resolve()
    full_analysis_dir = Path(full_analysis_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_rows = load_full_ckgrl_course_rows(full_analysis_dir, seeds=seeds)
    print("[P1 mechanisms] Building model-neutral course artifacts", flush=True)
    artifacts, artifact_stats = build_real_risk_artifacts(root)
    ablation_rows, audit_rows, export_paths = analyze_ablation_course_rows(
        root=root,
        seeds=seeds,
        artifacts=artifacts,
    )
    course_rows = combine_full_and_ablation_course_rows(full_rows, ablation_rows)
    seed_summary, model_summary = summarize_course_macro(course_rows)
    paired = paired_course_statistics(
        course_rows,
        comparisons=MECHANISM_COMPARISONS,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    claim_audit = build_mechanism_claim_audit(paired, cutoff=10)

    finite_columns = list(RISK_COLUMNS) + ["cold_proportion"]
    if not np.isfinite(course_rows[finite_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("mechanism course-macro output contains non-finite primary values")
    if paired["pair_count"].eq(0).any():
        raise ValueError("mechanism paired statistics contain empty comparisons")

    course_path = output_dir / "course_macro.csv"
    ablation_path = output_dir / "ablation_course_macro.csv"
    seed_summary_path = output_dir / "seed_summary.csv"
    model_summary_path = output_dir / "model_summary.csv"
    paired_path = output_dir / "paired_statistics.csv"
    claim_audit_path = output_dir / "claim_audit_top10.csv"
    audit_path = output_dir / "seed_export_audit.csv"
    course_rows.to_csv(course_path, index=False)
    ablation_rows.to_csv(ablation_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    claim_audit.to_csv(claim_audit_path, index=False)
    audit_rows.to_csv(audit_path, index=False)

    full_inputs = (
        full_analysis_dir / "analysis_manifest.json",
        full_analysis_dir / "course_macro.csv",
    )
    manifest = {
        "analysis": "p1_motivation_mechanisms",
        "seeds": list(map(int, seeds)),
        "cutoffs": [10, 20],
        "top_k": 20,
        "random_seed": int(random_seed),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "full_model_reused": True,
        "recomputed_models": list(MECHANISM_MODELS[1:]),
        "comparisons": [
            {
                "treatment": treatment,
                "baseline": baseline,
                "mechanism": mechanism,
            }
            for treatment, baseline, mechanism in MECHANISM_COMPARISONS
        ],
        "artifact_stats": artifact_stats,
        "full_analysis_input_sha256": {
            str(path.resolve()): _sha256(path) for path in full_inputs
        },
        "ablation_export_sha256": {
            str(path.resolve()): _sha256(path) for path in export_paths
        },
        "row_counts": {
            "course_macro": int(len(course_rows)),
            "ablation_course_macro": int(len(ablation_rows)),
            "seed_summary": int(len(seed_summary)),
            "model_summary": int(len(model_summary)),
            "paired_statistics": int(len(paired)),
            "claim_audit_top10": int(len(claim_audit)),
            "seed_export_audit": int(len(audit_rows)),
        },
        "outputs": {
            "course_macro": str(course_path),
            "ablation_course_macro": str(ablation_path),
            "seed_summary": str(seed_summary_path),
            "model_summary": str(model_summary_path),
            "paired_statistics": str(paired_path),
            "claim_audit_top10": str(claim_audit_path),
            "seed_export_audit": str(audit_path),
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[P1 mechanisms] wrote analysis to {output_dir}", flush=True)
    return manifest


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--full-analysis-dir",
        type=Path,
        default=(
            default_root
            / "paper_aaai27"
            / "figures"
            / "p1_topk_motivation_analysis"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            default_root
            / "paper_aaai27"
            / "figures"
            / "p1_motivation_mechanism_analysis"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--analysis-seed", type=int, default=2027)
    args = parser.parse_args()
    run_analysis(
        root=args.root,
        full_analysis_dir=args.full_analysis_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        random_seed=args.analysis_seed,
    )


if __name__ == "__main__":
    main()
