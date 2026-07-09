from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
SEEDS = [2025, 2026, 2027]
METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
METRIC_TOLERANCE = 5e-5
MAIN_SIGNIFICANCE_REFERENCES = {"MOOCCube": "CGRC", "Junyi": "ALDI", "COCO": "CCFCRec"}


@dataclass(frozen=True)
class ArtifactSpec:
    dataset: str
    method: str
    seed: int
    result_candidates: tuple[Path, ...]
    per_item_candidates: tuple[Path, ...]
    note: str = ""


def p(*parts: str | int) -> Path:
    return ROOT.joinpath(*(str(part) for part in parts))


def mooccube_split(seed: int) -> Path:
    return p("outputs", "content_delta_pop5", "static_item_cold_balanced", f"strict_item_cold_balanced_thr1_seed_{seed}")


def junyi_split(seed: int) -> Path:
    if seed == 2025:
        return p("outputs", "junyi", "official_prereq_seed2025", "strict_item_cold_balanced_thr1_seed_2025")
    return p("outputs", "junyi", "main_table_3seed", f"strict_item_cold_balanced_thr1_seed_{seed}")


def junyi_ours_split(seed: int) -> Path:
    return p("outputs", "junyi", "main_table_3seed", f"strict_item_cold_balanced_thr1_seed_{seed}")


def coco_split(seed: int) -> Path:
    return p("outputs", "coco", "single_seed_triage", "ours_full", f"strict_item_cold_balanced_thr1_seed_{seed}")


def add_specs(
    specs: list[ArtifactSpec],
    dataset: str,
    seed: int,
    method: str,
    base_dir: Path,
    result_file: str,
    per_item_file: str,
    alt_dirs: Iterable[Path] = (),
    note: str = "",
) -> None:
    dirs = (base_dir, *tuple(alt_dirs))
    specs.append(
        ArtifactSpec(
            dataset=dataset,
            method=method,
            seed=seed,
            result_candidates=tuple(d / result_file for d in dirs),
            per_item_candidates=tuple(d / per_item_file for d in dirs),
            note=note,
        )
    )


def build_specs() -> list[ArtifactSpec]:
    specs: list[ArtifactSpec] = []

    for seed in SEEDS:
        split = mooccube_split(seed)
        v1 = split / "main_table_balanced_itemmacro_v1"
        add_specs(specs, "MOOCCube", seed, "Popularity", v1, "popularity_static_result.json", "per_item_full_cold_popularity_static.csv")
        add_specs(specs, "MOOCCube", seed, "BPR", v1, "bpr_static_result.json", "per_item_full_cold_bpr_static.csv")
        add_specs(specs, "MOOCCube", seed, "LightGCN", v1, "lightgcn_static_result.json", "per_item_full_cold_lightgcn_static.csv")
        add_specs(
            specs,
            "MOOCCube",
            seed,
            "DropoutNet",
            split / "main_table_balanced_itemmacro_dropoutnet_official_teacher80_student120_v1",
            "dropoutnet_official_static_result.json",
            "per_item_full_cold_dropoutnet_official_static.csv",
            alt_dirs=(v1,),
            note="Primary table uses the official-protocol teacher80/student120 run.",
        )
        add_specs(specs, "MOOCCube", seed, "CCFCRec", v1, "ccfcrec_static_result.json", "per_item_full_cold_ccfcrec_static.csv")
        add_specs(specs, "MOOCCube", seed, "ALDI", v1, "aldi_official_static_result.json", "per_item_full_cold_aldi_official_static.csv")
        add_specs(
            specs,
            "MOOCCube",
            seed,
            "CGRC",
            split / "significance_cgrc_exact_reexport",
            "cgrc_paper_static_result.json",
            "per_item_full_cold_cgrc_paper_static.csv",
            alt_dirs=(split / "main_table_balanced_itemmacro_cgrc_paper_v1", split / "rq1_per_course_cgrc_export"),
            note="Exact re-export is the paired-test source; main-table aggregate and RQ1 export are fallbacks.",
        )
        add_specs(specs, "MOOCCube", seed, "USIM", v1, "usim_official_static_result.json", "per_item_full_cold_usim_official_static.csv")
        specs.append(
            ArtifactSpec(
                dataset="MOOCCube",
                method="CKG-RL",
                seed=seed,
                result_candidates=(
                    p(
                        "outputs",
                        "significance_per_item_exports",
                        "mooccube",
                        "ckg_rl_full",
                        f"strict_item_cold_balanced_thr1_seed_{seed}",
                        "final_fullrank_usim_feedback_fast3_content_delta_static.csv",
                    ),
                ),
                per_item_candidates=(
                    p(
                        "outputs",
                        "significance_per_item_exports",
                        "mooccube",
                        "ckg_rl_full",
                        f"strict_item_cold_balanced_thr1_seed_{seed}",
                        "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
                    ),
                ),
                note="Exact True/True main run.",
            )
        )

    for seed in SEEDS:
        split = junyi_split(seed)
        ours_split = junyi_ours_split(seed)
        suffix = "_strictfix" if seed in {2026, 2027} else ""
        add_specs(specs, "Junyi", seed, "Popularity", split / f"popularity_compare{suffix}", "popularity_static_result.json", "per_item_full_cold_popularity_static.csv")
        add_specs(specs, "Junyi", seed, "BPR", split / f"bpr_compare{suffix}", "bpr_static_result.json", "per_item_full_cold_bpr_static.csv")
        add_specs(specs, "Junyi", seed, "LightGCN", split / f"lightgcn_compare{suffix}", "lightgcn_static_result.json", "per_item_full_cold_lightgcn_static.csv")
        add_specs(specs, "Junyi", seed, "DropoutNet", split / f"dropoutnet_compare{suffix}", "dropoutnet_official_static_result.json", "per_item_full_cold_dropoutnet_official_static.csv")
        add_specs(specs, "Junyi", seed, "CCFCRec", split / f"ccfcrec_compare{suffix}", "ccfcrec_static_result.json", "per_item_full_cold_ccfcrec_static.csv")
        add_specs(specs, "Junyi", seed, "ALDI", split / f"aldi_compare{suffix}", "aldi_static_result.json", "per_item_full_cold_aldi_static.csv")
        add_specs(specs, "Junyi", seed, "CGRC", split / f"cgrc_paper_compare{suffix}", "cgrc_paper_static_result.json", "per_item_full_cold_cgrc_paper_static.csv")
        add_specs(specs, "Junyi", seed, "USIM", ours_split / "main_table_balanced_itemmacro_v1", "usim_official_static_result.json", "per_item_full_cold_usim_official_static.csv")
        specs.append(
            ArtifactSpec(
                dataset="Junyi",
                method="CKG-RL",
                seed=seed,
                result_candidates=(ours_split / "final_fullrank_usim_feedback_fast3_content_delta_static.csv",),
                per_item_candidates=(ours_split / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",),
            )
        )

    for seed in SEEDS:
        split = coco_split(seed)
        compare = split / "main_table_compare"
        usim_dir = split / "main_table_balanced_itemmacro_v1"
        add_specs(specs, "COCO", seed, "Popularity", compare, "popularity_static_result.json", "per_item_full_cold_popularity_static.csv")
        add_specs(specs, "COCO", seed, "BPR", compare, "bpr_static_result.json", "per_item_full_cold_bpr_static.csv")
        add_specs(specs, "COCO", seed, "LightGCN", compare, "lightgcn_static_result.json", "per_item_full_cold_lightgcn_static.csv")
        add_specs(specs, "COCO", seed, "DropoutNet", compare, "drop_static_result.json", "per_item_full_cold_drop_static.csv")
        add_specs(specs, "COCO", seed, "CCFCRec", compare, "ccfcrec_static_result.json", "per_item_full_cold_ccfcrec_static.csv")
        add_specs(specs, "COCO", seed, "ALDI", compare, "aldi_static_result.json", "per_item_full_cold_aldi_static.csv")
        add_specs(specs, "COCO", seed, "CGRC", compare, "cgrc_paper_static_result.json", "per_item_full_cold_cgrc_paper_static.csv")
        add_specs(specs, "COCO", seed, "USIM", usim_dir, "usim_official_static_result.json", "per_item_full_cold_usim_official_static.csv")
        specs.append(
            ArtifactSpec(
                dataset="COCO",
                method="CKG-RL",
                seed=seed,
                result_candidates=(split / "final_fullrank_usim_feedback_fast3_content_delta_static.csv",),
                per_item_candidates=(split / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",),
            )
        )

    return specs


def main_significance_required(dataset: str, method: str) -> bool:
    return method == "CKG-RL" or MAIN_SIGNIFICANCE_REFERENCES.get(dataset) == method


def main_significance_missing(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = frame.apply(
        lambda row: main_significance_required(str(row["dataset"]), str(row["method"])),
        axis=1,
    )
    return frame[required & frame["status"].ne("ready")].copy()


def first_existing(paths: Iterable[Path]) -> tuple[Path | None, int | None]:
    for idx, path in enumerate(paths):
        if path.exists():
            return path, idx
    return None, None


def load_json_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0] if data else {}
    if isinstance(data, dict):
        return data
    return {}


def result_metric(path: Path | None, metric: str) -> float:
    if path is None:
        return math.nan
    if path.suffix.lower() == ".json":
        row = load_json_result(path)
        block = row.get("full_cold_item_macro", {})
        if isinstance(block, dict) and metric in block:
            try:
                return float(block[metric])
            except (TypeError, ValueError):
                return math.nan
        return math.nan
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if frame.empty:
            return math.nan
        suffix = metric.lower().replace("@", "")
        candidates = [
            f"full_cold_item_macro_{suffix}",
            f"full_cold_item_macro_{suffix.replace('r', 'R').replace('n', 'N')}",
            metric,
        ]
        for col in candidates:
            if col in frame.columns:
                return float(frame[col].iloc[0])
    return math.nan


def per_item_summary(path: Path | None) -> dict[str, float | int]:
    if path is None:
        return {"per_item_count": 0, **{f"per_item_{m}": math.nan for m in METRICS}}
    frame = pd.read_csv(path)
    out: dict[str, float | int] = {"per_item_count": int(len(frame))}
    for metric in METRICS:
        out[f"per_item_{metric}"] = float(pd.to_numeric(frame[metric], errors="coerce").mean()) if metric in frame.columns else math.nan
    return out


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in build_specs():
        result_path, result_idx = first_existing(spec.result_candidates)
        per_item_path, per_item_idx = first_existing(spec.per_item_candidates)
        result_values = {metric: result_metric(result_path, metric) for metric in METRICS}
        per_item = per_item_summary(per_item_path)
        absdiffs: dict[str, float] = {}
        metrics_match = True
        for metric in METRICS:
            result_value = result_values[metric]
            per_item_value = float(per_item[f"per_item_{metric}"]) if not pd.isna(per_item[f"per_item_{metric}"]) else math.nan
            diff = abs(result_value - per_item_value) if not (pd.isna(result_value) or pd.isna(per_item_value)) else math.nan
            absdiffs[metric] = diff
            if pd.isna(diff) or diff > METRIC_TOLERANCE:
                metrics_match = False
        if result_path is None:
            status = "missing_result"
        elif per_item_path is None:
            status = "missing_per_item"
        elif result_idx == 0 and per_item_idx == 0 and metrics_match:
            status = "ready"
        elif result_idx == 0 and per_item_idx == 0:
            status = "metric_mismatch"
        else:
            status = "ready_alt"
        row: dict[str, object] = {
            "dataset": spec.dataset,
            "method": spec.method,
            "seed": spec.seed,
            "result_exists": result_path is not None,
            "per_item_exists": per_item_path is not None,
            "result_is_primary": result_idx == 0,
            "per_item_is_primary": per_item_idx == 0,
            "result_source": rel(result_path),
            "per_item_source": rel(per_item_path),
            "per_item_count": int(per_item["per_item_count"]),
            "status": status,
            "main_significance_required": main_significance_required(spec.dataset, spec.method),
            "note": spec.note,
            "result_candidates": " | ".join(rel(path) for path in spec.result_candidates),
            "per_item_candidates": " | ".join(rel(path) for path in spec.per_item_candidates),
        }
        for metric in METRICS:
            row[f"result_{metric}"] = result_values[metric]
            row[f"per_item_{metric}"] = float(per_item[f"per_item_{metric}"]) if not pd.isna(per_item[f"per_item_{metric}"]) else math.nan
            row[f"absdiff_{metric}"] = absdiffs[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = audit()
    out_path = FIG_DIR / "significance_input_audit.csv"
    missing_path = FIG_DIR / "significance_missing_inputs.csv"
    main_missing_path = FIG_DIR / "significance_main_missing_inputs.csv"
    frame.to_csv(out_path, index=False)
    frame[frame["status"].ne("ready")].to_csv(missing_path, index=False)
    main_missing = main_significance_missing(frame)
    main_missing.to_csv(main_missing_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {missing_path}")
    print(f"Wrote {main_missing_path}")
    summary = frame.groupby(["dataset", "status"]).size().reset_index(name="count")
    print(summary.to_string(index=False))
    if not main_missing.empty:
        print("\nMain-table significance needs attention:")
        cols = ["dataset", "method", "seed", "status", "result_source", "per_item_candidates"]
        print(main_missing.loc[:, cols].to_string(index=False))
    full_missing = frame[frame["status"].ne("ready")]
    if not full_missing.empty:
        print("\nFull per-item archive gaps (not all are needed for main-table significance):")
        cols = ["dataset", "method", "seed", "status", "result_source", "per_item_candidates"]
        print(full_missing.loc[:, cols].to_string(index=False))


if __name__ == "__main__":
    main()
