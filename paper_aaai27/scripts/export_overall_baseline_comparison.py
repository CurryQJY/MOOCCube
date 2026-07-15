from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_aaai27"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_significance_inputs as cold_audit
import export_warm_target_table as warm_export


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
DATASETS = ("MOOCCube", "Junyi", "COCO")
SEEDS = (2025, 2026, 2027)
METHOD_ORDER = (
    "Popularity",
    "BPR",
    "DropoutNet",
    "LightGCN",
    "CCFCRec",
    "ALDI",
    "KGRec",
    "CGRC",
    "PCGNN",
    "USIM",
    "SEMCo",
    "CKG-RL",
)
OUTPUT_DIR = PAPER / "figures/overall_baseline_comparison"


def weighted_overall(
    cold: float,
    cold_count: int,
    hot: float,
    hot_count: int,
) -> float:
    if not math.isfinite(float(cold)) or not math.isfinite(float(hot)):
        raise ValueError("cold and hot metrics must be finite")
    if cold_count <= 0 or hot_count <= 0:
        raise ValueError("cold and hot course counts must be positive")
    return (
        float(cold) * int(cold_count) + float(hot) * int(hot_count)
    ) / (int(cold_count) + int(hot_count))


def validate_direct_overall(
    reconstructed: float,
    direct: float,
    tolerance: float = 5e-5,
) -> None:
    if not math.isfinite(float(direct)) or abs(
        float(reconstructed) - float(direct)
    ) > float(tolerance):
        raise ValueError(
            "direct overall mismatch: "
            f"reconstructed={reconstructed:.12g}, direct={direct:.12g}"
        )


def unavailable_row(
    dataset: str,
    method: str,
    seed: int,
    reason: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "status": "unavailable_missing_warm_targets",
        "reason": reason,
        "aggregation_route": "unavailable",
        "cold_count": 0,
        "hot_count": 0,
        "cold_source": "",
        "hot_source": "",
        **{metric: math.nan for metric in METRICS},
    }


def build_seed_row(
    dataset: str,
    method: str,
    seed: int,
    cold: dict[str, float],
    hot: dict[str, float],
    cold_count: int,
    hot_count: int,
    cold_source: str,
    hot_source: str,
    direct: dict[str, float] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "status": "ready",
        "reason": "",
        "aggregation_route": "cold_hot_course_count_weighted",
        "cold_count": int(cold_count),
        "hot_count": int(hot_count),
        "cold_source": cold_source,
        "hot_source": hot_source,
    }
    for metric in METRICS:
        if metric not in cold or metric not in hot:
            raise ValueError(f"missing metric {metric} for {dataset}/{method}/{seed}")
        value = weighted_overall(
            cold[metric], cold_count, hot[metric], hot_count
        )
        if direct is not None and metric in direct:
            validate_direct_overall(value, direct[metric])
        row[metric] = value
    return row


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def per_item_metrics(path: Path) -> tuple[dict[str, float], int]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty per-item file: {path}")
    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"missing per-item metrics {missing}: {path}")
    metrics = {
        metric: float(pd.to_numeric(frame[metric], errors="raise").mean())
        for metric in METRICS
    }
    return metrics, int(len(frame))


def load_result_payload(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return frame.iloc[0].to_dict() if not frame.empty else {}
    raise ValueError(f"unsupported result source: {path}")


def result_group_metrics(
    path: Path,
    group: str,
) -> tuple[dict[str, float], int]:
    payload = load_result_payload(path)
    block_name = f"full_{group}_item_macro"
    block = payload.get(block_name, {})
    if isinstance(block, dict) and all(metric in block for metric in METRICS):
        count = int(payload.get(f"count_full_{group}_item_macro", 0) or 0)
        return {metric: float(block[metric]) for metric in METRICS}, count

    values: dict[str, float] = {}
    for metric in METRICS:
        suffix = metric.lower().replace("@", "")
        column = f"full_{group}_item_macro_{suffix}"
        if column not in payload:
            raise ValueError(f"missing {block_name} metric {metric}: {path}")
        values[metric] = float(payload[column])
    count = int(payload.get(f"full_{group}_item_macro_count", 0) or 0)
    return values, count


def load_group_metrics(
    result_path: Path,
    per_item_path: Path | None,
    group: str,
) -> tuple[dict[str, float], int, str]:
    if per_item_path is not None:
        metrics, count = per_item_metrics(per_item_path)
        return metrics, count, relative(per_item_path)
    metrics, count = result_group_metrics(result_path, group)
    if count <= 0:
        raise ValueError(f"missing {group} course count: {result_path}")
    return metrics, count, relative(result_path)


def collect_core_rows() -> list[dict[str, object]]:
    warm_by_key = {
        (spec.dataset, spec.method, spec.seed): spec
        for spec in warm_export.core_specs()
    }
    rows: list[dict[str, object]] = []
    for cold_spec in cold_audit.build_specs():
        key = (cold_spec.dataset, cold_spec.method, cold_spec.seed)
        warm_spec = warm_by_key[key]
        cold_result = first_existing(cold_spec.result_candidates)
        hot_result = first_existing(warm_spec.result_candidates)
        cold_path = first_existing(cold_spec.per_item_candidates)
        hot_path = first_existing(warm_spec.per_item_candidates)
        if cold_result is None or hot_result is None:
            raise ValueError(f"missing core result source for {key}")
        cold, cold_count, cold_source = load_group_metrics(
            cold_result, cold_path, "cold"
        )
        hot, hot_count, hot_source = load_group_metrics(
            hot_result, hot_path, "hot"
        )
        rows.append(
            build_seed_row(
                dataset=cold_spec.dataset,
                method=cold_spec.method,
                seed=cold_spec.seed,
                cold=cold,
                hot=hot,
                cold_count=cold_count,
                hot_count=hot_count,
                cold_source=cold_source,
                hot_source=hot_source,
                direct=None,
            )
        )
    return rows


def collect_semco_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in warm_export.semco_specs():
        result_path = first_existing(spec.result_candidates)
        hot_path = first_existing(spec.per_item_candidates)
        if result_path is None:
            raise ValueError(
                f"missing SEMCo result source for {spec.dataset}/{spec.seed}"
            )
        cold_path = None
        if hot_path is not None:
            candidate = Path(
                str(hot_path).replace(
                    "per_item_full_hot_", "per_item_full_cold_"
                )
            )
            cold_path = candidate if candidate.exists() else None
        cold, cold_count, cold_source = load_group_metrics(
            result_path, cold_path, "cold"
        )
        hot, hot_count, hot_source = load_group_metrics(
            result_path, hot_path, "hot"
        )
        rows.append(
            build_seed_row(
                dataset=spec.dataset,
                method="SEMCo",
                seed=spec.seed,
                cold=cold,
                hot=hot,
                cold_count=cold_count,
                hot_count=hot_count,
                cold_source=cold_source,
                hot_source=hot_source,
                direct=None,
            )
        )
    return rows


def collect_kgrec_rows() -> list[dict[str, object]]:
    summary_path = (
        PAPER
        / "baseline_sources/_kgrec_strict/_remaining_main_table_queue/main_table_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, object]] = []
    for dataset_block in summary["datasets"]:
        dataset = str(dataset_block["dataset"])
        for seed_block in dataset_block["seeds"]:
            seed = int(seed_block["seed"])
            report_path = Path(seed_block["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            test = report["test"]
            counts = test["counts"]
            row = build_seed_row(
                dataset=dataset,
                method="KGRec",
                seed=seed,
                cold={metric: float(test["full_cold_item_macro"][metric]) for metric in METRICS},
                hot={metric: float(test["full_hot_item_macro"][metric]) for metric in METRICS},
                cold_count=int(counts["cold_items"]),
                hot_count=int(counts["hot_items"]),
                cold_source=relative(report_path),
                hot_source=relative(report_path),
                direct={metric: float(test["full_all_item_macro"][metric]) for metric in METRICS},
            )
            row["aggregation_route"] = "direct_all_item_macro_validated"
            rows.append(row)
    return rows


def collect_pcgnn_rows() -> list[dict[str, object]]:
    root = PAPER / "baseline_sources/_pcgnn_strict"
    prefixes = {"MOOCCube": "mooccube", "Junyi": "junyi", "COCO": "coco"}
    rows: list[dict[str, object]] = []
    for dataset, prefix in prefixes.items():
        for seed in SEEDS:
            report_path = (
                root
                / f"{prefix}_seed{seed}_full_formal_kg_warm"
                / "pcgnn_strict_adapter_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            test = report.get("test", {})
            hot = test.get("full_hot_item_macro", {})
            hot_count = int(test.get("count_full_hot_item_macro", 0) or 0)
            if not hot or hot_count <= 0:
                row = unavailable_row(
                    dataset,
                    "PCGNN",
                    seed,
                    "retained strict report has no warm-target metrics",
                )
                row["cold_count"] = int(
                    test.get("count_full_cold_item_macro", 0) or 0
                )
                row["hot_count"] = hot_count
                row["cold_source"] = relative(report_path)
                row["hot_source"] = relative(report_path)
                rows.append(row)
                continue
            rows.append(
                build_seed_row(
                    dataset=dataset,
                    method="PCGNN",
                    seed=seed,
                    cold={metric: float(test["full_cold_item_macro"][metric]) for metric in METRICS},
                    hot={metric: float(hot[metric]) for metric in METRICS},
                    cold_count=int(test["count_full_cold_item_macro"]),
                    hot_count=hot_count,
                    cold_source=relative(report_path),
                    hot_source=relative(report_path),
                    direct=None,
                )
            )
    return rows


def collect_seed_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        *collect_core_rows(),
        *collect_semco_rows(),
        *collect_kgrec_rows(),
        *collect_pcgnn_rows(),
    ]
    coverage = pd.DataFrame(rows)
    duplicates = coverage.duplicated(["dataset", "method", "seed"], keep=False)
    if duplicates.any():
        duplicate_rows = coverage.loc[
            duplicates, ["dataset", "method", "seed"]
        ].to_dict("records")
        raise ValueError(f"duplicate dataset-method-seed rows: {duplicate_rows}")
    coverage["dataset"] = pd.Categorical(
        coverage["dataset"], categories=DATASETS, ordered=True
    )
    coverage["method"] = pd.Categorical(
        coverage["method"], categories=METHOD_ORDER, ordered=True
    )
    coverage = coverage.sort_values(["dataset", "method", "seed"]).reset_index(
        drop=True
    )
    detail = coverage.loc[coverage["status"] == "ready"].copy()
    return detail, coverage


def summarize_ready(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required_seeds = set(SEEDS)
    for (dataset, method), group in detail.groupby(
        ["dataset", "method"], observed=True, sort=False
    ):
        seeds = {int(seed) for seed in group["seed"]}
        if seeds != required_seeds:
            raise ValueError(
                f"{dataset}/{method} requires seeds 2025, 2026, 2027; "
                f"found {sorted(seeds)}"
            )
        row: dict[str, object] = {
            "dataset": str(dataset),
            "method": str(method),
            "seeds": "2025,2026,2027",
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)

    summary = pd.DataFrame(rows)
    for dataset in summary["dataset"].unique():
        mask = summary["dataset"] == dataset
        dataset_rows = summary.loc[mask]
        baseline_rows = dataset_rows[dataset_rows["method"] != "CKG-RL"]
        for metric in METRICS:
            summary.loc[mask, f"{metric}_rank"] = dataset_rows[metric].rank(
                method="min", ascending=False
            )
            best_index = baseline_rows[metric].idxmax()
            best_value = float(summary.loc[best_index, metric])
            best_method = str(summary.loc[best_index, "method"])
            summary.loc[mask, f"{metric}_strongest_baseline"] = best_value
            summary.loc[mask, f"{metric}_strongest_baseline_method"] = best_method
            summary.loc[mask, f"{metric}_relative_improvement"] = math.nan
            ours = mask & summary["method"].eq("CKG-RL")
            if best_value > 0 and ours.any():
                summary.loc[ours, f"{metric}_relative_improvement"] = (
                    summary.loc[ours, metric] - best_value
                ) / best_value

    for metric in METRICS:
        summary[f"{metric}_rank"] = summary[f"{metric}_rank"].astype(int)
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    method_order = {name: index for index, name in enumerate(METHOD_ORDER)}
    summary["_dataset_order"] = summary["dataset"].map(
        lambda value: dataset_order.get(str(value), len(dataset_order))
    )
    summary["_method_order"] = summary["method"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    return summary.sort_values(
        ["_dataset_order", "_method_order", "method"]
    ).drop(columns=["_dataset_order", "_method_order"]).reset_index(drop=True)


def build_wide(summary: pd.DataFrame) -> pd.DataFrame:
    main_metrics = ("R@5", "R@10", "N@5", "N@10")
    methods = [
        method
        for method in METHOD_ORDER
        if method != "PCGNN" and method in set(summary["method"])
    ]
    rows: list[dict[str, object]] = []
    for method in methods:
        row: dict[str, object] = {"method": method}
        for dataset in DATASETS:
            match = summary[
                summary["dataset"].eq(dataset) & summary["method"].eq(method)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"expected one summary row for {dataset}/{method}, found {len(match)}"
                )
            record = match.iloc[0]
            for metric in main_metrics:
                row[f"{dataset}_{metric}"] = float(record[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def plain_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include="category").columns:
        out[column] = out[column].astype(object)
    return out


def format_worksheet(worksheet) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10)
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = body_font
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for index, column in enumerate(worksheet.iter_cols(), start=1):
        width = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column
        )
        worksheet.column_dimensions[get_column_letter(index)].width = min(
            max(width + 2, 10), 48
        )


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_out = plain_frame(detail)
    summary_out = plain_frame(summary)
    coverage_out = plain_frame(coverage)
    wide = build_wide(summary_out)

    paths = {
        "workbook": output_dir / "overall_baseline_comparison.xlsx",
        "summary": output_dir / "overall_baseline_summary.csv",
        "detail": output_dir / "overall_baseline_seed_detail.csv",
        "wide": output_dir / "overall_baseline_wide.csv",
        "coverage": output_dir / "overall_baseline_coverage.csv",
    }
    summary_out.to_csv(paths["summary"], index=False, float_format="%.15g")
    detail_out.to_csv(paths["detail"], index=False, float_format="%.15g")
    wide.to_csv(paths["wide"], index=False, float_format="%.15g")
    coverage_out.to_csv(paths["coverage"], index=False, float_format="%.15g")

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        summary_out.to_excel(writer, sheet_name="Summary", index=False)
        detail_out.to_excel(writer, sheet_name="Seed_Detail", index=False)
        coverage_out.to_excel(writer, sheet_name="Coverage", index=False)
        for worksheet in writer.book.worksheets:
            format_worksheet(worksheet)
    return list(paths.values())


def main() -> None:
    detail, coverage = collect_seed_rows()
    summary = summarize_ready(detail)
    paths = write_outputs(detail, summary, coverage, OUTPUT_DIR)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
