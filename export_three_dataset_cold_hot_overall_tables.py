from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "paper_aaai27" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_significance_inputs as cold_audit
import export_warm_target_table as warm_export


OUT_DIR = ROOT / "outputs" / "three_dataset_cold_hot_overall_mainformat_export"
PDF_OUT_DIR = ROOT / "output" / "pdf"
DATASETS = ["MOOCCube", "Junyi", "COCO"]
SEEDS = [2025, 2026, 2027]
METHOD_ORDER = [
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
    "PAM",
    "SEMCo",
    "CKG-RL",
]
METRICS = ["R@5", "R@10", "N@5", "N@10"]
ALL_METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
METRIC_TO_COL = {
    "R@5": "R5",
    "R@10": "R10",
    "R@20": "R20",
    "N@5": "N5",
    "N@10": "N10",
    "N@20": "N20",
}
DATASET_TO_PAM_DIR = {
    "MOOCCube": "mooccube",
    "Junyi": "junyi",
    "COCO": "coco",
}
DATASET_TO_PCGNN_PREFIX = {
    "MOOCCube": "mooccube",
    "Junyi": "junyi",
    "COCO": "coco",
}
SEMCO_SUMMARY = {
    "MOOCCube": ROOT
    / "outputs"
    / "content_delta_pop5"
    / "semco_official_v1"
    / "summary_sparsemax_scale12_e5_b512"
    / "main_table_item_macro_detail.csv",
    "Junyi": ROOT
    / "outputs"
    / "junyi"
    / "semco_official_v1"
    / "summary_sparsemax_scale12_e5_b512"
    / "main_table_item_macro_detail.csv",
    "COCO": ROOT
    / "outputs"
    / "coco"
    / "semco_official_v1"
    / "summary_sparsemax_scale12_e5_b512"
    / "main_table_item_macro_detail.csv",
}
PAM_SUMMARY = {
    dataset: ROOT
    / "outputs"
    / "pam_validation_select_mooccube_seed2025_e3_gpu_20260720"
    / "aggregate"
    / subdir
    / "main_table_item_macro_detail.csv"
    for dataset, subdir in DATASET_TO_PAM_DIR.items()
}
KGREC_SUMMARY_JSON = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_kgrec_strict"
    / "_remaining_main_table_queue"
    / "main_table_summary.json"
)
MOOCCUBE_CGRC_MAIN_TABLE_DIR = "main_table_balanced_itemmacro_cgrc_paper_v1"


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_result_payload(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict):
            return payload
        return {}
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return frame.iloc[0].to_dict() if not frame.empty else {}
    raise ValueError(f"unsupported result source: {path}")


def per_item_metrics(path: Path) -> tuple[dict[str, float], int]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty per-item file: {path}")
    missing = [metric for metric in ALL_METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"missing per-item metrics {missing}: {path}")
    return {
        metric: float(pd.to_numeric(frame[metric], errors="raise").mean())
        for metric in ALL_METRICS
    }, int(len(frame))


def result_group_metrics(path: Path, group: str) -> tuple[dict[str, float], int]:
    payload = load_result_payload(path)
    block_name = f"full_{group}_item_macro"
    block = payload.get(block_name, {})
    if isinstance(block, dict) and all(metric in block for metric in ALL_METRICS):
        count = int(payload.get(f"count_full_{group}_item_macro", 0) or 0)
        return {metric: float(block[metric]) for metric in ALL_METRICS}, count

    values: dict[str, float] = {}
    for metric, suffix in METRIC_TO_COL.items():
        column = f"full_{group}_item_macro_{suffix.lower()}"
        if column not in payload:
            raise ValueError(f"missing {block_name} metric {metric}: {path}")
        values[metric] = float(payload[column])
    count = int(payload.get(f"full_{group}_item_macro_count", 0) or 0)
    return values, count


def load_group_metrics(
    result_path: Path,
    per_item_path: Path | None,
    group: str,
) -> tuple[dict[str, float], int]:
    if per_item_path is not None and per_item_path.exists():
        return per_item_metrics(per_item_path)
    metrics, count = result_group_metrics(result_path, group)
    if count <= 0:
        raise ValueError(f"missing {group} item count: {result_path}")
    return metrics, count


def make_record(
    dataset: str,
    method: str,
    seed: int,
    cold: dict[str, float],
    hot: dict[str, float] | None,
    cold_count: int,
    hot_count: int,
    note: str = "",
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "count_cold": float(cold_count),
        "count_hot": float(hot_count),
        "note": note,
    }
    for metric in ALL_METRICS:
        record[f"cold_{metric}"] = float(cold[metric])
        if hot is None:
            record[f"hot_{metric}"] = math.nan
            record[f"overall_{metric}"] = float(cold[metric])
        else:
            record[f"hot_{metric}"] = float(hot[metric])
            denom = cold_count + hot_count
            record[f"overall_{metric}"] = (
                float(cold[metric]) * cold_count + float(hot[metric]) * hot_count
            ) / denom
    return record


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
        cold_item = first_existing(cold_spec.per_item_candidates)
        hot_item = first_existing(warm_spec.per_item_candidates)
        if cold_spec.dataset == "MOOCCube" and cold_spec.method == "CGRC":
            main_table_result = (
                ROOT
                / "outputs"
                / "content_delta_pop5"
                / "static_item_cold_balanced"
                / f"strict_item_cold_balanced_thr1_seed_{cold_spec.seed}"
                / MOOCCUBE_CGRC_MAIN_TABLE_DIR
                / "cgrc_paper_static_result.json"
            )
            cold_result = main_table_result
            hot_result = main_table_result
            cold_item = None
            hot_item = None
        if cold_result is None or hot_result is None:
            raise FileNotFoundError(f"missing core source for {key}")
        cold, cold_count = load_group_metrics(cold_result, cold_item, "cold")
        hot, hot_count = load_group_metrics(hot_result, hot_item, "hot")
        rows.append(
            make_record(
                cold_spec.dataset,
                cold_spec.method,
                cold_spec.seed,
                cold,
                hot,
                cold_count,
                hot_count,
            )
        )
    return rows


def collect_kgrec_rows() -> list[dict[str, object]]:
    summary = json.loads(KGREC_SUMMARY_JSON.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, object]] = []
    for dataset_block in summary["datasets"]:
        dataset = str(dataset_block["dataset"])
        for seed_block in dataset_block["seeds"]:
            seed = int(seed_block["seed"])
            report_path = Path(seed_block["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            test = report["test"]
            counts = test["counts"]
            rows.append(
                make_record(
                    dataset,
                    "KGRec",
                    seed,
                    {metric: float(test["full_cold_item_macro"][metric]) for metric in ALL_METRICS},
                    {metric: float(test["full_hot_item_macro"][metric]) for metric in ALL_METRICS},
                    int(counts["cold_items"]),
                    int(counts["hot_items"]),
                )
            )
    return rows


def collect_pcgnn_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, prefix in DATASET_TO_PCGNN_PREFIX.items():
        for seed in SEEDS:
            report_path = (
                ROOT
                / "paper_aaai27"
                / "baseline_sources"
                / "_pcgnn_strict"
                / f"{prefix}_seed{seed}_full_formal_kg_warm"
                / "pcgnn_strict_adapter_report.json"
            )
            replay_path = report_path.with_name("pcgnn_hot_replay_report.json")
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            test = report["test"]
            replay = json.loads(replay_path.read_text(encoding="utf-8-sig"))
            replay_test = replay["test"]
            hot_count = int(replay_test.get("count_full_hot_item_macro", 0) or 0)
            hot_block = replay_test.get("full_hot_item_macro", {})
            if hot_count <= 0 or not hot_block:
                raise ValueError(f"missing PCGNN hot replay metrics: {replay_path}")
            rows.append(
                make_record(
                    dataset,
                    "PCGNN",
                    seed,
                    {metric: float(test["full_cold_item_macro"][metric]) for metric in ALL_METRICS},
                    {metric: float(hot_block[metric]) for metric in ALL_METRICS},
                    int(test["count_full_cold_item_macro"]),
                    hot_count,
                    "cold uses retained formal report; hot uses retained-checkpoint replay supplement",
                )
            )
    return rows


def collect_csv_detail_rows(
    dataset: str,
    method: str,
    path: Path,
) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        if str(row["model"]) != method and not (method == "SEMCo" and str(row["model"]).startswith("SEMCo")):
            continue
        cold = {
            metric: float(row[f"cold_{METRIC_TO_COL[metric]}"])
            for metric in ALL_METRICS
        }
        hot = {
            metric: float(row[f"hot_{METRIC_TO_COL[metric]}"])
            for metric in ALL_METRICS
        }
        rows.append(
            make_record(
                dataset,
                method,
                int(row["seed"]),
                cold,
                hot,
                int(row["count_cold"]),
                int(row["count_hot"]),
            )
        )
    if len(rows) != len(SEEDS):
        raise ValueError(f"{dataset}/{method} rows mismatch in {path}: {len(rows)}")
    return rows


def collect_pam_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, path in PAM_SUMMARY.items():
        rows.extend(collect_csv_detail_rows(dataset, "PAM", path))
    return rows


def collect_semco_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, path in SEMCO_SUMMARY.items():
        rows.extend(collect_csv_detail_rows(dataset, "SEMCo", path))
    return rows


def collect_seed_detail() -> pd.DataFrame:
    rows = [
        *collect_core_rows(),
        *collect_kgrec_rows(),
        *collect_pcgnn_rows(),
        *collect_pam_rows(),
        *collect_semco_rows(),
    ]
    detail = pd.DataFrame(rows)
    duplicate = detail.duplicated(["dataset", "method", "seed"], keep=False)
    if duplicate.any():
        dup_rows = detail.loc[duplicate, ["dataset", "method", "seed"]].to_dict("records")
        raise ValueError(f"duplicate rows: {dup_rows}")
    expected = set(SEEDS)
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            seeds = set(int(seed) for seed in detail.loc[
                detail["dataset"].eq(dataset) & detail["method"].eq(method), "seed"
            ])
            if seeds != expected:
                raise ValueError(f"{dataset}/{method} seeds mismatch: {sorted(seeds)}")
    return detail


def split_summary(detail: pd.DataFrame, split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    formatted_rows: list[dict[str, str]] = []
    numeric_rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        formatted = {"Method": method}
        numeric = {"Method": method}
        for dataset in DATASETS:
            rows = detail[detail["dataset"].eq(dataset) & detail["method"].eq(method)].sort_values("seed")
            for metric in METRICS:
                values = pd.to_numeric(rows[f"{split}_{metric}"], errors="raise").dropna()
                prefix = f"{dataset}_{metric}"
                if values.empty:
                    formatted[prefix] = "--"
                    numeric[f"{prefix}_mean"] = ""
                    numeric[f"{prefix}_std"] = ""
                    continue
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                formatted[prefix] = f"{mean:.4f} +/- {std:.4f}"
                numeric[f"{prefix}_mean"] = f"{mean:.10f}"
                numeric[f"{prefix}_std"] = f"{std:.10f}"
        formatted_rows.append(formatted)
        numeric_rows.append(numeric)
    return formatted_rows, numeric_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_cells(numeric_rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float | None]]:
    best: dict[str, float] = {}
    second: dict[str, float | None] = {}
    for dataset in DATASETS:
        for metric in METRICS:
            key = f"{dataset}_{metric}"
            values = []
            for row in numeric_rows:
                raw = row[f"{key}_mean"]
                if raw:
                    values.append(float(raw))
            ranked = sorted(set(values), reverse=True)
            best[key] = ranked[0] if ranked else math.nan
            second[key] = ranked[1] if len(ranked) > 1 else None
    return best, second


def latex_cell(mean: str, std: str, is_best: bool, is_second: bool) -> str:
    if not mean:
        return r"\texttt{--}"
    value = f"{float(mean):.4f}$\\pm${float(std):.4f}"
    if is_best:
        return rf"\textbf{{{value}}}"
    if is_second:
        return rf"\underline{{{value}}}"
    return value


def table_body(split: str, numeric_rows: list[dict[str, str]]) -> str:
    best, second = rank_cells(numeric_rows)
    split_title = {"cold": "Cold", "hot": "Hot", "overall": "Overall"}[split]
    notes = {
        "cold": (
            "The cold means match the AAAI main table to four decimals; standard deviations "
            "are computed from the retained seed-level exports."
        ),
        "hot": (
            "PCGNN hot values are supplemental retained-checkpoint replay results because the "
            "original formal PCGNN reports evaluated only cold targets."
        ),
        "overall": (
            "Overall values are reconstructed per seed by weighting cold and hot item-macro "
            "metrics by evaluated target-course counts. For PCGNN, cold values use the retained "
            "formal reports and hot values use the retained-checkpoint replay supplement."
        ),
    }
    caption = (
        f"{split_title} item-macro full-ranking results in the AAAI main-table layout. "
        f"Values are mean$\\pm$std over three seeds. Best and second-best available "
        f"results are bolded and underlined. {notes[split]}"
    )
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.35pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{MOOCCube} & \multicolumn{4}{c}{Junyi} & \multicolumn{4}{c}{COCO} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        r"Method & R@5 & R@10 & N@5 & N@10 & R@5 & R@10 & N@5 & N@10 & R@5 & R@10 & N@5 & N@10 \\",
        r"\midrule",
    ]
    for row in numeric_rows:
        method_label = r"\textbf{CKG-RL}" if row["Method"] == "CKG-RL" else row["Method"]
        cells = [method_label]
        for dataset in DATASETS:
            for metric in METRICS:
                key = f"{dataset}_{metric}"
                mean = row[f"{key}_mean"]
                std = row[f"{key}_std"]
                if not mean:
                    cells.append(r"\texttt{--}")
                    continue
                value = float(mean)
                is_best = math.isfinite(best[key]) and abs(value - best[key]) < 5e-12
                is_second = second[key] is not None and abs(value - second[key]) < 5e-12
                cells.append(latex_cell(mean, std, is_best, is_second))
        if row["Method"] == "CKG-RL":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:three-dataset-{split}-item-macro-mainformat}}",
    ])
    return "\n".join(lines)


def standalone_doc(title: str, bodies: list[str]) -> str:
    parts: list[str] = []
    for index, body in enumerate(bodies):
        if index:
            parts.append(r"\clearpage")
        parts.extend([r"\begin{table}[t]", r"\centering", body, r"\end{table}"])
    joined = "\n".join(parts)
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.45in,landscape]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{amsmath}}
\makeatletter
\setlength{{\@fptop}}{{0pt}}
\makeatother
\pagestyle{{empty}}
\title{{{title}}}
\date{{}}
\begin{{document}}
{joined}
\end{{document}}
"""


def run_pdflatex(tex_path: Path) -> Path:
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail = collect_seed_detail()
    detail_path = OUT_DIR / "three_dataset_cold_hot_overall_seed_detail.csv"
    detail.to_csv(detail_path, index=False)

    bodies: list[str] = []
    for split in ["cold", "hot", "overall"]:
        formatted, numeric = split_summary(detail, split)
        formatted_path = OUT_DIR / f"three_dataset_{split}_item_macro_mainformat_table.csv"
        numeric_path = OUT_DIR / f"three_dataset_{split}_item_macro_mainformat_table_numeric.csv"
        tex_path = OUT_DIR / f"three_dataset_{split}_item_macro_mainformat_table.tex"
        standalone_tex = OUT_DIR / f"three_dataset_{split}_item_macro_mainformat_table_standalone.tex"
        body = table_body(split, numeric)
        write_csv(formatted_path, formatted)
        write_csv(numeric_path, numeric)
        tex_path.write_text("\n".join([r"\begin{table*}[t]", r"\centering", body, r"\end{table*}", ""]), encoding="utf-8")
        standalone_tex.write_text(standalone_doc(f"Three-dataset {split} item-macro table", [body]), encoding="utf-8")
        stable_pdf = PDF_OUT_DIR / f"three_dataset_{split}_item_macro_mainformat_table.pdf"
        shutil.copy2(run_pdflatex(standalone_tex), stable_pdf)
        bodies.append(body)
        print(f"Wrote {formatted_path}")
        print(f"Wrote {numeric_path}")
        print(f"Wrote {tex_path}")
        print(f"Wrote {stable_pdf}")

    combined_tex = OUT_DIR / "three_dataset_cold_hot_overall_item_macro_mainformat_tables_standalone.tex"
    combined_tex.write_text(
        standalone_doc("Three-dataset cold hot overall item-macro tables", bodies),
        encoding="utf-8",
    )
    stable_combined_pdf = PDF_OUT_DIR / "three_dataset_cold_hot_overall_item_macro_mainformat_tables.pdf"
    shutil.copy2(run_pdflatex(combined_tex), stable_combined_pdf)
    print(f"Wrote {detail_path}")
    print(f"Wrote {stable_combined_pdf}")


if __name__ == "__main__":
    main()
