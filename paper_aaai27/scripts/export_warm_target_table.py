from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import audit_significance_inputs as cold_audit


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_aaai27"
SEEDS = [2025, 2026, 2027]
DATASETS = ["MOOCCube", "Junyi", "COCO"]
METHOD_ORDER = [
    "Popularity",
    "BPR",
    "LightGCN",
    "DropoutNet",
    "CCFCRec",
    "ALDI",
    "SEMCo",
    "CGRC",
    "PCGNN (adapted)",
    "USIM",
    "CKG-RL",
]
METRICS = ["R@5", "R@10", "N@5", "N@10"]
ALL_METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
TOLERANCE = 5e-5


@dataclass(frozen=True)
class WarmSpec:
    dataset: str
    method: str
    seed: int
    result_candidates: tuple[Path, ...]
    per_item_candidates: tuple[Path, ...]
    note: str = ""


def semco_specs() -> list[WarmSpec]:
    roots = {
        "MOOCCube": ROOT / "outputs" / "content_delta_pop5" / "semco_official_v1",
        "Junyi": ROOT / "outputs" / "junyi" / "semco_official_v1",
        "COCO": ROOT / "outputs" / "coco" / "semco_official_v1",
    }
    specs: list[WarmSpec] = []
    for dataset, root in roots.items():
        for seed in SEEDS:
            run = root / f"strict_item_cold_balanced_thr1_seed_{seed}" / "sparsemax_scale12_e5_b512"
            specs.append(
                WarmSpec(
                    dataset=dataset,
                    method="SEMCo",
                    seed=seed,
                    result_candidates=(run / "semco_official_static_result.json",),
                    per_item_candidates=(run / "per_item_full_hot_semco_official_static.csv",),
                    note="Official SEMCo adaptation used for the main cold-start table.",
                )
            )
    return specs


def pcgnn_specs() -> list[WarmSpec]:
    name = {"MOOCCube": "mooccube", "Junyi": "junyi", "COCO": "coco"}
    root = PAPER / "baseline_sources" / "_pcgnn_strict"
    specs: list[WarmSpec] = []
    for dataset, prefix in name.items():
        for seed in SEEDS:
            report = root / f"{prefix}_seed{seed}_full_formal_kg_warm" / "pcgnn_strict_adapter_report.json"
            specs.append(
                WarmSpec(
                    dataset=dataset,
                    method="PCGNN (adapted)",
                    seed=seed,
                    result_candidates=(report,),
                    per_item_candidates=(),
                    note="The current PCGNN report evaluates only cold targets; no same-protocol warm export is available.",
                )
            )
    return specs


def core_specs() -> list[WarmSpec]:
    specs: list[WarmSpec] = []
    for spec in cold_audit.build_specs():
        specs.append(
            WarmSpec(
                dataset=spec.dataset,
                method=spec.method,
                seed=spec.seed,
                result_candidates=spec.result_candidates,
                per_item_candidates=tuple(
                    Path(str(path).replace("per_item_full_cold_", "per_item_full_hot_"))
                    for path in spec.per_item_candidates
                ),
                note=spec.note,
            )
        )
    return specs


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_json_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def metrics_from_json(path: Path | None) -> dict[str, float]:
    if path is None or path.suffix.lower() != ".json":
        return {}
    row = load_json_row(path)
    block = row.get("full_hot_item_macro")
    if not isinstance(block, dict):
        block = row.get("test", {}).get("full_hot_item_macro", {}) if isinstance(row.get("test"), dict) else {}
    if not isinstance(block, dict):
        return {}
    values: dict[str, float] = {}
    for metric in ALL_METRICS:
        value = block.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[metric] = float(value)
    return values


def metrics_from_per_item(path: Path | None) -> tuple[dict[str, float], int]:
    if path is None:
        return {}, 0
    frame = pd.read_csv(path)
    values: dict[str, float] = {}
    for metric in ALL_METRICS:
        if metric in frame.columns:
            value = pd.to_numeric(frame[metric], errors="coerce").mean()
            if pd.notna(value):
                values[metric] = float(value)
    return values, int(len(frame))


def relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit_specs(specs: list[WarmSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        result_path = first_existing(spec.result_candidates)
        per_item_path = first_existing(spec.per_item_candidates)
        result = metrics_from_json(result_path)
        per_item, per_item_count = metrics_from_per_item(per_item_path)
        selected = per_item if all(metric in per_item for metric in METRICS) else result
        selected_source = "per_item" if selected is per_item else "result_json"
        differences = {
            metric: abs(result[metric] - per_item[metric])
            for metric in METRICS
            if metric in result and metric in per_item
        }
        if all(metric in selected for metric in METRICS):
            status = "ready"
            if differences and any(value > TOLERANCE for value in differences.values()):
                status = "metric_mismatch"
        elif result_path is None and per_item_path is None:
            status = "missing_result_and_per_item"
        elif result_path is None:
            status = "missing_result"
        elif per_item_path is None:
            status = "missing_warm_per_item"
        else:
            status = "missing_warm_metrics"
        row: dict[str, object] = {
            "dataset": spec.dataset,
            "method": spec.method,
            "seed": spec.seed,
            "status": status,
            "selected_source": selected_source if status in {"ready", "metric_mismatch"} else "",
            "result_source": relative(result_path),
            "per_item_source": relative(per_item_path),
            "per_item_count": per_item_count,
            "note": spec.note,
        }
        for metric in ALL_METRICS:
            row[metric] = selected.get(metric, math.nan)
            row[f"json_{metric}"] = result.get(metric, math.nan)
            row[f"per_item_{metric}"] = per_item.get(metric, math.nan)
            row[f"absdiff_{metric}"] = differences.get(metric, math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            group = frame[(frame["dataset"] == dataset) & (frame["method"] == method)].sort_values("seed")
            row: dict[str, object] = {"dataset": dataset, "method": method}
            complete = len(group) == len(SEEDS) and group["status"].eq("ready").all()
            row["status"] = "ready" if complete else "incomplete"
            row["missing_seeds"] = ",".join(str(seed) for seed in SEEDS if seed not in set(group.loc[group["status"].eq("ready"), "seed"]))
            for metric in METRICS:
                values = group[metric].dropna().astype(float).tolist()
                row[metric] = float(np.mean(values)) if complete else math.nan
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def latex_value(value: float, best: bool, second: bool) -> str:
    if not math.isfinite(value):
        return "--"
    text = f"{value:.4f}"
    if best:
        return f"\\textbf{{{text}}}"
    if second:
        return f"\\underline{{{text}}}"
    return text


def build_latex_table(summary: pd.DataFrame) -> str:
    ranks: dict[tuple[str, str], tuple[float, float]] = {}
    for dataset in DATASETS:
        ready = summary[(summary["dataset"] == dataset) & (summary["status"] == "ready")]
        for metric in METRICS:
            values = sorted({float(value) for value in ready[metric].dropna()}, reverse=True)
            first = values[0] if values else math.nan
            second = values[1] if len(values) > 1 else math.nan
            ranks[(dataset, metric)] = (first, second)

    rows: list[str] = []
    for method in METHOD_ORDER:
        cells = [method.replace("_", "\\_")]
        for dataset in DATASETS:
            record = summary[(summary["dataset"] == dataset) & (summary["method"] == method)].iloc[0]
            for metric in METRICS:
                value = float(record[metric]) if pd.notna(record[metric]) else math.nan
                best, second = ranks[(dataset, metric)]
                cells.append(
                    latex_value(
                        value,
                        math.isfinite(value) and abs(value - best) <= 1e-12,
                        math.isfinite(value) and abs(value - second) <= 1e-12,
                    )
                )
        line = " & ".join(cells) + r" \\"
        rows.append(line)

    return "\n".join(
        [
            r"\documentclass[letterpaper]{article}",
            r"\usepackage[submission]{aaai2027}",
            r"\usepackage{booktabs}",
            r"\frenchspacing",
            r"\pdfinfo{/TemplateVersion (2027.1)}",
            r"\setcounter{secnumdepth}{0}",
            r"\begin{document}",
            r"\thispagestyle{empty}",
            r"\makeatletter",
            r"\setlength{\@dblfptop}{0pt}",
            r"\setlength{\@dblfpbot}{0pt plus 1fil}",
            r"\makeatother",
            r"\begin{table*}[!t]",
            r"\centering",
            r"{\small",
            r"\setlength{\tabcolsep}{1.35pt}",
            r"\renewcommand{\arraystretch}{0.98}",
            r"\begin{tabular}{lcccccccccccc}",
            r"\toprule",
            r"& \multicolumn{4}{c}{MOOCCube} & \multicolumn{4}{c}{Junyi} & \multicolumn{4}{c}{COCO} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
            r"Method & R@5 & R@10 & N@5 & N@10 & R@5 & R@10 & N@5 & N@10 & R@5 & R@10 & N@5 & N@10 \\",
            r"\midrule",
            *rows[:-1],
            r"\midrule",
            rows[-1],
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Warm-target retention under the strict course-cold split. Test targets have at least one training interaction; candidates remain the full catalog after train-history masking. Values are three-seed course-macro means. Bold and underlines mark the best and second-best available means. ``--'' indicates that no same-protocol warm-target export was retained. CKG-RL checkpoints are selected on cold validation NDCG@10, not these diagnostics.}",
            r"\label{tab:warm-target-retention}",
            r"\end{table*}",
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    specs = [*core_specs(), *semco_specs(), *pcgnn_specs()]
    audit = audit_specs(specs)
    summary = summarize(audit)
    audit_path = PAPER / "warm_target_table_coverage.csv"
    values_path = PAPER / "warm_target_table_values.csv"
    tex_path = PAPER / "warm_target_table.tex"
    audit.to_csv(audit_path, index=False)
    summary.to_csv(values_path, index=False)
    tex_path.write_text(build_latex_table(summary), encoding="ascii")
    ready = int((audit["status"] == "ready").sum())
    print(f"Wrote {audit_path}")
    print(f"Wrote {values_path}")
    print(f"Wrote {tex_path}")
    print(f"Warm coverage: {ready}/{len(audit)} seed-method rows ready")
    print(audit.groupby(["dataset", "status"]).size().reset_index(name="count").to_string(index=False))


if __name__ == "__main__":
    main()
