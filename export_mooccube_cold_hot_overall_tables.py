from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

BASE_DETAIL = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "main_table_item_macro_final_audit"
    / "main_table_item_macro_detail.csv"
)
DROPOUTNET_DETAIL = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "main_table_balanced_itemmacro_dropoutnet_official_teacher80_student120_v1"
    / "main_table_item_macro_detail.csv"
)
CGRC_DETAIL = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "main_table_balanced_itemmacro_cgrc_paper_v1"
    / "main_table_item_macro_detail.csv"
)
USIM_DETAIL = (
    ROOT
    / "outputs"
    / "usim_official_3datasets_3seed"
    / "aggregate"
    / "mooccube"
    / "main_table_item_macro_detail.csv"
)
CKG_OLD2667_DETAIL = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "course_ablation_e60_3seed"
    / "full"
    / "fast3_static_runs_detail.csv"
)
ALDI_CORRECTED_DETAILS = {
    2025: ROOT
    / "outputs"
    / "score_parity"
    / "mooccube_seed2025"
    / "aldi_official_wsl_gpu_rerun_gate_20260717_192714"
    / "aldi_official_static_result.json",
    2026: ROOT
    / "outputs"
    / "score_parity"
    / "mooccube_seed2026"
    / "aldi_official_wsl_gpu_rerun_gate_20260717_203453"
    / "aldi_official_static_result.json",
    2027: ROOT
    / "outputs"
    / "score_parity"
    / "mooccube_seed2027"
    / "aldi_official_wsl_gpu_rerun_gate_20260717_205646"
    / "aldi_official_static_result.json",
}
KGREC_DETAILS = {
    seed: ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_kgrec_strict"
    / f"mooccube_seed{seed}_retuned_lr1e-5"
    / "kgrec_strict_adapter_report.json"
    for seed in [2025, 2026, 2027]
}
PCGNN_DETAILS = {
    seed: ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_pcgnn_strict"
    / f"mooccube_seed{seed}_full_formal_kg_warm"
    / "pcgnn_strict_adapter_report.json"
    for seed in [2025, 2026, 2027]
}
PAM_DETAIL = (
    ROOT
    / "outputs"
    / "pam_validation_select_mooccube_seed2025_e3_gpu_20260720"
    / "aggregate"
    / "mooccube"
    / "main_table_item_macro_detail.csv"
)
SEMCO_DETAIL = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "semco_official_v1"
    / "summary_sparsemax_scale12_e5_b512"
    / "main_table_item_macro_detail.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "course_ablation_e60_3seed"
    / "full"
    / "cold_hot_overall_table_export"
)
PDF_OUT_DIR = ROOT / "output" / "pdf"

METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
METRIC_TO_COL = {
    "R@5": "R5",
    "R@10": "R10",
    "R@20": "R20",
    "N@5": "N5",
    "N@10": "N10",
    "N@20": "N20",
}
METHOD_SPECS = [
    ("Popularity", "base", "Popularity"),
    ("BPR", "base", "BPR"),
    ("DropoutNet", "dropoutnet", "DropoutNet"),
    ("LightGCN", "base", "LightGCN"),
    ("CCFCRec", "base", "CCFCRec"),
    ("ALDI", "aldi_corrected", "ALDI-official-source"),
    ("KGRec", "kgrec", "KGRec"),
    ("CGRC", "cgrc", "CGRC-paper"),
    ("PCGNN", "pcgnn_cold_only", "PCGNN"),
    ("USIM", "usim", "USIM"),
    ("PAM", "pam", "PAM"),
    ("SEMCo", "semco", "SEMCo (official-code adapted)"),
    ("CKG-RL", "old2667", "USIM-Feedback-FAST3-ContentDelta"),
]


def require_files() -> None:
    for path in [
        BASE_DETAIL,
        DROPOUTNET_DETAIL,
        CGRC_DETAIL,
        USIM_DETAIL,
        CKG_OLD2667_DETAIL,
        PAM_DETAIL,
        SEMCO_DETAIL,
        *ALDI_CORRECTED_DETAILS.values(),
        *KGREC_DETAILS.values(),
        *PCGNN_DETAILS.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)


def read_standard_detail(path: Path, display: str, source_model: str) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    rows = df[df["model"] == source_model].copy()
    if rows.empty:
        raise ValueError(f"Missing model={source_model!r} in {path}")
    out: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        record: dict[str, object] = {
            "Method": display,
            "seed": int(row["seed"]),
            "count_cold": float(row["count_cold"]),
            "count_hot": float(row["count_hot"]),
        }
        for split in ["cold", "hot"]:
            for metric, suffix in METRIC_TO_COL.items():
                record[f"{split}_{metric}"] = float(row[f"{split}_{suffix}"])
        out.append(record)
    return out


def read_old2667_detail(display: str, source_model: str) -> list[dict[str, object]]:
    df = pd.read_csv(CKG_OLD2667_DETAIL)
    rows = df[df["model"] == source_model].copy()
    if rows.empty:
        raise ValueError(f"Missing model={source_model!r} in {CKG_OLD2667_DETAIL}")
    out: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        record: dict[str, object] = {
            "Method": display,
            "seed": int(row["seed"]),
            "count_cold": float(row["full_cold_item_macro_count"]),
            "count_hot": float(row["full_hot_item_macro_count"]),
        }
        for split in ["cold", "hot"]:
            src_prefix = f"full_{split}_item_macro"
            for metric, suffix in METRIC_TO_COL.items():
                record[f"{split}_{metric}"] = float(row[f"{src_prefix}_{suffix.lower()}"])
        out.append(record)
    return out


def read_corrected_aldi_detail(display: str, source_model: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for expected_seed, path in sorted(ALDI_CORRECTED_DETAILS.items()):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"Empty ALDI result: {path}")
            row = payload[0]
        elif isinstance(payload, dict):
            row = payload
        else:
            raise ValueError(f"Unsupported ALDI result payload: {path}")

        if row.get("model") != source_model:
            raise ValueError(f"Unexpected ALDI model in {path}: {row.get('model')!r}")
        seed = int(row.get("static_seed", expected_seed))
        if seed != expected_seed:
            raise ValueError(f"ALDI seed mismatch in {path}: got {seed}, expected {expected_seed}")
        record: dict[str, object] = {
            "Method": display,
            "seed": seed,
            "count_cold": float(row["count_full_cold_item_macro"]),
            "count_hot": float(row["count_full_hot_item_macro"]),
        }
        for split in ["cold", "hot"]:
            block = row[f"full_{split}_item_macro"]
            for metric in METRICS:
                record[f"{split}_{metric}"] = float(block[metric])
        out.append(record)
    return out


def read_kgrec_detail(display: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for expected_seed, path in sorted(KGREC_DETAILS.items()):
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        seed = int(row.get("seed", expected_seed))
        if seed != expected_seed:
            raise ValueError(f"KGRec seed mismatch in {path}: got {seed}, expected {expected_seed}")
        test = row["test"]
        counts = test["counts"]
        record: dict[str, object] = {
            "Method": display,
            "seed": seed,
            "count_cold": float(counts["cold_items"]),
            "count_hot": float(counts["hot_items"]),
        }
        for split in ["cold", "hot"]:
            block = test[f"full_{split}_item_macro"]
            for metric in METRICS:
                record[f"{split}_{metric}"] = float(block[metric])
        out.append(record)
    return out


def read_pcgnn_cold_only_detail(display: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for expected_seed, path in sorted(PCGNN_DETAILS.items()):
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        seed = expected_seed
        test = row["test"]
        record: dict[str, object] = {
            "Method": display,
            "seed": seed,
            "count_cold": float(test["count_full_cold_item_macro"]),
            "count_hot": 0.0,
        }
        for metric in METRICS:
            record[f"cold_{metric}"] = float(test["full_cold_item_macro"][metric])
            record[f"hot_{metric}"] = math.nan
        out.append(record)
    return out


def load_seed_detail() -> pd.DataFrame:
    source_paths = {
        "base": BASE_DETAIL,
        "dropoutnet": DROPOUTNET_DETAIL,
        "cgrc": CGRC_DETAIL,
        "usim": USIM_DETAIL,
        "pam": PAM_DETAIL,
        "semco": SEMCO_DETAIL,
    }
    all_rows: list[dict[str, object]] = []
    for display, source_name, source_model in METHOD_SPECS:
        if source_name == "old2667":
            all_rows.extend(read_old2667_detail(display, source_model))
        elif source_name == "aldi_corrected":
            all_rows.extend(read_corrected_aldi_detail(display, source_model))
        elif source_name == "kgrec":
            all_rows.extend(read_kgrec_detail(display))
        elif source_name == "pcgnn_cold_only":
            all_rows.extend(read_pcgnn_cold_only_detail(display))
        else:
            all_rows.extend(read_standard_detail(source_paths[source_name], display, source_model))

    detail = pd.DataFrame(all_rows)
    expected = {2025, 2026, 2027}
    for method in [spec[0] for spec in METHOD_SPECS]:
        seeds = set(int(seed) for seed in detail.loc[detail["Method"] == method, "seed"])
        if seeds != expected:
            raise ValueError(f"{method} seeds mismatch: got {sorted(seeds)}, expected {sorted(expected)}")
    return detail


def add_overall(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    denom = out["count_cold"] + out["count_hot"]
    for metric in METRICS:
        out[f"overall_{metric}"] = (
            out[f"cold_{metric}"] * out["count_cold"] + out[f"hot_{metric}"] * out["count_hot"]
        ) / denom
    cold_only = out["count_hot"].eq(0)
    for metric in METRICS:
        out.loc[cold_only, f"overall_{metric}"] = math.nan
    return out


def aggregate_split(detail: pd.DataFrame, split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    formatted_rows: list[dict[str, str]] = []
    numeric_rows: list[dict[str, str]] = []
    for method, _, _ in METHOD_SPECS:
        rows = detail[detail["Method"] == method].sort_values("seed")
        numeric: dict[str, str] = {
            "Method": method,
            "runs": str(len(rows)),
            "seeds": ",".join(str(int(seed)) for seed in rows["seed"]),
        }
        formatted: dict[str, str] = {"Method": method}
        for metric in METRICS:
            values = pd.to_numeric(rows[f"{split}_{metric}"], errors="raise")
            valid = values.dropna()
            if valid.empty:
                numeric[f"{metric}_mean"] = ""
                numeric[f"{metric}_std"] = ""
                formatted[metric] = "--"
                continue
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            numeric[f"{metric}_mean"] = f"{mean:.10f}"
            numeric[f"{metric}_std"] = f"{std:.10f}"
            formatted[metric] = f"{mean:.4f} +/- {std:.4f}"
        numeric_rows.append(numeric)
        formatted_rows.append(formatted)
    return formatted_rows, numeric_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_and_second(numeric_rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float | None]]:
    best: dict[str, float] = {}
    second: dict[str, float | None] = {}
    for metric in METRICS:
        values = []
        for row in numeric_rows:
            raw = row[f"{metric}_mean"]
            if raw == "":
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
        ranked = sorted(set(values), reverse=True)
        best[metric] = ranked[0] if ranked else math.nan
        second[metric] = ranked[1] if len(ranked) > 1 else None
    return best, second


def latex_cell(mean: str, std: str, is_best: bool, is_second: bool) -> str:
    if mean == "":
        return r"\texttt{--}"
    value = f"{float(mean):.4f}$\\pm${float(std):.4f}"
    if is_best:
        return rf"\textbf{{{value}}}"
    if is_second:
        return rf"\underline{{{value}}}"
    return value


def table_body(split: str, formatted_rows: list[dict[str, str]], numeric_rows: list[dict[str, str]]) -> str:
    best, second = best_and_second(numeric_rows)
    split_title = {"cold": "Cold", "hot": "Hot", "overall": "Overall"}[split]
    lines = [
        rf"\caption{{MOOCCube {split_title.lower()} item-macro full-ranking results. Values are mean$\pm$std over three seeds. For overall, each seed is count-weighted over cold and hot item-macro partitions before aggregation. Best results are bolded and second-best results are underlined. Unavailable same-protocol warm metrics are shown as --.}}",
        rf"\label{{tab:mooccube-{split}-item-macro}}",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Recall \(\uparrow\)} & \multicolumn{3}{c}{NDCG \(\uparrow\)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & \multicolumn{1}{c}{@5} & \multicolumn{1}{c}{@10} & \multicolumn{1}{c}{@20} & \multicolumn{1}{c}{@5} & \multicolumn{1}{c}{@10} & \multicolumn{1}{c}{@20} \\",
        r"\midrule",
    ]
    numeric_by_method = {row["Method"]: row for row in numeric_rows}
    for formatted in formatted_rows:
        method = formatted["Method"]
        numeric = numeric_by_method[method]
        cells = [method]
        for metric in METRICS:
            mean = numeric[f"{metric}_mean"]
            std = numeric[f"{metric}_std"]
            if mean == "":
                cells.append(r"\texttt{--}")
                continue
            score = float(mean)
            is_best = math.isfinite(best[metric]) and abs(score - best[metric]) < 5e-12
            is_second = second[metric] is not None and abs(score - second[metric]) < 5e-12
            cells.append(latex_cell(mean, std, is_best, is_second))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    return "\n".join(lines)


def paper_table(split: str, body: str) -> str:
    return "\n".join([r"\begin{table*}[t]", r"\centering", body, r"\end{table*}", ""])


def standalone_doc(title: str, bodies: list[str]) -> str:
    tables = []
    for index, body in enumerate(bodies):
        if index:
            tables.append(r"\clearpage")
        tables.extend([r"\begin{table}[t]", r"\centering", body, r"\end{table}"])
    joined = "\n".join(tables)
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{amsmath}}
\pagestyle{{empty}}
\title{{{title}}}
\date{{}}
\begin{{document}}
{joined}
\end{{document}}
"""


def run_pdflatex(tex_path: Path) -> Path:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Run twice so references settle if LaTeX decides to emit labels.
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
    require_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail = add_overall(load_seed_detail())
    seed_detail_path = OUT_DIR / "mooccube_cold_hot_overall_seed_detail.csv"
    detail.to_csv(seed_detail_path, index=False)

    bodies: list[str] = []
    for split in ["cold", "hot", "overall"]:
        formatted_rows, numeric_rows = aggregate_split(detail, split)
        formatted_csv = OUT_DIR / f"mooccube_{split}_item_macro_table.csv"
        numeric_csv = OUT_DIR / f"mooccube_{split}_item_macro_table_numeric.csv"
        tex_path = OUT_DIR / f"mooccube_{split}_item_macro_table.tex"
        standalone_tex = OUT_DIR / f"mooccube_{split}_item_macro_table_standalone.tex"

        body = table_body(split, formatted_rows, numeric_rows)
        write_csv(formatted_csv, formatted_rows)
        write_csv(numeric_csv, numeric_rows)
        tex_path.write_text(paper_table(split, body), encoding="utf-8")
        standalone_tex.write_text(
            standalone_doc(f"MOOCCube {split} item-macro table", [body]),
            encoding="utf-8",
        )
        pdf_path = run_pdflatex(standalone_tex)
        stable_pdf = PDF_OUT_DIR / f"mooccube_{split}_item_macro_table.pdf"
        shutil.copy2(pdf_path, stable_pdf)
        bodies.append(body)

        print(f"Wrote {formatted_csv}")
        print(f"Wrote {numeric_csv}")
        print(f"Wrote {tex_path}")
        print(f"Wrote {stable_pdf}")

    combined_tex = OUT_DIR / "mooccube_cold_hot_overall_item_macro_tables_standalone.tex"
    combined_tex.write_text(
        standalone_doc("MOOCCube cold hot overall item-macro tables", bodies),
        encoding="utf-8",
    )
    combined_pdf = run_pdflatex(combined_tex)
    stable_combined_pdf = PDF_OUT_DIR / "mooccube_cold_hot_overall_item_macro_tables.pdf"
    shutil.copy2(combined_pdf, stable_combined_pdf)
    print(f"Wrote {seed_detail_path}")
    print(f"Wrote {stable_combined_pdf}")


if __name__ == "__main__":
    main()
