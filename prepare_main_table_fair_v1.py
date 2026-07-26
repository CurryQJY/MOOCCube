"""
Prepare and aggregate the unified static item-cold main table.

This script materializes non-rerun results (FAST3 and existing CGRC) into the
main-table output directory and writes a compact CSV summary from all available
baseline JSON files in that directory.
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("outputs/content_delta_pop5/static_item_cold/strict_item_cold_thr1_seed_2025")
PAPER_MAIN_MODELS = [
    "Popularity",
    "BPR",
    "LightGCN",
    "DropoutNet",
    "GAR",
    "ContentProfile",
    "CGRC",
    "ALDI (official-source)",
    "FAST3",
]
PAPER_DISPLAY_NAMES = {
    "ALDI (official-source)": "ALDI",
}
PAPER_METRIC_COLS = [
    ("full_cold_R10", "Cold R@10"),
    ("full_cold_N10", "Cold N@10"),
    ("full_hot_R10", "Hot R@10"),
    ("full_hot_N10", "Hot N@10"),
]
PAPER_ITEM_MACRO_METRIC_COLS = [
    ("full_cold_item_macro_R10", "Cold R@10"),
    ("full_cold_item_macro_N10", "Cold N@10"),
    ("full_hot_item_macro_R10", "Hot R@10"),
    ("full_hot_item_macro_N10", "Hot N@10"),
]
RESULT_FILES = [
    "popularity_static_result.json",
    "bpr_static_result.json",
    "lightgcn_static_result.json",
    "lightgcl_static_result.json",
    "drop_static_result.json",
    "gar_static_result.json",
    "content_profile_static_result.json",
    "aldi_static_result.json",
    "aldi_official_static_result.json",
    "ccfcrec_static_result.json",
    "cgrc_static_result.json",
    "fast3_static_result.json",
]


def _metric_block_from_report(report_df: pd.DataFrame, col: str):
    out = {}
    for _, row in report_df.iterrows():
        metric = str(row["metric"])
        out[metric] = float(row[col])
    return out


def _materialize_fast3(root: Path, out_dir: Path):
    report_path = root / "final_report_usim_feedback_fast3_content_delta_static.csv"
    full_path = root / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if not report_path.exists() or not full_path.exists():
        print(f"Skip FAST3 materialization: missing {report_path} or {full_path}")
        return

    report_df = pd.read_csv(report_path)
    full_df = pd.read_csv(full_path)
    if report_df.empty or full_df.empty:
        print("Skip FAST3 materialization: empty CSV")
        return

    full_row = full_df.iloc[0].to_dict()
    out = {
        "model": "FAST3",
        "model_display": "FAST3",
        "source": "USIM-Feedback-FAST3-ContentDelta",
        "protocol": "static_item_cold",
        "sample_cold": _metric_block_from_report(report_df, "sampled_cold"),
        "sample_hot": _metric_block_from_report(report_df, "sampled_hot"),
        "full_cold": _metric_block_from_report(report_df, "full_cold"),
        "full_hot": _metric_block_from_report(report_df, "full_hot"),
        "count_sample_cold": int(full_row.get("sampled_cold_count", 0)),
        "count_sample_hot": int(full_row.get("sampled_hot_count", 0)),
        "count_full_cold": int(full_row.get("full_cold_count", 0)),
        "count_full_hot": int(full_row.get("full_hot_count", 0)),
        "best_epoch": None,
        "best_metric": "cold",
        "note": "Materialized from static FAST3 CSV outputs.",
    }
    if "full_cold_item_macro_n10" in full_row:
        out["full_cold_item_macro"] = {
            "R@5": float(full_row.get("full_cold_item_macro_r5", 0.0)),
            "R@10": float(full_row.get("full_cold_item_macro_r10", 0.0)),
            "R@20": float(full_row.get("full_cold_item_macro_r20", 0.0)),
            "N@5": float(full_row.get("full_cold_item_macro_n5", 0.0)),
            "N@10": float(full_row.get("full_cold_item_macro_n10", 0.0)),
            "N@20": float(full_row.get("full_cold_item_macro_n20", 0.0)),
        }
        out["full_hot_item_macro"] = {
            "R@5": float(full_row.get("full_hot_item_macro_r5", 0.0)),
            "R@10": float(full_row.get("full_hot_item_macro_r10", 0.0)),
            "R@20": float(full_row.get("full_hot_item_macro_r20", 0.0)),
            "N@5": float(full_row.get("full_hot_item_macro_n5", 0.0)),
            "N@10": float(full_row.get("full_hot_item_macro_n10", 0.0)),
            "N@20": float(full_row.get("full_hot_item_macro_n20", 0.0)),
        }
        out["count_full_cold_item_macro"] = int(full_row.get("full_cold_item_macro_count", 0))
        out["count_full_hot_item_macro"] = int(full_row.get("full_hot_item_macro_count", 0))
    target = out_dir / "fast3_static_result.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump([out], f, ensure_ascii=False, indent=2)
    print(f"Wrote {target}")


def _copy_existing_cgrc(root: Path, out_dir: Path):
    src = root / "cgrc_static_result.json"
    dst = out_dir / "cgrc_static_result.json"
    if dst.exists():
        return
    if not src.exists():
        print(f"Skip CGRC copy: missing {src}")
        return
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data:
        data[0]["protocol"] = "static_item_cold"
        data[0].setdefault("best_metric", "cold")
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.copystat(src, dst)
    print(f"Materialized {src} -> {dst}")


def _get_metric(obj, section: str, metric: str):
    block = obj.get(section)
    if isinstance(block, dict):
        return block.get(metric)
    return obj.get(f"{section}_{metric}")


def _result_row(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)[0]
    row = {
        "file": path.name,
        "model": obj.get("model_display") or obj.get("model") or path.stem,
        "protocol": obj.get("protocol"),
        "best_epoch": obj.get("best_epoch"),
        "best_metric": obj.get("best_metric"),
        "best_val_full_cold_n10": obj.get("best_val_full_cold_n10"),
        "count_sample_cold": obj.get("count_sample_cold"),
        "count_sample_hot": obj.get("count_sample_hot"),
        "count_full_cold": obj.get("count_full_cold"),
        "count_full_hot": obj.get("count_full_hot"),
        "count_full_cold_item_macro": obj.get("count_full_cold_item_macro"),
        "count_full_hot_item_macro": obj.get("count_full_hot_item_macro"),
        "source": obj.get("source"),
        "note": obj.get("note"),
    }
    for section in [
        "sample_cold",
        "sample_hot",
        "full_cold",
        "full_hot",
        "full_cold_item_macro",
        "full_hot_item_macro",
    ]:
        prefix = section.replace("sample", "samp")
        for metric in ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]:
            value = _get_metric(obj, section, metric)
            row[f"{section}_{metric}"] = value
            row[f"{prefix}_{metric.replace('@', '')}"] = value
    return row


def _format_plain(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _format_latex(value, best_value, second_value) -> str:
    if value is None or pd.isna(value):
        return "-"
    val = float(value)
    formatted = f"{val:.4f}"
    if best_value is not None and abs(val - best_value) < 5e-13:
        return rf"\textbf{{{formatted}}}"
    if second_value is not None and abs(val - second_value) < 5e-13:
        return rf"\underline{{{formatted}}}"
    return formatted


def _top_values(values):
    vals = sorted({float(v) for v in values if v is not None and not pd.isna(v)}, reverse=True)
    best = vals[0] if vals else None
    second = vals[1] if len(vals) > 1 else None
    return best, second


def _write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(path: Path, df: pd.DataFrame, metric_cols) -> None:
    best_second = {}
    for src_col, label in metric_cols:
        best_second[label] = _top_values(df[label].tolist())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Static item-cold recommendation results under full-ranking evaluation.}",
        r"\label{tab:static-item-cold-main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & \multicolumn{2}{c}{Cold} & \multicolumn{2}{c}{Hot} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & R@10 & N@10 & R@10 & N@10 \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        raw_model = str(row["Model"])
        model = raw_model if raw_model.startswith(r"\textbf{") else _latex_escape(raw_model)
        vals = []
        for _, label in metric_cols:
            best, second = best_second[label]
            vals.append(_format_latex(row[label], best, second))
        lines.append(model + " & " + " & ".join(vals) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_narrow_tables(detail: pd.DataFrame, out_dir: Path) -> None:
    if detail.empty:
        return
    base_df = detail.copy()
    for col in [
        "full_cold_R10",
        "full_cold_N10",
        "full_hot_R10",
        "full_hot_N10",
        "full_cold_item_macro_R10",
        "full_cold_item_macro_N10",
        "full_hot_item_macro_R10",
        "full_hot_item_macro_N10",
    ]:
        if col not in base_df.columns:
            base_df[col] = pd.NA
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")

    def write_variant(metric_cols, suffix: str, label: str) -> None:
        if not all(src in base_df.columns for src, _ in metric_cols):
            print(f"Skip {label} narrow table: missing metric columns")
            return
        if base_df[[src for src, _ in metric_cols]].dropna(how="all").empty:
            print(f"Skip {label} narrow table: no metric values")
            return

        df = base_df.copy()
        df = df[df["model"].isin(PAPER_MAIN_MODELS)].copy()
        order = {name: idx for idx, name in enumerate(PAPER_MAIN_MODELS)}
        df["__order"] = df["model"].map(order)
        df = df.sort_values("__order")

        raw_cols = ["model"] + [src for src, _ in metric_cols]
        raw = df[raw_cols].rename(columns={"model": "Model", **{src: metric_label for src, metric_label in metric_cols}})
        raw["Model"] = raw["Model"].replace(PAPER_DISPLAY_NAMES)
        suffix_part = f"_{suffix}" if suffix else ""

        csv_path = out_dir / f"main_table_fair_v1_paper_narrow{suffix_part}.csv"
        raw.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        pretty = raw.copy()
        for _, metric_label in metric_cols:
            pretty[metric_label] = pretty[metric_label].map(_format_plain)
        md_path = out_dir / f"main_table_fair_v1_paper_narrow{suffix_part}.md"
        _write_markdown_table(md_path, pretty)
        print(f"Wrote {md_path}")

        tex_path = out_dir / f"main_table_fair_v1_paper_narrow{suffix_part}.tex"
        _write_latex_table(tex_path, raw, metric_cols)
        print(f"Wrote {tex_path}")

    mode = os.environ.get("MAIN_TABLE_METRIC_MODE", "interaction").strip().lower()
    if mode == "item_macro":
        write_variant(PAPER_ITEM_MACRO_METRIC_COLS, "", "item-macro")
        write_variant(PAPER_METRIC_COLS, "interaction", "interaction")
    else:
        write_variant(PAPER_METRIC_COLS, "", "interaction")
    write_variant(PAPER_ITEM_MACRO_METRIC_COLS, "item_macro", "item-macro")


def main():
    root = Path(os.environ.get("USIM_STATIC_SPLIT_DIR", str(DEFAULT_ROOT)))
    out_dir = Path(os.environ.get("USIM_BASELINE_OUTPUT_DIR", str(root / "main_table_fair_v1")))
    out_dir.mkdir(parents=True, exist_ok=True)

    _materialize_fast3(root, out_dir)
    _copy_existing_cgrc(root, out_dir)

    rows = []
    for filename in RESULT_FILES:
        path = out_dir / filename
        if path.exists():
            rows.append(_result_row(path))
        else:
            print(f"Missing result: {path}")

    detail = pd.DataFrame(rows)
    detail_path = out_dir / "main_table_fair_v1_summary.csv"
    detail.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}")
    _write_paper_narrow_tables(detail, out_dir)


if __name__ == "__main__":
    main()
