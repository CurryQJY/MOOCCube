from __future__ import annotations

from pathlib import Path

import pandas as pd


BASELINE_SUMMARY = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_item_macro_final_audit/main_table_item_macro_summary.csv"
)
OLD2667_SUMMARY = Path(
    "outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv"
)
OUT_DIR = Path(
    "outputs/content_delta_pop5/course_ablation_e60_3seed/full/baseline_comparison_export"
)

MODEL_ORDER = [
    "Popularity",
    "BPR",
    "LightGCN",
    "DropoutNet",
    "ContentProfile",
    "CCFCRec",
    "ALDI",
    "CGRC-paper",
    "USIM",
    "PAM",
    "FS-GNN",
    "M2VAE",
    "SAGERec",
    "SEMCo",
    "CourseAware-MLP",
    "CKG-RL-old2667",
]

METRIC_KEYS = ["R5", "R10", "R20", "N5", "N10", "N20"]


def _old_row(raw: pd.Series) -> dict:
    out = {
        "model": "CKG-RL-old2667",
        "runs": int(raw["runs"]),
        "seeds": raw["seeds"],
        "mean_best_epoch": 60.0,
        "count_cold_mean": float(raw["full_cold_item_macro_count_mean"]),
        "count_hot_mean": float(raw["full_hot_item_macro_count_mean"]),
    }
    for split in ["cold", "hot"]:
        src = f"full_{split}_item_macro"
        for metric in METRIC_KEYS:
            lower = metric.lower()
            out[f"{split}_{metric}_mean"] = float(raw[f"{src}_{lower}_mean"])
            out[f"{split}_{metric}_std"] = float(raw[f"{src}_{lower}_std"])
    return out


def _weighted_overall(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cold_count = pd.to_numeric(out["count_cold_mean"], errors="coerce")
    hot_count = pd.to_numeric(out["count_hot_mean"], errors="coerce")
    denom = cold_count + hot_count
    for metric in METRIC_KEYS:
        cold = pd.to_numeric(out[f"cold_{metric}_mean"], errors="coerce")
        hot = pd.to_numeric(out[f"hot_{metric}_mean"], errors="coerce")
        out[f"overall_{metric}_mean"] = (cold * cold_count + hot * hot_count) / denom
    return out


def _format_mean_std(row: pd.Series, prefix: str, metric: str) -> str:
    mean = float(row[f"{prefix}_{metric}_mean"])
    std_col = f"{prefix}_{metric}_std"
    if std_col in row and pd.notna(row[std_col]):
        return f"{mean:.4f} +/- {float(row[std_col]):.4f}"
    return f"{mean:.4f}"


def _paper_narrow(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Cold R@10": _format_mean_std(row, "cold", "R10"),
                "Cold N@10": _format_mean_std(row, "cold", "N10"),
                "Hot R@10": _format_mean_std(row, "hot", "R10"),
                "Hot N@10": _format_mean_std(row, "hot", "N10"),
                "Overall R@10": f"{float(row['overall_R10_mean']):.4f}",
                "Overall N@10": f"{float(row['overall_N10_mean']):.4f}",
            }
        )
    return pd.DataFrame(rows)


def _main_table_full(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        out = {"Model": row["model"]}
        for split_name, prefix in [("Cold", "cold"), ("Hot", "hot")]:
            for metric_label, metric_prefix in [("R", "R"), ("N", "N")]:
                for k in [5, 10, 20]:
                    metric = f"{metric_prefix}{k}"
                    out[f"{split_name} {metric_label}@{k}"] = _format_mean_std(row, prefix, metric)
        rows.append(out)
    return pd.DataFrame(rows)


def _latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def _latex_table(paper: pd.DataFrame) -> str:
    numeric_cols = [col for col in paper.columns if col != "Model"]
    best = {}
    second = {}
    for col in numeric_cols:
        vals = []
        for value in paper[col]:
            vals.append(float(str(value).split()[0]))
        unique = sorted(set(vals), reverse=True)
        best[col] = unique[0]
        second[col] = unique[1] if len(unique) > 1 else None

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{MOOCCube baseline comparison with the legacy CKG-RL .2667 run. Values are three-seed item-macro means; cold/hot columns include standard deviations. Overall is count-weighted by the mean number of cold and hot items.}",
        r"\label{tab:old2667-baseline-comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Cold R@10 & Cold N@10 & Hot R@10 & Hot N@10 & Overall R@10 & Overall N@10 \\",
        r"\midrule",
    ]
    for _, row in paper.iterrows():
        cells = [_latex_escape(row["Model"])]
        for col in numeric_cols:
            value = str(row[col]).replace("+/-", r"$\pm$")
            score = float(str(row[col]).split()[0])
            if abs(score - best[col]) < 5e-12:
                value = rf"\textbf{{{value}}}"
            elif second[col] is not None and abs(score - second[col]) < 5e-12:
                value = rf"\underline{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _latex_main_table(paper: pd.DataFrame) -> str:
    numeric_cols = [col for col in paper.columns if col != "Model"]
    best = {}
    second = {}
    for col in numeric_cols:
        vals = [float(str(value).split()[0]) for value in paper[col]]
        unique = sorted(set(vals), reverse=True)
        best[col] = unique[0]
        second[col] = unique[1] if len(unique) > 1 else None

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{MOOCCube strict item-cold full-ranking item-macro results with the legacy CKG-RL .2667 run. Values are mean$\pm$std over three seeds. Best results are bolded and second-best results are underlined.}",
        r"\label{tab:old2667-main-table-full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{6}{c}{Cold item-macro} & \multicolumn{6}{c}{Hot item-macro} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r" & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 \\",
        r"\midrule",
    ]
    ordered_cols = [
        "Cold R@5",
        "Cold R@10",
        "Cold R@20",
        "Cold N@5",
        "Cold N@10",
        "Cold N@20",
        "Hot R@5",
        "Hot R@10",
        "Hot R@20",
        "Hot N@5",
        "Hot N@10",
        "Hot N@20",
    ]
    for _, row in paper.iterrows():
        cells = [_latex_escape(row["Model"])]
        for col in ordered_cols:
            value = str(row[col]).replace("+/-", r"$\pm$")
            score = float(str(row[col]).split()[0])
            if abs(score - best[col]) < 5e-12:
                value = rf"\textbf{{{value}}}"
            elif second[col] is not None and abs(score - second[col]) < 5e-12:
                value = rf"\underline{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    if not BASELINE_SUMMARY.exists():
        raise FileNotFoundError(BASELINE_SUMMARY)
    if not OLD2667_SUMMARY.exists():
        raise FileNotFoundError(OLD2667_SUMMARY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(BASELINE_SUMMARY)
    old = pd.read_csv(OLD2667_SUMMARY)
    if old.empty:
        raise ValueError(f"empty old .2667 summary: {OLD2667_SUMMARY}")
    combined = pd.concat([baseline, pd.DataFrame([_old_row(old.iloc[0])])], ignore_index=True)
    combined = _weighted_overall(combined)
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    combined["__order"] = combined["model"].map(order).fillna(999)
    combined = combined.sort_values(["__order", "model"]).drop(columns="__order")

    full_path = OUT_DIR / "old2667_vs_baselines_full.csv"
    narrow_path = OUT_DIR / "old2667_vs_baselines_paper_narrow.csv"
    latex_path = OUT_DIR / "old2667_vs_baselines_paper_narrow.tex"
    main_table_path = OUT_DIR / "old2667_vs_baselines_main_table_full.csv"
    main_table_tex_path = OUT_DIR / "old2667_vs_baselines_main_table_full.tex"

    combined.to_csv(full_path, index=False)
    paper = _paper_narrow(combined)
    paper.to_csv(narrow_path, index=False)
    latex_path.write_text(_latex_table(paper), encoding="utf-8")
    main_table = _main_table_full(combined)
    main_table.to_csv(main_table_path, index=False)
    main_table_tex_path.write_text(_latex_main_table(main_table), encoding="utf-8")

    print(f"Wrote {full_path}")
    print(f"Wrote {narrow_path}")
    print(f"Wrote {latex_path}")
    print(f"Wrote {main_table_path}")
    print(f"Wrote {main_table_tex_path}")


if __name__ == "__main__":
    main()
