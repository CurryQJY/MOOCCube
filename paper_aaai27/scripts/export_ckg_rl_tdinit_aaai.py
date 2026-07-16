from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "outputs"
    / "cbi_faithful_single_seed2025"
    / "strict_item_cold_balanced_thr1_seed_2025"
    / "final_report_usim_feedback_fast3_content_delta_static.csv"
)
PAPER_TEX = ROOT / "paper_aaai27" / "main.tex"
OUTPUT_DIR = ROOT / "paper_aaai27" / "figures" / "ckg_rl_tdinit_comparison"
METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")


def load_rows(source: Path = SOURCE) -> pd.DataFrame:
    lines = PAPER_TEX.read_text(encoding="utf-8").splitlines()
    main_line = next((line for line in lines if line.startswith(r"\textbf{Full CKG-RL}")), None)
    if main_line is None:
        raise ValueError(f"missing Full CKG-RL ablation row: {PAPER_TEX}")
    main_values = [float(value) for value in re.findall(r"0\.\d+", main_line)]
    if len(main_values) < len(METRICS):
        raise ValueError(f"incomplete Full CKG-RL metrics: {PAPER_TEX}")
    report = pd.read_csv(source).set_index("metric")
    missing = [metric for metric in METRICS if metric not in report.index]
    if missing:
        raise ValueError(f"missing CBI cold metrics {missing}: {source}")
    no_sg = dict(zip(METRICS, main_values[: len(METRICS)], strict=True))
    tdinit = {metric: float(report.loc[metric, "full_cold_item_macro"]) for metric in METRICS}
    imp = {metric: (tdinit[metric] - no_sg[metric]) / no_sg[metric] for metric in METRICS}
    return pd.DataFrame(
        [
            {"Method": "CKG-RL", **no_sg},
            {"Method": "CKG-RL+TDInit", **tdinit},
            {"Method": "Imp.", **imp},
        ]
    )


def write_outputs(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    tex_path = output_dir / "ckg_rl_tdinit_aaai.tex"
    csv_path = output_dir / "ckg_rl_tdinit_all_metrics.csv"
    csv_frame = rows.copy()
    csv_frame.insert(1, "Scope", [
        "ItemMacro Cold; three-seed mean",
        "ItemMacro Cold; seed 2025 single run",
        "Relative delta vs CKG-RL",
    ])
    csv_frame.insert(2, "Source", str(SOURCE.relative_to(ROOT)))
    csv_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def cell(value: float, bold: bool = False, percent: bool = False) -> str:
        text = f"{float(value) * (100 if percent else 1):+.1f}\\%" if percent else f"{float(value):.4f}"
        return rf"\textbf{{{text}}}" if bold else text

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"{",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(METRICS) + " " + (chr(92) * 2),
        r"\midrule",
    ]
    for index, row in rows.iterrows():
        method = str(row["Method"])
        if method == "Imp.":
            values = [cell(row[metric], percent=True) for metric in METRICS]
            lines.append(r"\emph{Imp.} & " + " & ".join(values) + " " + (chr(92) * 2))
        else:
            label = rf"\textbf{{{method}}}" if method == "CKG-RL+TDInit" else method
            values = [cell(row[metric], bold=method == "CKG-RL+TDInit") for metric in METRICS]
            lines.append(f"{label} & " + " & ".join(values) + " " + (chr(92) * 2))
        if index == 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"}",
            r"\caption{MOOCCube strict course-cold item-macro comparison across all reported cutoffs. CKG-RL reports the paper's three-seed mean, while CKG-RL+TDInit is the CBI seed-2025 run. Imp. is the relative improvement of CKG-RL+TDInit over CKG-RL.}",
            r"\label{tab:ckg-rl-tdinit-all-metrics}",
            r"\end{table*}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path, csv_path


def main() -> None:
    for path in write_outputs():
        print(path)


if __name__ == "__main__":
    main()
