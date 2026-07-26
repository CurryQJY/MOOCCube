from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("R@5", "r5"),
    ("R@10", "r10"),
    ("R@20", "r20"),
    ("N@5", "n5"),
    ("N@10", "n10"),
    ("N@20", "n20"),
)
PANELS = ("cold", "hot", "overall")
PANEL_TITLES = {
    "cold": "Cold",
    "hot": "Hot",
    "overall": "Overall",
}
BASELINE_LABEL = "CKG-RL"
TDINIT_LABEL = "CKG-RL+TDInit"


def count_weighted_overall(
    cold_value: float,
    hot_value: float,
    cold_count: float,
    hot_count: float,
) -> float:
    total = cold_count + hot_count
    if total <= 0:
        raise ValueError("Cold and Hot item counts must sum to a positive value")
    return (cold_value * cold_count + hot_value * hot_count) / total


def _load_seed_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No experiment rows found in {path}")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"Duplicate seed {seed} in {path}")
        result[seed] = row
    return result


def _metric_value(row: dict[str, str], panel: str, suffix: str) -> float:
    if panel in {"cold", "hot"}:
        return float(row[f"full_{panel}_item_macro_{suffix}"])
    if panel != "overall":
        raise ValueError(f"Unknown panel: {panel}")
    return count_weighted_overall(
        cold_value=float(row[f"full_cold_item_macro_{suffix}"]),
        hot_value=float(row[f"full_hot_item_macro_{suffix}"]),
        cold_count=float(row["full_cold_item_macro_count"]),
        hot_count=float(row["full_hot_item_macro_count"]),
    )


def _summarize(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, float]]:
    rows = list(rows)
    summary: dict[str, dict[str, float]] = {}
    for panel in PANELS:
        summary[panel] = {}
        for label, suffix in METRICS:
            values = [_metric_value(row, panel, suffix) for row in rows]
            summary[panel][label] = statistics.mean(values)
    return summary


def build_comparison(
    baseline_csv: Path,
    tdinit_csv: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    baseline_rows = _load_seed_rows(baseline_csv)
    tdinit_rows = _load_seed_rows(tdinit_csv)
    if baseline_rows.keys() != tdinit_rows.keys():
        raise ValueError(
            "Baseline and TDInit seeds differ: "
            f"{sorted(baseline_rows)} vs {sorted(tdinit_rows)}"
        )

    baseline = _summarize(baseline_rows[seed] for seed in sorted(baseline_rows))
    tdinit = _summarize(tdinit_rows[seed] for seed in sorted(tdinit_rows))
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for panel in PANELS:
        improvement = {
            metric: (tdinit[panel][metric] / baseline[panel][metric] - 1.0) * 100.0
            for metric, _ in METRICS
        }
        comparison[panel] = {
            BASELINE_LABEL: baseline[panel],
            TDINIT_LABEL: tdinit[panel],
            "Imp.": improvement,
        }
    return comparison


def _write_csv(
    comparison: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    headers = ["Panel", "Method", *(metric for metric, _ in METRICS)]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for panel in PANELS:
            for method in (BASELINE_LABEL, TDINIT_LABEL, "Imp."):
                values = comparison[panel][method]
                if method == "Imp.":
                    formatted = [f"{values[metric]:+.1f}%" for metric, _ in METRICS]
                else:
                    formatted = [f"{values[metric]:.6f}" for metric, _ in METRICS]
                writer.writerow([PANEL_TITLES[panel], method, *formatted])


def _latex_value(value: float, bold: bool) -> str:
    text = f"{value:.4f}"
    return f"\\textbf{{{text}}}" if bold else text


def _write_latex(
    comparison: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cold, hot, and overall item-macro performance of CKG-RL and CKG-RL+TDInit.}",
        r"\label{tab:tdinit_cold_hot_overall}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
    ]
    for panel_index, panel in enumerate(PANELS):
        if panel_index:
            lines.append(r"\midrule")
        lines.extend(
            [
                rf"\multicolumn{{7}}{{l}}{{\textbf{{{PANEL_TITLES[panel]}}}}} \\",
                r"Method & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 \\",
                r"\midrule",
            ]
        )
        baseline = comparison[panel][BASELINE_LABEL]
        tdinit = comparison[panel][TDINIT_LABEL]
        baseline_cells = [
            _latex_value(baseline[metric], baseline[metric] >= tdinit[metric])
            for metric, _ in METRICS
        ]
        tdinit_cells = [
            _latex_value(tdinit[metric], tdinit[metric] > baseline[metric])
            for metric, _ in METRICS
        ]
        improvement_cells = [
            f"{comparison[panel]['Imp.'][metric]:+.1f}\\%" for metric, _ in METRICS
        ]
        lines.extend(
            [
                f"{BASELINE_LABEL} & " + " & ".join(baseline_cells) + r" \\",
                f"{TDINIT_LABEL} & " + " & ".join(tdinit_cells) + r" \\",
                r"\midrule",
                r"\textit{Imp.} & " + " & ".join(improvement_cells) + r" \\",
            ]
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]\footnotesize",
            r"\item TDInit denotes the current CBI method. Values are three-seed item-macro means. Overall is computed per seed by weighting cold and hot item-macro values by their evaluated course counts. Imp. is the signed relative change over CKG-RL.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _style_table(
    axis: plt.Axes,
    panel: str,
    comparison: dict[str, dict[str, dict[str, float]]],
) -> None:
    axis.axis("off")
    axis.set_title(
        f"{PANEL_TITLES[panel]} (Item-macro, 3-seed mean)",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=8,
    )
    baseline = comparison[panel][BASELINE_LABEL]
    tdinit = comparison[panel][TDINIT_LABEL]
    improvement = comparison[panel]["Imp."]
    rows = [
        [BASELINE_LABEL, *(f"{baseline[metric]:.4f}" for metric, _ in METRICS)],
        [TDINIT_LABEL, *(f"{tdinit[metric]:.4f}" for metric, _ in METRICS)],
        ["Imp.", *(f"{improvement[metric]:+.1f}%" for metric, _ in METRICS)],
    ]
    columns = ["Method", *(metric for metric, _ in METRICS)]
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13],
        bbox=[0.0, 0.02, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)

    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        cell.PAD = 0.012
        if column == 0:
            cell.get_text().set_ha("left")
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(1.2)
            cell.get_text().set_fontsize(17)
        elif row == 2:
            cell.visible_edges = "B"
            cell.set_linewidth(1.2)
        elif row == 3:
            cell.visible_edges = "B"
            cell.set_linewidth(1.6)
            if column == 0:
                cell.get_text().set_fontstyle("italic")

    for column, (metric, _) in enumerate(METRICS, start=1):
        if tdinit[metric] > baseline[metric]:
            table[(2, column)].get_text().set_fontweight("bold")
        else:
            table[(1, column)].get_text().set_fontweight("bold")
    table[(2, 0)].get_text().set_fontweight("bold")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_tables(
    comparison: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    for panel in PANELS:
        fig, axis = plt.subplots(figsize=(14.0, 3.0))
        _style_table(axis, panel, comparison)
        _save_figure(fig, output_dir / f"tdinit_comparison_{panel}")

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 9.0))
    for axis, panel in zip(axes, PANELS):
        _style_table(axis, panel, comparison)
    fig.subplots_adjust(hspace=0.34)
    _save_figure(fig, output_dir / "tdinit_comparison_all")


def generate_tables(baseline_csv: Path, tdinit_csv: Path, output_dir: Path) -> None:
    baseline_csv = Path(baseline_csv)
    tdinit_csv = Path(tdinit_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison(baseline_csv, tdinit_csv)
    _write_csv(comparison, output_dir / "tdinit_comparison.csv")
    _write_latex(comparison, output_dir / "tdinit_comparison.tex")
    _render_tables(comparison, output_dir)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=root
        / "outputs"
        / "significance_per_item_exports"
        / "mooccube"
        / "ckg_rl_full"
        / "fast3_static_runs_detail.csv",
    )
    parser.add_argument(
        "--tdinit",
        type=Path,
        default=root
        / "outputs"
        / "cbi_anchor_sim_3seed_serial"
        / "fast3_static_runs_detail.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_aaai27" / "figures" / "ckg_rl_tdinit_3seed_latest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tables(args.baseline, args.tdinit, args.output_dir)


if __name__ == "__main__":
    main()
