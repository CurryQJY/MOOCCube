from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURE_DIR = ROOT / "paper_aaai27" / "figures"
DEFAULT_COURSE_PATH = FIGURE_DIR / "validation_motivation_analysis" / "course_macro.csv"
DEFAULT_SUMMARY_PATH = FIGURE_DIR / "mooccube_validation_motivation_summary.csv"
DEFAULT_OUTPUT_BASE = FIGURE_DIR / "mooccube_validation_motivation"

STRUCTURAL_METRICS = (
    "cold_prerequisite_gap",
    "cold_concept_continuity",
    "cold_difficulty_gap",
    "cold_structural_redundancy",
)
METRIC_LABELS = {
    "cold_prerequisite_gap": r"Prerequisite gap ($\downarrow$)",
    "cold_concept_continuity": r"Concept continuity ($\uparrow$)",
    "cold_difficulty_gap": r"Difficulty gap ($\downarrow$)",
    "cold_structural_redundancy": r"Structural redundancy ($\downarrow$)",
}
MODEL_STYLE = {
    "pcgnn": {
        "label": "PCGNN",
        "color": "#C7772E",
        "marker": "o",
        "linestyle": "-",
    },
    "cgrc": {
        "label": "CGRC",
        "color": "#5E6266",
        "marker": "s",
        "linestyle": "--",
    },
}
GRID = "#D5D5D5"
DARK = "#252525"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.75,
            "font.size": 6.8,
            "axes.labelsize": 6.6,
            "axes.titlesize": 7.1,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "lines.linewidth": 1.15,
        }
    )


def validate_validation_figure_inputs(
    course_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    course_required = {
        "analysis_split",
        "model",
        "seed",
        "target_item_id",
        "ndcg_at_10",
        "cold_proportion",
        "effective_coverage",
        "missingness",
        *STRUCTURAL_METRICS,
    }
    summary_required = {
        "analysis_split",
        "model",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "unit_count",
        "observed_unit_count",
        "effective_coverage",
        "missingness",
    }
    if missing := course_required.difference(course_rows.columns):
        raise ValueError(f"course input missing columns: {sorted(missing)}")
    if missing := summary_required.difference(summary.columns):
        raise ValueError(f"summary input missing columns: {sorted(missing)}")
    if set(course_rows["analysis_split"]) != {"validation"}:
        raise ValueError("Figure 1 accepts validation-only course rows")
    if set(summary["analysis_split"]) != {"validation"}:
        raise ValueError("Figure 1 accepts validation-only summary rows")
    expected_models = set(MODEL_STYLE)
    if set(course_rows["model"]) != expected_models or set(summary["model"]) != expected_models:
        raise ValueError("Figure 1 must contain PCGNN and CGRC only")
    if course_rows.duplicated(["model", "seed", "target_item_id"]).any():
        raise ValueError("course input contains duplicate seed/target units")
    if not np.isfinite(course_rows["ndcg_at_10"].to_numpy(dtype=float)).all():
        raise ValueError("course-level NDCG@10 must be finite")

    structural = summary.loc[summary["metric"].isin(STRUCTURAL_METRICS)].copy()
    expected_pairs = {
        (model, metric)
        for model in MODEL_STYLE
        for metric in STRUCTURAL_METRICS
    }
    actual_pairs = set(zip(structural["model"], structural["metric"]))
    if actual_pairs != expected_pairs or len(structural) != len(expected_pairs):
        raise ValueError("structural summary must contain both models for all four metrics")
    interval_values = structural[["mean", "ci_low", "ci_high"]].to_numpy(dtype=float)
    if not np.isfinite(interval_values).all():
        raise ValueError("structural summary intervals must be finite")
    if np.any(interval_values[:, 1] > interval_values[:, 0]) or np.any(
        interval_values[:, 0] > interval_values[:, 2]
    ):
        raise ValueError("structural summary intervals do not contain their means")


def _summary_value(summary: pd.DataFrame, model: str, metric: str) -> float:
    selected = summary.loc[summary["model"].eq(model) & summary["metric"].eq(metric)]
    if len(selected) != 1:
        raise ValueError(f"expected one {model}/{metric} summary row")
    return float(selected.iloc[0]["mean"])


def _save_three_formats(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        output = output_base.with_suffix(suffix)
        fig.savefig(
            output,
            dpi=450,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
        outputs.append(output)
    return outputs


def draw_validation_motivation(
    course_rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_base: Path = DEFAULT_OUTPUT_BASE,
) -> list[Path]:
    validate_validation_figure_inputs(course_rows, summary)
    configure_style()

    fig, (ax_exposure, ax_structure) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.35, 4.25),
        gridspec_kw={"height_ratios": (1.03, 1.16)},
        constrained_layout=False,
    )

    ax_exposure.axvspan(0.0, 0.10, color="#ECECEC", alpha=0.85, lw=0, zorder=0)
    ax_exposure.axvline(0.10, color="#777777", lw=0.75, ls=":", zorder=1)
    max_ndcg = 0.0
    for model in ("pcgnn", "cgrc"):
        style = MODEL_STYLE[model]
        values = np.sort(
            course_rows.loc[course_rows["model"].eq(model), "ndcg_at_10"].to_numpy(
                dtype=float
            )
        )
        cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
        max_ndcg = max(max_ndcg, float(values[-1]))
        ax_exposure.step(
            values,
            cumulative,
            where="post",
            color=style["color"],
            ls=style["linestyle"],
            lw=1.25,
            zorder=3,
        )
        label_index = min(len(values) - 1, max(0, int(round(0.80 * len(values))) - 1))
        label_y = cumulative[label_index] + (0.035 if model == "pcgnn" else -0.035)
        ax_exposure.text(
            values[label_index] + 0.012,
            label_y,
            style["label"],
            color=style["color"],
            fontsize=6.4,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.3},
            zorder=4,
        )

    annotation_y = {"pcgnn": 0.30, "cgrc": 0.18}
    for model in ("pcgnn", "cgrc"):
        style = MODEL_STYLE[model]
        median = _summary_value(summary, model, "median_ndcg_at_10")
        low_fraction = _summary_value(summary, model, "low_ndcg_at_10")
        cold_share = _summary_value(summary, model, "cold_proportion")
        coverage = _summary_value(summary, model, "effective_coverage")
        ax_exposure.text(
            0.018,
            annotation_y[model],
            (
                f"{style['label']}: med {median:.3f} | "
                rf"$\leq$.10 {low_fraction:.0%} | cold {cold_share:.1%} | cov {coverage:.1%}"
            ),
            transform=ax_exposure.transAxes,
            color=style["color"],
            fontsize=5.65,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.35},
            zorder=5,
        )

    ax_exposure.set_xlim(0.0, max(0.35, min(1.0, max_ndcg * 1.10)))
    ax_exposure.set_ylim(0.0, 1.02)
    ax_exposure.set_xlabel("Course-level NDCG@10")
    ax_exposure.set_ylabel("Cumulative fraction")
    ax_exposure.set_title(
        "(a) Validation cold-course exposure",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax_exposure.grid(axis="y", color=GRID, lw=0.5, ls=":", zorder=0)

    y_positions = np.arange(len(STRUCTURAL_METRICS))[::-1]
    offsets = {"pcgnn": 0.11, "cgrc": -0.11}
    interval_max = 0.0
    for model in ("pcgnn", "cgrc"):
        style = MODEL_STYLE[model]
        model_rows = summary.loc[
            summary["model"].eq(model) & summary["metric"].isin(STRUCTURAL_METRICS)
        ].set_index("metric")
        for position, metric in zip(y_positions, STRUCTURAL_METRICS):
            row = model_rows.loc[metric]
            mean = float(row["mean"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            y = float(position + offsets[model])
            interval_max = max(interval_max, high)
            ax_structure.plot(
                [low, high],
                [y, y],
                color=style["color"],
                ls=style["linestyle"],
                lw=1.15,
                solid_capstyle="butt",
                zorder=2,
            )
            ax_structure.vlines(
                [low, high],
                y - 0.045,
                y + 0.045,
                color=style["color"],
                lw=0.75,
                zorder=2,
            )
            ax_structure.scatter(
                mean,
                y,
                marker=style["marker"],
                s=22,
                facecolor="white",
                edgecolor=style["color"],
                linewidth=1.0,
                zorder=3,
                label=style["label"] if metric == STRUCTURAL_METRICS[0] else None,
            )

    ax_structure.set_yticks(y_positions)
    ax_structure.set_yticklabels([METRIC_LABELS[metric] for metric in STRUCTURAL_METRICS])
    ax_structure.set_xlim(0.0, min(1.0, max(0.30, interval_max + 0.07)))
    ax_structure.set_ylim(-0.55, len(STRUCTURAL_METRICS) - 0.45)
    ax_structure.set_xlabel(
        "Absolute cold-only proxy\n(conditional on a cold course being recommended)"
    )
    ax_structure.set_title(
        "(b) Cold-only structural proxies",
        loc="left",
        fontweight="bold",
        pad=3,
        fontsize=6.8,
    )
    ax_structure.grid(axis="x", color=GRID, lw=0.5, ls=":", zorder=0)
    ax_structure.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.1,
    )
    pcgnn_coverage = _summary_value(summary, "pcgnn", "effective_coverage")
    cgrc_coverage = _summary_value(summary, "cgrc", "effective_coverage")
    ax_structure.text(
        0.0,
        -0.32,
        (
            "Coverage / missingness: "
            f"PCGNN {pcgnn_coverage:.1%} / {1.0 - pcgnn_coverage:.1%}; "
            f"CGRC {cgrc_coverage:.1%} / {1.0 - cgrc_coverage:.1%}"
        ),
        transform=ax_structure.transAxes,
        fontsize=5.9,
        color=DARK,
        ha="left",
        va="top",
    )

    for axis in (ax_exposure, ax_structure):
        axis.tick_params(length=2.2, pad=1.4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#444444")
        axis.spines["bottom"].set_color("#444444")

    fig.subplots_adjust(left=0.33, right=0.985, top=0.975, bottom=0.15, hspace=0.54)
    outputs = _save_three_formats(fig, output_base)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw validation-only Figure 1 motivation diagnostics")
    parser.add_argument("--course-csv", type=Path, default=DEFAULT_COURSE_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()
    outputs = draw_validation_motivation(
        pd.read_csv(args.course_csv),
        pd.read_csv(args.summary_csv),
        output_base=args.output_base,
    )
    for output in outputs:
        print(f"[Validation motivation] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
