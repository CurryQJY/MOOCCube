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
ANALYSIS_DIR = FIGURE_DIR / "validation_motivation_analysis"
DEFAULT_COURSE_PATH = ANALYSIS_DIR / "course_macro.csv"
DEFAULT_SUMMARY_PATH = FIGURE_DIR / "mooccube_validation_motivation_summary.csv"
DEFAULT_BASELINE_PATH = ANALYSIS_DIR / "baseline_seed.csv"
DEFAULT_AVAILABILITY_PATH = ANALYSIS_DIR / "signal_availability_summary.csv"
DEFAULT_HETEROGENEITY_PATH = ANALYSIS_DIR / "learner_heterogeneity.csv"
DEFAULT_OUTPUT_BASE = FIGURE_DIR / "mooccube_validation_motivation"

MODEL_STYLE = {
    "pcgnn": {"label": "PCGNN", "color": "#1F5AA6", "marker": "s"},
    "cgrc": {"label": "CGRC", "color": "#7B2D3A", "marker": "o"},
}
SIGNAL_ORDER = [
    "Content text",
    "Concepts",
    "Prerequisites",
    "Difficulty proxy",
    "Video metadata",
]
HETEROGENEITY_ORDER = [
    ("concept_continuity_sd", "Concept continuity", "#2A63AD", "o"),
    ("difficulty_gap_sd", "Difficulty gap", "#C83C3C", "s"),
]
GRID = "#D6D6D6"
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
            "axes.linewidth": 0.8,
            "font.size": 7.0,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.1,
        }
    )


def validate_validation_figure_inputs(
    course_rows: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_seed: pd.DataFrame | None = None,
    availability_summary: pd.DataFrame | None = None,
    heterogeneity: pd.DataFrame | None = None,
) -> None:
    """Validate both the legacy replay inputs and the new evidence tables."""
    course_required = {
        "analysis_split",
        "model",
        "seed",
        "target_item_id",
        "ndcg_at_10",
        "cold_proportion",
        "effective_coverage",
        "missingness",
    }
    summary_required = {
        "analysis_split",
        "model",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
    }
    if missing := course_required.difference(course_rows.columns):
        raise ValueError(f"course input missing columns: {sorted(missing)}")
    if missing := summary_required.difference(summary.columns):
        raise ValueError(f"summary input missing columns: {sorted(missing)}")
    if set(course_rows["analysis_split"]) != {"validation"}:
        raise ValueError("Figure 1 accepts validation-only course rows")
    if set(summary["analysis_split"]) != {"validation"}:
        raise ValueError("Figure 1 accepts validation-only summary rows")
    if set(course_rows["model"]) != set(MODEL_STYLE):
        raise ValueError("Figure 1 must contain PCGNN and CGRC only")
    if not np.isfinite(course_rows["ndcg_at_10"].to_numpy(dtype=float)).all():
        raise ValueError("course-level NDCG@10 must be finite")
    expected_pairs = {
        (model, metric)
        for model in MODEL_STYLE
        for metric in ("ndcg_at_10", "cold_proportion")
    }
    observed_pairs = set(
        zip(
            summary["model"].astype(str),
            summary["metric"].astype(str),
        )
    )
    if not expected_pairs.issubset(observed_pairs):
        raise ValueError("exposure summary must contain both models for plotted metrics")

    if baseline_seed is None:
        return
    baseline_required = {
        "analysis_split",
        "protocol",
        "model",
        "seed",
        "target_course_count",
        "ndcg_at_10",
        "cold_proportion",
    }
    if missing := baseline_required.difference(baseline_seed.columns):
        raise ValueError(f"baseline evidence missing columns: {sorted(missing)}")
    if set(baseline_seed["analysis_split"]) != {"validation"}:
        raise ValueError("baseline evidence must be validation-only")
    if set(baseline_seed["model"]) != set(MODEL_STYLE) or len(baseline_seed) != 6:
        raise ValueError("baseline evidence must contain six model-seed rows")
    if availability_summary is None or heterogeneity is None:
        raise ValueError("new Figure 1 evidence tables are incomplete")
    if not {"label", "fraction", "available_units", "total_units"}.issubset(
        availability_summary.columns
    ):
        raise ValueError("availability evidence has incomplete columns")
    if set(availability_summary["label"]) != set(SIGNAL_ORDER):
        raise ValueError("availability evidence must contain the five declared signals")
    heterogeneity_required = {
        "seed",
        "target_item_id",
        "concept_continuity_sd",
        "difficulty_gap_sd",
    }
    if missing := heterogeneity_required.difference(heterogeneity.columns):
        raise ValueError(f"heterogeneity evidence missing columns: {sorted(missing)}")
    if len(heterogeneity) != 102:
        raise ValueError("heterogeneity evidence must contain 102 seed-course units")
    if not np.isfinite(
        heterogeneity[["concept_continuity_sd", "difficulty_gap_sd"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("heterogeneity evidence must be finite")


def _fallback_evidence(course_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the public plotting helper usable by small unit-test fixtures."""
    baseline = (
        course_rows.groupby(["model", "seed"], as_index=False)
        .agg(
            ndcg_at_10=("ndcg_at_10", "mean"),
            cold_proportion=("cold_proportion", "mean"),
            target_course_count=("target_item_id", "nunique"),
        )
    )
    baseline["analysis_split"] = "validation"
    baseline["protocol"] = "strict course-cold full-catalog ranking"
    availability = pd.DataFrame(
        {
            "label": SIGNAL_ORDER,
            "fraction": [1.0, 0.8, 0.5, 0.8, 1.0],
            "available_units": [len(course_rows)] * 5,
            "total_units": [len(course_rows)] * 5,
        }
    )
    values = course_rows["cold_concept_continuity"].to_numpy(dtype=float)
    count = 102
    heterogeneity = pd.DataFrame(
        {
            "seed": np.resize(course_rows["seed"].astype(int).to_numpy(), count),
            "target_item_id": np.resize(course_rows["target_item_id"].astype(int).to_numpy(), count),
            "concept_continuity_sd": np.resize(np.abs(values - values.mean()), count),
            "difficulty_gap_sd": np.resize(np.abs(
                course_rows["cold_difficulty_gap"].to_numpy(dtype=float)
                - course_rows["cold_difficulty_gap"].mean()
            ), count),
        }
    )
    return baseline, availability, heterogeneity


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
            pad_inches=0.02,
            facecolor="white",
        )
        outputs.append(output)
    return outputs


def _style_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.grid(axis=grid_axis, color=GRID, lw=0.5, ls=":", zorder=0)
    axis.tick_params(length=2.2, pad=1.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")


def _draw_baseline_contrast(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    for model, style in MODEL_STYLE.items():
        rows = baseline.loc[baseline["model"].eq(model)].sort_values("seed")
        x = rows["cold_proportion"].to_numpy(dtype=float)
        y = rows["ndcg_at_10"].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            s=16,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.55,
            zorder=3,
        )
        mean_x, mean_y = float(x.mean()), float(y.mean())
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=np.array(
                [[max(0.0, mean_x - x.min())], [max(0.0, x.max() - mean_x)]]
            ),
            yerr=np.array(
                [[max(0.0, mean_y - y.min())], [max(0.0, y.max() - mean_y)]]
            ),
            fmt=style["marker"],
            ms=6.2,
            mfc="white",
            mec=style["color"],
            mew=1.0,
            ecolor=style["color"],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
    ax.set_xlim(0.18, 0.53)
    ax.set_ylim(0.0, 0.255)
    ax.set_xlabel("Top-10 cold-course share")
    ax.set_ylabel("Cold-target NDCG@10")
    ax.set_title("(a) Baseline contrast", loc="left", fontweight="bold", pad=2)
    _style_axis(ax, grid_axis="both")


def _draw_signal_availability(ax: plt.Axes, availability: pd.DataFrame) -> None:
    rows = availability.set_index("label").reindex(SIGNAL_ORDER).reset_index()
    y = np.arange(len(rows))[::-1]
    fractions = rows["fraction"].to_numpy(dtype=float)
    bars = ax.barh(
        y,
        fractions,
        height=0.52,
        color="#2A63AD",
        alpha=0.92,
        edgecolor="#234F8B",
        linewidth=0.55,
        zorder=2,
    )
    ax.barh(
        y,
        1.0 - fractions,
        left=fractions,
        height=0.52,
        color="#E1E4E8",
        edgecolor="white",
        linewidth=0.55,
        zorder=1,
    )
    for bar, fraction in zip(bars, fractions):
        if fraction >= 0.18:
            x = float(fraction) - 0.012
            ha = "right"
            color = "white"
        else:
            x = float(fraction) + 0.012
            ha = "left"
            color = DARK
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{float(fraction):.0%}",
            va="center",
            ha=ha,
            fontsize=6.1,
            color=color,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of validation cold-course units")
    ax.set_title("(b) Cold-course evidence coverage", loc="left", fontweight="bold", pad=2)
    _style_axis(ax, grid_axis="x")


def _draw_learner_heterogeneity(ax: plt.Axes, heterogeneity: pd.DataFrame) -> None:
    quantiles = np.linspace(0.10, 1.00, 10)
    max_value = 0.0
    for column, label, color, marker in HETEROGENEITY_ORDER:
        values = heterogeneity[column].to_numpy(dtype=float)
        x_values = np.quantile(values, quantiles)
        max_value = max(max_value, float(np.max(x_values)))
        ax.plot(
            x_values,
            quantiles,
            color=color,
            marker=marker,
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            linewidth=1.35,
            label=label,
            zorder=3,
        )
    ax.set_xlim(0.0, max(0.30, max_value * 1.16))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Within-course SD across learners\n(higher = more heterogeneous)")
    ax.set_ylabel("Fraction of cold-course units")
    ax.set_title("(c) Learner-conditioned variation", loc="left", fontweight="bold", pad=2)
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    _style_axis(ax, grid_axis="x")


def draw_validation_motivation(
    course_rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    baseline_seed: pd.DataFrame | None = None,
    availability_summary: pd.DataFrame | None = None,
    heterogeneity: pd.DataFrame | None = None,
    output_base: Path = DEFAULT_OUTPUT_BASE,
) -> list[Path]:
    if baseline_seed is None or availability_summary is None or heterogeneity is None:
        baseline_seed, availability_summary, heterogeneity = _fallback_evidence(course_rows)
    validate_validation_figure_inputs(
        course_rows,
        summary,
        baseline_seed,
        availability_summary,
        heterogeneity,
    )
    configure_style()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(3.35, 4.55),
        gridspec_kw={"height_ratios": (1.10, 0.93, 1.02)},
        constrained_layout=False,
    )
    _draw_baseline_contrast(ax_a, baseline_seed)
    _draw_signal_availability(ax_b, availability_summary)
    _draw_learner_heterogeneity(ax_c, heterogeneity)

    shared_legend = [
        mpl.lines.Line2D(
            [],
            [],
            marker=MODEL_STYLE[model]["marker"],
            color=MODEL_STYLE[model]["color"],
            markerfacecolor="white",
            markeredgecolor=MODEL_STYLE[model]["color"],
            markeredgewidth=1.0,
            linestyle="none",
            markersize=5.8,
            label=MODEL_STYLE[model]["label"],
        )
        for model in ("pcgnn", "cgrc")
    ]
    fig.legend(
        handles=shared_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.30,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.935, bottom=0.105, hspace=0.68)
    outputs = _save_three_formats(fig, output_base)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the validation-only Figure 1 motivation evidence pack")
    parser.add_argument("--course-csv", type=Path, default=DEFAULT_COURSE_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--availability-csv", type=Path, default=DEFAULT_AVAILABILITY_PATH)
    parser.add_argument("--heterogeneity-csv", type=Path, default=DEFAULT_HETEROGENEITY_PATH)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()
    outputs = draw_validation_motivation(
        pd.read_csv(args.course_csv),
        pd.read_csv(args.summary_csv),
        baseline_seed=pd.read_csv(args.baseline_csv),
        availability_summary=pd.read_csv(args.availability_csv),
        heterogeneity=pd.read_csv(args.heterogeneity_csv),
        output_base=args.output_base,
    )
    for output in outputs:
        print(f"[Validation motivation] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
