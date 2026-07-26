from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MAIN_METRICS = (
    "prerequisite_gap",
    "concept_continuity",
    "difficulty_gap",
    "structural_redundancy",
)
METRIC_LABELS = {
    "prerequisite_gap": "Prerequisite gap",
    "concept_continuity": "Concept continuity",
    "difficulty_gap": "Difficulty gap",
    "structural_redundancy": "Structural redundancy",
}
PALETTE = {
    "ckg_rl": "#2F6B9A",
    "pcgnn": "#C4772E",
    "cgrc": "#9A9A9A",
    "positive": "#2F7D6D",
    "negative": "#B24C45",
    "neutral": "#777777",
    "grid": "#D8D8D8",
    "ink": "#222222",
}
COMPARISON_SPECS = (
    ("ckg_rl_vs_pcgnn", "vs PCGNN (course)", "pcgnn", "o"),
    ("ckg_rl_vs_cgrc", "vs CGRC (cold-start)", "cgrc", "s"),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 7.0,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _difference_column(frame: pd.DataFrame) -> str:
    if "mean_difference" in frame:
        return "mean_difference"
    if "mean_difference_ckg_rl_minus_cgrc" in frame:
        return "mean_difference_ckg_rl_minus_cgrc"
    raise ValueError("paired statistics lack a mean-difference column")


def favorable_effect_rows(
    paired: pd.DataFrame,
    comparison: str | None = None,
) -> pd.DataFrame:
    selected = paired.loc[
        paired["cutoff"].eq(10) & paired["metric"].isin(MAIN_METRICS)
    ].copy()
    if comparison is not None:
        if "comparison" not in selected:
            raise ValueError("paired statistics lack comparison labels")
        selected = selected.loc[selected["comparison"].eq(comparison)].copy()
    elif "comparison" in selected and selected["comparison"].nunique() != 1:
        raise ValueError("comparison must be selected when multiple baselines exist")
    selected["metric"] = pd.Categorical(
        selected["metric"],
        categories=MAIN_METRICS,
        ordered=True,
    )
    selected = selected.sort_values("metric").reset_index(drop=True)
    if selected["metric"].astype(str).tolist() != list(MAIN_METRICS):
        raise ValueError("paired statistics do not contain the four Top-10 risks")
    selected["raw_difference"] = selected[_difference_column(selected)].astype(float)
    lower_better = selected["direction"].eq("lower")
    selected["favorable_effect"] = np.where(
        lower_better,
        -selected["raw_difference"],
        selected["raw_difference"],
    )
    selected["favorable_ci_low"] = np.where(
        lower_better,
        -selected["bootstrap_ci_high"].astype(float),
        selected["bootstrap_ci_low"].astype(float),
    )
    selected["favorable_ci_high"] = np.where(
        lower_better,
        -selected["bootstrap_ci_low"].astype(float),
        selected["bootstrap_ci_high"].astype(float),
    )
    return selected


def validate_robustness_inputs(
    sensitivity: pd.DataFrame,
    rank_profile: pd.DataFrame,
) -> None:
    grid = sensitivity.loc[sensitivity["metric"].eq("difficulty_gap")]
    cells = set(zip(grid["scale"], grid["readiness_k"].astype(int)))
    expected = {
        (scale, readiness_k)
        for scale in ("p90", "p95", "max")
        for readiness_k in (3, 5, 10)
    }
    if cells != expected:
        raise ValueError("robustness input must contain nine difficulty settings")
    for metric in MAIN_METRICS:
        ranks = set(
            rank_profile.loc[rank_profile["metric"].eq(metric), "rank"].astype(int)
        )
        if ranks != set(range(1, 11)):
            raise ValueError(f"rank profile is incomplete for {metric}")


def _save_three_formats(fig, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        path = base.with_suffix(suffix)
        fig.savefig(
            path,
            dpi=450,
            bbox_inches="tight",
            pad_inches=0.015,
            facecolor="white",
        )
        outputs.append(path)
    return outputs


def draw_main_figure(
    paired: pd.DataFrame,
    model_summary: pd.DataFrame,
    base: Path,
) -> list[Path]:
    effects_by_comparison = {
        comparison: favorable_effect_rows(paired, comparison=comparison)
        for comparison, _, _, _ in COMPARISON_SPECS
    }
    exposure = model_summary.loc[
        model_summary["cutoff"].eq(10)
        & model_summary["model"].isin(("ckg_rl", "pcgnn", "cgrc"))
    ].set_index("model")
    if set(exposure.index) != {"ckg_rl", "pcgnn", "cgrc"}:
        raise ValueError("model summary lacks Top-10 cold exposure for all three models")

    configure_style()
    fig = plt.figure(figsize=(3.35, 3.02))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.65, 0.92), hspace=0.68)
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(MAIN_METRICS))[::-1]
    all_limits = []
    ax.axvline(0.0, color="#555555", lw=0.8, ls="--", zorder=1)
    for offset, (comparison, label, color_key, marker) in zip(
        (0.13, -0.13),
        COMPARISON_SPECS,
    ):
        oriented = effects_by_comparison[comparison]
        effects = oriented["favorable_effect"].to_numpy(dtype=float)
        low = oriented["favorable_ci_low"].to_numpy(dtype=float)
        high = oriented["favorable_ci_high"].to_numpy(dtype=float)
        all_limits.extend(low)
        all_limits.extend(high)
        ax.errorbar(
            effects,
            y + offset,
            xerr=np.vstack((effects - low, high - effects)),
            fmt=marker,
            ms=4.0,
            mfc=PALETTE[color_key],
            mec="white",
            mew=0.45,
            ecolor=PALETTE[color_key],
            elinewidth=1.05,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[metric] for metric in MAIN_METRICS])
    bound = max(0.042, float(np.max(np.abs(all_limits))) * 1.25)
    ax.set_xlim(-bound, bound)
    ax.set_xlabel("Favorable effect (positive = CKG-RL better)")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.5, ls=":", zorder=0)
    ax.tick_params(length=2.4, pad=1.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    ax.text(-0.17, 1.08, "(a)", transform=ax.transAxes, fontweight="bold", va="top")

    ax_b = fig.add_subplot(grid[1])
    labels = ("CKG-RL", "PCGNN", "CGRC")
    models = ("ckg_rl", "pcgnn", "cgrc")
    values = np.array([exposure.loc[model, "cold_proportion_mean"] for model in models])
    errors = np.array([exposure.loc[model, "cold_proportion_sd"] for model in models])
    bars = ax_b.barh(
        np.arange(3)[::-1],
        values,
        xerr=errors,
        height=0.50,
        color=tuple(PALETTE[model] for model in models),
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2.0, "ecolor": "#444444"},
    )
    for bar, value, error in zip(bars, values, errors):
        ax_b.text(
            value + error + 0.014,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=6.4,
        )
    ax_b.set_yticks(np.arange(3)[::-1])
    ax_b.set_yticklabels(labels)
    ax_b.set_xlim(0, max(0.62, float((values + errors).max()) + 0.08))
    ax_b.set_xlabel("Cold-course proportion in Top-10")
    ax_b.grid(axis="x", color=PALETTE["grid"], lw=0.5, ls=":", zorder=0)
    ax_b.tick_params(length=2.4, pad=1.5)
    ax_b.text(-0.17, 1.20, "(b)", transform=ax_b.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.37, right=0.98, top=0.97, bottom=0.14)
    outputs = _save_three_formats(fig, base)
    plt.close(fig)
    return outputs


def _favorable_from_raw(frame: pd.DataFrame) -> np.ndarray:
    raw = frame[_difference_column(frame)].to_numpy(dtype=float)
    lower = frame["direction"].eq("lower").to_numpy()
    return np.where(lower, -raw, raw)


def draw_robustness_figure(
    sensitivity: pd.DataFrame,
    rank_profile: pd.DataFrame,
    base: Path,
) -> list[Path]:
    validate_robustness_inputs(sensitivity, rank_profile)
    configure_style()
    fig = plt.figure(figsize=(3.35, 3.05))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.25), hspace=0.58)
    ax = fig.add_subplot(grid[0])
    difficulty = sensitivity.loc[sensitivity["metric"].eq("difficulty_gap")].copy()
    difficulty["favorable"] = -difficulty["mean_difference_ckg_rl_minus_cgrc"]
    matrix = np.empty((3, 3), dtype=float)
    for row, scale in enumerate(("p90", "p95", "max")):
        for column, readiness_k in enumerate((3, 5, 10)):
            value = difficulty.loc[
                difficulty["scale"].eq(scale)
                & difficulty["readiness_k"].eq(readiness_k),
                "favorable",
            ]
            matrix[row, column] = float(value.iloc[0])
    limit = max(0.005, float(np.abs(matrix).max()) * 1.12)
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="#111111",
            )
    ax.set_xticks(range(3), labels=("Top-3", "Top-5", "Top-10"))
    ax.set_yticks(range(3), labels=("P90", "P95", "Max"))
    ax.set_xlabel("Readiness depth")
    ax.set_ylabel("Complexity scale")
    ax.tick_params(length=0, pad=1.6)
    ax.text(-0.19, 1.12, "(a)", transform=ax.transAxes, fontweight="bold", va="top")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
    colorbar.set_label("Favorable difficulty effect", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    ax_b = fig.add_subplot(grid[1])
    metric_colors = {
        "prerequisite_gap": "#2F7D6D",
        "concept_continuity": "#B24C45",
        "difficulty_gap": "#7A6A9A",
        "structural_redundancy": "#2F6B9A",
    }
    markers = ("o", "s", "^", "D")
    ax_b.axhline(0.0, color="#555555", lw=0.8, ls="--", zorder=1)
    for marker, metric in zip(markers, MAIN_METRICS):
        rows = rank_profile.loc[rank_profile["metric"].eq(metric)].sort_values("rank")
        ax_b.plot(
            rows["rank"],
            _favorable_from_raw(rows),
            color=metric_colors[metric],
            marker=marker,
            ms=2.8,
            lw=1.0,
            label=METRIC_LABELS[metric],
        )
    ax_b.set_xlim(0.7, 10.3)
    ax_b.set_xticks((1, 2, 4, 6, 8, 10))
    ax_b.set_xlabel("Recommendation rank")
    ax_b.set_ylabel("Favorable effect")
    ax_b.grid(axis="y", color=PALETTE["grid"], lw=0.5, ls=":", zorder=0)
    ax_b.legend(ncol=2, loc="best", handlelength=1.5, columnspacing=0.9)
    ax_b.tick_params(length=2.4, pad=1.5)
    ax_b.text(-0.19, 1.09, "(b)", transform=ax_b.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.25, right=0.94, top=0.97, bottom=0.12)
    outputs = _save_three_formats(fig, base)
    plt.close(fig)
    return outputs


def draw_all(
    *,
    analysis_dir: Path,
    robustness_dir: Path,
    figure_dir: Path,
) -> list[Path]:
    analysis_dir = Path(analysis_dir)
    robustness_dir = Path(robustness_dir)
    figure_dir = Path(figure_dir)
    paired = pd.read_csv(analysis_dir / "paired_statistics.csv")
    model_summary = pd.read_csv(analysis_dir / "model_summary.csv")
    sensitivity = pd.read_csv(robustness_dir / "difficulty_sensitivity_paired.csv")
    rank_profile = pd.read_csv(robustness_dir / "rank_profile_paired.csv")
    outputs = draw_main_figure(
        paired,
        model_summary,
        figure_dir / "mooccube_p1_topk_motivation",
    )
    outputs.extend(
        draw_robustness_figure(
            sensitivity,
            rank_profile,
            figure_dir / "mooccube_p1_risk_robustness",
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures" / "p1_topk_motivation_analysis",
    )
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures" / "p1_risk_robustness",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures",
    )
    args = parser.parse_args()
    outputs = draw_all(
        analysis_dir=args.analysis_dir,
        robustness_dir=args.robustness_dir,
        figure_dir=args.figure_dir,
    )
    for output in outputs:
        print(f"[P1-FIG] wrote {output}")


if __name__ == "__main__":
    main()
