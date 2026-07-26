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


RISK_METRICS = (
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
COMPARISONS = (
    (
        "ckg_rl_vs_ckg_rl_wo_course_reward",
        "Full vs w/o course reward",
        "#2F6B9A",
        "o",
    ),
    (
        "ckg_rl_vs_ckg_rl_wo_simulator",
        "Full vs w/o simulator",
        "#C4772E",
        "s",
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 7.0,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.1,
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


def favorable_mechanism_rows(
    paired: pd.DataFrame,
    *,
    comparison: str,
    cold_only: bool,
) -> pd.DataFrame:
    metrics = tuple(
        f"cold_{metric}" if cold_only else metric for metric in RISK_METRICS
    )
    selected = paired.loc[
        paired["cutoff"].eq(10)
        & paired["comparison"].eq(comparison)
        & paired["metric"].isin(metrics)
    ].copy()
    selected["display_metric"] = selected["metric"].str.removeprefix("cold_")
    selected["display_metric"] = pd.Categorical(
        selected["display_metric"],
        categories=RISK_METRICS,
        ordered=True,
    )
    selected = selected.sort_values("display_metric").reset_index(drop=True)
    if selected["display_metric"].astype(str).tolist() != list(RISK_METRICS):
        raise ValueError("mechanism statistics do not contain the four Top-10 risks")

    selected["raw_difference"] = selected["mean_difference"].astype(float)
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


def draw_mechanism_figure(paired: pd.DataFrame, base: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.18), sharey=True)
    all_intervals = []
    for axis, cold_only, title in zip(
        axes,
        (False, True),
        ("All Top-10 recommendations", "Recommended cold courses only"),
    ):
        y = np.arange(len(RISK_METRICS))[::-1]
        axis.axvline(0.0, color="#555555", lw=0.8, ls="--", zorder=1)
        for offset, (comparison, label, color, marker) in zip(
            (0.13, -0.13),
            COMPARISONS,
        ):
            rows = favorable_mechanism_rows(
                paired,
                comparison=comparison,
                cold_only=cold_only,
            )
            effect = rows["favorable_effect"].to_numpy(dtype=float)
            low = rows["favorable_ci_low"].to_numpy(dtype=float)
            high = rows["favorable_ci_high"].to_numpy(dtype=float)
            all_intervals.extend(low)
            all_intervals.extend(high)
            axis.errorbar(
                effect,
                y + offset,
                xerr=np.vstack((effect - low, high - effect)),
                fmt=marker,
                ms=4.2,
                mfc=color,
                mec="white",
                mew=0.45,
                ecolor=color,
                elinewidth=1.05,
                capsize=2.0,
                label=label,
                zorder=3,
            )
        axis.set_yticks(y)
        axis.set_yticklabels([METRIC_LABELS[metric] for metric in RISK_METRICS])
        axis.set_title(title, pad=5.0)
        axis.set_xlabel("Favorable mechanism effect (positive = full better)")
        axis.grid(axis="x", color="#D8D8D8", lw=0.5, ls=":", zorder=0)
        axis.tick_params(length=2.4, pad=1.5)

    bound = max(0.018, float(np.max(np.abs(all_intervals))) * 1.15)
    for axis in axes:
        axis.set_xlim(-bound, bound)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    axes[0].text(-0.20, 1.12, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.08, 1.12, "(b)", transform=axes[1].transAxes, fontweight="bold")
    figure.subplots_adjust(left=0.19, right=0.99, top=0.80, bottom=0.23, wspace=0.18)
    outputs = _save_three_formats(figure, Path(base))
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=(
            ROOT
            / "paper_aaai27"
            / "figures"
            / "p1_motivation_mechanism_analysis"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures",
    )
    args = parser.parse_args()
    paired = pd.read_csv(args.analysis_dir / "paired_statistics.csv")
    outputs = draw_mechanism_figure(
        paired,
        args.figure_dir / "mooccube_p1_motivation_mechanisms",
    )
    for output in outputs:
        print(f"[P1-MECHANISM-FIG] wrote {output}")


if __name__ == "__main__":
    main()
