"""Figure 1 (revised motivation): ranking quality is decoupled from pedagogical quality.

This replaces the earlier baseline exposure-vs-ranking contrast, which a
per-course audit showed did not hold (CGRC dominated PCGNN on both exposure and
ranking, and the two axes were near-collinear, Spearman rho=0.98). The revised
motivation rests on a single, model-internal fact that survives that audit:
among cold courses that a strong recommender does rank well (high NDCG@10),
ranking quality is *positively* correlated with structural redundancy and
*orthogonal* to prerequisite safety. Optimizing click-relevance therefore does
not buy pedagogical quality -- it must be placed inside the objective.

Panels:
  (a) NDCG@10 vs structural redundancy  -- MISALIGNED (rho ~ +0.53)
  (b) NDCG@10 vs prerequisite gap       -- ORTHOGONAL  (rho ~ -0.15, n.s.)

All statistics are computed over cold courses that are actually exposed
(recall@10 > 0); non-exposed courses carry no ranked list to diagnose and are
reported separately in the text as exposure failures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper_aaai27" / "figures"
ANALYSIS_DIR = FIGURE_DIR / "validation_motivation_analysis"
DEFAULT_COURSE_PATH = ANALYSIS_DIR / "course_macro.csv"
DEFAULT_OUTPUT_BASE = FIGURE_DIR / "mooccube_pedagogical_decoupling"

MODEL_STYLE = {
    "pcgnn": {"label": "PCGNN", "color": "#1F5AA6", "marker": "s"},
    "cgrc": {"label": "CGRC", "color": "#7B2D3A", "marker": "o"},
}
GRID = "#D6D6D6"
DARK = "#252525"

# (column, panel title, x-axis label, expected relationship tag)
PANELS = [
    (
        "cold_structural_redundancy",
        "(a) Ranking quality rewards redundancy",
        "Structural redundancy (lower is better)",
        "misaligned",
    ),
    (
        "cold_prerequisite_gap",
        "(b) Ranking quality ignores prerequisites",
        "Prerequisite gap (lower is better)",
        "orthogonal",
    ),
]


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
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.2,
        }
    )


def load_exposed(course_path: Path, metric_k: int = 10) -> pd.DataFrame:
    df = pd.read_csv(course_path)
    df = df[df["cutoff"] == metric_k].copy()
    required = {
        "model",
        "seed",
        "target_item_id",
        "recall_at_10",
        "ndcg_at_10",
        "cold_structural_redundancy",
        "cold_prerequisite_gap",
    }
    if missing := required.difference(df.columns):
        raise ValueError(f"course input missing columns: {sorted(missing)}")
    if set(df["model"]) != set(MODEL_STYLE):
        raise ValueError("expected exactly PCGNN and CGRC rows")
    exposed = df[df["recall_at_10"] > 0].copy()
    if exposed.empty:
        raise ValueError("no exposed cold courses to plot")
    return exposed


def _annotate_corr(ax: plt.Axes, x: np.ndarray, y: np.ndarray, tag: str) -> None:
    rho, p = spearmanr(x, y)
    p_txt = "p < 0.001" if p < 1e-3 else f"p = {p:.2f}"
    label = {"misaligned": "misaligned", "orthogonal": "orthogonal"}[tag]
    ax.text(
        0.03,
        0.97,
        rf"$\rho = {rho:+.2f}$" + f"\n{p_txt}\n({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6, alpha=0.9),
    )


def _fit_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3:
        return
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xs, np.polyval(coef, xs), color=DARK, lw=1.0, ls="--", alpha=0.75, zorder=2)


def build_figure(exposed: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), constrained_layout=True)
    for ax, (col, title, xlabel, tag) in zip(axes, PANELS):
        x_all = exposed[col].to_numpy(dtype=float)
        y_all = exposed["ndcg_at_10"].to_numpy(dtype=float)
        for model, style in MODEL_STYLE.items():
            sub = exposed[exposed["model"] == model]
            ax.scatter(
                sub[col].to_numpy(dtype=float),
                sub["ndcg_at_10"].to_numpy(dtype=float),
                s=20,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.7,
                label=style["label"],
                zorder=3,
            )
        _fit_line(ax, x_all, y_all)
        _annotate_corr(ax, x_all, y_all, tag)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@10 (ranking quality)")
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor=GRID,
        handletextpad=0.3,
        borderpad=0.4,
    )
    return fig


def save_three_formats(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png", ".svg"):
        target = output_base.with_suffix(suffix)
        fig.savefig(target, dpi=300, bbox_inches="tight")
        outputs.append(target)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-path", type=Path, default=DEFAULT_COURSE_PATH)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    exposed = load_exposed(args.course_path)
    fig = build_figure(exposed)
    outputs = save_three_formats(fig, args.output_base)
    plt.close(fig)
    for path in outputs:
        print(f"[decoupling] wrote {path}")

    # Provenance: print the numbers that back the caption.
    for col, _title, _xlabel, tag in PANELS:
        rho, p = spearmanr(exposed[col], exposed["ndcg_at_10"])
        print(f"[decoupling] NDCG vs {col}: rho={rho:+.3f} p={p:.2g} ({tag}) n={len(exposed)}")


if __name__ == "__main__":
    main()
