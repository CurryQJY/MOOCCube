"""Pilot experiment C: does baseline ranking failure line up with pedagogical risk?

We reuse the exact per-course data already produced by ``draw_method_motivation``:

* ``risk``     -- per (seed, course) neutral, method-free pedagogical signals.
* ``exposure`` -- per (seed, course) CGRC full-catalog cold NDCG@10.

Both are course-level, so we merge on ``[seed, item_id]`` and show that the
strongest content-graph baseline (CGRC) ranks pedagogically risky cold courses
systematically worse. Signals use the *raw* definitions (prerequisite coverage
gap, concept discontinuity, difficulty gap) rather than the method's gated
reward terms, so the diagnostic does not depend on CKG-RL's own knobs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.draw_method_motivation import (  # noqa: E402
    BLUE,
    DARK,
    GRID,
    RED,
    collect_data,
    configure_style,
)

PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
BASE = FIG_DIR / "mooccube_motivation_risk_vs_exposure"

# Each panel is oriented so that "more risk" points right, and we expect NDCG@10
# to fall. concept_gap = 1 - concept_match keeps the three panels visually
# consistent (higher x = more pedagogical mismatch).
RISK_PANELS = [
    ("prereq_gap", "Prerequisite gap\n(uncovered prereqs)"),
    ("concept_gap", "Concept discontinuity\n(1 - concept overlap)"),
    ("difficulty_gap", "Difficulty gap\n(too hard for learner)"),
]
N_BINS = 3
BIN_LABELS = ["Low", "Mid", "High"]


def _tercile_by_rank(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Assign equal-count buckets by rank, robust to ties/zero-inflation."""
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    edges = np.linspace(0, len(values), n_bins + 1)
    bucket = np.digitize(order, edges[1:-1], right=False)
    return bucket.astype(int)


def build_merged(risk: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    merged = risk.merge(
        exposure[["seed", "item_id", "count", "N@10"]],
        on=["seed", "item_id"],
        how="inner",
    )
    merged["concept_gap"] = (1.0 - merged["concept_match"]).clip(0.0, 1.0)
    return merged


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col, _ in RISK_PANELS:
        rho = float(merged[[col, "N@10"]].corr(method="spearman").iloc[0, 1])
        buckets = _tercile_by_rank(merged[col].to_numpy(dtype=float))
        low = float(merged.loc[buckets == 0, "N@10"].mean())
        high = float(merged.loc[buckets == N_BINS - 1, "N@10"].mean())
        ratio = float(low / high) if high > 1e-9 else float("inf")
        rows.append(
            {
                "signal": col,
                "spearman_rho_vs_ndcg10": rho,
                "ndcg10_low_risk_tercile": low,
                "ndcg10_high_risk_tercile": high,
                "low_over_high_ratio": ratio,
                "n_courses": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def draw(merged: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(RISK_PANELS),
        figsize=(3.35, 1.78),
        sharey=True,
        constrained_layout=False,
    )

    xpos = np.arange(N_BINS)
    for ax, (col, title) in zip(axes, RISK_PANELS):
        vals = merged[col].to_numpy(dtype=float)
        ndcg = merged["N@10"].to_numpy(dtype=float)
        buckets = _tercile_by_rank(vals)

        means = np.array([ndcg[buckets == b].mean() for b in range(N_BINS)])
        sems = np.array(
            [
                ndcg[buckets == b].std(ddof=1) / max(1.0, np.sqrt((buckets == b).sum()))
                for b in range(N_BINS)
            ]
        )
        # Bar shade darkens from low risk (blue) to high risk (red).
        colors = [BLUE, "#8a6f86", RED]
        ax.bar(
            xpos,
            means,
            width=0.66,
            color=colors,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            xpos,
            means,
            yerr=sems,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.7,
            capsize=2.0,
            zorder=4,
        )

        rho = float(merged[[col, "N@10"]].corr(method="spearman").iloc[0, 1])
        ax.text(
            0.5,
            0.94,
            rf"$\rho$={rho:+.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.6,
            color=DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        )

        ax.set_xticks(xpos)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel(title, fontsize=6.2, labelpad=2.0)
        ax.grid(axis="y", color=GRID, lw=0.55, linestyle="--", zorder=0)
        ax.tick_params(length=2.4, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("CGRC cold\nNDCG@10", fontsize=6.6)
    fig.text(
        0.5,
        0.995,
        f"Higher pedagogical risk -> worse cold-course exposure (n={len(merged)})",
        ha="center",
        va="top",
        fontsize=6.6,
        color=DARK,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.86, bottom=0.30, wspace=0.16)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(BASE.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    risk, exposure, _ = collect_data()
    merged = build_merged(risk, exposure)
    draw(merged)
    summary = summarize(merged)
    merged.to_csv(FIG_DIR / "mooccube_motivation_risk_vs_exposure_data.csv", index=False)
    summary.to_csv(FIG_DIR / "mooccube_motivation_risk_vs_exposure_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
