"""Pilot experiment C (final figure): concept continuity structures cold-course exposure.

Single, decisive motivation panel. We reuse the merged per-course table from
``draw_motivation_risk_vs_exposure`` (neutral raw signals x CGRC cold NDCG@10)
and show the one axis that strongly predicts baseline failure: concept overlap
between the cold course and the learner's history. This directly motivates
placing concept signals inside retrieval and reward design.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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
from paper_aaai27.scripts.draw_motivation_risk_vs_exposure import (  # noqa: E402
    _tercile_by_rank,
    build_merged,
)

FIG_DIR = ROOT / "paper_aaai27" / "figures"
BASE = FIG_DIR / "mooccube_motivation_concept_exposure"
N_QUANTILE = 5


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    edges = np.linspace(0, len(values), n_bins + 1)
    return np.digitize(order, edges[1:-1], right=False).astype(int)


def draw(merged) -> dict[str, float]:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x = merged["concept_match"].to_numpy(dtype=float)
    y = merged["N@10"].to_numpy(dtype=float)

    rho = float(merged[["concept_match", "N@10"]].corr(method="spearman").iloc[0, 1])
    terc = _tercile_by_rank(x)
    top = float(y[terc == 2].mean())
    bot = float(y[terc == 0].mean())
    ratio = float(top / bot) if bot > 1e-9 else float("inf")

    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=False)

    # Low-continuity "no-exposure" risk zone (bottom concept-overlap tercile).
    thr = float(np.sort(x)[len(x) // 3])
    ax.axvspan(x.min() - 0.01, thr, color=RED, alpha=0.06, lw=0, zorder=0)

    ax.scatter(x, y, s=8, color="#5a6b8c", alpha=0.32, linewidth=0, zorder=2)

    # Quintile trend: mean N@10 per equal-count concept-overlap bin.
    qb = _quantile_bins(x, N_QUANTILE)
    bx = np.array([x[qb == b].mean() for b in range(N_QUANTILE)])
    by = np.array([y[qb == b].mean() for b in range(N_QUANTILE)])
    bsem = np.array(
        [y[qb == b].std(ddof=1) / max(1.0, np.sqrt((qb == b).sum())) for b in range(N_QUANTILE)]
    )
    ax.plot(bx, by, color=BLUE, lw=1.6, marker="o", ms=4.5, mfc="white", mec=BLUE, zorder=4)
    ax.errorbar(bx, by, yerr=bsem, fmt="none", ecolor=BLUE, elinewidth=0.8, capsize=2.2, zorder=3)

    # Linear trend line for visual slope.
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color=DARK, lw=0.9, ls="--", alpha=0.7, zorder=3)

    ax.text(
        0.035,
        0.955,
        rf"Spearman $\rho$ = {rho:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
    )
    ax.text(
        0.965,
        0.06,
        f"Top vs bottom third:\nNDCG@10 {top:.2f} vs {bot:.2f} ({ratio:.1f}x)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "pad": 2.2},
    )
    ax.text(
        thr / 2 if thr > 0 else 0.01,
        0.52,
        "low continuity\n(no exposure)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        color=RED,
    )

    ax.set_xlabel("Concept overlap of cold course with learner history")
    ax.set_ylabel("CGRC cold-course NDCG@10")
    ax.set_xlim(x.min() - 0.01, x.max() + 0.01)
    ax.set_ylim(bottom=-0.01)
    ax.grid(axis="y", color=GRID, lw=0.55, linestyle="--", zorder=0)
    ax.tick_params(length=2.4, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.text(
        0.5,
        0.995,
        f"Concept continuity structures cold-course exposure (CGRC, n={len(x)})",
        ha="center",
        va="top",
        fontsize=6.8,
        color=DARK,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.915, bottom=0.155)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(BASE.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return {"spearman_rho": rho, "top_tercile_ndcg10": top, "bottom_tercile_ndcg10": bot, "ratio": ratio, "n": len(x)}


def main() -> None:
    risk, exposure, _ = collect_data()
    merged = build_merged(risk, exposure)
    stats = draw(merged)
    for key, val in stats.items():
        print(f"{key}: {val}")
    print(f"Wrote {BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
