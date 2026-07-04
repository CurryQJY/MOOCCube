from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "mooccube_hparam_sensitivity_single_column.pdf"
OUT_PNG = ROOT / "figures" / "mooccube_hparam_sensitivity_single_column.png"

DEFAULT = {"beta": 0.20, "reward": 1.00, "horizon": 5}

SWEEPS = [
    {
        "key": "beta",
        "title": r"Knowledge-sampling weight $\beta$",
        "x": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "r10": [0.2667, 0.2691, 0.2716, 0.2694, 0.2667, 0.2636, 0.2659, 0.2685, 0.2700],
        "n10": [0.1965, 0.1974, 0.2004, 0.1977, 0.1962, 0.1933, 0.1960, 0.1948, 0.1973],
    },
    {
        "key": "reward",
        "title": "Reward-shaping scale",
        "x": [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
        "r10": [0.2663, 0.2626, 0.2654, 0.2661, 0.2667, 0.2665, 0.2659, 0.2651],
        "n10": [0.1951, 0.1938, 0.1958, 0.1945, 0.1962, 0.1953, 0.1964, 0.1952],
    },
    {
        "key": "horizon",
        "title": r"Simulator horizon $T$",
        "x": [1, 3, 5, 7, 10],
        "r10": [0.2661, 0.2666, 0.2667, 0.2659, 0.2568],
        "n10": [0.1966, 0.1967, 0.1962, 0.1950, 0.1903],
    },
]


def default_score(x_values, y_values, default_x):
    return y_values[x_values.index(default_x)]


def style_axis(ax):
    ax.grid(axis="y", color="#d6d6d6", lw=0.55, linestyle="--", zorder=0)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(3.25, 4.35),
        sharey=False,
        constrained_layout=False,
    )
    colors = {"r10": "#1f5aa6", "n10": "#c9403a"}

    for row, sweep in enumerate(SWEEPS):
        x = sweep["x"]
        series = [
            ("r10", "Recall@10", sweep["r10"]),
            ("n10", "NDCG@10", sweep["n10"]),
        ]
        for col, (key, metric, y) in enumerate(series):
            ax = axes[row][col]
            default_x = DEFAULT[sweep["key"]]
            default_y = default_score(x, y, default_x)
            ax.plot(
                x,
                y,
                color=colors[key],
                marker="o" if key == "r10" else "s",
                markersize=3.2,
                linewidth=1.35,
                markeredgewidth=0.0,
            )
            ax.scatter(
                [default_x],
                [default_y],
                s=20,
                facecolors="white",
                edgecolors=colors[key],
                linewidths=1.05,
                zorder=4,
            )
            ax.axvline(default_x, color="#666666", lw=0.8, linestyle=":")
            style_axis(ax)
            panel_min = min(y)
            panel_max = max(y)
            pad = max(0.0007, (panel_max - panel_min) * 0.18)
            ax.set_ylim(panel_min - pad, panel_max + pad)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.text(
                0.97,
                0.93,
                metric,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=colors[key],
                fontsize=6.8,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
            )
            if col == 0:
                ax.text(
                    0.0,
                    1.06,
                    sweep["title"],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold",
                )
            if row == 2:
                ax.set_xlabel("Sweep value", fontsize=7, labelpad=2)
            if sweep["key"] == "beta":
                ax.set_xticks([0.0, 0.2, 0.4])
            elif sweep["key"] == "reward":
                ax.set_xticks([0.0, 1.0, 2.0])
            else:
                ax.set_xticks([1, 5, 10])

    fig.supylabel("Raw score", x=0.02, fontsize=7.2, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.99, top=0.93, bottom=0.08, wspace=0.33, hspace=0.62)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
