from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO = PAPER_DIR.parent
PAPER_FIG_DIR = PAPER_DIR / "figures"
OUT_DIR = PAPER_DIR / "figures" / "provisional_updated"

MAIN_TEX = PAPER_DIR / "main.tex"
ABLATION_TREND = REPO / "ablation_hparam_reference_style" / "mooccube_complete_ablation_core_reference_style_summary.csv"
HPARAM_TREND = REPO / "outputs" / "content_delta_pop5" / "course_hparam_validation_figures" / "mooccube_hparam_validation_summary.csv"
CURRENT_VALIDATION = (
    REPO
    / "outputs"
    / "significance_per_item_exports"
    / "mooccube"
    / "ckg_rl_true_true_hparam_grid"
    / "validation_figures"
    / "mooccube_hparam_validation_summary.csv"
)

MAIN_METRICS = [
    ("R@5", "cold_r5_mean", "main_r5"),
    ("R@10", "cold_r10_mean", "main_r10"),
    ("R@20", "cold_r20_mean", "main_r20"),
    ("N@5", "cold_n5_mean", "main_n5"),
    ("N@10", "cold_n10_mean", "main_n10"),
    ("N@20", "cold_n20_mean", "main_n20"),
]

ABLATION_COLORS = {
    "full": "#2F6FA3",
    "wo_course_reward": "#45A58C",
    "wo_course_candidate": "#8EAD96",
    "wo_prereq_aux": "#EFC15D",
    "wo_all_course_signals": "#D84B2A",
}

ABLATION_HATCHES = {
    "full": "",
    "wo_course_reward": "",
    "wo_course_candidate": "////",
    "wo_prereq_aux": "\\\\\\\\",
    "wo_all_course_signals": "...",
}

COMPACT_ABLATION_LABELS = {
    "Full CKG-RL": "Full CKG-RL",
    "No educational rewards": "w/o Edu. Rewards",
    "No knowledge-guided sampler": "w/o KG Sampling",
    "No prerequisite auxiliary loss": "w/o Prereq. Aux.",
    "No course-knowledge inputs": "w/o Course Signals",
}


class CenterGridLegendHandle:
    pass


class CenterGridLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=ABLATION_COLORS["full"],
            edgecolor="#1E1E1E",
            linewidth=0.48,
            transform=trans,
        )
        artists = [rect]
        xmid = xdescent + width / 2.0
        artists.append(
            Line2D(
                [xmid, xmid],
                [ydescent, ydescent + height],
                color="#1E1E1E",
                linewidth=0.95,
                solid_capstyle="butt",
                transform=trans,
            )
        )
        ymid = ydescent + height / 2.0
        artists.append(
            Line2D(
                [xdescent, xdescent + width],
                [ymid, ymid],
                color="#1E1E1E",
                linewidth=0.95,
                solid_capstyle="butt",
                transform=trans,
            )
        )
        return artists


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.4,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "hatch.linewidth": 0.85,
            "axes.unicode_minus": False,
            "savefig.dpi": 600,
        }
    )


def save_all(fig: mpl.figure.Figure, out_base: Path, tight_pdf: bool = False) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_base.with_suffix(".pdf")
    save_kwargs = {"dpi": 300}
    if tight_pdf:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.012})
    else:
        # Keep standalone PDF pages as a stable canvas with visible margins, so
        # the plot sits centered on the page instead of being tightly cropped.
        save_kwargs.update({"dpi": 300})
    try:
        fig.savefig(pdf_path, **save_kwargs)
    except PermissionError:
        fallback = out_base.with_name(out_base.name + "_layout_fixed").with_suffix(".pdf")
        fig.savefig(fallback, **save_kwargs)
        print(f"PDF destination locked; wrote {fallback}")
    for suffix in (".svg", ".png"):
        fig.savefig(out_base.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.012)


def _clean_latex_numbers(line: str) -> list[float]:
    line = re.sub(r"\\sigmark\{[^}]*\}", "", line)
    line = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", line)
    line = re.sub(r"\\underline\{([^{}]+)\}", r"\1", line)
    return [float(x) for x in re.findall(r"\b\d+\.\d+\b", line)]


def _clean_latex_mean_numbers(line: str) -> list[float]:
    line = re.sub(r"\\sd\{[^}]*\}", "", line)
    return _clean_latex_numbers(line)


def read_mooccube_ckg_main_values() -> dict[str, float]:
    text = MAIN_TEX.read_text(encoding="utf-8")
    out: dict[str, float] | None = None
    for line in text.splitlines():
        if r"\textbf{CKG-RL}" in line:
            values = _clean_latex_numbers(line)
            if len(values) < 4:
                raise ValueError("Could not parse MOOCCube CKG-RL values from main table.")
            out = {
                "main_r5": values[0],
                "main_n5": values[1],
                "main_r10": values[2],
                "main_n10": values[3],
            }
            break
    if out is None:
        raise ValueError("Could not find CKG-RL row in main.tex.")

    supplement = PAPER_DIR / "supplement_tables.tex"
    if supplement.exists():
        for line in supplement.read_text(encoding="utf-8").splitlines():
            if r"\textbf{CKG-RL}" in line:
                values = _clean_latex_mean_numbers(line)
                if len(values) >= 6:
                    out["main_r20"] = values[2]
                    out["main_n20"] = values[5]
                    return out
    full_old = pd.read_csv(ABLATION_TREND).loc[lambda df: df["variant"].eq("full")].iloc[0]
    out["main_r20"] = float(full_old["cold_r20_mean"])
    out["main_n20"] = float(full_old["cold_n20_mean"])
    return out


def build_provisional_ablation() -> pd.DataFrame:
    current = read_mooccube_ckg_main_values()
    trend = pd.read_csv(ABLATION_TREND)
    full_old = trend.loc[trend["variant"].eq("full")].iloc[0]

    rows: list[dict[str, object]] = []
    for row in trend.itertuples(index=False):
        out: dict[str, object] = {
            "variant": row.variant,
            "label": row.label,
            "display_label": COMPACT_ABLATION_LABELS.get(str(row.label), str(row.label)),
            "source": "current main-table full + previous ablation drop trend",
        }
        for metric_label, old_col, main_key in MAIN_METRICS:
            old_full = float(full_old[old_col])
            old_value = float(row._asdict()[old_col])
            drop = old_full - old_value
            out[f"{metric_label}_previous_full"] = old_full
            out[f"{metric_label}_previous_value"] = old_value
            out[f"{metric_label}_drop_from_previous_full"] = drop
            out[f"{metric_label}_provisional_value"] = current[main_key] - drop
            out[f"{metric_label}_current_full"] = current[main_key]
        rows.append(out)
    return pd.DataFrame(rows)


def _score_axis(values: np.ndarray) -> tuple[float, float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 0.003)
    if span <= 0.008:
        step = 0.002
    elif span <= 0.014:
        step = 0.003
    else:
        step = 0.005
    pad = max(step * 0.75, span * 0.18)
    lower = np.floor((lo - pad) / step) * step
    upper = np.ceil((hi + pad) / step) * step
    return max(0.0, float(lower)), float(upper), step


def _draw_center_grid_bar(ax: mpl.axes.Axes, bar: mpl.patches.Rectangle) -> None:
    x0 = float(bar.get_x())
    x1 = x0 + float(bar.get_width())
    xmid = (x0 + x1) / 2.0
    y0 = max(float(ax.get_ylim()[0]), float(bar.get_y()))
    y1 = float(bar.get_y() + bar.get_height())
    if y1 <= y0:
        return

    def add_line(xs: list[float], ys: list[float]) -> None:
        line = Line2D(xs, ys, color="#1E1E1E", linewidth=0.85, solid_capstyle="butt", zorder=4)
        line.set_clip_path(bar)
        ax.add_line(line)

    add_line([xmid, xmid], [y0, y1])
    for y in np.linspace(y0, y1, 11)[1:-1]:
        add_line([x0, x1], [float(y), float(y)])


def plot_ablation(table: pd.DataFrame, out_base: Path, include_embedded_caption: bool = True) -> None:
    panels = [
        ("R@5", "R@5_provisional_value"),
        ("R@10", "R@10_provisional_value"),
        ("R@20", "R@20_provisional_value"),
        ("N@5", "N@5_provisional_value"),
        ("N@10", "N@10_provisional_value"),
        ("N@20", "N@20_provisional_value"),
    ]
    x = np.arange(len(table), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(4.45, 4.85))
    axes = axes.ravel()
    legend_handles = []
    legend_labels = []
    for row in table.itertuples(index=False):
        if str(row.variant) == "full":
            legend_handles.append(CenterGridLegendHandle())
        else:
            legend_handles.append(
                Patch(
                    facecolor=ABLATION_COLORS.get(str(row.variant), "#777777"),
                    edgecolor="#1E1E1E",
                    linewidth=0.48,
                    hatch=ABLATION_HATCHES.get(str(row.variant), ""),
                )
            )
        legend_labels.append(str(row.display_label))

    for ax, (title, col) in zip(axes, panels):
        values = table[col].astype(float).to_numpy()
        full_bars: list[mpl.patches.Rectangle] = []
        for idx, row in enumerate(table.itertuples(index=False)):
            bars = ax.bar(
                x[idx],
                values[idx],
                width=0.56,
                color=ABLATION_COLORS.get(str(row.variant), "#777777"),
                edgecolor="#1E1E1E",
                linewidth=0.48,
                hatch=ABLATION_HATCHES.get(str(row.variant), ""),
                label=str(row.display_label),
            )
            if str(row.variant) == "full":
                full_bars.append(bars[0])
        lower, upper, step = _score_axis(values)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        for bar in full_bars:
            _draw_center_grid_bar(ax, bar)
        ax.set_title(title, fontweight="bold", fontsize=10.8, pad=2)
        ax.set_xticks([])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.42, color="#C7C7C7", alpha=0.78)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        ncol=3,
        frameon=False,
        handlelength=1.55,
        handleheight=1.05,
        columnspacing=0.95,
        labelspacing=0.22,
        borderaxespad=0.0,
        fontsize=6.7,
        handler_map={CenterGridLegendHandle: CenterGridLegendHandler()},
    )
    if include_embedded_caption:
        fig.text(
            0.5,
            0.115,
            "Figure 3: Ablation study on key components of CKG-RL.",
            fontsize=9.8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    fig.text(0.045, 0.590, "Recall", rotation=90, fontsize=13.8, fontweight="bold", va="center", ha="center")
    fig.text(0.045, 0.315, "NDCG", rotation=90, fontsize=13.8, fontweight="bold", va="center", ha="center")
    fig.subplots_adjust(
        top=0.790,
        bottom=0.205 if include_embedded_caption else 0.145,
        left=0.145,
        right=0.985,
        wspace=0.43,
        hspace=0.58,
    )
    save_all(fig, out_base, tight_pdf=not include_embedded_caption)
    plt.close(fig)


def read_current_default_validation() -> tuple[float, float]:
    if CURRENT_VALIDATION.exists():
        current = pd.read_csv(CURRENT_VALIDATION)
        default = current[(current["variant"] == "main_default") & (current["sweep"] == "beta")]
        if not default.empty:
            row = default.iloc[0]
            return float(row["val_r10_mean"]), float(row["val_n10_mean"])

    # Fallback: compute from the current default validation metric logs.
    rows = []
    for path in (REPO / "outputs" / "significance_per_item_exports" / "mooccube" / "ckg_rl_full").rglob(
        "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"
    ):
        frame = pd.read_csv(path)
        idx = pd.to_numeric(frame["Val_full_cold_N@10"], errors="coerce").idxmax()
        best = frame.loc[idx]
        rows.append(
            {
                "r10": float(best["Val_full_cold_R@10"]),
                "n10": float(best["Val_full_cold_N@10"]),
            }
        )
    if not rows:
        raise ValueError("No current default validation metrics found.")
    frame = pd.DataFrame(rows)
    return float(frame["r10"].mean()), float(frame["n10"].mean())


def build_provisional_hparam() -> pd.DataFrame:
    trend = pd.read_csv(HPARAM_TREND)
    current_r10, current_n10 = read_current_default_validation()
    default = trend[(trend["variant"] == "main_default") & (trend["sweep"] == "beta")].iloc[0]
    old_r10 = float(default["val_r10_mean"])
    old_n10 = float(default["val_n10_mean"])

    rows = []
    for row in trend.itertuples(index=False):
        record = row._asdict()
        rows.append(
            {
                **record,
                "val_r10_previous_mean": float(record["val_r10_mean"]),
                "val_n10_previous_mean": float(record["val_n10_mean"]),
                "val_r10_previous_std": float(record["val_r10_std"]),
                "val_n10_previous_std": float(record["val_n10_std"]),
                "val_r10_delta_from_previous_default": float(record["val_r10_mean"]) - old_r10,
                "val_n10_delta_from_previous_default": float(record["val_n10_mean"]) - old_n10,
                "val_r10_provisional_mean": current_r10 + float(record["val_r10_mean"]) - old_r10,
                "val_n10_provisional_mean": current_n10 + float(record["val_n10_mean"]) - old_n10,
                "current_default_val_r10": current_r10,
                "current_default_val_n10": current_n10,
                "source": "current default validation + previous validation sweep trend",
            }
        )
    return pd.DataFrame(rows)


def _default_score(rows: pd.DataFrame, x: float, col: str) -> float | None:
    match = rows[rows["x"].astype(float).eq(float(x))]
    if match.empty:
        return None
    return float(match.iloc[0][col])


def _style_axis(ax) -> None:
    ax.grid(axis="y", color="#D6D6D6", lw=0.48, linestyle="--", zorder=0)
    ax.tick_params(axis="both", labelsize=6.2, length=2.2, pad=1.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_hparam(summary: pd.DataFrame, out_base: Path, include_embedded_caption: bool = True) -> None:
    sweeps = [
        ("beta", r"Knowledge-sampling weight $\beta$", 0.20),
        ("reward", "Reward-shaping scale", 1.00),
        ("horizon", r"Simulator horizon $T$", 5.00),
    ]
    metric_specs = [
        ("Recall@10", "val_r10_provisional_mean", "#1F5AA6", "o"),
        ("NDCG@10", "val_n10_provisional_mean", "#C9403A", "s"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(3.70, 4.85), sharey=False)

    for row_idx, (sweep_key, sweep_title, default_x) in enumerate(sweeps):
        sweep_rows = summary[summary["sweep"] == sweep_key].sort_values("point_order")
        x_values = sweep_rows["x"].astype(float).tolist()
        for col_idx, (metric_label, mean_col, color, marker) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            y_values = sweep_rows[mean_col].astype(float).tolist()
            ax.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                markersize=3.0,
                linewidth=1.12,
                markeredgewidth=0.0,
                zorder=3,
            )
            default_y = _default_score(sweep_rows, default_x, mean_col)
            if default_y is not None:
                ax.axvline(default_x, color="#666666", lw=0.75, linestyle=":")

            _style_axis(ax)
            ymin = min(y_values)
            ymax = max(y_values)
            pad = max(0.0009, (ymax - ymin) * 0.18)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            if col_idx == 0:
                ax.text(
                    0.0,
                    1.07,
                    sweep_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold",
                )
            if row_idx == 2:
                ax.set_xlabel("Sweep value", fontsize=6.8, labelpad=2)
            ax.set_xticks(x_values)
            if sweep_key in {"beta", "reward"}:
                ax.set_xticklabels([f"{float(x):.2f}" for x in x_values], rotation=55, ha="right", fontsize=5.0)
            else:
                ax.set_xticklabels([f"{int(float(x))}" for x in x_values], fontsize=6.2)

    fig.supylabel("Validation score", x=0.045, fontsize=7.0, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color="#1F5AA6", marker="o", lw=1.12, markersize=3.6, label="Recall@10"),
        Line2D([0], [0], color="#C9403A", marker="s", lw=1.12, markersize=3.6, label="NDCG@10"),
        Line2D([0], [0], color="#666666", lw=0.75, linestyle=":", label="Default"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.088 if include_embedded_caption else 0.060),
        ncol=3,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.45,
        borderaxespad=0.0,
        fontsize=6.4,
    )
    if include_embedded_caption:
        fig.text(
            0.5,
            0.038,
            "Figure 4: Hyper-parameter sensitivity on MOOCCube.",
            fontsize=9.4,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    fig.subplots_adjust(left=0.20, right=0.94, top=0.91, bottom=0.18, wspace=0.40, hspace=0.90)
    save_all(fig, out_base, tight_pdf=not include_embedded_caption)
    plt.close(fig)


def main() -> None:
    setup_style()
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ablation = build_provisional_ablation()
    ablation.to_csv(OUT_DIR / "mooccube_ablation_updated_main_table_provisional.csv", index=False)
    plot_ablation(ablation, OUT_DIR / "mooccube_ablation_updated_main_table_provisional", include_embedded_caption=True)
    plot_ablation(ablation, PAPER_FIG_DIR / "mooccube_ablation_updated_main_table_aaai_main", include_embedded_caption=False)

    hparam = build_provisional_hparam()
    hparam.to_csv(OUT_DIR / "mooccube_hparam_validation_updated_default_provisional.csv", index=False)
    plot_hparam(hparam, OUT_DIR / "mooccube_hparam_validation_updated_default_provisional", include_embedded_caption=True)
    plot_hparam(hparam, PAPER_FIG_DIR / "mooccube_hparam_validation_updated_default_aaai_main", include_embedded_caption=False)

    print(f"Wrote {OUT_DIR / 'mooccube_ablation_updated_main_table_provisional.pdf'}")
    print(f"Wrote {OUT_DIR / 'mooccube_hparam_validation_updated_default_provisional.pdf'}")
    print(f"Wrote {PAPER_FIG_DIR / 'mooccube_ablation_updated_main_table_aaai_main.pdf'}")
    print(f"Wrote {PAPER_FIG_DIR / 'mooccube_hparam_validation_updated_default_aaai_main.pdf'}")
    print(f"Wrote data CSVs under {OUT_DIR}")


if __name__ == "__main__":
    main()
