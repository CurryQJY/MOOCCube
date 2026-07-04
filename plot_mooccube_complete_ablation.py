from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.8,
        "axes.labelsize": 9.4,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "savefig.dpi": 600,
    }
)


METRICS = [
    ("cold_r5", "R@5"),
    ("cold_n5", "N@5"),
    ("cold_r10", "R@10"),
    ("cold_n10", "N@10"),
    ("cold_r20", "R@20"),
    ("cold_n20", "N@20"),
]


GROUPS = [
    (
        "Core component ablations",
        [
            ("wo_course_reward", "No educational rewards", "course_ablation_e60_3seed/wo_course_reward"),
            (
                "wo_course_candidate",
                "No knowledge-guided sampler",
                "course_ablation_e60_3seed/wo_course_candidate",
            ),
            ("wo_prereq_aux", "No prerequisite auxiliary loss", "course_ablation_e60_3seed/wo_prereq_aux"),
            (
                "wo_all_course_signals",
                "No course-knowledge inputs",
                "course_ablation_e60_3seed/wo_all_course_signals",
            ),
        ],
    ),
    (
        "Course-knowledge signal ablations",
        [
            ("wo_concept_match", "No concept-matching reward", "signal_summary"),
            ("wo_prereq_signal", "No prerequisite-readiness signal", "signal_summary"),
            ("wo_difficulty_signal", "No difficulty-adaptation signal", "signal_summary"),
            ("wo_redundancy_signal", "No redundancy-control penalty", "signal_summary"),
        ],
    ),
    (
        "Mechanism diagnostics",
        [
            (
                "wo_forced_cold_masking",
                "No forced-cold ID masking",
                "course_core_ablation_e60_3seed/wo_forced_cold_masking",
            ),
            (
                "wo_simulator_t0",
                "No simulator rollout",
                "course_core_ablation_e60_3seed/wo_simulator_t0",
            ),
        ],
    ),
]


REFERENCE_GROUP_FIGURES = [
    (
        "Core component ablations",
        "mooccube_complete_ablation_core_reference_style",
        [
            ("full", "Full CKG-RL", "#2F6FA3"),
            ("wo_course_reward", "No educational rewards", "#45A58C"),
            ("wo_course_candidate", "No knowledge-guided sampler", "#8EAD96"),
            ("wo_prereq_aux", "No prerequisite auxiliary loss", "#EFC15D"),
            ("wo_all_course_signals", "No course-knowledge inputs", "#D84B2A"),
        ],
    ),
    (
        "Course-knowledge signal ablations",
        "mooccube_complete_ablation_signal_reference_style",
        [
            ("full", "Full CKG-RL", "#2F6FA3"),
            ("wo_concept_match", "No concept-matching reward", "#45A58C"),
            ("wo_prereq_signal", "No prerequisite-readiness signal", "#8EAD96"),
            ("wo_difficulty_signal", "No difficulty-adaptation signal", "#EFC15D"),
            ("wo_redundancy_signal", "No redundancy-control penalty", "#D84B2A"),
        ],
    ),
    (
        "Mechanism diagnostics",
        "mooccube_complete_ablation_mechanism_reference_style",
        [
            ("full", "Full CKG-RL", "#2F6FA3"),
            ("wo_forced_cold_masking", "No forced-cold ID masking", "#45A58C"),
            ("wo_simulator_t0", "No simulator rollout", "#D84B2A"),
        ],
    ),
]


def save_pub(fig: mpl.figure.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def read_summary(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty summary file: {path}")
    return frame.iloc[0]


def item_macro_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "cold_r5": float(row["full_cold_item_macro_r5_mean"]),
        "cold_n5": float(row["full_cold_item_macro_n5_mean"]),
        "cold_r10": float(row["full_cold_item_macro_r10_mean"]),
        "cold_n10": float(row["full_cold_item_macro_n10_mean"]),
        "cold_r20": float(row["full_cold_item_macro_r20_mean"]),
        "cold_n20": float(row["full_cold_item_macro_n20_mean"]),
    }


def signal_summary_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "cold_r5": float(row["cold_r5_mean"]),
        "cold_n5": float(row["cold_n5_mean"]),
        "cold_r10": float(row["cold_r10_mean"]),
        "cold_n10": float(row["cold_n10_mean"]),
        "cold_r20": float(row["cold_r20_mean"]),
        "cold_n20": float(row["cold_n20_mean"]),
    }


def build_complete_table(repo: Path) -> pd.DataFrame:
    result_root = repo / "outputs" / "content_delta_pop5"
    figure_root = repo / "ablation_hparam_reference_style"
    full_row = read_summary(result_root / "course_ablation_e60_3seed" / "full" / "fast3_static_multiseed_summary.csv")
    full = item_macro_metrics(full_row)
    signal_summary = pd.read_csv(figure_root / "mooccube_signal_ablation_summary.csv").set_index("variant")

    rows: list[dict[str, object]] = []
    for group, variants in GROUPS:
        for variant, label, source in variants:
            if source == "signal_summary":
                metrics = signal_summary_metrics(signal_summary.loc[variant])
            else:
                row = read_summary(result_root / source / "fast3_static_multiseed_summary.csv")
                metrics = item_macro_metrics(row)
            out = {"group": group, "variant": variant, "label": label}
            for metric, _ in METRICS:
                out[f"{metric}_mean"] = metrics[metric]
                out[f"{metric}_drop"] = full[metric] - metrics[metric]
                out[f"{metric}_rel_drop_pct"] = (full[metric] - metrics[metric]) / full[metric] * 100.0
                out[f"{metric}_full"] = full[metric]
            rows.append(out)
    return pd.DataFrame(rows)


def plot_complete_delta_heatmap(table: pd.DataFrame, out_base: Path) -> None:
    labels = table["label"].tolist()
    groups = table["group"].tolist()
    raw_drop = table[[f"{metric}_drop" for metric, _ in METRICS]].to_numpy(dtype=float)
    full_values = [float(table[f"{metric}_full"].iloc[0]) for metric, _ in METRICS]

    vmax = float(np.nanmax(np.abs(raw_drop)))
    vmax = max(0.0085, np.ceil(vmax * 10000.0) / 10000.0)
    vmin = -vmax

    display_rows: list[np.ndarray] = []
    display_labels: list[str] = []
    header_rows: list[int] = []
    row_map: list[int | None] = []
    start = 0
    for group in dict.fromkeys(groups):
        header_rows.append(len(display_rows))
        display_rows.append(np.full(len(METRICS), np.nan))
        display_labels.append(group)
        row_map.append(None)
        for idx in range(start, len(groups)):
            if groups[idx] != group:
                start = idx
                break
            display_rows.append(raw_drop[idx])
            display_labels.append(labels[idx])
            row_map.append(idx)
        else:
            start = len(groups)

    drop = np.vstack(display_rows)
    masked = np.ma.masked_invalid(drop)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")

    fig, ax = plt.subplots(figsize=(7.2, 4.95), constrained_layout=True)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([f"{name}\nFull={base:.4f}" for (_, name), base in zip(METRICS, full_values)])
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_yticklabels(display_labels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0)

    for tick, row in zip(ax.get_yticklabels(), row_map):
        if row is None:
            tick.set_fontweight("bold")
            tick.set_color("#333333")
        else:
            tick.set_color("#111111")

    for i in range(drop.shape[0]):
        if np.isnan(drop[i]).all():
            continue
        for j in range(drop.shape[1]):
            value = drop[i, j]
            color = "white" if abs(value) > vmax * 0.58 else "#1A1A1A"
            ax.text(
                j,
                i,
                f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=color,
            )

    for header in header_rows:
        ax.axhline(header - 0.5, color="#333333", linewidth=0.45)
        ax.axhline(header + 0.5, color="#FFFFFF", linewidth=0.8)

    ax.set_title(
        "MOOCCube Component Ablation under Strict Item-Cold Full-Ranking Evaluation",
        fontsize=9.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Primary cold item-macro metrics")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.08)
    cbar.set_label("Drop from Full CKG-RL (positive = performance decreases after removal)")
    cbar.ax.tick_params(labelsize=7.0)

    ax.text(
        0.0,
        -0.23,
        "Cell values are mean differences over three seeds; positive cells support the removed component.",
        fontsize=7.1,
        color="#333333",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    save_pub(fig, out_base)
    plt.close(fig)


def plot_complete_absolute_bars(table: pd.DataFrame, out_base: Path) -> None:
    group_names = list(dict.fromkeys(table["group"].tolist()))
    metric_pairs = [
        (("cold_r5_mean", "R@5"), ("cold_n5_mean", "N@5")),
        (("cold_r10_mean", "R@10"), ("cold_n10_mean", "N@10")),
        (("cold_r20_mean", "R@20"), ("cold_n20_mean", "N@20")),
    ]
    colors = ["#2F6FA3", "#45A58C", "#8EAD96", "#EFC15D", "#D84B2A", "#7E6AA8"]

    fig, axes = plt.subplots(
        len(group_names),
        len(metric_pairs),
        figsize=(7.2, 6.2),
        constrained_layout=True,
        sharey=False,
    )
    if len(group_names) == 1:
        axes = np.array([axes])

    for row_idx, group in enumerate(group_names):
        sub = table[table["group"] == group].copy()
        x = np.arange(len(sub))
        for col_idx, ((r_col, r_name), (n_col, n_name)) in enumerate(metric_pairs):
            ax = axes[row_idx, col_idx]
            width = 0.36
            ax.bar(
                x - width / 2,
                sub[r_col],
                width,
                color=colors[0],
                edgecolor="#222222",
                linewidth=0.45,
                label="Recall" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.bar(
                x + width / 2,
                sub[n_col],
                width,
                color=colors[3],
                edgecolor="#222222",
                linewidth=0.45,
                label="NDCG" if row_idx == 0 and col_idx == 0 else None,
            )
            vals = np.r_[sub[r_col].to_numpy(dtype=float), sub[n_col].to_numpy(dtype=float)]
            ymin = np.floor((vals.min() - 0.006) * 1000) / 1000
            ymax = np.ceil((vals.max() + 0.006) * 1000) / 1000
            ax.set_ylim(max(0, ymin), ymax)
            r_key = r_col.replace("_mean", "_full")
            n_key = n_col.replace("_mean", "_full")
            ax.axhline(
                float(sub[r_key].iloc[0]),
                color=colors[0],
                linestyle="--",
                linewidth=0.75,
                alpha=0.85,
            )
            ax.axhline(
                float(sub[n_key].iloc[0]),
                color=colors[3],
                linestyle=":",
                linewidth=0.85,
                alpha=0.95,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(sub["label"], rotation=35, ha="right", rotation_mode="anchor")
            ax.set_title(f"{r_name} / {n_name}", fontsize=8.5, fontweight="bold")
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, linestyle="--")
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(group, fontweight="bold")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "MOOCCube Component Ablation: Absolute Cold Item-Macro Scores",
        y=1.045,
        fontsize=9.6,
        fontweight="bold",
    )
    save_pub(fig, out_base)
    plt.close(fig)


def _reference_group_table(
    table: pd.DataFrame,
    group_name: str,
    variants: list[tuple[str, str, str]],
) -> pd.DataFrame:
    full_values = {
        f"{metric}_mean": float(table[f"{metric}_full"].iloc[0])
        for metric, _ in METRICS
    }
    full_row = {
        "variant": "full",
        "label": "Full CKG-RL",
        "color": "#2F6FA3",
        **full_values,
    }

    source = table.loc[table["group"] == group_name].set_index("variant")
    rows = [full_row]
    for variant, label, color in variants:
        if variant == "full":
            continue
        if variant not in source.index:
            raise KeyError(f"Missing variant {variant!r} in {group_name!r}")
        row = source.loc[variant]
        rows.append(
            {
                "variant": variant,
                "label": label,
                "color": color,
                **{f"{metric}_mean": float(row[f"{metric}_mean"]) for metric, _ in METRICS},
            }
        )
    return pd.DataFrame(rows)


def _fine_tick_step(values: np.ndarray) -> float:
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= 0.004:
        return 0.001
    if span <= 0.010:
        return 0.002
    return 0.005


def _zoomed_ylim_for_step(values: np.ndarray, step: float) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, step * 2.0)
    pad = max(step, span * 0.24)
    lower = np.floor((lo - pad) / step) * step
    upper = np.ceil((hi + pad) / step) * step
    return max(0.0, float(lower)), float(upper)


def plot_group_reference_style(summary: pd.DataFrame, out_base: Path) -> None:
    panels = [
        ("Recall@5", "cold_r5_mean"),
        ("Recall@10", "cold_r10_mean"),
        ("Recall@20", "cold_r20_mean"),
        ("NDCG@5", "cold_n5_mean"),
        ("NDCG@10", "cold_n10_mean"),
        ("NDCG@20", "cold_n20_mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.35))
    axes = axes.ravel()
    x = np.arange(len(summary))
    width = 0.68 if len(summary) >= 5 else 0.54

    for ax, (title, mean_col) in zip(axes, panels):
        means = summary[mean_col].astype(float).to_numpy()
        ax.bar(
            x,
            means,
            width=width,
            color=summary["color"].tolist(),
            edgecolor="#1E1E1E",
            linewidth=0.68,
        )
        step = _fine_tick_step(means)
        ax.set_ylim(*_zoomed_ylim_for_step(means, step))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        ax.set_title(title, fontsize=9.4, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_ylabel("Score" if ax in (axes[0], axes[3]) else "", fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="#BFBFBF", alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)

    handles = [
        mpl.patches.Patch(facecolor=row.color, edgecolor="#1E1E1E", label=row.label)
        for row in summary.itertuples(index=False)
    ]
    labels = [row.label for row in summary.itertuples(index=False)]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fancybox=False,
        columnspacing=1.25,
        handleheight=1.25,
        handlelength=1.95,
        labelspacing=0.70,
        borderaxespad=0.0,
        prop={"size": 11.4},
    )
    fig.subplots_adjust(top=0.68, bottom=0.085, left=0.065, right=0.995, wspace=0.27, hspace=0.40)
    save_pub(fig, out_base)
    plt.close(fig)


def _single_column_label(label: str) -> str:
    compact = {
        "Full CKG-RL": "Full CKG-RL",
        "No educational rewards": "No edu. rewards",
        "No knowledge-guided sampler": "No knowledge sampler",
        "No prerequisite auxiliary loss": "No prereq. aux. loss",
        "No course-knowledge inputs": "No course knowledge",
        "No concept-matching reward": "No concept match",
        "No prerequisite-readiness signal": "No prereq. readiness",
        "No difficulty-adaptation signal": "No difficulty adapt.",
        "No redundancy-control penalty": "No redundancy penalty",
        "No forced-cold ID masking": "No cold-ID mask",
        "No simulator rollout": "No simulator rollout",
    }
    return compact.get(label, label)


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


def plot_group_single_column_bars(summary: pd.DataFrame, out_base: Path) -> None:
    """Single-column ablation bars showing the full model and ablated variants."""
    metric_sets = [
        ("Recall", [("R@5", "cold_r5_mean"), ("R@10", "cold_r10_mean"), ("R@20", "cold_r20_mean")]),
        ("NDCG", [("N@5", "cold_n5_mean"), ("N@10", "cold_n10_mean"), ("N@20", "cold_n20_mean")]),
    ]
    hatches = ["++", "", "///", "\\\\\\", "...", "xxx"]

    if "full" in set(summary["variant"]):
        full = summary.loc[summary["variant"] == "full"]
        variants = pd.concat([full, summary.loc[summary["variant"] != "full"]], ignore_index=True)
    else:
        variants = summary.reset_index(drop=True)
    if len(variants) < 2:
        raise ValueError("Single-column ablation bars require the full model and at least one ablated variant.")

    x = np.arange(len(variants), dtype=float)
    width = 0.72 if len(variants) <= 3 else 0.66
    fig, axes = plt.subplots(2, 3, figsize=(3.35, 4.12))

    legend_handles = []
    legend_labels = []
    for row_idx, (ylabel, panels) in enumerate(metric_sets):
        for col_idx, (title, col) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            values = variants[col].astype(float).to_numpy()
            for idx, row in enumerate(variants.itertuples(index=False)):
                label = _single_column_label(row.label)
                bar = ax.bar(
                    x[idx],
                    values[idx],
                    width=width,
                    color=row.color,
                    edgecolor="#1E1E1E",
                    linewidth=0.50,
                    hatch=hatches[idx % len(hatches)],
                    label=label,
                )[0]
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(bar)
                    legend_labels.append(label)
            lower, upper, step = _score_axis(values)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
            ax.set_title(title, fontsize=8.2, fontweight="bold", pad=2)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6.2, pad=1.5)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.42, color="#C7C7C7", alpha=0.75)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontweight="bold", labelpad=4)
            else:
                ax.set_ylabel("")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.85,
        borderaxespad=0.0,
        labelspacing=0.24,
        prop={"size": 6.6},
    )
    fig.text(
        0.17,
        0.012,
        "Bars show absolute scores; each panel uses a local zoomed y-axis.",
        fontsize=6.5,
        color="#333333",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.17, right=0.99, wspace=0.34, hspace=0.34)
    save_pub(fig, out_base)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent
    out_dir = repo / "ablation_hparam_reference_style"
    table = build_complete_table(repo)
    table.to_csv(out_dir / "mooccube_complete_ablation_summary.csv", index=False)
    plot_complete_delta_heatmap(table, out_dir / "mooccube_complete_ablation_delta_heatmap")
    plot_complete_absolute_bars(table, out_dir / "mooccube_complete_ablation_absolute_bars")
    for group_name, filename, variants in REFERENCE_GROUP_FIGURES:
        summary = _reference_group_table(table, group_name, variants)
        summary.to_csv(out_dir / f"{filename}_summary.csv", index=False)
        plot_group_reference_style(summary, out_dir / filename)
        plot_group_single_column_bars(summary, out_dir / f"{filename}_single_column")
    print(out_dir / "mooccube_complete_ablation_delta_heatmap.png")
    print(out_dir / "mooccube_complete_ablation_absolute_bars.png")
    for _, filename, _ in REFERENCE_GROUP_FIGURES:
        print(out_dir / f"{filename}.png")
        print(out_dir / f"{filename}_single_column.png")


if __name__ == "__main__":
    main()
