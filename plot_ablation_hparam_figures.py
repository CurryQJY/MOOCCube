from __future__ import annotations

import argparse
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
        "font.size": 8.5,
        "axes.labelsize": 10.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.85,
        "axes.unicode_minus": False,
        "savefig.dpi": 600,
    }
)


SEEDS = (2025, 2026, 2027)

ABLATION_VARIANTS = [
    ("wo_course_reward", "w/o Reward", "#45A58C"),
    ("wo_course_candidate", "w/o Sampling", "#8EAD96"),
    ("wo_prereq_aux", "w/o Prereq Aux", "#EFC15D"),
    ("wo_all_course_signals", "w/o All Signals", "#D84B2A"),
    ("full", "Full", "#2F6FA3"),
]

CORE_ABLATION_VARIANTS = [
    ("full", "Full CKG-RL", "#2F6FA3", "latex_main_table"),
    ("wo_course_reward", "w/o Reward", "#45A58C", "experiment_summary"),
    ("wo_course_candidate", "w/o Sampling", "#8EAD96", "experiment_summary"),
    ("wo_prereq_aux", "w/o Prereq Aux", "#EFC15D", "experiment_summary"),
    ("wo_all_course_signals", "w/o All Knowledge", "#D84B2A", "experiment_summary"),
]

MAIN_TABLE_REFERENCE_VARIANTS = [
    ("cgrc_main", "CGRC ref.", "#8C8C8C", "latex_main_table"),
]

SIGNAL_ABLATION_VARIANTS = [
    ("full", "Full", "#2F6FA3", "latex_main_table"),
    ("wo_concept_match", "w/o Concept", "#45A58C", "experiment_summary"),
    ("wo_prereq_signal", "w/o Prereq", "#8EAD96", "experiment_summary"),
    ("wo_difficulty_signal", "w/o Difficulty", "#EFC15D", "experiment_summary"),
    ("wo_redundancy_signal", "w/o Redundancy", "#D84B2A", "experiment_summary"),
]

# Values copied from paper_wsdm/main.tex, Table~\ref{tab:main-item-cold}.
LATEX_MOOCCUBE_MAIN_TABLE = {
    "content_cbf": {
        "cold_r5_mean": 0.1525,
        "cold_r5_std": 0.0027,
        "cold_r10_mean": 0.1866,
        "cold_r10_std": 0.0017,
        "cold_r20_mean": 0.2319,
        "cold_r20_std": 0.0015,
        "cold_n5_mean": 0.1194,
        "cold_n5_std": 0.0064,
        "cold_n10_mean": 0.1303,
        "cold_n10_std": 0.0057,
        "cold_n20_mean": 0.1417,
        "cold_n20_std": 0.0059,
    },
    "cgrc_main": {
        "cold_r5_mean": 0.2121,
        "cold_r5_std": 0.0205,
        "cold_r10_mean": 0.2589,
        "cold_r10_std": 0.0171,
        "cold_r20_mean": 0.3141,
        "cold_r20_std": 0.0168,
        "cold_n5_mean": 0.1695,
        "cold_n5_std": 0.0190,
        "cold_n10_mean": 0.1845,
        "cold_n10_std": 0.0182,
        "cold_n20_mean": 0.1984,
        "cold_n20_std": 0.0178,
    },
    "full": {
        "cold_r5_mean": 0.2214,
        "cold_r5_std": 0.0126,
        "cold_r10_mean": 0.2667,
        "cold_r10_std": 0.0150,
        "cold_r20_mean": 0.3172,
        "cold_r20_std": 0.0092,
        "cold_n5_mean": 0.1818,
        "cold_n5_std": 0.0112,
        "cold_n10_mean": 0.1962,
        "cold_n10_std": 0.0115,
        "cold_n20_mean": 0.2090,
        "cold_n20_std": 0.0104,
    },
}

METRIC_LABELS = {
    "full_cold_item_macro_r10": "Cold item-macro Recall@10",
    "full_cold_item_macro_n10": "Cold item-macro NDCG@10",
    "full_cold_item_macro_r20": "Cold item-macro Recall@20",
    "full_cold_item_macro_n20": "Cold item-macro NDCG@20",
}


def save_pub(fig: mpl.figure.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def read_summary_row(summary_path: Path) -> pd.Series:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"Empty summary: {summary_path}")
    return frame.iloc[0]


def build_ablation_summary(root: Path) -> pd.DataFrame:
    rows = []
    for variant, label, color in ABLATION_VARIANTS:
        row = read_summary_row(root / variant / "fast3_static_multiseed_summary.csv")
        rows.append(
            {
                "variant": variant,
                "label": label,
                "color": color,
                "cold_r20_mean": float(row["full_cold_item_macro_r20_mean"]),
                "cold_n20_mean": float(row["full_cold_item_macro_n20_mean"]),
                "hot_r20_mean": float(row["full_hot_item_macro_r20_mean"]),
                "hot_n20_mean": float(row["full_hot_item_macro_n20_mean"]),
                "cold_r20_std": float(row["full_cold_item_macro_r20_std"]),
                "cold_n20_std": float(row["full_cold_item_macro_n20_std"]),
                "hot_r20_std": float(row["full_hot_item_macro_r20_std"]),
                "hot_n20_std": float(row["full_hot_item_macro_n20_std"]),
                "runs": int(row["runs"]),
                "seeds": str(row["seeds"]),
            }
        )
    return pd.DataFrame(rows)


def _cold_metric_values(row: pd.Series | dict[str, float], prefix: str = "") -> dict[str, float]:
    def value(name: str) -> float:
        return float(row[f"{prefix}{name}"])

    return {
        "cold_r5_mean": value("cold_r5_mean"),
        "cold_r5_std": value("cold_r5_std"),
        "cold_r10_mean": value("cold_r10_mean"),
        "cold_r10_std": value("cold_r10_std"),
        "cold_n5_mean": value("cold_n5_mean"),
        "cold_n5_std": value("cold_n5_std"),
        "cold_n10_mean": value("cold_n10_mean"),
        "cold_n10_std": value("cold_n10_std"),
        "cold_r20_mean": value("cold_r20_mean"),
        "cold_r20_std": value("cold_r20_std"),
        "cold_n20_mean": value("cold_n20_mean"),
        "cold_n20_std": value("cold_n20_std"),
    }


def _cold_item_macro_from_summary(row: pd.Series) -> dict[str, float]:
    return {
        "cold_r5_mean": float(row["full_cold_item_macro_r5_mean"]),
        "cold_r5_std": float(row["full_cold_item_macro_r5_std"]),
        "cold_r10_mean": float(row["full_cold_item_macro_r10_mean"]),
        "cold_r10_std": float(row["full_cold_item_macro_r10_std"]),
        "cold_n5_mean": float(row["full_cold_item_macro_n5_mean"]),
        "cold_n5_std": float(row["full_cold_item_macro_n5_std"]),
        "cold_n10_mean": float(row["full_cold_item_macro_n10_mean"]),
        "cold_n10_std": float(row["full_cold_item_macro_n10_std"]),
        "cold_r20_mean": float(row["full_cold_item_macro_r20_mean"]),
        "cold_r20_std": float(row["full_cold_item_macro_r20_std"]),
        "cold_n20_mean": float(row["full_cold_item_macro_n20_mean"]),
        "cold_n20_std": float(row["full_cold_item_macro_n20_std"]),
    }


def build_component_ablation_summary(
    root: Path,
    variants: list[tuple[str, str, str, str]],
    include_content_cbf_reference: bool = False,
) -> pd.DataFrame:
    rows = []
    for variant, label, color, source in variants:
        if source == "latex_main_table":
            metrics = _cold_metric_values(LATEX_MOOCCUBE_MAIN_TABLE[variant])
            runs = len(SEEDS)
            seeds = ",".join(str(seed) for seed in SEEDS)
        else:
            row = read_summary_row(root / variant / "fast3_static_multiseed_summary.csv")
            metrics = _cold_item_macro_from_summary(row)
            runs = int(row["runs"])
            seeds = str(row["seeds"])
        rows.append(
            {
                "variant": variant,
                "label": label,
                "color": color,
                "source": source,
                "runs": runs,
                "seeds": seeds,
                **metrics,
            }
        )

    if include_content_cbf_reference:
        rows.append(
            {
                "variant": "content_cbf",
                "label": "Content-CBF ref.",
                "color": "#B7B7B7",
                "source": "latex_main_table",
                "runs": len(SEEDS),
                "seeds": ",".join(str(seed) for seed in SEEDS),
                **_cold_metric_values(LATEX_MOOCCUBE_MAIN_TABLE["content_cbf"]),
            }
        )
    return pd.DataFrame(rows)


def _zoomed_ylim(values: np.ndarray, pad_ratio: float = 0.18) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 1e-4)
    return lo - span * pad_ratio, hi + span * pad_ratio


def _apply_fine_y_ticks(ax: mpl.axes.Axes, major_step: float = 0.005) -> None:
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_step))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))


def plot_ablation_reference_style(summary: pd.DataFrame, out_base: Path) -> None:
    groups = ["Cold", "Hot"]
    panels = [
        ("Recall@20", ["cold_r20_mean", "hot_r20_mean"]),
        ("NDCG@20", ["cold_n20_mean", "hot_n20_mean"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.95, 2.85), sharex=True)
    x = np.arange(len(groups))
    width = 0.14
    offsets = (np.arange(len(summary)) - (len(summary) - 1) / 2.0) * width

    for ax, (ylabel, cols) in zip(axes, panels):
        all_values = []
        for offset, row in zip(offsets, summary.itertuples(index=False)):
            values = [getattr(row, cols[0]), getattr(row, cols[1])]
            all_values.extend(values)
            ax.bar(
                x + offset,
                values,
                width=width,
                color=row.color,
                edgecolor="#1E1E1E",
                linewidth=0.85,
                label=row.label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_ylim(*_zoomed_ylim(np.asarray(all_values)))
        _apply_fine_y_ticks(ax)
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
        fancybox=False,
        edgecolor="#D5D5D5",
        columnspacing=0.75,
        handlelength=1.9,
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.99, wspace=0.34)
    save_pub(fig, out_base)
    plt.close(fig)


def plot_mooccube_component_ablation(
    summary: pd.DataFrame,
    out_base: Path,
) -> None:
    panels = [
        ("Recall@5", "cold_r5_mean"),
        ("Recall@10", "cold_r10_mean"),
        ("Recall@20", "cold_r20_mean"),
        ("NDCG@5", "cold_n5_mean"),
        ("NDCG@10", "cold_n10_mean"),
        ("NDCG@20", "cold_n20_mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15))
    axes = axes.ravel()
    x = np.arange(len(summary))
    width = 0.68

    for ax, (title, mean_col) in zip(axes, panels):
        means = summary[mean_col].astype(float).to_numpy()
        colors = summary["color"].tolist()
        ax.bar(
            x,
            means,
            width=width,
            color=colors,
            edgecolor="#1E1E1E",
            linewidth=0.75,
        )
        ax.set_title(title, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_ylim(*_zoomed_ylim(means, pad_ratio=0.18))
        _apply_fine_y_ticks(ax)
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
        ncol=min(3, len(labels)),
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fancybox=False,
        edgecolor="#D5D5D5",
        columnspacing=0.9,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.075, right=0.995, wspace=0.34, hspace=0.42)
    save_pub(fig, out_base)
    plt.close(fig)


def write_latex_rows(summary: pd.DataFrame, path: Path) -> None:
    def cell(mean: float, std: float, bold: bool = False) -> str:
        text = f"{mean:.4f}\\sd{{{std:.4f}}}"
        return f"\\textbf{{{mean:.4f}}}\\sd{{{std:.4f}}}" if bold else text

    lines = []
    for row in summary.itertuples(index=False):
        bold = row.variant == "full"
        lines.append(
            f"{row.label} & "
            f"{cell(row.cold_r10_mean, row.cold_r10_std, bold)} & "
            f"{cell(row.cold_n10_mean, row.cold_n10_std, bold)} & "
            f"{cell(row.cold_r20_mean, row.cold_r20_std, bold)} & "
            f"{cell(row.cold_n20_mean, row.cold_n20_std, bold)} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_hparam_summary(points_path: Path) -> pd.DataFrame:
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    points = pd.read_csv(points_path)
    beta_map = {"sample_beta_0p10": 0.10, "main_default": 0.20}
    points = points.loc[points["variant"].isin(beta_map)].copy()
    points["beta"] = points["variant"].map(beta_map).astype(float)
    rows = []
    for (metric, beta), group in points.groupby(["metric", "beta"], sort=True):
        values = group["value"].astype(float)
        rows.append(
            {
                "metric": metric,
                "metric_label": str(group["metric_label"].iloc[0]),
                "beta": float(beta),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "beta"]).reset_index(drop=True)


def build_hparam_points(points_path: Path) -> pd.DataFrame:
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    points = pd.read_csv(points_path)
    beta_map = {"sample_beta_0p10": 0.10, "main_default": 0.20}
    points = points.loc[points["variant"].isin(beta_map)].copy()
    points["beta"] = points["variant"].map(beta_map).astype(float)
    points["seed_label"] = points["seed"].astype(int).map(lambda seed: f"Seed {seed}")
    return points.sort_values(["metric", "seed", "beta"]).reset_index(drop=True)


def plot_hparam_beta_reference_style(points: pd.DataFrame, out_base: Path) -> None:
    panels = [
        ("full_cold_item_macro_r10", "Recall@10"),
        ("full_cold_item_macro_n10", "NDCG@10"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.95, 2.85))
    seed_styles = {
        2025: ("#1E5AA6", "o"),
        2026: ("#D94848", "s"),
        2027: ("#19A957", "D"),
    }

    for ax, (metric, ylabel) in zip(axes, panels):
        sub = points.loc[points["metric"] == metric].copy()
        all_values = sub["value"].astype(float).to_numpy()
        for seed in sorted(sub["seed"].astype(int).unique()):
            seed_sub = sub.loc[sub["seed"].astype(int) == seed].sort_values("beta")
            color, marker = seed_styles.get(seed, ("#777777", "o"))
            ax.plot(
                seed_sub["beta"].astype(float),
                seed_sub["value"].astype(float),
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.25,
                label=f"Seed {seed}",
            )
        ax.set_xticks([0.10, 0.20])
        ax.set_xticklabels(["0.10", "0.20"])
        ax.set_xlabel("Course Sample Beta", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.55, color="#BFBFBF", alpha=0.85)
        ax.set_axisbelow(True)
        ax.set_ylim(*_zoomed_ylim(all_values, pad_ratio=0.20))
        _apply_fine_y_ticks(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.03),
        frameon=True,
        fancybox=False,
        edgecolor="#D5D5D5",
        handlelength=2.1,
        columnspacing=1.1,
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.10, right=0.99, wspace=0.34)
    save_pub(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ablation and hyperparameter figures in reference style.")
    parser.add_argument("--out-dir", default="output/figures/ablation_hparam_reference_style")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ablation_root = Path("outputs/content_delta_pop5/course_ablation_e60_3seed")
    hparam_points = Path(
        "outputs/content_delta_pop5/course_hparam_sensitivity_e60_3seed/"
        "figures/mooccube_hparam_sensitivity_preview_partial_points.csv"
    )

    ablation = build_ablation_summary(ablation_root)
    core_ablation = build_component_ablation_summary(ablation_root, CORE_ABLATION_VARIANTS)
    signal_ablation = build_component_ablation_summary(
        Path("outputs/content_delta_pop5/course_signal_ablation_e60_3seed"),
        SIGNAL_ABLATION_VARIANTS,
    )
    core_ablation_with_refs = build_component_ablation_summary(
        ablation_root,
        CORE_ABLATION_VARIANTS + MAIN_TABLE_REFERENCE_VARIANTS,
        include_content_cbf_reference=True,
    )
    hparam = build_hparam_summary(hparam_points)
    hparam_seed_points = build_hparam_points(hparam_points)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(out_dir / "mooccube_ablation_reference_style_summary.csv", index=False)
    core_ablation.to_csv(out_dir / "mooccube_core_ablation_latex_main_aligned_summary.csv", index=False)
    signal_ablation.to_csv(out_dir / "mooccube_signal_ablation_summary.csv", index=False)
    core_ablation_with_refs.to_csv(out_dir / "mooccube_core_ablation_with_main_table_refs.csv", index=False)
    write_latex_rows(
        core_ablation_with_refs,
        out_dir / "mooccube_core_ablation_with_main_table_refs_latex_rows.tex",
    )
    write_latex_rows(
        signal_ablation,
        out_dir / "mooccube_signal_ablation_latex_rows.tex",
    )
    hparam.to_csv(out_dir / "mooccube_hparam_beta_reference_style_summary.csv", index=False)
    hparam_seed_points.to_csv(out_dir / "mooccube_hparam_beta_reference_style_points.csv", index=False)
    plot_ablation_reference_style(ablation, out_dir / "mooccube_ablation_reference_style")
    plot_mooccube_component_ablation(core_ablation, out_dir / "mooccube_core_ablation_reference_style")
    plot_mooccube_component_ablation(signal_ablation, out_dir / "mooccube_signal_ablation_reference_style")
    plot_hparam_beta_reference_style(hparam_seed_points, out_dir / "mooccube_hparam_beta_reference_style")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
