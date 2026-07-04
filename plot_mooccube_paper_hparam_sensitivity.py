from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


METRICS = [
    ("full_cold_item_macro_r10", "Cold item-macro Recall@10"),
    ("full_cold_item_macro_n10", "Cold item-macro NDCG@10"),
]


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    root: Path


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def read_final(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle))


def collect_points(variants: list[Variant], expected_seeds: set[int], allow_partial: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order, variant in enumerate(variants):
        finals = sorted(variant.root.rglob("final_fullrank_usim_feedback_fast3_content_delta_static.csv"))
        for final_path in finals:
            seed = seed_from_path(final_path)
            if seed is None or seed not in expected_seeds:
                continue
            record = read_final(final_path)
            for metric, metric_label in METRICS:
                rows.append(
                    {
                        "variant": variant.name,
                        "label": variant.label,
                        "variant_order": order,
                        "seed": seed,
                        "metric": metric,
                        "metric_label": metric_label,
                        "value": float(record[metric]),
                        "source_file": str(final_path),
                    }
                )
    points = pd.DataFrame(rows)
    if points.empty:
        raise SystemExit("No final metric files found.")

    complete_keys: list[tuple[str, str]] = []
    for (variant, metric), group in points.groupby(["variant", "metric"]):
        seeds = set(int(x) for x in group["seed"].unique())
        if allow_partial or seeds == expected_seeds:
            complete_keys.append((variant, metric))

    keep = pd.Series(False, index=points.index)
    for variant, metric in complete_keys:
        keep |= (points["variant"] == variant) & (points["metric"] == metric)
    filtered = points[keep].copy()
    if filtered.empty:
        raise SystemExit(
            "No complete variants found. Re-run with --allow-partial for a diagnostic preview."
        )
    return filtered


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    summary = (
        points.groupby(["variant_order", "variant", "label", "metric", "metric_label"], as_index=False)
        .agg(
            n=("value", "size"),
            seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.unique()))),
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .sort_values(["metric", "variant_order"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def plot_sensitivity(points: pd.DataFrame, summary: pd.DataFrame, out_base: Path, title_suffix: str) -> None:
    label_rows = summary[["variant_order", "variant", "label"]].drop_duplicates().sort_values("variant_order")
    labels = label_rows["label"].tolist()
    label_counts = (
        points.groupby("label")["seed"]
        .nunique()
        .reindex(labels)
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    y_pos = np.arange(len(labels))
    label_to_y = {label: idx for idx, label in enumerate(labels)}

    colors = {
        "main_default": "#4C566A",
        "sample_beta_0p10": "#0072B2",
        "sample_beta_0p30": "#56B4E9",
        "reward_scale_0p5": "#009E73",
        "reward_scale_0p75": "#56A773",
        "reward_scale_1p25": "#D99A3D",
        "reward_scale_1p5": "#E69F00",
        "prereq_gate_0p35": "#D55E00",
        "prereq_gate_0p70": "#A64218",
    }

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.2, 3.8), sharey=True)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (metric, metric_label) in zip(axes, METRICS):
        metric_points = points[points["metric"] == metric]
        metric_summary = summary[summary["metric"] == metric]
        baseline = metric_summary[metric_summary["variant"] == "main_default"]
        baseline_mean = float(baseline["mean"].iloc[0]) if not baseline.empty else None

        for _, row in metric_points.iterrows():
            y = label_to_y[row["label"]]
            seed = int(row["seed"])
            jitter = ((seed % 10) - 5) * 0.012
            ax.scatter(
                row["value"],
                y + jitter,
                s=26,
                color=colors.get(str(row["variant"]), "#333333"),
                alpha=0.68,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )

        for _, row in metric_summary.iterrows():
            y = label_to_y[row["label"]]
            ax.errorbar(
                row["mean"],
                y,
                xerr=row["std"] if row["n"] > 1 else 0,
                fmt="D",
                markersize=4.6,
                color=colors.get(str(row["variant"]), "#333333"),
                ecolor=colors.get(str(row["variant"]), "#333333"),
                elinewidth=1.1,
                capsize=2.4,
                zorder=4,
            )
        if baseline_mean is not None:
            ax.axvline(baseline_mean, color="#777777", lw=0.9, ls=(0, (3, 2)), zorder=1)

        ax.set_title(metric_label, fontsize=9)
        ax.set_xlabel("Full-ranking test score", fontsize=8)
        ax.grid(axis="x", color="#D7D7D7", lw=0.55, alpha=0.75)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"{label} (n={label_counts[label]})" for label in labels], fontsize=7.3)
    axes[0].invert_yaxis()
    fig.suptitle(f"MOOCCube main-table hyperparameter sensitivity{title_suffix}", fontsize=10)
    fig.text(
        0.02,
        0.02,
        "Points: individual seeds; diamonds: mean; bars: SD. Dashed line: main-table default.",
        fontsize=6.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=600 if suffix == ".tiff" else 300, bbox_inches="tight")
    plt.close(fig)


def _tight_metric_ylim(values: list[float]) -> tuple[float, float]:
    ymin, ymax = min(values), max(values)
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax) * 0.01, 0.001)
    pad = max(span * 0.35, 0.00065)
    return ymin - pad, ymax + pad


def plot_line_grid_sensitivity(summary: pd.DataFrame, out_base: Path, title_suffix: str) -> None:
    metric_specs = [
        ("full_cold_item_macro_r10", "Recall@10", "#1F5A9F", "o"),
        ("full_cold_item_macro_n10", "NDCG@10", "#E54843", "s"),
    ]
    panels = [
        (
            "Course Sample beta",
            "Course Sample \u03b2",
            [
                ("sample_beta_0p10", "0.10"),
                ("main_default", "0.20"),
                ("sample_beta_0p30", "0.30"),
            ],
        ),
        (
            "Reward Scale",
            "Reward Scale",
            [
                ("reward_scale_0p5", "0.5x"),
                ("reward_scale_0p75", "0.75x"),
                ("main_default", "1.0x"),
                ("reward_scale_1p25", "1.25x"),
                ("reward_scale_1p5", "1.5x"),
            ],
        ),
        (
            "Prerequisite Gate",
            "Prerequisite Gate",
            [
                ("main_default", "0.20"),
                ("prereq_gate_0p35", "0.35"),
                ("prereq_gate_0p70", "0.70"),
            ],
        ),
    ]

    lookup = {
        (str(row.variant), str(row.metric)): row
        for row in summary.itertuples(index=False)
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2, 3.15))
    outer_grid = fig.add_gridspec(1, len(panels), wspace=0.38)

    legend_handles = []
    for panel_idx, (panel_title, x_label, variants) in enumerate(panels):
        inner_grid = outer_grid[0, panel_idx].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.12)
        ax_top = fig.add_subplot(inner_grid[0])
        ax_bottom = fig.add_subplot(inner_grid[1], sharex=ax_top)
        metric_axes = {
            metric_specs[0][0]: ax_top,
            metric_specs[1][0]: ax_bottom,
        }
        metric_values = {metric: [] for metric, _, _, _ in metric_specs}

        xs = np.arange(len(variants))
        xlabels = [label for _, label in variants]

        for metric, metric_label, color, marker in metric_specs:
            ax = metric_axes[metric]
            y_values: list[float] = []
            x_values: list[int] = []
            for x, (variant, _) in enumerate(variants):
                row = lookup.get((variant, metric))
                if row is None:
                    continue
                x_values.append(x)
                y_values.append(float(row.mean))
            if not x_values:
                continue
            (line,) = ax.plot(
                x_values,
                y_values,
                marker=marker,
                markersize=5.2,
                linewidth=1.35,
                color=color,
                label=metric_label,
            )
            if panel_title == panels[0][0]:
                legend_handles.append(line)
            metric_values[metric].extend(y_values)

        for ax, (metric, metric_label, _, _) in zip((ax_top, ax_bottom), metric_specs):
            if metric_values[metric]:
                ax.set_ylim(*_tight_metric_ylim(metric_values[metric]))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.55, color="#A8A8A8", alpha=0.65)
            ax.tick_params(axis="y", labelsize=7.4)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            ax.set_ylabel(metric_label, fontsize=9.2, fontweight="bold", labelpad=5)

        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_bottom.tick_params(axis="x", length=0)
        ax_bottom.set_xticks(xs)
        ax_bottom.set_xticklabels(xlabels, fontsize=8)
        ax_bottom.set_xlabel(x_label, fontsize=10, fontweight="bold", labelpad=5)

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        bbox_to_anchor=(0.5, 1.0),
        prop={"weight": "bold", "size": 10},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.82)

    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=600 if suffix == ".tiff" else 300, bbox_inches="tight")
    plt.close(fig)


def make_delta_points(points: pd.DataFrame) -> pd.DataFrame:
    baseline = points[points["variant"] == "main_default"][
        ["seed", "metric", "value"]
    ].rename(columns={"value": "baseline_value"})
    delta = points.merge(baseline, on=["seed", "metric"], how="left")
    delta["delta"] = delta["value"] - delta["baseline_value"]
    return delta


def plot_delta_grid_sensitivity(points: pd.DataFrame, out_base: Path, title_suffix: str) -> None:
    metric_specs = [
        ("full_cold_item_macro_r10", "Recall@10", "#1F5A9F", "o"),
        ("full_cold_item_macro_n10", "NDCG@10", "#E54843", "s"),
    ]
    panels = [
        (
            "Course Sample beta",
            "Course Sample \u03b2",
            [
                ("sample_beta_0p10", "0.10"),
                ("main_default", "0.20"),
                ("sample_beta_0p30", "0.30"),
            ],
        ),
        (
            "Reward Scale",
            "Reward Scale",
            [
                ("reward_scale_0p5", "0.5x"),
                ("reward_scale_0p75", "0.75x"),
                ("main_default", "1.0x"),
                ("reward_scale_1p25", "1.25x"),
                ("reward_scale_1p5", "1.5x"),
            ],
        ),
        (
            "Prerequisite Gate",
            "Prerequisite Gate",
            [
                ("main_default", "0.20"),
                ("prereq_gate_0p35", "0.35"),
                ("prereq_gate_0p70", "0.70"),
            ],
        ),
    ]
    delta = make_delta_points(points)
    delta_summary = (
        delta.groupby(["variant", "metric"], as_index=False)
        .agg(mean_delta=("delta", "mean"), std_delta=("delta", "std"), n=("delta", "size"))
        .fillna({"std_delta": 0.0})
    )
    lookup = {
        (str(row.variant), str(row.metric)): row
        for row in delta_summary.itertuples(index=False)
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.75))
    axes = np.asarray(axes).ravel()
    legend_handles = []

    for ax, (panel_title, x_label, variants) in zip(axes, panels):
        xs = np.arange(len(variants))
        xlabels = [label for _, label in variants]

        for metric, metric_label, color, marker in metric_specs:
            x_values: list[int] = []
            y_values: list[float] = []
            for x, (variant, _) in enumerate(variants):
                row = lookup.get((variant, metric))
                if row is None:
                    continue
                x_values.append(x)
                y_values.append(float(row.mean_delta))

                seed_rows = delta[(delta["variant"] == variant) & (delta["metric"] == metric)]
                for _, seed_row in seed_rows.iterrows():
                    seed_jitter = ((int(seed_row["seed"]) % 10) - 6) * 0.018
                    ax.scatter(
                        x + seed_jitter,
                        seed_row["delta"],
                        s=18,
                        marker=marker,
                        color=color,
                        alpha=0.32,
                        linewidth=0,
                        zorder=2,
                    )
            if not x_values:
                continue
            (line,) = ax.plot(
                x_values,
                y_values,
                marker=marker,
                markersize=5.2,
                linewidth=1.35,
                color=color,
                label=metric_label,
                zorder=4,
            )
            if panel_title == panels[0][0]:
                legend_handles.append(line)

        ax.axhline(0, color="#444444", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("\u0394 cold item-macro score", fontsize=10, fontweight="bold")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.55, color="#A8A8A8", alpha=0.65)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        plotted = []
        for metric, _, _, _ in metric_specs:
            for variant, _ in variants:
                row = lookup.get((variant, metric))
                if row is not None:
                    plotted.append(float(row.mean_delta))
                seed_rows = delta[(delta["variant"] == variant) & (delta["metric"] == metric)]
                plotted.extend(float(x) for x in seed_rows["delta"].tolist())
        if plotted:
            limit = max(abs(min(plotted)), abs(max(plotted)), 0.0025)
            ax.set_ylim(-limit * 1.25, limit * 1.25)

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        bbox_to_anchor=(0.5, 1.0),
        prop={"weight": "bold", "size": 10},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.4)

    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=600 if suffix == ".tiff" else 300, bbox_inches="tight")
    delta.to_csv(out_base.with_name(out_base.name + "_points.csv"), index=False)
    delta_summary.to_csv(out_base.with_name(out_base.name + "_summary.csv"), index=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--hparam-root",
        type=Path,
        default=Path("outputs/content_delta_pop5/course_hparam_sensitivity_e60_3seed"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/content_delta_pop5/course_ablation_e60_3seed/full"),
    )
    parser.add_argument(
        "--wide-root",
        type=Path,
        default=Path("outputs/content_delta_pop5/course_hparam_wide_seed2025"),
        help="Root containing completed wide-grid seed2025 plus supplemented multiseed candidates.",
    )
    parser.add_argument("--seeds", default="2025,2026,2027")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--style",
        choices=["dot", "line-grid", "delta-grid", "both"],
        default="line-grid",
        help="Figure style. line-grid matches the parameter-sensitivity reference figure.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    hparam_root = args.hparam_root if args.hparam_root.is_absolute() else repo / args.hparam_root
    baseline_root = args.baseline_root if args.baseline_root.is_absolute() else repo / args.baseline_root
    wide_root = args.wide_root if args.wide_root.is_absolute() else repo / args.wide_root
    out_dir = args.out_dir if args.out_dir is not None else hparam_root / "figures"
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_seeds = {int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()}
    variants = [
        Variant("main_default", "Default (beta=0.20)", baseline_root),
        Variant("sample_beta_0p10", "Sample beta 0.10", hparam_root / "sample_beta_0p10"),
        Variant("sample_beta_0p30", "Sample beta 0.30", hparam_root / "sample_beta_0p30"),
        Variant("reward_scale_0p5", "Reward weights x0.5", hparam_root / "reward_scale_0p5"),
        Variant("reward_scale_0p75", "Reward weights x0.75", wide_root / "reward_scale_0p75"),
        Variant("reward_scale_1p25", "Reward weights x1.25", wide_root / "reward_scale_1p25"),
        Variant("reward_scale_1p5", "Reward weights x1.5", hparam_root / "reward_scale_1p5"),
        Variant("prereq_gate_0p35", "Prereq gate 0.35", hparam_root / "prereq_gate_0p35"),
        Variant("prereq_gate_0p70", "Prereq gate 0.70", wide_root / "prereq_gate_0p70"),
    ]

    points = collect_points(variants, expected_seeds, allow_partial=args.allow_partial)
    if not args.allow_partial:
        found = set(points["variant"].unique())
        missing = [variant.name for variant in variants if variant.name not in found]
        if missing:
            raise SystemExit(
                "Missing complete variants for paper figure: "
                + ", ".join(missing)
                + ". Re-run with --allow-partial for a diagnostic preview."
            )
    summary = summarize(points)
    tag = "preview_partial" if args.allow_partial else "paper_complete"
    points.to_csv(out_dir / f"mooccube_hparam_sensitivity_{tag}_points.csv", index=False)
    summary.to_csv(out_dir / f"mooccube_hparam_sensitivity_{tag}_summary.csv", index=False)
    title_suffix = " (partial preview)" if args.allow_partial else ""
    if args.style in {"dot", "both"}:
        plot_sensitivity(points, summary, out_dir / f"mooccube_hparam_sensitivity_{tag}_dot", title_suffix)
    if args.style in {"line-grid", "both"}:
        plot_line_grid_sensitivity(
            summary,
            out_dir / f"mooccube_hparam_sensitivity_{tag}_linegrid",
            title_suffix,
        )
    if args.style in {"delta-grid", "both"}:
        plot_delta_grid_sensitivity(
            points,
            out_dir / f"mooccube_hparam_sensitivity_{tag}_deltagrid",
            title_suffix,
        )
    print(f"Wrote {out_dir / f'mooccube_hparam_sensitivity_{tag}_summary.csv'}")
    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
