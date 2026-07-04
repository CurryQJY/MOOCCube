from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SEEDS = (2025, 2026, 2027)
METRICS = (
    ("full_cold_item_macro_r10", "Recall@10", "#1F5A9F"),
    ("full_cold_item_macro_n10", "NDCG@10", "#D64B45"),
)


@dataclass(frozen=True)
class PointSpec:
    family: str
    x: float
    tick: str
    variant: str
    root: Path
    is_default: bool = False


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def read_final_fullrank(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle))


def collect_metric_values(point: PointSpec, expected_seeds: set[int]) -> dict[str, list[float]]:
    values = {metric: [] for metric, _, _ in METRICS}
    seen_seeds: list[int] = []
    for final_path in sorted(point.root.rglob("final_fullrank_usim_feedback_fast3_content_delta_static.csv")):
        seed = seed_from_path(final_path)
        if seed is None or seed not in expected_seeds:
            continue
        record = read_final_fullrank(final_path)
        seen_seeds.append(seed)
        for metric, _, _ in METRICS:
            values[metric].append(float(record[metric]))
    if set(seen_seeds) != expected_seeds:
        missing = ",".join(str(seed) for seed in sorted(expected_seeds - set(seen_seeds)))
        raise RuntimeError(f"{point.variant} is not complete for seeds {missing}")
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_records(points: list[PointSpec], expected_seeds: set[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for point in points:
        values_by_metric = collect_metric_values(point, expected_seeds)
        for metric, metric_label, color in METRICS:
            values = values_by_metric[metric]
            records.append(
                {
                    "family": point.family,
                    "x": point.x,
                    "tick": point.tick,
                    "variant": point.variant,
                    "metric": metric,
                    "metric_label": metric_label,
                    "color": color,
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                    "seeds": ",".join(str(seed) for seed in sorted(expected_seeds)),
                    "is_default": point.is_default,
                }
            )
    return records


def family_points(repo: Path) -> list[PointSpec]:
    baseline = repo / "outputs/content_delta_pop5/course_ablation_e60_3seed/full"
    sensitivity = repo / "outputs/content_delta_pop5/course_hparam_sensitivity_e60_3seed"
    wide = repo / "outputs/content_delta_pop5/course_hparam_wide_seed2025"
    sim = repo / "outputs/content_delta_pop5/course_hparam_sim_steps_e60_3seed"

    points: list[PointSpec] = [
        PointSpec("Knowledge-sampling weight beta", 0.00, "0", "sample_beta_0p00", wide / "sample_beta_0p00"),
        PointSpec("Knowledge-sampling weight beta", 0.05, "0.05", "sample_beta_0p05", wide / "sample_beta_0p05"),
        PointSpec("Knowledge-sampling weight beta", 0.10, "0.1", "sample_beta_0p10", sensitivity / "sample_beta_0p10"),
        PointSpec("Knowledge-sampling weight beta", 0.15, "0.15", "sample_beta_0p15", wide / "sample_beta_0p15"),
        PointSpec("Knowledge-sampling weight beta", 0.20, "0.2", "main_default", baseline, True),
        PointSpec("Knowledge-sampling weight beta", 0.25, "0.25", "sample_beta_0p25", wide / "sample_beta_0p25"),
        PointSpec("Knowledge-sampling weight beta", 0.30, "0.3", "sample_beta_0p30", sensitivity / "sample_beta_0p30"),
        PointSpec("Knowledge-sampling weight beta", 0.40, "0.4", "sample_beta_0p40", wide / "sample_beta_0p40"),
        PointSpec("Knowledge-sampling weight beta", 0.50, "0.5", "sample_beta_0p50", wide / "sample_beta_0p50"),
        PointSpec("Reward-shaping scale", 0.00, "0", "reward_scale_0p00", wide / "reward_scale_0p00"),
        PointSpec("Reward-shaping scale", 0.25, "0.25", "reward_scale_0p25", wide / "reward_scale_0p25"),
        PointSpec("Reward-shaping scale", 0.50, "0.5", "reward_scale_0p5", sensitivity / "reward_scale_0p5"),
        PointSpec("Reward-shaping scale", 0.75, "0.75", "reward_scale_0p75", wide / "reward_scale_0p75"),
        PointSpec("Reward-shaping scale", 1.00, "1", "main_default", baseline, True),
        PointSpec("Reward-shaping scale", 1.25, "1.25", "reward_scale_1p25", wide / "reward_scale_1p25"),
        PointSpec("Reward-shaping scale", 1.50, "1.5", "reward_scale_1p5", sensitivity / "reward_scale_1p5"),
        PointSpec("Reward-shaping scale", 2.00, "2", "reward_scale_2p00", wide / "reward_scale_2p00"),
        PointSpec("Simulator horizon T", 1.00, "1", "sim_steps_1", sim / "sim_steps_1"),
        PointSpec("Simulator horizon T", 3.00, "3", "sim_steps_3", sim / "sim_steps_3"),
        PointSpec("Simulator horizon T", 5.00, "5", "main_default", baseline, True),
        PointSpec("Simulator horizon T", 7.00, "7", "sim_steps_7", sim / "sim_steps_7"),
        PointSpec("Simulator horizon T", 10.00, "10", "sim_steps_10", sim / "sim_steps_10"),
    ]
    return points


def write_source_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "x",
        "tick",
        "variant",
        "metric",
        "metric_label",
        "mean",
        "std",
        "n",
        "seeds",
        "is_default",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def set_tight_ylim(ax: plt.Axes, ys: list[float], errs: list[float]) -> None:
    low = min(ys)
    high = max(ys)
    span = max(high - low, 1e-4)
    ax.set_ylim(low - span * 0.24, high + span * 0.24)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))


def draw(records: list[dict[str, object]], out_base: Path, caption_text: str | None) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "lines.linewidth": 1.6,
        }
    )

    families = ["Knowledge-sampling weight beta", "Reward-shaping scale", "Simulator horizon T"]
    family_labels = {
        "Knowledge-sampling weight beta": r"Knowledge-sampling weight $\beta$",
        "Reward-shaping scale": "Reward-shaping scale",
        "Simulator horizon T": r"Simulator horizon $T$",
    }
    display_ticks = {
        "Knowledge-sampling weight beta": [0.00, 0.10, 0.20, 0.30, 0.50],
        "Reward-shaping scale": [0.00, 0.50, 1.00, 1.50, 2.00],
        "Simulator horizon T": [1.00, 3.00, 5.00, 7.00, 10.00],
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.05), constrained_layout=False)

    for col, family in enumerate(families):
        family_records = [record for record in records if record["family"] == family]
        tick_by_x = {float(record["x"]): str(record["tick"]) for record in family_records}
        x_values = display_ticks[family]
        default_x = next(float(record["x"]) for record in family_records if record["is_default"])

        for row, (metric, metric_label, color) in enumerate(METRICS):
            ax = axes[row, col]
            metric_records = sorted(
                [record for record in family_records if record["metric"] == metric],
                key=lambda record: float(record["x"]),
            )
            xs = [float(record["x"]) for record in metric_records]
            ys = [float(record["mean"]) for record in metric_records]
            errs = [float(record["std"]) for record in metric_records]
            marker = "o" if metric.endswith("_r10") else "s"

            ax.axvline(default_x, color="#444444", linestyle=":", linewidth=0.9, alpha=0.85, zorder=1)
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=4.6,
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=3,
            )
            ax.grid(True, color="#C7C7C7", lw=0.55, alpha=0.78, linestyle="--")
            ax.set_xticks(x_values)
            ax.set_xticklabels([tick_by_x[x] for x in x_values], fontsize=7.8)
            ax.tick_params(axis="both", labelsize=7.8, length=3.6, width=0.85, pad=2.8)
            set_tight_ylim(ax, ys, errs)

            ax.set_ylabel(metric_label if col == 0 else "", fontsize=9.4, fontweight="bold", labelpad=5.8)
            if row == 0:
                ax.set_title(family_labels[family], fontsize=9.7, fontweight="bold", pad=6.0)
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("")

    bottom = 0.23 if caption_text else 0.13
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=bottom, wspace=0.27, hspace=0.16)
    if caption_text:
        fig.text(
            0.5,
            0.035,
            caption_text,
            ha="center",
            va="bottom",
            fontsize=11.2,
            fontweight="bold",
            family="serif",
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_single_column(records: list[dict[str, object]], out_base: Path, caption_text: str | None) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.25,
        }
    )

    families = ["Knowledge-sampling weight beta", "Reward-shaping scale", "Simulator horizon T"]
    family_labels = {
        "Knowledge-sampling weight beta": r"Knowledge-sampling weight $\beta$",
        "Reward-shaping scale": "Reward-shaping scale",
        "Simulator horizon T": r"Simulator horizon $T$",
    }
    display_ticks = {
        "Knowledge-sampling weight beta": [0.00, 0.10, 0.20, 0.30, 0.50],
        "Reward-shaping scale": [0.00, 0.50, 1.00, 1.50, 2.00],
        "Simulator horizon T": [1.00, 3.00, 5.00, 7.00, 10.00],
    }

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.55), constrained_layout=False)
    for ax, family in zip(axes, families):
        family_records = [record for record in records if record["family"] == family]
        tick_by_x = {float(record["x"]): str(record["tick"]) for record in family_records}
        default_x = next(float(record["x"]) for record in family_records if record["is_default"])

        all_changes: list[float] = []
        for metric, metric_label, color in METRICS:
            metric_records = sorted(
                [record for record in family_records if record["metric"] == metric],
                key=lambda record: float(record["x"]),
            )
            default_mean = next(float(record["mean"]) for record in metric_records if record["is_default"])
            xs = [float(record["x"]) for record in metric_records]
            changes = [(float(record["mean"]) - default_mean) / default_mean * 100.0 for record in metric_records]
            all_changes.extend(changes)
            marker = "o" if metric.endswith("_r10") else "s"
            ax.plot(
                xs,
                changes,
                color=color,
                marker=marker,
                markersize=3.6,
                markerfacecolor=color,
                markeredgecolor=color,
                label=metric_label,
                zorder=3,
            )

        ax.axhline(0.0, color="#555555", linestyle="-", linewidth=0.65, alpha=0.80, zorder=1)
        ax.axvline(default_x, color="#444444", linestyle=":", linewidth=0.85, alpha=0.85, zorder=1)
        span = max(max(all_changes) - min(all_changes), 0.8)
        ax.set_ylim(min(all_changes) - span * 0.25, max(all_changes) + span * 0.25)
        ax.set_title(family_labels[family], fontsize=8.8, fontweight="bold", pad=3.0)
        ticks = display_ticks[family]
        ax.set_xticks(ticks)
        ax.set_xticklabels([tick_by_x[x] for x in ticks], fontsize=7.2)
        ax.tick_params(axis="both", labelsize=7.2, length=3.0, width=0.8, pad=2.0)
        ax.grid(True, axis="y", color="#C7C7C7", lw=0.45, alpha=0.75, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.36),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        prop={"size": 7.2},
    )
    axes[1].set_ylabel("Change from default (%)", fontsize=8.6, fontweight="bold", labelpad=5.0)

    bottom = 0.18 if caption_text else 0.08
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=bottom, hspace=0.42)
    if caption_text:
        fig.text(
            0.5,
            0.035,
            caption_text,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            family="serif",
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("outputs/content_delta_pop5/hparam_preview/mooccube_hparam_preview"),
    )
    parser.add_argument(
        "--no-embedded-caption",
        action="store_true",
        help="omit the bottom Figure X caption for LaTeX-managed captions",
    )
    parser.add_argument(
        "--single-column",
        action="store_true",
        help="draw a single-column manuscript version using relative changes from defaults",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_base = args.out_base if args.out_base.is_absolute() else repo / args.out_base
    expected_seeds = set(SEEDS)
    records = build_records(family_points(repo), expected_seeds)
    write_source_csv(records, out_base.with_name(out_base.name + "_source.csv"))
    caption_text = None if args.no_embedded_caption else "Figure X: MOOCCube hyperparameter sensitivity under strict item-cold evaluation."
    if args.single_column:
        draw_single_column(records, out_base, caption_text)
    else:
        draw(records, out_base, caption_text)
    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")
    print(f"Wrote {out_base.with_suffix('.svg')}")
    print(f"Wrote {out_base.with_suffix('.tiff')}")
    print(f"Wrote {out_base.with_name(out_base.name + '_source.csv')}")


if __name__ == "__main__":
    main()
