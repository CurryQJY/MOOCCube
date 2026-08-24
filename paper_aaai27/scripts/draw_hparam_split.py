from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO = PAPER_DIR.parent
OUT = PAPER_DIR / "figures" / "mooccube_hparam_sensitivity_single_column.pdf"
OUT_PNG = PAPER_DIR / "figures" / "mooccube_hparam_sensitivity_single_column.png"
REPORT_DIR = REPO / "outputs" / "content_delta_pop5" / "course_hparam_validation_figures"
SUMMARY_INPUT = REPORT_DIR / "mooccube_hparam_validation_summary.csv"

METRICS_FILE = "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"
EXPECTED_SEEDS = (2025, 2026, 2027)
DEFAULT = {"beta": 0.20, "reward": 1.00, "horizon": 5}
SELECT_METRIC = "Val_full_cold_N@10"
RECALL_METRIC = "Val_full_cold_R@10"


@dataclass(frozen=True)
class PointSpec:
    sweep: str
    title: str
    x: float
    label: str
    variant: str
    roots: tuple[Path, ...]


def repo_path(path: str) -> Path:
    return REPO / path


DEFAULT_ROOTS = (
    repo_path("outputs/significance_per_item_exports/mooccube/ckg_rl_full"),
)
TRUE_TRUE_HPARAM_ROOT = "outputs/significance_per_item_exports/mooccube/ckg_rl_true_true_hparam_grid"


SWEEPS: list[dict[str, object]] = [
    {
        "key": "beta",
        "title": r"Knowledge-sampling weight $\beta$",
        "points": [
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.00, "0.00", "beta_0p00", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p00"),)),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.10, "0.10", "beta_0p10", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p10"),)),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.15, "0.15", "beta_0p15", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p15"),)),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.20, "0.20", "main_default", DEFAULT_ROOTS),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.25, "0.25", "beta_0p25", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p25"),)),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.30, "0.30", "beta_0p30", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p30"),)),
            PointSpec("beta", r"Knowledge-sampling weight $\beta$", 0.50, "0.50", "beta_0p50", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/beta_0p50"),)),
        ],
    },
    {
        "key": "reward",
        "title": "Reward-shaping scale",
        "points": [
            PointSpec("reward", "Reward-shaping scale", 0.00, "0.00", "reward_0p00", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/reward_0p00"),)),
            PointSpec("reward", "Reward-shaping scale", 0.50, "0.50", "reward_0p50", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/reward_0p50"),)),
            PointSpec("reward", "Reward-shaping scale", 1.00, "1.00", "main_default", DEFAULT_ROOTS),
            PointSpec("reward", "Reward-shaping scale", 1.50, "1.50", "reward_1p50", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/reward_1p50"),)),
            PointSpec("reward", "Reward-shaping scale", 2.00, "2.00", "reward_2p00", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/reward_2p00"),)),
        ],
    },
    {
        "key": "horizon",
        "title": r"Simulator horizon $T$",
        "points": [
            PointSpec("horizon", r"Simulator horizon $T$", 1.0, "1", "horizon_1", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/horizon_1"),)),
            PointSpec("horizon", r"Simulator horizon $T$", 3.0, "3", "horizon_3", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/horizon_3"),)),
            PointSpec("horizon", r"Simulator horizon $T$", 5.0, "5", "main_default", DEFAULT_ROOTS),
            PointSpec("horizon", r"Simulator horizon $T$", 7.0, "7", "horizon_7", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/horizon_7"),)),
            PointSpec("horizon", r"Simulator horizon $T$", 10.0, "10", "horizon_10", (repo_path(f"{TRUE_TRUE_HPARAM_ROOT}/horizon_10"),)),
        ],
    },
]


def find_metrics_file(spec: PointSpec, seed: int) -> Path | None:
    rel = Path(f"strict_item_cold_balanced_thr1_seed_{seed}") / METRICS_FILE
    for root in spec.roots:
        path = root / rel
        if path.exists():
            return path
    return None


def read_best_validation(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    for column in ("Epoch", SELECT_METRIC, RECALL_METRIC):
        if column not in frame.columns:
            raise ValueError(f"{path} missing required column {column}")
    frame[SELECT_METRIC] = pd.to_numeric(frame[SELECT_METRIC], errors="coerce")
    frame[RECALL_METRIC] = pd.to_numeric(frame[RECALL_METRIC], errors="coerce")
    valid = frame.dropna(subset=[SELECT_METRIC, RECALL_METRIC])
    if valid.empty:
        raise ValueError(f"{path} has no numeric validation metric rows")
    idx = valid[SELECT_METRIC].idxmax()
    row = valid.loc[idx]
    return {
        "best_epoch": int(row["Epoch"]),
        "val_r10_at_best_n10": float(row[RECALL_METRIC]),
        "val_n10": float(row[SELECT_METRIC]),
    }


def collect_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for sweep_order, sweep in enumerate(SWEEPS):
        for point_order, spec in enumerate(sweep["points"]):
            found_seeds: list[int] = []
            missing_seeds: list[int] = []
            for seed in EXPECTED_SEEDS:
                metrics_path = find_metrics_file(spec, seed)
                if metrics_path is None:
                    missing_seeds.append(seed)
                    continue
                best = read_best_validation(metrics_path)
                found_seeds.append(seed)
                rows.append(
                    {
                        "sweep": spec.sweep,
                        "sweep_order": sweep_order,
                        "point_order": point_order,
                        "variant": spec.variant,
                        "x": spec.x,
                        "label": spec.label,
                        "seed": seed,
                        "eval_split": "validation",
                        "selection_metric": SELECT_METRIC,
                        "recall_metric": RECALL_METRIC,
                        **best,
                        "source_file": metrics_path.as_posix(),
                    }
                )
            missing_rows.append(
                {
                    "sweep": spec.sweep,
                    "point_order": point_order,
                    "variant": spec.variant,
                    "x": spec.x,
                    "label": spec.label,
                    "expected_seeds": ",".join(str(s) for s in EXPECTED_SEEDS),
                    "found_seeds": ",".join(str(s) for s in found_seeds),
                    "missing_seeds": ",".join(str(s) for s in missing_seeds),
                    "n_found": len(found_seeds),
                    "n_missing": len(missing_seeds),
                    "complete": len(missing_seeds) == 0,
                    "eval_split": "validation",
                    "selection_metric": SELECT_METRIC,
                    "root_candidates": " | ".join(str(root) for root in spec.roots),
                }
            )

    points = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)
    if points.empty:
        raise SystemExit("No validation metric files found for the configured sweeps.")
    return points, missing


def summarize(points: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    summary = (
        points.groupby(["sweep", "sweep_order", "point_order", "variant", "x", "label"], as_index=False)
        .agg(
            n=("seed", "size"),
            seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.unique()))),
            val_r10_mean=("val_r10_at_best_n10", "mean"),
            val_r10_std=("val_r10_at_best_n10", "std"),
            val_n10_mean=("val_n10", "mean"),
            val_n10_std=("val_n10", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
        )
        .sort_values(["sweep_order", "point_order"])
    )
    summary[["val_r10_std", "val_n10_std"]] = summary[["val_r10_std", "val_n10_std"]].fillna(0.0)
    summary["eval_split"] = "validation"
    summary["selection_metric"] = SELECT_METRIC
    missing_small = missing[["sweep", "variant", "x", "n_missing", "missing_seeds", "complete"]]
    return summary.merge(missing_small, on=["sweep", "variant", "x"], how="left")


def default_score(rows: pd.DataFrame, x_value: float, metric: str) -> float | None:
    match = rows[rows["x"] == x_value]
    if match.empty:
        return None
    return float(match.iloc[0][metric])


def compact_tick_label(label: str) -> str:
    return label


def set_point_xticks(ax, rows: pd.DataFrame, key: str) -> None:
    ax.set_xticks(rows["x"].astype(float).tolist())
    if key in {"beta", "reward"}:
        labels = [f"{float(x):.2f}" for x in rows["x"]]
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.0)
    else:
        labels = [str(int(float(x))) for x in rows["x"]]
        ax.set_xticklabels(labels, fontsize=6.6)


def style_axis(ax):
    ax.grid(axis="y", color="#d6d6d6", lw=0.55, linestyle="--", zorder=0)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def plot_validation(summary: pd.DataFrame) -> None:
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
    metric_specs = [
        ("r10", "Recall@10", "val_r10_mean", "val_r10_std", "o"),
        ("n10", "NDCG@10", "val_n10_mean", "val_n10_std", "s"),
    ]

    for row_idx, sweep in enumerate(SWEEPS):
        key = str(sweep["key"])
        title = str(sweep["title"])
        sweep_rows = summary[summary["sweep"] == key].sort_values("point_order")
        x_values = sweep_rows["x"].astype(float).tolist()

        for col_idx, (metric_key, metric_label, mean_col, std_col, marker) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            y_values = sweep_rows[mean_col].astype(float).tolist()
            ax.plot(
                x_values,
                y_values,
                color=colors[metric_key],
                marker=marker,
                markersize=3.2,
                linewidth=1.35,
                markeredgewidth=0.0,
            )

            default_x = float(DEFAULT[key])
            default_y = default_score(sweep_rows, default_x, mean_col)
            if default_y is not None:
                ax.scatter(
                    [default_x],
                    [default_y],
                    s=20,
                    facecolors="white",
                    edgecolors=colors[metric_key],
                    linewidths=1.05,
                    zorder=4,
                )
                ax.axvline(default_x, color="#666666", lw=0.8, linestyle=":")

            style_axis(ax)
            panel_min = min(y_values)
            panel_max = max(y_values)
            pad = max(0.0007, (panel_max - panel_min) * 0.18)
            ax.set_ylim(panel_min - pad, panel_max + pad)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.text(
                0.97,
                0.93,
                metric_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=colors[metric_key],
                fontsize=6.8,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
            )
            if col_idx == 0:
                ax.text(
                    0.0,
                    1.06,
                    title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold",
                )
            if row_idx == 2:
                ax.set_xlabel("Sweep value", fontsize=7, labelpad=2)
            set_point_xticks(ax, sweep_rows, key)

    fig.supylabel("Validation score", x=0.02, fontsize=7.2, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.99, top=0.93, bottom=0.10, wspace=0.33, hspace=0.78)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_reports(points: pd.DataFrame, summary: pd.DataFrame, missing: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(REPORT_DIR / "mooccube_hparam_validation_points.csv", index=False)
    summary.to_csv(REPORT_DIR / "mooccube_hparam_validation_summary.csv", index=False)
    missing.to_csv(REPORT_DIR / "mooccube_hparam_validation_missing.csv", index=False)

    tex_rows = []
    for row in summary.itertuples(index=False):
        missing_note = "" if int(row.n_missing) == 0 else f" (missing {row.missing_seeds})"
        tex_rows.append(
            {
                "Sweep": row.sweep,
                "Value": row.label,
                "R@10": f"{row.val_r10_mean:.4f}$\\pm${row.val_r10_std:.4f}",
                "N@10": f"{row.val_n10_mean:.4f}$\\pm${row.val_n10_std:.4f}",
                "Seeds": f"{row.seeds}{missing_note}",
            }
        )
    with (REPORT_DIR / "mooccube_hparam_validation_latex_rows.tex").open("w", encoding="utf-8", newline="") as handle:
        for row in tex_rows:
            handle.write(
                f"{row['Sweep']} & {row['Value']} & {row['R@10']} & {row['N@10']} & {row['Seeds']} \\\\\n"
            )


def main() -> None:
    if SUMMARY_INPUT.exists():
        summary = pd.read_csv(SUMMARY_INPUT)
        points = pd.read_csv(REPORT_DIR / "mooccube_hparam_validation_points.csv")
        missing = pd.read_csv(REPORT_DIR / "mooccube_hparam_validation_missing.csv")
    else:
        points, missing = collect_points()
        summary = summarize(points, missing)
        write_reports(points, summary, missing)
    plot_validation(summary)

    total_missing = int(missing["n_missing"].sum())
    incomplete = missing[missing["n_missing"] > 0]
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote reports under {REPORT_DIR}")
    print(f"Expected point-seed files: {len(missing) * len(EXPECTED_SEEDS)}")
    print(f"Missing point-seed files: {total_missing}")
    if not incomplete.empty:
        print("Incomplete points:")
        for row in incomplete.itertuples(index=False):
            print(f"  {row.sweep} {row.label} ({row.variant}): missing {row.missing_seeds}")


if __name__ == "__main__":
    main()
