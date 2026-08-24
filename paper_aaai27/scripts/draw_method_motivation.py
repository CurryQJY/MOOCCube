from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
BASE = FIG_DIR / "mooccube_method_motivation"

from paper_aaai27.scripts.analyze_method_motivation import (  # noqa: E402
    METRICS as ALIGNMENT_METRICS,
)

DATA = ROOT / "processed_data_hin_clean_pop5" / "stream_data.pkl"
SPLIT_ROOT = ROOT / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
SEEDS = [2025, 2026, 2027]

BLUE = "#1f5aa6"
RED = "#c9403a"
TEAL = "#4aa68d"
MUTED = "#9aa99a"
GRID = "#d6d6d6"
DARK = "#222222"

RISK_COLUMNS = [
    ("prereq_gap", "Prereq. gap", RED, "\\\\"),
    ("difficulty_gap", "Difficulty gap", RED, "//"),
    ("redundancy", "Redundancy", MUTED, "xx"),
    ("concept_bonus", "Concept bonus", TEAL, ""),
]

ALIGNMENT_LABELS = {
    "prerequisite_gap": "Prerequisite gap",
    "concept_continuity": "Concept continuity",
    "difficulty_gap": "Difficulty gap",
    "structural_redundancy": "Structural redundancy",
}

MODEL_COLORS = {
    "ckg_rl": "#2F6B9A",
    "pcgnn": "#C4772E",
    "cgrc": "#9A9A9A",
}

MODEL_HATCHES = {
    "pcgnn": "///",
    "cgrc": "...",
}


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
            "font.size": 7.0,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "hatch.linewidth": 0.6,
        }
    )


def _validate_seed_course_units(frame: pd.DataFrame, description: str) -> None:
    required = {"seed", "target_item_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{description} missing columns: {missing}")
    if len(frame) != 204:
        raise ValueError(f"{description} must contain 204 seed-course units")
    if frame.duplicated(["seed", "target_item_id"]).any():
        raise ValueError(f"{description} contains duplicate seed-course units")


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def build_existing_method_diagnostics(
    course_macro: pd.DataFrame,
    exposure: pd.DataFrame,
    model_summary: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
) -> dict[str, dict]:
    pcgnn = course_macro.loc[
        course_macro["model"].eq("pcgnn") & course_macro["cutoff"].eq(10)
    ].copy()
    cgrc = exposure.loc[exposure["model"].eq("cgrc")].copy()
    _validate_seed_course_units(pcgnn, "PCGNN structural diagnostics")
    _validate_seed_course_units(cgrc, "CGRC exposure diagnostics")

    metrics = {}
    for offset, metric in enumerate(("prerequisite_gap", "difficulty_gap")):
        if metric not in pcgnn:
            raise ValueError(f"PCGNN diagnostics missing {metric}")
        mean, ci_low, ci_high = _bootstrap_mean_interval(
            pcgnn[metric].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            random_seed=2027 + offset,
        )
        metrics[metric] = {
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    if "N@10" not in cgrc:
        raise ValueError("CGRC exposure diagnostics missing N@10")
    cgrc_summary = model_summary.loc[
        model_summary["model"].eq("cgrc") & model_summary["cutoff"].eq(10)
    ]
    if len(cgrc_summary) != 1:
        raise ValueError("expected one CGRC Top-10 model summary row")

    ndcg_values = cgrc["N@10"].to_numpy(dtype=float)
    return {
        "pcgnn": {
            "count": len(pcgnn),
            "metrics": metrics,
        },
        "cgrc": {
            "count": len(cgrc),
            "ndcg_values": ndcg_values,
            "low_ndcg_threshold": 0.10,
            "low_ndcg_fraction": float((ndcg_values <= 0.10).mean()),
            "cold_proportion": float(
                cgrc_summary.iloc[0]["cold_proportion_mean"]
            ),
        },
    }


def diagnostics_frame(diagnostics: dict[str, dict]) -> pd.DataFrame:
    pcgnn = diagnostics["pcgnn"]
    cgrc = diagnostics["cgrc"]
    rows = []
    for metric, values in pcgnn["metrics"].items():
        rows.append(
            {
                "model": "pcgnn",
                "diagnostic": metric,
                "value": values["mean"],
                "ci_low": values["ci_low"],
                "ci_high": values["ci_high"],
                "unit_count": pcgnn["count"],
                "threshold": float("nan"),
            }
        )
    rows.extend(
        [
            {
                "model": "cgrc",
                "diagnostic": "fraction_ndcg_at_10_le_0_10",
                "value": cgrc["low_ndcg_fraction"],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "unit_count": cgrc["count"],
                "threshold": cgrc["low_ndcg_threshold"],
            },
            {
                "model": "cgrc",
                "diagnostic": "cold_proportion_at_10",
                "value": cgrc["cold_proportion"],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "unit_count": cgrc["count"],
                "threshold": float("nan"),
            },
        ]
    )
    return pd.DataFrame(rows)


def draw_existing_method_motivation_figure(
    course_macro: pd.DataFrame,
    exposure: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_base: Path,
    *,
    n_bootstrap: int = 10_000,
) -> list[Path]:
    diagnostics = build_existing_method_diagnostics(
        course_macro,
        exposure,
        model_summary,
        n_bootstrap=n_bootstrap,
    )

    configure_style()
    fig = plt.figure(figsize=(3.35, 3.75))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.82, 1.18, 0.58),
        hspace=0.78,
    )
    fig.suptitle(
        "Why existing methods fall short",
        x=0.52,
        y=0.982,
        fontsize=7.8,
        fontweight="bold",
    )

    ax_a = fig.add_subplot(grid[0])
    metric_order = ("prerequisite_gap", "difficulty_gap")
    metric_labels = ("Prerequisite gap", "Difficulty gap")
    metric_rows = diagnostics["pcgnn"]["metrics"]
    means = np.array([metric_rows[metric]["mean"] for metric in metric_order])
    lows = np.array([metric_rows[metric]["ci_low"] for metric in metric_order])
    highs = np.array([metric_rows[metric]["ci_high"] for metric in metric_order])
    y = np.arange(len(metric_order))[::-1]
    bars = ax_a.barh(
        y,
        means,
        height=0.45,
        color=MODEL_COLORS["pcgnn"],
        edgecolor="#333333",
        linewidth=0.75,
        hatch=MODEL_HATCHES["pcgnn"],
        zorder=2,
    )
    ax_a.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.2,
        zorder=3,
    )
    for bar, value in zip(bars, means):
        ax_a.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=6.4,
            fontweight="bold",
        )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(metric_labels)
    ax_a.set_xlim(0.0, max(0.72, float(highs.max()) + 0.07))
    ax_a.set_xlabel(r"Held-out gap score ($\downarrow$ better)")
    ax_a.set_title(
        "PCGNN: structural mismatch",
        loc="left",
        color=MODEL_COLORS["pcgnn"],
        fontweight="bold",
        pad=2,
    )
    ax_a.grid(axis="x", color=GRID, lw=0.5, ls=":", zorder=0)
    ax_a.text(-0.17, 1.10, "(a)", transform=ax_a.transAxes, fontweight="bold")

    ax_b = fig.add_subplot(grid[1])
    cgrc = diagnostics["cgrc"]
    bins = np.linspace(0.0, 0.85, 12)
    ax_b.axvspan(
        0.0,
        cgrc["low_ndcg_threshold"],
        color="#d9d9d9",
        alpha=0.45,
        lw=0,
        zorder=0,
    )
    ax_b.hist(
        cgrc["ndcg_values"],
        bins=bins,
        color=MODEL_COLORS["cgrc"],
        edgecolor="#333333",
        linewidth=0.65,
        hatch=MODEL_HATCHES["cgrc"],
        zorder=2,
    )
    ax_b.axvline(
        cgrc["low_ndcg_threshold"],
        color="#333333",
        lw=0.9,
        ls="--",
        zorder=3,
    )
    histogram_counts, _ = np.histogram(cgrc["ndcg_values"], bins=bins)
    ax_b.text(
        0.125,
        float(histogram_counts.max()) * 0.88,
        rf"{cgrc['low_ndcg_fraction']:.0%} at NDCG@10 $\leq$ 0.10",
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.7,
        },
    )
    ax_b.text(
        0.98,
        0.91,
        f"Top-10 cold share: {cgrc['cold_proportion']:.1%}",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    ax_b.set_xlim(0.0, 0.85)
    ax_b.set_xlabel("Course-level NDCG@10")
    ax_b.set_ylabel("Cold courses")
    ax_b.set_title(
        "CGRC: weak cold-course ranking",
        loc="left",
        color="#666666",
        fontweight="bold",
        pad=2,
    )
    ax_b.grid(axis="y", color=GRID, lw=0.5, ls=":", zorder=0)
    ax_b.text(-0.17, 1.10, "(b)", transform=ax_b.transAxes, fontweight="bold")

    ax_c = fig.add_subplot(grid[2])
    ax_c.set_axis_off()
    ax_c.add_patch(
        Rectangle(
            (0.0, 0.02),
            1.0,
            0.96,
            transform=ax_c.transAxes,
            facecolor="#EAF2F8",
            edgecolor=MODEL_COLORS["ckg_rl"],
            linewidth=1.0,
        )
    )
    ax_c.plot(
        [0.5, 0.5],
        [0.12, 0.80],
        transform=ax_c.transAxes,
        color=MODEL_COLORS["ckg_rl"],
        lw=0.65,
    )
    ax_c.text(
        0.02,
        0.87,
        "CKG-RL response",
        transform=ax_c.transAxes,
        color=MODEL_COLORS["ckg_rl"],
        fontsize=7.0,
        fontweight="bold",
        va="top",
    )
    ax_c.text(
        0.25,
        0.43,
        "Structure -> knowledge signals\nsampling | rewards | supervision",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        linespacing=1.15,
    )
    ax_c.text(
        0.75,
        0.43,
        "Cold ranking -> cold embeddings\nanchoring | masking | simulation",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        linespacing=1.15,
    )

    for axis in (ax_a, ax_b):
        axis.tick_params(length=2.4, pad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.30, right=0.99, top=0.90, bottom=0.03)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        output = output_base.with_suffix(suffix)
        fig.savefig(
            output,
            dpi=450,
            bbox_inches="tight",
            pad_inches=0.015,
            facecolor="white",
        )
        outputs.append(output)
    plt.close(fig)
    return outputs


def _course_terms_for_seed(
    seed: int,
    item_concept_overlap: np.ndarray,
    item_prereq_mat: np.ndarray,
    item_prereq_cnt: np.ndarray,
) -> pd.DataFrame:
    root = SPLIT_ROOT / f"strict_item_cold_balanced_thr1_seed_{seed}"
    train = pd.read_pickle(root / "static_train.pkl")
    test = pd.read_pickle(root / "static_test.pkl")
    cold = test[test["_split_source"].eq("strict_item_cold_test")][["u_idx", "i_idx"]].copy()

    n_items = int(item_concept_overlap.shape[0])
    train_pop = train.groupby("i_idx").size().reindex(range(n_items), fill_value=0).to_numpy(dtype=np.float32)
    max_pop = float(np.log1p(train_pop).max()) or 1.0
    item_difficulty = 1.0 - np.log1p(train_pop) / max_pop
    user_hist = train.groupby("u_idx")["i_idx"].apply(
        lambda series: np.array(sorted(set(map(int, series))), dtype=np.int32)
    ).to_dict()

    rows: list[dict[str, float | int]] = []
    for user_idx, item_idx in cold.itertuples(index=False):
        user_idx = int(user_idx)
        item_idx = int(item_idx)
        seen = user_hist.get(user_idx)
        if seen is None or len(seen) < 1:
            continue

        prereq_gap = 0.0
        if item_prereq_cnt[item_idx] > 0:
            prereq_seen = float(item_prereq_mat[item_idx, seen].sum())
            prereq_gap = float(np.clip(1.0 - prereq_seen / max(float(item_prereq_cnt[item_idx]), 1.0), 0.0, 1.0))

        concept_match = float(np.clip(item_concept_overlap[item_idx, seen].sum() / max(len(seen), 1), 0.0, 1.0))
        redundant_thr = 0.70
        concept_min = 0.12
        concept_bonus = float(np.clip((concept_match - concept_min) / (redundant_thr - concept_min), 0.0, 1.0))
        redundancy = float(np.clip((concept_match - redundant_thr) / (1.0 - redundant_thr), 0.0, 1.0))
        prereq_safe = 1.0 if prereq_gap <= 0.20 else 0.0
        concept_bonus = concept_bonus * prereq_safe * (1.0 - redundancy)

        readiness = min(1.0, len(seen) / 5.0)
        difficulty_gap = float(max(0.0, item_difficulty[item_idx] - readiness))
        fit_penalty = 0.08 * prereq_gap + 0.03 * difficulty_gap + 0.02 * redundancy - 0.04 * concept_bonus

        rows.append(
            {
                "seed": seed,
                "item_id": item_idx,
                "prereq_gap": prereq_gap,
                "concept_match": concept_match,
                "concept_bonus": concept_bonus,
                "difficulty_gap": difficulty_gap,
                "redundancy": redundancy,
                "fit_penalty": fit_penalty,
                "history_len": len(seen),
            }
        )

    pair_terms = pd.DataFrame(rows)
    return (
        pair_terms.groupby(["seed", "item_id"], as_index=False)
        .agg(
            test_pairs=("fit_penalty", "size"),
            prereq_gap=("prereq_gap", "mean"),
            concept_match=("concept_match", "mean"),
            concept_bonus=("concept_bonus", "mean"),
            difficulty_gap=("difficulty_gap", "mean"),
            redundancy=("redundancy", "mean"),
            fit_penalty=("fit_penalty", "mean"),
            history_len=("history_len", "mean"),
        )
    )


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from fast3_delta.course_artifacts import build_course_artifacts

    full_df = pd.read_pickle(DATA)
    n_items = int(full_df["i_idx"].max()) + 1
    artifacts, stats = build_course_artifacts(
        full_df,
        n_items=n_items,
        relation_dir=str(ROOT / "MOOCCube" / "relations"),
        prereq_min_support=30,
        prereq_max_per_item=5,
        prereq_min_items=1,
        prereq_max_forward=20,
        concept_overlap_mode="plain",
        prereq_graph_source="concept",
        prereq_concept_score_thr=0.10,
        prereq_concept_min_hits=1,
        prereq_concept_file="prerequisite-dependency.json",
    )
    item_concept_overlap = artifacts["item_concept_overlap"].cpu().numpy().astype(np.float32)
    item_prereq_mat = artifacts["item_prereq_item_mat"].cpu().numpy().astype(np.float32)
    item_prereq_cnt = artifacts["item_prereq_item_cnt"].cpu().numpy().astype(np.float32)

    risk_frames = [
        _course_terms_for_seed(seed, item_concept_overlap, item_prereq_mat, item_prereq_cnt)
        for seed in SEEDS
    ]
    risk = pd.concat(risk_frames, ignore_index=True)

    exposure_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        cgrc_path = (
            SPLIT_ROOT
            / f"strict_item_cold_balanced_thr1_seed_{seed}"
            / "rq1_per_course_cgrc_export"
            / "per_item_full_cold_cgrc_paper_static.csv"
        )
        frame = pd.read_csv(cgrc_path)
        frame["seed"] = seed
        frame["source"] = str(cgrc_path.relative_to(ROOT))
        exposure_frames.append(frame)
    exposure = pd.concat(exposure_frames, ignore_index=True)

    summary_rows = [
        {"stat": "cold_course_seed_rows", "value": float(len(exposure))},
        {"stat": "cgrc_n10_mean", "value": float(exposure["N@10"].mean())},
        {"stat": "cgrc_n10_median", "value": float(exposure["N@10"].median())},
        {"stat": "cgrc_n10_le_0p05", "value": float((exposure["N@10"] <= 0.05).mean())},
        {"stat": "cgrc_n10_le_0p10", "value": float((exposure["N@10"] <= 0.10).mean())},
        {"stat": "prereq_gap_mean", "value": float(risk["prereq_gap"].mean())},
        {"stat": "difficulty_gap_mean", "value": float(risk["difficulty_gap"].mean())},
        {"stat": "concept_bonus_zero_rate", "value": float((risk["concept_bonus"] <= 1e-12).mean())},
        {"stat": "items_with_concept", "value": float(stats.get("items_with_concept", 0))},
        {"stat": "items_with_prereq", "value": float(stats.get("items_with_prereq", 0))},
    ]
    summary = pd.DataFrame(summary_rows)
    return risk, exposure, summary


def draw(risk: pd.DataFrame, exposure: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.25, 3.45),
        gridspec_kw={"height_ratios": [0.96, 1.20]},
        constrained_layout=False,
    )

    x = np.sort(exposure["N@10"].to_numpy(dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / max(len(x), 1)
    ax_a.axvspan(0, 0.10, color=RED, alpha=0.055, lw=0, zorder=0)
    ax_a.plot(x, y, color=BLUE, lw=1.35, zorder=3)
    for cutoff, label_y in [(0.05, 0.30), (0.10, 0.47)]:
        frac = float((exposure["N@10"] <= cutoff).mean())
        ax_a.axvline(cutoff, color="#666666", lw=0.8, ls=":", zorder=2)
        ax_a.text(
            cutoff + 0.012,
            label_y,
            f"{frac:.0%} <= {cutoff:.2f}",
            fontsize=6.4,
            color=DARK,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
    ax_a.set_xlim(0, min(0.86, max(0.22, float(x.max()) + 0.035)))
    ax_a.set_ylim(0, 1.02)
    ax_a.text(
        0.98,
        0.08,
        f"n={len(x)}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    ax_a.set_xlabel("CGRC cold-course NDCG@10")
    ax_a.set_ylabel("Cumulative fraction")
    ax_a.grid(axis="y", color=GRID, lw=0.55, linestyle="--", zorder=0)
    ax_a.text(-0.16, 1.07, "(a)", transform=ax_a.transAxes, fontsize=7.4, fontweight="bold", va="top")

    rng = np.random.default_rng(2027)
    positions = np.arange(len(RISK_COLUMNS))[::-1]
    box_data = [risk[col].to_numpy(dtype=float) for col, _, _, _ in RISK_COLUMNS]
    box = ax_b.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 0.9},
        whiskerprops={"color": "#333333", "linewidth": 0.75},
        capprops={"color": "#333333", "linewidth": 0.75},
        boxprops={"linewidth": 0.75, "color": "#333333"},
    )
    for patch, (_, _, color, hatch) in zip(box["boxes"], RISK_COLUMNS):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_hatch(hatch)

    for pos, (col, _, color, _) in zip(positions, RISK_COLUMNS):
        vals = risk[col].to_numpy(dtype=float)
        if len(vals) > 160:
            idx = rng.choice(len(vals), size=160, replace=False)
            vals = vals[idx]
        jitter = rng.normal(0.0, 0.055, size=len(vals))
        ax_b.scatter(
            vals,
            np.full_like(vals, pos, dtype=float) + jitter,
            s=4.5,
            color=color,
            alpha=0.26,
            linewidth=0,
            zorder=3,
        )

    ax_b.set_yticks(positions)
    ax_b.set_yticklabels([label for _, label, _, _ in RISK_COLUMNS])
    ax_b.set_xlim(-0.02, 1.05)
    ax_b.set_xlabel("Mean signal per cold-test pair (0-1)")
    ax_b.grid(axis="x", color=GRID, lw=0.55, linestyle="--", zorder=0)
    ax_b.text(-0.16, 1.06, "(b)", transform=ax_b.transAxes, fontsize=7.4, fontweight="bold", va="top")

    for ax in (ax_a, ax_b):
        ax.tick_params(length=2.4, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(left=0.30, right=0.99, top=0.965, bottom=0.12, hspace=0.45)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(BASE.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def _validate_alignment_rows(paired: pd.DataFrame) -> pd.DataFrame:
    required = {
        "metric",
        "direction",
        "pair_count",
        "favorable_alignment_effect",
        "favorable_ci_low",
        "favorable_ci_high",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"paired alignment input missing columns: {missing}")
    rows = paired.set_index("metric").reindex(ALIGNMENT_METRICS)
    if rows.index.has_duplicates or rows["pair_count"].isna().any():
        raise ValueError("paired alignment input must contain each required metric once")
    return rows.reset_index()


def draw_motivation_figure(
    exposure: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    cold_share_top10: float,
    output_base: Path = BASE,
) -> list[Path]:
    if "N@10" not in exposure:
        raise ValueError("exposure input must contain N@10")
    rows = _validate_alignment_rows(paired)
    configure_style()
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.25, 3.35),
        gridspec_kw={"height_ratios": [0.95, 1.05]},
        constrained_layout=False,
    )

    x = np.sort(exposure["N@10"].to_numpy(dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / max(len(x), 1)
    ax_a.axvspan(0, 0.10, color=RED, alpha=0.055, lw=0, zorder=0)
    ax_a.plot(x, y, color=BLUE, lw=1.35, zorder=3)
    for cutoff, label_y in [(0.05, 0.30), (0.10, 0.47)]:
        fraction = float((exposure["N@10"] <= cutoff).mean())
        ax_a.axvline(cutoff, color="#666666", lw=0.8, ls=":", zorder=2)
        ax_a.text(
            cutoff + 0.012,
            label_y,
            f"{fraction:.0%} <= {cutoff:.2f}",
            fontsize=6.4,
            color=DARK,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
    ax_a.set_xlim(0, min(0.86, max(0.22, float(x.max()) + 0.035)))
    ax_a.set_ylim(0, 1.02)
    ax_a.text(
        0.98,
        0.20,
        f"n={len(x)}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#444444",
    )
    ax_a.text(
        0.98,
        0.08,
        f"Top-10 cold share: {float(cold_share_top10):.1%}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#444444",
    )
    ax_a.set_xlabel("CGRC cold-course NDCG@10")
    ax_a.set_ylabel("Cumulative fraction")
    ax_a.grid(axis="y", color=GRID, lw=0.55, linestyle="--", zorder=0)
    ax_a.text(-0.16, 1.07, "(a)", transform=ax_a.transAxes, fontsize=7.4, fontweight="bold", va="top")

    effects = rows["favorable_alignment_effect"].to_numpy(dtype=float)
    lows = rows["favorable_ci_low"].to_numpy(dtype=float)
    highs = rows["favorable_ci_high"].to_numpy(dtype=float)
    positions = np.arange(len(rows))[::-1]
    colors = [
        TEAL if low > 0.0 else RED if high < 0.0 else "#777777"
        for low, high in zip(lows, highs)
    ]
    for pos, effect, low, high, color in zip(positions, effects, lows, highs, colors):
        ax_b.errorbar(
            effect,
            pos,
            xerr=[[effect - low], [high - effect]],
            fmt="o",
            ms=4.4,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.4,
            capthick=0.8,
            zorder=3,
        )
        offset = 0.004 if effect >= 0.0 else -0.004
        ax_b.text(
            effect + offset,
            pos,
            f"{effect:+.3f}",
            ha="left" if effect >= 0.0 else "right",
            va="center",
            fontsize=6.2,
            color=DARK,
        )
    extent = max(0.075, float(np.nanmax(np.abs(np.concatenate([lows, highs])))) * 1.55)
    ax_b.axvline(0.0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax_b.set_xlim(-extent, extent)
    ax_b.set_ylim(-0.45, len(rows) - 0.55)
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels([ALIGNMENT_LABELS[metric] for metric in rows["metric"]])
    ax_b.set_xlabel("Favorable Top-10 alignment vs. ranks 11-20")
    ax_b.grid(axis="x", color=GRID, lw=0.55, linestyle="--", zorder=0)
    ax_b.text(-0.16, 1.07, "(b)", transform=ax_b.transAxes, fontsize=7.4, fontweight="bold", va="top")

    for ax in (ax_a, ax_b):
        ax.tick_params(length=2.4, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(left=0.34, right=0.99, top=0.97, bottom=0.12, hspace=0.50)
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        output = output_base.with_suffix(suffix)
        fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.015)
        outputs.append(output)
    plt.close(fig)
    return outputs


def collect_exposure_data() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = (
            SPLIT_ROOT
            / f"strict_item_cold_balanced_thr1_seed_{seed}"
            / "rq1_per_course_cgrc_export"
            / "per_item_full_cold_cgrc_paper_static.csv"
        )
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"item_id": "target_item_id"})
        frame["model"] = "cgrc"
        frame["seed"] = int(seed)
        frame["source"] = str(path.relative_to(ROOT))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    analysis_dir = FIG_DIR / "p1_topk_motivation_analysis"
    course_macro = pd.read_csv(
        analysis_dir / "course_macro.csv"
    )
    model_summary = pd.read_csv(
        analysis_dir / "model_summary.csv"
    )
    exposure = collect_exposure_data()
    outputs = draw_existing_method_motivation_figure(
        course_macro,
        exposure,
        model_summary,
        BASE,
    )
    diagnostics = build_existing_method_diagnostics(
        course_macro,
        exposure,
        model_summary,
    )
    diagnostics_frame(diagnostics).to_csv(
        FIG_DIR / "mooccube_method_motivation_existing_diagnostics.csv",
        index=False,
    )
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
