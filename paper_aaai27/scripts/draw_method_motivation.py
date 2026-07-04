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

from fast3_delta.course_artifacts import build_course_artifacts


PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
BASE = FIG_DIR / "mooccube_method_motivation"

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
        }
    )


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


def main() -> None:
    risk, exposure, summary = collect_data()
    draw(risk, exposure)
    risk.to_csv(FIG_DIR / "mooccube_method_motivation_course_risk_data.csv", index=False)
    exposure.to_csv(FIG_DIR / "mooccube_method_motivation_cgrc_exposure_data.csv", index=False)
    summary.to_csv(FIG_DIR / "mooccube_method_motivation_summary.csv", index=False)
    print(f"Wrote {BASE.with_suffix('.pdf')}")
    print(f"Wrote {BASE.with_suffix('.png')}")
    print(f"Wrote {BASE.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
