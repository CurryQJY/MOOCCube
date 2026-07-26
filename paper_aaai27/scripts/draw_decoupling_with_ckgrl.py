"""Overlay CKG-RL onto the Figure-1 decoupling axes (held-out TEST split).

Figure 1 shows that for the two baselines, ranking quality (course-macro
cold-target NDCG@10) is *decoupled* from pedagogical structure: better-ranked
cold courses are MORE redundant and no safer on prerequisites. The open
question the motivation raises but never answers on-figure is:

    Does CKG-RL break this decoupling -- i.e. reach the region of HIGH NDCG
    with LOW redundancy / LOW prerequisite gap that the baselines cannot?

This script answers it by placing all three models (CGRC, PCGNN, CKG-RL) on the
same two decoupling axes, reusing the exact Figure-1 primitives
(build_real_risk_artifacts, _seed_inputs, analyze_export_record) on the frozen
TEST exports. No retraining; descriptive overlay only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_topk_motivation import (  # noqa: E402
    _seed_inputs,
    analyze_export_record,
    build_real_risk_artifacts,
)

SEEDS = (2025, 2026, 2027)
METRIC_K = 10

MODEL_STYLE = {
    "pcgnn": {"label": "PCGNN", "color": "#1F5AA6", "marker": "s"},
    "cgrc": {"label": "CGRC", "color": "#7B2D3A", "marker": "o"},
    "ckg_rl": {"label": "CKG-RL (ours)", "color": "#C6771A", "marker": "^"},
}

PANELS = [
    ("cold_structural_redundancy", "(a) NDCG@10 vs. structural redundancy", "Structural redundancy (lower is better)"),
    ("cold_prerequisite_gap", "(b) NDCG@10 vs. prerequisite gap", "Prerequisite gap (lower is better)"),
]

GRID = "#D6D6D6"
DARK = "#252525"


def export_paths(root: Path, seed: int) -> dict[str, Path]:
    split_id = f"strict_item_cold_balanced_thr1_seed_{seed}"
    tm = Path(root) / "outputs" / "test_motivation"
    return {
        "cgrc": tm / "cgrc" / split_id / "top20_test.jsonl",
        "pcgnn": tm / "pcgnn" / split_id / "pcgnn_top20.jsonl",
        "ckg_rl": Path(root) / "outputs" / "p1_motivation_topk" / "ckg_rl" / split_id / "top20_cold_test.jsonl",
    }


def analyze_seed(root: Path, seed: int, artifacts, n_items: int) -> pd.DataFrame:
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    _pairs, histories, popularity = _seed_inputs(split_root, seed, n_items)
    perf: dict[tuple, dict] = {}
    struct: dict[tuple, dict] = {}
    cols = [c for c, *_ in PANELS]

    for model, path in export_paths(root, seed).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing export: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                user_id = int(record["user_id"])
                target = int(record["target_item_id"])
                history = histories.get(user_id, np.empty(0, dtype=np.int64))
                record.setdefault("sample_index", 0)
                record["model"] = model
                record["seed"] = int(seed)
                _, list_rows = analyze_export_record(
                    record,
                    history_item_ids=history,
                    train_popularity=popularity,
                    artifacts=artifacts,
                    cutoffs=(METRIC_K,),
                )
                row = list_rows[0]
                key = (model, target)
                s = struct.setdefault(key, {c: 0.0 for c in cols} | {"n": 0})
                for c in cols:
                    s[c] += float(row[c])
                s["n"] += 1
                prefix = [int(i) for i in record["recommended_item_ids"][:METRIC_K]]
                rank = prefix.index(target) + 1 if target in prefix else None
                p = perf.setdefault(key, {"hit": 0.0, "ndcg": 0.0, "count": 0})
                p["count"] += 1
                p["hit"] += float(rank is not None)
                p["ndcg"] += 1.0 / math.log2(rank + 1.0) if rank is not None else 0.0

    rows = []
    for key, p in perf.items():
        model, target = key
        s = struct[key]
        entry = {
            "model": model,
            "seed": int(seed),
            "target_item_id": int(target),
            "recall_at_10": p["hit"] / p["count"],
            "ndcg_at_10": p["ndcg"] / p["count"],
        }
        for c in cols:
            entry[c] = s[c] / s["n"] if s["n"] else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
    })


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean via nonparametric bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_figure(exposed: pd.DataFrame, output_base: Path) -> None:
    """Centroid + 95% bootstrap CI per model on each decoupling axis.

    The scientific claim is about *where each model's cold courses sit on
    average*, not the intra-model scatter; plotting 245 overlapping points
    obscures that. We therefore show the mean position with two-sided
    (x = structural proxy, y = NDCG) bootstrap error bars.
    """
    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.0), constrained_layout=True)
    for ax, (col, title, xlabel) in zip(axes, PANELS):
        for model, style in MODEL_STYLE.items():
            sub = exposed[exposed["model"] == model]
            if sub.empty:
                continue
            x = sub[col].to_numpy(dtype=float)
            y = sub["ndcg_at_10"].to_numpy(dtype=float)
            mx, my = x.mean(), y.mean()
            xlo, xhi = _bootstrap_ci(x, seed=1)
            ylo, yhi = _bootstrap_ci(y, seed=2)
            ax.errorbar(
                mx, my,
                xerr=[[mx - xlo], [xhi - mx]],
                yerr=[[my - ylo], [yhi - my]],
                fmt=style["marker"], markersize=8,
                color=style["color"], ecolor=style["color"],
                elinewidth=1.2, capsize=3, capthick=1.2,
                markeredgecolor="white", markeredgewidth=0.7,
                label=style["label"], zorder=4,
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@10 (ranking quality)")
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="best", frameon=True, framealpha=0.9, handletextpad=0.3)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"[overlay] wrote {output_base.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-base", type=Path,
                        default=ROOT / "paper_aaai27" / "figures" / "mooccube_decoupling_with_ckgrl")
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "paper_aaai27" / "figures" / "test_motivation_analysis" / "course_macro_all_models.csv")
    args = parser.parse_args()

    if args.csv.is_file():
        print(f"[overlay] reusing cached per-course table: {args.csv}", flush=True)
        course = pd.read_csv(args.csv)
    else:
        print("[overlay] building structural artifacts (same as Figure 1) ...", flush=True)
        artifacts, stats = build_real_risk_artifacts(args.root)
        n_items = int(stats["n_items"])
        frames = [analyze_seed(args.root, s, artifacts, n_items) for s in SEEDS]
        course = pd.concat(frames, ignore_index=True)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        course.to_csv(args.csv, index=False)

    exposed = course[course["recall_at_10"] > 0].copy()
    print(f"\n[overlay] exposed cold courses per model: {exposed['model'].value_counts().to_dict()}")
    print("\n[overlay] MEDIAN position on each axis (exposed courses):")
    print("  " + "-" * 70)
    print(f"  {'model':14s} {'NDCG@10':>9s} {'redundancy':>12s} {'prereq_gap':>12s}")
    for model in ("cgrc", "pcgnn", "ckg_rl"):
        sub = exposed[exposed["model"] == model]
        print(f"  {model:14s} {sub['ndcg_at_10'].median():9.3f} "
              f"{sub['cold_structural_redundancy'].median():12.3f} "
              f"{sub['cold_prerequisite_gap'].median():12.3f}")
    print("  " + "-" * 70)

    # does the decoupling (rho) change when CKG-RL is included?
    print("\n[overlay] Spearman(NDCG, redundancy) within each model:")
    for model in ("cgrc", "pcgnn", "ckg_rl"):
        sub = exposed[exposed["model"] == model]
        rho, p = spearmanr(sub["ndcg_at_10"], sub["cold_structural_redundancy"])
        print(f"  {model:14s} rho={rho:+.3f} (p={p:.1e}, n={len(sub)})")

    build_figure(exposed, args.output_base)


if __name__ == "__main__":
    main()
