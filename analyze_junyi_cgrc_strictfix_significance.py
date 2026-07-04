"""Junyi Ours vs CGRC-paper strictfix per-course paired significance."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT = Path("D:/DeskTop/MOOCCube")
OUT_DIR = ROOT / "outputs/junyi/cgrc_strictfix_significance"
METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]

PAIRS = {
    "2025": (
        ROOT / "outputs/junyi/mask_ablation/mask_tt/strict_item_cold_balanced_thr1_seed_2025/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
        ROOT / "outputs/junyi/mask_ablation/mask_tt/strict_item_cold_balanced_thr1_seed_2025/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
    ),
    "2026": (
        ROOT / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
        ROOT / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
    ),
    "2027": (
        ROOT / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
        ROOT / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
    ),
}


def read_by_item(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["item_id"]: row for row in csv.DictReader(f)}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_ci(diffs: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(samples, 2.5), percentile(samples, 97.5)


def randomization_pvalue(diffs: list[float], n_perm: int, rng: random.Random) -> float:
    obs = abs(mean(diffs))
    extreme = 0
    n = len(diffs)
    for _ in range(n_perm):
        value = sum(d if rng.getrandbits(1) else -d for d in diffs) / n
        if abs(value) >= obs - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def summarize(scope: str, diffs_by_metric: dict[str, list[float]], seed_offset: int) -> list[dict[str, object]]:
    rows = []
    for idx, metric in enumerate(METRICS):
        diffs = diffs_by_metric[metric]
        rng = random.Random(20260610 + seed_offset * 101 + idx)
        ci_low, ci_high = bootstrap_ci(diffs, 10000, rng)
        p_value = randomization_pvalue(diffs, 200000, rng)
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "paired_units": len(diffs),
                "mean_diff": mean(diffs),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "randomization_p": p_value,
                "wins": sum(1 for d in diffs if d > 1e-12),
                "ties": sum(1 for d in diffs if abs(d) <= 1e-12),
                "losses": sum(1 for d in diffs if d < -1e-12),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    summary_rows = []
    pooled = {metric: [] for metric in METRICS}

    for seed_idx, (seed, (ours_path, cgrc_path)) in enumerate(PAIRS.items()):
        ours = read_by_item(ours_path)
        cgrc = read_by_item(cgrc_path)
        if set(ours) != set(cgrc):
            raise ValueError(f"seed={seed} item ids differ: ours={len(ours)} cgrc={len(cgrc)}")
        ids = sorted(ours, key=lambda x: int(x))
        count_mismatch = [item_id for item_id in ids if ours[item_id]["count"] != cgrc[item_id]["count"]]
        if count_mismatch:
            raise ValueError(f"seed={seed} count mismatch items: {count_mismatch[:10]}")

        seed_diffs = {metric: [] for metric in METRICS}
        for item_id in ids:
            row = {"seed": seed, "item_id": item_id, "count": ours[item_id]["count"]}
            for metric in METRICS:
                ours_value = float(ours[item_id][metric])
                cgrc_value = float(cgrc[item_id][metric])
                diff = ours_value - cgrc_value
                row[f"{metric}_ours"] = ours_value
                row[f"{metric}_cgrc"] = cgrc_value
                row[f"{metric}_diff"] = diff
                seed_diffs[metric].append(diff)
                pooled[metric].append(diff)
            detail_rows.append(row)
        summary_rows.extend(summarize(f"seed_{seed}", seed_diffs, seed_idx))

    summary_rows.extend(summarize("pooled_seed_course", pooled, 99))

    detail_path = OUT_DIR / "per_course_ours_vs_cgrc_strictfix_detail.csv"
    summary_path = OUT_DIR / "per_course_ours_vs_cgrc_strictfix_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "item_id", "count"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_ours", f"{metric}_cgrc", f"{metric}_diff"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scope",
            "metric",
            "paired_units",
            "mean_diff",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "randomization_p",
            "wins",
            "ties",
            "losses",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
