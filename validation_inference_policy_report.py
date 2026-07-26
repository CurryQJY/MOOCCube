"""Aggregate validation cold item-macro inference-policy results."""

import argparse
from pathlib import Path

import pandas as pd


POLICIES = ["static", "ppo", "greedy_similarity", "course_fit", "random"]
METRIC_COLUMNS = {
    "R@5": "full_cold_item_macro_r5",
    "R@10": "full_cold_item_macro_r10",
    "R@20": "full_cold_item_macro_r20",
    "N@5": "full_cold_item_macro_n5",
    "N@10": "full_cold_item_macro_n10",
    "N@20": "full_cold_item_macro_n20",
}


def rank_policies(rows):
    """Rank policies by mean cold item-macro N@10, then R@10."""
    aggregations = {"seeds": ("seed", "nunique")}
    for metric in METRIC_COLUMNS:
        if metric in rows.columns:
            aggregations[f"mean_{metric}"] = (metric, "mean")
            aggregations[f"std_{metric}"] = (metric, "std")
    summary = rows.groupby("policy", as_index=False).agg(**aggregations)
    if "mean_N@10" not in summary or "mean_R@10" not in summary:
        raise ValueError("N@10 and R@10 are required to rank policies")
    static_rows = summary[summary["policy"] == "static"]
    if not static_rows.empty:
        static_n10 = float(static_rows.iloc[0]["mean_N@10"])
        static_r10 = float(static_rows.iloc[0]["mean_R@10"])
        summary["delta_N@10_vs_static"] = summary["mean_N@10"] - static_n10
        summary["delta_R@10_vs_static"] = summary["mean_R@10"] - static_r10
    return summary.sort_values(
        ["mean_N@10", "mean_R@10"],
        ascending=[False, False],
        ignore_index=True,
    )


def collect_policy_rows(root, seeds):
    root = Path(root)
    rows = []
    filename = "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    for policy in POLICIES:
        for seed in seeds:
            path = root / policy / f"strict_item_cold_balanced_thr1_seed_{seed}" / filename
            if not path.exists():
                continue
            raw = pd.read_csv(path).iloc[0]
            row = {"policy": policy, "seed": int(seed)}
            row.update({metric: float(raw[column]) for metric, column in METRIC_COLUMNS.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    details = collect_policy_rows(root, args.seeds)
    if details.empty:
        raise SystemExit("No validation policy results found")
    ranking = rank_policies(details)
    expected_seeds = len(set(args.seeds))
    incomplete = ranking[ranking["seeds"] != expected_seeds]
    if not incomplete.empty:
        names = ",".join(incomplete["policy"].astype(str))
        raise SystemExit(f"Incomplete validation policy results: {names}")
    details.to_csv(root / "validation_policy_by_seed.csv", index=False)
    ranking.to_csv(root / "validation_policy_ranking.csv", index=False)
    (root / "validation_policy_summary.txt").write_text(
        ranking.to_string(index=False) + "\n",
        encoding="utf-8",
    )
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
