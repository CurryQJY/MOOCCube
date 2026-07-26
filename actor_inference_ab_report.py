"""Build aggregate and per-course reports for Actor inference A/B runs."""

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
FULLRANK_COLUMNS = {
    "R@5": "full_cold_item_macro_r5",
    "R@10": "full_cold_item_macro_r10",
    "R@20": "full_cold_item_macro_r20",
    "N@5": "full_cold_item_macro_n5",
    "N@10": "full_cold_item_macro_n10",
    "N@20": "full_cold_item_macro_n20",
}


def compare_per_item(static_path, actor_path, seed):
    """Join per-course arm outputs and summarize Actor-minus-static changes."""
    static = pd.read_csv(static_path)
    actor = pd.read_csv(actor_path)
    merged = static.merge(
        actor,
        on=["item_id", "count"],
        suffixes=("_static", "_actor"),
        validate="one_to_one",
    ).sort_values("item_id", ignore_index=True)

    summary = {"seed": int(seed), "cold_items": int(len(merged))}
    for metric in METRICS:
        static_col = f"{metric}_static"
        actor_col = f"{metric}_actor"
        if static_col not in merged or actor_col not in merged:
            continue
        delta_col = f"delta_{metric}"
        merged[delta_col] = merged[actor_col] - merged[static_col]
        summary[delta_col] = float(merged[delta_col].mean())
        summary[f"positive_{metric}"] = float((merged[delta_col] > 0).mean())
    return merged, summary


def compare_fullrank(static_path, actor_path, seed):
    """Return long-form item-macro metrics for one seed."""
    static = pd.read_csv(static_path).iloc[0]
    actor = pd.read_csv(actor_path).iloc[0]
    rows = []
    for metric, column in FULLRANK_COLUMNS.items():
        static_value = float(static[column])
        actor_value = float(actor[column])
        rows.append(
            {
                "seed": int(seed),
                "metric": metric,
                "static": static_value,
                "actor": actor_value,
                "delta": actor_value - static_value,
            }
        )
    return rows


def _seed_dir(root, mode, seed):
    return root / mode / f"strict_item_cold_balanced_thr1_seed_{seed}"


def _read_audit(path, seed, mode):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"seed": int(seed), "mode": mode, **payload}


def build_reports(root, seeds):
    """Write all complete static/Actor seed-pair reports under ``root``."""
    root = Path(root)
    metric_rows = []
    item_details = []
    item_summaries = []
    audit_rows = []
    completed = []
    fullrank_name = "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    per_item_name = "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"

    for seed in seeds:
        static_dir = _seed_dir(root, "static", seed)
        actor_dir = _seed_dir(root, "actor", seed)
        required = [
            static_dir / fullrank_name,
            actor_dir / fullrank_name,
            static_dir / per_item_name,
            actor_dir / per_item_name,
        ]
        if not all(path.exists() for path in required):
            continue

        metric_rows.extend(compare_fullrank(required[0], required[1], seed))
        detail, summary = compare_per_item(required[2], required[3], seed)
        item_details.append(detail.assign(seed=int(seed)))
        item_summaries.append(summary)
        for mode, directory in (("static", static_dir), ("actor", actor_dir)):
            audit = _read_audit(directory / "actor_inference_audit.json", seed, mode)
            if audit is not None:
                audit_rows.append(audit)
        completed.append(int(seed))

    if not completed:
        raise ValueError("No complete static/actor seed pairs found")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "actor_inference_ab_metrics.csv", index=False)
    aggregate = (
        metrics.groupby("metric", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            static_mean=("static", "mean"),
            actor_mean=("actor", "mean"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(root / "actor_inference_ab_aggregate.csv", index=False)
    pd.DataFrame(item_summaries).to_csv(root / "actor_inference_ab_seed_summary.csv", index=False)
    pd.concat(item_details, ignore_index=True).to_csv(root / "actor_inference_ab_per_item.csv", index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(root / "actor_inference_ab_audit.csv", index=False)

    lines = [f"completed_seeds={','.join(map(str, completed))}"]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"{row.metric}: static={row.static_mean:.9f} actor={row.actor_mean:.9f} "
            f"delta={row.delta_mean:+.9f}"
        )
    (root / "actor_inference_ab_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    completed = build_reports(args.root, args.seeds)
    print(f">> Actor inference A/B report: completed_seeds={completed}")


if __name__ == "__main__":
    main()
