"""Report the locked PPO-loss training by inference-policy factorial."""

import argparse
import json
from pathlib import Path

import pandas as pd


CELLS = ("on_static", "on_course_fit", "off_static", "off_course_fit")
METRIC_COLUMNS = {
    "R@5": "full_cold_item_macro_r5",
    "R@10": "full_cold_item_macro_r10",
    "R@20": "full_cold_item_macro_r20",
    "N@5": "full_cold_item_macro_n5",
    "N@10": "full_cold_item_macro_n10",
    "N@20": "full_cold_item_macro_n20",
}
FINAL_CSV = "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
AUDIT_JSON = "actor_inference_audit.json"
SEED_DIR = "strict_item_cold_balanced_thr1_seed_{seed}"


def summarize_factorial(rows):
    pivot = rows.pivot(index=["seed", "metric"], columns="cell", values="value")
    missing = [cell for cell in CELLS if cell not in pivot.columns]
    if missing:
        raise ValueError(f"Factorial rows are missing cells: {missing}")
    pivot = pivot.reset_index()
    pivot["training_effect_static"] = pivot["on_static"] - pivot["off_static"]
    pivot["training_effect_course_fit"] = (
        pivot["on_course_fit"] - pivot["off_course_fit"]
    )
    pivot["inference_effect_ppo_on"] = pivot["on_course_fit"] - pivot["on_static"]
    pivot["inference_effect_ppo_off"] = (
        pivot["off_course_fit"] - pivot["off_static"]
    )
    pivot["interaction"] = (
        pivot["inference_effect_ppo_on"] - pivot["inference_effect_ppo_off"]
    )

    value_columns = list(CELLS) + [
        "training_effect_static",
        "training_effect_course_fit",
        "inference_effect_ppo_on",
        "inference_effect_ppo_off",
        "interaction",
    ]
    records = []
    for metric, group in pivot.groupby("metric", sort=False):
        record = {"metric": metric, "seeds": int(group["seed"].nunique())}
        for column in value_columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_std"] = float(group[column].std(ddof=1))
        records.append(record)
    summary = pd.DataFrame(records)
    metric_order = {metric: index for index, metric in enumerate(METRIC_COLUMNS)}
    pivot = pivot.sort_values(
        ["metric", "seed"], key=lambda series: series.map(metric_order) if series.name == "metric" else series
    ).reset_index(drop=True)
    summary = summary.sort_values(
        "metric", key=lambda series: series.map(metric_order)
    ).reset_index(drop=True)
    return pivot, summary


def load_factorial(roots, seeds):
    rows = []
    for cell in CELLS:
        if cell not in roots:
            raise KeyError(f"Missing root for cell {cell}")
        for seed in seeds:
            directory = Path(roots[cell]) / SEED_DIR.format(seed=seed)
            path = directory / FINAL_CSV
            if not path.exists():
                raise FileNotFoundError(f"Missing {cell} seed={seed}: {path}")
            if cell == "off_course_fit":
                validate_off_course_fit_audit(directory / AUDIT_JSON)
            raw = pd.read_csv(path)
            if len(raw) != 1:
                raise ValueError(f"Expected one result row for {cell} seed={seed}: {path}")
            result = raw.iloc[0]
            for metric, column in METRIC_COLUMNS.items():
                rows.append(
                    {
                        "seed": int(seed),
                        "metric": metric,
                        "cell": cell,
                        "value": float(result[column]),
                    }
                )
    return pd.DataFrame(rows)


def validate_off_course_fit_audit(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing off_course_fit audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    composition = audit.get("refined_item_composition", {})
    checks = {
        "evaluation_target": audit.get("evaluation_target") == "test",
        "mode": audit.get("mode") == "course_fit",
        "actor_calls": int(audit.get("actor_calls", -1)) == 0,
        "episode_calls": int(audit.get("episode_calls", 0)) > 0,
        "history_all_train_only": audit.get("history_all_train_only") is True,
        "target_seen_candidate_pairs": int(
            audit.get("target_seen_candidate_pairs", -1)
        )
        == 0,
        "refined_total": int(composition.get("total_unique", -1)) == 102,
        "refined_train_present": int(composition.get("train_present", -1)) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid off_course_fit audit {path}: {', '.join(failed)}")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    roots = {
        cell: getattr(args, f"{cell}_root")
        for cell in CELLS
    }
    rows = load_factorial(roots, args.seeds)
    by_seed, summary = summarize_factorial(rows)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(output_root / "ppo_loss_factorial_by_seed.csv", index=False)
    summary.to_csv(output_root / "ppo_loss_factorial_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
