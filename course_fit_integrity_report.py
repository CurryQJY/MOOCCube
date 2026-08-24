"""Compare target-exclusion course-fit replays against frozen test references."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SEED_DIR = "strict_item_cold_balanced_thr1_seed_{seed}"
FINAL_CSV = "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
PER_ITEM_CSV = "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
AUDIT_JSON = "actor_inference_audit.json"
COLD_ITEM_MACRO_COLUMNS = [
    "full_cold_item_macro_r5",
    "full_cold_item_macro_r10",
    "full_cold_item_macro_r20",
    "full_cold_item_macro_n5",
    "full_cold_item_macro_n10",
    "full_cold_item_macro_n20",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_seed(baseline_root, audited_root, seed):
    baseline_dir = Path(baseline_root) / SEED_DIR.format(seed=seed)
    audited_dir = Path(audited_root) / SEED_DIR.format(seed=seed)
    baseline = pd.read_csv(baseline_dir / FINAL_CSV).iloc[0]
    audited = pd.read_csv(audited_dir / FINAL_CSV).iloc[0]
    differences = [
        abs(float(audited[column]) - float(baseline[column]))
        for column in COLD_ITEM_MACRO_COLUMNS
    ]
    max_abs_metric_diff = round(max(differences, default=0.0), 12)
    per_item_equal = file_sha256(baseline_dir / PER_ITEM_CSV) == file_sha256(
        audited_dir / PER_ITEM_CSV
    )
    audit = json.loads((audited_dir / AUDIT_JSON).read_text(encoding="utf-8"))
    composition = audit.get("refined_item_composition", {})

    checks = {
        "test_target": audit.get("evaluation_target") == "test",
        "course_fit_mode": audit.get("mode") == "course_fit",
        "history_train_only": audit.get("history_all_train_only") is True
        and set(audit.get("history_source_counts", {})) == {"train_only"},
        "target_seen_zero": int(audit.get("target_seen_candidate_pairs", -1)) == 0
        and int(audit.get("target_rows_with_seen_candidate", -1)) == 0,
        "behavior_target_zero": int(audit.get("behavior_target_non_null_calls", -1)) == 0,
        "exclude_target_true": audit.get(
            "effective_course_match_exclude_target"
        )
        == [True],
        "refined_composition_valid": composition
        == {
            "total_unique": 102,
            "train_present": 0,
            "validation_only": 34,
            "test_only": 68,
            "validation_and_test": 0,
            "neither_validation_nor_test": 0,
        },
        "aggregate_metrics_equal": max_abs_metric_diff == 0.0,
        "per_item_sha256_equal": per_item_equal,
    }
    return {
        "seed": int(seed),
        "max_abs_metric_diff": max_abs_metric_diff,
        **checks,
        "passed": all(checks.values()),
    }


def build_report(baseline_root, audited_root, seeds):
    return pd.DataFrame(
        [audit_seed(baseline_root, audited_root, seed) for seed in seeds]
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--audited-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    rows = build_report(args.baseline_root, args.audited_root, args.seeds)
    output_root = Path(args.output_root or args.audited_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_root / "course_fit_integrity_by_seed.csv", index=False)
    passed = bool(len(rows) == len(args.seeds) and rows["passed"].all())
    summary = {
        "passed": passed,
        "seeds_requested": [int(seed) for seed in args.seeds],
        "seeds_checked": int(len(rows)),
        "all_aggregate_metrics_equal": bool(rows["aggregate_metrics_equal"].all()),
        "all_per_item_sha256_equal": bool(rows["per_item_sha256_equal"].all()),
        "all_history_train_only": bool(rows["history_train_only"].all()),
        "all_target_seen_zero": bool(rows["target_seen_zero"].all()),
    }
    (output_root / "course_fit_integrity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(rows.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
