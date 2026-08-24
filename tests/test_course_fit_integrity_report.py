import json
from pathlib import Path

import pandas as pd

import course_fit_integrity_report as report


METRICS = {
    "full_cold_item_macro_r5": 0.21,
    "full_cold_item_macro_r10": 0.31,
    "full_cold_item_macro_r20": 0.41,
    "full_cold_item_macro_n5": 0.17,
    "full_cold_item_macro_n10": 0.22,
    "full_cold_item_macro_n20": 0.25,
}


def write_seed(root, seed, metrics=None, audit=None):
    directory = root / f"strict_item_cold_balanced_thr1_seed_{seed}"
    directory.mkdir(parents=True)
    pd.DataFrame([{**METRICS, **(metrics or {})}]).to_csv(
        directory / "final_fullrank_usim_feedback_fast3_content_delta_static.csv",
        index=False,
    )
    (directory / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv").write_text(
        "item_id,count,R@5,R@10,R@20,N@5,N@10,N@20\n1,2,0.5,0.5,1.0,0.4,0.4,0.6\n",
        encoding="utf-8",
    )
    if audit is not None:
        (directory / "actor_inference_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
    return directory


def valid_audit():
    return {
        "evaluation_target": "test",
        "mode": "course_fit",
        "history_all_train_only": True,
        "history_source_counts": {"train_only": 2},
        "target_seen_candidate_pairs": 0,
        "target_rows_with_seen_candidate": 0,
        "behavior_target_non_null_calls": 0,
        "effective_course_match_exclude_target": [True],
        "refined_item_composition": {
            "total_unique": 102,
            "train_present": 0,
            "validation_only": 34,
            "test_only": 68,
            "validation_and_test": 0,
            "neither_validation_nor_test": 0,
        },
    }


def test_audit_seed_passes_identical_outputs_and_clean_provenance(tmp_path):
    baseline = tmp_path / "baseline"
    audited = tmp_path / "audited"
    write_seed(baseline, 2025)
    write_seed(audited, 2025, audit=valid_audit())

    row = report.audit_seed(baseline, audited, 2025)

    assert row["passed"] is True
    assert row["max_abs_metric_diff"] == 0.0
    assert row["per_item_sha256_equal"] is True


def test_audit_seed_rejects_target_seen_candidate(tmp_path):
    baseline = tmp_path / "baseline"
    audited = tmp_path / "audited"
    write_seed(baseline, 2025)
    audit = valid_audit()
    audit["target_seen_candidate_pairs"] = 1
    write_seed(audited, 2025, audit=audit)

    row = report.audit_seed(baseline, audited, 2025)

    assert row["passed"] is False
    assert row["target_seen_zero"] is False


def test_audit_seed_rejects_cold_metric_difference(tmp_path):
    baseline = tmp_path / "baseline"
    audited = tmp_path / "audited"
    write_seed(baseline, 2025)
    write_seed(
        audited,
        2025,
        metrics={"full_cold_item_macro_n10": 0.23},
        audit=valid_audit(),
    )

    row = report.audit_seed(baseline, audited, 2025)

    assert row["passed"] is False
    assert row["max_abs_metric_diff"] == 0.01
