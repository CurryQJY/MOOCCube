import math
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_p1_topk_motivation import (
    CourseMacroAccumulator,
    RiskArtifacts,
    aggregate_course_macro,
    analyze_seed_export_pair,
    analyze_export_record,
    analyze_recommendation_list,
    build_structural_complexity,
    compute_item_risks,
    paired_course_statistics,
    robust_normalize_nonnegative,
    validate_native_export_audit,
    summarize_course_macro,
    validate_export_record,
    validate_export_provenance,
    _model_export_paths,
    _native_export_metrics,
    _seed_inputs,
)


def _artifacts() -> RiskArtifacts:
    prereq = np.zeros((6, 6), dtype=np.float32)
    prereq[3, [0, 1]] = 1.0

    concept = np.zeros((6, 6), dtype=np.float32)
    concept[3, 0] = 0.50
    concept[3, 2] = 0.25

    video = np.zeros((6, 6), dtype=np.float32)
    video[4, 0] = 0.20
    video[0, 4] = 0.75

    same_family = np.zeros((6, 6), dtype=bool)
    same_family[3, 2] = True

    return RiskArtifacts(
        prerequisite_matrix=prereq,
        concept_overlap=concept,
        video_containment=video,
        same_family=same_family,
        structural_complexity=np.array([0.20, 0.40, 0.60, 0.90, 0.70, 0.10]),
    )


def test_robust_normalization_clips_outliers_at_the_requested_quantile():
    values = np.array([0.0, 1.0, 2.0, 100.0])

    normalized = robust_normalize_nonnegative(values, upper_quantile=0.75)

    scale = np.quantile(values, 0.75)
    assert normalized == pytest.approx(np.clip(values / scale, 0.0, 1.0))


def test_item_risks_follow_the_four_model_neutral_definitions():
    risks = compute_item_risks(
        recommended_item_ids=[3, 4, 5],
        history_item_ids=[0, 2],
        artifacts=_artifacts(),
    )

    assert risks["prerequisite_gap"] == pytest.approx([0.5, 0.0, 0.0])
    assert risks["concept_continuity"] == pytest.approx([0.375, 0.0, 0.0])
    assert risks["difficulty_gap"] == pytest.approx([0.5, 0.3, 0.0])
    assert risks["structural_redundancy"] == pytest.approx([1.0, 0.75, 0.0])


def test_readiness_uses_only_the_most_advanced_requested_courses():
    artifacts = _artifacts()
    risks = compute_item_risks(
        recommended_item_ids=[3],
        history_item_ids=[0, 1, 2, 5],
        artifacts=artifacts,
        readiness_k=2,
    )

    assert risks["difficulty_gap"] == pytest.approx([0.4])


def test_readiness_uses_all_available_courses_when_history_is_short():
    risks = compute_item_risks(
        recommended_item_ids=[3],
        history_item_ids=[0, 2],
        artifacts=_artifacts(),
        readiness_k=5,
    )

    assert risks["difficulty_gap"] == pytest.approx([0.5])


def test_binary_prerequisite_matrix_is_reused_without_copying():
    artifacts = _artifacts()
    binary = artifacts.prerequisite_matrix.astype(bool)
    artifacts = RiskArtifacts(
        prerequisite_matrix=binary,
        concept_overlap=artifacts.concept_overlap,
        video_containment=artifacts.video_containment,
        same_family=artifacts.same_family,
        structural_complexity=artifacts.structural_complexity,
    )

    risks = compute_item_risks([3], [0, 2], artifacts)

    assert risks["prerequisite_gap"] == pytest.approx([0.5])
    assert np.asarray(artifacts.prerequisite_matrix, dtype=bool) is binary


def test_list_analysis_reports_cold_proportion_and_cold_only_means():
    result = analyze_recommendation_list(
        recommended_item_ids=[3, 4],
        history_item_ids=[0, 2],
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
    )

    assert result["recommended_count"] == 2
    assert result["cold_recommendation_count"] == 1
    assert result["cold_proportion"] == pytest.approx(0.5)
    assert result["prerequisite_gap"] == pytest.approx(0.25)
    assert result["cold_prerequisite_gap"] == pytest.approx(0.5)
    assert result["cold_concept_continuity"] == pytest.approx(0.375)


def test_list_without_cold_recommendation_keeps_cold_only_risks_missing():
    result = analyze_recommendation_list(
        recommended_item_ids=[4, 5],
        history_item_ids=[0, 2],
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
    )

    assert result["cold_recommendation_count"] == 0
    assert result["cold_proportion"] == 0.0
    assert math.isnan(result["cold_prerequisite_gap"])
    assert math.isnan(result["cold_concept_continuity"])
    assert math.isnan(result["cold_difficulty_gap"])
    assert math.isnan(result["cold_structural_redundancy"])


def test_course_macro_averages_lists_and_does_not_zero_fill_missing_cold_risk():
    lists = pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "cutoff": 10,
                "prerequisite_gap": 0.2,
                "concept_continuity": 0.4,
                "difficulty_gap": 0.3,
                "structural_redundancy": 0.1,
                "cold_proportion": 0.5,
                "cold_prerequisite_gap": 0.8,
                "cold_concept_continuity": 0.2,
                "cold_difficulty_gap": 0.6,
                "cold_structural_redundancy": 0.4,
            },
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "cutoff": 10,
                "prerequisite_gap": 0.4,
                "concept_continuity": 0.6,
                "difficulty_gap": 0.1,
                "structural_redundancy": 0.3,
                "cold_proportion": 0.0,
                "cold_prerequisite_gap": np.nan,
                "cold_concept_continuity": np.nan,
                "cold_difficulty_gap": np.nan,
                "cold_structural_redundancy": np.nan,
            },
        ]
    )

    course = aggregate_course_macro(lists).iloc[0]

    assert course["list_count"] == 2
    assert course["cold_list_count"] == 1
    assert course["prerequisite_gap"] == pytest.approx(0.3)
    assert course["concept_continuity"] == pytest.approx(0.5)
    assert course["cold_proportion"] == pytest.approx(0.25)
    assert course["cold_prerequisite_gap"] == pytest.approx(0.8)


def test_structural_complexity_equal_weights_robustly_normalized_counts():
    prerequisite_counts = np.array([0.0, 1.0, 2.0, 100.0])
    concept_counts = np.array([0.0, 2.0, 4.0, 8.0])

    complexity = build_structural_complexity(
        prerequisite_counts,
        concept_counts,
        upper_quantile=0.75,
    )

    expected = 0.5 * (
        robust_normalize_nonnegative(prerequisite_counts, 0.75)
        + robust_normalize_nonnegative(concept_counts, 0.75)
    )
    assert complexity == pytest.approx(expected)


def _paired_course_rows() -> pd.DataFrame:
    rows = []
    for seed, ckg_value, cgrc_value in [
        (2025, 0.10, 0.30),
        (2026, 0.20, 0.40),
        (2027, 0.30, 0.50),
    ]:
        for target in [11, 12]:
            for model, value in [("ckg_rl", ckg_value), ("cgrc", cgrc_value)]:
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "target_item_id": target,
                        "cutoff": 10,
                        "list_count": 1,
                        "cold_list_count": 1,
                        "prerequisite_gap": value,
                        "concept_continuity": 1.0 - value,
                        "difficulty_gap": value,
                        "structural_redundancy": value,
                        "cold_proportion": 0.5,
                        "cold_prerequisite_gap": value,
                        "cold_concept_continuity": 1.0 - value,
                        "cold_difficulty_gap": value,
                        "cold_structural_redundancy": value,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_statistics_use_matched_seed_target_rows_and_fixed_rng():
    course = _paired_course_rows()

    first = paired_course_statistics(
        course,
        n_bootstrap=500,
        n_permutations=1000,
        random_seed=77,
    )
    second = paired_course_statistics(
        course,
        n_bootstrap=500,
        n_permutations=1000,
        random_seed=77,
    )

    pd.testing.assert_frame_equal(first, second)
    prereq = first[
        (first["cutoff"] == 10)
        & (first["metric"] == "prerequisite_gap")
    ].iloc[0]
    assert prereq["pair_count"] == 6
    assert prereq["mean_difference_ckg_rl_minus_cgrc"] == pytest.approx(-0.2)
    assert prereq["bootstrap_ci_low"] == pytest.approx(-0.2)
    assert prereq["bootstrap_ci_high"] == pytest.approx(-0.2)
    assert prereq["interpretation"] == "supports"


def test_paired_statistics_supports_primary_and_secondary_model_comparisons():
    course = _paired_course_rows()
    pcgnn = course[course["model"].eq("ckg_rl")].copy()
    pcgnn["model"] = "pcgnn"
    for metric in (
        "prerequisite_gap",
        "difficulty_gap",
        "structural_redundancy",
        "cold_prerequisite_gap",
        "cold_difficulty_gap",
        "cold_structural_redundancy",
    ):
        pcgnn[metric] += 0.1
    for metric in ("concept_continuity", "cold_concept_continuity"):
        pcgnn[metric] -= 0.1
    course = pd.concat([course, pcgnn], ignore_index=True)

    paired = paired_course_statistics(
        course,
        comparisons=(
            ("ckg_rl", "pcgnn", "primary"),
            ("ckg_rl", "cgrc", "secondary"),
        ),
        n_bootstrap=200,
        n_permutations=400,
        random_seed=77,
    )

    prereq = paired[
        (paired["cutoff"].eq(10))
        & (paired["metric"].eq("prerequisite_gap"))
    ].sort_values("comparison_role")
    assert set(prereq["comparison"]) == {
        "ckg_rl_vs_cgrc",
        "ckg_rl_vs_pcgnn",
    }
    assert set(prereq["treatment"]) == {"ckg_rl"}
    assert set(prereq["baseline"]) == {"pcgnn", "cgrc"}
    primary = prereq[prereq["comparison_role"].eq("primary")].iloc[0]
    secondary = prereq[prereq["comparison_role"].eq("secondary")].iloc[0]
    assert primary["mean_difference"] == pytest.approx(-0.1)
    assert secondary["mean_difference"] == pytest.approx(-0.2)


def test_model_summary_uses_seed_means_and_reports_three_seed_sd():
    seed_summary, model_summary = summarize_course_macro(_paired_course_rows())

    assert len(seed_summary) == 6
    ckg = model_summary[
        (model_summary["model"] == "ckg_rl")
        & (model_summary["cutoff"] == 10)
    ].iloc[0]
    assert ckg["seed_count"] == 3
    assert ckg["prerequisite_gap_mean"] == pytest.approx(0.2)
    assert ckg["prerequisite_gap_sd"] == pytest.approx(0.1)


def _export_record() -> dict:
    return {
        "model": "ckg_rl",
        "seed": 2025,
        "sample_index": 0,
        "user_id": 7,
        "target_item_id": 3,
        "target_popularity": 0,
        "recommended_item_ids": [3, 4, 5],
        "recommended_scores": [0.9, 0.8, 0.7],
    }


def test_export_record_validation_checks_metadata_order_and_seen_leakage():
    validate_export_record(
        _export_record(),
        expected_model="ckg_rl",
        expected_seed=2025,
        expected_sample_index=0,
        expected_pair=(7, 3),
        expected_target_popularity=0,
        history_item_ids=[0, 2],
        expected_top_k=3,
    )

    leaked = _export_record()
    leaked["recommended_item_ids"] = [3, 2, 5]
    with pytest.raises(ValueError, match="seen-item leakage"):
        validate_export_record(
            leaked,
            expected_model="ckg_rl",
            expected_seed=2025,
            expected_sample_index=0,
            expected_pair=(7, 3),
            expected_target_popularity=0,
            history_item_ids=[0, 2],
            expected_top_k=3,
        )


def test_export_record_analysis_emits_recommendation_and_cutoff_rows():
    recommendations, lists = analyze_export_record(
        _export_record(),
        history_item_ids=[0, 2],
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
        cutoffs=(2, 3),
    )

    assert [row["rank"] for row in recommendations] == [1, 2, 3]
    assert recommendations[0]["recommended_item_id"] == 3
    assert recommendations[0]["is_cold"] == 1
    assert [row["cutoff"] for row in lists] == [2, 3]
    assert lists[0]["cold_proportion"] == pytest.approx(0.5)
    assert lists[1]["cold_proportion"] == pytest.approx(1 / 3)


def test_export_record_analysis_accepts_precomputed_readiness():
    _, lists = analyze_export_record(
        _export_record(),
        history_item_ids=[0, 2],
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
        cutoffs=(3,),
        precomputed_readiness=0.8,
    )

    assert lists[0]["difficulty_gap"] == pytest.approx((0.1 + 0.0 + 0.0) / 3)


def test_streaming_course_accumulator_matches_dataframe_aggregation():
    lists = pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "cutoff": 10,
                "prerequisite_gap": 0.2,
                "concept_continuity": 0.4,
                "difficulty_gap": 0.3,
                "structural_redundancy": 0.1,
                "cold_proportion": 0.5,
                "cold_prerequisite_gap": 0.8,
                "cold_concept_continuity": 0.2,
                "cold_difficulty_gap": 0.6,
                "cold_structural_redundancy": 0.4,
            },
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "cutoff": 10,
                "prerequisite_gap": 0.4,
                "concept_continuity": 0.6,
                "difficulty_gap": 0.1,
                "structural_redundancy": 0.3,
                "cold_proportion": 0.0,
                "cold_prerequisite_gap": np.nan,
                "cold_concept_continuity": np.nan,
                "cold_difficulty_gap": np.nan,
                "cold_structural_redundancy": np.nan,
            },
        ]
    )
    accumulator = CourseMacroAccumulator()
    for row in lists.to_dict(orient="records"):
        accumulator.update(row)

    expected = aggregate_course_macro(lists)
    actual = accumulator.to_frame()

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_seed_pair_streams_both_models_against_the_same_split_row(tmp_path):
    ckg_record = _export_record()
    cgrc_record = dict(ckg_record)
    cgrc_record["model"] = "cgrc"
    cgrc_record["recommended_item_ids"] = [4, 3, 5]
    cgrc_record["recommended_scores"] = [0.85, 0.75, 0.65]
    ckg_path = tmp_path / "ckg.jsonl"
    cgrc_path = tmp_path / "cgrc.jsonl"
    ckg_path.write_text(json.dumps(ckg_record) + "\n", encoding="utf-8")
    cgrc_path.write_text(json.dumps(cgrc_record) + "\n", encoding="utf-8")
    recommendation_rows = []
    list_rows = []
    accumulator = CourseMacroAccumulator()

    audit = analyze_seed_export_pair(
        seed=2025,
        ckg_rl_path=ckg_path,
        cgrc_path=cgrc_path,
        expected_pairs=[(7, 3)],
        histories={7: [0, 2]},
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
        recommendation_sink=recommendation_rows.append,
        list_sink=list_rows.append,
        course_accumulator=accumulator,
        expected_top_k=3,
        cutoffs=(2, 3),
        metric_k=2,
    )

    assert len(recommendation_rows) == 6
    assert len(list_rows) == 4
    assert len(accumulator.to_frame()) == 4
    assert audit["ckg_rl"]["record_count"] == 1
    assert audit["ckg_rl"]["R@2"] == 1.0
    assert audit["ckg_rl"]["N@2"] == 1.0
    assert audit["cgrc"]["R@2"] == 1.0
    assert audit["cgrc"]["N@2"] == pytest.approx(1.0 / math.log2(3.0))


def test_seed_export_analysis_streams_three_models_against_the_same_split_row(tmp_path):
    records = {}
    rankings = {
        "ckg_rl": ([3, 4, 5], [0.90, 0.80, 0.70]),
        "pcgnn": ([4, 3, 5], [0.85, 0.75, 0.65]),
        "cgrc": ([4, 5, 3], [0.83, 0.73, 0.63]),
    }
    model_paths = {}
    for model, (items, scores) in rankings.items():
        record = dict(_export_record())
        record["model"] = model
        record["recommended_item_ids"] = items
        record["recommended_scores"] = scores
        records[model] = record
        path = tmp_path / f"{model}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        model_paths[model] = path
    recommendation_rows = []
    list_rows = []
    accumulator = CourseMacroAccumulator()

    audit = analyze_seed_export_pair(
        seed=2025,
        model_paths=model_paths,
        expected_pairs=[(7, 3)],
        histories={7: [0, 2]},
        train_popularity=np.array([5, 5, 5, 0, 7, 3]),
        artifacts=_artifacts(),
        recommendation_sink=recommendation_rows.append,
        list_sink=list_rows.append,
        course_accumulator=accumulator,
        expected_top_k=3,
        cutoffs=(2, 3),
        metric_k=2,
    )

    assert len(recommendation_rows) == 9
    assert len(list_rows) == 6
    assert len(accumulator.to_frame()) == 6
    assert set(audit) == {"ckg_rl", "pcgnn", "cgrc"}
    assert audit["ckg_rl"]["N@2"] == pytest.approx(1.0)
    assert audit["pcgnn"]["N@2"] == pytest.approx(1.0 / math.log2(3.0))
    assert audit["cgrc"]["N@2"] == pytest.approx(0.0)


def test_seed_export_analysis_pair_aligns_a_model_with_native_record_order(tmp_path):
    first = _export_record()
    second = {
        **_export_record(),
        "sample_index": 1,
        "user_id": 8,
        "target_item_id": 4,
        "target_popularity": 0,
        "recommended_item_ids": [4, 3, 5],
        "recommended_scores": [0.88, 0.78, 0.68],
    }
    model_paths = {}
    for model in ("ckg_rl", "cgrc"):
        rows = []
        for record in (first, second):
            rows.append(json.dumps({**record, "model": model}))
        path = tmp_path / f"{model}.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        model_paths[model] = path
    pcgnn_second = {**second, "model": "pcgnn", "sample_index": 0}
    pcgnn_first = {**first, "model": "pcgnn", "sample_index": 1}
    pcgnn_path = tmp_path / "pcgnn.jsonl"
    pcgnn_path.write_text(
        "\n".join(map(json.dumps, (pcgnn_second, pcgnn_first))) + "\n",
        encoding="utf-8",
    )
    model_paths["pcgnn"] = pcgnn_path
    recommendation_rows = []
    accumulator = CourseMacroAccumulator()

    audit = analyze_seed_export_pair(
        seed=2025,
        model_paths=model_paths,
        pair_keyed_models=("pcgnn",),
        expected_pairs=[(7, 3), (8, 4)],
        histories={7: [0, 2], 8: [0, 1]},
        train_popularity=np.array([5, 5, 5, 0, 0, 3]),
        artifacts=_artifacts(),
        recommendation_sink=recommendation_rows.append,
        list_sink=lambda row: None,
        course_accumulator=accumulator,
        expected_top_k=3,
        cutoffs=(2,),
        metric_k=2,
    )

    pcgnn_users = [
        row["user_id"] for row in recommendation_rows if row["model"] == "pcgnn"
    ]
    assert pcgnn_users == [7, 7, 7, 8, 8, 8]
    assert audit["pcgnn"]["record_count"] == 2


def _file_binding(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_pcgnn_native_metrics_are_loaded_from_checkpoint_replay(tmp_path):
    export_dir = (
        tmp_path
        / "paper_aaai27"
        / "baseline_sources"
        / "_pcgnn_strict"
        / "mooccube_seed2025_full_formal_kg_warm"
        / "p1_top20_export"
    )
    export_dir.mkdir(parents=True)
    replay = {
        "record_count": 11,
        "metrics": {
            "rows_full_cold": 11,
            "count_full_cold_item_macro": 3,
            "full_cold_item_macro": {"R@10": 0.25, "N@10": 0.125},
        },
    }
    (export_dir / "pcgnn_replay_result.json").write_text(
        json.dumps(replay),
        encoding="utf-8",
    )

    native = _native_export_metrics(tmp_path, 2025, "pcgnn")

    assert native == {
        "count_full_cold": 11,
        "count_full_cold_item_macro": 3,
        "course_macro_R@10": pytest.approx(0.25),
        "course_macro_N@10": pytest.approx(0.125),
    }


def test_ckgrl_ablation_native_metrics_use_the_variant_replay_directory(tmp_path):
    model = "ckg_rl_wo_simulator"
    eval_dir = (
        tmp_path
        / "outputs"
        / "p1_motivation_topk"
        / model
        / "strict_item_cold_balanced_thr1_seed_2025"
        / "eval"
    )
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "full_cold_count": 11,
                "full_cold_r10": 0.2,
                "full_cold_n10": 0.1,
                "full_cold_item_macro_r10": 0.3,
                "full_cold_item_macro_n10": 0.15,
            }
        ]
    ).to_csv(
        eval_dir / "final_fullrank_usim_feedback_fast3_content_delta_static.csv",
        index=False,
    )

    native = _native_export_metrics(tmp_path, 2025, model)

    assert native == {
        "count_full_cold": 11,
        "R@10": pytest.approx(0.2),
        "N@10": pytest.approx(0.1),
        "course_macro_R@10": pytest.approx(0.3),
        "course_macro_N@10": pytest.approx(0.15),
    }


def test_model_export_paths_keep_pcgnn_in_its_formal_export_directory(tmp_path):
    paths = _model_export_paths(tmp_path, 2025)

    assert set(paths) == {"ckg_rl", "pcgnn", "cgrc"}
    assert paths["ckg_rl"] == (
        tmp_path
        / "outputs"
        / "p1_motivation_topk"
        / "ckg_rl"
        / "strict_item_cold_balanced_thr1_seed_2025"
        / "top20_cold_test.jsonl"
    )
    assert paths["cgrc"] == (
        tmp_path
        / "outputs"
        / "p1_motivation_topk"
        / "cgrc"
        / "strict_item_cold_balanced_thr1_seed_2025"
        / "top20_cold_test.jsonl"
    )
    assert paths["pcgnn"] == (
        tmp_path
        / "paper_aaai27"
        / "baseline_sources"
        / "_pcgnn_strict"
        / "mooccube_seed2025_full_formal_kg_warm"
        / "p1_top20_export"
        / "pcgnn_top20.jsonl"
    )


def test_pcgnn_provenance_binds_topk_replay_and_unchanged_checkpoint(tmp_path):
    export_dir = tmp_path / "p1_top20_export"
    export_dir.mkdir()
    topk = export_dir / "pcgnn_top20.jsonl"
    replay = export_dir / "pcgnn_replay_result.json"
    checkpoint = tmp_path / "best_model.pt"
    report = tmp_path / "report.json"
    config = tmp_path / "config.yaml"
    split = tmp_path / "static_test.pkl"
    script = tmp_path / "exporter.py"
    topk.write_text('{"sample_index": 0}\n', encoding="utf-8")
    replay.write_text('{"record_count": 1}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    report.write_text("{}\n", encoding="utf-8")
    config.write_text("dataset: demo\n", encoding="utf-8")
    split.write_bytes(b"split")
    script.write_text("print('ok')\n", encoding="utf-8")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "model": "pcgnn",
        "seed": 2025,
        "top_k": 20,
        "record_count": 1,
        "status": "checkpoint_replay_valid",
        "native_report_test_reproduced": True,
        "checkpoint": {
            **_file_binding(checkpoint),
            "sha256_before": checkpoint_digest,
            "sha256_after": checkpoint_digest,
        },
        "report": _file_binding(report),
        "config": _file_binding(config),
        "split_files": [_file_binding(split)],
        "script_files": [_file_binding(script)],
        "topk_output": _file_binding(topk),
        "replay_result": _file_binding(replay),
    }
    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    validated = validate_export_provenance(
        tmp_path,
        model="pcgnn",
        seed=2025,
        export_path=topk,
        expected_count=1,
    )

    assert validated["status"] == "checkpoint_replay_valid"
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="PCGNN checkpoint provenance mismatch"):
        validate_export_provenance(
            tmp_path,
            model="pcgnn",
            seed=2025,
            export_path=topk,
            expected_count=1,
        )


def test_ckgrl_ablation_provenance_uses_the_ckgrl_manifest_schema(tmp_path):
    model = "ckg_rl_wo_course_reward"
    export_dir = tmp_path / "variant"
    eval_dir = export_dir / "eval"
    eval_dir.mkdir(parents=True)
    topk = export_dir / "top20_cold_test.jsonl"
    checkpoint = tmp_path / "finished.pt"
    source_manifest = tmp_path / "source_manifest.json"
    topk.write_text('{"sample_index": 0}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    source_manifest.write_text(
        json.dumps({"split": {"seed": 2025}}),
        encoding="utf-8",
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "model": model,
        "seed": 2025,
        "top_k": 20,
        "record_count": 1,
        "topk_output": str(topk.resolve()),
        "checkpoint_hashes": {str(checkpoint.resolve()): checkpoint_hash},
        "source_manifest": str(source_manifest.resolve()),
    }
    (eval_dir / "p1_topk_export_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    validated = validate_export_provenance(
        tmp_path,
        model=model,
        seed=2025,
        export_path=topk,
        expected_count=1,
    )

    assert validated["model"] == model


def test_native_coverage_gate_rejects_count_mismatch():
    audit = {
        "record_count": 10,
        "R@10": 0.2,
        "N@10": 0.1,
        "course_macro_R@10": 0.3,
        "course_macro_N@10": 0.2,
    }
    native = {
        "count_full_cold": 9,
        "R@10": 0.2,
        "N@10": 0.1,
        "course_macro_R@10": 0.3,
        "course_macro_N@10": 0.2,
    }

    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_native_export_audit(audit, native)


def test_native_metric_gate_compares_only_metrics_provided_by_the_model():
    audit = {
        "record_count": 10,
        "R@10": 0.2,
        "N@10": 0.1,
        "course_macro_R@10": 0.3,
        "course_macro_N@10": 0.2,
    }
    native = {
        "count_full_cold": 10,
        "count_full_cold_item_macro": 2,
        "course_macro_R@10": 0.3,
        "course_macro_N@10": 0.2,
    }

    deltas = validate_native_export_audit(audit, native)

    assert deltas == {
        "course_macro_R@10": pytest.approx(0.0),
        "course_macro_N@10": pytest.approx(0.0),
    }


def test_direct_script_context_can_resolve_repository_modules(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "paper_aaai27" / "scripts" / "analyze_p1_topk_motivation.py"
    relation_dir = tmp_path / "relations"
    relation_dir.mkdir()
    (relation_dir / "course-concept.json").write_text("", encoding="utf-8")
    code = f"""
import runpy
import sys
from pathlib import Path

import pandas as pd

root = Path({str(root)!r}).resolve()
script = Path({str(script)!r}).resolve()
filtered = []
for entry in sys.path:
    try:
        if Path(entry or '.').resolve() == root:
            continue
    except OSError:
        pass
    filtered.append(entry)
sys.path = [str(script.parent)] + filtered
namespace = runpy.run_path(str(script), run_name='p1_direct_test')
frame = pd.DataFrame(columns=['i_idx', 'course_id'])
namespace['_item_concept_counts'](frame, Path({str(relation_dir)!r}), 0)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_seed_inputs_builds_array_histories_with_groupby_apply(tmp_path):
    split_root = tmp_path / "splits"
    seed_root = split_root / "strict_item_cold_balanced_thr1_seed_2025"
    seed_root.mkdir(parents=True)
    pd.DataFrame(
        {"u_idx": [7, 7, 8], "i_idx": [2, 0, 1]}
    ).to_pickle(seed_root / "static_train.pkl")
    pd.DataFrame(
        {
            "u_idx": [7, 8],
            "i_idx": [3, 4],
            "_split_source": ["strict_item_cold_test", "other"],
        }
    ).to_pickle(seed_root / "static_test.pkl")

    pairs, histories, popularity = _seed_inputs(split_root, 2025, n_items=6)

    assert pairs == [(7, 3)]
    assert histories[7].tolist() == [0, 2]
    assert histories[8].tolist() == [1]
    assert popularity.tolist() == [1, 1, 1, 0, 0, 0]
