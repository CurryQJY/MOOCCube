import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_p1_topk_motivation import RiskArtifacts
from paper_aaai27.scripts.analyze_validation_motivation import (
    analyze_validation_seed,
    seed_stratified_interval,
    summarize_validation_course_rows,
    validate_validation_export_record,
    validation_seed_inputs,
)


def _artifacts() -> RiskArtifacts:
    prerequisite = np.zeros((6, 6), dtype=bool)
    prerequisite[3, 0] = True
    prerequisite[4, 1] = True
    concept = np.eye(6, dtype=np.float64)
    concept[3, 0] = concept[0, 3] = 0.6
    concept[4, 1] = concept[1, 4] = 0.4
    return RiskArtifacts(
        prerequisite_matrix=prerequisite,
        concept_overlap=concept,
        video_containment=np.zeros((6, 6), dtype=np.float64),
        same_family=np.eye(6, dtype=bool),
        structural_complexity=np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.4]),
    )


def _record(model: str, recommendations, scores=None, analysis_split="validation"):
    if scores is None:
        scores = [0.9, 0.8, 0.7]
    return {
        "model": model,
        "seed": 2025,
        "analysis_split": analysis_split,
        "sample_index": 0,
        "user_id": 7,
        "target_item_id": 3,
        "target_popularity": 0,
        "recommended_item_ids": list(recommendations),
        "recommended_scores": list(scores),
    }


def test_validation_seed_inputs_use_only_validation_cold_rows_and_train_history(tmp_path):
    split_root = tmp_path / "splits"
    seed_root = split_root / "strict_item_cold_balanced_thr1_seed_2025"
    seed_root.mkdir(parents=True)
    pd.DataFrame(
        {"u_idx": [7, 7, 8], "i_idx": [1, 0, 2]}
    ).to_pickle(seed_root / "static_train.pkl")
    pd.DataFrame(
        {
            "u_idx": [7, 8],
            "i_idx": [3, 4],
            "_split_source": ["strict_item_cold_val", "other"],
        }
    ).to_pickle(seed_root / "static_val.pkl")
    pd.DataFrame(
        {
            "u_idx": [7],
            "i_idx": [5],
            "_split_source": ["strict_item_cold_test"],
        }
    ).to_pickle(seed_root / "static_test.pkl")

    pairs, histories, popularity = validation_seed_inputs(split_root, 2025, n_items=6)

    assert pairs == [(7, 3)]
    assert histories[7].tolist() == [0, 1]
    assert histories[8].tolist() == [2]
    assert popularity.tolist() == [1, 1, 1, 0, 0, 0]


def test_validation_record_rejects_test_metadata_and_invalid_topk():
    kwargs = {
        "expected_model": "pcgnn",
        "expected_seed": 2025,
        "expected_sample_index": 0,
        "expected_pair": (7, 3),
        "expected_target_popularity": 0,
        "history_item_ids": [0, 1],
        "expected_top_k": 3,
    }

    validate_validation_export_record(_record("pcgnn", [3, 4, 5]), **kwargs)
    with pytest.raises(ValueError, match="validation-only"):
        validate_validation_export_record(
            _record("pcgnn", [3, 4, 5], analysis_split="test"),
            **kwargs,
        )
    with pytest.raises(ValueError, match="descending"):
        validate_validation_export_record(
            _record("pcgnn", [3, 4, 5], scores=[0.8, 0.9, 0.7]),
            **kwargs,
        )


def test_validation_seed_analysis_aggregates_lists_before_courses(tmp_path):
    paths = {}
    for model, items in {"pcgnn": [3, 2, 4], "cgrc": [2, 5, 3]}.items():
        path = tmp_path / f"{model}.jsonl"
        path.write_text(json.dumps(_record(model, items)) + "\n", encoding="utf-8")
        paths[model] = path

    course_rows, audit = analyze_validation_seed(
        seed=2025,
        expected_pairs=[(7, 3)],
        histories={7: np.array([0, 1])},
        train_popularity=np.array([2, 2, 2, 0, 0, 0]),
        artifacts=_artifacts(),
        model_paths=paths,
        expected_top_k=3,
        metric_k=2,
    )

    assert set(course_rows["model"]) == {"pcgnn", "cgrc"}
    assert set(course_rows["analysis_split"]) == {"validation"}
    pcgnn = course_rows.loc[course_rows["model"].eq("pcgnn")].iloc[0]
    cgrc = course_rows.loc[course_rows["model"].eq("cgrc")].iloc[0]
    assert pcgnn["list_count"] == 1
    assert pcgnn["ndcg_at_10"] == pytest.approx(1.0)
    assert cgrc["ndcg_at_10"] == pytest.approx(0.0)
    assert pcgnn["cold_proportion"] == pytest.approx(0.5)
    assert audit["pcgnn"]["target_course_count"] == 1
    assert audit["cgrc"]["record_count"] == 1


def test_validation_seed_analysis_pair_aligns_pcgnn_native_order(tmp_path):
    first = _record("pcgnn", [3, 2, 5])
    second = {
        **_record("pcgnn", [4, 2, 5]),
        "sample_index": 1,
        "user_id": 8,
        "target_item_id": 4,
    }
    pcgnn_path = tmp_path / "pcgnn.jsonl"
    pcgnn_path.write_text(
        "\n".join(
            [
                json.dumps({**second, "sample_index": 0}),
                json.dumps({**first, "sample_index": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cgrc_path = tmp_path / "cgrc.jsonl"
    cgrc_path.write_text(
        "\n".join(
            [
                json.dumps({**first, "model": "cgrc", "sample_index": 0}),
                json.dumps({**second, "model": "cgrc", "sample_index": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    course_rows, audit = analyze_validation_seed(
        seed=2025,
        expected_pairs=[(7, 3), (8, 4)],
        histories={7: np.array([0, 1]), 8: np.array([0, 1])},
        train_popularity=np.array([2, 2, 2, 0, 0, 0]),
        artifacts=_artifacts(),
        model_paths={"pcgnn": pcgnn_path, "cgrc": cgrc_path},
        expected_top_k=3,
        metric_k=2,
    )

    assert len(course_rows) == 4
    assert audit["pcgnn"]["record_count"] == 2
    assert audit["pcgnn"]["target_course_count"] == 2


def test_validation_seed_analysis_preserves_cold_only_missingness(tmp_path):
    path = tmp_path / "pcgnn.jsonl"
    path.write_text(json.dumps(_record("pcgnn", [2, 1, 3])) + "\n", encoding="utf-8")

    course_rows, _ = analyze_validation_seed(
        seed=2025,
        expected_pairs=[(7, 3)],
        histories={7: np.array([0])},
        train_popularity=np.array([2, 2, 2, 0, 0, 0]),
        artifacts=_artifacts(),
        model_paths={"pcgnn": path},
        expected_top_k=3,
        metric_k=2,
    )

    row = course_rows.iloc[0]
    assert row["effective_coverage"] == 0.0
    assert row["missingness"] == 1.0
    assert math.isnan(row["cold_prerequisite_gap"])
    assert math.isnan(row["cold_concept_continuity"])


def test_seed_stratified_interval_is_deterministic_and_seed_weighted():
    rows = pd.DataFrame(
        [
            *({"seed": 2025, "value": 0.0} for _ in range(20)),
            {"seed": 2026, "value": 1.0},
        ]
    )

    first = seed_stratified_interval(
        rows,
        value_column="value",
        n_bootstrap=500,
        random_seed=2027,
    )
    second = seed_stratified_interval(
        rows,
        value_column="value",
        n_bootstrap=500,
        random_seed=2027,
    )

    assert first == second
    assert first[0] == pytest.approx(0.5)
    assert first[1:] == pytest.approx((0.5, 0.5))


def test_summary_reports_observed_units_coverage_and_equal_seed_means():
    rows = []
    for model in ("pcgnn", "cgrc"):
        for seed, value in ((2025, 0.2), (2026, 0.4), (2027, 0.6)):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "target_item_id": seed,
                    "analysis_split": "validation",
                    "list_count": 10,
                    "cold_list_count": 5,
                    "ndcg_at_10": value,
                    "low_ndcg_at_10": float(value <= 0.1),
                    "cold_proportion": 0.25,
                    "effective_coverage": 0.5,
                    "missingness": 0.5,
                    "cold_prerequisite_gap": np.nan if seed == 2025 else value,
                    "cold_concept_continuity": value,
                    "cold_difficulty_gap": value,
                    "cold_structural_redundancy": value,
                }
            )

    summary = summarize_validation_course_rows(
        pd.DataFrame(rows),
        n_bootstrap=200,
        random_seed=2027,
    )

    assert set(summary["model"]) == {"pcgnn", "cgrc"}
    assert set(summary["analysis_split"]) == {"validation"}
    prereq = summary.loc[
        summary["metric"].eq("cold_prerequisite_gap")
        & summary["model"].eq("pcgnn")
    ].iloc[0]
    assert prereq["mean"] == pytest.approx(0.5)
    assert prereq["unit_count"] == 3
    assert prereq["observed_unit_count"] == 2
    assert prereq["effective_coverage"] == pytest.approx(0.5)
    assert prereq["missingness"] == pytest.approx(0.5)


def test_real_validation_splits_have_102_course_units_per_model():
    root = Path(__file__).resolve().parents[1]
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    counts = []
    for seed in (2025, 2026, 2027):
        pairs, _, _ = validation_seed_inputs(split_root, seed, n_items=819)
        targets = {target for _, target in pairs}
        assert len(targets) == 34
        counts.append(len(targets))

    assert sum(counts) == 102
