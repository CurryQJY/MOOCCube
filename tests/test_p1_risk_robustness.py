import numpy as np
import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_p1_risk_robustness import (
    PartialMeanAccumulator,
    build_readiness_caches,
    history_size_bin,
    paired_group_statistics,
    summarize_recommendation_chunk,
    validate_primary_reproduction,
)


def test_readiness_cache_uses_most_complex_available_courses():
    histories = {
        1: np.array([0, 1, 2, 3]),
        2: np.array([0, 3]),
    }
    complexities = {
        "p90": np.array([0.1, 0.3, 0.8, 0.5]),
        "p95": np.array([0.2, 0.4, 0.9, 0.6]),
    }

    caches = build_readiness_caches(
        histories,
        complexities,
        readiness_ks=(2, 5),
        n_users=4,
    )

    assert caches[("p90", 2)][1] == pytest.approx(0.65)
    assert caches[("p90", 5)][1] == pytest.approx(0.425)
    assert caches[("p95", 2)][2] == pytest.approx(0.4)
    assert caches[("p95", 5)][2] == pytest.approx(0.4)
    assert caches[("p90", 2)][3] == 0.0
    assert caches[("p90", 2)].dtype == np.float64


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, "1-2"), (2, "1-2"), (3, "3-4"), (4, "3-4"), (5, "5-9"), (9, "5-9"), (10, "10+")],
)
def test_history_size_bins_are_stable(size, expected):
    assert history_size_bin(size) == expected


def test_partial_accumulator_preserves_list_weighting_and_missingness():
    accumulator = PartialMeanAccumulator(
        key_columns=("model", "seed", "target_item_id"),
        value_columns=("difficulty_gap", "cold_difficulty_gap"),
    )
    accumulator.update(
        pd.DataFrame(
            [
                {
                    "model": "ckg_rl",
                    "seed": 2025,
                    "target_item_id": 3,
                    "difficulty_gap": 0.2,
                    "cold_difficulty_gap": 0.8,
                },
                {
                    "model": "ckg_rl",
                    "seed": 2025,
                    "target_item_id": 3,
                    "difficulty_gap": 0.4,
                    "cold_difficulty_gap": np.nan,
                },
            ]
        )
    )

    row = accumulator.to_frame().iloc[0]

    assert row["list_count"] == 2
    assert row["difficulty_gap"] == pytest.approx(0.3)
    assert row["cold_difficulty_gap"] == pytest.approx(0.8)


def _recommendation_frame():
    rows = []
    for model in ("ckg_rl", "cgrc"):
        for sample_index, user_id, items in [
            (0, 1, [2, 3]),
            (1, 2, [3, 1]),
        ]:
            for rank, item_id in enumerate(items, start=1):
                rows.append(
                    {
                        "model": model,
                        "seed": 2025,
                        "sample_index": sample_index,
                        "user_id": user_id,
                        "target_item_id": 4,
                        "rank": rank,
                        "recommended_item_id": item_id,
                        "is_cold": int(item_id == 2),
                        "prerequisite_gap": 0.1 * rank,
                        "concept_continuity": 0.3 * rank,
                        "difficulty_gap": 0.0,
                        "structural_redundancy": 0.05 * rank,
                    }
                )
    return pd.DataFrame(rows)


def test_chunk_summary_builds_sensitivity_rank_and_history_rows():
    frame = _recommendation_frame()
    complexities = {"p95": np.array([0.1, 0.2, 0.8, 0.6, 0.9])}
    readiness = {
        2025: {
            ("p95", 5): np.array([0.0, 0.5, 0.4]),
        }
    }
    history_counts = {2025: np.array([0, 6, 2])}

    lists, rank_rows = summarize_recommendation_chunk(
        frame,
        complexities=complexities,
        readiness_caches=readiness,
        history_counts=history_counts,
        cutoff=2,
    )

    assert len(lists) == 4
    assert len(rank_rows) == 8
    first = lists[(lists["model"] == "ckg_rl") & (lists["sample_index"] == 0)].iloc[0]
    assert first["difficulty_gap__p95__k5"] == pytest.approx(0.2)
    assert first["cold_difficulty_gap__p95__k5"] == pytest.approx(0.3)
    second = lists[(lists["model"] == "ckg_rl") & (lists["sample_index"] == 1)].iloc[0]
    assert np.isnan(second["cold_difficulty_gap__p95__k5"])
    assert set(lists["history_bin"]) == {"1-2", "5-9"}


def test_grouped_paired_statistics_keep_matching_seed_course_units():
    rows = []
    for target in (3, 4, 5):
        for model, value in (("ckg_rl", 0.2), ("cgrc", 0.4)):
            rows.append(
                {
                    "model": model,
                    "seed": 2025,
                    "target_item_id": target,
                    "scale": "p95",
                    "readiness_k": 5,
                    "difficulty_gap": value,
                }
            )
    paired = paired_group_statistics(
        pd.DataFrame(rows),
        group_columns=("scale", "readiness_k"),
        metrics={"difficulty_gap": "lower"},
        n_bootstrap=200,
        n_permutations=500,
        random_seed=9,
    )

    row = paired.iloc[0]
    assert row["pair_count"] == 3
    assert row["mean_difference_ckg_rl_minus_cgrc"] == pytest.approx(-0.2)
    assert row["interpretation"] == "supports"


def test_primary_reproduction_rejects_course_macro_drift():
    observed = pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "difficulty_gap__p95__k5": 0.2,
            }
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 3,
                "cutoff": 10,
                "difficulty_gap": 0.3,
            }
        ]
    )

    with pytest.raises(ValueError, match="primary difficulty reproduction"):
        validate_primary_reproduction(observed, frozen, tolerance=1e-12)
