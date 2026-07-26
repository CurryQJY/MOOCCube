from pathlib import Path

import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_figure1_motivation import (
    build_baseline_seed_summary,
    build_prerequisite_sources,
)


def _baseline_rows() -> pd.DataFrame:
    rows = []
    for model, offset in (("pcgnn", 0.0), ("cgrc", 0.1)):
        for seed in (2025, 2026, 2027):
            rows.append(
                {
                    "analysis_split": "validation",
                    "model": model,
                    "seed": seed,
                    "target_item_id": seed - 2000,
                    "ndcg_at_10": 0.04 + offset,
                    "cold_proportion": 0.35 - offset,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_summary_aggregates_within_seed_before_pooling():
    summary = build_baseline_seed_summary(_baseline_rows())

    assert len(summary) == 6
    assert summary.groupby("model")["target_course_count"].sum().to_dict() == {
        "cgrc": 3,
        "pcgnn": 3,
    }
    assert summary.loc[summary["model"].eq("pcgnn"), "ndcg_at_10"].mean() == pytest.approx(0.04)
    assert summary.loc[summary["model"].eq("cgrc"), "cold_proportion"].mean() == pytest.approx(0.25)


def test_concept_prerequisite_availability_matches_retained_edge_rule(tmp_path: Path):
    relation_dir = tmp_path
    (relation_dir / "prerequisite-dependency.json").write_text(
        "p\tt\nq\tt\n", encoding="utf-8"
    )
    concept_sets = [{"p"}, {"t"}, {"q"}, set()]

    counts, sources = build_prerequisite_sources(concept_sets, relation_dir)

    assert counts.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert sources[1] == {0, 2}
