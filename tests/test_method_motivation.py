from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from paper_aaai27.scripts.analyze_method_motivation import (
    METRICS,
    aggregate_rank_frame,
    build_bucket_course_rows,
    paired_rank_alignment,
)
from paper_aaai27.scripts.draw_method_motivation import (
    MODEL_HATCHES,
    build_existing_method_diagnostics,
    draw_motivation_figure,
    draw_existing_method_motivation_figure,
)


ROOT = Path(__file__).resolve().parents[1]


def _recommendation_rows():
    rows = []
    settings = [
        {
            "seed": 2025,
            "target_item_id": 11,
            "interactions": 1,
            "top": {
                "prerequisite_gap": 0.2,
                "concept_continuity": 0.6,
                "difficulty_gap": 0.1,
                "structural_redundancy": 0.1,
                "is_cold": 0.3,
            },
            "bottom": {
                "prerequisite_gap": 0.4,
                "concept_continuity": 0.4,
                "difficulty_gap": 0.2,
                "structural_redundancy": 0.2,
                "is_cold": 0.2,
            },
        },
        {
            "seed": 2026,
            "target_item_id": 22,
            "interactions": 3,
            "top": {
                "prerequisite_gap": 0.8,
                "concept_continuity": 0.7,
                "difficulty_gap": 0.4,
                "structural_redundancy": 0.5,
                "is_cold": 0.5,
            },
            "bottom": {
                "prerequisite_gap": 0.4,
                "concept_continuity": 0.5,
                "difficulty_gap": 0.2,
                "structural_redundancy": 0.3,
                "is_cold": 0.3,
            },
        },
    ]
    for setting in settings:
        for interaction in range(setting["interactions"]):
            for rank in range(1, 21):
                values = setting["top" if rank <= 10 else "bottom"]
                rows.append(
                    {
                        "model": "cgrc",
                        "seed": setting["seed"],
                        "sample_index": interaction,
                        "target_item_id": setting["target_item_id"],
                        "rank": rank,
                        **values,
                    }
                )
    # A non-CGRC row must not affect the baseline-only diagnosis.
    rows.append(
        {
            "model": "ckg_rl",
            "seed": 2025,
            "sample_index": 999,
            "target_item_id": 11,
            "rank": 1,
            "prerequisite_gap": 1.0,
            "concept_continuity": 0.0,
            "difficulty_gap": 1.0,
            "structural_redundancy": 1.0,
            "is_cold": 1.0,
        }
    )
    return pd.DataFrame(rows)


def test_rank_alignment_is_cgrc_only_and_course_macro():
    rank_rows = aggregate_rank_frame(_recommendation_rows(), model="cgrc", top_k=20)
    assert len(rank_rows) == 40
    assert rank_rows.groupby(["seed", "target_item_id"])["rank"].nunique().eq(20).all()

    bucket_rows = build_bucket_course_rows(rank_rows, cutoff=10, top_k=20)
    paired = paired_rank_alignment(
        bucket_rows,
        n_bootstrap=200,
        n_permutations=500,
        random_seed=2027,
    ).set_index("metric")

    # Course 11 has raw prerequisite difference -0.2; course 22 has +0.4.
    # Course-macro aggregation is therefore +0.1, not the interaction-weighted +0.25.
    prerequisite = paired.loc["prerequisite_gap"]
    assert prerequisite["pair_count"] == 2
    assert prerequisite["raw_difference_top10_minus_bottom10"] == pytest.approx(0.1)
    assert prerequisite["favorable_alignment_effect"] == pytest.approx(-0.1)

    concept = paired.loc["concept_continuity"]
    assert concept["raw_difference_top10_minus_bottom10"] == pytest.approx(0.2)
    assert concept["favorable_alignment_effect"] == pytest.approx(0.2)


def test_rank_alignment_rejects_incomplete_top20():
    rows = aggregate_rank_frame(_recommendation_rows(), model="cgrc", top_k=20)
    incomplete = rows[~((rows["seed"] == 2025) & (rows["rank"] == 20))]

    with pytest.raises(ValueError, match="complete ranks 1..20"):
        build_bucket_course_rows(incomplete, cutoff=10, top_k=20)


def test_draw_motivation_exports_all_formats(tmp_path):
    exposure = pd.DataFrame({"N@10": [0.01, 0.04, 0.08, 0.2, 0.5]})
    directions = {
        "prerequisite_gap": "lower",
        "concept_continuity": "higher",
        "difficulty_gap": "lower",
        "structural_redundancy": "lower",
    }
    effects = {
        "prerequisite_gap": (0.0, -0.01, 0.01),
        "concept_continuity": (0.05, 0.04, 0.06),
        "difficulty_gap": (0.01, 0.005, 0.015),
        "structural_redundancy": (-0.06, -0.07, -0.05),
    }
    paired = pd.DataFrame(
        [
            {
                "metric": metric,
                "direction": directions[metric],
                "pair_count": 204,
                "favorable_alignment_effect": effects[metric][0],
                "favorable_ci_low": effects[metric][1],
                "favorable_ci_high": effects[metric][2],
            }
            for metric in METRICS
        ]
    )
    output_base = tmp_path / "motivation"

    outputs = draw_motivation_figure(
        exposure,
        paired,
        cold_share_top10=0.246,
        output_base=output_base,
    )

    assert {Path(path).suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs)


def test_existing_method_motivation_figure_diagnoses_without_ckg_scores(tmp_path):
    pcgnn_rows = pd.DataFrame(
        [
            {
                "model": "pcgnn",
                "seed": 2025 + index // 68,
                "target_item_id": index,
                "cutoff": 10,
                "prerequisite_gap": 0.55 + (index % 5) * 0.02,
                "difficulty_gap": 0.07 + (index % 4) * 0.01,
            }
            for index in range(204)
        ]
    )
    decoy_ckg_rows = pcgnn_rows.assign(model="ckg_rl")
    course_macro = pd.concat([pcgnn_rows, decoy_ckg_rows], ignore_index=True)
    exposure = pd.DataFrame(
        {
            "model": "cgrc",
            "seed": [2025 + index // 68 for index in range(204)],
            "target_item_id": list(range(204)),
            "N@10": [0.05] * 94 + [0.20] * 110,
        }
    )
    model_summary = pd.DataFrame(
        [{"model": "cgrc", "cutoff": 10, "cold_proportion_mean": 0.246}]
    )

    diagnostics = build_existing_method_diagnostics(
        course_macro,
        exposure,
        model_summary,
        n_bootstrap=100,
    )

    assert diagnostics["pcgnn"]["count"] == 204
    assert diagnostics["cgrc"]["count"] == 204
    assert diagnostics["cgrc"]["low_ndcg_fraction"] == pytest.approx(94 / 204)
    assert "ckg_rl" not in diagnostics
    assert MODEL_HATCHES["pcgnn"] == "///"
    assert MODEL_HATCHES["cgrc"] in {"...", "xx"}

    outputs = draw_existing_method_motivation_figure(
        course_macro,
        exposure,
        model_summary,
        tmp_path / "mooccube_method_motivation",
        n_bootstrap=100,
    )
    assert {Path(path).suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs)


def test_analysis_script_runs_directly_from_repository_root():
    script = ROOT / "paper_aaai27" / "scripts" / "analyze_method_motivation.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Analyze CGRC-only method motivation" in completed.stdout


def test_manuscript_positions_cgrc_as_generic_transfer_not_course_model():
    manuscript = (ROOT / "paper_aaai27" / "main.tex").read_text(encoding="utf-8")

    assert "CGRC as a generic cold-start transfer baseline, not a course recommender" in manuscript
    assert "PCGNN as the course-specific counterpart" in manuscript
    assert "Why existing methods fall short" in manuscript
    assert "46\\%" in manuscript
    assert "Figure~\\ref{fig:p1-topk-motivation}" in manuscript
    assert "course-specific PCGNN" in manuscript
    assert "generic cold-start CGRC" in manuscript
