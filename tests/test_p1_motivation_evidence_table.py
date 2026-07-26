from pathlib import Path

import pandas as pd
import pytest

from paper_aaai27.scripts.build_p1_motivation_evidence_table import (
    build_motivation_evidence,
    render_motivation_table,
    run,
)


def _paired_rows():
    rows = []
    for baseline, values in {
        "pcgnn": {
            "prerequisite_gap": (-0.0450, -0.0550, -0.0350, 0.00001, "supports"),
            "difficulty_gap": (-0.0120, -0.0150, -0.0090, 0.00001, "supports"),
            "cold_proportion": (0.0850, 0.0640, 0.1070, 0.00001, "descriptive"),
            "concept_continuity": (0.0100, 0.0010, 0.0200, 0.02000, "supports"),
        },
        "cgrc": {
            "prerequisite_gap": (-0.0195, -0.0280, -0.0111, 0.00003, "supports"),
            "structural_redundancy": (-0.0176, -0.0202, -0.0151, 0.00001, "supports"),
            "cold_proportion": (0.2615, 0.2493, 0.2738, 0.00001, "descriptive"),
        },
    }.items():
        for metric, (delta, low, high, p_value, interpretation) in values.items():
            rows.append(
                {
                    "treatment": "ckg_rl",
                    "baseline": baseline,
                    "cutoff": 10,
                    "metric": metric,
                    "pair_count": 204,
                    "mean_difference": delta,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "permutation_p_value": p_value,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def _comparison_summary():
    values = {
        "ckg_rl": (0.5977, 0.0844, 0.0533, 0.5073),
        "pcgnn": (0.6427, 0.0963, 0.0417, 0.4218),
        "cgrc": (0.6172, 0.0824, 0.0710, 0.2457),
    }
    return pd.DataFrame(
        [
            {
                "model": model,
                "cutoff": 10,
                "prerequisite_gap_mean": prereq,
                "difficulty_gap_mean": difficulty,
                "structural_redundancy_mean": redundancy,
                "cold_proportion_mean": exposure,
            }
            for model, (prereq, difficulty, redundancy, exposure) in values.items()
        ]
    )


def _mechanism_paired():
    return pd.DataFrame(
        [
            {
                "treatment": "ckg_rl",
                "baseline": "ckg_rl_wo_course_reward",
                "comparison_role": "course_reward",
                "cutoff": 10,
                "metric": metric,
                "pair_count": 204,
                "mean_difference": delta,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "permutation_p_value": p_value,
                "interpretation": interpretation,
            }
            for metric, delta, low, high, p_value, interpretation in (
                ("cold_prerequisite_gap", -0.0132, -0.0205, -0.0056, 0.00092, "supports"),
                ("cold_difficulty_gap", -0.0034, -0.0056, -0.0011, 0.00367, "supports"),
                ("cold_proportion", 0.0269, 0.0189, 0.0346, 0.00001, "descriptive"),
            )
        ]
        + [
            {
                "treatment": "ckg_rl",
                "baseline": "ckg_rl_wo_simulator",
                "comparison_role": "simulator",
                "cutoff": 10,
                "metric": "cold_proportion",
                "pair_count": 204,
                "mean_difference": 0.0130,
                "bootstrap_ci_low": 0.0056,
                "bootstrap_ci_high": 0.0208,
                "permutation_p_value": 0.00104,
                "interpretation": "descriptive",
            }
        ]
    )


def _mechanism_summary():
    return pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "cutoff": 10,
                "cold_prerequisite_gap_mean": 0.5994,
                "cold_difficulty_gap_mean": 0.0864,
                "cold_proportion_mean": 0.5073,
            },
            {
                "model": "ckg_rl_wo_course_reward",
                "cutoff": 10,
                "cold_prerequisite_gap_mean": 0.6126,
                "cold_difficulty_gap_mean": 0.0898,
                "cold_proportion_mean": 0.4804,
            },
        ]
    )


def test_evidence_table_contains_only_predeclared_supported_motivation_claims():
    evidence = build_motivation_evidence(
        _paired_rows(),
        _comparison_summary(),
        _mechanism_paired(),
        _mechanism_summary(),
    )

    assert len(evidence) == 7
    assert not evidence["metric"].str.contains("concept").any()
    assert not evidence["contrast"].str.contains("simulator").any()
    assert (evidence["improvement"] > 0).all()
    assert (evidence["ci_low"].dropna() > 0).all()
    assert set(evidence["evidence_group"]) == {
        "Course-structure limitation",
        "Strict cold-start effectiveness",
        "Course-reward mechanism",
    }


def test_latex_table_marks_exposure_as_descriptive_and_names_exclusions():
    evidence = build_motivation_evidence(
        _paired_rows(),
        _comparison_summary(),
        _mechanism_paired(),
        _mechanism_summary(),
    )

    latex = render_motivation_table(evidence)

    assert "\\toprule" in latex
    assert "\\label{tab:p1-motivation-evidence}" in latex
    assert "Course-reward exposure is descriptive" in latex
    assert "inconclusive or adverse outcomes are excluded" in latex
    assert "+0.0274 (+10.6\\%)" in latex
    assert "w/o simulator" not in latex


def test_run_writes_csv_and_latex_from_analysis_outputs(tmp_path):
    comparison_dir = tmp_path / "comparison"
    mechanism_dir = tmp_path / "mechanism"
    output_dir = tmp_path / "tables"
    comparison_dir.mkdir()
    mechanism_dir.mkdir()
    _paired_rows().to_csv(comparison_dir / "paired_statistics.csv", index=False)
    _comparison_summary().to_csv(comparison_dir / "model_summary.csv", index=False)
    _mechanism_paired().to_csv(mechanism_dir / "paired_statistics.csv", index=False)
    _mechanism_summary().to_csv(mechanism_dir / "model_summary.csv", index=False)

    outputs = run(comparison_dir, mechanism_dir, output_dir)

    assert outputs == {
        "csv": output_dir / "mooccube_p1_motivation_evidence.csv",
        "latex": output_dir / "mooccube_p1_motivation_evidence.tex",
    }
    assert outputs["csv"].is_file()
    assert outputs["latex"].is_file()
    written = pd.read_csv(outputs["csv"])
    assert len(written) == 7
    assert written["pair_count"].eq(204).all()


def test_missing_supporting_interval_is_rejected():
    paired = _paired_rows()
    paired.loc[
        (paired["baseline"].eq("pcgnn"))
        & (paired["metric"].eq("prerequisite_gap")),
        "bootstrap_ci_high",
    ] = 0.01

    with pytest.raises(ValueError, match="does not support the motivation claim"):
        build_motivation_evidence(
            paired,
            _comparison_summary(),
            _mechanism_paired(),
            _mechanism_summary(),
        )


def test_supplement_integrates_the_supported_motivation_table():
    root = Path(__file__).resolve().parents[1]
    main = (root / "paper_aaai27" / "main.tex").read_text(encoding="utf-8")
    supplement = (root / "paper_aaai27" / "supplement_tables.tex").read_text(
        encoding="utf-8"
    )

    table_input = "\\input{tables/mooccube_p1_motivation_evidence.tex}"
    assert table_input not in main
    assert "Table~\\ref{tab:p1-motivation-evidence}" not in main
    assert "\\usepackage{threeparttable}" not in main
    assert supplement.count(table_input) == 1
    assert "\\usepackage{threeparttable}" in supplement


def test_main_paper_describes_both_baselines_in_both_motivation_figures():
    root = Path(__file__).resolve().parents[1]
    main = (root / "paper_aaai27" / "main.tex").read_text(encoding="utf-8")

    method_caption = main.split("\\label{fig:method-motivation}", 1)[0].rsplit(
        "\\caption{", 1
    )[1]
    audit_caption = main.split("\\label{fig:p1-topk-motivation}", 1)[0].rsplit(
        "\\caption{", 1
    )[1]
    for caption in (method_caption, audit_caption):
        assert "PCGNN" in caption
        assert "CGRC" in caption
    assert "Recall@10" not in method_caption
    assert "NDCG@10 for CKG-RL" not in method_caption
    assert "Validation-only" in method_caption
    assert "102 matched" in method_caption
    assert "coverage/missingness" in method_caption.lower()
