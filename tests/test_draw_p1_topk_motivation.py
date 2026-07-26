from pathlib import Path

import pandas as pd
import pytest

import paper_aaai27.scripts.draw_p1_topk_motivation as draw_module
from paper_aaai27.scripts.draw_p1_topk_motivation import (
    MAIN_METRICS,
    draw_all,
    draw_main_figure,
    favorable_effect_rows,
    validate_robustness_inputs,
)


def _paired_frame():
    directions = {
        "prerequisite_gap": "lower",
        "concept_continuity": "higher",
        "difficulty_gap": "lower",
        "structural_redundancy": "lower",
    }
    values = {
        "prerequisite_gap": (-0.02, -0.03, -0.01),
        "concept_continuity": (-0.03, -0.04, -0.02),
        "difficulty_gap": (0.002, -0.001, 0.005),
        "structural_redundancy": (-0.018, -0.02, -0.01),
    }
    rows = []
    for comparison, baseline, role, scale in (
        ("ckg_rl_vs_pcgnn", "pcgnn", "primary", 1.0),
        ("ckg_rl_vs_cgrc", "cgrc", "secondary", 0.5),
    ):
        for metric in MAIN_METRICS:
            mean, low, high = values[metric]
            rows.append(
                {
                    "comparison": comparison,
                    "comparison_role": role,
                    "treatment": "ckg_rl",
                    "baseline": baseline,
                    "cutoff": 10,
                    "metric": metric,
                    "direction": directions[metric],
                    "pair_count": 204,
                    "mean_difference": mean * scale,
                    "bootstrap_ci_low": low * scale,
                    "bootstrap_ci_high": high * scale,
                    "permutation_p_value": 0.01,
                    "interpretation": "supports",
                }
            )
    return pd.DataFrame(rows)


def _model_frame():
    return pd.DataFrame(
        [
            {"model": "ckg_rl", "cutoff": 10, "cold_proportion_mean": 0.50, "cold_proportion_sd": 0.04},
            {"model": "pcgnn", "cutoff": 10, "cold_proportion_mean": 0.35, "cold_proportion_sd": 0.02},
            {"model": "cgrc", "cutoff": 10, "cold_proportion_mean": 0.25, "cold_proportion_sd": 0.01},
        ]
    )


def _robustness_frames():
    sensitivity = []
    for scale in ("p90", "p95", "max"):
        for readiness_k in (3, 5, 10):
            sensitivity.append(
                {
                    "scale": scale,
                    "readiness_k": readiness_k,
                    "metric": "difficulty_gap",
                    "direction": "lower",
                    "pair_count": 204,
                    "mean_difference_ckg_rl_minus_cgrc": 0.002,
                    "bootstrap_ci_low": -0.001,
                    "bootstrap_ci_high": 0.005,
                    "permutation_p_value": 0.2,
                    "interpretation": "inconclusive",
                }
            )
    rank = []
    for rank_index in range(1, 11):
        for metric in MAIN_METRICS:
            rank.append(
                {
                    "rank": rank_index,
                    "metric": metric,
                    "direction": "higher" if metric == "concept_continuity" else "lower",
                    "pair_count": 204,
                    "mean_difference_ckg_rl_minus_cgrc": -0.01,
                    "bootstrap_ci_low": -0.02,
                    "bootstrap_ci_high": 0.0,
                    "permutation_p_value": 0.1,
                    "interpretation": "inconclusive",
                }
            )
    return pd.DataFrame(sensitivity), pd.DataFrame(rank)


def test_favorable_effect_orientation_preserves_raw_difference():
    oriented = favorable_effect_rows(
        _paired_frame(),
        comparison="ckg_rl_vs_pcgnn",
    )

    assert oriented["metric"].tolist() == list(MAIN_METRICS)
    prereq = oriented[oriented["metric"] == "prerequisite_gap"].iloc[0]
    concept = oriented[oriented["metric"] == "concept_continuity"].iloc[0]
    assert prereq["raw_difference"] == pytest.approx(-0.02)
    assert prereq["favorable_effect"] == pytest.approx(0.02)
    assert prereq["favorable_ci_low"] == pytest.approx(0.01)
    assert prereq["favorable_ci_high"] == pytest.approx(0.03)
    assert concept["raw_difference"] == pytest.approx(-0.03)
    assert concept["favorable_effect"] == pytest.approx(-0.03)


def test_robustness_inputs_require_full_grid_and_ranks():
    sensitivity, rank = _robustness_frames()
    validate_robustness_inputs(sensitivity, rank)

    with pytest.raises(ValueError, match="nine difficulty settings"):
        validate_robustness_inputs(sensitivity.iloc[:-1], rank)


def test_draw_all_exports_pdf_svg_and_png_for_both_figures(tmp_path):
    analysis_dir = tmp_path / "analysis"
    robustness_dir = tmp_path / "robustness"
    figure_dir = tmp_path / "figures"
    analysis_dir.mkdir()
    robustness_dir.mkdir()
    _paired_frame().to_csv(analysis_dir / "paired_statistics.csv", index=False)
    _model_frame().to_csv(analysis_dir / "model_summary.csv", index=False)
    sensitivity, rank = _robustness_frames()
    sensitivity.to_csv(robustness_dir / "difficulty_sensitivity_paired.csv", index=False)
    rank.to_csv(robustness_dir / "rank_profile_paired.csv", index=False)

    outputs = draw_all(
        analysis_dir=analysis_dir,
        robustness_dir=robustness_dir,
        figure_dir=figure_dir,
    )

    assert len(outputs) == 6
    for output in outputs:
        assert Path(output).is_file()
        assert Path(output).stat().st_size > 0


def test_main_figure_legend_stays_above_the_effect_plot(monkeypatch, tmp_path):
    captured = {}

    def capture_figure(fig, base):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(draw_module, "_save_three_formats", capture_figure)

    draw_main_figure(_paired_frame(), _model_frame(), tmp_path / "figure")

    figure = captured["figure"]
    figure.canvas.draw()
    effect_axis = figure.axes[0]
    renderer = figure.canvas.get_renderer()
    legend_box = effect_axis.get_legend().get_window_extent(renderer)
    axis_box = effect_axis.get_window_extent(renderer)
    assert legend_box.y0 >= axis_box.y1
