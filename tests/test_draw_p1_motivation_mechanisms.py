from pathlib import Path

import pandas as pd
import pytest

import paper_aaai27.scripts.draw_p1_motivation_mechanisms as draw_module
from paper_aaai27.scripts.draw_p1_motivation_mechanisms import (
    RISK_METRICS,
    draw_mechanism_figure,
    favorable_mechanism_rows,
)


def _paired_frame() -> pd.DataFrame:
    directions = {
        "prerequisite_gap": "lower",
        "concept_continuity": "higher",
        "difficulty_gap": "lower",
        "structural_redundancy": "lower",
    }
    rows = []
    for comparison, role, scale in (
        ("ckg_rl_vs_ckg_rl_wo_course_reward", "course_reward", 1.0),
        ("ckg_rl_vs_ckg_rl_wo_simulator", "simulator", 0.5),
    ):
        for prefix in ("", "cold_"):
            for index, metric in enumerate(RISK_METRICS, start=1):
                raw = -0.01 * index * scale
                rows.append(
                    {
                        "comparison": comparison,
                        "comparison_role": role,
                        "cutoff": 10,
                        "metric": f"{prefix}{metric}",
                        "direction": directions[metric],
                        "mean_difference": raw,
                        "bootstrap_ci_low": raw - 0.002,
                        "bootstrap_ci_high": raw + 0.002,
                    }
                )
    return pd.DataFrame(rows)


def test_favorable_mechanism_rows_orient_lower_and_higher_metrics():
    rows = favorable_mechanism_rows(
        _paired_frame(),
        comparison="ckg_rl_vs_ckg_rl_wo_course_reward",
        cold_only=False,
    )

    assert rows["display_metric"].tolist() == list(RISK_METRICS)
    prerequisite = rows.loc[rows["display_metric"].eq("prerequisite_gap")].iloc[0]
    continuity = rows.loc[rows["display_metric"].eq("concept_continuity")].iloc[0]
    assert prerequisite["favorable_effect"] == pytest.approx(0.01)
    assert prerequisite["favorable_ci_low"] == pytest.approx(0.008)
    assert prerequisite["favorable_ci_high"] == pytest.approx(0.012)
    assert continuity["favorable_effect"] == pytest.approx(-0.02)


def test_favorable_mechanism_rows_select_cold_only_metrics():
    rows = favorable_mechanism_rows(
        _paired_frame(),
        comparison="ckg_rl_vs_ckg_rl_wo_simulator",
        cold_only=True,
    )

    assert rows["metric"].str.startswith("cold_").all()
    assert rows["display_metric"].tolist() == list(RISK_METRICS)


def test_mechanism_figure_exports_pdf_svg_and_png(tmp_path):
    outputs = draw_mechanism_figure(
        _paired_frame(),
        tmp_path / "mooccube_p1_motivation_mechanisms",
    )

    assert len(outputs) == 3
    for output in outputs:
        assert Path(output).is_file()
        assert Path(output).stat().st_size > 0


def test_mechanism_figure_legend_stays_above_both_panels(monkeypatch, tmp_path):
    captured = {}

    def capture_figure(fig, base):
        captured["figure"] = fig
        return []

    monkeypatch.setattr(draw_module, "_save_three_formats", capture_figure)
    draw_mechanism_figure(_paired_frame(), tmp_path / "figure")

    figure = captured["figure"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = figure.legends[0].get_window_extent(renderer)
    for axis in figure.axes:
        assert legend_box.y0 >= axis.get_window_extent(renderer).y1
