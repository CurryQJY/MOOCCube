from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_draw_hparam_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "paper_aaai27" / "scripts" / "draw_hparam_split.py"
    spec = importlib.util.spec_from_file_location("draw_hparam_split", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_true_true_hparam_figure_reads_validation_histories_not_test_finals():
    module = load_draw_hparam_module()

    assert module.SELECT_METRIC == "Val_full_cold_N@10"
    assert module.RECALL_METRIC == "Val_full_cold_R@10"

    configured_variants = {
        point.variant
        for sweep in module.SWEEPS
        for point in sweep["points"]
        if point.variant != "main_default"
    }
    assert configured_variants == {
        "beta_0p00",
        "beta_0p10",
        "beta_0p15",
        "beta_0p25",
        "beta_0p30",
        "beta_0p50",
        "reward_0p00",
        "reward_0p50",
        "reward_1p50",
        "reward_2p00",
        "horizon_1",
        "horizon_3",
        "horizon_7",
        "horizon_10",
    }

    points, missing = module.collect_points()
    assert not points.empty
    assert not missing.empty
    assert set(points["eval_split"]) == {"validation"}
    assert set(points["selection_metric"]) == {"Val_full_cold_N@10"}
    assert points["source_file"].str.endswith(module.METRICS_FILE).all()
    assert not points["source_file"].str.contains("final_fullrank", regex=False).any()
    assert not points["source_file"].str.contains("content_delta_pop5", regex=False).any()
    assert points["source_file"].str.contains(
        "significance_per_item_exports/mooccube/ckg_rl_true_true_hparam_grid|"
        "significance_per_item_exports/mooccube/ckg_rl_full",
        regex=True,
    ).all()


def test_true_true_hparam_summary_keeps_validation_provenance():
    module = load_draw_hparam_module()

    points, missing = module.collect_points()
    summary = module.summarize(points, missing)

    assert "eval_split" in summary.columns
    assert "selection_metric" in summary.columns
    assert set(summary["eval_split"]) == {"validation"}
    assert set(summary["selection_metric"]) == {"Val_full_cold_N@10"}
