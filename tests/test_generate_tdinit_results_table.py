from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_aaai27" / "scripts" / "generate_tdinit_results_table.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_tdinit_results_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_weighted_overall_is_computed_within_each_seed():
    module = load_module()
    result = module.count_weighted_overall(
        cold_value=0.25,
        hot_value=0.50,
        cold_count=1,
        hot_count=3,
    )
    assert result == pytest.approx(0.4375)


def test_default_baseline_is_latest_main_table(monkeypatch: pytest.MonkeyPatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = module.parse_args()
    assert args.baseline == (
        ROOT
        / "outputs"
        / "significance_per_item_exports"
        / "mooccube"
        / "ckg_rl_full"
        / "fast3_static_runs_detail.csv"
    )
    assert args.output_dir == (
        ROOT / "paper_aaai27" / "figures" / "ckg_rl_tdinit_3seed_latest"
    )


def test_generator_exports_combined_and_split_tables(tmp_path: Path):
    module = load_module()
    baseline = ROOT / "outputs" / "significance_per_item_exports" / "mooccube" / "ckg_rl_full" / "fast3_static_runs_detail.csv"
    tdinit = ROOT / "outputs" / "cbi_anchor_sim_3seed_serial" / "fast3_static_runs_detail.csv"

    module.generate_tables(baseline, tdinit, tmp_path)

    expected = {
        "tdinit_comparison.csv",
        "tdinit_comparison.tex",
        "tdinit_comparison_all.png",
        "tdinit_comparison_all.pdf",
        "tdinit_comparison_cold.png",
        "tdinit_comparison_cold.pdf",
        "tdinit_comparison_hot.png",
        "tdinit_comparison_hot.pdf",
        "tdinit_comparison_overall.png",
        "tdinit_comparison_overall.pdf",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
