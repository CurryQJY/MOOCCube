from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_aaai27" / "scripts" / "export_ckg_rl_tdinit_aaai.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_ckg_rl_tdinit_aaai", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loads_six_item_macro_cold_metrics_and_imp() -> None:
    module = load_module()
    rows = module.load_rows()

    assert list(rows["Method"]) == ["CKG-RL", "CKG-RL+TDInit", "Imp."]
    assert rows.loc[0, "R@10"] == pytest.approx(0.2863)
    assert rows.loc[1, "R@10"] == pytest.approx(0.3470857762779366)
    assert rows.loc[2, "R@20"] == pytest.approx(0.3602, abs=5e-5)


def test_writes_aaai_table_with_all_metrics(tmp_path: Path) -> None:
    module = load_module()
    tex_path, csv_path = module.write_outputs(tmp_path)
    tex = tex_path.read_text(encoding="utf-8")

    assert tex_path.name == "ckg_rl_tdinit_aaai.tex"
    assert csv_path.name == "ckg_rl_tdinit_all_metrics.csv"
    assert "\\begin{table*}[t]" in tex
    assert "R@20" in tex and "N@20" in tex
    assert "\\textbf{CKG-RL+TDInit}" in tex
    assert "\\emph{Imp.}" in tex
    assert "+36.0\\%" in tex
    assert tex.rstrip().endswith("\\end{table*}")
