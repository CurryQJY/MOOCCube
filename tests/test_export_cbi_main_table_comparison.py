from __future__ import annotations

import importlib.util
from pathlib import Path

import openpyxl
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_aaai27" / "scripts" / "export_cbi_main_table_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_cbi_main_table_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_comparison_matches_paper_and_cbi_sources() -> None:
    module = load_module()
    frame, cbi, seed_matched = module.collect_comparison()

    assert list(frame["Method"])[-1] == "CBI-Faithful"
    paper_ckg = frame[frame["Method"] == "CKG-RL"].iloc[0]
    cbi_row = frame[frame["Method"] == "CBI-Faithful"].iloc[0]
    assert paper_ckg["R@10"] == pytest.approx(0.2863)
    assert cbi_row["R@10"] == pytest.approx(0.3470857762779366)
    assert cbi["N@10"] == pytest.approx(0.2250153918070929)
    assert seed_matched["R@10"] == pytest.approx(0.27321624805264144)


def test_write_outputs_creates_auditable_workbook(tmp_path: Path) -> None:
    module = load_module()
    frame, cbi, seed_matched = module.collect_comparison()
    paths = module.write_outputs(frame, cbi, seed_matched, tmp_path)

    assert {path.name for path in paths} == {
        "cbi_vs_main_table.csv",
        "cbi_vs_main_table.xlsx",
        "cbi_vs_main_table_aaai.tex",
    }
    workbook = openpyxl.load_workbook(tmp_path / "cbi_vs_main_table.xlsx", data_only=False)
    assert workbook.sheetnames == [
        "Main_Comparison",
        "CBI_Deltas",
        "Seed_Matched",
        "Full_CBI_Metrics",
        "Provenance",
    ]
    assert workbook["Main_Comparison"].freeze_panes == "A2"
    assert workbook["CBI_Deltas"]["D2"].value == "=B2-C2"
    assert workbook["Seed_Matched"]["D2"].value == "=B2-C2"


def test_write_aaai_table_uses_paper_style_and_scope_note(tmp_path: Path) -> None:
    module = load_module()
    frame, cbi, _ = module.collect_comparison()
    path = module.write_aaai_table(frame, cbi, tmp_path)
    content = path.read_text(encoding="utf-8")

    assert path.name == "cbi_vs_main_table_aaai.tex"
    assert "\\begin{table*}[t]" in content
    assert "\\begin{tabular*}{\\textwidth}" in content
    assert "\\textbf{CBI-Faithful}$^\\dagger$" in content
    assert "\\emph{Imp. vs. CKG-RL}" in content
    assert "single seed 2025" in content
    assert "0.3471" in content
    assert "+21.2\\%" in content
