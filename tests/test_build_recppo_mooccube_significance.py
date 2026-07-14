from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper_aaai27/scripts/build_recppo_mooccube_significance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_recppo_mooccube_significance", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_write_latex_uses_superscript_significance_marks(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "method": MODULE.CGRC_NAME,
                **{
                    f"{metric}_{stat}": value
                    for metric in MODULE.METRICS
                    for stat, value in (("mean", 0.1), ("std", 0.01))
                },
            },
            {
                "method": MODULE.RECPPO_NAME,
                **{
                    f"{metric}_{stat}": value
                    for metric in MODULE.METRICS
                    for stat, value in (("mean", 0.2), ("std", 0.02))
                },
            },
        ]
    )
    significance = pd.DataFrame(
        [
            {"metric": metric, "p_bonferroni_12": p_value}
            for metric, p_value in zip(
                MODULE.METRICS, (0.0001, 0.01, 0.1, 0.001)
            )
        ]
    )
    output = tmp_path / "table.tex"

    MODULE.write_latex(table, significance, output)

    latex = output.read_text(encoding="utf-8")
    assert r"0.2000 $\pm$ 0.0200\textsuperscript{\scriptsize ***}" in latex
    assert r"0.2000 $\pm$ 0.0200\textsuperscript{\scriptsize *}" in latex
    assert r"0.2000 $\pm$ 0.0200***" not in latex

