from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_aaai27_rl_component_gate_serial.ps1"


def test_main_reproduction_runner_has_full_only_mode_and_exact_inactive_pseudo_config():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "[switch]$FullOnly" in text
    assert "PseudoColdRatio = 0.30" in text
    assert "PseudoColdMinPop = 5" in text
    assert 'PseudoColdMode = "batch_random"' in text
    assert "if ($FullOnly)" in text
    assert "FULL_ONLY_END" in text
