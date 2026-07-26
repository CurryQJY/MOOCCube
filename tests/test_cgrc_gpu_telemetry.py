from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_aaai27" / "scripts" / "monitor_cgrc_gpu_telemetry.py"


def load_module():
    spec = importlib.util.spec_from_file_location("monitor_cgrc_gpu_telemetry", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_telemetry_parser_normalizes_nvidia_smi_csv_rows():
    module = load_module()

    row = module.parse_nvidia_smi_row(
        "2026/07/23 10:00:00.000, 0, P0, 87, 1024, 12227, 120.4, 65, 2205"
    )

    assert row == {
        "timestamp": "2026/07/23 10:00:00.000",
        "gpu_index": "0",
        "pstate": "P0",
        "gpu_utilization_pct": "87",
        "memory_used_mib": "1024",
        "memory_total_mib": "12227",
        "power_draw_w": "120.4",
        "temperature_c": "65",
        "sm_clock_mhz": "2205",
    }


def test_gpu_telemetry_command_scopes_sampling_to_requested_device():
    module = load_module()

    command = module.build_nvidia_smi_command("nvidia-smi", device=0)

    assert command[0] == "nvidia-smi"
    assert "-i" in command
    assert command[command.index("-i") + 1] == "0"
    assert any("utilization.gpu" in arg for arg in command)
    assert any("clocks.current.sm" in arg for arg in command)
