from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path
from typing import Sequence


TELEMETRY_FIELDS = (
    "timestamp",
    "gpu_index",
    "pstate",
    "gpu_utilization_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
)


def build_nvidia_smi_command(executable: str, *, device: int) -> list[str]:
    query = ",".join(
        (
            "timestamp",
            "index",
            "pstate",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "power.draw",
            "temperature.gpu",
            "clocks.current.sm",
        )
    )
    return [
        executable,
        "-i",
        str(device),
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]


def parse_nvidia_smi_row(line: str) -> dict[str, str]:
    fields = next(csv.reader([line]))
    if len(fields) != len(TELEMETRY_FIELDS):
        raise ValueError(
            f"Expected {len(TELEMETRY_FIELDS)} nvidia-smi fields, got {len(fields)}"
        )
    return {
        name: value.strip()
        for name, value in zip(TELEMETRY_FIELDS, fields)
    }


def collect_sample(executable: str, *, device: int) -> dict[str, str]:
    completed = subprocess.run(
        build_nvidia_smi_command(executable, device=device),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no telemetry rows")
    return parse_nvidia_smi_row(lines[0])


def monitor(
    output_path: Path,
    *,
    stop_path: Path,
    interval_seconds: float,
    executable: str,
    device: int,
    max_samples: int | None = None,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS)
        writer.writeheader()
        while not stop_path.exists():
            writer.writerow(collect_sample(executable, device=device))
            handle.flush()
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                return
            time.sleep(interval_seconds)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    monitor(
        args.output,
        stop_path=args.stop_file,
        interval_seconds=args.interval_seconds,
        executable=args.nvidia_smi,
        device=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
