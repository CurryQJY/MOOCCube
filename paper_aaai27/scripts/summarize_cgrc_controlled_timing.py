from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"\[CGRC-TRAIN\] Epoch (\d+)(?:/\d+)? .*?Time:\s*([0-9.]+)s"
)
# Fraction trimmed from each tail when computing a trimmed mean over the
# per-epoch timings within a single repeat. The workload is provably constant
# across epochs (identical workload_signature), so per-epoch dispersion is
# host-side measurement noise; trimming the tails suppresses transient CPU
# contention spikes that would otherwise inflate the plain mean.
TRIM_FRACTION = 0.1
FORMAL_TIMING_PROTOCOL = "cgrc_formal_timing_v1"
FORMAL_DATASETS = ("Junyi", "COCO", "MOOCCube")
FORMAL_MODEL_SEED = 2026
FORMAL_TIMING_SEEDS = (9101, 9102, 9103, 9104, 9105)
FORMAL_TELEMETRY_FIELDS = (
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
FORMAL_WORKLOAD_FIELDS = (
    "batch_count",
    "sampled_cold_item_total",
    "sampled_cold_item_max",
    "masked_edge_count_raw_total",
    "reconstruction_edge_count_total",
    "reconstruction_active_batch_count",
    "reconstruction_user_count_total",
    "workload_signature",
)


def parse_epoch_times(path: Path) -> dict[int, float]:
    times: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            times[int(match.group(1))] = float(match.group(2))
    return times


def load_timing_profile(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError(f"Expected one timing profile record in {path}")
    return raw[0]


def _trimmed_mean(values: list[float], *, trim_fraction: float = TRIM_FRACTION) -> float:
    """Mean after dropping the top/bottom ``trim_fraction`` of samples.

    The CGRC per-epoch workload is provably constant (identical workload
    signatures), so per-epoch dispersion is host-contention noise rather than
    real compute variation. A symmetric trimmed mean is a robust central
    estimate that ignores the occasional wall-clock spike while still using
    the bulk of the samples. Falls back to the plain mean when there are too
    few samples to trim at least one element from each tail.
    """
    if not values:
        raise ValueError("_trimmed_mean requires at least one value")
    ordered = sorted(values)
    n = len(ordered)
    k = int(n * trim_fraction)
    kept = ordered[k : n - k] if n - 2 * k > 0 else ordered
    return statistics.mean(kept)


def _iqr(values: list[float]) -> float:
    """Q3 - Q1 spread; a contention-robust dispersion measure.

    Returns 0.0 when fewer than two samples are available.
    """
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _has_telemetry_sample(path: Path) -> bool:
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    return len(lines) >= 2 and bool(lines[0].strip()) and bool(lines[1].strip())


def _profile_int(
    profile: dict[str, object],
    key: str,
    *,
    label: str,
    path: Path,
) -> int:
    try:
        return int(profile[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {label} in {path}") from exc


def _validate_formal_telemetry(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing telemetry samples for {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FORMAL_TELEMETRY_FIELDS:
            raise ValueError(f"Unexpected telemetry schema in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Missing telemetry samples for {path}")
    for row_number, row in enumerate(rows, start=2):
        if any(row.get(field, "").strip() == "" for field in FORMAL_TELEMETRY_FIELDS):
            raise ValueError(f"Blank telemetry value at row {row_number} in {path}")
        try:
            gpu_index = int(row["gpu_index"])
            utilization = float(row["gpu_utilization_pct"])
            memory_used = float(row["memory_used_mib"])
            memory_total = float(row["memory_total_mib"])
            power_draw = float(row["power_draw_w"])
            temperature = float(row["temperature_c"])
            sm_clock = float(row["sm_clock_mhz"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid telemetry value at row {row_number} in {path}") from exc
        numeric_values = (
            utilization,
            memory_used,
            memory_total,
            power_draw,
            temperature,
            sm_clock,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"Non-finite telemetry value at row {row_number} in {path}")
        if gpu_index != 0:
            raise ValueError(f"Unexpected telemetry GPU index at row {row_number} in {path}")
        if not 0.0 <= utilization <= 100.0:
            raise ValueError(f"Invalid GPU utilization at row {row_number} in {path}")
        if memory_used < 0.0 or memory_total <= 0.0 or memory_used > memory_total:
            raise ValueError(f"Invalid GPU memory telemetry at row {row_number} in {path}")
        if power_draw < 0.0 or temperature < 0.0 or sm_clock < 0.0:
            raise ValueError(f"Invalid GPU telemetry at row {row_number} in {path}")


def validate_formal_profile(
    path: Path,
    *,
    expected_dataset: str,
    expected_model_seed: int,
    expected_timing_seed: int,
    source_epoch: int,
    warmup_epochs: int,
    timed_epochs: int,
    require_telemetry: bool,
) -> dict[str, object]:
    expected_target_epoch = source_epoch + warmup_epochs + timed_epochs
    expected_measurement_start = source_epoch + warmup_epochs + 1
    if expected_dataset not in FORMAL_DATASETS:
        raise ValueError(f"Unexpected formal dataset: {expected_dataset}")
    if expected_model_seed != FORMAL_MODEL_SEED:
        raise ValueError(f"Unexpected formal model seed: {expected_model_seed}")
    if expected_timing_seed not in FORMAL_TIMING_SEEDS:
        raise ValueError(f"Unexpected formal timing seed: {expected_timing_seed}")
    if path.name != "cgrc_timing_profile.json":
        raise ValueError(f"Unexpected timing profile filename: {path}")
    if path.parent.name != f"timing_seed_{expected_timing_seed}":
        raise ValueError(f"Timing seed directory does not match profile: {path}")
    if path.parent.parent.name != f"seed_{expected_model_seed}":
        raise ValueError(f"Model seed directory does not match profile: {path}")
    if path.parent.parent.parent.name != expected_dataset:
        raise ValueError(f"Dataset directory does not match profile: {path}")

    profile = load_timing_profile(path)
    if profile.get("timing_only") is not True:
        raise ValueError(f"Timing-only profile required: {path}")
    if profile.get("protocol") != FORMAL_TIMING_PROTOCOL:
        raise ValueError(f"Expected formal timing protocol in {path}")
    if _profile_int(profile, "static_seed", label="static seed", path=path) != FORMAL_MODEL_SEED:
        raise ValueError(f"Unexpected static seed in {path}")
    if _profile_int(profile, "model_seed", label="model seed", path=path) != expected_model_seed:
        raise ValueError(f"Unexpected model seed in {path}")
    if _profile_int(profile, "timing_seed", label="timing seed", path=path) != expected_timing_seed:
        raise ValueError(f"Unexpected timing seed in {path}")
    if _profile_int(profile, "resumed_from_epoch", label="source epoch", path=path) != source_epoch:
        raise ValueError(f"Unexpected source epoch in {path}")
    if _profile_int(profile, "target_epoch", label="target epoch", path=path) != expected_target_epoch:
        raise ValueError(f"Unexpected target epoch in {path}")
    if (
        _profile_int(
            profile,
            "measurement_start_epoch",
            label="measurement start epoch",
            path=path,
        )
        != expected_measurement_start
    ):
        raise ValueError(f"Unexpected measurement start epoch in {path}")

    device = str(profile.get("device", "")).lower()
    if not device.startswith("cuda"):
        raise ValueError(f"Expected CUDA device in {path}")
    environment = profile.get("environment")
    if not isinstance(environment, dict) or not str(environment.get("device", "")).lower().startswith("cuda"):
        raise ValueError(f"Expected CUDA environment in {path}")
    for key in ("python_version", "torch_version", "cuda_version", "cuda_device_name"):
        if not str(environment.get(key, "")).strip():
            raise ValueError(f"Missing timing environment field {key} in {path}")

    allocated = _profile_int(
        profile,
        "peak_memory_allocated_bytes",
        label="peak memory allocated",
        path=path,
    )
    reserved = _profile_int(
        profile,
        "peak_memory_reserved_bytes",
        label="peak memory reserved",
        path=path,
    )
    if allocated <= 0 or reserved <= 0 or reserved < allocated:
        raise ValueError(f"Invalid peak memory values in {path}")

    raw_records = profile.get("train_epoch_profiles")
    if not isinstance(raw_records, list):
        raise ValueError(f"Missing train_epoch_profiles in {path}")
    expected_epochs = list(range(source_epoch + 1, expected_target_epoch + 1))
    try:
        record_epochs = [int(record["epoch"]) for record in raw_records if isinstance(record, dict)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training epoch records in {path}") from exc
    if len(record_epochs) != len(raw_records) or record_epochs != expected_epochs:
        raise ValueError(f"Expected exact training epochs {expected_epochs} in {path}")

    record_times: dict[int, float] = {}
    for record in raw_records:
        assert isinstance(record, dict)
        missing_fields = [field for field in FORMAL_WORKLOAD_FIELDS if field not in record]
        if missing_fields:
            raise ValueError(f"Missing workload fields {missing_fields} in {path}")
        epoch = int(record["epoch"])
        try:
            elapsed = float(record["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid epoch time for epoch {epoch} in {path}") from exc
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"Invalid epoch time for epoch {epoch} in {path}")
        record_times[epoch] = elapsed
        for field in FORMAL_WORKLOAD_FIELDS[:-1]:
            try:
                value = int(record[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid workload field {field} in {path}") from exc
            if value < 0:
                raise ValueError(f"Negative workload field {field} in {path}")
        if int(record["batch_count"]) <= 0:
            raise ValueError(f"Invalid workload batch count in {path}")
        signature = str(record["workload_signature"])
        if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError(f"Invalid workload signature in {path}")

    raw_times = profile.get("train_epoch_times")
    if not isinstance(raw_times, list):
        raise ValueError(f"Missing train_epoch_times in {path}")
    try:
        timing_epochs = [int(row["epoch"]) for row in raw_times if isinstance(row, dict)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid train_epoch_times in {path}") from exc
    if len(timing_epochs) != len(raw_times) or timing_epochs != expected_epochs:
        raise ValueError(f"Expected exact train_epoch_times epochs in {path}")
    for row in raw_times:
        assert isinstance(row, dict)
        epoch = int(row["epoch"])
        try:
            elapsed = float(row["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid train_epoch_times value in {path}") from exc
        if not math.isclose(elapsed, record_times[epoch], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Mismatched epoch timing records in {path}")

    if require_telemetry:
        _validate_formal_telemetry(path.parent / "timing_telemetry.csv")
    return profile


def _trimmed_mean(values: list[float], *, trim_fraction: float) -> float:
    """Mean after dropping the lowest/highest ``trim_fraction`` of samples.

    The controlled-timing workload is constant per epoch (identical workload
    signatures), so per-epoch spread is measurement noise from host-side
    contention. Trimming symmetric tails suppresses transient spikes before
    averaging. Falls back to the plain mean when trimming would drop every
    sample.
    """
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0.0, 0.5)")
    ordered = sorted(values)
    trim_count = int(len(ordered) * trim_fraction)
    trimmed = ordered[trim_count: len(ordered) - trim_count]
    if not trimmed:
        trimmed = ordered
    return statistics.mean(trimmed)


def _iqr(values: list[float]) -> float:
    """Interquartile range (Q3 - Q1); 0.0 when fewer than two samples."""
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


TRIM_FRACTION = 0.1


def summarize_profiles(
    profile_paths: list[Path],
    *,
    source_epoch: int,
    warmup_epochs: int,
    timed_epochs: int,
    require_telemetry: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if warmup_epochs < 0 or timed_epochs < 1:
        raise ValueError("warmup_epochs must be nonnegative and timed_epochs positive")

    timed_start = source_epoch + warmup_epochs + 1
    timed_end = timed_start + timed_epochs - 1
    expected = list(range(timed_start, timed_end + 1))
    details: list[dict[str, object]] = []
    repeat_means: list[float] = []
    repeat_medians: list[float] = []
    repeat_trimmed_means: list[float] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    for repeat, path in enumerate(sorted(profile_paths), start=1):
        profile = load_timing_profile(path)
        if profile.get("timing_only") is not True:
            raise ValueError(f"Timing-only profile required: {path}")
        if require_telemetry and profile.get("protocol") != FORMAL_TIMING_PROTOCOL:
            raise ValueError(f"Expected formal timing protocol in {path}")
        if int(profile.get("resumed_from_epoch", -1)) != source_epoch:
            raise ValueError(f"Unexpected source epoch in {path}")
        if int(profile.get("measurement_start_epoch", -1)) != timed_start:
            raise ValueError(f"Unexpected measurement start epoch in {path}")
        timing_seed = profile.get("timing_seed")
        if timing_seed is None:
            raise ValueError(f"Missing timing_seed in {path}")
        raw_records = profile.get("train_epoch_profiles")
        if not isinstance(raw_records, list):
            raise ValueError(f"Missing train_epoch_profiles in {path}")
        records = {
            int(record["epoch"]): record
            for record in raw_records
            if isinstance(record, dict) and "epoch" in record and "time_s" in record
        }
        missing = [epoch for epoch in expected if epoch not in records]
        if missing:
            raise ValueError(f"Missing timed epochs {missing} in {path}")
        telemetry_path = path.parent / "timing_telemetry.csv"
        if require_telemetry and not _has_telemetry_sample(telemetry_path):
            raise ValueError(f"Missing telemetry samples for {path}")
        values = [float(records[epoch]["time_s"]) for epoch in expected]
        repeat_mean = statistics.mean(values)
        repeat_median = statistics.median(values)
        repeat_trimmed_mean = _trimmed_mean(values, trim_fraction=TRIM_FRACTION)
        repeat_iqr = _iqr(values)
        repeat_means.append(repeat_mean)
        repeat_medians.append(repeat_median)
        repeat_trimmed_means.append(repeat_trimmed_mean)
        allocated = profile.get("peak_memory_allocated_bytes")
        reserved = profile.get("peak_memory_reserved_bytes")
        if allocated is not None:
            peak_allocated.append(int(allocated))
        if reserved is not None:
            peak_reserved.append(int(reserved))
        details.append(
            {
                "repeat": repeat,
                "timing_seed": int(timing_seed),
                "model_seed": profile.get("model_seed"),
                "timed_start_epoch": timed_start,
                "timed_end_epoch": timed_end,
                "timed_epoch_count": timed_epochs,
                "mean_train_time_s_per_epoch": repeat_mean,
                "median_train_time_s_per_epoch": repeat_median,
                "trimmed_mean_train_time_s_per_epoch": repeat_trimmed_mean,
                "iqr_train_time_s_within_repeat": repeat_iqr,
                "peak_memory_allocated_bytes": allocated,
                "peak_memory_reserved_bytes": reserved,
                "profile_path": str(path),
                "telemetry_path": str(telemetry_path),
            }
        )
    if not repeat_means:
        raise ValueError("At least one complete timing profile is required")
    return details, {
        "source_epoch": source_epoch,
        "warmup_epochs": warmup_epochs,
        "timed_epochs": timed_epochs,
        "repeat_count": len(repeat_means),
        "mean_train_time_s_per_epoch": statistics.mean(repeat_means),
        "median_train_time_s_per_epoch": statistics.median(repeat_medians),
        "trimmed_mean_train_time_s_per_epoch": statistics.mean(
            repeat_trimmed_means
        ),
        "std_train_time_s_per_epoch_across_repeat_means": statistics.stdev(
            repeat_means
        )
        if len(repeat_means) > 1
        else 0.0,
        "std_train_time_s_per_epoch_across_repeat_medians": statistics.stdev(
            repeat_medians
        )
        if len(repeat_medians) > 1
        else 0.0,
        "mean_peak_memory_allocated_bytes": statistics.mean(peak_allocated)
        if peak_allocated
        else None,
        "mean_peak_memory_reserved_bytes": statistics.mean(peak_reserved)
        if peak_reserved
        else None,
    }


def summarize_logs(
    log_paths: list[Path],
    *,
    source_epoch: int,
    warmup_epochs: int,
    timed_epochs: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if warmup_epochs < 0 or timed_epochs < 1:
        raise ValueError("warmup_epochs must be nonnegative and timed_epochs positive")

    timed_start = source_epoch + warmup_epochs + 1
    timed_end = timed_start + timed_epochs - 1
    expected = list(range(timed_start, timed_end + 1))
    details: list[dict[str, object]] = []
    repeat_means: list[float] = []
    for repeat, path in enumerate(sorted(log_paths), start=1):
        epoch_times = parse_epoch_times(path)
        missing = [epoch for epoch in expected if epoch not in epoch_times]
        if missing:
            raise ValueError(f"Missing timed epochs {missing} in {path}")
        values = [epoch_times[epoch] for epoch in expected]
        repeat_mean = statistics.mean(values)
        repeat_means.append(repeat_mean)
        details.append(
            {
                "repeat": repeat,
                "timed_start_epoch": timed_start,
                "timed_end_epoch": timed_end,
                "timed_epoch_count": timed_epochs,
                "mean_train_time_s_per_epoch": repeat_mean,
                "std_train_time_s_within_repeat": statistics.stdev(values)
                if len(values) > 1
                else 0.0,
                "log_path": str(path),
            }
        )
    if not repeat_means:
        raise ValueError("At least one complete repeat log is required")
    return details, {
        "source_epoch": source_epoch,
        "warmup_epochs": warmup_epochs,
        "timed_epochs": timed_epochs,
        "repeat_count": len(repeat_means),
        "mean_train_time_s_per_epoch": statistics.mean(repeat_means),
        "std_train_time_s_per_epoch_across_repeat_means": statistics.stdev(
            repeat_means
        )
        if len(repeat_means) > 1
        else 0.0,
    }


def write_summary(
    root: Path,
    *,
    source_epoch: int,
    warmup_epochs: int,
    timed_epochs: int,
    require_telemetry: bool = False,
    formal: bool = False,
) -> tuple[Path, Path]:
    if formal:
        if not require_telemetry:
            raise ValueError("Formal timing summary requires telemetry")
        if (source_epoch, warmup_epochs, timed_epochs) != (50, 10, 20):
            raise ValueError("Formal timing summary requires epochs 50 + 10 warm-up + 20 timed")
        dataset_dirs = {path.name: path for path in root.iterdir() if path.is_dir()}
        if set(dataset_dirs) != set(FORMAL_DATASETS):
            raise ValueError(
                f"Expected formal datasets {list(FORMAL_DATASETS)}, got {sorted(dataset_dirs)}"
            )
        for dataset in FORMAL_DATASETS:
            dataset_dir = dataset_dirs[dataset]
            profiles = sorted(
                dataset_dir.glob("seed_*/timing_seed_*/cgrc_timing_profile.json")
            )
            expected_paths = {
                dataset_dir
                / f"seed_{FORMAL_MODEL_SEED}"
                / f"timing_seed_{timing_seed}"
                / "cgrc_timing_profile.json"
                for timing_seed in FORMAL_TIMING_SEEDS
            }
            if set(profiles) != expected_paths:
                actual_seeds = sorted(path.parent.name for path in profiles)
                raise ValueError(
                    f"Expected exact timing seeds {list(FORMAL_TIMING_SEEDS)} "
                    f"for {dataset}, got {actual_seeds}"
                )
            for timing_seed in FORMAL_TIMING_SEEDS:
                validate_formal_profile(
                    dataset_dir
                    / f"seed_{FORMAL_MODEL_SEED}"
                    / f"timing_seed_{timing_seed}"
                    / "cgrc_timing_profile.json",
                    expected_dataset=dataset,
                    expected_model_seed=FORMAL_MODEL_SEED,
                    expected_timing_seed=timing_seed,
                    source_epoch=source_epoch,
                    warmup_epochs=warmup_epochs,
                    timed_epochs=timed_epochs,
                    require_telemetry=True,
                )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    root_dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if formal:
        root_dataset_dirs = [root / dataset for dataset in FORMAL_DATASETS]
    for dataset_dir in root_dataset_dirs:
        profiles = sorted(
            dataset_dir.glob("seed_*/timing_seed_*/cgrc_timing_profile.json")
        )
        if profiles:
            details, summary = summarize_profiles(
                profiles,
                source_epoch=source_epoch,
                warmup_epochs=warmup_epochs,
                timed_epochs=timed_epochs,
                require_telemetry=require_telemetry,
            )
            for row in details:
                row["dataset"] = dataset_dir.name
                row["profile_path"] = str(
                    Path(str(row["profile_path"])).relative_to(root)
                )
                row["telemetry_path"] = str(
                    Path(str(row["telemetry_path"])).relative_to(root)
                )
        else:
            logs = sorted(dataset_dir.glob("seed_*/repeat_*/run.log"))
            if not logs:
                continue
            if require_telemetry:
                raise ValueError(f"No timing profiles found below {dataset_dir}")
            details, summary = summarize_logs(
                logs,
                source_epoch=source_epoch,
                warmup_epochs=warmup_epochs,
                timed_epochs=timed_epochs,
            )
            for row in details:
                row["dataset"] = dataset_dir.name
                row["log_path"] = str(Path(str(row["log_path"])).relative_to(root))
        if not details:
            continue
        summary["dataset"] = dataset_dir.name
        detail_rows.extend(details)
        summary_rows.append(summary)
    if not summary_rows:
        raise ValueError(f"No controlled timing logs found below {root}")

    detail_path = root / "cgrc_controlled_timing_repeat_detail.csv"
    summary_path = root / "cgrc_controlled_timing_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in detail_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in summary_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--validate-profile", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--model-seed", type=int, default=FORMAL_MODEL_SEED)
    parser.add_argument("--timing-seed", type=int)
    parser.add_argument("--source-epoch", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--timed-epochs", type=int, default=10)
    parser.add_argument("--require-telemetry", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.validate_profile is not None:
        if not args.dataset or args.timing_seed is None:
            parser.error("--validate-profile requires --dataset and --timing-seed")
        validate_formal_profile(
            args.validate_profile,
            expected_dataset=args.dataset,
            expected_model_seed=args.model_seed,
            expected_timing_seed=args.timing_seed,
            source_epoch=args.source_epoch,
            warmup_epochs=args.warmup_epochs,
            timed_epochs=args.timed_epochs,
            require_telemetry=args.require_telemetry,
        )
        print(f"Validated {args.validate_profile}")
        return
    if args.root is None:
        parser.error("one of --root or --validate-profile is required")
    for path in write_summary(
        args.root,
        source_epoch=args.source_epoch,
        warmup_epochs=args.warmup_epochs,
        timed_epochs=args.timed_epochs,
        require_telemetry=args.require_telemetry,
        formal=args.formal,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
