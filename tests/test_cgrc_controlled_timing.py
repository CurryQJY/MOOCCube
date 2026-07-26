from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "paper_aaai27" / "scripts"
FORMAL_DATASETS = ("Junyi", "COCO", "MOOCCube")
FORMAL_TIMING_SEEDS = (9101, 9102, 9103, 9104, 9105)


def _load_summary_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("summarize_cgrc_controlled_timing")
    finally:
        sys.path.pop(0)


def _write_formal_profile(
    root: Path,
    dataset: str,
    timing_seed: int,
    *,
    profile_updates: dict[str, object] | None = None,
    telemetry_text: str | None = None,
) -> Path:
    run_dir = root / dataset / "seed_2026" / f"timing_seed_{timing_seed}"
    run_dir.mkdir(parents=True)
    signature = f"{timing_seed:064x}"
    records = [
        {
            "epoch": epoch,
            "time_s": float(epoch),
            "batch_count": 2,
            "sampled_cold_item_total": 4,
            "sampled_cold_item_max": 2,
            "masked_edge_count_raw_total": 8,
            "reconstruction_edge_count_total": 6,
            "reconstruction_active_batch_count": 2,
            "reconstruction_user_count_total": 4,
            "workload_signature": signature,
        }
        for epoch in range(51, 81)
    ]
    profile: dict[str, object] = {
        "model": "CGRC-paper",
        "protocol": "cgrc_formal_timing_v1",
        "timing_only": True,
        "static_seed": 2026,
        "model_seed": 2026,
        "timing_seed": timing_seed,
        "resumed_from_epoch": 50,
        "target_epoch": 80,
        "measurement_start_epoch": 61,
        "device": "cuda",
        "train_epoch_times": [
            {"epoch": record["epoch"], "time_s": record["time_s"]}
            for record in records
        ],
        "train_epoch_profiles": records,
        "peak_memory_allocated_bytes": 1024,
        "peak_memory_reserved_bytes": 2048,
        "environment": {
            "python_version": "3.12.10",
            "torch_version": "2.8.0+cu129",
            "cuda_version": "12.9",
            "device": "cuda",
            "cuda_device_name": "NVIDIA GeForce RTX 5070",
        },
    }
    if profile_updates:
        profile.update(profile_updates)
    profile_path = run_dir / "cgrc_timing_profile.json"
    profile_path.write_text(json.dumps([profile]), encoding="utf-8")
    if telemetry_text is None:
        telemetry_text = (
            "timestamp,gpu_index,pstate,gpu_utilization_pct,memory_used_mib,"
            "memory_total_mib,power_draw_w,temperature_c,sm_clock_mhz\n"
            "2026/07/23 10:00:00.000,0,P0,87,1024,12227,120.4,65,2205\n"
        )
    (run_dir / "timing_telemetry.csv").write_text(
        telemetry_text,
        encoding="utf-8",
    )
    return profile_path


def _build_formal_root(root: Path) -> None:
    for dataset in FORMAL_DATASETS:
        for timing_seed in FORMAL_TIMING_SEEDS:
            _write_formal_profile(root, dataset, timing_seed)


def test_controlled_timing_summary_uses_repeat_means(tmp_path: Path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        module = importlib.import_module("summarize_cgrc_controlled_timing")
    finally:
        sys.path.pop(0)

    logs = []
    for repeat, values in enumerate(
        ([80.0, 70.0, 31.0, 32.0], [79.0, 69.0, 33.0, 34.0]), start=1
    ):
        path = tmp_path / f"repeat_{repeat}" / "run.log"
        path.parent.mkdir()
        path.write_text(
            "\n".join(
                f"[CGRC-TRAIN] Epoch {epoch}/54 Time: {value:.2f}s"
                for epoch, value in zip(range(51, 55), values)
            ),
            encoding="utf-8",
        )
        logs.append(path)

    details, summary = module.summarize_logs(
        logs,
        source_epoch=50,
        warmup_epochs=2,
        timed_epochs=2,
    )

    assert [row["repeat"] for row in details] == [1, 2]
    assert [row["mean_train_time_s_per_epoch"] for row in details] == [31.5, 33.5]
    assert summary["repeat_count"] == 2
    assert summary["mean_train_time_s_per_epoch"] == 32.5
    assert math.isclose(
        summary["std_train_time_s_per_epoch_across_repeat_means"],
        math.sqrt(2.0),
    )


def test_controlled_timing_summary_reads_complete_profiles_and_telemetry(tmp_path: Path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        module = importlib.import_module("summarize_cgrc_controlled_timing")
    finally:
        sys.path.pop(0)

    for timing_seed, values in ((9101, (31.0, 33.0)), (9102, (35.0, 37.0))):
        run_dir = tmp_path / "Junyi" / "seed_2026" / f"timing_seed_{timing_seed}"
        run_dir.mkdir(parents=True)
        profile = {
            "timing_only": True,
            "protocol": "cgrc_formal_timing_v1",
            "static_seed": 2026,
            "model_seed": 2026,
            "timing_seed": timing_seed,
            "resumed_from_epoch": 50,
            "target_epoch": 62,
            "measurement_start_epoch": 61,
            "peak_memory_allocated_bytes": 1024,
            "peak_memory_reserved_bytes": 2048,
            "train_epoch_profiles": [
                {
                    "epoch": epoch,
                    "time_s": value,
                    "batch_count": 1,
                    "sampled_cold_item_total": 1,
                    "masked_edge_count_raw_total": 1,
                    "reconstruction_edge_count_total": 1,
                    "reconstruction_active_batch_count": 1,
                    "reconstruction_user_count_total": 1,
                    "workload_signature": f"sig-{timing_seed}-{epoch}",
                }
                for epoch, value in zip((61, 62), values)
            ],
        }
        (run_dir / "cgrc_timing_profile.json").write_text(
            json.dumps([profile]), encoding="utf-8"
        )
        (run_dir / "timing_telemetry.csv").write_text(
            "timestamp,gpu_utilization_pct\n2026-07-23T00:00:00,90\n",
            encoding="utf-8",
        )

    detail_path, summary_path = module.write_summary(
        tmp_path,
        source_epoch=50,
        warmup_epochs=10,
        timed_epochs=2,
        require_telemetry=True,
    )

    with detail_path.open(newline="", encoding="utf-8") as handle:
        details = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))

    assert [row["timing_seed"] for row in details] == ["9101", "9102"]
    assert [float(row["mean_train_time_s_per_epoch"]) for row in details] == [32.0, 36.0]
    assert summary[0]["repeat_count"] == "2"
    assert float(summary[0]["mean_train_time_s_per_epoch"]) == 34.0


def test_formal_timing_summary_rejects_diagnostic_profiles(tmp_path: Path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        module = importlib.import_module("summarize_cgrc_controlled_timing")
    finally:
        sys.path.pop(0)

    run_dir = tmp_path / "Junyi" / "seed_2026" / "timing_seed_9101"
    run_dir.mkdir(parents=True)
    profile = {
        "timing_only": True,
        "protocol": "timing_only",
        "timing_seed": 9101,
        "resumed_from_epoch": 50,
        "measurement_start_epoch": 51,
        "train_epoch_profiles": [{"epoch": 51, "time_s": 1.0}],
    }
    (run_dir / "cgrc_timing_profile.json").write_text(
        json.dumps([profile]), encoding="utf-8"
    )
    (run_dir / "timing_telemetry.csv").write_text(
        "timestamp,gpu_utilization_pct\n2026-07-23T00:00:00,90\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="formal timing protocol"):
        module.write_summary(
            tmp_path,
            source_epoch=50,
            warmup_epochs=0,
            timed_epochs=1,
            require_telemetry=True,
        )


def test_controlled_timing_launcher_is_isolated_and_prioritizes_junyi():
    source = (ROOT / "run_cgrc_controlled_timing.ps1").read_text(encoding="utf-8")

    assert '[string[]]$Datasets = @("Junyi", "COCO", "MOOCCube")' in source
    assert "$jobs = @($jobs | Where-Object { $Datasets -contains $_.Dataset })" in source
    assert source.index('Dataset = "Junyi"') < source.index('Dataset = "COCO"')
    assert source.index('Dataset = "COCO"') < source.index('Dataset = "MOOCCube"')
    coco_start = source.index('Dataset = "COCO"')
    mooccube_start = source.index('Dataset = "MOOCCube"')
    assert "ReconUserChunk = 256" in source[coco_start:mooccube_start]
    assert '$env:CGRC_PAPER_RECON_USER_CHUNK = "$($job.ReconUserChunk)"' in source
    assert 'CGRC_PAPER_SAVE_CKPT = "0"' in source
    assert 'CGRC_PAPER_AUTO_RESUME = "1"' in source
    assert 'CGRC_PAPER_FORCE_FRESH = "0"' in source
    assert 'CGRC_PAPER_PROGRESS_INTERVAL = "0"' in source
    assert 'CGRC_PAPER_EVAL_SPLIT = "validation"' in source
    assert "$sourceEpoch + $WarmupEpochs + $TimedEpochs" in source
    assert "cgrc_formal_timing_v1" in source
    assert "build_revision_tables.py" not in source
    assert "export_efficiency_table.py" not in source


def test_formal_timing_launcher_locks_protocol_and_records_telemetry():
    source = (ROOT / "run_cgrc_controlled_timing.ps1").read_text(encoding="utf-8")

    assert "[int]$WarmupEpochs = 10" in source
    assert "[int]$TimedEpochs = 20" in source
    assert "[int[]]$TimingSeeds = @(9101, 9102, 9103, 9104, 9105)" in source
    assert 'CGRC_PAPER_TIMING_SEED = "$timingSeed"' in source
    assert 'CGRC_PAPER_TIMING_MEASURE_START_EPOCH = "$timedStartEpoch"' in source
    assert "CGRC_PAPER_TIMING_PROTOCOL = $protocolId" in source
    assert "monitor_cgrc_gpu_telemetry.py" in source
    assert "timing_telemetry.csv" in source
    assert "--require-telemetry" in source
    assert 'timing_seed_$timingSeed' in source


def test_formal_launcher_starts_telemetry_without_a_cmd_wrapper():
    source = (ROOT / "run_cgrc_controlled_timing.ps1").read_text(encoding="utf-8")

    assert '$telemetryLauncher = Join-Path $Repo "py.bat"' in source
    assert "Start-Process -FilePath $telemetryLauncher" in source
    assert "-RedirectStandardError $telemetryErrorPath" in source
    assert "Start-Process -FilePath \"cmd.exe\" -ArgumentList @(\"/d\", \"/c\", $telemetryCommand)" not in source


def test_formal_summary_requires_the_exact_15_block_matrix(tmp_path: Path):
    module = _load_summary_module()
    _build_formal_root(tmp_path)

    detail_path, summary_path = module.write_summary(
        tmp_path,
        source_epoch=50,
        warmup_epochs=10,
        timed_epochs=20,
        require_telemetry=True,
        formal=True,
    )

    with detail_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 15
    with summary_path.open(newline="", encoding="utf-8") as handle:
        assert {row["dataset"] for row in csv.DictReader(handle)} == set(FORMAL_DATASETS)

    missing = (
        tmp_path
        / "MOOCCube"
        / "seed_2026"
        / "timing_seed_9105"
        / "cgrc_timing_profile.json"
    )
    missing.unlink()
    with pytest.raises(ValueError, match="exact timing seeds"):
        module.write_summary(
            tmp_path,
            source_epoch=50,
            warmup_epochs=10,
            timed_epochs=20,
            require_telemetry=True,
            formal=True,
        )


@pytest.mark.parametrize(
    ("profile_updates", "match"),
    [
        ({"target_epoch": 79}, "target epoch"),
        ({"model_seed": 2025}, "model seed"),
        ({"static_seed": 2025}, "static seed"),
        ({"device": "cpu"}, "CUDA device"),
        ({"peak_memory_allocated_bytes": None}, "peak memory"),
        ({"environment": {"device": "cpu"}}, "CUDA environment"),
    ],
)
def test_formal_profile_rejects_incomplete_protocol_metadata(
    tmp_path: Path,
    profile_updates: dict[str, object],
    match: str,
):
    module = _load_summary_module()
    profile_path = _write_formal_profile(
        tmp_path,
        "Junyi",
        9101,
        profile_updates=profile_updates,
    )

    with pytest.raises(ValueError, match=match):
        module.validate_formal_profile(
            profile_path,
            expected_dataset="Junyi",
            expected_model_seed=2026,
            expected_timing_seed=9101,
            source_epoch=50,
            warmup_epochs=10,
            timed_epochs=20,
            require_telemetry=True,
        )


def test_formal_profile_requires_all_warmup_epochs(tmp_path: Path):
    module = _load_summary_module()
    profile_path = _write_formal_profile(tmp_path, "Junyi", 9101)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))[0]
    profile["train_epoch_profiles"] = profile["train_epoch_profiles"][1:]
    profile_path.write_text(json.dumps([profile]), encoding="utf-8")

    with pytest.raises(ValueError, match="training epochs"):
        module.validate_formal_profile(
            profile_path,
            expected_dataset="Junyi",
            expected_model_seed=2026,
            expected_timing_seed=9101,
            source_epoch=50,
            warmup_epochs=10,
            timed_epochs=20,
            require_telemetry=True,
        )


def test_formal_profile_requires_workload_fields(tmp_path: Path):
    module = _load_summary_module()
    profile_path = _write_formal_profile(tmp_path, "Junyi", 9101)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))[0]
    profile["train_epoch_profiles"][0].pop("workload_signature")
    profile_path.write_text(json.dumps([profile]), encoding="utf-8")

    with pytest.raises(ValueError, match="workload fields"):
        module.validate_formal_profile(
            profile_path,
            expected_dataset="Junyi",
            expected_model_seed=2026,
            expected_timing_seed=9101,
            source_epoch=50,
            warmup_epochs=10,
            timed_epochs=20,
            require_telemetry=True,
        )


def test_formal_profile_rejects_malformed_telemetry(tmp_path: Path):
    module = _load_summary_module()
    profile_path = _write_formal_profile(
        tmp_path,
        "Junyi",
        9101,
        telemetry_text="timestamp,gpu_utilization_pct\nbad,not-a-number\n",
    )

    with pytest.raises(ValueError, match="telemetry schema"):
        module.validate_formal_profile(
            profile_path,
            expected_dataset="Junyi",
            expected_model_seed=2026,
            expected_timing_seed=9101,
            source_epoch=50,
            warmup_epochs=10,
            timed_epochs=20,
            require_telemetry=True,
        )


def test_formal_launcher_locks_exact_protocol_and_validates_existing_results():
    source = (ROOT / "run_cgrc_controlled_timing.ps1").read_text(encoding="utf-8")

    assert "9101,9102,9103,9104,9105" in source
    assert "Junyi,COCO,MOOCCube" in source
    assert "--validate-profile" in source
    assert "--formal" in source


def test_timing_only_profile_records_training_epochs_without_evaluation(monkeypatch):
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("cgrc_paper_static_hin")
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("CGRC_PAPER_TIMING_ONLY", "1")
    assert module.Config(10, 20).timing_only is True

    profile = module.build_timing_profile(
        static_seed=2026,
        start_epoch=50,
        n_epochs=62,
        device="cuda:0",
        epoch_times=[(51, 1.25), (52, 1.5)],
    )
    assert profile["timing_only"] is True
    assert profile["static_seed"] == 2026
    assert profile["resumed_from_epoch"] == 50
    assert profile["train_epoch_times"] == [
        {"epoch": 51, "time_s": 1.25},
        {"epoch": 52, "time_s": 1.5},
    ]


def test_timing_profile_records_auditable_protocol_metadata():
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("cgrc_paper_static_hin")
    finally:
        sys.path.pop(0)

    profile = module.build_timing_profile(
        static_seed=2026,
        model_seed=2026,
        timing_seed=9101,
        start_epoch=50,
        n_epochs=80,
        measurement_start_epoch=61,
        device="cuda:0",
        epoch_records=[
            {
                "epoch": 61,
                "time_s": 12.5,
                "batch_count": 4,
                "sampled_cold_item_total": 8,
                "masked_edge_count_raw_total": 16,
                "reconstruction_edge_count_total": 12,
                "reconstruction_active_batch_count": 3,
                "reconstruction_user_count_total": 9,
                "workload_signature": "abc123",
            }
        ],
        peak_memory_allocated_bytes=1024,
        peak_memory_reserved_bytes=2048,
        environment={"python_version": "3.11", "torch_version": "test"},
        timing_protocol="cgrc_formal_timing_v1",
    )

    assert profile["model_seed"] == 2026
    assert profile["timing_seed"] == 9101
    assert profile["protocol"] == "cgrc_formal_timing_v1"
    assert profile["measurement_start_epoch"] == 61
    assert profile["peak_memory_allocated_bytes"] == 1024
    assert profile["peak_memory_reserved_bytes"] == 2048
    assert profile["environment"] == {
        "python_version": "3.11",
        "torch_version": "test",
    }
    assert profile["train_epoch_profiles"] == [
        {
            "epoch": 61,
            "time_s": 12.5,
            "batch_count": 4,
            "sampled_cold_item_total": 8,
            "masked_edge_count_raw_total": 16,
            "reconstruction_edge_count_total": 12,
            "reconstruction_active_batch_count": 3,
            "reconstruction_user_count_total": 9,
            "workload_signature": "abc123",
        }
    ]


def test_timing_only_config_separates_sampling_seed_from_model_seed(monkeypatch):
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("cgrc_paper_static_hin")
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("CGRC_PAPER_STATIC_SEED", "2026")
    monkeypatch.setenv("CGRC_PAPER_SEED", "2026")
    monkeypatch.setenv("CGRC_PAPER_TIMING_SEED", "9101")
    monkeypatch.setenv("CGRC_PAPER_TIMING_MEASURE_START_EPOCH", "61")

    config = module.Config(10, 20)

    assert config.static_seed == 2026
    assert config.seed == 2026
    assert config.timing_seed == 9101
    assert config.timing_measure_start_epoch == 61


def test_timing_workload_summary_is_deterministic_and_auditable():
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("cgrc_paper_static_hin")
    finally:
        sys.path.pop(0)

    def build_summary():
        tracker = module.TimingWorkloadSummary()
        tracker.record_batch(
            cold_item_ids=np.array([2, 7], dtype=np.int64),
            raw_edge_count=6,
            reconstruction_edge_count=4,
            reconstruction_user_count=3,
        )
        tracker.record_batch(
            cold_item_ids=np.array([], dtype=np.int64),
            raw_edge_count=0,
            reconstruction_edge_count=0,
            reconstruction_user_count=0,
        )
        return tracker.as_dict()

    summary = build_summary()
    assert summary["batch_count"] == 2
    assert summary["sampled_cold_item_total"] == 2
    assert summary["sampled_cold_item_max"] == 2
    assert summary["masked_edge_count_raw_total"] == 6
    assert summary["reconstruction_edge_count_total"] == 4
    assert summary["reconstruction_active_batch_count"] == 1
    assert summary["reconstruction_user_count_total"] == 3
    assert len(summary["workload_signature"]) == 64
    assert summary["workload_signature"] == build_summary()["workload_signature"]


def test_timing_only_loop_reseeds_sampling_and_records_workload():
    source = (ROOT / "cgrc_paper_static_hin.py").read_text(encoding="utf-8")

    assert "setup_seed(cfg.timing_seed)" in source
    assert "TimingWorkloadSummary()" in source
    assert "torch.cuda.reset_peak_memory_stats" in source
    assert "epoch_records=timing_epoch_records" in source
    assert "measurement_start_epoch=cfg.timing_measure_start_epoch" in source
    assert "environment=collect_timing_environment(device)" in source


def test_workload_audit_is_disabled_for_non_timing_training():
    source = (ROOT / "cgrc_paper_static_hin.py").read_text(encoding="utf-8")

    assert "TimingWorkloadSummary() if cfg.timing_only else None" in source
    assert "if timing_workload is not None:" in source


def test_timing_only_launcher_uses_a_dedicated_completion_marker():
    source = (ROOT / "run_cgrc_controlled_timing.ps1").read_text(encoding="utf-8")

    assert "[switch]$TimingOnly" in source
    assert "cgrc_timing_profile.json" in source
    assert 'CGRC_PAPER_TIMING_ONLY = if ($TimingOnly.IsPresent) { "1" } else { "0" }' in source


def test_timing_only_syncs_before_the_training_timer_starts():
    source = (ROOT / "cgrc_paper_static_hin.py").read_text(encoding="utf-8")

    assert "_sync_device(device)\n        epoch_start = time.perf_counter()" in source
