#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPORT_NAME = "final_report_usim_feedback_fast3_content_delta_static.csv"
HISTORY_NAME = "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"
CANDIDATE_DIR = Path(
    "outputs/cbi_faithful_single_seed2025/strict_item_cold_balanced_thr1_seed_2025"
)
BASELINE_DIR = Path(
    "outputs/content_delta_pop5/course_ablation_e60_3seed/full/"
    "strict_item_cold_balanced_thr1_seed_2025"
)
LOG_PATH = Path("background_logs/cbi_faithful_single_seed2025/training.log")
OUTPUT_ROOT = Path("outputs/cbi_faithful_single_seed2025")
REQUIRED_METRICS = {
    f"{prefix}@{cutoff}" for prefix in ("R", "N") for cutoff in (5, 10, 20)
}
DELTA_PATTERN = re.compile(
    r"DeltaNorm\[mean=(?P<mean>[0-9.eE+-]+),\s*"
    r"max=(?P<max>[0-9.eE+-]+),\s*"
    r"eff_mean=(?P<eff_mean>[0-9.eE+-]+),\s*"
    r"eff_max=(?P<eff_max>[0-9.eE+-]+),\s*"
    r"clip=(?P<clip>[0-9.eE+-]+)%\]"
)


def read_report(path: Path | str) -> dict[str, dict[str, float]]:
    report_path = Path(path)
    rows: dict[str, dict[str, float]] = {}
    with report_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row["metric"].strip()
            rows[metric] = {
                "cold": float(row["full_cold_item_macro"]),
                "hot": float(row["full_hot_item_macro"]),
            }
    missing = REQUIRED_METRICS - rows.keys()
    if missing:
        raise ValueError(
            f"missing strict item-macro metrics in {report_path}: {sorted(missing)}"
        )
    return rows


def select_validation_epoch(path: Path | str) -> dict[str, float | int]:
    history_path = Path(path)
    with history_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"validation history is empty: {history_path}")
    best = max(rows, key=lambda row: float(row["Val_full_cold_N@10"]))
    return {
        "epoch": int(best["Epoch"]),
        "cold_r10": float(best["Val_full_cold_R@10"]),
        "hot_r10": float(best["Val_full_hot_R@10"]),
        "cold_n10": float(best["Val_full_cold_N@10"]),
        "hot_n10": float(best["Val_full_hot_N@10"]),
    }


def parse_delta_stats(path: Path | str) -> dict[str, float]:
    log_path = Path(path)
    matches = list(DELTA_PATTERN.finditer(log_path.read_text(encoding="utf-8")))
    if not matches:
        raise ValueError(f"no DeltaNorm diagnostics found in {log_path}")
    values = matches[-1].groupdict()
    return {
        "mean_norm": float(values["mean"]),
        "max_norm": float(values["max"]),
        "effective_mean_norm": float(values["eff_mean"]),
        "effective_max_norm": float(values["eff_max"]),
        "clipped_ratio": float(values["clip"]) / 100.0,
    }


def _relative_delta(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / baseline


def build_comparison(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> dict:
    metrics = {}
    for metric in sorted(REQUIRED_METRICS, key=lambda name: (name[0], int(name[2:]))):
        candidate_cold = candidate[metric]["cold"]
        baseline_cold = baseline[metric]["cold"]
        candidate_hot = candidate[metric]["hot"]
        baseline_hot = baseline[metric]["hot"]
        metrics[metric] = {
            "candidate_cold": candidate_cold,
            "baseline_cold": baseline_cold,
            "cold_delta": round(candidate_cold - baseline_cold, 12),
            "cold_relative": _relative_delta(candidate_cold, baseline_cold),
            "candidate_hot": candidate_hot,
            "baseline_hot": baseline_hot,
            "hot_delta": round(candidate_hot - baseline_hot, 12),
            "hot_relative": _relative_delta(candidate_hot, baseline_hot),
        }

    n10_gain = metrics["N@10"]["cold_delta"] >= 0.003
    r10_guard = metrics["R@10"]["cold_delta"] >= -0.002
    baseline_hot_n10 = metrics["N@10"]["baseline_hot"]
    candidate_hot_n10 = metrics["N@10"]["candidate_hot"]
    hot_guard = baseline_hot_n10 <= 0.0 or candidate_hot_n10 >= 0.95 * baseline_hot_n10
    return {
        "metrics": metrics,
        "screening": {
            "promising": bool(n10_gain and r10_guard and hot_guard),
            "cold_n10_gain_at_least_0p003": bool(n10_gain),
            "cold_r10_drop_not_below_minus_0p002": bool(r10_guard),
            "hot_n10_retains_at_least_95_percent": bool(hot_guard),
        },
    }


def _write_csv(path: Path, comparison: dict) -> None:
    fieldnames = [
        "metric",
        "candidate_cold",
        "baseline_cold",
        "cold_delta",
        "cold_relative",
        "candidate_hot",
        "baseline_hot",
        "hot_delta",
        "hot_relative",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in comparison["metrics"].items():
            writer.writerow({"metric": metric, **values})


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# CBI-Faithful Seed-2025 Screening Result",
        "",
        "This is a one-seed exploratory screening result. It cannot change the AAAI main table or support a statistical claim.",
        "",
        f"- Promising: `{payload['comparison']['screening']['promising']}`",
        f"- Validation-selected epoch: `{payload['validation']['epoch']}`",
        f"- Delta clipped ratio: `{payload['delta_stats']['clipped_ratio']:.2%}`",
        "",
        "| Metric | Candidate cold | Baseline cold | Cold delta | Candidate hot | Baseline hot | Hot delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, values in payload["comparison"]["metrics"].items():
        lines.append(
            "| {metric} | {cc} | {bc} | {cd} | {ch} | {bh} | {hd} |".format(
                metric=metric,
                cc=_format_metric(values["candidate_cold"]),
                bc=_format_metric(values["baseline_cold"]),
                cd=_format_metric(values["cold_delta"]),
                ch=_format_metric(values["candidate_hot"]),
                bh=_format_metric(values["baseline_hot"]),
                hd=_format_metric(values["hot_delta"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    candidate_dir: Path = CANDIDATE_DIR,
    baseline_dir: Path = BASELINE_DIR,
    log_path: Path = LOG_PATH,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    candidate_report = candidate_dir / REPORT_NAME
    baseline_report = baseline_dir / REPORT_NAME
    history_path = candidate_dir / HISTORY_NAME
    comparison = build_comparison(
        read_report(candidate_report),
        read_report(baseline_report),
    )
    payload = {
        "scope": {
            "dataset": "MOOCCube",
            "seed": 2025,
            "protocol": "strict_item_cold_balanced",
            "status": "one_seed_screening_only",
        },
        "paths": {
            "candidate_report": str(candidate_report.resolve()),
            "baseline_report": str(baseline_report.resolve()),
            "history": str(history_path.resolve()),
            "training_log": str(log_path.resolve()),
        },
        "validation": select_validation_epoch(history_path),
        "delta_stats": parse_delta_stats(log_path),
        "comparison": comparison,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "cbi_comparison.json"
    csv_path = output_root / "cbi_comparison.csv"
    markdown_path = output_root / "cbi_comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(csv_path, comparison)
    _write_markdown(markdown_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    payload = summarize(
        candidate_dir=args.candidate_dir,
        baseline_dir=args.baseline_dir,
        log_path=args.log_path,
        output_root=args.output_root,
    )
    print(json.dumps(payload["comparison"]["screening"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
