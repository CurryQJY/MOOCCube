#!/usr/bin/env python3
"""Summarize the isolated CBI-constrained simulator seed-2025 run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from summarize_cbi_faithful_seed2025 import (
    HISTORY_NAME,
    REPORT_NAME,
    build_comparison,
    parse_delta_stats,
    read_report,
    select_validation_epoch,
)


CANDIDATE_ROOT = Path("outputs/cbi_trust_sim_single_seed2025")
CANDIDATE_DIR = CANDIDATE_ROOT / "strict_item_cold_balanced_thr1_seed_2025"
ORIGINAL_CBI_DIR = Path(
    "outputs/cbi_faithful_single_seed2025/strict_item_cold_balanced_thr1_seed_2025"
)
BASELINE_DIR = Path(
    "outputs/content_delta_pop5/course_ablation_e60_3seed/full/"
    "strict_item_cold_balanced_thr1_seed_2025"
)
LOG_PATH = Path("background_logs/cbi_trust_sim_single_seed2025/training.log")
MANIFEST_PATH = CANDIDATE_ROOT / "run_manifest.json"
TRUST_PATTERN = re.compile(
    r"\[CBI-TRUST\]\s+episodes=(?P<episodes>\d+)\s+"
    r"projected=(?P<projected>[0-9.eE+-]+)%\s+"
    r"min_cos=(?P<min_cos>[0-9.eE+-]+)\s+"
    r"mean_cos=(?P<mean_cos>[0-9.eE+-]+)\s+"
    r"floor=(?P<floor>[0-9.eE+-]+)"
)


def parse_trust_history(path: Path | str) -> list[dict]:
    log_path = Path(path)
    matches = list(TRUST_PATTERN.finditer(log_path.read_text(encoding="utf-8")))
    if not matches:
        raise ValueError(f"no CBI trust diagnostics found in {log_path}")
    history = []
    for epoch, match in enumerate(matches, start=1):
        values = match.groupdict()
        row = {
            "epoch": epoch,
            "episodes": int(values["episodes"]),
            "projected_ratio": float(values["projected"]) / 100.0,
            "min_cosine": float(values["min_cos"]),
            "mean_cosine": float(values["mean_cos"]),
            "cosine_floor": float(values["floor"]),
        }
        if row["episodes"] < 1:
            raise ValueError(f"epoch {epoch} has no training trust episodes")
        if row["min_cosine"] + 1e-6 < row["cosine_floor"]:
            raise ValueError(
                f"epoch {epoch} violates trust floor: "
                f"{row['min_cosine']} < {row['cosine_floor']}"
            )
        history.append(row)
    return history


def _write_comparison_csv(path: Path, comparisons: dict) -> None:
    fields = [
        "reference",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reference, comparison in comparisons.items():
            for metric, values in comparison["metrics"].items():
                writer.writerow({"reference": reference, "metric": metric, **values})


def _write_markdown(path: Path, payload: dict) -> None:
    trust = payload["best_epoch_trust"]
    lines = [
        "# CBI-Constrained Simulator Seed-2025 Result",
        "",
        "This is an isolated one-seed structural validation and cannot replace the AAAI main table.",
        "",
        f"- Best validation epoch: `{payload['validation']['epoch']}`",
        f"- Trust projected ratio: `{trust['projected_ratio']:.2%}`",
        f"- Trust minimum cosine: `{trust['min_cosine']:.6f}`",
        f"- Trust cosine floor: `{trust['cosine_floor']:.6f}`",
        "",
        "| Reference | Metric | Candidate cold | Reference cold | Cold delta | Candidate hot | Reference hot | Hot delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for reference, comparison in payload["comparisons"].items():
        for metric, values in comparison["metrics"].items():
            lines.append(
                f"| {reference} | {metric} | {values['candidate_cold']:.6f} | "
                f"{values['baseline_cold']:.6f} | {values['cold_delta']:+.6f} | "
                f"{values['candidate_hot']:.6f} | {values['baseline_hot']:.6f} | "
                f"{values['hot_delta']:+.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    candidate_dir: Path = CANDIDATE_DIR,
    original_cbi_dir: Path = ORIGINAL_CBI_DIR,
    baseline_dir: Path = BASELINE_DIR,
    log_path: Path = LOG_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_root: Path | None = None,
) -> dict:
    output_root = output_root or candidate_dir.parent / "comparison"
    validation = select_validation_epoch(candidate_dir / HISTORY_NAME)
    trust_history = parse_trust_history(log_path)
    best_epoch = int(validation["epoch"])
    if best_epoch > len(trust_history):
        raise ValueError(
            f"best epoch {best_epoch} exceeds trust history length {len(trust_history)}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    protected_before = manifest.get("protected_files_before") or {}
    protected_after = manifest.get("protected_files_after") or {}
    if protected_before != protected_after:
        raise ValueError("protected shared files changed during trust-sim run")

    candidate = read_report(candidate_dir / REPORT_NAME)
    comparisons = {
        "original_cbi": build_comparison(candidate, read_report(original_cbi_dir / REPORT_NAME)),
        "seed2025_baseline": build_comparison(candidate, read_report(baseline_dir / REPORT_NAME)),
    }
    payload = {
        "scope": {
            "dataset": "MOOCCube",
            "seed": 2025,
            "protocol": "strict_item_cold_balanced",
            "status": "one_seed_structural_validation",
        },
        "validation": validation,
        "best_epoch_trust": trust_history[best_epoch - 1],
        "trust_history": trust_history,
        "delta_stats": parse_delta_stats(log_path),
        "comparisons": comparisons,
        "manifest_status": manifest.get("status"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cbi_trust_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_comparison_csv(output_root / "cbi_trust_comparison.csv", comparisons)
    _write_markdown(output_root / "cbi_trust_comparison.md", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--original-cbi-dir", type=Path, default=ORIGINAL_CBI_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    payload = summarize(
        candidate_dir=args.candidate_dir,
        original_cbi_dir=args.original_cbi_dir,
        baseline_dir=args.baseline_dir,
        log_path=args.log_path,
        manifest_path=args.manifest_path,
        output_root=args.output_root,
    )
    print(json.dumps(payload["best_epoch_trust"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
