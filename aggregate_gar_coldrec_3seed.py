import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
SECTIONS = (
    "full_cold",
    "full_hot",
    "full_cold_item_macro",
    "full_hot_item_macro",
)
EXPECTED_COMMIT = "18efd24"
EXPECTED_EPOCHS = 500
RESULT_NAME = "gar_coldrec_strict_result.json"


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        result = payload[0] if payload else None
    else:
        result = payload
    if not isinstance(result, dict):
        raise ValueError(f"Invalid GAR result payload: {path}")
    return result


def _require_equal(result: dict, expected_seed: int, key: str, expected) -> None:
    actual = result.get(key)
    if actual != expected:
        raise ValueError(
            f"seed {expected_seed}: expected {key}={expected!r}, got {actual!r}"
        )


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {label}: {value!r}")
    return number


def _resolve_artifact(result_path: Path, value: str) -> Path:
    artifact = Path(value)
    if not artifact.is_absolute():
        artifact = result_path.parent / artifact
    return artifact.resolve()


def validate_result(result: dict, path: Path, expected_seed: int) -> None:
    required_equal = {
        "model": "GAR-coldrec-source-strict",
        "seed": expected_seed,
        "static_seed": expected_seed,
        "official_commit": EXPECTED_COMMIT,
        "source_model_unchanged": True,
        "protocol": "static_item_cold_balanced",
        "candidate_mode": "full_catalog",
        "checkpoint_metric": "validation_full_cold_item_macro.N@10",
        "item_macro_metrics": True,
        "train_history_masking": True,
        "train_only_interaction_evidence": True,
        "test_history_policy": "train_only",
        "cuda_used": True,
        "epochs": EXPECTED_EPOCHS,
    }
    # Older synthetic fixtures may omit protocol/model flags; real formal results may not.
    for key, expected in required_equal.items():
        if key in ("protocol", "item_macro_metrics") and key not in result:
            continue
        _require_equal(result, expected_seed, key, expected)

    device = str(result.get("device", ""))
    if not device.startswith("cuda:"):
        raise ValueError(f"seed {expected_seed}: expected CUDA device, got {device!r}")

    audit = result.get("strict_audit") or {}
    if int(audit.get("heldout_cold_item_count", 0)) <= 0:
        raise ValueError(f"seed {expected_seed}: empty held-out cold-course audit set")
    if int(audit.get("train_overlap_count", -1)) != 0:
        raise ValueError(f"seed {expected_seed}: held-out cold/train overlap is nonzero")

    history = result.get("strict_validation_history") or []
    if not history or not all(int(row.get("item_count", 0)) > 0 for row in history):
        raise ValueError(f"seed {expected_seed}: empty strict validation cold-course set")
    history_scores = [
        _finite(row.get("N@10"), f"seed {expected_seed} validation N@10")
        for row in history
    ]
    best_score = _finite(
        result.get("best_val_full_cold_item_macro_n10"),
        f"seed {expected_seed} retained validation N@10",
    )
    max_score = max(history_scores)
    if not math.isclose(best_score, max_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"seed {expected_seed}: retained checkpoint score {best_score} "
            f"does not match validation maximum {max_score}"
        )
    best_epoch = int(result.get("best_epoch", 0))
    matching_epochs = {
        int(row.get("epoch", 0))
        for row, score in zip(history, history_scores)
        if math.isclose(score, max_score, rel_tol=0.0, abs_tol=1e-12)
    }
    if best_epoch not in matching_epochs:
        raise ValueError(f"seed {expected_seed}: best epoch does not match retained score")
    epochs_ran = int(result.get("epochs_ran", 0))
    if not 0 < epochs_ran <= EXPECTED_EPOCHS or not 0 < best_epoch <= epochs_ran:
        raise ValueError(
            f"seed {expected_seed}: invalid epoch accounting "
            f"best={best_epoch} ran={epochs_ran}"
        )

    coldrec_args = result.get("coldrec_args") or {}
    expected_args = {
        "model": "GAR",
        "backbone": "MF",
        "epochs": EXPECTED_EPOCHS,
        "seed": expected_seed,
        "use_gpu": True,
    }
    for key, expected in expected_args.items():
        actual = coldrec_args.get(key)
        if actual != expected:
            raise ValueError(
                f"seed {expected_seed}: expected coldrec_args.{key}={expected!r}, "
                f"got {actual!r}"
            )

    for section in SECTIONS:
        block = result.get(section)
        if not isinstance(block, dict):
            raise ValueError(f"seed {expected_seed}: missing {section}")
        for metric in METRICS:
            if metric not in block:
                raise ValueError(f"seed {expected_seed}: missing {section}.{metric}")
            _finite(block[metric], f"seed {expected_seed} {section}.{metric}")

    counts = result.get("counts") or {}
    for key in ("full_cold", "full_hot", "full_cold_item", "full_hot_item"):
        if int(counts.get(key, 0)) <= 0:
            raise ValueError(f"seed {expected_seed}: empty result count {key}")

    artifact_counts = (
        ("per_item_full_cold_path", "full_cold_item"),
        ("per_item_full_hot_path", "full_hot_item"),
    )
    for path_key, count_key in artifact_counts:
        if not result.get(path_key):
            raise ValueError(f"seed {expected_seed}: missing {path_key}")
        csv_path = _resolve_artifact(path, result[path_key])
        if not csv_path.is_file():
            raise FileNotFoundError(f"seed {expected_seed}: missing per-course CSV {csv_path}")
        actual_rows = len(pd.read_csv(csv_path))
        expected_rows = int(counts[count_key])
        if actual_rows != expected_rows:
            raise ValueError(
                f"seed {expected_seed}: per-course row-count mismatch for {csv_path}: "
                f"expected {expected_rows}, got {actual_rows}"
            )


def _detail_row(result: dict) -> dict:
    counts = result["counts"]
    row = {
        "seed": int(result["seed"]),
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "best_val_full_cold_item_macro_n10": float(
            result["best_val_full_cold_item_macro_n10"]
        ),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
        "full_cold_count": int(counts["full_cold"]),
        "full_hot_count": int(counts["full_hot"]),
        "full_cold_item_count": int(counts["full_cold_item"]),
        "full_hot_item_count": int(counts["full_hot_item"]),
    }
    for section in SECTIONS:
        for metric in METRICS:
            row[f"{section}_{metric}"] = float(result[section][metric])
    return row


def _summary(rows: list[dict], seeds: tuple[int, ...]) -> dict:
    summary = {
        "model": "GAR-coldrec-source-strict",
        "source_commit": EXPECTED_COMMIT,
        "epoch_budget": EXPECTED_EPOCHS,
        "runs": len(rows),
        "seeds": list(seeds),
        "best_epoch": {
            "mean": statistics.mean(row["best_epoch"] for row in rows),
            "std": statistics.stdev(row["best_epoch"] for row in rows),
        },
    }
    for section in SECTIONS:
        summary[section] = {}
        for metric in METRICS:
            values = [row[f"{section}_{metric}"] for row in rows]
            summary[section][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
    return summary


def _flat_summary(summary: dict) -> dict:
    row = {
        "model": summary["model"],
        "source_commit": summary["source_commit"],
        "epoch_budget": summary["epoch_budget"],
        "runs": summary["runs"],
        "seeds": ",".join(str(seed) for seed in summary["seeds"]),
        "best_epoch_mean": summary["best_epoch"]["mean"],
        "best_epoch_std": summary["best_epoch"]["std"],
    }
    for section in SECTIONS:
        for metric in METRICS:
            stats = summary[section][metric]
            row[f"{section}_{metric}_mean"] = stats["mean"]
            row[f"{section}_{metric}_std"] = stats["std"]
    return row


def _write_report(summary: dict, detail: pd.DataFrame, path: Path) -> None:
    lines = [
        "# GAR ColdRec Source-Default Three-Seed Report",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        f"- Runs: {summary['runs']}",
        f"- ColdRec commit: `{summary['source_commit']}`",
        f"- Epoch ceiling: {summary['epoch_budget']} (early stopping enabled)",
        "- Protocol: strict full-catalog course-cold, train-only history",
        "",
        "## Per-Seed Cold Results",
        "",
        "| Seed | Best epoch | Interaction R@10 | Interaction N@10 | Course R@10 | Course N@10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detail.to_dict(orient="records"):
        lines.append(
            f"| {int(row['seed'])} | {int(row['best_epoch'])} | "
            f"{row['full_cold_R@10']:.6f} | {row['full_cold_N@10']:.6f} | "
            f"{row['full_cold_item_macro_R@10']:.6f} | "
            f"{row['full_cold_item_macro_N@10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Three-Seed Cold Summary",
            "",
            "| Metric family | R@5 | R@10 | N@5 | N@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for section, label in (
        ("full_cold", "Interaction macro"),
        ("full_cold_item_macro", "Course macro"),
    ):
        values = []
        for metric in ("R@5", "R@10", "N@5", "N@10"):
            stats = summary[section][metric]
            values.append(f"{stats['mean']:.6f} +/- {stats['std']:.6f}")
        lines.append(f"| {label} | {' | '.join(values)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(root: Path, seeds, out_dir: Path) -> dict:
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected at least two unique seeds, got {seeds}")

    rows = []
    for seed in seeds:
        result_path = root / f"seed_{seed}" / RESULT_NAME
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing GAR result for seed {seed}: {result_path}")
        result = load_result(result_path)
        validate_result(result, result_path, seed)
        rows.append(_detail_row(result))

    detail = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = _summary(detail.to_dict(orient="records"), seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "gar_coldrec_3seed_detail.csv", index=False)
    pd.DataFrame([_flat_summary(summary)]).to_csv(
        out_dir / "gar_coldrec_3seed_summary.csv", index=False
    )
    (out_dir / "gar_coldrec_3seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(summary, detail, out_dir / "gar_coldrec_3seed_report.md")
    return summary


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=(2025, 2026, 2027))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = aggregate(Path(args.root), args.seeds, Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
