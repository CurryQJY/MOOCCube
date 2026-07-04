import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


EPOCH_CSV = "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"
SUMMARY_CSV = "mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv"
MANIFEST = "static_protocol_manifest.json"
VARIANT_META = "tuning_variant.json"
RUN_TAG = "strict_item_cold_balanced_thr1_seed_2025"


def parse_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def best_validation_row(epoch_csv: Path) -> Dict:
    rows = read_csv_rows(epoch_csv)
    if not rows:
        return {}
    best = max(rows, key=lambda r: parse_float(r.get("Val_full_cold_N@10"), -1.0))
    return best


def summary_eval_row(summary_csv: Path, eval_name: str) -> Dict:
    for row in read_csv_rows(summary_csv):
        if row.get("Eval") == eval_name:
            return row
    return {}


def discover_runs(root: Path) -> Iterable[Path]:
    for manifest in root.rglob(MANIFEST):
        yield manifest.parent


def variant_meta_for_run(root: Path, run_dir: Path) -> Dict:
    # Expected layout: root / variant_id / strict_item... / manifest.
    try:
        variant_dir = run_dir.parent
    except IndexError:
        variant_dir = run_dir
    meta = read_json(variant_dir / VARIANT_META)
    if meta:
        return meta
    return {
        "id": variant_dir.name,
        "group": "unknown",
        "param": "unknown",
        "value": "",
        "seed": 2025,
        "tuned_config": {},
    }


def build_summary_row(root: Path, run_dir: Path) -> Dict:
    meta = variant_meta_for_run(root, run_dir)
    manifest = read_json(run_dir / MANIFEST)
    best = best_validation_row(run_dir / EPOCH_CSV)
    full_item = summary_eval_row(run_dir / SUMMARY_CSV, "full_rank_item_macro")
    full_inter = summary_eval_row(run_dir / SUMMARY_CSV, "full_rank")

    tuned = meta.get("tuned_config", {}) or {}
    model_cfg = manifest.get("model_config", {}) if isinstance(manifest, dict) else {}
    split_cfg = manifest.get("split", {}) if isinstance(manifest, dict) else {}

    row = {
        "variant_id": meta.get("id", run_dir.parent.name),
        "group": meta.get("group", ""),
        "param": meta.get("param", ""),
        "value": meta.get("value", ""),
        "seed": meta.get("seed", split_cfg.get("seed", "")),
        "selection_metric": meta.get("selection_metric", "validation full-ranking cold item-macro NDCG@10"),
        "best_epoch": parse_int(best.get("Epoch")),
        "best_val_cold_R@10": parse_float(best.get("Val_full_cold_R@10")),
        "best_val_cold_N@10": parse_float(best.get("Val_full_cold_N@10")),
        "best_val_hot_R@10": parse_float(best.get("Val_full_hot_R@10")),
        "best_val_hot_N@10": parse_float(best.get("Val_full_hot_N@10")),
        "test_cold_R@5": parse_float(full_item.get("Cold_R@5")),
        "test_cold_R@10": parse_float(full_item.get("Cold_R@10")),
        "test_cold_R@20": parse_float(full_item.get("Cold_R@20")),
        "test_cold_N@5": parse_float(full_item.get("Cold_N@5")),
        "test_cold_N@10": parse_float(full_item.get("Cold_N@10")),
        "test_cold_N@20": parse_float(full_item.get("Cold_N@20")),
        "test_hot_N@10": parse_float(full_item.get("Hot_N@10")),
        "test_interaction_cold_N@10": parse_float(full_inter.get("Cold_N@10")),
        "test_interaction_hot_N@10": parse_float(full_inter.get("Hot_N@10")),
        "cold_item_count": parse_int(full_item.get("ColdSamples")),
        "hot_item_count": parse_int(full_item.get("HotSamples")),
        "course_sample_beta": tuned.get("course_sample_beta", model_cfg.get("feedback_course_sample_beta", "")),
        "course_reward_scale": tuned.get("course_reward_scale", ""),
        "course_prereq_w": tuned.get("course_prereq_w", model_cfg.get("feedback_course_prereq_weight", "")),
        "course_concept_w": tuned.get("course_concept_w", model_cfg.get("feedback_course_concept_weight", "")),
        "course_diff_w": tuned.get("course_diff_w", model_cfg.get("feedback_course_difficulty_weight", "")),
        "course_redundant_w": tuned.get("course_redundant_w", model_cfg.get("feedback_course_redundant_weight", "")),
        "prereq_aux_weight": tuned.get("prereq_aux_weight", model_cfg.get("prereq_aux_weight", "")),
        "epochs": model_cfg.get("n_epochs", meta.get("epochs", "")),
        "patience": model_cfg.get("early_stop_patience", meta.get("patience", "")),
        "output_dir": str(run_dir),
    }
    return row


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_row(rows: List[Dict]) -> Optional[Dict]:
    for row in rows:
        if row.get("variant_id") == "default":
            return row
    return None


def make_sensitivity_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    default = default_row(rows)
    group_defaults = {
        "beta": ("CourseSampleBeta", 0.20),
        "reward": ("CourseRewardScale", 1.00),
        "prereq_aux": ("PrereqAuxWeight", 0.03),
    }

    for row in rows:
        if row.get("group") in group_defaults:
            out.append(row)

    if default:
        for group, (param, value) in group_defaults.items():
            copied = dict(default)
            copied["variant_id"] = f"{group}_default"
            copied["group"] = group
            copied["param"] = param
            copied["value"] = value
            out.append(copied)

    return sorted(
        out,
        key=lambda r: (
            str(r.get("group", "")),
            parse_float(r.get("value"), 0.0),
            str(r.get("variant_id", "")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/mooccubex/tuning_v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root
    rows = [build_summary_row(root, run_dir) for run_dir in discover_runs(root)]
    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("group", "")),
            parse_float(r.get("value"), 0.0),
            str(r.get("variant_id", "")),
        ),
    )

    summary_path = out_dir / "tuning_summary.csv"
    sensitivity_path = out_dir / "tuning_sensitivity_long.csv"
    write_csv(summary_path, rows)
    write_csv(sensitivity_path, make_sensitivity_rows(rows))

    if rows:
        best = max(rows, key=lambda r: parse_float(r.get("best_val_cold_N@10"), -1.0))
        print(f"Wrote {summary_path}")
        print(f"Wrote {sensitivity_path}")
        print(
            "Best by validation cold item-macro N@10: "
            f"{best['variant_id']} ({best['param']}={best['value']}) "
            f"epoch={best['best_epoch']} val_N@10={best['best_val_cold_N@10']:.6f} "
            f"test_N@10={best['test_cold_N@10']:.6f}"
        )
    else:
        print(f"No completed tuning runs found under {root}")


if __name__ == "__main__":
    main()
