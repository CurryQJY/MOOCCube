import csv
import json
import statistics
from pathlib import Path


ROOT = Path(r"D:\DeskTop\MOOCCube")
OUT_ROOT = ROOT / "outputs" / "junyi" / "mask_ablation"
MAIN_ROOT = ROOT / "outputs" / "junyi" / "main_table_3seed"

METRICS = ["r5", "r10", "r20", "n5", "n10", "n20"]


def split_dir(root: Path, seed: int) -> Path:
    return root / f"strict_item_cold_balanced_thr1_seed_{seed}"


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_final_fullrank(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


def load_run(condition: str, seed: int, candidates: list[Path], expected_known: bool, expected_same: bool) -> dict:
    errors = []
    for root in candidates:
        d = split_dir(root, seed)
        manifest_path = d / "static_protocol_manifest.json"
        result_path = d / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if not manifest_path.exists() or not result_path.exists():
            errors.append(f"missing under {d}")
            continue

        manifest = read_manifest(manifest_path)
        cfg = manifest.get("model_config", {})
        actual_known = bool(cfg.get("mask_known_pos_neg"))
        actual_same = bool(cfg.get("mask_same_item_neg"))
        if actual_known != expected_known or actual_same != expected_same:
            errors.append(
                f"flag mismatch under {d}: known={actual_known}, same={actual_same}"
            )
            continue

        row = read_final_fullrank(result_path)
        out = {
            "condition": condition,
            "seed": seed,
            "source_dir": str(d),
            "mask_known_pos_neg": actual_known,
            "mask_same_item_neg": actual_same,
        }
        for prefix in [
            "full_cold_item_macro",
            "full_hot_item_macro",
            "full_cold",
            "full_hot",
        ]:
            for metric in METRICS:
                out[f"{prefix}_{metric}"] = float(row[f"{prefix}_{metric}"])
        return out

    raise FileNotFoundError(
        f"No valid {condition} seed={seed} run found. Details: {' | '.join(errors)}"
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return sum(values) / len(values), statistics.stdev(values)


def main() -> None:
    tt_root = OUT_ROOT / "mask_tt"
    ff_root = OUT_ROOT / "mask_ff"

    detail_rows = []
    for seed in [2025, 2026, 2027]:
        detail_rows.append(
            load_run(
                "mask_tt",
                seed,
                [tt_root, MAIN_ROOT],
                expected_known=True,
                expected_same=True,
            )
        )
        detail_rows.append(
            load_run(
                "mask_ff",
                seed,
                [ff_root],
                expected_known=False,
                expected_same=False,
            )
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_ROOT / "junyi_mask_ablation_detail.csv"
    summary_path = OUT_ROOT / "junyi_mask_ablation_summary.csv"

    fieldnames = list(detail_rows[0].keys())
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = []
    for condition in ["mask_tt", "mask_ff"]:
        rows = [r for r in detail_rows if r["condition"] == condition]
        summary = {
            "condition": condition,
            "seeds": ";".join(str(r["seed"]) for r in rows),
            "runs": len(rows),
            "mask_known_pos_neg": rows[0]["mask_known_pos_neg"],
            "mask_same_item_neg": rows[0]["mask_same_item_neg"],
        }
        for prefix in [
            "full_cold_item_macro",
            "full_hot_item_macro",
            "full_cold",
            "full_hot",
        ]:
            for metric in METRICS:
                vals = [r[f"{prefix}_{metric}"] for r in rows]
                mean, std = mean_std(vals)
                summary[f"{prefix}_{metric}_mean"] = mean
                summary[f"{prefix}_{metric}_std"] = std
        summary_rows.append(summary)

    # Difference is true/true minus false/false. Positive means masking helped.
    tt = summary_rows[0]
    ff = summary_rows[1]
    delta = {
        "condition": "delta_tt_minus_ff",
        "seeds": "2025;2026;2027",
        "runs": 3,
        "mask_known_pos_neg": "",
        "mask_same_item_neg": "",
    }
    for key, value in tt.items():
        if key.endswith("_mean"):
            delta[key] = value - ff[key]
            ff_value = ff[key]
            delta[key.replace("_mean", "_rel_pct")] = (
                ((value / ff_value) - 1.0) * 100.0 if ff_value else ""
            )
    summary_rows.append(delta)

    all_fields = []
    for row in summary_rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    for row in summary_rows:
        if row["condition"] == "delta_tt_minus_ff":
            continue
        print(
            row["condition"],
            "cold_item_N@10=",
            f"{row['full_cold_item_macro_n10_mean']:.4f}"
            f"+/-{row['full_cold_item_macro_n10_std']:.4f}",
            "cold_item_R@10=",
            f"{row['full_cold_item_macro_r10_mean']:.4f}"
            f"+/-{row['full_cold_item_macro_r10_std']:.4f}",
        )
    print(
        "delta_tt_minus_ff",
        "cold_item_N@10_abs=",
        f"{delta['full_cold_item_macro_n10_mean']:.4f}",
        "cold_item_N@10_rel_pct=",
        f"{delta['full_cold_item_macro_n10_rel_pct']:.2f}",
    )


if __name__ == "__main__":
    main()
