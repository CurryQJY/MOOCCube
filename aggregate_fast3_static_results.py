import argparse
import re
from pathlib import Path

import pandas as pd


METRIC_COLS = [
    "full_cold_r5",
    "full_cold_r10",
    "full_cold_r20",
    "full_cold_n5",
    "full_cold_n10",
    "full_cold_n20",
    "full_hot_r5",
    "full_hot_r10",
    "full_hot_r20",
    "full_hot_n5",
    "full_hot_n10",
    "full_hot_n20",
    "full_cold_item_macro_r5",
    "full_cold_item_macro_r10",
    "full_cold_item_macro_r20",
    "full_cold_item_macro_n5",
    "full_cold_item_macro_n10",
    "full_cold_item_macro_n20",
    "full_hot_item_macro_r5",
    "full_hot_item_macro_r10",
    "full_hot_item_macro_r20",
    "full_hot_item_macro_n5",
    "full_hot_item_macro_n10",
    "full_hot_item_macro_n20",
    "full_cold_count",
    "full_hot_count",
    "full_cold_item_macro_count",
    "full_hot_item_macro_count",
]


def parse_run_tag(path):
    tag = path.parent.name
    threshold = None
    seed = None
    match = re.search(r"thr(\d+)", tag)
    if match:
        threshold = int(match.group(1))
    match = re.search(r"seed[_-]?(\d+)", tag)
    if match:
        seed = int(match.group(1))
    return tag, threshold, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/content_delta_pop5/static_item_cold",
        help="Directory containing per-seed static FAST3 runs.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <root>/fast3_static_multiseed_summary.csv.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/final_fullrank_usim_feedback_fast3_content_delta_static.csv"))
    if not files:
        raise SystemExit(f"No final full-rank files found under {root}")

    rows = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        tag, threshold, seed = parse_run_tag(path)
        row["run_tag"] = tag
        row["threshold"] = threshold
        row["seed"] = seed
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail_path = root / "fast3_static_runs_detail.csv"
    detail.to_csv(detail_path, index=False)

    group_cols = ["model", "protocol", "threshold"]
    metric_cols = [col for col in METRIC_COLS if col in detail.columns]
    summary_parts = []
    for keys, group in detail.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        out["runs"] = int(len(group))
        out["seeds"] = ",".join(str(int(x)) for x in sorted(group["seed"].dropna().unique()))
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            out[f"{col}_mean"] = float(values.mean())
            out[f"{col}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        summary_parts.append(out)
    summary = pd.DataFrame(summary_parts)
    out_path = Path(args.out) if args.out else root / "fast3_static_multiseed_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
