"""Replace the DropoutNet row in the 3-seed main-table summary with the
official-protocol DropoutNet run.

This keeps the existing baseline audit table intact and writes a new summary
CSV that can be passed to ``export_main_table_3seed_docx.py --baseline-csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASE = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_item_macro_final_audit/main_table_item_macro_summary.csv"
)
DEFAULT_OFFICIAL = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_balanced_itemmacro_dropoutnet_official_v1/main_table_item_macro_summary.csv"
)
DEFAULT_OUT = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_item_macro_final_audit_with_dropoutnet_official/main_table_item_macro_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"Missing base summary: {args.base}")
    if not args.official.exists():
        raise FileNotFoundError(f"Missing official DropoutNet summary: {args.official}")

    base = pd.read_csv(args.base)
    official = pd.read_csv(args.official)
    official = official[official["model"].astype(str).eq("DropoutNet")]
    if official.empty:
        raise ValueError(f"No DropoutNet row found in {args.official}")
    if official.shape[0] != 1:
        raise ValueError(f"Expected one DropoutNet row in {args.official}, found {official.shape[0]}")

    out = base[~base["model"].astype(str).eq("DropoutNet")].copy()
    out = pd.concat([out, official], ignore_index=True, sort=False)

    order = {
        "Popularity": 0,
        "BPR": 1,
        "LightGCN": 2,
        "DropoutNet": 3,
        "ContentProfile": 4,
        "CCFCRec": 5,
        "ALDI": 6,
    }
    out["__order"] = out["model"].map(order).fillna(999)
    out = out.sort_values(["__order", "model"]).drop(columns=["__order"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
