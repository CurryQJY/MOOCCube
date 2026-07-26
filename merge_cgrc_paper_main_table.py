"""Add the CGRC-paper row to the finalized 3-seed main-table summary.

This keeps the previous summary intact and writes a new CSV that can be passed
to ``export_main_table_3seed_docx.py --baseline-csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASE = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120/"
    "main_table_item_macro_summary.csv"
)
DEFAULT_CGRC = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_balanced_itemmacro_cgrc_paper_v1/main_table_item_macro_summary.csv"
)
DEFAULT_OUT = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/"
    "main_table_item_macro_summary.csv"
)

MODEL_ORDER = {
    "Popularity": 0,
    "BPR": 1,
    "LightGCN": 2,
    "DropoutNet": 3,
    "ContentProfile": 4,
    "CCFCRec": 5,
    "ALDI": 6,
    "CGRC-paper": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cgrc", type=Path, default=DEFAULT_CGRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"Missing base summary: {args.base}")
    if not args.cgrc.exists():
        raise FileNotFoundError(f"Missing CGRC-paper summary: {args.cgrc}")

    base = pd.read_csv(args.base)
    cgrc = pd.read_csv(args.cgrc)
    cgrc = cgrc[cgrc["model"].astype(str).eq("CGRC-paper")]
    if cgrc.empty:
        raise ValueError(f"No CGRC-paper row found in {args.cgrc}")
    if cgrc.shape[0] != 1:
        raise ValueError(f"Expected one CGRC-paper row in {args.cgrc}, found {cgrc.shape[0]}")

    out = base[~base["model"].astype(str).eq("CGRC-paper")].copy()
    out = pd.concat([out, cgrc], ignore_index=True, sort=False)
    out["__order"] = out["model"].map(MODEL_ORDER).fillna(999)
    out = out.sort_values(["__order", "model"]).drop(columns=["__order"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
