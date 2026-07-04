import argparse
from pathlib import Path

import pandas as pd


RUNS = [
    {
        "name": "masktrue_no_sage_seed2025",
        "root": "outputs/content_delta_pop5/fn_mask_ab/aligned_oldcfg_mask_e60_3seed/strict_item_cold_balanced_thr1_seed_2025",
    },
    {
        "name": "old_pseudo_e60_seed2025",
        "root": "outputs/content_delta_pop5/pseudo_cold_itemmacro_v1/old_main_plus_pseudo_e60/strict_item_cold_balanced_thr1_seed_2025",
    },
    {
        "name": "sage_tail0p002_seed2025",
        "root": "outputs/content_delta_pop5/sage_lite_v1/S5_tailratio_0p002_e60_resume_from_s4e8/strict_item_cold_balanced_thr1_seed_2025",
    },
    {
        "name": "p0_pseudo_only_seed2025",
        "root": "outputs/content_delta_pop5/pseudo_cold_sage_v1/P0_pseudo_only_e60_seeds_2025/strict_item_cold_balanced_thr1_seed_2025",
    },
    {
        "name": "p1_pseudo_sage_seed2025",
        "root": "outputs/content_delta_pop5/pseudo_cold_sage_v1/P1_pseudo_sage_e60_seeds_2025/strict_item_cold_balanced_thr1_seed_2025",
    },
    {
        "name": "p2_pseudo_sage_twoexpert_seed2025",
        "root": "outputs/content_delta_pop5/pseudo_cold_sage_v1/P2_pseudo_sage_twoexpert_e60_seeds_2025/strict_item_cold_balanced_thr1_seed_2025",
    },
]


METRICS = [
    "full_cold_item_macro_r10",
    "full_cold_item_macro_n10",
    "full_hot_item_macro_r10",
    "full_hot_item_macro_n10",
    "full_cold_item_macro_r20",
    "full_cold_item_macro_n20",
    "full_hot_item_macro_r20",
    "full_hot_item_macro_n20",
]


def read_manifest_flags(path: Path) -> dict:
    manifest = path / "static_protocol_manifest.json"
    if not manifest.exists():
        return {}
    try:
        import json

        cfg = json.loads(manifest.read_text(encoding="utf-8")).get("model_config", {})
    except Exception:
        return {}
    keys = [
        "use_pseudo_cold_train",
        "pseudo_cold_mode",
        "pseudo_cold_ratio",
        "use_sage_lite",
        "sage_only_cold_or_tail",
        "sage_tail_pop_ratio",
        "sage_use_two_expert",
        "mask_known_pos_neg",
        "mask_same_item_neg",
        "n_epochs",
    ]
    return {key: cfg.get(key) for key in keys}


def read_run(repo: Path, spec: dict) -> dict:
    root = repo / spec["root"]
    final_csv = root / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    row = {
        "run": spec["name"],
        "root": spec["root"],
        "status": "missing",
    }
    row.update(read_manifest_flags(root))
    if not final_csv.exists():
        return row
    df = pd.read_csv(final_csv)
    if df.empty:
        row["status"] = "empty"
        return row
    vals = df.iloc[0].to_dict()
    row["status"] = "complete"
    for metric in METRICS:
        row[metric] = vals.get(metric)
    return row


def add_deltas(df: pd.DataFrame, baseline_run: str) -> pd.DataFrame:
    if baseline_run not in set(df["run"]):
        return df
    base = df[df["run"] == baseline_run].iloc[0]
    for metric in METRICS:
        if metric in df.columns and pd.notna(base.get(metric)):
            df[f"{metric}_imp_vs_{baseline_run}"] = pd.to_numeric(df[metric], errors="coerce") - float(base[metric])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--out",
        default="outputs/content_delta_pop5/pseudo_cold_sage_v1/mooccube_pseudocold_sage_comparison_seed2025.csv",
    )
    parser.add_argument("--baseline", default="sage_tail0p002_seed2025")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    rows = [read_run(repo, spec) for spec in RUNS]
    df = pd.DataFrame(rows)
    df = add_deltas(df, args.baseline)
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df[["run", "status"] + [m for m in METRICS[:4] if m in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
