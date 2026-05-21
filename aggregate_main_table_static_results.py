import argparse
import json
import re
from pathlib import Path

import pandas as pd


RESULT_FILES = [
    "popularity_static_result.json",
    "bpr_static_result.json",
    "lightgcn_static_result.json",
    "lightgcl_static_result.json",
    "drop_static_result.json",
    "dropoutnet_official_static_result.json",
    "gar_static_result.json",
    "content_profile_static_result.json",
    "aldi_static_result.json",
    "aldi_official_static_result.json",
    "ccfcrec_static_result.json",
    "cgrc_static_result.json",
    "cgrc_paper_static_result.json",
    "fast3_static_result.json",
]

MODEL_ORDER = [
    "Popularity",
    "BPR",
    "LightGCN",
    "DropoutNet",
    "GAR",
    "ContentProfile",
    "CGRC",
    "CGRC-paper",
    "CCFCRec",
    "ALDI",
    "LightGCL",
    "FAST3",
]

DISPLAY_ALIASES = {
    "ALDI (official-source)": "ALDI",
    "ALDI (official-adapted)": "ALDI",
    "LightGCL (official-adapted)": "LightGCL",
    "CCFCRec (official-adapted)": "CCFCRec",
}

METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]


def parse_seed(path: Path):
    match = re.search(r"seed[_-]?(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def metric_value(obj, section, metric):
    block = obj.get(section)
    if isinstance(block, dict):
        return block.get(metric)
    return obj.get(f"{section}_{metric}")


def display_name(obj, path):
    name = obj.get("model_display") or obj.get("model") or path.stem
    return DISPLAY_ALIASES.get(name, name)


def epoch_tag(obj):
    tag = obj.get("epoch_tag")
    if tag:
        return tag
    teacher = obj.get("teacher_epochs")
    student = obj.get("student_epochs")
    if teacher is not None and student is not None:
        return f"teacher{teacher}_student{student}"
    return None


def collect_rows(root: Path, split_glob: str, result_subdir: str, metric_mode: str):
    if metric_mode == "item_macro":
        cold_section = "full_cold_item_macro"
        hot_section = "full_hot_item_macro"
        cold_count = "count_full_cold_item_macro"
        hot_count = "count_full_hot_item_macro"
    elif metric_mode == "interaction":
        cold_section = "full_cold"
        hot_section = "full_hot"
        cold_count = "count_full_cold"
        hot_count = "count_full_hot"
    else:
        raise ValueError("metric_mode must be item_macro or interaction")

    rows = []
    for split_dir in sorted(root.glob(split_glob)):
        if not split_dir.is_dir():
            continue
        seed = parse_seed(split_dir)
        result_dir = split_dir / result_subdir
        if not result_dir.exists():
            continue
        for filename in RESULT_FILES:
            path = result_dir / filename
            if not path.exists():
                continue
            obj = load_json(path)
            row = {
                "seed": seed,
                "split": split_dir.name,
                "result_subdir": result_subdir,
                "file": filename,
                "model": display_name(obj, path),
                "protocol": obj.get("protocol"),
                "best_epoch": obj.get("best_epoch"),
                "best_metric": obj.get("best_metric"),
                "teacher_epochs": obj.get("teacher_epochs"),
                "student_epochs": obj.get("student_epochs"),
                "epoch_tag": epoch_tag(obj),
                "count_cold": obj.get(cold_count),
                "count_hot": obj.get(hot_count),
            }
            for metric in METRICS:
                suffix = metric.replace("@", "")
                row[f"cold_{suffix}"] = metric_value(obj, cold_section, metric)
                row[f"hot_{suffix}"] = metric_value(obj, hot_section, metric)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame):
    metric_cols = [col for col in detail.columns if re.match(r"^(cold|hot)_[RN]\d+$", col)]
    rows = []
    for model, group in detail.groupby("model", dropna=False):
        out = {"model": model, "runs": int(len(group))}
        seeds = [int(x) for x in sorted(group["seed"].dropna().unique())]
        out["seeds"] = ",".join(str(x) for x in seeds)
        for col in ["teacher_epochs", "student_epochs", "epoch_tag"]:
            if col in group.columns:
                vals = group[col].dropna().astype(str).unique().tolist()
                out[col] = ",".join(vals)
        out["mean_best_epoch"] = pd.to_numeric(group["best_epoch"], errors="coerce").mean()
        out["count_cold_mean"] = pd.to_numeric(group["count_cold"], errors="coerce").mean()
        out["count_hot_mean"] = pd.to_numeric(group["count_hot"], errors="coerce").mean()
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            out[f"{col}_mean"] = float(values.mean()) if len(values) else None
            out[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    summary = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    summary["__order"] = summary["model"].map(order).fillna(999)
    return summary.sort_values(["__order", "model"]).drop(columns=["__order"])


def write_paper_table(summary: pd.DataFrame, out_path: Path):
    cols = [
        ("cold_R10_mean", "Cold R@10"),
        ("cold_N10_mean", "Cold N@10"),
        ("hot_R10_mean", "Hot R@10"),
        ("hot_N10_mean", "Hot N@10"),
    ]
    out = summary[["model"] + [src for src, _ in cols]].copy()
    out = out.rename(columns={"model": "Model", **{src: label for src, label in cols}})
    out.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/content_delta_pop5/static_item_cold_balanced_itemmacro_v1")
    parser.add_argument("--split-glob", default="strict_item_cold_balanced_thr1_seed_*")
    parser.add_argument("--result-subdir", default="main_table_balanced_itemmacro_v1")
    parser.add_argument("--metric-mode", choices=["item_macro", "interaction"], default="item_macro")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / f"main_table_{args.metric_mode}_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = collect_rows(root, args.split_glob, args.result_subdir, args.metric_mode)
    if detail.empty:
        raise SystemExit(f"No result JSON files found under {root}/*/{args.result_subdir}")

    detail_path = out_dir / f"main_table_{args.metric_mode}_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = summarize(detail)
    summary_path = out_dir / f"main_table_{args.metric_mode}_summary.csv"
    summary.to_csv(summary_path, index=False)

    paper_path = out_dir / f"main_table_{args.metric_mode}_paper_narrow.csv"
    write_paper_table(summary, paper_path)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {paper_path}")


if __name__ == "__main__":
    main()
