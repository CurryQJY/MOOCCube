import json
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fast3_delta.course_artifacts import _read_relation_pairs


DATA_DIR = Path("processed_data_hin_x")
SPLIT_DIR = Path(
    "outputs/mooccubex/course_ckpt_v1/full_e15/"
    "strict_item_cold_balanced_thr1_seed_2025"
)
OUT_DIR = Path("outputs/course_signal_diagnosis")
DEFAULT_THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.12]
DEFAULT_RELATION_CASES = [
    ("MOOCCubeX", "MOOCCubeX/relations"),
    ("MOOCCubeX-Aug", "MOOCCubeX/relations_aug"),
]


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _parse_relation_cases(case_args):
    if not case_args:
        return list(DEFAULT_RELATION_CASES)
    cases = []
    for raw in case_args:
        raw = str(raw)
        if "=" not in raw:
            raise ValueError(
                f"Invalid relation case '{raw}'. Expected format: Label=path/to/relations"
            )
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(
                f"Invalid relation case '{raw}'. Expected non-empty label and path."
            )
        cases.append((label, path))
    return cases


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _load_concept_sets(df, relation_dir):
    idx_course = (
        df[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .sort_values("i_idx")
        .reset_index(drop=True)
    )
    n_items = int(df["i_idx"].max()) + 1
    course_to_idx = {
        str(row.course_id): int(row.i_idx)
        for row in idx_course.itertuples(index=False)
    }
    concept_sets = [set() for _ in range(n_items)]
    for course_id, concept_id in _read_relation_pairs(str(Path(relation_dir) / "course-concept.json")):
        idx = course_to_idx.get(str(course_id))
        if idx is not None and concept_id:
            concept_sets[idx].add(str(concept_id))
    return concept_sets


def _build_train_seen(train_df):
    seen = defaultdict(list)
    for row in train_df[["u_idx", "i_idx"]].itertuples(index=False):
        seen[int(row.u_idx)].append(int(row.i_idx))
    return {u: sorted(set(items)) for u, items in seen.items()}


def _directed_overlap(target_concepts, seen_concepts):
    if not target_concepts or not seen_concepts:
        return 0.0
    return len(target_concepts & seen_concepts) / float(len(target_concepts))


def _threshold_activation_stats(values, by_item, thresholds):
    item_means = [sum(v) / len(v) for v in by_item.values() if v]
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        row_active = sum(1 for value in values if value >= threshold)
        item_active = sum(1 for value in item_means if value >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "row_active": int(row_active),
                "row_active_ratio_ge_threshold": _safe_ratio(row_active, len(values)),
                "item_active": int(item_active),
                "item_active_ratio_ge_threshold": _safe_ratio(item_active, len(item_means)),
            }
        )
    return rows


def _quantile(values_sorted, q):
    if not values_sorted:
        return 0.0
    idx = min(len(values_sorted) - 1, max(0, int(q * (len(values_sorted) - 1))))
    return values_sorted[idx]


def _compute_case(label, relation_dir, df, train_df, test_df, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    train_items = set(int(x) for x in train_df["i_idx"].unique())
    test_cold_df = test_df[~test_df["i_idx"].astype(int).isin(train_items)][["u_idx", "i_idx"]]
    concept_sets = _load_concept_sets(df, relation_dir)
    seen = _build_train_seen(train_df)

    values = []
    by_item = defaultdict(list)
    for row in test_cold_df.itertuples(index=False):
        u_idx = int(row.u_idx)
        i_idx = int(row.i_idx)
        target = concept_sets[i_idx]
        seen_items = seen.get(u_idx, [])
        if not target or not seen_items:
            val = 0.0
        else:
            total = 0.0
            for seen_i in seen_items:
                total += _directed_overlap(target, concept_sets[seen_i])
            val = total / float(len(seen_items))
        values.append(val)
        by_item[i_idx].append(val)

    item_means = [sum(v) / len(v) for v in by_item.values() if v]
    values_sorted = sorted(values)
    base = {
        "label": label,
        "relation_dir": str(relation_dir),
        "test_cold_rows": int(len(values)),
        "test_cold_items": int(len(by_item)),
        "row_mean_concept_match": sum(values) / max(1, len(values)),
        "row_p50_concept_match": _quantile(values_sorted, 0.50),
        "row_p90_concept_match": _quantile(values_sorted, 0.90),
        "row_p95_concept_match": _quantile(values_sorted, 0.95),
        "row_p99_concept_match": _quantile(values_sorted, 0.99),
        "item_mean_concept_match": sum(item_means) / max(1, len(item_means)),
    }
    rows = []
    for stats in _threshold_activation_stats(values, by_item, thresholds):
        row = dict(base)
        row.update(stats)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Relation case in Label=relation_dir format. Can be repeated.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated concept_match thresholds to diagnose.",
    )
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    cases = _parse_relation_cases(args.case)

    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    df = _load_pickle(data_dir / "stream_data.pkl")
    train_df = _load_pickle(split_dir / "static_train.pkl")
    test_df = _load_pickle(split_dir / "static_test.pkl")
    rows = []
    for label, relation_dir in cases:
        rows.extend(_compute_case(label, relation_dir, df, train_df, test_df, thresholds))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mooccubex_effective_concept_match.json"
    out_csv = out_dir / "mooccubex_effective_concept_match.csv"
    out_wide_csv = out_dir / "mooccubex_effective_concept_match_wide.csv"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    wide_df = result_df.pivot(
        index="label",
        columns="threshold",
        values=["row_active_ratio_ge_threshold", "item_active_ratio_ge_threshold"],
    )
    wide_df.columns = [f"{metric}@{threshold:g}" for metric, threshold in wide_df.columns]
    wide_df.reset_index().to_csv(out_wide_csv, index=False, encoding="utf-8-sig")
    print(result_df.to_string(index=False))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_wide_csv}")


if __name__ == "__main__":
    main()
