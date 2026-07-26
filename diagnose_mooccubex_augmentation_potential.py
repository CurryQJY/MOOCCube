import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from fast3_delta.course_artifacts import _read_relation_pairs


DATA_DIR = Path("processed_data_hin_x")
MOOCX_DIR = Path("MOOCCubeX")
SPLIT_DIR = Path(
    "outputs/mooccubex/course_ckpt_v1/full_e15/"
    "strict_item_cold_balanced_thr1_seed_2025"
)
OUT_DIR = Path("outputs/course_signal_diagnosis")


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _clean_text(text):
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_real_prereq_text(text):
    text = _clean_text(text)
    if not text:
        return False
    norm = re.sub(r"[\s,，.。;；:：、/\\()（）\\[\\]【】\"'<>《》]+", "", text).lower()
    none_markers = {
        "无",
        "暂无",
        "无要求",
        "没有",
        "无先修",
        "无需",
        "不需要",
        "不限",
        "无特殊要求",
        "none",
        "null",
        "no",
        "nothing",
        "noprerequisites",
        "noprerequisite",
        "noprerequisitesarerequired",
        "noprerequisiteisrequired",
        "norequirement",
        "norequirements",
    }
    if norm in none_markers:
        return False
    if norm.startswith("无") and len(norm) <= 6:
        return False
    if norm.startswith("noprerequisite") and len(norm) <= 40:
        return False
    return True


def _load_course_meta(path):
    meta = {}
    bad = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            cid = str(row.get("id") or "").strip()
            if cid:
                meta[cid] = row
    return meta, bad


def _resource_title_terms(row, max_terms=30):
    terms = []
    for res in row.get("resource", []) or []:
        for title in res.get("titles", []) or []:
            title = _clean_text(title)
            if title and title.lower() not in {"default", "none", "null"}:
                terms.append(f"TITLE::{title[:80]}")
                if len(terms) >= max_terms:
                    return terms
    return terms


def _course_concept_sets(observed_course_ids, original_pairs, course_meta, mode):
    original = defaultdict(set)
    for cid, concept in original_pairs:
        if cid in observed_course_ids and concept:
            original[cid].add(f"KG::{concept}")

    concept_sets = {}
    for cid in observed_course_ids:
        row = course_meta.get(cid, {})
        concepts = set()
        if "kg" in mode:
            concepts.update(original.get(cid, set()))
        if "field" in mode:
            for field in row.get("field", []) or []:
                field = _clean_text(field)
                if field:
                    concepts.add(f"FIELD::{field}")
        if "title" in mode:
            concepts.update(_resource_title_terms(row))
        concept_sets[cid] = concepts
    return concept_sets


def _coverage_for_subset(prefix, course_ids, course_meta, original_relation_courses):
    n = len(course_ids)
    with_meta = 0
    with_original_concept = 0
    with_field = 0
    with_prereq_text = 0
    with_resource = 0
    with_resource_title = 0
    with_about = 0
    for cid in course_ids:
        row = course_meta.get(cid)
        if row:
            with_meta += 1
        if cid in original_relation_courses:
            with_original_concept += 1
        fields = row.get("field", []) if row else []
        if any(_clean_text(x) for x in fields or []):
            with_field += 1
        prereq = row.get("prerequisites", "") if row else ""
        if _has_real_prereq_text(prereq):
            with_prereq_text += 1
        resources = row.get("resource", []) if row else []
        if resources:
            with_resource += 1
        if _resource_title_terms(row or {}, max_terms=1):
            with_resource_title += 1
        if _clean_text(row.get("about", "") if row else ""):
            with_about += 1
    return {
        f"{prefix}_courses": n,
        f"{prefix}_with_meta": with_meta,
        f"{prefix}_with_meta_ratio": _safe_ratio(with_meta, n),
        f"{prefix}_with_original_concept": with_original_concept,
        f"{prefix}_with_original_concept_ratio": _safe_ratio(with_original_concept, n),
        f"{prefix}_with_field": with_field,
        f"{prefix}_with_field_ratio": _safe_ratio(with_field, n),
        f"{prefix}_with_prereq_text": with_prereq_text,
        f"{prefix}_with_prereq_text_ratio": _safe_ratio(with_prereq_text, n),
        f"{prefix}_with_resource": with_resource,
        f"{prefix}_with_resource_ratio": _safe_ratio(with_resource, n),
        f"{prefix}_with_resource_title": with_resource_title,
        f"{prefix}_with_resource_title_ratio": _safe_ratio(with_resource_title, n),
        f"{prefix}_with_about": with_about,
        f"{prefix}_with_about_ratio": _safe_ratio(with_about, n),
    }


def _concept_graph_stats(label, concept_sets, subset_course_ids):
    n_items = len(concept_sets)
    df = Counter()
    for concepts in concept_sets.values():
        for c in concepts:
            df[c] += 1

    nonempty = [cid for cid, concepts in concept_sets.items() if concepts]
    avg_concepts = sum(len(v) for v in concept_sets.values()) / max(1, n_items)
    max_df = max(df.values()) if df else 0

    # Average directed overlap used by the model: |C_i cap C_j| / |C_i|.
    overlap_sum = 0.0
    pair_overlap_sum = 0.0
    active = 0
    for cid in subset_course_ids:
        concepts = concept_sets.get(cid, set())
        if not concepts:
            continue
        active += 1
        overlap_sum += sum(df[c] for c in concepts) / (len(concepts) * max(1, n_items))
        covered = set()
        # This union is fine for diagnostics; subsets here are small.
        for c in concepts:
            # Avoid materializing inverse lists; approximate pair overlap by
            # inclusion-exclusion upper bound when many generic concepts exist.
            pass
        pair_overlap_sum += min(1.0, sum(df[c] for c in concepts) / max(1, n_items))

    return {
        f"{label}_unique_concepts": len(df),
        f"{label}_courses_with_concepts": len(nonempty),
        f"{label}_courses_with_concepts_ratio": _safe_ratio(len(nonempty), n_items),
        f"{label}_avg_concepts_per_course": avg_concepts,
        f"{label}_max_concept_df": int(max_df),
        f"{label}_max_concept_df_ratio": _safe_ratio(max_df, n_items),
        f"{label}_subset_active_courses": active,
        f"{label}_subset_avg_directed_overlap_to_all": _safe_ratio(overlap_sum, active),
        f"{label}_subset_pair_overlap_upper": _safe_ratio(pair_overlap_sum, active),
    }


def main():
    df = _load_pickle(DATA_DIR / "stream_data.pkl")
    train_df = _load_pickle(SPLIT_DIR / "static_train.pkl")
    test_df = _load_pickle(SPLIT_DIR / "static_test.pkl")
    course_meta, bad_meta = _load_course_meta(MOOCX_DIR / "entities" / "course.json")
    original_pairs = _read_relation_pairs(str(MOOCX_DIR / "relations" / "course-concept.json"))
    prereq_pairs = _read_relation_pairs(str(MOOCX_DIR / "relations" / "prerequisite-dependency.json"))

    observed_courses = set(str(x) for x in df["course_id"].unique())
    i_to_course = (
        df[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .set_index("i_idx")["course_id"]
        .astype(str)
        .to_dict()
    )
    train_items = set(int(x) for x in train_df["i_idx"].unique())
    test_items = set(int(x) for x in test_df["i_idx"].unique())
    test_cold_items = sorted(test_items - train_items)
    test_cold_courses = [i_to_course[i] for i in test_cold_items if i in i_to_course]
    original_relation_courses = set(cid for cid, _ in original_pairs)

    output = {
        "dataset": "MOOCCubeX",
        "observed_courses": len(observed_courses),
        "course_meta_rows": len(course_meta),
        "course_meta_bad_json_rows": bad_meta,
        "original_course_concept_pairs": len(original_pairs),
        "original_course_concept_courses": len(original_relation_courses),
        "original_prereq_concept_pairs": len(prereq_pairs),
        "test_cold_courses": len(test_cold_courses),
    }
    output.update(_coverage_for_subset("observed", sorted(observed_courses), course_meta, original_relation_courses))
    output.update(_coverage_for_subset("test_cold", test_cold_courses, course_meta, original_relation_courses))

    modes = {
        "kg_only": ("kg",),
        "field_only": ("field",),
        "kg_plus_field": ("kg", "field"),
        "kg_plus_field_title": ("kg", "field", "title"),
    }
    graph_rows = []
    for label, mode in modes.items():
        sets = _course_concept_sets(observed_courses, original_pairs, course_meta, mode)
        stats = _concept_graph_stats(label, sets, test_cold_courses)
        output.update(stats)
        graph_rows.append({"mode": label, **stats})

    # Field distribution helps detect whether field-only concepts are too coarse.
    field_counter = Counter()
    for cid in observed_courses:
        for field in course_meta.get(cid, {}).get("field", []) or []:
            field = _clean_text(field)
            if field:
                field_counter[field] += 1
    field_rows = [
        {"field": field, "courses": count, "course_ratio": _safe_ratio(count, len(observed_courses))}
        for field, count in field_counter.most_common(30)
    ]
    title_counter = Counter()
    for cid in observed_courses:
        for title in set(_resource_title_terms(course_meta.get(cid, {}), max_terms=200)):
            title_counter[title.replace("TITLE::", "", 1)] += 1
    title_rows = [
        {"title_term": title, "courses": count, "course_ratio": _safe_ratio(count, len(observed_courses))}
        for title, count in title_counter.most_common(50)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "mooccubex_augmentation_potential.json"
    out_csv = OUT_DIR / "mooccubex_augmentation_potential_summary.csv"
    out_fields = OUT_DIR / "mooccubex_field_distribution.csv"
    out_titles = OUT_DIR / "mooccubex_resource_title_distribution.csv"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([output]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(field_rows).to_csv(out_fields, index=False, encoding="utf-8-sig")
    pd.DataFrame(title_rows).to_csv(out_titles, index=False, encoding="utf-8-sig")

    summary_cols = [
        "observed_courses",
        "test_cold_courses",
        "observed_with_original_concept_ratio",
        "test_cold_with_original_concept_ratio",
        "observed_with_field_ratio",
        "test_cold_with_field_ratio",
        "observed_with_prereq_text_ratio",
        "test_cold_with_prereq_text_ratio",
        "kg_only_subset_avg_directed_overlap_to_all",
        "field_only_subset_avg_directed_overlap_to_all",
        "kg_plus_field_subset_avg_directed_overlap_to_all",
        "kg_plus_field_title_subset_avg_directed_overlap_to_all",
        "field_only_max_concept_df_ratio",
    ]
    print(pd.DataFrame([output])[summary_cols].to_string(index=False))
    print("\nTop fields:")
    print(pd.DataFrame(field_rows).head(12).to_string(index=False))
    print("\nTop resource title terms:")
    print(pd.DataFrame(title_rows).head(12).to_string(index=False))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_fields}")
    print(f"Saved: {out_titles}")


if __name__ == "__main__":
    main()
