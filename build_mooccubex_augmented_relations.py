import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fast3_delta.course_artifacts import _read_relation_pairs


def dedupe_pairs_preserve_order(pairs):
    seen = set()
    out = []
    for course_id, concept_id in pairs:
        key = (str(course_id), str(concept_id))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def build_semantic_cluster_pairs(course_ids, labels_by_level):
    pairs = []
    for level in sorted(labels_by_level):
        labels = labels_by_level[level]
        if len(labels) != len(course_ids):
            raise ValueError(
                f"Label count mismatch for level={level}: "
                f"expected {len(course_ids)}, got {len(labels)}"
            )
        for course_id, label in zip(course_ids, labels):
            concept_id = f"SEM_CLUSTER_L{int(level)}_{int(label):05d}"
            pairs.append((str(course_id), concept_id))
    return pairs


def _normalize_np_rows(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _course_concept_sets(original_pairs, observed_courses):
    observed = set(str(x) for x in observed_courses)
    concepts = defaultdict(set)
    for course_id, concept_id in original_pairs:
        course_id = str(course_id)
        concept_id = str(concept_id)
        if course_id in observed and concept_id:
            concepts[course_id].add(concept_id)
    return concepts


def _allowed_propagation_concepts(course_to_concepts, min_df, max_course_frac):
    n_source = max(1, len(course_to_concepts))
    df = Counter()
    for concepts in course_to_concepts.values():
        for concept_id in concepts:
            df[str(concept_id)] += 1
    min_df = max(1, int(min_df))
    max_df = max(min_df, int(np.floor(float(max_course_frac) * n_source)))
    allowed = {
        concept_id
        for concept_id, count in df.items()
        if count >= min_df and count <= max_df
    }
    return allowed, df, max_df


def build_propagated_concept_pairs(
    course_ids,
    embeddings,
    original_pairs,
    top_m=8,
    min_similarity=0.60,
    min_concept_support=0.08,
    max_concepts_per_course=16,
    min_concept_df=2,
    max_concept_course_frac=0.25,
    only_missing=True,
):
    """Propagate real concepts from similar annotated courses to unannotated courses.

    The output keeps the existing unweighted relation format intentionally. Confidence
    is enforced by neighbor similarity, concept document-frequency filtering, and
    normalized support across the selected neighbors.
    """
    course_ids = [str(x) for x in course_ids]
    x = _normalize_np_rows(embeddings)
    if x.shape[0] != len(course_ids):
        raise ValueError(
            f"Embedding row count mismatch: embeddings={x.shape[0]}, courses={len(course_ids)}"
        )

    course_to_concepts = _course_concept_sets(original_pairs, course_ids)
    allowed, concept_df, max_df = _allowed_propagation_concepts(
        course_to_concepts,
        min_df=min_concept_df,
        max_course_frac=max_concept_course_frac,
    )
    source_indices = [
        idx
        for idx, course_id in enumerate(course_ids)
        if course_to_concepts.get(course_id)
    ]
    source_indices = [
        idx
        for idx in source_indices
        if course_to_concepts[course_ids[idx]] & allowed
    ]
    target_indices = [
        idx
        for idx, course_id in enumerate(course_ids)
        if (not only_missing) or (not course_to_concepts.get(course_id))
    ]

    stats = {
        "enabled": True,
        "source_courses": int(len(source_indices)),
        "target_courses": int(len(target_indices)),
        "allowed_concepts": int(len(allowed)),
        "concept_df_min": int(max(1, int(min_concept_df))),
        "concept_df_max": int(max_df),
        "max_concept_course_frac": float(max_concept_course_frac),
        "top_m": int(top_m),
        "min_similarity": float(min_similarity),
        "min_concept_support": float(min_concept_support),
        "max_concepts_per_course": int(max_concepts_per_course),
        "only_missing": bool(only_missing),
        "propagated_courses": 0,
        "propagated_pairs": 0,
        "mean_selected_neighbors": 0.0,
    }
    if not source_indices or not target_indices or not allowed:
        return [], stats

    source_x = x[np.asarray(source_indices, dtype=np.int64)]
    source_concepts = [
        sorted(course_to_concepts[course_ids[idx]] & allowed)
        for idx in source_indices
    ]
    top_m = max(1, int(top_m))
    max_concepts_per_course = max(1, int(max_concepts_per_course))
    min_similarity = float(min_similarity)
    min_concept_support = float(min_concept_support)

    pairs = []
    selected_neighbor_counts = []
    source_pos_by_idx = {int(src_idx): pos for pos, src_idx in enumerate(source_indices)}
    for target_idx in target_indices:
        sims = source_x @ x[target_idx]
        source_self_pos = source_pos_by_idx.get(int(target_idx))
        if source_self_pos is not None:
            sims[source_self_pos] = -np.inf
        keep = np.where(sims >= min_similarity)[0]
        if keep.size == 0:
            selected_neighbor_counts.append(0)
            continue
        order = keep[np.argsort(-sims[keep])[:top_m]]
        positive_sims = np.maximum(sims[order].astype(np.float64), 0.0)
        total_sim = float(positive_sims.sum())
        if total_sim <= 1e-12:
            selected_neighbor_counts.append(0)
            continue

        scores = defaultdict(float)
        hits = defaultdict(int)
        for pos, sim in zip(order, positive_sims):
            for concept_id in source_concepts[int(pos)]:
                scores[concept_id] += float(sim)
                hits[concept_id] += 1

        ranked = []
        for concept_id, score in scores.items():
            support = float(score) / total_sim
            if support >= min_concept_support:
                ranked.append((support, float(score), int(hits[concept_id]), str(concept_id)))
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        chosen = ranked[:max_concepts_per_course]
        if chosen:
            target_course = course_ids[target_idx]
            for _, _, _, concept_id in chosen:
                pairs.append((target_course, concept_id))
        selected_neighbor_counts.append(int(order.size))

    propagated_courses = len({course_id for course_id, _ in pairs})
    stats["propagated_courses"] = int(propagated_courses)
    stats["propagated_pairs"] = int(len(pairs))
    stats["mean_selected_neighbors"] = (
        float(np.mean(selected_neighbor_counts)) if selected_neighbor_counts else 0.0
    )
    stats["source_concepts_raw"] = int(len(concept_df))
    return pairs, stats


def _load_stream_df(data_dir):
    path = Path(data_dir) / "stream_data.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path)


def _load_content_embeddings(data_dir):
    path = Path(data_dir) / "content_emb.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _normalize_embeddings(emb):
    emb = torch.as_tensor(emb, dtype=torch.float32)
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()


def _fit_cluster_labels(x, levels, seed, batch_size):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for semantic relation augmentation.") from exc

    labels_by_level = {}
    n_samples = x.shape[0]
    for level in levels:
        k = max(2, min(int(level), n_samples))
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=int(seed),
            batch_size=min(int(batch_size), max(32, n_samples)),
            n_init=10,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        labels_by_level[k] = model.fit_predict(x).astype(int).tolist()
    return labels_by_level


def _write_pairs_tsv(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for course_id, concept_id in pairs:
            f.write(f"{course_id}\t{concept_id}\n")


def build_augmented_relations(
    data_dir,
    source_relation_dir,
    output_relation_dir,
    levels=(64, 128, 256),
    seed=2025,
    batch_size=1024,
    use_semantic_clusters=True,
    propagate_real_concepts=False,
    propagation_top_m=8,
    propagation_min_similarity=0.60,
    propagation_min_concept_support=0.08,
    propagation_max_concepts_per_course=16,
    propagation_min_concept_df=2,
    propagation_max_concept_course_frac=0.25,
    propagation_only_missing=True,
):
    data_dir = Path(data_dir)
    source_relation_dir = Path(source_relation_dir)
    output_relation_dir = Path(output_relation_dir)
    output_relation_dir.mkdir(parents=True, exist_ok=True)

    df = _load_stream_df(data_dir)
    idx_course = (
        df[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .sort_values("i_idx")
        .reset_index(drop=True)
    )
    item_indices = idx_course["i_idx"].astype(int).tolist()
    course_ids = idx_course["course_id"].astype(str).tolist()
    observed_courses = set(course_ids)

    content_emb = _load_content_embeddings(data_dir)
    max_i_idx = max(item_indices)
    if content_emb.shape[0] <= max_i_idx:
        raise ValueError(
            f"content_emb rows={content_emb.shape[0]} cannot cover max i_idx={max_i_idx}"
        )
    x = _normalize_embeddings(content_emb[item_indices])

    original_pairs_raw = _read_relation_pairs(str(source_relation_dir / "course-concept.json"))
    original_pairs = [
        (course_id, concept_id)
        for course_id, concept_id in original_pairs_raw
        if course_id in observed_courses and concept_id
    ]

    propagated_pairs = []
    propagation_stats = {"enabled": False}
    if propagate_real_concepts:
        propagated_pairs, propagation_stats = build_propagated_concept_pairs(
            course_ids,
            x,
            original_pairs,
            top_m=propagation_top_m,
            min_similarity=propagation_min_similarity,
            min_concept_support=propagation_min_concept_support,
            max_concepts_per_course=propagation_max_concepts_per_course,
            min_concept_df=propagation_min_concept_df,
            max_concept_course_frac=propagation_max_concept_course_frac,
            only_missing=propagation_only_missing,
        )

    labels_by_level = {}
    semantic_pairs = []
    if use_semantic_clusters and levels:
        labels_by_level = _fit_cluster_labels(x, levels=levels, seed=seed, batch_size=batch_size)
        semantic_pairs = build_semantic_cluster_pairs(course_ids, labels_by_level)

    augmented_pairs = dedupe_pairs_preserve_order(
        original_pairs + propagated_pairs + semantic_pairs
    )

    _write_pairs_tsv(output_relation_dir / "course-concept.json", augmented_pairs)
    _write_pairs_tsv(output_relation_dir / "concept-course.txt", [(b, a) for a, b in augmented_pairs])

    prereq_src = source_relation_dir / "prerequisite-dependency.json"
    if prereq_src.exists():
        shutil.copyfile(prereq_src, output_relation_dir / "prerequisite-dependency.json")

    cluster_rows = []
    for pos, (i_idx, course_id) in enumerate(zip(item_indices, course_ids)):
        row = {"i_idx": int(i_idx), "course_id": course_id}
        for level, labels in sorted(labels_by_level.items()):
            row[f"sem_cluster_l{level}"] = int(labels[pos])
        cluster_rows.append(row)
    pd.DataFrame(cluster_rows).to_csv(
        output_relation_dir / "semantic_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    original_courses = {course_id for course_id, _ in original_pairs}
    propagated_courses = {course_id for course_id, _ in propagated_pairs}
    semantic_courses = {course_id for course_id, _ in semantic_pairs}
    augmented_courses = {course_id for course_id, _ in augmented_pairs}
    manifest = {
        "data_dir": str(data_dir),
        "source_relation_dir": str(source_relation_dir),
        "output_relation_dir": str(output_relation_dir),
        "seed": int(seed),
        "semantic_cluster_levels": [int(k) for k in sorted(labels_by_level)],
        "observed_courses": int(len(observed_courses)),
        "original_pairs_raw": int(len(original_pairs_raw)),
        "original_pairs_observed": int(len(original_pairs)),
        "original_courses_observed": int(len(original_courses)),
        "propagated_pairs": int(len(propagated_pairs)),
        "propagated_courses": int(len(propagated_courses)),
        "propagation": propagation_stats,
        "semantic_pairs": int(len(semantic_pairs)),
        "augmented_pairs": int(len(augmented_pairs)),
        "original_course_coverage": len(original_courses) / max(1, len(observed_courses)),
        "propagated_course_coverage": (
            len(original_courses | propagated_courses)
            / max(1, len(observed_courses))
        ),
        "semantic_course_coverage": len(semantic_courses) / max(1, len(observed_courses)),
        "augmented_course_coverage": len(augmented_courses) / max(1, len(observed_courses)),
        "note": (
            "Augmentation uses precomputed content embeddings and course metadata only. "
            "No validation/test interactions are used."
        ),
    }
    with open(output_relation_dir / "augmentation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="processed_data_hin_x")
    parser.add_argument("--source-relation-dir", default="MOOCCubeX/relations")
    parser.add_argument("--output-relation-dir", default="MOOCCubeX/relations_aug")
    parser.add_argument("--levels", default="64,128,256")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--disable-semantic-clusters",
        action="store_true",
        help="Do not add SEM_CLUSTER pseudo concepts.",
    )
    parser.add_argument(
        "--propagate-real-concepts",
        action="store_true",
        help="Propagate original real concepts from similar annotated courses.",
    )
    parser.add_argument("--propagation-top-m", type=int, default=8)
    parser.add_argument("--propagation-min-similarity", type=float, default=0.60)
    parser.add_argument("--propagation-min-concept-support", type=float, default=0.08)
    parser.add_argument("--propagation-max-concepts-per-course", type=int, default=16)
    parser.add_argument("--propagation-min-concept-df", type=int, default=2)
    parser.add_argument("--propagation-max-concept-course-frac", type=float, default=0.25)
    parser.add_argument(
        "--propagate-all-courses",
        action="store_true",
        help="Also add propagated concepts to courses that already have original concepts.",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    manifest = build_augmented_relations(
        data_dir=args.data_dir,
        source_relation_dir=args.source_relation_dir,
        output_relation_dir=args.output_relation_dir,
        levels=levels,
        seed=args.seed,
        batch_size=args.batch_size,
        use_semantic_clusters=not args.disable_semantic_clusters,
        propagate_real_concepts=args.propagate_real_concepts,
        propagation_top_m=args.propagation_top_m,
        propagation_min_similarity=args.propagation_min_similarity,
        propagation_min_concept_support=args.propagation_min_concept_support,
        propagation_max_concepts_per_course=args.propagation_max_concepts_per_course,
        propagation_min_concept_df=args.propagation_min_concept_df,
        propagation_max_concept_course_frac=args.propagation_max_concept_course_frac,
        propagation_only_missing=not args.propagate_all_courses,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
