import json
import hashlib
import math
import os
import re
from collections import defaultdict

import torch


def _empty_course_stats(n_items):
    return {
        "items_with_concept": 0,
        "items_with_prereq": 0,
        "items_with_video": 0,
        "redundant_family_groups": 0,
        "hard_density": 0.0,
        "prereq_edges_kept": 0,
        "prereq_edges_raw": 0,
        "prereq_users": 0,
        "n_items": int(n_items),
    }


def _sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_signature(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    stat = os.stat(path)
    return {
        "path": path,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _course_mapping_digest(idx_to_course):
    payload = "\n".join(
        f"{idx}\t{course_id if course_id is not None else ''}"
        for idx, course_id in enumerate(idx_to_course)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _course_artifact_cache_dir(relation_dir):
    disabled = os.environ.get("USIM_COURSE_ARTIFACT_CACHE_DISABLE", "0").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    cache_dir = os.environ.get("USIM_COURSE_ARTIFACT_CACHE_DIR", "").strip()
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(relation_dir)), ".course_artifact_cache")
    return cache_dir


def _course_artifact_cache_key(
    idx_to_course,
    n_items,
    relation_dir,
    prereq_min_support,
    prereq_max_per_item,
    prereq_min_items,
    prereq_max_forward,
    concept_overlap_mode,
    prereq_graph_source,
    prereq_concept_score_thr,
    prereq_concept_min_hits,
    prereq_concept_file,
    prereq_hybrid_alpha,
    prereq_hybrid_strong_concept_thr,
    weighted_prereq_edges,
):
    relation_dir_abs = os.path.abspath(relation_dir)
    entity_dir = os.path.join(os.path.dirname(relation_dir_abs), "entities")
    payload = {
        "cache_version": 1,
        "n_items": int(n_items),
        "idx_to_course_sha256": _course_mapping_digest(idx_to_course),
        "relation_dir": relation_dir_abs,
        "files": {
            "course_concept": _file_signature(os.path.join(relation_dir_abs, "course-concept.json")),
            "prereq_concept": _file_signature(os.path.join(relation_dir_abs, prereq_concept_file)),
            "course_entity": _file_signature(os.path.join(entity_dir, "course.json")),
        },
        "params": {
            "prereq_min_support": int(prereq_min_support),
            "prereq_max_per_item": int(prereq_max_per_item),
            "prereq_min_items": int(prereq_min_items),
            "prereq_max_forward": int(prereq_max_forward),
            "concept_overlap_mode": str(concept_overlap_mode),
            "prereq_graph_source": str(prereq_graph_source),
            "prereq_concept_score_thr": float(prereq_concept_score_thr),
            "prereq_concept_min_hits": int(prereq_concept_min_hits),
            "prereq_concept_file": str(prereq_concept_file),
            "prereq_hybrid_alpha": float(prereq_hybrid_alpha),
            "prereq_hybrid_strong_concept_thr": float(prereq_hybrid_strong_concept_thr),
            "weighted_prereq_edges": bool(weighted_prereq_edges),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_course_artifact_cache(cache_path, cache_key):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"[COURSE-CACHE] Failed to load {cache_path}: {exc}")
        return None
    if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
        return None
    artifacts = payload.get("artifacts")
    stats = dict(payload.get("stats") or {})
    if not isinstance(artifacts, dict):
        return None
    stats["course_artifact_cache_status"] = "hit"
    stats["course_artifact_cache_path"] = cache_path
    print(f"[COURSE-CACHE] Loaded course artifacts from {cache_path}")
    return artifacts, stats


def _save_course_artifact_cache(cache_path, cache_key, artifacts, stats):
    if not cache_path:
        return None
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    payload = {
        "cache_key": cache_key,
        "artifacts": artifacts,
        "stats": stats,
    }
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        return None
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return str(exc)


def _read_relation_pairs(filepath):
    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                a, b = line.split("\t", 1)
            elif "," in line:
                a, b = line.split(",", 1)
                if a == "start_id" and b == "end_id":
                    continue
            else:
                continue
            pairs.append((a.strip(), b.strip()))
    return pairs


def _parse_subject_from_course_id(course_id):
    cid = str(course_id)
    m = re.search(r"\+([A-Za-z]+)\d+", cid)
    if m:
        return m.group(1).upper()
    m = re.search(r"course-v1:([^+]+)\+", cid)
    if m:
        return m.group(1).upper()
    return "UNK"


def _iter_entity_objects(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            return data if isinstance(data, list) else []
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def _extract_course_unit_ids(course_obj):
    unit_ids = []
    for unit_id in course_obj.get("video_order", []) or []:
        unit_id = str(unit_id).strip()
        if unit_id:
            unit_ids.append(unit_id)
    if unit_ids:
        return unit_ids

    for resource in course_obj.get("resource", []) or []:
        unit_id = str(resource.get("resource_id") or "").strip()
        if unit_id:
            unit_ids.append(unit_id)
    return unit_ids


def _normalize_course_family_key(course_id, core_id=None):
    for raw in [course_id, core_id]:
        if raw is None:
            continue
        cid = str(raw).strip()
        if not cid:
            continue
        if "+" in cid:
            prefix, suffix = cid.rsplit("+", 1)
            if re.fullmatch(r"(?i)(sp|20\d{2}_t\d+(?:_[a-z]+)?|_?20\d{2}_?|20\d{2})", suffix):
                return prefix
        return cid
    return None


def _build_behavior_prereq_candidates(df, prereq_min_support=30, prereq_max_forward=20):
    edge_support = defaultdict(int)
    user_seq_count = 0
    if {"u_idx", "i_idx", "timestamp"}.issubset(df.columns):
        seq_df = df[["u_idx", "i_idx", "timestamp"]].sort_values(["u_idx", "timestamp"])
        max_forward = max(1, int(prereq_max_forward))
        for _, group in seq_df.groupby("u_idx", sort=False):
            seq_raw = [int(x) for x in group["i_idx"].tolist()]
            if len(seq_raw) < 2:
                continue
            seq = []
            seen_local = set()
            for item in seq_raw:
                if item in seen_local:
                    continue
                seen_local.add(item)
                seq.append(item)
            if len(seq) < 2:
                continue
            user_seq_count += 1
            for p, a in enumerate(seq):
                end = min(len(seq), p + 1 + max_forward)
                for q in range(p + 1, end):
                    b = seq[q]
                    if a != b:
                        edge_support[(a, b)] += 1
    incoming = defaultdict(list)
    for (a, b), sup in edge_support.items():
        if sup >= int(prereq_min_support):
            incoming[int(b)].append((int(a), float(sup), int(sup)))
    stats = {
        "prereq_source": "behavior",
        "prereq_edges_raw": int(len(edge_support)),
        "prereq_users": int(user_seq_count),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_forward": int(prereq_max_forward),
    }
    return incoming, stats


def _build_concept_prereq_candidates(
    concept_sets,
    relation_dir="MOOCCube/relations",
    prereq_concept_file="prerequisite-dependency.json",
    prereq_concept_score_thr=0.10,
    prereq_concept_min_hits=1,
):
    prereq_path = os.path.join(relation_dir, prereq_concept_file)
    prereq_pairs = _read_relation_pairs(prereq_path)
    incoming_concepts = defaultdict(set)
    for prereq_concept, target_concept in prereq_pairs:
        if prereq_concept and target_concept and prereq_concept != target_concept:
            incoming_concepts[target_concept].add(prereq_concept)
    concept_required_sets = []
    for cset in concept_sets:
        required = set()
        for concept in cset:
            required.update(incoming_concepts.get(concept, ()))
        required.difference_update(cset)
        concept_required_sets.append(required)
    incoming = defaultdict(list)
    raw_edge_count = 0
    courses_with_required_concepts = 0
    n_items = len(concept_sets)
    min_hits = max(1, int(prereq_concept_min_hits))
    score_thr = max(0.0, float(prereq_concept_score_thr))
    for b in range(n_items):
        required = concept_required_sets[b]
        if not required:
            continue
        courses_with_required_concepts += 1
        denom = float(len(required))
        for a in range(n_items):
            if a == b or not concept_sets[a]:
                continue
            hits = len(concept_sets[a] & required)
            if hits < min_hits:
                continue
            score = hits / denom
            if score >= score_thr:
                incoming[b].append((a, float(score), int(hits)))
                raw_edge_count += 1
    stats = {
        "prereq_source": "concept",
        "prereq_edges_raw": int(raw_edge_count),
        "prereq_users": 0,
        "prereq_min_support": 0,
        "prereq_max_forward": 0,
        "prereq_concept_pairs": int(len(prereq_pairs)),
        "prereq_concept_score_thr": float(score_thr),
        "prereq_concept_min_hits": int(min_hits),
        "courses_with_required_concepts": int(courses_with_required_concepts),
        "prereq_concept_file": str(prereq_concept_file),
    }
    return incoming, stats


def _build_hybrid_prereq_candidates(
    df,
    concept_sets,
    relation_dir="MOOCCube/relations",
    prereq_min_support=30,
    prereq_max_forward=20,
    prereq_concept_file="prerequisite-dependency.json",
    prereq_concept_score_thr=0.20,
    prereq_concept_min_hits=2,
    hybrid_alpha=0.70,
    hybrid_strong_concept_thr=0.35,
):
    behavior_incoming, behavior_stats = _build_behavior_prereq_candidates(
        df,
        prereq_min_support=prereq_min_support,
        prereq_max_forward=prereq_max_forward,
    )
    concept_incoming, concept_stats = _build_concept_prereq_candidates(
        concept_sets,
        relation_dir=relation_dir,
        prereq_concept_file=prereq_concept_file,
        prereq_concept_score_thr=prereq_concept_score_thr,
        prereq_concept_min_hits=prereq_concept_min_hits,
    )

    alpha = min(1.0, max(0.0, float(hybrid_alpha)))
    strong_thr = max(float(prereq_concept_score_thr), float(hybrid_strong_concept_thr))
    incoming = defaultdict(list)
    raw_edge_count = 0

    for b, concept_edges in concept_incoming.items():
        behavior_edges = {
            int(src): float(score)
            for src, score, _ in behavior_incoming.get(b, [])
        }
        max_behavior = max(behavior_edges.values()) if behavior_edges else 0.0
        for src, concept_score, concept_hits in concept_edges:
            src = int(src)
            concept_score = float(concept_score)
            behavior_score = behavior_edges.get(src, 0.0)
            behavior_norm = behavior_score / max_behavior if max_behavior > 0.0 else 0.0
            if behavior_score <= 0.0 and concept_score < strong_thr:
                continue
            hybrid_score = alpha * concept_score + (1.0 - alpha) * behavior_norm
            incoming[int(b)].append((src, float(hybrid_score), int(concept_hits)))
            raw_edge_count += 1

    stats = {
        "prereq_source": "hybrid",
        "prereq_edges_raw": int(raw_edge_count),
        "prereq_users": int(behavior_stats.get("prereq_users", 0)),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_forward": int(prereq_max_forward),
        "prereq_concept_pairs": int(concept_stats.get("prereq_concept_pairs", 0)),
        "prereq_concept_score_thr": float(prereq_concept_score_thr),
        "prereq_concept_min_hits": int(prereq_concept_min_hits),
        "courses_with_required_concepts": int(concept_stats.get("courses_with_required_concepts", 0)),
        "prereq_concept_file": str(prereq_concept_file),
        "prereq_hybrid_alpha": float(alpha),
        "prereq_hybrid_strong_concept_thr": float(strong_thr),
        "behavior_prereq_edges_raw": int(behavior_stats.get("prereq_edges_raw", 0)),
        "concept_prereq_edges_raw": int(concept_stats.get("prereq_edges_raw", 0)),
    }
    return incoming, stats


def _build_item_concept_overlap(concept_sets, mode="plain"):
    mode = (mode or "plain").strip().lower()
    n_items = len(concept_sets)
    item_concept_overlap = torch.zeros((n_items, n_items), dtype=torch.float32)

    if mode == "plain":
        for i in range(n_items):
            c_i = concept_sets[i]
            denom = len(c_i)
            if denom < 1:
                continue
            for j in range(n_items):
                c_j = concept_sets[j]
                if not c_j:
                    continue
                inter = len(c_i & c_j)
                if inter > 0:
                    item_concept_overlap[i, j] = inter / float(denom)
        return item_concept_overlap

    if mode == "idf":
        concept_df = defaultdict(int)
        for cset in concept_sets:
            for concept in cset:
                concept_df[concept] += 1
        n_courses = max(1, n_items)
        concept_idf = {
            concept: math.log((n_courses + 1.0) / (df + 1.0)) + 1.0
            for concept, df in concept_df.items()
        }
        for i in range(n_items):
            c_i = concept_sets[i]
            if not c_i:
                continue
            denom = sum(concept_idf.get(c, 1.0) for c in c_i)
            if denom <= 0.0:
                continue
            for j in range(n_items):
                c_j = concept_sets[j]
                if not c_j:
                    continue
                inter = c_i & c_j
                if inter:
                    numer = sum(concept_idf.get(c, 1.0) for c in inter)
                    item_concept_overlap[i, j] = numer / denom
        return item_concept_overlap

    raise ValueError(f"Unsupported concept overlap mode: {mode}")


def build_course_artifacts(
    df,
    n_items,
    relation_dir="MOOCCube/relations",
    prereq_min_support=30,
    prereq_max_per_item=5,
    prereq_min_items=1,
    prereq_max_forward=20,
    concept_overlap_mode=None,
    prereq_graph_source=None,
    prereq_concept_score_thr=None,
    prereq_concept_min_hits=None,
    prereq_concept_file=None,
    prereq_hybrid_alpha=None,
    prereq_hybrid_strong_concept_thr=None,
):
    weighted_prereq_edges = os.environ.get("USIM_FB_PREREQ_WEIGHTED_EDGES", "0") == "1"
    concept_overlap_mode = (concept_overlap_mode or os.environ.get("USIM_CONCEPT_OVERLAP_MODE", "plain")).strip().lower()
    prereq_graph_source = (prereq_graph_source or os.environ.get("USIM_PREREQ_GRAPH_SOURCE", "behavior")).strip().lower()
    prereq_concept_score_thr = float(
        prereq_concept_score_thr if prereq_concept_score_thr is not None
        else os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", "0.10")
    )
    prereq_concept_min_hits = int(
        prereq_concept_min_hits if prereq_concept_min_hits is not None
        else os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", "1")
    )
    prereq_concept_file = prereq_concept_file or os.environ.get("USIM_PREREQ_CONCEPT_FILE", "prerequisite-dependency.json")
    prereq_hybrid_alpha = float(
        prereq_hybrid_alpha if prereq_hybrid_alpha is not None
        else os.environ.get("USIM_PREREQ_HYBRID_ALPHA", "0.70")
    )
    prereq_hybrid_strong_concept_thr = float(
        prereq_hybrid_strong_concept_thr if prereq_hybrid_strong_concept_thr is not None
        else os.environ.get("USIM_PREREQ_HYBRID_STRONG_CONCEPT_THR", "0.35")
    )
    idx_course = df[["i_idx", "course_id"]].drop_duplicates(subset=["i_idx"])
    idx_to_course = [None] * n_items
    for row in idx_course.itertuples(index=False):
        i_idx = int(row.i_idx)
        if 0 <= i_idx < n_items:
            idx_to_course[i_idx] = str(row.course_id)
    cache_dir = _course_artifact_cache_dir(relation_dir)
    cache_key = _course_artifact_cache_key(
        idx_to_course=idx_to_course,
        n_items=n_items,
        relation_dir=relation_dir,
        prereq_min_support=prereq_min_support,
        prereq_max_per_item=prereq_max_per_item,
        prereq_min_items=prereq_min_items,
        prereq_max_forward=prereq_max_forward,
        concept_overlap_mode=concept_overlap_mode,
        prereq_graph_source=prereq_graph_source,
        prereq_concept_score_thr=prereq_concept_score_thr,
        prereq_concept_min_hits=prereq_concept_min_hits,
        prereq_concept_file=prereq_concept_file,
        prereq_hybrid_alpha=prereq_hybrid_alpha,
        prereq_hybrid_strong_concept_thr=prereq_hybrid_strong_concept_thr,
        weighted_prereq_edges=weighted_prereq_edges,
    )
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt") if cache_dir else None
    cached = _load_course_artifact_cache(cache_path, cache_key)
    if cached is not None:
        return cached

    course_to_idx = {cid: idx for idx, cid in enumerate(idx_to_course) if cid is not None}
    concept_sets = [set() for _ in range(n_items)]
    video_sets = [set() for _ in range(n_items)]
    family_keys = [None] * n_items
    course_concept_file = os.path.join(relation_dir, "course-concept.json")
    for cid, concept in _read_relation_pairs(course_concept_file):
        idx = course_to_idx.get(cid)
        if idx is not None and concept:
            concept_sets[idx].add(concept)
    entity_dir = os.path.join(os.path.dirname(relation_dir), "entities")
    course_entity_file = os.path.join(entity_dir, "course.json")
    for course_obj in _iter_entity_objects(course_entity_file):
        cid = str(course_obj.get("id") or "").strip()
        idx = course_to_idx.get(cid)
        if idx is None:
            continue
        family_keys[idx] = _normalize_course_family_key(cid, course_obj.get("core_id"))
        video_sets[idx] = set(_extract_course_unit_ids(course_obj))
    item_prereq_item_mat = torch.zeros((n_items, n_items), dtype=torch.float32)
    item_prereq_item_cnt = torch.zeros(n_items, dtype=torch.float32)
    if prereq_graph_source == "behavior":
        incoming, prereq_stats = _build_behavior_prereq_candidates(
            df,
            prereq_min_support=prereq_min_support,
            prereq_max_forward=prereq_max_forward,
        )
    elif prereq_graph_source == "concept":
        incoming, prereq_stats = _build_concept_prereq_candidates(
            concept_sets,
            relation_dir=relation_dir,
            prereq_concept_file=prereq_concept_file,
            prereq_concept_score_thr=prereq_concept_score_thr,
            prereq_concept_min_hits=prereq_concept_min_hits,
        )
    elif prereq_graph_source == "hybrid":
        incoming, prereq_stats = _build_hybrid_prereq_candidates(
            df,
            concept_sets,
            relation_dir=relation_dir,
            prereq_min_support=prereq_min_support,
            prereq_max_forward=prereq_max_forward,
            prereq_concept_file=prereq_concept_file,
            prereq_concept_score_thr=prereq_concept_score_thr,
            prereq_concept_min_hits=prereq_concept_min_hits,
            hybrid_alpha=prereq_hybrid_alpha,
            hybrid_strong_concept_thr=prereq_hybrid_strong_concept_thr,
        )
    else:
        raise ValueError(f"Unsupported prereq_graph_source: {prereq_graph_source}")
    kept_edge_count = 0
    for b, src_list in incoming.items():
        src_list.sort(key=lambda x: (-float(x[1]), -int(x[2]), int(x[0])))
        kept = src_list[:max(1, int(prereq_max_per_item))]
        if len(kept) < int(prereq_min_items):
            continue
        idx_list = torch.tensor([src for src, _, _ in kept], dtype=torch.long)
        if weighted_prereq_edges:
            weight_tensor = torch.tensor([float(score) for _, score, _ in kept], dtype=torch.float32)
            max_weight = float(weight_tensor.max().item()) if weight_tensor.numel() > 0 else 0.0
            if max_weight > 0.0:
                weight_tensor = weight_tensor / max_weight
            item_prereq_item_mat[b, idx_list] = weight_tensor
            item_prereq_item_cnt[b] = float(weight_tensor.sum().item())
        else:
            item_prereq_item_mat[b, idx_list] = 1.0
            item_prereq_item_cnt[b] = float(len(kept))
        kept_edge_count += len(kept)
    item_concept_overlap = _build_item_concept_overlap(concept_sets, mode=concept_overlap_mode)
    item_video_contain = torch.zeros((n_items, n_items), dtype=torch.float32)
    item_same_family = torch.zeros((n_items, n_items), dtype=torch.bool)
    for i in range(n_items):
        v_i = video_sets[i]
        for j in range(n_items):
            if v_i and video_sets[j]:
                inter_video = len(v_i & video_sets[j])
                if inter_video > 0:
                    item_video_contain[i, j] = inter_video / float(len(v_i))
            if family_keys[i] and family_keys[i] == family_keys[j]:
                item_same_family[i, j] = True
    subjects = [_parse_subject_from_course_id(cid) if cid is not None else "UNK" for cid in idx_to_course]
    item_hard_adj = torch.zeros((n_items, n_items), dtype=torch.bool)
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            same_subject = subjects[i] != "UNK" and subjects[i] == subjects[j]
            same_concept = item_concept_overlap[i, j] > 0
            if same_subject or same_concept:
                item_hard_adj[i, j] = True
    items_with_concept = int(sum(1 for c in concept_sets if len(c) > 0))
    items_with_prereq = int((item_prereq_item_cnt > 0).sum().item())
    items_with_video = int(sum(1 for vids in video_sets if len(vids) > 0))
    family_group_sizes = defaultdict(int)
    for key in family_keys:
        if key:
            family_group_sizes[key] += 1
    hard_density = float(item_hard_adj.float().mean().item())
    stats = {
        "prereq_source": prereq_graph_source,
        "items_with_concept": items_with_concept,
        "items_with_prereq": items_with_prereq,
        "items_with_video": items_with_video,
        "redundant_family_groups": int(sum(1 for v in family_group_sizes.values() if v > 1)),
        "hard_density": hard_density,
        "prereq_edges_kept": int(kept_edge_count),
        "prereq_edges_raw": int(prereq_stats.get("prereq_edges_raw", 0)),
        "prereq_users": int(prereq_stats.get("prereq_users", 0)),
        "prereq_min_support": int(prereq_min_support),
        "prereq_max_per_item": int(prereq_max_per_item),
        "prereq_max_forward": int(prereq_max_forward),
        "concept_overlap_mode": concept_overlap_mode,
        "prereq_concept_pairs": int(prereq_stats.get("prereq_concept_pairs", 0)),
        "prereq_concept_score_thr": float(prereq_stats.get("prereq_concept_score_thr", prereq_concept_score_thr)),
        "prereq_concept_min_hits": int(prereq_stats.get("prereq_concept_min_hits", prereq_concept_min_hits)),
        "courses_with_required_concepts": int(prereq_stats.get("courses_with_required_concepts", 0)),
        "prereq_concept_file": str(prereq_stats.get("prereq_concept_file", prereq_concept_file)),
        "prereq_hybrid_alpha": float(prereq_stats.get("prereq_hybrid_alpha", prereq_hybrid_alpha)),
        "prereq_hybrid_strong_concept_thr": float(
            prereq_stats.get("prereq_hybrid_strong_concept_thr", prereq_hybrid_strong_concept_thr)
        ),
        "behavior_prereq_edges_raw": int(prereq_stats.get("behavior_prereq_edges_raw", 0)),
        "concept_prereq_edges_raw": int(prereq_stats.get("concept_prereq_edges_raw", 0)),
        "prereq_weighted_edges": weighted_prereq_edges,
        "course_artifact_cache_status": "miss" if cache_path else "disabled",
    }
    if cache_path:
        stats["course_artifact_cache_path"] = cache_path
    artifacts = {
        "item_hard_adj": item_hard_adj,
        "item_prereq_item_mat": item_prereq_item_mat,
        "item_prereq_item_cnt": item_prereq_item_cnt,
        "item_concept_overlap": item_concept_overlap,
        "item_video_contain": item_video_contain,
        "item_same_family": item_same_family,
    }
    cache_error = _save_course_artifact_cache(cache_path, cache_key, artifacts, stats)
    if cache_error:
        stats["course_artifact_cache_error"] = cache_error
        print(f"[COURSE-CACHE] Failed to save {cache_path}: {cache_error}")
    elif cache_path:
        print(f"[COURSE-CACHE] Saved course artifacts to {cache_path}")
    return artifacts, stats


