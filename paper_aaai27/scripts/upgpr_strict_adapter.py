from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np


Pair = tuple[int, int]
RankRecord = tuple[int, int | None]


def warm_only_negative_support(train_pairs: Iterable[Pair], n_items: int) -> np.ndarray:
    """Return the unnormalized UPGPR CF negative support for warm items."""
    support = np.zeros(int(n_items), dtype=np.float64)
    for _user, item in train_pairs:
        item_id = int(item)
        if item_id < 0 or item_id >= n_items:
            raise ValueError(f"Training item {item_id} is outside [0, {n_items}).")
        support[item_id] = 1.0
    if not np.any(support):
        raise ValueError("At least one warm training item is required.")
    return support


def reconstruct_cold_item_embeddings(
    item_embeddings: np.ndarray,
    relation_embeddings: Mapping[str, np.ndarray],
    tail_embeddings: Mapping[str, np.ndarray],
    item_relation_tails: Mapping[str, Mapping[int, Sequence[int]]],
    relation_tail_types: Mapping[str, str],
    cold_item_ids: Iterable[int],
    warm_item_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    """Reconstruct cold item rows from allowed static TransE triples.

    For a triple ``item + relation ~= tail``, the item estimate is
    ``tail - relation``. Multiple static edges are averaged.
    """
    reconstructed = np.asarray(item_embeddings).copy()
    if reconstructed.ndim != 2:
        raise ValueError("item_embeddings must be a rank-2 array.")

    warm_items = None if warm_item_ids is None else {int(item) for item in warm_item_ids}
    warm_tail_support: dict[str, set[int]] = {}
    if warm_items is not None:
        for relation, by_item in item_relation_tails.items():
            warm_tail_support[relation] = {
                int(tail)
                for item in warm_items
                for tail in by_item.get(item, ())
            }

    audit: dict[int, dict[str, object]] = {}
    for raw_item in sorted(set(cold_item_ids)):
        item = int(raw_item)
        if item < 0 or item >= reconstructed.shape[0]:
            raise ValueError(f"Cold item {item} is outside the embedding table.")
        estimates: list[np.ndarray] = []
        relations_used: list[str] = []
        discarded_unanchored_edges = 0
        for relation, by_item in item_relation_tails.items():
            if relation not in relation_embeddings or relation not in relation_tail_types:
                continue
            tail_type = relation_tail_types[relation]
            if tail_type not in tail_embeddings:
                continue
            relation_vector = np.asarray(relation_embeddings[relation]).reshape(-1)
            tail_table = np.asarray(tail_embeddings[tail_type])
            relation_had_edge = False
            for raw_tail in by_item.get(item, ()):
                tail = int(raw_tail)
                if warm_items is not None and tail not in warm_tail_support.get(relation, set()):
                    discarded_unanchored_edges += 1
                    continue
                if tail < 0 or tail >= tail_table.shape[0]:
                    raise ValueError(f"Tail {tail} for {relation} is outside {tail_type}.")
                estimates.append(tail_table[tail] - relation_vector)
                relation_had_edge = True
            if relation_had_edge:
                relations_used.append(relation)
        if estimates:
            reconstructed[item] = np.mean(np.stack(estimates, axis=0), axis=0)
        audit[item] = {
            "static_edge_count": len(estimates),
            "relations_used": relations_used,
            "reconstructed": bool(estimates),
            "discarded_unanchored_edges": discarded_unanchored_edges,
        }
    return reconstructed, audit


def build_strict_candidates(
    warm_item_ids: Iterable[int],
    cold_target: int,
    train_history: Iterable[int],
) -> np.ndarray:
    """Build all-warm plus current-cold-target candidates with history masking."""
    target = int(cold_target)
    history = {int(item) for item in train_history}
    candidates = ({int(item) for item in warm_item_ids} - history) | {target}
    return np.asarray(sorted(candidates), dtype=np.int64)


def rank_target(
    user_vector: np.ndarray,
    item_embeddings: np.ndarray,
    candidate_ids: Sequence[int],
    target: int,
) -> int | None:
    """Rank one target with deterministic item-id tie breaking (rank starts at 1)."""
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    target_id = int(target)
    if target_id not in set(candidates.tolist()):
        return None
    scores = np.asarray(item_embeddings)[candidates] @ np.asarray(user_vector)
    order = np.lexsort((candidates, -scores))
    ranked = candidates[order]
    return int(np.flatnonzero(ranked == target_id)[0]) + 1


def rank_target_with_path_priority(
    user_vector: np.ndarray,
    item_embeddings: np.ndarray,
    candidate_ids: Sequence[int],
    target: int,
    endpoint_probabilities: Mapping[int, float],
) -> int | None:
    """Total-rank candidates by path reachability, TransE score, and path probability."""
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    target_id = int(target)
    if target_id not in set(candidates.tolist()):
        return None
    transe_scores = np.asarray(item_embeddings)[candidates] @ np.asarray(user_vector)
    reached = np.asarray(
        [1 if int(item) in endpoint_probabilities else 0 for item in candidates],
        dtype=np.int8,
    )
    path_probabilities = np.asarray(
        [float(endpoint_probabilities.get(int(item), 0.0)) for item in candidates],
        dtype=np.float64,
    )
    order = np.lexsort((candidates, -path_probabilities, -transe_scores, -reached))
    ranked = candidates[order]
    return int(np.flatnonzero(ranked == target_id)[0]) + 1


def compute_item_macro_metrics(
    rank_records: Iterable[RankRecord],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float | int]:
    """Average per-example ranks within each cold target, then across targets."""
    normalized_ks = tuple(sorted({int(k) for k in ks}))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ks must contain positive integers.")

    grouped: dict[int, list[int | None]] = defaultdict(list)
    for target, rank in rank_records:
        grouped[int(target)].append(None if rank is None else int(rank))

    metrics: dict[str, float | int] = {"count": len(grouped)}
    for k in normalized_ks:
        per_item_recall: list[float] = []
        per_item_ndcg: list[float] = []
        for ranks in grouped.values():
            recalls = [1.0 if rank is not None and rank <= k else 0.0 for rank in ranks]
            ndcgs = [
                1.0 / math.log2(rank + 1.0) if rank is not None and rank <= k else 0.0
                for rank in ranks
            ]
            per_item_recall.append(float(np.mean(recalls)))
            per_item_ndcg.append(float(np.mean(ndcgs)))
        metrics[f"R@{k}"] = float(np.mean(per_item_recall)) if per_item_recall else 0.0
        metrics[f"N@{k}"] = float(np.mean(per_item_ndcg)) if per_item_ndcg else 0.0
    return metrics


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SPLIT_ROOT = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "strict_item_cold_balanced_thr1_seed_2025"
)
DEFAULT_STREAM = ROOT / "processed_data_hin_clean_pop5" / "stream_data.pkl"
DEFAULT_RELATIONS = ROOT / "MOOCCube" / "relations"
DEFAULT_UPGPR_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "UPGPR-courserec"
DEFAULT_OUTPUT = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_upgpr_strict"
    / "mooccube_seed2025_feasibility"
)
DEFAULT_FORMAL_THROUGHPUT_OUTPUT = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_upgpr_strict"
    / "mooccube_seed2025_fulltrain_throughput"
)


def _balanced_rows(frame, max_rows: int):
    if max_rows <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    groups = {
        int(item): group.sort_values("_row_id", kind="stable")
        for item, group in frame.groupby("i_idx", sort=True)
    }
    offsets = {item: 0 for item in groups}
    selected: list[int] = []
    while len(selected) < max_rows:
        added = False
        for item in sorted(groups):
            offset = offsets[item]
            group = groups[item]
            if offset >= len(group):
                continue
            selected.append(int(group.index[offset]))
            offsets[item] += 1
            added = True
            if len(selected) >= max_rows:
                break
        if not added:
            break
    return frame.loc[selected].copy()


def _read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def build_feasibility_input(
    split_root: Path,
    stream_path: Path,
    relations_dir: Path,
    max_validation_rows: int,
    max_test_rows: int,
    max_history_per_user: int,
    full_train: bool = False,
):
    import pandas as pd

    from paper_aaai27.scripts.course_baseline_adaptability import TinyAdapterInput, safe_token

    train = pd.read_pickle(split_root / "static_train.pkl")
    validation = pd.read_pickle(split_root / "static_val.pkl")
    test = pd.read_pickle(split_root / "static_test.pkl")
    stream = pd.read_pickle(stream_path)

    validation_cold = validation[validation["_split_source"].eq("strict_item_cold_val")]
    test_cold = test[test["_split_source"].eq("strict_item_cold_test")]
    selected_validation = _balanced_rows(validation_cold, max_validation_rows)
    selected_test = _balanced_rows(test_cold, max_test_rows)

    eval_users = set(selected_validation["u_idx"].astype(int)) | set(selected_test["u_idx"].astype(int))
    if full_train:
        selected_train = train.sort_values("_row_id", kind="stable")
    else:
        history = train[train["u_idx"].isin(eval_users)].sort_values(
            ["u_idx", "timestamp", "_row_id"], kind="stable"
        )
        history = history.groupby("u_idx", sort=False).tail(max_history_per_user)
        warm_anchors = train.sort_values("_row_id", kind="stable").drop_duplicates("i_idx")
        selected_train = (
            pd.concat([warm_anchors, history], ignore_index=False)
            .drop_duplicates(["u_idx", "i_idx"], keep="first")
            .sort_values("_row_id", kind="stable")
        )

    selected_users = sorted(set(selected_train["u_idx"].astype(int)) | eval_users)
    user_to_local = {user: local for local, user in enumerate(selected_users)}

    def remap(frame) -> list[Pair]:
        return [
            (user_to_local[int(row.u_idx)], int(row.i_idx))
            for row in frame[["u_idx", "i_idx"]].itertuples(index=False)
            if int(row.u_idx) in user_to_local
        ]

    idx_to_course = (
        stream[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .sort_values("i_idx")
        .set_index("i_idx")["course_id"]
        .to_dict()
    )
    n_items = int(stream["i_idx"].max()) + 1
    course_tokens = [safe_token(idx_to_course.get(item, f"course_{item}")) for item in range(n_items)]
    course_to_idx = {str(course): int(item) for item, course in idx_to_course.items()}

    course_concepts: dict[int, list[str]] = {item: [] for item in range(n_items)}
    for course, concept in _read_tsv_pairs(relations_dir / "course-concept.json"):
        item = course_to_idx.get(course)
        if item is not None and len(course_concepts[item]) < 8:
            course_concepts[item].append(safe_token(concept))

    course_teachers: dict[int, list[str]] = {item: [] for item in range(n_items)}
    for teacher, course in _read_tsv_pairs(relations_dir / "teacher-course.json"):
        item = course_to_idx.get(course)
        if item is not None and len(course_teachers[item]) < 4:
            course_teachers[item].append(safe_token(teacher))

    course_schools: dict[int, str] = {}
    for school, course in _read_tsv_pairs(relations_dir / "school-course.json"):
        item = course_to_idx.get(course)
        if item is not None and item not in course_schools:
            course_schools[item] = safe_token(school)

    data = TinyAdapterInput(
        n_users=len(selected_users),
        n_items=n_items,
        train_pairs=remap(selected_train),
        val_pairs=remap(selected_validation),
        test_pairs=remap(selected_test),
        course_tokens=course_tokens,
        course_concepts=course_concepts,
        course_teachers=course_teachers,
        course_schools=course_schools,
    )
    selection = {
        "global_train_rows": int(len(train)),
        "global_warm_items": int(train["i_idx"].nunique()),
        "global_validation_cold_items": int(validation_cold["i_idx"].nunique()),
        "global_test_cold_items": int(test_cold["i_idx"].nunique()),
        "full_train": bool(full_train),
        "selected_users": len(selected_users),
        "selected_train_rows": len(data.train_pairs),
        "selected_validation_rows": len(data.val_pairs),
        "selected_test_rows": len(data.test_pairs),
        "selected_validation_cold_items": len({item for _user, item in data.val_pairs}),
        "selected_test_cold_items": len({item for _user, item in data.test_pairs}),
    }
    return data, selection


def _labels(pairs: Iterable[Pair]) -> dict[int, list[int]]:
    labels: dict[int, list[int]] = defaultdict(list)
    for user, item in pairs:
        labels[int(user)].append(int(item))
    return dict(labels)


def _prepare_official_artifacts(
    upgpr_source: Path,
    data_dir: Path,
    tmp_dir: Path,
    config: dict[str, object],
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    test_pairs: list[Pair],
) -> dict[str, object]:
    sys.path.insert(0, str(upgpr_source))
    try:
        from data_utils import Dataset
        from easydict import EasyDict as edict
        from knowledge_graph import KnowledgeGraph
        from utils import save_dataset, save_kg, save_labels

        kg_args = edict(config["KG_ARGS"])
        dataset = Dataset(str(data_dir), kg_args, set_name="train")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        save_dataset(str(tmp_dir), dataset, False)
        kg = KnowledgeGraph(dataset, kg_args, use_user_relations=False, use_entity_relations=False)
        kg.compute_degrees()
        save_kg(str(tmp_dir), kg, False)
        save_labels(str(tmp_dir), _labels(train_pairs), mode="train", use_wandb=False)
        save_labels(str(tmp_dir), _labels(validation_pairs), mode="validation", use_wandb=False)
        save_labels(str(tmp_dir), _labels(test_pairs), mode="test", use_wandb=False)
        support = np.asarray(dataset.interactions.item_uniform_distrib)
        return {
            "interaction_rows": int(dataset.interactions.size),
            "cf_negative_support_items": np.flatnonzero(support > 0).astype(int).tolist(),
        }
    finally:
        sys.path.remove(str(upgpr_source))


def _base_config(
    source_config: Path,
    data_dir: Path,
    tmp_dir: Path,
    seed: int,
    embedding_epochs: int,
    policy_epochs: int,
    device: str,
    profile: str = "feasibility",
) -> dict[str, object]:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    gpu = "0" if device == "cuda" else "-1"
    for section in ("PREPROCESS", "TRAIN_EMBEDS", "TRAIN_AGENT", "TEST_AGENT"):
        config[section]["data_dir"] = str(data_dir)
        config[section]["tmp_dir"] = str(tmp_dir)
        config[section]["seed"] = seed
        config[section]["gpu"] = gpu
        config[section]["use_wandb"] = False
    config["TRAIN_EMBEDS"].update(epochs=embedding_epochs, min_epochs=0)
    config["TRAIN_AGENT"].update(epochs=policy_epochs, min_epochs=0)
    config["TEST_AGENT"].update(epochs=policy_epochs, run_eval=False)
    if profile == "feasibility":
        config["TRAIN_EMBEDS"].update(batch_size=256, embed_size=64, steps_per_checkpoint=20)
        config["TRAIN_AGENT"].update(batch_size=64, max_acts=100, hidden=[64, 32])
        config["TEST_AGENT"].update(max_acts=100, hidden=[64, 32], topk=[10, 3, 1])
    elif profile != "formal-throughput":
        raise ValueError(f"Unknown UPGPR profile: {profile}")
    return config


def _run_stage(
    python_executable: Path,
    source_dir: Path,
    script: str,
    config_path: Path,
    log_path: Path,
) -> dict[str, object]:
    started = time.time()
    completed = subprocess.run(
        [str(python_executable), "-u", script, "--config", str(config_path)],
        cwd=source_dir,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-4000:]
        raise RuntimeError(f"{script} failed with exit code {completed.returncode}:\n{tail}")
    return {"seconds": time.time() - started, "log": str(log_path)}


def _read_relation_rows(processed_dir: Path, relation_files: Mapping[str, str]) -> dict[str, dict[int, list[int]]]:
    result: dict[str, dict[int, list[int]]] = {}
    for relation, filename in relation_files.items():
        by_item: dict[int, list[int]] = {}
        with (processed_dir / filename).open("r", encoding="utf-8") as handle:
            for item, line in enumerate(handle):
                by_item[item] = [int(value) for value in line.strip().split() if value]
        result[relation] = by_item
    return result


def _reconstruct_saved_embeddings(
    tmp_dir: Path,
    processed_dir: Path,
    config: Mapping[str, object],
    cold_item_ids: set[int],
    warm_item_ids: set[int],
) -> dict[int, dict[str, object]]:
    embed_path = tmp_dir / "transe_embed.pkl"
    with embed_path.open("rb") as handle:
        embeddings = pickle.load(handle)
    item_relation = config["KG_ARGS"]["item_relation"]
    relation_files = {relation: values[0] for relation, values in item_relation.items()}
    relation_tail_types = {relation: values[1] for relation, values in item_relation.items()}
    relation_vectors = {relation: embeddings[relation][0] for relation in item_relation}
    tail_tables = {tail_type: embeddings[tail_type] for tail_type in set(relation_tail_types.values())}
    relation_rows = _read_relation_rows(processed_dir, relation_files)
    reconstructed, audit = reconstruct_cold_item_embeddings(
        embeddings["item"],
        relation_vectors,
        tail_tables,
        relation_rows,
        relation_tail_types,
        cold_item_ids,
        warm_item_ids=warm_item_ids,
    )
    embeddings["item"] = reconstructed
    with embed_path.open("wb") as handle:
        pickle.dump(embeddings, handle)
    return audit


def _predict_policy_paths(
    upgpr_source: Path,
    tmp_dir: Path,
    config: Mapping[str, object],
    split: str,
    device: str,
) -> Path:
    import torch
    from easydict import EasyDict as edict

    sys.path.insert(0, str(upgpr_source))
    try:
        from test_agent import predict_paths

        args = edict(config["TEST_AGENT"])
        args.device = torch.device(device)
        args.log_dir = str(tmp_dir / args.name)
        policy_file = Path(args.log_dir) / f"tmp_policy_model_epoch_{args.epochs}.ckpt"
        path_file = Path(args.log_dir) / f"policy_paths_{split}_epoch_{args.epochs}.pkl"
        predict_paths(str(policy_file), str(path_file), args, edict(config["KG_ARGS"]), data=split)
        return path_file
    finally:
        sys.path.remove(str(upgpr_source))


def _path_endpoint_probabilities(path_file: Path, users: Iterable[int]) -> dict[int, dict[int, float]]:
    with path_file.open("rb") as handle:
        results = pickle.load(handle)
    endpoints: dict[int, dict[int, float]] = {int(user): {} for user in users}
    for path, probabilities in zip(results["paths"], results["probs"]):
        if path[-1][1] == "item" and int(path[0][2]) in endpoints:
            user = int(path[0][2])
            item = int(path[-1][2])
            probability = float(np.prod(np.asarray(probabilities, dtype=np.float64)))
            endpoints[user][item] = max(probability, endpoints[user].get(item, -np.inf))
    return endpoints


def _evaluate_split(
    pairs: list[Pair],
    train_history: Mapping[int, Sequence[int]],
    warm_items: set[int],
    embeddings: Mapping[str, object],
    endpoints: Mapping[int, Mapping[int, float]],
) -> dict[str, object]:
    user_embeddings = np.asarray(embeddings["user"])
    enroll_vector = np.asarray(embeddings["enroll"][0])
    item_embeddings = np.asarray(embeddings["item"])
    policy_rank_records: list[RankRecord] = []
    transe_rank_records: list[RankRecord] = []
    reached_records: list[RankRecord] = []
    reached_targets = 0
    for user, target in pairs:
        candidates = build_strict_candidates(warm_items, target, train_history.get(user, ()))
        user_vector = user_embeddings[user] + enroll_vector
        user_endpoints = endpoints.get(user, {})
        policy_rank_records.append(
            (
                target,
                rank_target_with_path_priority(
                    user_vector,
                    item_embeddings,
                    candidates,
                    target,
                    user_endpoints,
                ),
            )
        )
        transe_rank_records.append(
            (target, rank_target(user_vector, item_embeddings, candidates, target))
        )
        reached = sorted(set(candidates.tolist()) & set(user_endpoints))
        if target in reached:
            reached_targets += 1
            reached_records.append((target, rank_target(user_vector, item_embeddings, reached, target)))
        else:
            reached_records.append((target, None))
    return {
        "full_cold_item_macro": compute_item_macro_metrics(policy_rank_records, ks=(5, 10)),
        "transe_full_candidate_fallback_item_macro": compute_item_macro_metrics(
            transe_rank_records,
            ks=(5, 10),
        ),
        "native_path_proxy_item_macro": compute_item_macro_metrics(reached_records, ks=(5, 10)),
        "target_path_reachability": reached_targets / len(pairs) if pairs else 0.0,
        "rows": len(pairs),
        "cold_items": len({item for _user, item in pairs}),
    }


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    validation = report["validation"]["full_cold_item_macro"]
    test = report["test"]["full_cold_item_macro"]
    lines = [
        "# UPGPR Strict Feasibility Audit",
        "",
        f"- Verdict: **{report['verdict']}**",
        f"- Seed: `{report['seed']}`",
        f"- Split: `{report['split_root']}`",
        f"- Device: `{report['device']}`",
        f"- Validation R@10 / N@10: `{validation['R@10']:.6f}` / `{validation['N@10']:.6f}`",
        f"- Test R@10 / N@10: `{test['R@10']:.6f}` / `{test['N@10']:.6f}`",
        f"- Test target path reachability: `{report['test']['target_path_reachability']:.6f}`",
        "",
        "This is a capped one-seed feasibility run. It is not eligible for the paper main table.",
        "Cold collaborative positives and CF negatives are excluded; cold course embeddings are reconstructed only from warm-anchored static metadata via mean(tail - relation).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feasibility(args: argparse.Namespace) -> dict[str, object]:
    from paper_aaai27.scripts.course_baseline_adaptability import write_upgpr_processed_dataset

    output_root = args.output
    if output_root is None:
        output_root = DEFAULT_FORMAL_THROUGHPUT_OUTPUT if args.profile == "formal-throughput" else DEFAULT_OUTPUT
    output = output_root.resolve()
    processed_dir = output / "processed_files"
    tmp_dir = output / "tmp"
    output.mkdir(parents=True, exist_ok=True)

    data, selection = build_feasibility_input(
        args.split_root.resolve(),
        args.stream.resolve(),
        args.relations.resolve(),
        args.max_validation_rows,
        args.max_test_rows,
        args.max_history_per_user,
        full_train=args.full_train or args.profile == "formal-throughput",
    )
    export_report = write_upgpr_processed_dataset(processed_dir, data)
    config = _base_config(
        args.upgpr_root / "config" / "UPGPR" / "mooc.json",
        processed_dir,
        tmp_dir,
        args.seed,
        args.embedding_epochs,
        args.policy_epochs,
        args.device,
        args.profile,
    )
    config["TRAIN_AGENT"]["max_train_steps"] = args.max_policy_steps
    config["TRAIN_AGENT"]["checkpoint_every_steps"] = args.checkpoint_every_steps
    config_path = output / "upgpr_strict_feasibility_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    source_dir = args.upgpr_root / "src" / "UPGPR"
    prep = _prepare_official_artifacts(
        source_dir,
        processed_dir,
        tmp_dir,
        config,
        data.train_pairs,
        data.val_pairs,
        data.test_pairs,
    )

    train_items = {item for _user, item in data.train_pairs}
    validation_cold = {item for _user, item in data.val_pairs}
    test_cold = {item for _user, item in data.test_pairs}
    cold_items = validation_cold | test_cold
    boundary = {
        "cold_items_absent_from_train_positives": cold_items.isdisjoint(train_items),
        "cold_items_absent_from_cf_negative_support": cold_items.isdisjoint(prep["cf_negative_support_items"]),
        "enrolments_equals_train": (
            (processed_dir / "enrolments.txt").read_text(encoding="utf-8")
            == (processed_dir / "train.txt").read_text(encoding="utf-8")
        ),
        "candidate_mode": "all_warm_plus_current_cold_target",
        "train_history_masked": True,
    }

    stages: dict[str, object] = {}
    if args.skip_embedding_training:
        embed_path = tmp_dir / "transe_embed.pkl"
        if not embed_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing embedding artifact: {embed_path}")
        stages["embedding_training"] = {"reused_existing_embedding": True, "path": str(embed_path)}
    else:
        stages["embedding_training"] = _run_stage(
            args.python,
            source_dir,
            "train_transe_model.py",
            config_path,
            output / "train_embeddings.log",
        )
    reconstruction = _reconstruct_saved_embeddings(
        tmp_dir,
        processed_dir,
        config,
        cold_items,
        train_items,
    )
    reconstructed_items = {item for item, row in reconstruction.items() if row["reconstructed"]}
    boundary["all_selected_cold_items_reconstructed"] = reconstructed_items == cold_items

    stages["policy_training"] = _run_stage(
        args.python,
        source_dir,
        "train_agent.py",
        config_path,
        output / "train_policy.log",
    )
    progress_path = tmp_dir / "train_agent" / "policy_training_progress.json"
    if progress_path.exists():
        stages["policy_training"]["progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
    validation_paths = _predict_policy_paths(source_dir, tmp_dir, config, "validation", args.device)
    test_paths = _predict_policy_paths(source_dir, tmp_dir, config, "test", args.device)

    with (tmp_dir / "transe_embed.pkl").open("rb") as handle:
        embeddings = pickle.load(handle)
    train_labels = _labels(data.train_pairs)
    validation_endpoints = _path_endpoint_probabilities(
        validation_paths,
        (user for user, _item in data.val_pairs),
    )
    test_endpoints = _path_endpoint_probabilities(
        test_paths,
        (user for user, _item in data.test_pairs),
    )
    validation_metrics = _evaluate_split(
        data.val_pairs, train_labels, train_items, embeddings, validation_endpoints
    )
    test_metrics = _evaluate_split(data.test_pairs, train_labels, train_items, embeddings, test_endpoints)

    gates = {
        **boundary,
        "validation_metrics_nonempty": validation_metrics["rows"] > 0,
        "test_metrics_nonempty": test_metrics["rows"] > 0,
        "official_embedding_training_completed": True,
        "official_policy_training_completed": True,
        "official_path_prediction_completed": validation_paths.exists() and test_paths.exists(),
    }
    boolean_gates = [value for value in gates.values() if isinstance(value, bool)]
    verdict = "FEASIBLE_FOR_FORMALIZATION" if all(boolean_gates) else "FEASIBILITY_GATE_FAILED"
    report = {
        "model": "UPGPR (adapted)",
        "audit_type": f"single_seed_{args.profile}",
        "verdict": verdict,
        "main_table_eligible": False,
        "seed": args.seed,
        "split_root": str(args.split_root.resolve()),
        "requested_device": args.device,
        "device": args.device,
        "python": str(args.python),
        "profile": args.profile,
        "training_config": {
            "embedding_epochs": config["TRAIN_EMBEDS"]["epochs"],
            "embedding_batch_size": config["TRAIN_EMBEDS"]["batch_size"],
            "embedding_dim": config["TRAIN_EMBEDS"]["embed_size"],
            "policy_epochs": config["TRAIN_AGENT"]["epochs"],
            "policy_batch_size": config["TRAIN_AGENT"]["batch_size"],
            "policy_max_acts": config["TRAIN_AGENT"]["max_acts"],
            "policy_hidden": config["TRAIN_AGENT"]["hidden"],
            "max_policy_steps": config["TRAIN_AGENT"].get("max_train_steps", -1),
        },
        "selection": selection,
        "export": export_report,
        "strict_gates": gates,
        "reconstruction": {
            "method": "mean_warm_anchored_tail_minus_relation",
            "selected_cold_items": len(cold_items),
            "reconstructed_cold_items": len(reconstructed_items),
            "items": reconstruction,
        },
        "stages": stages,
        "validation": validation_metrics,
        "test": test_metrics,
        "protocol_notes": [
            "Official UPGPR TransE and policy optimization are executed for one epoch each.",
            "The strict scorer totally ranks all warm candidates plus the current cold target by policy reachability, official TransE path score, then path probability.",
            "A pure TransE full-candidate fallback is reported separately as a diagnostic.",
            "Native policy reachability is reported separately and is not substituted for full-candidate ranking.",
            "This single-seed audit cannot be inserted into the paper main table.",
        ],
    }
    report_path = output / "upgpr_strict_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output / "upgpr_strict_feasibility_report.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict UPGPR item-cold feasibility audit.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--upgpr-root", type=Path, default=DEFAULT_UPGPR_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--embedding-epochs", type=int, default=1)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--max-validation-rows", type=int, default=68)
    parser.add_argument("--max-test-rows", type=int, default=136)
    parser.add_argument("--max-history-per-user", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=-1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile", choices=("feasibility", "formal-throughput"), default="feasibility")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--skip-embedding-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run_feasibility(parse_args())
    print(json.dumps({
        "verdict": report["verdict"],
        "validation": report["validation"],
        "test": report["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
