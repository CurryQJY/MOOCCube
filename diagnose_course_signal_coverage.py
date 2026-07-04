import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fast3_delta.config import FeedbackConfig
from fast3_delta.course_artifacts import build_course_artifacts


DEFAULT_OUTPUT_DIR = (
    "outputs/mooccubex/relations_aug_cmin001_e3/"
    "strict_item_cold_balanced_thr1_seed_2025"
)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_manifest(output_dir):
    path = Path(output_dir) / "static_protocol_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_manifest_env(manifest):
    env = manifest.get("env") or {}
    for key, value in env.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)


def _load_splits(output_dir):
    output_dir = Path(output_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = output_dir / f"static_{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = pd.read_pickle(path)
    return splits


def _resolve_meta_counts(manifest, splits):
    data_dir = (manifest.get("data") or {}).get("data_dir")
    meta_path = Path(data_dir) / "meta.json" if data_dir else None
    meta = {}
    if meta_path and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    n_users = int(meta.get("n_users", (manifest.get("data") or {}).get("users", 0)))
    n_items = int(meta.get("n_items", (manifest.get("data") or {}).get("items", 0)))
    max_user = max(int(df["u_idx"].max()) for df in splits.values() if len(df) > 0)
    max_item = max(int(df["i_idx"].max()) for df in splits.values() if len(df) > 0)
    n_users = max(n_users, max_user + 1)
    n_items = max(n_items, max_item + 1)
    return n_users, n_items


def _build_user_seen(train_df):
    user_seen = {}
    for uid, values in train_df.groupby("u_idx", sort=False)["i_idx"]:
        user_seen[int(uid)] = np.asarray(sorted(set(int(x) for x in values)), dtype=np.int32)
    return user_seen


def _build_item_pop(train_df, n_items):
    counts = train_df["i_idx"].astype(int).value_counts()
    pop = np.zeros(int(n_items), dtype=np.float32)
    for item_id, cnt in counts.items():
        idx = int(item_id)
        if 0 <= idx < n_items:
            pop[idx] = float(cnt)
    return pop


def _maybe_sample(df, max_rows, seed):
    max_rows = int(max_rows or 0)
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True), False
    return df.sample(n=max_rows, random_state=int(seed)).reset_index(drop=True), True


def _safe_ratio(numer, denom):
    denom = max(1, int(denom))
    return float(numer) / float(denom)


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "zero_batches": 0}
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "zero_batches": int((values <= 1e-12).sum()),
    }


def _batch_summary(mask, batch_size):
    n = int(mask.size)
    if n == 0:
        return _quantiles([])
    batch_ids = np.arange(n, dtype=np.int64) // int(batch_size)
    counts = np.bincount(batch_ids)
    hits = np.bincount(batch_ids, weights=mask.astype(np.float64), minlength=counts.size)
    return _quantiles(hits / np.maximum(counts, 1))


def _summarize_group(split_name, label, item_ids, rows, terms, cfg, item_has_concept, item_has_prereq):
    mask = rows["mask"]
    n = int(mask.sum())
    if n == 0:
        return None
    idx = np.where(mask)[0]
    items = item_ids[idx]
    active = terms["active"][idx]
    concept_match = terms["concept_match"][idx]
    concept_bonus = terms["concept_bonus"][idx]
    prereq_defined = terms["prereq_defined"][idx]
    prereq_gap = terms["prereq_gap"][idx]
    redundant = terms["redundant"][idx]
    difficulty_gap = terms["difficulty_gap"][idx]
    weighted = terms["weighted_adjust"][idx]
    any_signal = terms["any_signal"][idx]
    return {
        "split": split_name,
        "group": label,
        "rows": n,
        "items": int(np.unique(items).size),
        "active_rate": float(active.mean()),
        "item_has_concept_rate": float(item_has_concept[items].mean()) if items.size else 0.0,
        "item_has_prereq_rate": float(item_has_prereq[items].mean()) if items.size else 0.0,
        "concept_match_nonzero_rate": float((concept_match > 1e-12).mean()),
        "concept_match_mean": float(concept_match.mean()),
        "concept_bonus_nonzero_rate": float((concept_bonus > 1e-12).mean()),
        "concept_bonus_mean": float(concept_bonus.mean()),
        "prereq_defined_rate": float(prereq_defined.mean()),
        "prereq_gap_nonzero_rate": float((prereq_gap > 1e-12).mean()),
        "prereq_gap_mean": float(prereq_gap.mean()),
        "redundant_nonzero_rate": float((redundant > 1e-12).mean()),
        "redundant_mean": float(redundant.mean()),
        "difficulty_gap_nonzero_rate": float((difficulty_gap > 1e-12).mean()),
        "difficulty_gap_mean": float(difficulty_gap.mean()),
        "any_course_signal_rate": float(any_signal.mean()),
        "weighted_adjust_mean": float(weighted.mean()),
        "weighted_adjust_abs_mean": float(np.abs(weighted).mean()),
        "weighted_adjust_nonzero_rate": float((np.abs(weighted) > 1e-12).mean()),
        "batch_any_signal": _batch_summary(any_signal, cfg.batch_size),
        "batch_concept_bonus": _batch_summary(concept_bonus > 1e-12, cfg.batch_size),
        "batch_prereq_gap": _batch_summary(prereq_gap > 1e-12, cfg.batch_size),
        "batch_redundant": _batch_summary(redundant > 1e-12, cfg.batch_size),
    }


def _item_rows(split_name, group_label, item_ids, mask, terms, item_has_concept, item_has_prereq):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    items = item_ids[idx]
    n_items = int(max(item_ids.max(initial=0), 0)) + 1
    rows = []
    counts = np.bincount(items, minlength=n_items).astype(np.float64)
    valid_items = np.where(counts > 0)[0]
    for item_id in valid_items:
        m = items == item_id
        src_idx = idx[m]
        rows.append(
            {
                "split": split_name,
                "group": group_label,
                "i_idx": int(item_id),
                "rows": int(counts[item_id]),
                "item_has_concept": bool(item_has_concept[item_id]),
                "item_has_prereq": bool(item_has_prereq[item_id]),
                "concept_match_mean": float(terms["concept_match"][src_idx].mean()),
                "concept_bonus_nonzero_rate": float((terms["concept_bonus"][src_idx] > 1e-12).mean()),
                "concept_bonus_mean": float(terms["concept_bonus"][src_idx].mean()),
                "prereq_gap_nonzero_rate": float((terms["prereq_gap"][src_idx] > 1e-12).mean()),
                "prereq_gap_mean": float(terms["prereq_gap"][src_idx].mean()),
                "redundant_nonzero_rate": float((terms["redundant"][src_idx] > 1e-12).mean()),
                "redundant_mean": float(terms["redundant"][src_idx].mean()),
                "difficulty_gap_mean": float(terms["difficulty_gap"][src_idx].mean()),
                "any_course_signal_rate": float(terms["any_signal"][src_idx].mean()),
                "weighted_adjust_mean": float(terms["weighted_adjust"][src_idx].mean()),
                "weighted_adjust_abs_mean": float(np.abs(terms["weighted_adjust"][src_idx]).mean()),
            }
        )
    return rows


def _compute_split_terms(df, split_name, cfg, user_seen, overlap, prereq_mat, prereq_cnt, item_pop):
    n = len(df)
    item_ids = df["i_idx"].astype(np.int32).to_numpy(copy=True)
    user_ids = df["u_idx"].astype(np.int32).to_numpy(copy=True)
    pop = df["popularity"].astype(np.float32).to_numpy(copy=True)

    concept_match = np.zeros(n, dtype=np.float32)
    prereq_seen_ratio = np.zeros(n, dtype=np.float32)
    prereq_defined = prereq_cnt[item_ids] > 1e-12
    seen_counts = np.zeros(n, dtype=np.float32)

    grouped = defaultdict(list)
    for pos, uid in enumerate(user_ids):
        grouped[int(uid)].append(pos)

    for uid, positions in grouped.items():
        seen = user_seen.get(uid)
        positions = np.asarray(positions, dtype=np.int64)
        if seen is None or seen.size == 0:
            continue
        seen_counts[positions] = float(seen.size)
        items = item_ids[positions]

        # overlap[item, seen] = how much the target item's concepts are already
        # represented in the user's train-only history.
        max_cells = 1_500_000
        chunk_size = max(1, int(max_cells // max(1, seen.size)))
        match_mode = str(getattr(cfg, "feedback_course_match_mode", "mean")).strip().lower()
        match_topk = max(1, int(getattr(cfg, "feedback_course_match_topk", 5)))
        exclude_target = bool(getattr(cfg, "feedback_course_match_exclude_target", False))
        seen_pos = {int(item_id): col for col, item_id in enumerate(seen)} if exclude_target else {}
        for start in range(0, items.size, chunk_size):
            end = min(items.size, start + chunk_size)
            chunk_items = items[start:end]
            scores = overlap[chunk_items][:, seen]
            effective_seen = np.full(scores.shape[0], scores.shape[1], dtype=np.float32)
            if exclude_target:
                scores = scores.copy()
                for row_idx, item_id in enumerate(chunk_items):
                    col = seen_pos.get(int(item_id))
                    if col is not None:
                        scores[row_idx, col] = 0.0
                        effective_seen[row_idx] -= 1.0
                effective_seen = np.maximum(effective_seen, 0.0)
            if match_mode == "max":
                match_vals = scores.max(axis=1)
            elif match_mode == "topk":
                k = min(match_topk, scores.shape[1])
                if k < scores.shape[1]:
                    top_vals = np.partition(scores, -k, axis=1)[:, -k:]
                else:
                    top_vals = scores
                denom = np.maximum(1.0, np.minimum(effective_seen, float(k)))
                match_vals = top_vals.sum(axis=1) / denom
            else:
                match_vals = scores.sum(axis=1) / np.maximum(1.0, effective_seen)
            concept_match[positions[start:end]] = match_vals

        prereq_seen_all = prereq_mat[:, seen].sum(axis=1)
        cnt = prereq_cnt[items]
        valid = cnt > 1e-12
        vals = np.zeros(items.size, dtype=np.float32)
        vals[valid] = prereq_seen_all[items[valid]] / np.maximum(cnt[valid], 1e-12)
        prereq_seen_ratio[positions] = np.clip(vals, 0.0, 1.0)

    prereq_raw_gap = np.zeros(n, dtype=np.float32)
    valid_prereq = prereq_cnt[item_ids] > 1e-12
    prereq_raw_gap[valid_prereq] = 1.0 - prereq_seen_ratio[valid_prereq]
    prereq_raw_gap = np.clip(prereq_raw_gap, 0.0, 1.0)

    gate = float(min(1.0, max(0.0, cfg.feedback_course_prereq_gate)))
    if cfg.feedback_prereq_soft_penalty:
        prereq_gap = np.maximum(prereq_raw_gap - gate, 0.0) / max(1e-6, 1.0 - gate)
        prereq_safe = np.clip(1.0 - prereq_gap, 0.0, 1.0)
    else:
        prereq_gap = prereq_raw_gap
        prereq_safe = (prereq_raw_gap <= gate).astype(np.float32)

    redundant_thr = float(min(0.99, max(0.0, cfg.feedback_course_redundant_thr)))
    concept_min = float(min(redundant_thr - 1e-3, max(0.0, cfg.feedback_course_concept_min)))
    concept_band = max(1e-6, redundant_thr - concept_min)
    concept_bonus = np.clip((concept_match - concept_min) / concept_band, 0.0, 1.0)
    redundant = np.clip((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr), 0.0, 1.0)
    redundant_gate = float(min(1.0, max(0.0, cfg.feedback_course_redundant_concept_gate)))
    seen_active = (seen_counts >= 1.0).astype(np.float32)
    concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant_gate * redundant)
    redundant = redundant * seen_active

    max_log = max(float(np.log1p(item_pop.max())), 1.0)
    item_difficulty = np.clip(1.0 - np.log1p(item_pop) / max_log, 0.0, 1.0)
    user_readiness = np.clip(seen_counts / max(1.0, float(cfg.feedback_course_warm_seen)), 0.0, 1.0)
    difficulty_gap = np.maximum(item_difficulty[item_ids] - user_readiness, 0.0)

    if cfg.feedback_course_only_cold:
        active = (pop < float(cfg.cold_threshold)).astype(np.float32)
    else:
        active = np.ones(n, dtype=np.float32)
    concept_bonus *= active
    prereq_gap *= active
    redundant *= active
    difficulty_gap *= active

    weighted_adjust = (
        float(cfg.feedback_course_concept_weight) * concept_bonus
        - float(cfg.feedback_course_prereq_weight) * prereq_gap
        - float(cfg.feedback_course_difficulty_weight) * difficulty_gap
        - float(cfg.feedback_course_redundant_weight) * redundant
    ).astype(np.float32)
    any_signal = (
        (concept_bonus > 1e-12)
        | (prereq_gap > 1e-12)
        | (difficulty_gap > 1e-12)
        | (redundant > 1e-12)
    )

    terms = {
        "pop": pop,
        "seen_counts": seen_counts,
        "active": active,
        "concept_match": concept_match,
        "concept_bonus": concept_bonus.astype(np.float32),
        "prereq_defined": prereq_defined.astype(np.float32),
        "prereq_gap": prereq_gap.astype(np.float32),
        "redundant": redundant.astype(np.float32),
        "difficulty_gap": difficulty_gap.astype(np.float32),
        "weighted_adjust": weighted_adjust,
        "any_signal": any_signal,
    }
    cold_mask = pop < float(cfg.cold_threshold)
    rows = {
        "all": np.ones(n, dtype=bool),
        "cold": cold_mask,
        "hot": ~cold_mask,
        "mask": None,
    }
    return item_ids, rows, terms


def main():
    parser = argparse.ArgumentParser(description="Diagnose FAST3 course reward signal coverage.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = _load_manifest(output_dir)
    _apply_manifest_env(manifest)

    full_splits = _load_splits(output_dir)
    n_users, n_items = _resolve_meta_counts(manifest, full_splits)
    cfg = FeedbackConfig(n_users, n_items, content_dim=768)

    sampled = {}
    splits = {}
    splits["train"], sampled["train"] = _maybe_sample(full_splits["train"], args.max_train_rows, args.seed)
    splits["val"], sampled["val"] = _maybe_sample(full_splits["val"], args.max_val_rows, args.seed)
    splits["test"], sampled["test"] = _maybe_sample(full_splits["test"], args.max_test_rows, args.seed)

    artifact_df = pd.concat(
        [df[["i_idx", "course_id", "u_idx", "timestamp"]] for df in full_splits.values()],
        ignore_index=True,
    )
    course_artifacts, course_stats = build_course_artifacts(
        artifact_df,
        cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    overlap = course_artifacts["item_concept_overlap"].detach().cpu().numpy().astype(np.float32)
    prereq_mat = course_artifacts["item_prereq_item_mat"].detach().cpu().numpy().astype(np.float32)
    prereq_cnt = course_artifacts["item_prereq_item_cnt"].detach().cpu().numpy().astype(np.float32)
    item_has_concept = np.diag(overlap) > 1e-12
    item_has_prereq = prereq_cnt > 1e-12
    item_pop = _build_item_pop(full_splits["train"], cfg.n_items)
    user_seen = _build_user_seen(full_splits["train"])
    print(
        "Loaded splits: train={}{} val={}{} test={}{} | users={} items={}".format(
            len(splits["train"]),
            " sampled" if sampled["train"] else "",
            len(splits["val"]),
            " sampled" if sampled["val"] else "",
            len(splits["test"]),
            " sampled" if sampled["test"] else "",
            cfg.n_users,
            cfg.n_items,
        ),
        flush=True,
    )
    print(
        "Course artifacts: concept_items={} prereq_items={} prereq_edges_kept={} hard_density={:.6f}".format(
            course_stats.get("items_with_concept", 0),
            course_stats.get("items_with_prereq", 0),
            course_stats.get("prereq_edges_kept", 0),
            float(course_stats.get("hard_density", 0.0)),
        ),
        flush=True,
    )

    split_rows = []
    item_rows = []
    for split_name in ["train", "val", "test"]:
        print(f"Diagnosing split={split_name} rows={len(splits[split_name])} ...", flush=True)
        item_ids, rows, terms = _compute_split_terms(
            splits[split_name],
            split_name,
            cfg,
            user_seen,
            overlap,
            prereq_mat,
            prereq_cnt,
            item_pop,
        )
        for group_name in ["all", "cold", "hot"]:
            rows["mask"] = rows[group_name]
            summary = _summarize_group(
                split_name,
                group_name,
                item_ids,
                rows,
                terms,
                cfg,
                item_has_concept,
                item_has_prereq,
            )
            if summary:
                split_rows.append(summary)
            if group_name in {"cold", "hot"}:
                item_rows.extend(
                    _item_rows(
                        split_name,
                        group_name,
                        item_ids,
                        rows[group_name],
                        terms,
                        item_has_concept,
                        item_has_prereq,
                    )
                )

    config_snapshot = {
        "output_dir": str(output_dir),
        "sampled": sampled,
        "cold_threshold": int(cfg.cold_threshold),
        "course_only_cold": bool(cfg.feedback_course_only_cold),
        "course_sample_only_cold": bool(cfg.feedback_course_sample_only_cold),
        "course_match_mode": str(getattr(cfg, "feedback_course_match_mode", "mean")),
        "course_match_topk": int(getattr(cfg, "feedback_course_match_topk", 5)),
        "course_match_exclude_target": bool(getattr(cfg, "feedback_course_match_exclude_target", False)),
        "concept_min": float(cfg.feedback_course_concept_min),
        "redundant_thr": float(cfg.feedback_course_redundant_thr),
        "prereq_gate": float(cfg.feedback_course_prereq_gate),
        "prereq_graph_source": str(cfg.prereq_graph_source),
        "concept_weight": float(cfg.feedback_course_concept_weight),
        "prereq_weight": float(cfg.feedback_course_prereq_weight),
        "difficulty_weight": float(cfg.feedback_course_difficulty_weight),
        "redundant_weight": float(cfg.feedback_course_redundant_weight),
        "course_stats": course_stats,
    }

    summary_path = output_dir / "course_signal_coverage_summary.json"
    split_csv_path = output_dir / "course_signal_coverage_by_split.csv"
    item_csv_path = output_dir / "course_signal_coverage_by_item.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config_snapshot,
                "by_split": split_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    flat_rows = []
    for row in split_rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for key, value in row.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat[f"{key}_{sub_key}"] = sub_val
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(split_csv_path, index=False)
    pd.DataFrame(item_rows).to_csv(item_csv_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved split CSV: {split_csv_path}")
    print(f"Saved item CSV: {item_csv_path}")
    test_cold = next((r for r in split_rows if r["split"] == "test" and r["group"] == "cold"), None)
    train_all = next((r for r in split_rows if r["split"] == "train" and r["group"] == "all"), None)
    if train_all:
        print(
            "TRAIN all: any={:.4f}, concept_bonus={:.4f}, prereq_gap={:.4f}, redundant={:.4f}, abs_adj={:.6f}".format(
                train_all["any_course_signal_rate"],
                train_all["concept_bonus_nonzero_rate"],
                train_all["prereq_gap_nonzero_rate"],
                train_all["redundant_nonzero_rate"],
                train_all["weighted_adjust_abs_mean"],
            )
        )
    if test_cold:
        print(
            "TEST cold: any={:.4f}, concept_bonus={:.4f}, prereq_defined={:.4f}, prereq_gap={:.4f}, redundant={:.4f}, abs_adj={:.6f}".format(
                test_cold["any_course_signal_rate"],
                test_cold["concept_bonus_nonzero_rate"],
                test_cold["prereq_defined_rate"],
                test_cold["prereq_gap_nonzero_rate"],
                test_cold["redundant_nonzero_rate"],
                test_cold["weighted_adjust_abs_mean"],
            )
        )


if __name__ == "__main__":
    main()
