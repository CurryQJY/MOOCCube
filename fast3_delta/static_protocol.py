import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .eval import build_llm_score_tensor


class StreamDataset(Dataset):
    def __init__(self, df, llm_scores):
        user_ids = [int(x) for x in df["u_idx"].values]
        item_ids = [int(x) for x in df["i_idx"].values]
        self.u = torch.tensor(user_ids, dtype=torch.long)
        self.i = torch.tensor(item_ids, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)
        self.llm_s = build_llm_score_tensor(llm_scores, user_ids, item_ids)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {"u": self.u[idx], "i": self.i[idx], "pop": self.pop[idx], "llm": self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item["u"] for item in batch])
    i = torch.stack([item["i"] for item in batch])
    pop = torch.stack([item["pop"] for item in batch])
    llm = torch.stack([item["llm"] for item in batch])
    return {"u": u, "i": i}, pop, llm


def add_user_seen_from_df(user_seen_items, src_df):
    for u_idx, i_idx in zip(src_df["u_idx"].values, src_df["i_idx"].values):
        uid = int(u_idx)
        if uid not in user_seen_items:
            user_seen_items[uid] = set()
        user_seen_items[uid].add(int(i_idx))
    return user_seen_items


def clone_user_seen(user_seen_items):
    return {uid: set(items) for uid, items in user_seen_items.items()}


def static_seed():
    return int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025")))


def load_shared_static_split(split_dir):
    train_path = os.path.join(split_dir, "static_train.pkl")
    val_path = os.path.join(split_dir, "static_val.pkl")
    test_path = os.path.join(split_dir, "static_test.pkl")
    missing = [p for p in (train_path, val_path, test_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "USIM_STATIC_SPLIT_DIR is set but split files are missing: "
            + ", ".join(missing)
        )

    train_df = pd.read_pickle(train_path).copy()
    val_df = pd.read_pickle(val_path).copy()
    test_df = pd.read_pickle(test_path).copy()
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if "_split_source" not in split_df.columns:
            split_df["_split_source"] = f"shared_{split_name}"
        if "_row_id" not in split_df.columns:
            split_df["_row_id"] = np.arange(len(split_df), dtype=np.int64)

    summary_path = os.path.join(split_dir, "static_split_summary.json")
    split_info = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            split_info.update(loaded)

    total_rows = max(1, len(train_df) + len(val_df) + len(test_df))
    train_users = set(train_df["u_idx"].astype(int)) if "u_idx" in train_df.columns else set()
    val_users = set(val_df["u_idx"].astype(int)) if "u_idx" in val_df.columns else set()
    test_users = set(test_df["u_idx"].astype(int)) if "u_idx" in test_df.columns else set()
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    split_mode = os.environ.get("USIM_STATIC_SPLIT_MODE", "shared_static_split").strip().lower()
    split_info.setdefault("seed", static_seed())
    split_info.setdefault("split_mode", split_mode)
    split_info.setdefault("split_family", split_info.get("split_mode", "shared_static_split"))
    split_info.setdefault("train_ratio", train_ratio)
    split_info.setdefault("val_ratio", val_ratio)
    split_info.setdefault("test_ratio", test_ratio)
    split_info["train_rows"] = int(len(train_df))
    split_info["val_rows"] = int(len(val_df))
    split_info["test_rows"] = int(len(test_df))
    split_info.setdefault("actual_train_ratio", float(len(train_df) / total_rows))
    split_info.setdefault("actual_val_ratio", float(len(val_df) / total_rows))
    split_info.setdefault("actual_test_ratio", float(len(test_df) / total_rows))
    split_info.setdefault("val_user_seen_ratio", float(len(val_users & train_users) / max(1, len(val_users))))
    split_info.setdefault("test_user_seen_ratio", float(len(test_users & train_users) / max(1, len(test_users))))
    split_info.setdefault("train_item_coverage_moves", 0)
    split_info["static_split_loaded"] = True
    split_info["static_split_dir"] = str(split_dir)
    split_info["static_split_summary_loaded"] = bool(os.path.exists(summary_path))
    print(
        f"Loaded shared static split from {split_dir}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df, split_info


def split_exact_warm_user(source_df, seed, train_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    base_train_idx = []
    remaining_idx = []
    for _, group in source_df.groupby("u_idx", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        if idx.size == 0:
            continue
        base_train_idx.append(int(idx[0]))
        if idx.size > 1:
            remaining_idx.extend(int(x) for x in idx[1:])

    n_total = len(source_df)
    n_train_target = int(round(n_total * train_ratio))
    n_train_target = min(n_total, max(len(base_train_idx), n_train_target))
    n_val_target = int(round(n_total * val_ratio))
    n_val_target = max(0, min(n_val_target, n_total - n_train_target))

    remaining_idx = np.array(remaining_idx, dtype=np.int64)
    rng.shuffle(remaining_idx)
    extra_train = max(0, n_train_target - len(base_train_idx))
    extra_train = min(extra_train, remaining_idx.size)

    train_idx = list(base_train_idx) + [int(x) for x in remaining_idx[:extra_train]]
    tail_idx = remaining_idx[extra_train:]
    val_idx = [int(x) for x in tail_idx[:n_val_target]]
    test_idx = [int(x) for x in tail_idx[n_val_target:]]
    return train_idx, val_idx, test_idx


def split_user_leave_one_out(source_df, seed, train_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    for _, group in source_df.groupby("u_idx", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        if n >= 3:
            n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
            n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
            if n_val + n_test >= n:
                n_val = 1 if val_ratio > 0 else 0
                n_test = 1
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_idx.extend(int(x) for x in idx[:n_train])
        if n_val > 0:
            val_idx.extend(int(x) for x in idx[n_train:n_train + n_val])
        if n_test > 0:
            test_idx.extend(int(x) for x in idx[n_train + n_val:n_train + n_val + n_test])
    return train_idx, val_idx, test_idx


def loc_split(source_df, idx, split_source, seed=None, shuffle=False):
    if len(idx) == 0:
        out = source_df.iloc[0:0].copy()
    else:
        out = source_df.loc[idx].copy()
    if shuffle and len(out) > 0:
        out = out.sample(frac=1.0, random_state=seed)
    out["_split_source"] = split_source
    return out.reset_index(drop=True)


def ensure_train_item_coverage(train_df, val_df, test_df, required_items):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    required_items = set(int(x) for x in required_items)
    train_items = set(train_df["i_idx"].astype(int))
    missing_items = sorted(required_items - train_items)
    moved_rows = 0

    for item_id in missing_items:
        moved = None
        for split_name in ("val", "test"):
            src_df = val_df if split_name == "val" else test_df
            hit_idx = src_df.index[src_df["i_idx"].astype(int) == item_id]
            if len(hit_idx) < 1:
                continue
            row_idx = hit_idx[0]
            moved = src_df.loc[[row_idx]].copy()
            moved["_split_source"] = moved["_split_source"].astype(str) + "_coverage_train"
            if split_name == "val":
                val_df = val_df.drop(index=row_idx)
            else:
                test_df = test_df.drop(index=row_idx)
            break

        if moved is not None:
            train_df = pd.concat([train_df, moved], ignore_index=True)
            moved_rows += 1

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        moved_rows,
    )


def make_balanced_item_folds(item_counts, eligible_items, n_folds):
    eligible_items = [int(x) for x in eligible_items]
    n_folds = int(n_folds)
    if n_folds < 3:
        raise ValueError(f"USIM_STATIC_COLD_ITEM_FOLDS must be >= 3, got {n_folds}")
    if len(eligible_items) < n_folds:
        raise ValueError(
            f"Not enough eligible items ({len(eligible_items)}) for {n_folds} balanced folds"
        )

    sorted_items = (
        item_counts.loc[eligible_items]
        .sort_values(ascending=False, kind="mergesort")
        .index.astype(int)
        .tolist()
    )
    base_size = len(sorted_items) // n_folds
    extra = len(sorted_items) % n_folds
    capacities = [base_size + (1 if fold_id < extra else 0) for fold_id in range(n_folds)]
    folds = [[] for _ in range(n_folds)]
    fold_sums = [0 for _ in range(n_folds)]

    for item_id in sorted_items:
        candidates = [idx for idx in range(n_folds) if len(folds[idx]) < capacities[idx]]
        fold_id = min(candidates, key=lambda idx: (fold_sums[idx], len(folds[idx]), idx))
        folds[fold_id].append(int(item_id))
        fold_sums[fold_id] += int(item_counts.loc[item_id])

    return folds, fold_sums


def sample_strict_item_cold_items(item_counts, eligible_items, seed, split_mode):
    val_item_ratio = float(os.environ.get("USIM_STATIC_VAL_COLD_ITEM_RATIO", "0.05"))
    test_item_ratio = float(os.environ.get("USIM_STATIC_COLD_ITEM_RATIO", "0.10"))
    n_val_items = max(1, int(round(eligible_items.size * val_item_ratio)))
    n_test_items = max(1, int(round(eligible_items.size * test_item_ratio)))
    if n_val_items + n_test_items >= eligible_items.size:
        n_val_items = max(1, min(n_val_items, eligible_items.size // 4))
        n_test_items = max(1, min(n_test_items, eligible_items.size // 4))

    if split_mode in {"strict_item_cold_balanced", "item_cold_balanced", "balanced_item_cold"}:
        n_folds = int(os.environ.get("USIM_STATIC_COLD_ITEM_FOLDS", "20"))
        n_folds = min(n_folds, int(eligible_items.size))
        folds, fold_sums = make_balanced_item_folds(item_counts, eligible_items, n_folds)
        n_val_folds = max(1, int(round(n_folds * val_item_ratio)))
        n_test_folds = max(1, int(round(n_folds * test_item_ratio)))
        if n_val_folds + n_test_folds >= n_folds:
            n_val_folds = max(1, min(n_val_folds, n_folds // 4))
            n_test_folds = max(1, min(n_test_folds, n_folds // 4))
        if n_val_folds + n_test_folds >= n_folds:
            raise ValueError(
                "Invalid balanced item-cold fold allocation: "
                f"folds={n_folds}, val_folds={n_val_folds}, test_folds={n_test_folds}"
            )

        rng_items = np.random.default_rng(seed)
        fold_order = np.arange(n_folds)
        rng_items.shuffle(fold_order)
        val_fold_ids = [int(x) for x in fold_order[:n_val_folds]]
        test_fold_ids = [int(x) for x in fold_order[n_val_folds:n_val_folds + n_test_folds]]
        val_cold_items = {int(item_id) for fold_id in val_fold_ids for item_id in folds[fold_id]}
        test_cold_items = {int(item_id) for fold_id in test_fold_ids for item_id in folds[fold_id]}
        fold_sums_arr = np.asarray(fold_sums, dtype=np.float64)
        return val_cold_items, test_cold_items, {
            "strict_item_cold_sampling": "balanced_item_folds",
            "strict_item_cold_folds": int(n_folds),
            "strict_item_cold_val_folds": int(n_val_folds),
            "strict_item_cold_test_folds": int(n_test_folds),
            "strict_item_cold_val_fold_ids": val_fold_ids,
            "strict_item_cold_test_fold_ids": test_fold_ids,
            "strict_item_cold_fold_item_count_min": int(min(len(fold) for fold in folds)),
            "strict_item_cold_fold_item_count_max": int(max(len(fold) for fold in folds)),
            "strict_item_cold_fold_pop_sum_min": int(fold_sums_arr.min()),
            "strict_item_cold_fold_pop_sum_mean": float(fold_sums_arr.mean()),
            "strict_item_cold_fold_pop_sum_max": int(fold_sums_arr.max()),
            "strict_item_cold_fold_pop_sum_std": float(fold_sums_arr.std(ddof=1)) if n_folds > 1 else 0.0,
        }

    rng_items = np.random.default_rng(seed)
    shuffled_items = eligible_items.copy()
    rng_items.shuffle(shuffled_items)
    val_cold_items = {int(x) for x in shuffled_items[:n_val_items]}
    test_cold_items = {int(x) for x in shuffled_items[n_val_items:n_val_items + n_test_items]}
    return val_cold_items, test_cold_items, {
        "strict_item_cold_sampling": "random_items",
        "strict_item_cold_target_val_items": int(n_val_items),
        "strict_item_cold_target_test_items": int(n_test_items),
    }


def static_split_df(df):
    split_dir = os.environ.get("USIM_STATIC_SPLIT_DIR", "").strip()
    if split_dir:
        return load_shared_static_split(split_dir)

    seed = static_seed()
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))
    split_mode = os.environ.get("USIM_STATIC_SPLIT_MODE", "user_threshold_exact").strip().lower()
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    work_df = df.copy().reset_index(drop=True)
    work_df["_row_id"] = np.arange(len(work_df), dtype=np.int64)

    strict_item_cold = split_mode in {
        "item_cold",
        "cold_item",
        "strict_item_cold",
        "strict_item_cold_balanced",
        "item_cold_balanced",
        "balanced_item_cold",
    }
    coverage_moves = 0
    if strict_item_cold:
        item_counts = work_df["i_idx"].astype(int).value_counts()
        min_inter = int(os.environ.get("USIM_STATIC_COLD_ITEM_MIN_INTER", "5"))
        eligible_items = item_counts[item_counts >= min_inter].index.to_numpy(copy=True)
        if eligible_items.size < 3:
            raise ValueError(f"Not enough items for strict item-cold split: eligible_items={eligible_items.size}")
        val_cold_items, test_cold_items, cold_sampling_info = sample_strict_item_cold_items(
            item_counts,
            eligible_items,
            seed,
            split_mode,
        )
        heldout_items = val_cold_items | test_cold_items
        source_df = work_df[~work_df["i_idx"].astype(int).isin(heldout_items)].copy()
        train_idx, val_idx, test_idx = split_exact_warm_user(source_df, seed, train_ratio, val_ratio)
        train_df = loc_split(source_df, train_idx, "strict_item_cold_train", seed=seed, shuffle=True)
        val_warm_df = loc_split(source_df, val_idx, "strict_item_cold_warm_val")
        test_warm_df = loc_split(source_df, test_idx, "strict_item_cold_warm_test")
        train_df, val_warm_df, test_warm_df, coverage_moves = ensure_train_item_coverage(
            train_df,
            val_warm_df,
            test_warm_df,
            source_df["i_idx"].astype(int).unique(),
        )
        train_users = set(train_df["u_idx"].astype(int))
        val_cold_df = work_df[
            work_df["i_idx"].astype(int).isin(val_cold_items)
            & work_df["u_idx"].astype(int).isin(train_users)
        ].copy()
        test_cold_df = work_df[
            work_df["i_idx"].astype(int).isin(test_cold_items)
            & work_df["u_idx"].astype(int).isin(train_users)
        ].copy()
        val_cold_df["_split_source"] = "strict_item_cold_val"
        test_cold_df["_split_source"] = "strict_item_cold_test"
        val_df = pd.concat([val_warm_df, val_cold_df], ignore_index=True)
        test_df = pd.concat([test_warm_df, test_cold_df], ignore_index=True)
        split_family = "strict_item_cold"
    elif split_mode in {"user", "per_user", "user_history", "user-stratified", "user_leave_one_out"}:
        train_idx, val_idx, test_idx = split_user_leave_one_out(work_df, seed, train_ratio, val_ratio)
        train_df = loc_split(work_df, train_idx, "user_leave_one_out_train", seed=seed, shuffle=True)
        val_df = loc_split(work_df, val_idx, "user_leave_one_out_val")
        test_df = loc_split(work_df, test_idx, "user_leave_one_out_test")
        split_family = "user_leave_one_out"
    elif split_mode in {"global", "random"}:
        rng = np.random.default_rng(seed)
        idx = work_df.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_ratio))
        n_val = int(round(len(idx) * val_ratio))
        train_df = loc_split(work_df, idx[:n_train], "global_train", seed=seed, shuffle=True)
        val_df = loc_split(work_df, idx[n_train:n_train + n_val], "global_val")
        test_df = loc_split(work_df, idx[n_train + n_val:], "global_test")
        split_family = "global"
    elif split_mode in {"threshold", "user_threshold", "user_threshold_exact", "user_exact"}:
        train_idx, val_idx, test_idx = split_exact_warm_user(work_df, seed, train_ratio, val_ratio)
        train_df = loc_split(work_df, train_idx, "user_threshold_exact_train", seed=seed, shuffle=True)
        val_df = loc_split(work_df, val_idx, "user_threshold_exact_val")
        test_df = loc_split(work_df, test_idx, "user_threshold_exact_test")
        train_df, val_df, test_df, coverage_moves = ensure_train_item_coverage(
            train_df,
            val_df,
            test_df,
            work_df["i_idx"].astype(int).unique(),
        )
        split_family = "user_threshold_exact"
    else:
        raise ValueError(f"Unsupported USIM_STATIC_SPLIT_MODE={split_mode!r}")

    train_users = set(train_df["u_idx"].astype(int))
    val_users = set(val_df["u_idx"].astype(int))
    test_users = set(test_df["u_idx"].astype(int))
    split_info = {
        "seed": seed,
        "split_mode": split_mode,
        "split_family": split_family,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "actual_train_ratio": float(len(train_df) / max(1, len(work_df))),
        "actual_val_ratio": float(len(val_df) / max(1, len(work_df))),
        "actual_test_ratio": float(len(test_df) / max(1, len(work_df))),
        "val_user_seen_ratio": float(len(val_users & train_users) / max(1, len(val_users))),
        "test_user_seen_ratio": float(len(test_users & train_users) / max(1, len(test_users))),
        "train_item_coverage_moves": int(coverage_moves),
    }
    if strict_item_cold:
        val_item_pop = item_counts.loc[list(val_cold_items)].astype(int) if val_cold_items else pd.Series(dtype=int)
        test_item_pop = item_counts.loc[list(test_cold_items)].astype(int) if test_cold_items else pd.Series(dtype=int)
        split_info.update(
            {
                "val_cold_items": int(len(val_cold_items)),
                "test_cold_items": int(len(test_cold_items)),
                "strict_item_cold_min_inter": int(os.environ.get("USIM_STATIC_COLD_ITEM_MIN_INTER", "5")),
                "strict_item_cold_eligible_items": int(len(eligible_items)),
                "strict_item_cold_val_item_pop_sum": int(val_item_pop.sum()) if len(val_item_pop) else 0,
                "strict_item_cold_test_item_pop_sum": int(test_item_pop.sum()) if len(test_item_pop) else 0,
                "strict_item_cold_val_item_pop_mean": float(val_item_pop.mean()) if len(val_item_pop) else 0.0,
                "strict_item_cold_test_item_pop_mean": float(test_item_pop.mean()) if len(test_item_pop) else 0.0,
            }
        )
        split_info.update(cold_sampling_info)
    return train_df, val_df, test_df, split_info


def apply_train_popularity(train_df, val_df, test_df, cfg):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_counts = train_df["i_idx"].astype(int).value_counts().astype(int)
    for split_df in (train_df, val_df, test_df):
        if "raw_popularity" not in split_df.columns and "popularity" in split_df.columns:
            split_df["raw_popularity"] = split_df["popularity"]
        split_df["popularity"] = (
            split_df["i_idx"].astype(int).map(train_counts).fillna(0).astype(int)
        )
    item_train_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    for item_id, pop_value in train_counts.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_train_pop[idx] = int(pop_value)
    return train_df, val_df, test_df, item_train_pop


def static_split_counts(train_df, val_df, test_df, cfg):
    rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows.append(
            {
                "split": name,
                "rows": int(len(split_df)),
                "users": int(split_df["u_idx"].nunique()),
                "items": int(split_df["i_idx"].nunique()),
                "cold_rows": int((split_df["popularity"] < cfg.cold_threshold).sum()),
                "hot_rows": int((split_df["popularity"] >= cfg.cold_threshold).sum()),
                "zero_train_pop_rows": int((split_df["popularity"] == 0).sum()),
                "cold_threshold": int(cfg.cold_threshold),
            }
        )
    return rows


def write_static_split_artifacts(train_df, val_df, test_df, split_info, cfg, output_path_fn):
    split_info_path = output_path_fn("static_split_summary.json")
    split_counts_path = output_path_fn("static_split_counts.csv")
    split_sources_path = output_path_fn("static_split_sources.csv")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    pd.DataFrame(static_split_counts(train_df, val_df, test_df, cfg)).to_csv(split_counts_path, index=False)

    source_rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for source, count in split_df["_split_source"].value_counts().sort_index().items():
            source_rows.append({"split": name, "split_source": source, "rows": int(count)})
    pd.DataFrame(source_rows).to_csv(split_sources_path, index=False)

    exports = {
        "split_summary": split_info_path,
        "split_counts": split_counts_path,
        "split_sources": split_sources_path,
    }
    export_split = os.environ.get("USIM_STATIC_EXPORT_SPLIT", "1") == "1"
    export_shared_split = os.environ.get("USIM_STATIC_EXPORT_SHARED_SPLIT", "0") == "1"
    shared_split_loaded = bool(split_info.get("static_split_loaded", False))
    if export_split and (not shared_split_loaded or export_shared_split):
        train_path = output_path_fn("static_train.pkl")
        val_path = output_path_fn("static_val.pkl")
        test_path = output_path_fn("static_test.pkl")
        assignments_path = output_path_fn("static_split_assignments.csv")
        train_df.to_pickle(train_path)
        val_df.to_pickle(val_path)
        test_df.to_pickle(test_path)
        assign_parts = []
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            cols = ["_row_id", "u_idx", "i_idx", "_split_source"]
            part = split_df[cols].copy()
            part["split"] = split_name
            assign_parts.append(part)
        pd.concat(assign_parts, ignore_index=True).to_csv(assignments_path, index=False)
        exports.update(
            {
                "train_split": train_path,
                "val_split": val_path,
                "test_split": test_path,
                "split_assignments": assignments_path,
            }
        )
    elif export_split and shared_split_loaded:
        exports["split_export_skipped"] = "shared_static_split_loaded"
    return exports


_add_user_seen_from_df = add_user_seen_from_df
_apply_train_popularity = apply_train_popularity
_clone_user_seen = clone_user_seen
_ensure_train_item_coverage = ensure_train_item_coverage
_loc_split = loc_split
_make_balanced_item_folds = make_balanced_item_folds
_sample_strict_item_cold_items = sample_strict_item_cold_items
_split_exact_warm_user = split_exact_warm_user
_split_user_leave_one_out = split_user_leave_one_out
_static_seed = static_seed
_static_split_counts = static_split_counts
_static_split_df = static_split_df
