import json
import os

import pandas as pd
import torch


DATA_DIR_ENV = "MOOC_DATA_DIR"


def resolve_data_dir(default="processed_data"):
    return os.environ.get(DATA_DIR_ENV, default)


def get_processed_paths(data_dir=None, default="processed_data"):
    resolved = data_dir or resolve_data_dir(default)
    return {
        "data_dir": resolved,
        "stream": os.path.join(resolved, "stream_data.pkl"),
        "meta": os.path.join(resolved, "meta.json"),
        "content": os.path.join(resolved, "content_emb.pt"),
        "llm": os.path.join(resolved, "llm_scores.pkl"),
    }


def load_processed_bundle(data_dir=None, default="processed_data", map_location="cpu"):
    paths = get_processed_paths(data_dir=data_dir, default=default)
    missing = [path for key, path in paths.items() if key in {"stream", "meta", "content"} and not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing processed data files under "
            f"'{paths['data_dir']}'. Expected: {', '.join(missing)}. "
            f"Set {DATA_DIR_ENV} or prepare that directory first."
        )

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_pickle(paths["stream"])
    content_emb = torch.load(paths["content"], map_location=map_location)
    return paths["data_dir"], meta, df, content_emb


def _llm_score_path(candidate):
    if candidate.endswith(".pkl"):
        return candidate
    return os.path.join(candidate, "llm_scores.pkl")


def _summarize_llm_score_compatibility(llm_scores, df, cold_threshold=5, n_users=None, n_items=None):
    df_pairs = set(zip(df["u_idx"].astype(int), df["i_idx"].astype(int)))
    cold_df = df[df["popularity"] < int(cold_threshold)]
    cold_pairs = set(zip(cold_df["u_idx"].astype(int), cold_df["i_idx"].astype(int)))

    stats = {
        "pair_total": 0,
        "pair_in_range": 0,
        "pair_in_df": 0,
        "pair_cold_hits": 0,
        "item_total": 0,
        "item_in_range": 0,
    }

    for key in llm_scores.keys():
        if isinstance(key, tuple) and len(key) == 2:
            try:
                u_idx = int(key[0])
                i_idx = int(key[1])
            except (TypeError, ValueError):
                continue
            stats["pair_total"] += 1
            in_range = True
            if n_users is not None:
                in_range = in_range and 0 <= u_idx < int(n_users)
            if n_items is not None:
                in_range = in_range and 0 <= i_idx < int(n_items)
            if in_range:
                stats["pair_in_range"] += 1
            pair = (u_idx, i_idx)
            if pair in df_pairs:
                stats["pair_in_df"] += 1
                if pair in cold_pairs:
                    stats["pair_cold_hits"] += 1
        else:
            try:
                item_idx = int(key)
            except (TypeError, ValueError):
                continue
            stats["item_total"] += 1
            if n_items is None or 0 <= item_idx < int(n_items):
                stats["item_in_range"] += 1

    pair_total = max(1, stats["pair_total"])
    item_total = max(1, stats["item_total"])
    stats["pair_match_ratio"] = stats["pair_in_df"] / pair_total
    stats["pair_cold_ratio"] = stats["pair_cold_hits"] / pair_total
    stats["pair_in_range_ratio"] = stats["pair_in_range"] / pair_total
    stats["item_in_range_ratio"] = stats["item_in_range"] / item_total
    return stats


def load_llm_scores_for_stream(
    data_dir,
    df,
    cold_threshold=5,
    n_users=None,
    n_items=None,
    fallback_data_dirs=None,
    verbose=True,
):
    if os.environ.get("USIM_DISABLE_LLM_SCORE", "0") == "1":
        if verbose:
            print("   LLM scores disabled by USIM_DISABLE_LLM_SCORE=1.")
        return {}, None, {"disabled": True}

    candidates = [_llm_score_path(data_dir)]
    for candidate in fallback_data_dirs or []:
        path = _llm_score_path(candidate)
        if path not in candidates:
            candidates.append(path)

    loaded = []
    for idx, path in enumerate(candidates):
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            llm_scores = pd.read_pickle(f)
        stats = _summarize_llm_score_compatibility(
            llm_scores,
            df,
            cold_threshold=cold_threshold,
            n_users=n_users,
            n_items=n_items,
        )
        loaded.append(
            {
                "path": path,
                "scores": llm_scores,
                "stats": stats,
                "is_primary": idx == 0,
            }
        )

    if not loaded:
        if verbose:
            print(f"   No llm_scores.pkl found for {data_dir}.")
        return {}, None, None

    primary = loaded[0]
    best = max(
        loaded,
        key=lambda x: (
            x["stats"]["pair_cold_hits"],
            x["stats"]["pair_in_df"],
            x["stats"]["pair_in_range"],
            -x["stats"]["pair_total"],
        ),
    )

    primary_stats = primary["stats"]
    primary_bad = (
        primary_stats["pair_total"] > 0
        and (
            primary_stats["pair_in_df"] == 0
            or (
                primary_stats["pair_match_ratio"] < 0.05
                and primary_stats["pair_cold_hits"] == 0
            )
        )
    )

    chosen = best if (best is not primary and primary_bad) else primary

    if verbose:
        for entry in loaded:
            stats = entry["stats"]
            label = "primary" if entry["is_primary"] else "fallback"
            print(
                "   LLM score candidate "
                f"[{label}] {entry['path']}: "
                f"pairs={stats['pair_total']}, "
                f"in_df={stats['pair_in_df']}, "
                f"cold_hits={stats['pair_cold_hits']}, "
                f"in_range={stats['pair_in_range']}"
            )
        if chosen is not primary:
            print(
                "   Warning: primary llm_scores.pkl looks incompatible with the current stream; "
                f"using {chosen['path']} instead."
            )

    strict_mode = os.environ.get("USIM_LLM_SCORE_STRICT", "0") == "1"
    if strict_mode and chosen is not primary:
        raise ValueError(
            "Primary llm_scores.pkl is incompatible with the current stream. "
            "Set USIM_LLM_SCORE_STRICT=0 to allow fallback, or regenerate matching llm_scores.pkl."
        )

    return chosen["scores"], chosen["path"], chosen["stats"]
