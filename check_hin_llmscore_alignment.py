import json
import os
import pickle

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_STREAM_DIR = os.path.join(BASE_DIR, "processed_data")
HIN_STREAM_DIR = os.path.join(BASE_DIR, "processed_data_hin")


def load_stream(data_dir):
    path = os.path.join(data_dir, "stream_data.pkl")
    return pd.read_pickle(path)


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scores(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def stream_alignment(base_df, hin_df):
    base_cols = base_df[["u_idx", "i_idx", "timestamp", "popularity"]].copy()
    hin_cols = hin_df[["u_idx", "i_idx", "timestamp", "popularity"]].copy()
    merged = base_cols.merge(
        hin_cols,
        on=["u_idx", "i_idx"],
        suffixes=("_base", "_hin"),
        how="inner",
    )
    return {
        "base_rows": len(base_df),
        "hin_rows": len(hin_df),
        "pair_overlap": len(merged),
        "timestamp_equal": int((merged["timestamp_base"] == merged["timestamp_hin"]).sum()),
        "popularity_equal": int((merged["popularity_base"] == merged["popularity_hin"]).sum()),
    }


def score_alignment(scores, hin_df, hin_meta, cold_threshold=5):
    if scores is None:
        return None

    n_users = int(hin_meta.get("n_users", 10**18))
    n_items = int(hin_meta.get("n_items", 10**18))
    pair_set = set(zip(hin_df["u_idx"].astype(int), hin_df["i_idx"].astype(int)))
    cold_df = hin_df[hin_df["popularity"] < cold_threshold]
    cold_pairs = set(zip(cold_df["u_idx"].astype(int), cold_df["i_idx"].astype(int)))

    pair_keys = [k for k in scores.keys() if isinstance(k, tuple) and len(k) == 2]
    item_keys = [k for k in scores.keys() if isinstance(k, int)]
    pair_keys_int = [(int(k[0]), int(k[1])) for k in pair_keys]

    in_df = sum(1 for key in pair_keys_int if key in pair_set)
    cold_hits = sum(1 for key in pair_keys_int if key in cold_pairs)
    out_user_range = sum(1 for u, _ in pair_keys_int if u < 0 or u >= n_users)
    out_item_range = sum(1 for _, i in pair_keys_int if i < 0 or i >= n_items)

    return {
        "total": len(scores),
        "pair_keys": len(pair_keys),
        "item_keys": len(item_keys),
        "pair_in_hin": in_df,
        "cold_hits": cold_hits,
        "out_user_range": out_user_range,
        "out_item_range": out_item_range,
    }


def print_block(title, stats):
    print(title)
    if stats is None:
        print("  missing")
        return
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    base_df = load_stream(BASE_STREAM_DIR)
    hin_df = load_stream(HIN_STREAM_DIR)
    hin_meta = load_meta(HIN_STREAM_DIR)

    stream_stats = stream_alignment(base_df, hin_df)
    print_block("STREAM_ALIGNMENT processed_data -> processed_data_hin", stream_stats)

    expected_rows = stream_stats["base_rows"]
    stream_ok = (
        stream_stats["hin_rows"] == expected_rows
        and stream_stats["pair_overlap"] == expected_rows
        and stream_stats["timestamp_equal"] == expected_rows
        and stream_stats["popularity_equal"] == expected_rows
    )
    print(f"  stream_aligned: {stream_ok}")

    for label, path in [
        ("HIN_PRIMARY_LLM", os.path.join(HIN_STREAM_DIR, "llm_scores.pkl")),
        ("BASE_FALLBACK_LLM", os.path.join(BASE_STREAM_DIR, "llm_scores.pkl")),
    ]:
        stats = score_alignment(load_scores(path), hin_df, hin_meta, cold_threshold=5)
        print_block(label, stats)
        if stats is not None:
            llm_ok = (
                stats["pair_keys"] > 0
                and stats["pair_in_hin"] == stats["pair_keys"]
                and stats["cold_hits"] == stats["pair_keys"]
                and stats["out_user_range"] == 0
                and stats["out_item_range"] == 0
            )
            print(f"  llm_aligned_for_cold: {llm_ok}")


if __name__ == "__main__":
    main()
