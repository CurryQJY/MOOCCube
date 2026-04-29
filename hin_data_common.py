import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def setup_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed fixed: {seed}")


def load_hin_processed(data_dir: str = "processed_data_hin") -> Tuple[Dict, pd.DataFrame, torch.Tensor]:
    meta_path = os.path.join(data_dir, "meta.json")
    stream_path = os.path.join(data_dir, "stream_data.pkl")
    content_path = os.path.join(data_dir, "content_emb.pt")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing file: {meta_path}")
    if not os.path.exists(stream_path):
        raise FileNotFoundError(f"Missing file: {stream_path}")
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Missing file: {content_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_pickle(stream_path)
    content_emb = torch.load(content_path, map_location="cpu")

    required = {"u_idx", "i_idx", "popularity"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"stream_data.pkl missing columns: {sorted(miss)}")

    return meta, df, content_emb


def static_split_df(
    df: pd.DataFrame,
    seed: int = 2025,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratio, require 0 < train,val and train+val < 1")

    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"Split too small: total={n_total}, train={n_train}, val={n_val}, test={n_test}"
        )

    train_df = df_shuffled.iloc[:n_train].copy()
    val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = df_shuffled.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def split_dataframe_by_periods(df: pd.DataFrame, period_type: str = "M"):
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df = df.copy()
        df["datetime"] = df["timestamp"]
    df["period_id"] = df["datetime"].dt.to_period(period_type)
    periods = []
    for key in sorted(df["period_id"].unique()):
        periods.append(df[df["period_id"] == key].reset_index(drop=True))
    return periods


def add_user_seen_from_df(user_seen_items: Dict[int, set], src_df: pd.DataFrame) -> Dict[int, set]:
    for u_idx, i_idx in zip(src_df["u_idx"].values, src_df["i_idx"].values):
        uid = int(u_idx)
        if uid not in user_seen_items:
            user_seen_items[uid] = set()
        user_seen_items[uid].add(int(i_idx))
    return user_seen_items


def clone_user_seen(user_seen_items: Dict[int, set]) -> Dict[int, set]:
    return {uid: set(items) for uid, items in user_seen_items.items()}


def build_user_seen(src_df: pd.DataFrame) -> Dict[int, set]:
    return add_user_seen_from_df({}, src_df)


class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.tensor(df["u_idx"].values, dtype=torch.long)
        self.i = torch.tensor(df["i_idx"].values, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.u)

    def __getitem__(self, idx: int):
        return {"u": self.u[idx], "i": self.i[idx], "pop": self.pop[idx]}


def collate_interactions(batch):
    u = torch.stack([item["u"] for item in batch])
    i = torch.stack([item["i"] for item in batch])
    pop = torch.stack([item["pop"] for item in batch])
    return {"u": u, "i": i}, pop
