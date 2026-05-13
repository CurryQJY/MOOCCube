"""
filter_hin_x.py
---------------
对 processed_data_hin_x 做用户过滤，减少内存占用。
- 保留交互 >= MIN_INTERACTIONS 次的用户
- 保持 item 编码不变 (匹配已有 content_emb.pt)
- 仅重新编码 user ID 并更新 meta.json
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "./processed_data_hin_x"
MIN_INTERACTIONS = 20


def main():
    print("=" * 60)
    print(f"过滤 {DATA_DIR}: 保留交互 >= {MIN_INTERACTIONS} 的用户")
    print("=" * 60)

    pkl_path = os.path.join(DATA_DIR, "stream_data.pkl")
    meta_path = os.path.join(DATA_DIR, "meta.json")

    # 备份
    backup = pkl_path + ".full_backup"
    if not os.path.exists(backup):
        shutil.copy2(pkl_path, backup)
        print(f"[备份] {backup}")
    else:
        print(f"[备份] 已存在，跳过")

    meta_backup = meta_path + ".full_backup"
    if not os.path.exists(meta_backup):
        shutil.copy2(meta_path, meta_backup)

    # 加载 (优先从备份读取原始完整数据)
    print("\n[1/4] 加载数据...")
    src_pkl = backup if os.path.exists(backup) else pkl_path
    print(f"  数据源: {src_pkl}")
    df = pd.read_pickle(src_pkl)
    src_meta = meta_backup if os.path.exists(meta_backup) else meta_path
    with open(src_meta, "r") as f:
        meta = json.load(f)

    print(f"  原始: {len(df)} 记录, {df['u_idx'].nunique()} 用户, {df['i_idx'].nunique()} 课程")

    # 过滤
    print(f"\n[2/4] 过滤用户 (>= {MIN_INTERACTIONS} 次交互)...")
    user_counts = df['user_id'].value_counts()
    keep_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    df_filtered = df[df['user_id'].isin(keep_users)].copy()

    # 检查课程是否全部保留
    items_before = df['i_idx'].nunique()
    items_after = df_filtered['i_idx'].nunique()
    print(f"  过滤后: {len(df_filtered)} 记录, {df_filtered['user_id'].nunique()} 用户")
    print(f"  课程: {items_before} -> {items_after} (丢失 {items_before - items_after})")

    if items_after < items_before:
        print(f"  ⚠ 有 {items_before - items_after} 门课程失去了所有交互")
        print(f"  保持 i_idx 不变以匹配 content_emb.pt")

    # 重新编码 user ID (item ID 保持不变)
    print(f"\n[3/4] 重新编码 user ID...")
    user_enc = LabelEncoder()
    df_filtered['u_idx'] = user_enc.fit_transform(df_filtered['user_id'])

    # 重算 popularity
    df_filtered = df_filtered.sort_values('timestamp').reset_index(drop=True)
    df_filtered['popularity'] = df_filtered.groupby('i_idx').cumcount()

    n_users = len(user_enc.classes_)
    n_items = meta['n_items']  # 保持不变

    # 保存
    print(f"\n[4/4] 保存...")
    df_filtered.to_pickle(pkl_path)

    new_meta = {"n_users": n_users, "n_items": n_items}
    if "content_dim" in meta:
        new_meta["content_dim"] = meta["content_dim"]
    with open(meta_path, "w") as f:
        json.dump(new_meta, f)

    # 清空 llm_scores (旧的 key 不匹配了)
    llm_path = os.path.join(DATA_DIR, "llm_scores.pkl")
    with open(llm_path, "wb") as f:
        pickle.dump({}, f)

    # 内存估算
    emb_mem_mb = n_users * 128 * 4 / 1024 / 1024
    adam_mem_mb = emb_mem_mb * 2
    total_mb = emb_mem_mb + adam_mem_mb

    print(f"\n{'=' * 60}")
    print(f"完成!")
    print(f"{'=' * 60}")
    print(f"  记录: {len(df_filtered):,}")
    print(f"  用户: {n_users:,}")
    print(f"  课程: {n_items:,}")
    print(f"  用户嵌入: ~{emb_mem_mb:.0f}MB | Adam: ~{adam_mem_mb:.0f}MB | 合计: ~{total_mb:.0f}MB")
    dt = pd.to_datetime(df_filtered['timestamp'], unit='s')
    print(f"  时间: {dt.min().date()} ~ {dt.max().date()}")
    print(f"  月份: {dt.dt.to_period('M').nunique()}")


if __name__ == "__main__":
    main()
