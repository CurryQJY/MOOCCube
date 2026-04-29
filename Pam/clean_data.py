import pandas as pd
import pickle
import os

# 路径配置
DATA_DIR = "processed_data_video"
INPUT_FILE = os.path.join(DATA_DIR, "stream_data.pkl")
OUTPUT_FILE = os.path.join(DATA_DIR, "stream_data_5core.pkl")


def k_core_filter(df, k=5):
    print(f"原始数据: {len(df)} 条, 用户: {df['u'].nunique()}, 物品: {df['i'].nunique()}")

    cycle = 0
    while True:
        cycle += 1
        # 1. 过滤用户
        u_counts = df['u'].value_counts()
        valid_u = u_counts[u_counts >= k].index
        df = df[df['u'].isin(valid_u)]

        # 2. 过滤物品
        i_counts = df['i'].value_counts()
        valid_i = i_counts[i_counts >= k].index
        df = df[df['i'].isin(valid_i)]

        print(f"Cycle {cycle}: 剩余 {len(df)} 条 (U: {df['u'].nunique()}, I: {df['i'].nunique()})")

        # 如果不再变化，停止
        if len(u_counts[u_counts < k]) == 0 and len(i_counts[i_counts < k]) == 0:
            break

    return df


def main():
    if not os.path.exists(INPUT_FILE):
        print("找不到 stream_data.pkl")
        return

    df = pd.read_pickle(INPUT_FILE)

    # 执行 5-core 过滤
    df_clean = k_core_filter(df, k=5)

    # 重新编码 ID (Re-indexing)
    # 因为过滤后 ID 会断层，必须重新映射
    print("重新建立索引映射...")
    u_map = {u: i for i, u in enumerate(df_clean['u'].unique())}
    i_map = {i: x for x, i in enumerate(df_clean['i'].unique())}

    df_clean['u_idx'] = df_clean['u'].map(u_map)
    df_clean['i_idx'] = df_clean['i'].map(i_map)

    # 重新计算 Popularity
    df_clean['popularity'] = df_clean.groupby('i_idx')['i_idx'].transform('count')

    # 保存映射 (训练时需要知道总数)
    meta = {
        'n_users': len(u_map),
        'n_items': len(i_map),
        'u_map': u_map,
        'i_map': i_map
    }

    df_clean.to_pickle(OUTPUT_FILE)
    with open(os.path.join(DATA_DIR, "meta_5core.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ 清洗完成！已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()