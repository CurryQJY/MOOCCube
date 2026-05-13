"""
data_process_inter_x.py
-----------------------
从 MOOCCubeX.inter 完整交互文件处理数据，产出与 train_x.py 兼容的输出。

与 data_process_x.py 的区别:
  - 直接读取 .inter 文件 (1180万条真实交互，带真实时间戳)
  - 不再使用合成时间戳
  - 复用已有的 content_emb.pt (按 LabelEncoder 字母序排列)
  - 过滤到已有内容嵌入的 1,173 个课程

输出目录: ./processed_data_x/
  - stream_data.pkl   (交互数据, 真实时间戳)
  - meta.json         (元信息)
  - llm_scores.pkl    (空占位)
  - content_emb.pt    (不修改, 直接复用)
"""

import pandas as pd
import numpy as np
import torch
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ================= 配置 =================
INTER_FILE = "./MOOCCubeX/MOOCCubeX.inter"
OUTPUT_DIR = "./processed_data_x"
EXISTING_EMB = os.path.join(OUTPUT_DIR, "content_emb.pt")

# 从已有数据中提取 1173 个有效课程 ID
# (与 content_emb.pt 的 LabelEncoder 字母序对应)
EXISTING_PKL = os.path.join(OUTPUT_DIR, "stream_data.pkl")

# 最少交互次数 (过滤低活跃用户)
MIN_USER_INTERACTIONS = 2
# =========================================


def get_valid_items():
    """从已有 stream_data.pkl 中提取 1173 个有效课程 ID"""
    if os.path.exists(EXISTING_PKL):
        df = pd.read_pickle(EXISTING_PKL)
        items = sorted(df['course_id'].unique())
        print(f"   从已有 pkl 提取 {len(items)} 个有效课程 ID")
        return items

    # 如果没有 pkl，从 course.json 提取
    course_file = "./MOOCCubeX/entities/course.json"
    vid2cid = {}
    ccid2vid = {}
    course_ids = set()

    # 加载 video-course 映射
    ccid_map = "./MOOCCubeX/relations/video_id-ccid.txt"
    if os.path.exists(ccid_map):
        with open(ccid_map, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ccid2vid[parts[1]] = parts[0]

    if os.path.exists(course_file):
        with open(course_file, 'r', encoding='utf-8') as f:
            first = f.read(1)
            f.seek(0)
            if first == '[':
                courses = json.load(f)
            else:
                courses = [json.loads(line) for line in f]
        for c in courses:
            cid = c.get('id')
            if cid:
                course_ids.add(cid)

    print(f"   从 course.json 提取 {len(course_ids)} 个课程 ID")
    return sorted(course_ids)


def main():
    print("=" * 60)
    print("MOOCCubeX 完整交互数据处理 (.inter → processed_data_x)")
    print("=" * 60)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 获取有效课程集合
    print("\n[1/5] 获取有效课程集合...")
    valid_items = set(get_valid_items())
    print(f"   有效课程数: {len(valid_items)}")

    # 2. 读取 .inter 文件
    print(f"\n[2/5] 读取 {INTER_FILE}...")
    records = []
    skipped_items = 0
    total_lines = 0

    with open(INTER_FILE, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in tqdm(f, desc="读取交互"):
            total_lines += 1
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            user_id, item_id, timestamp = parts[0], parts[1], float(parts[2])

            if item_id not in valid_items:
                skipped_items += 1
                continue

            records.append({
                'user_id': user_id,
                'course_id': item_id,
                'timestamp': timestamp
            })

    print(f"   总行数: {total_lines}")
    print(f"   保留记录: {len(records)}")
    print(f"   过滤掉 (课程不在有效集): {skipped_items}")

    if len(records) == 0:
        print("ERROR: 没有有效记录!")
        return

    df = pd.DataFrame(records)

    # 3. 去重 + 过滤
    print(f"\n[3/5] 去重与过滤...")
    # 每个 (user, course) 保留最早的一次交互
    df_clean = df.groupby(['user_id', 'course_id'])['timestamp'].min().reset_index()
    print(f"   去重后: {len(df_clean)} 条 unique user-course pairs")

    # 过滤低活跃用户
    user_counts = df_clean['user_id'].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    df_clean = df_clean[df_clean['user_id'].isin(active_users)]
    print(f"   过滤后 (用户至少{MIN_USER_INTERACTIONS}次交互): {len(df_clean)} 条")

    # 按时间排序
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)

    # 时间范围
    dt_min = pd.to_datetime(df_clean['timestamp'].min(), unit='s')
    dt_max = pd.to_datetime(df_clean['timestamp'].max(), unit='s')
    print(f"   时间跨度: {dt_min} ~ {dt_max}")

    # 4. 编码
    print(f"\n[4/5] 编码 user/item ID...")
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df_clean['u_idx'] = user_enc.fit_transform(df_clean['user_id'])
    df_clean['i_idx'] = item_enc.fit_transform(df_clean['course_id'])

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)
    print(f"   用户数: {n_users}")
    print(f"   课程数: {n_items}")

    # 计算 popularity (每个 item 的累计出现次数)
    df_clean['popularity'] = df_clean.groupby('i_idx').cumcount()

    # 验证 item 编码与已有 content_emb.pt 一致
    if os.path.exists(EXISTING_EMB):
        emb = torch.load(EXISTING_EMB, map_location='cpu')
        if emb.shape[0] == n_items:
            print(f"   ✓ content_emb.pt 维度匹配 ({emb.shape[0]} items)")
        else:
            print(f"   ⚠ content_emb.pt 维度不匹配! 期望 {n_items}, 实际 {emb.shape[0]}")
            print(f"     需要重新生成 content_emb.pt!")

    # 5. 保存
    print(f"\n[5/5] 保存...")

    # stream_data.pkl
    pkl_path = os.path.join(OUTPUT_DIR, 'stream_data.pkl')
    df_clean.to_pickle(pkl_path)
    print(f"   stream_data.pkl: {len(df_clean)} rows")

    # meta.json
    meta = {"n_users": n_users, "n_items": n_items}
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"   meta.json: {meta}")

    # llm_scores.pkl (空占位)
    llm_path = os.path.join(OUTPUT_DIR, 'llm_scores.pkl')
    with open(llm_path, 'wb') as f:
        pickle.dump({}, f)
    print(f"   llm_scores.pkl: 空占位")

    print(f"\n   content_emb.pt: 复用已有文件 (不修改)")

    # 统计摘要
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"  交互记录  : {len(df_clean):,}")
    print(f"  用户数    : {n_users:,}")
    print(f"  课程数    : {n_items:,}")
    print(f"  时间跨度  : {dt_min.date()} ~ {dt_max.date()}")
    print(f"  月份数    : {df_clean.assign(dt=pd.to_datetime(df_clean['timestamp'], unit='s')).dt.dt.to_period('M').nunique()}")
    print(f"  输出目录  : {OUTPUT_DIR}")
    print(f"  平均交互/用户: {len(df_clean)/n_users:.1f}")
    print(f"  平均交互/课程: {len(df_clean)/n_items:.1f}")


if __name__ == "__main__":
    main()
