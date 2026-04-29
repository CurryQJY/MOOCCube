import pandas as pd
import pickle
import os
import json

DATA_DIR = "processed_data_video"
META_FILE = os.path.join(DATA_DIR, "video_meta.pkl")  # 之前生成的 meta
CLEAN_META = os.path.join(DATA_DIR, "meta_5core.pkl")  # 5-core meta
OUT_MAP = os.path.join(DATA_DIR, "vid2cid_map.pkl")


def main():
    # 1. 加载视频元数据 (V_ID -> {cid: C_ID})
    with open(META_FILE, 'rb') as f:
        v_meta = pickle.load(f)

    # 2. 加载清洗后的 ID 映射 (i_idx -> V_ID)
    with open(CLEAN_META, 'rb') as f:
        clean_meta = pickle.load(f)
    i_map_rev = {v: k for k, v in clean_meta['i_map'].items()}  # i_idx -> V_Raw

    # 3. 构建 i_idx -> c_idx 映射
    # 先收集所有课程 ID 并编码
    all_cids = set()
    temp_map = {}  # i_idx -> raw_cid

    for idx in range(clean_meta['n_items']):
        raw_vid = i_map_rev[idx]
        if raw_vid in v_meta:
            cid = v_meta[raw_vid]['cid']
            all_cids.add(cid)
            temp_map[idx] = cid

    # 课程编码
    c_list = sorted(list(all_cids))
    cid2idx = {c: i for i, c in enumerate(c_list)}

    # 生成最终 tensor 映射表
    # i2c_map[item_idx] = course_idx
    i2c_map = {}
    for i_idx, raw_cid in temp_map.items():
        i2c_map[i_idx] = cid2idx[raw_cid]

    print(f"提取课程关系: {len(i2c_map)} 个视频归属于 {len(cid2idx)} 门课程")

    data = {
        'n_courses': len(cid2idx),
        'i2c': i2c_map
    }
    with open(OUT_MAP, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()