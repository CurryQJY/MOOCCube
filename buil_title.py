import pandas as pd
import json
import os

# ================= 配置 =================
# 1. 您的处理后数据（用于获取 Integer ID -> Course ID 的关系）
STREAM_PATH = "processed_data/stream_data.pkl"

# 2. MOOCCube 原始课程文件 (请确认您的路径)
COURSE_JSON_PATH = "MOOCCube/entities/course.json"
# 或者 D:/Datasets/Mooccube/course.json

# 3. 输出文件
OUTPUT_PATH = "processed_data/course_titles.json"


# =======================================

def build_mapping():
    print(f"📂 Loading stream data from {STREAM_PATH}...")
    if not os.path.exists(STREAM_PATH):
        print("❌ Stream data not found!")
        return

    df = pd.read_pickle(STREAM_PATH)

    # 1. 建立 item_id (Int) -> course_id (String) 的映射
    #    例如: 105 -> "C_course-v1:TsinghuaX+..."

    # 自动识别列名
    if 'i_idx' in df.columns:
        item_col = 'i_idx'
    elif 'item_id' in df.columns:
        item_col = 'item_id'
    else:
        print("❌ Error: Cannot find item_id column in dataframe")
        return

    if 'course_id' not in df.columns:
        print("❌ Error: Cannot find course_id column in dataframe (needed for linking)")
        return

    # 去重提取映射关系
    id_map_df = df[[item_col, 'course_id']].drop_duplicates(subset=[item_col])
    int_to_str_id = dict(zip(id_map_df[item_col], id_map_df['course_id']))

    print(f"✅ Loaded {len(int_to_str_id)} items from stream data.")

    # -------------------------------------------------------
    print(f"📂 Loading titles from {COURSE_JSON_PATH}...")
    if not os.path.exists(COURSE_JSON_PATH):
        print(f"❌ Error: {COURSE_JSON_PATH} does not exist!")
        return

    course_name_map = {}

    # MOOCCube 的 course.json 通常是一个大的 JSON 列表，或者是按行存储的 JSON 对象
    # 我们写得健壮一点，支持两种格式
    try:
        with open(COURSE_JSON_PATH, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # 尝试情况 A: 整个文件是一个大的 JSON 列表 [{}, {}]
            if content.startswith('['):
                print("   Detected JSON List format...")
                data_list = json.loads(content)
                for item in data_list:
                    if 'id' in item and 'name' in item:
                        course_name_map[item['id']] = item['name']
            else:
                # 尝试情况 B: JSON Lines (每行一个对象)
                print("   Detected JSON Lines format...")
                f.seek(0)  # 回到文件头
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'id' in item and 'name' in item:
                            course_name_map[item['id']] = item['name']
                    except:
                        continue

    except Exception as e:
        print(f"❌ Error reading course.json: {e}")
        return

    print(f"✅ Loaded {len(course_name_map)} raw course titles from course.json.")

    # -------------------------------------------------------
    # 3. 合并： int_id -> string_id -> title
    final_titles = {}
    match_count = 0

    for int_id, str_id in int_to_str_id.items():
        if str_id in course_name_map:
            final_titles[int(int_id)] = course_name_map[str_id]
            match_count += 1
        else:
            # 没找到名字，为了防止报错，填入 ID 作为兜底
            # 也有可能 str_id 需要去掉前缀 (比如去掉 "C_")，您可以根据实际情况调整
            # 尝试去掉 "C_" 前缀再找一次
            clean_id = str_id.replace("C_", "")
            if clean_id in course_name_map:
                final_titles[int(int_id)] = course_name_map[clean_id]
                match_count += 1
            else:
                final_titles[int(int_id)] = f"Course {str_id}"

    print(f"🎉 Matched {match_count} / {len(int_to_str_id)} titles!")

    if match_count == 0:
        print("⚠️ Warning: 0 matches found! Please check if IDs in stream_data match IDs in course.json.")
        print(f"   Example Stream ID: {list(int_to_str_id.values())[0]}")
        print(f"   Example JSON ID:   {list(course_name_map.keys())[0]}")

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_titles, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved titles to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_mapping()