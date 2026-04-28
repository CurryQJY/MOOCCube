import json
import csv
import os

# ================= 配置路径 =================
# 请确保这三个文件名与您实际存放的文件名一致
FILE_USER = '../MOOCCubeX/entities/user.json'
FILE_COURSE = '../MOOCCubeX/entities/course.json'
FILE_USER_VIDEO = '../MOOCCubeX/relations/user-video.json'

OUTPUT_DIR = '../MOOCCubeX'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'MOOCCubeX.inter')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 1. 建立 Video -> Course 的映射 =================
# LightGCN 推荐的是课程，所以我们需要把视频观看记录映射回课程 ID
print(f"正在读取 {FILE_COURSE} 以构建映射关系...")
video2course = {}
valid_course_ids = set()

try:
    with open(FILE_COURSE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                course_id = data.get('id')  # e.g., "C_584313"

                if not course_id: continue

                valid_course_ids.add(course_id)

                # 提取该课程下的所有视频资源
                resources = data.get('resource', [])
                for res in resources:
                    vid = res.get('resource_id')
                    if vid:
                        video2course[vid] = course_id
            except json.JSONDecodeError:
                continue
    print(f"映射构建完成：包含 {len(valid_course_ids)} 门课程，{len(video2course)} 个视频关系。")

except FileNotFoundError:
    print(f"警告：找不到 {FILE_COURSE}，将无法利用视频观看数据进行增强，仅使用 user.json。")

# ================= 2. 提取交互数据 (去重) =================
interactions = set()  # 使用集合自动去重 (user_id, course_id)
count_enroll = 0
count_view = 0

# --- 处理显式选课 (user.json) ---
print(f"正在处理 {FILE_USER} (选课记录)...")
try:
    with open(FILE_USER, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                user_data = json.loads(line)
                uid = user_data.get('id')
                # user.json 中的 course_order 是整数列表 [682129, ...]
                course_list = user_data.get('course_order', [])

                if uid and course_list:
                    for cid_raw in course_list:
                        # 统一 ID 格式：将数字转换为 "C_xxxx" 格式以匹配 course.json
                        cid_str = f"C_{cid_raw}"
                        interactions.add((uid, cid_str))
                        count_enroll += 1
            except:
                continue
except FileNotFoundError:
    print(f"错误：找不到 {FILE_USER}，这是必需文件！")

# --- 处理隐式观看 (user-video.json) [数据增强] ---
# 如果您想复现最强效果，这一步很关键
print(f"正在处理 {FILE_USER_VIDEO} (观看记录)...")
try:
    with open(FILE_USER_VIDEO, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                uid = log_data.get('user_id')
                seq = log_data.get('seq', [])

                if uid and seq:
                    for item in seq:
                        vid = item.get('video_id')
                        # 通过之前的字典映射回课程 ID
                        if vid in video2course:
                            cid = video2course[vid]
                            interactions.add((uid, cid))
                            count_view += 1
            except:
                continue
except FileNotFoundError:
    print("提示：未找到视频日志，跳过数据增强步骤。")

# ================= 3. 写入 RecBole 格式文件 =================
print(f"正在写入数据至 {OUTPUT_FILE} ...")
print(f"统计：原始选课 {count_enroll} 条，视频观看映射 {count_view} 条。")
print(f"去重后最终交互对数量：{len(interactions)}")

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    # 写入表头：Token表示这是离散ID
    writer.writerow(['user_id:token', 'item_id:token'])
    for uid, cid in interactions:
        writer.writerow([uid, cid])

print("数据预处理完成！")
