import json
import csv
import os
import random
from collections import Counter

# ================= 配置区 =================
# 输入文件路径 (请确保 user.json 路径正确)
FILE_USER = '../MOOCCubeX//entities/user.json'
OUTPUT_DIR = 'dataset/MOOCCubeX_Paper'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'MOOCCubeX_Paper.inter')

# 目标规模 (参考论文 Table 1)
TARGET_COURSE_NUM = 300  # 论文是 289
TARGET_YEAR = '2020'  # 论文只用了 2020 年
MIN_USER_INTER = 2  # 至少选修 2 门课

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("正在读取数据并进行‘论文级’过滤...")

# 1. 第一遍扫描：统计 2020 年的课程热度
course_counter = Counter()
valid_records = []  # 暂存 (user, course, time)

with open(FILE_USER, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            u_obj = json.loads(line)
            uid = u_obj.get('id')
            courses = u_obj.get('course_order', [])
            times = u_obj.get('enroll_time', [])

            if not uid or not courses or len(courses) != len(times):
                continue

            for i, t_str in enumerate(times):
                # 过滤条件1：只看 2020 年
                if t_str and t_str.startswith(TARGET_YEAR):
                    cid = f"C_{courses[i]}"
                    course_counter[cid] += 1
                    valid_records.append({
                        'uid': uid,
                        'cid': cid,
                        'time': t_str
                    })
        except:
            continue

print(f"2020年原始交互记录数: {len(valid_records)}")
print(f"2020年涉及课程总数: {len(course_counter)}")

# 2. 筛选 Top-300 热门课程
# 论文里只有 289 门课，我们取 Top 300 模拟
top_courses = set([c[0] for c in course_counter.most_common(TARGET_COURSE_NUM)])
print(f"已锁定 Top-{len(top_courses)} 热门课程。")

# 3. 第二遍过滤：只保留核心课程的交互
final_user_inters = {}  # uid -> list of cids

for record in valid_records:
    if record['cid'] in top_courses:
        uid = record['uid']
        if uid not in final_user_inters:
            final_user_inters[uid] = []
        final_user_inters[uid].append(record)

# 4. 用户过滤：剔除交互太少的用户
qualified_users = [u for u, inters in final_user_inters.items() if len(inters) >= MIN_USER_INTER]

print(f"筛选后合格用户数: {len(qualified_users)}")

# 5. (可选) 如果用户太多，随机采样到 1.5万 以接近论文的 1万
if len(qualified_users) > 15000:
    print(f"用户数量 ({len(qualified_users)}) 远超论文 (1万)，进行随机采样...")
    selected_users = set(random.sample(qualified_users, 15000))
else:
    selected_users = set(qualified_users)

# 6. 写入最终 .inter 文件
print(f"正在写入 {OUTPUT_FILE} ...")
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    # 写入 RecBole 表头 (SASRec 需要 timestamp)
    writer.writerow(['user_id:token', 'item_id:token', 'timestamp:float'])

    write_count = 0
    import time

    for uid in selected_users:
        records = final_user_inters[uid]
        for r in records:
            # 转换时间戳
            try:
                ts = time.mktime(time.strptime(r['time'], "%Y-%m-%d %H:%M:%S"))
            except:
                ts = 0
            writer.writerow([uid, r['cid'], ts])
            write_count += 1

print("=" * 30)
print(f"数据处理完成！")
print(f"最终统计:")
print(f"用户数: {len(selected_users)} (论文: 10,930)")
print(f"课程数: {len(top_courses)} (论文: 289)")
print(f"交互数: {write_count} (论文: 25,189)")
print("=" * 30)
print("请使用新的数据集路径 'dataset/MOOCCubeX_Paper' 运行模型。")
