import json
import csv
import os
import time

# === 配置 ===
FILE_USER = '../MOOCCubeX/entities/user.json'  # 确保路径正确
OUTPUT_FILE = '../MOOCCubeX/MOOCCubeX.inter'  # 覆盖旧文件或存为新文件


# 辅助函数：将时间字符串转为 float 时间戳
def time_str_to_float(t_str):
    try:
        # MOOCCubeX 的时间格式通常为 "2019-10-12 10:28:02"
        timeArray = time.strptime(t_str, "%Y-%m-%d %H:%M:%S")
        return float(time.mktime(timeArray))
    except:
        return 0.0


print("正在生成带有时间戳的 SASRec 训练数据...")

data_list = []
count = 0

with open(FILE_USER, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            user_data = json.loads(line)
            uid = user_data.get('id')
            # 获取课程ID列表 和 选课时间列表
            course_ids = user_data.get('course_order', [])
            enroll_times = user_data.get('enroll_time', [])

            # 确保数据对齐
            if uid and course_ids and len(course_ids) == len(enroll_times):
                for i in range(len(course_ids)):
                    cid_raw = course_ids[i]
                    t_str = enroll_times[i]

                    # 格式化 ID
                    cid = f"C_{cid_raw}"
                    # 转换时间
                    ts = time_str_to_float(t_str)

                    if ts > 0:
                        data_list.append([uid, cid, ts])
                        count += 1
        except Exception as e:
            continue

print(f"共提取 {count} 条带时间戳的交互记录。")

# 写入 .inter 文件
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    # SASRec 必须包含 timestamp 列
    writer.writerow(['user_id:token', 'item_id:token', 'timestamp:float'])
    writer.writerows(data_list)

print("完成！请继续配置模型。")
