import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ================= 配置 =================
DATA_DIR = '../MOOCCubeX'
INTERACTION_FILE = '../MOOCCubeX/relations/user-video.json'  # 你的日志文件名
COURSE_FILE = '../MOOCCubeX/entities/course.json'
OUTPUT_CSV = os.path.join(DATA_DIR, 'MOOCCubeX.csv')
OUTPUT_CONTENT = os.path.join(DATA_DIR, 'MOOCCubeX_item_content.npy')
MIN_INTERACTION = 5  # 过滤少于5次交互的用户


def load_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def process():
    print("1. Loading Course Metadata...")
    # --- 1. 建立 Video -> Course 映射 ---
    video2course = {}
    course_texts = {}  # {course_raw_id: "title + desc"}

    with open(os.path.join(DATA_DIR, COURSE_FILE), 'r', encoding='utf-8') as f:
        # course.json 通常是一个大的 json list，或者 json lines
        # 根据你的 snippet，它看起来像 json lines
        for line in f:
            c_data = json.loads(line)
            cid = c_data['id']
            # 提取文本用于 BERT
            name = c_data.get('name', '')
            about = c_data.get('about', '') or ''
            course_texts[cid] = f"{name} {about}".strip()

            # 建立视频到课程的索引
            if 'resource' in c_data:
                for res in c_data['resource']:
                    # 你的 snippet 显示 resource_id 类似 "V_xxx"
                    vid = res.get('resource_id')
                    if vid:
                        video2course[vid] = cid

    print(f"   Loaded {len(course_texts)} courses, {len(video2course)} videos.")

    # --- 2. 处理交互日志 (Video View -> Course View) ---
    print("2. Processing User Interactions...")
    user_course_pairs = set()

    # 读取 user_video.json
    raw_iter = load_json_lines(os.path.join(DATA_DIR, INTERACTION_FILE))

    for u_data in tqdm(raw_iter):
        raw_uid = u_data['user_id']

        # 遍历该用户看过的所有视频序列
        for seq_item in u_data.get('seq', []):
            vid = seq_item.get('video_id')

            # 找到该视频属于哪个课程
            if vid in video2course:
                cid = video2course[vid]
                user_course_pairs.add((raw_uid, cid))

    print(f"   Found {len(user_course_pairs)} raw user-course interactions.")

    # --- 3. ID Remapping (映射到 0~N) ---
    print("3. Remapping IDs...")
    # 转为 DataFrame 处理更方便
    df = pd.DataFrame(list(user_course_pairs), columns=['raw_uid', 'raw_cid'])

    # 过滤冷门课程/用户 (可选)
    # user_counts = df['raw_uid'].value_counts()
    # valid_users = user_counts[user_counts >= MIN_INTERACTION].index
    # df = df[df['raw_uid'].isin(valid_users)]

    # 生成映射字典
    unique_users = df['raw_uid'].unique()
    unique_courses = df['raw_cid'].unique()

    user2id = {u: i for i, u in enumerate(unique_users)}
    course2id = {c: i for i, c in enumerate(unique_courses)}

    df['user'] = df['raw_uid'].map(user2id)
    df['item'] = df['raw_cid'].map(course2id)

    # 保存 CSV (split.py 需要的格式)
    df[['user', 'item']].to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved interactions to {OUTPUT_CSV}. Users: {len(user2id)}, Items: {len(course2id)}")

    # --- 4. 提取 BERT 特征 ---
    print("4. Extracting BERT Features (This may take a while)...")

    # 按 ID 顺序准备文本
    # 注意：unique_courses 里的顺序就是 course2id 的顺序吗？
    # course2id 是 enumerate(unique_courses) 生成的，所以 unique_courses[i] 就是 ID 为 i 的课程
    ordered_texts = []
    for raw_cid in unique_courses:
        # 如果日志里的课程在 course.json 没找到文本，给个空字符
        ordered_texts.append(course_texts.get(raw_cid, "Unknown Course"))

    # 加载 BERT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # MOOC通常是中文
    model = BertModel.from_pretrained('bert-base-chinese').to(device)
    model.eval()

    embs = []
    batch_size = 32

    for i in tqdm(range(0, len(ordered_texts), batch_size)):
        batch_texts = ordered_texts[i: i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            # 取 [CLS] token 作为句向量
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls_emb)

    final_embs = np.concatenate(embs, axis=0)
    np.save(OUTPUT_CONTENT, final_embs)
    print(f"   Saved content features to {OUTPUT_CONTENT}. Shape: {final_embs.shape}")


if __name__ == '__main__':
    process()
