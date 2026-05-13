import pandas as pd
import numpy as np
import torch
import os
import re
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= 配置 =================
BASE_DIR = "../MOOCCubeX"
COURSE_FILE = os.path.join(BASE_DIR, "entities/course.json")
USER_VIDEO_FILE = os.path.join(BASE_DIR, "relations/user-video.json")
CCID_MAP_FILE = os.path.join(BASE_DIR, "relations/video_id-ccid.txt")
OUTPUT_DIR = "./processed_data_video"

# 强力压缩时间 (保持稠密)
START_YEAR = 2019
END_YEAR = 2020
MIN_SEQ_LEN = 5  # 只有看视频超过5个的用户才参与训练


# =======================================

class VideoLevelProcessor:
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        self.ccid2vid = {}
        self.vid2meta = {}  # V_ID -> {text, course_id}

    def _load_json_flexible(self, file_path):
        """ 修复了读取指针的问题 """
        if not os.path.exists(file_path):
            print(f"Error: 文件不存在 {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first = f.read(1)
                f.seek(0)  # 【关键修复】必须把指针拨回去！
                if first == '[':
                    return json.load(f)
                else:
                    return [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def process(self):
        print("1. 构建视频元数据 (Video Metadata)...")
        # 1. 加载 CCID 映射
        if os.path.exists(CCID_MAP_FILE):
            with open(CCID_MAP_FILE, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="索引表"):
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.ccid2vid[p[1]] = p[0]  # CCID -> V_xxx

        # 2. 从 Course 中提取 Video 标题
        courses = self._load_json_flexible(COURSE_FILE)
        print(f"   -> 成功加载课程文件: {len(courses)} 门")

        if len(courses) == 0:
            print("❌ 严重错误: 课程文件加载为空，请检查路径！")
            return

        for c in tqdm(courses, desc="解析资源"):
            cid = c.get('id')
            c_name = c.get('name', '')
            # 提取所有资源
            for res in c.get('resource', []):
                vid = res.get('resource_id')
                titles = res.get('titles', [])
                # 组合文本: 课程名 + 章节名 + 视频名
                v_title = " ".join([str(t) for t in titles if t])
                full_text = f"{c_name} {v_title}".strip()

                if vid:
                    self.vid2meta[vid] = {'text': full_text, 'cid': cid}

        print(f"   -> 视频库规模: {len(self.vid2meta)}")

        if len(self.vid2meta) == 0:
            print("❌ 严重错误: 没有提取到任何视频元数据，无法继续！")
            return

        # 3. 处理日志
        print("2. 解析日志 (Video Level)...")
        raw_data = []
        start_ts = float(pd.Timestamp(f"{START_YEAR}-01-01").timestamp())
        end_ts = float(pd.Timestamp(f"{END_YEAR}-12-31").timestamp())

        with open(USER_VIDEO_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting"):
                try:
                    obj = json.loads(line)
                    uid = obj.get('user_id')
                    seq = obj.get('seq', [])

                    if not uid or len(seq) < MIN_SEQ_LEN: continue

                    # 随机入学时间
                    base_time = np.random.randint(start_ts, end_ts - 86400 * 30)

                    user_interactions = []
                    for idx, item in enumerate(seq):
                        # 兼容不同格式
                        raw_id = None
                        if isinstance(item, str):
                            raw_id = item
                        elif isinstance(item, dict):
                            raw_id = item.get('video_id') or item.get('ccid') or item.get('id')

                        if not raw_id: continue

                        # ID 翻译: Raw -> V_xxx
                        vid = self.ccid2vid.get(raw_id, raw_id)

                        # 必须是“已知视频”才保留 (否则没有 Embedding)
                        if vid not in self.vid2meta: continue

                        # 模拟时间 (分钟级间隔)
                        ts = base_time + idx * 60 + np.random.randint(0, 30)
                        user_interactions.append({'u': uid, 'i': vid, 'ts': ts})

                    if len(user_interactions) >= MIN_SEQ_LEN:
                        raw_data.extend(user_interactions)
                except:
                    continue

        df = pd.DataFrame(raw_data)
        print(f"   -> 有效交互数: {len(df)}")

        if len(df) == 0:
            print("❌ 错误: 匹配后数据为 0。可能原因: user-video 里的 ID 和 course 里的 ID 还是对不上。")
            return

        print(f"   -> 覆盖用户数: {df['u'].nunique()}")

        # 4. 编码与保存
        print("3. 编码与 BERT 特征提取...")
        df['u_idx'] = self.user_enc.fit_transform(df['u'])
        df['i_idx'] = self.item_enc.fit_transform(df['i'])
        df['popularity'] = df.groupby('i_idx')['i_idx'].transform('count')

        # 准备 BERT 文本 (按 i_idx 顺序)
        all_vids = self.item_enc.inverse_transform(range(len(self.item_enc.classes_)))
        texts = [self.vid2meta[v]['text'] for v in all_vids]

        # 提取 BERT
        print("   -> Running BERT...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        model = AutoModel.from_pretrained('bert-base-chinese')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        embs = []
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
            batch = texts[i:i + batch_size]
            # 截断以加快速度
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state[:, 0, :].cpu()
                embs.append(out)
        content_emb = torch.cat(embs, dim=0)

        # 保存
        df.to_pickle(os.path.join(OUTPUT_DIR, 'stream_data.pkl'))
        torch.save(content_emb, os.path.join(OUTPUT_DIR, 'content_emb.pt'))

        # 保存视频元数据 (供 LLM 打分脚本使用)
        with open(os.path.join(OUTPUT_DIR, 'video_meta.pkl'), 'wb') as f:
            pickle.dump(self.vid2meta, f)

        meta = {"n_users": len(self.user_enc.classes_), "n_items": len(self.item_enc.classes_)}
        with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        print(f"Done! 视频级数据已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    VideoLevelProcessor().process()