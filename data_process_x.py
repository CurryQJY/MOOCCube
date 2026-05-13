import pandas as pd
import numpy as np
import torch
import os
import re
import json
import random
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = "./MOOCCubeX"
COURSE_FILE = os.path.join(BASE_DIR, "entities/course.json")
USER_VIDEO_FILE = os.path.join(BASE_DIR, "relations/user-video.json")
CCID_MAP_FILE = os.path.join(BASE_DIR, "relations/video_id-ccid.txt")
OUTPUT_DIR = "./processed_data_x"

# 【强力压缩配置】
START_YEAR = 2019
END_YEAR = 2020


# ===========================================

class MOOCCubeXCompressedProcessor:
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        self.vid2cid = {}
        self.ccid2vid = {}
        self.course_texts = {}

    def _load_json_flexible(self, file_path):
        if not os.path.exists(file_path): return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if f.read(1) == '[':
                    f.seek(0)
                    return json.load(f)
                else:
                    f.seek(0)
                    return [json.loads(line) for line in f]
        except:
            return []

    def _build_mapping(self):
        print("1. 构建映射...")
        if os.path.exists(CCID_MAP_FILE):
            with open(CCID_MAP_FILE, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="索引表"):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: self.ccid2vid[parts[1]] = parts[0]

        courses = self._load_json_flexible(COURSE_FILE)
        for course in tqdm(courses, desc="课程表"):
            cid = course.get('id')
            name = course.get('name', '')
            desc = course.get('about', '') or course.get('intro', '') or ''
            full_text = re.sub(r'<[^>]+>', '', f"{name} {desc}".strip())[:512]
            if cid: self.course_texts[cid] = full_text
            for res in course.get('resource', []):
                vid = res.get('resource_id')
                if vid and cid: self.vid2cid[vid] = cid

    def _extract_bert(self, texts, batch_size=64):
        print("   提取 BERT 特征...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        model = AutoModel.from_pretrained('bert-base-chinese')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        res = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch = [t if t and t.strip() else "未知课程" for t in batch]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                res.append(model(**inputs).last_hidden_state[:, 0, :].cpu())
        return torch.cat(res, dim=0)

    def process(self):
        self._build_mapping()
        print(f"2. 解析日志 (强力压缩至 {START_YEAR}-{END_YEAR})...")

        valid_data = []

        # 定义绝对边界
        start_ts = float(pd.Timestamp(f"{START_YEAR}-01-01").timestamp())
        end_ts = float(pd.Timestamp(f"{END_YEAR}-12-31").timestamp())

        # 调试计数器
        out_of_bound_fixed = 0

        with open(USER_VIDEO_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理中"):
                try:
                    obj = json.loads(line)
                    user_id = obj.get('user_id')
                    seq = obj.get('seq', [])

                    if not user_id or not seq: continue

                    # 随机分配入学时间 (保证在区间内)
                    user_base_time = np.random.randint(start_ts, end_ts - 86400)  # 留一天余量

                    for idx, item in enumerate(seq):
                        raw_id = None
                        if isinstance(item, str):
                            raw_id = item
                        elif isinstance(item, dict):
                            raw_id = item.get('video_id') or item.get('ccid') or item.get('id')

                        if not raw_id: continue

                        final_cid = None
                        if raw_id in self.ccid2vid:
                            final_cid = self.vid2cid.get(self.ccid2vid[raw_id])
                        elif raw_id in self.vid2cid:
                            final_cid = self.vid2cid[raw_id]

                        if final_cid:
                            # === 强力压缩逻辑 ===
                            # 1. 间隔改为 60 秒 (模拟一口气刷课)
                            # 2. 增加一点点微秒级抖动防止完全重复
                            offset = (idx * 60) + np.random.randint(0, 30)
                            simulated_ts = user_base_time + offset

                            # === 硬性截断 ===
                            # 如果算出来的时间超过了 end_ts，强制设为 end_ts 附近的时间
                            if simulated_ts > end_ts:
                                # 倒退回 end_ts 前的一段随机时间
                                simulated_ts = end_ts - np.random.randint(100, 10000)
                                out_of_bound_fixed += 1

                            valid_data.append({
                                'user_id': user_id,
                                'course_id': final_cid,
                                'timestamp': simulated_ts
                            })
                except:
                    continue

        df = pd.DataFrame(valid_data)
        print(f"   生成记录: {len(df)}")
        print(f"   触发边界强行修复次数: {out_of_bound_fixed}")

        if len(df) == 0: return

        print("3. 聚合去重...")
        df['timestamp'] = df['timestamp'].astype(float)
        df_clean = df.groupby(['user_id', 'course_id'])['timestamp'].min().reset_index()
        df_clean = df_clean.sort_values('timestamp')

        # 验证时间
        dt_min = pd.to_datetime(df_clean['timestamp'].min(), unit='s')
        dt_max = pd.to_datetime(df_clean['timestamp'].max(), unit='s')
        print(f"   >>> 最终时间跨度: {dt_min} 至 {dt_max}")

        print("4. 编码与保存...")
        df_clean['u_idx'] = self.user_enc.fit_transform(df_clean['user_id'])
        df_clean['i_idx'] = self.item_enc.fit_transform(df_clean['course_id'])

        final_cids = self.item_enc.inverse_transform(range(len(self.item_enc.classes_)))
        final_texts = [self.course_texts.get(cid, "未知课程") for cid in final_cids]

        content_emb = self._extract_bert(final_texts)
        df_clean['popularity'] = df_clean.groupby('i_idx').cumcount()

        df_clean.to_pickle(os.path.join(self.output_dir, 'stream_data.pkl'))
        torch.save(content_emb, os.path.join(self.output_dir, 'content_emb.pt'))

        meta = {"n_users": len(self.user_enc.classes_), "n_items": len(self.item_enc.classes_)}
        with open(os.path.join(self.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        print(f"Done! 强力压缩完成。")


if __name__ == "__main__":
    MOOCCubeXCompressedProcessor().process()