import pandas as pd
import numpy as np
import torch
import os
import re
import json
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class DataProcessor:
    def __init__(self, inter_file, course_file, output_dir='./processed_data'):
        self.inter_file = inter_file
        self.course_file = course_file
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _load_course_metadata(self):
        """
        加载 course.json，提取 {course_id: text} 映射。
        Text = name + about
        """
        print(f"1. 正在加载课程元数据: {self.course_file} ...")
        course_texts = {}

        # 统计读取行数
        count = 0
        with open(self.course_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析课程JSON"):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    cid = data.get('id')
                    # 拼接 标题 + 简介 (处理可能的 None 值)
                    name = data.get('name', '') or ''
                    # about 可能是 HTML，简单去除标签能更好，但直接拼也不影响 BERT
                    about = data.get('about', '') or ''
                    # 简单清洗 HTML 标签 (可选)
                    about = re.sub(r'<[^>]+>', '', about)

                    text = f"{name} {about}".strip()
                    if cid and text:
                        course_texts[cid] = text
                    count += 1
                except json.JSONDecodeError:
                    continue

        print(f"   成功加载 {len(course_texts)} 门课程的元数据。")
        return course_texts

    def _extract_bert_features(self, texts, batch_size=32):
        """
        使用中文 BERT 提取特征
        """
        print("   正在加载 BERT 模型 (bert-base-chinese)...")
        try:
            # 针对 MOOCCube 的中文内容，使用中文 BERT
            tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            model = AutoModel.from_pretrained('bert-base-chinese')
        except:
            print("   无法下载 bert-base-chinese，尝试使用 bert-base-uncased...")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        all_embs = []
        print(f"   正在提取特征 (Device: {device})...")

        # 不计算梯度，节省显存
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
                batch = texts[i: i + batch_size]
                # 截断长度设为 128 (涵盖大部分简介核心)
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # 取 [CLS] token (Batch, 768)
                emb = outputs.last_hidden_state[:, 0, :].cpu()
                all_embs.append(emb)

        return torch.cat(all_embs, dim=0)

    def _extract_time_from_course_id(self, course_id):
        """ 从课程 ID 解析时间戳 (例如 _2015_T1) """
        cid_str = str(course_id)
        match = re.search(r'(20[0-2]\d)', cid_str)
        if match:
            year = int(match.group(1))
            base_time = (year - 1970) * 31536000
            # 加上随机偏移防止时间重叠
            return base_time + np.random.randint(0, 3600 * 24 * 30)
        else:
            # 默认 2017 年
            return 1483228800 + np.random.randint(0, 31536000)

    def process(self):
        # --- 1. 加载课程文本 ---
        course_texts = self._load_course_metadata()
        valid_course_ids = set(course_texts.keys())

        # --- 2. 加载交互数据 (TSV) ---
        print(f"2. 正在加载交互数据: {self.inter_file} ...")
        # 直接按 TSV 读取
        df = pd.read_csv(
            self.inter_file,
            sep='\t',
            header=None,
            names=['user_id', 'course_id', 'raw_time'],
            on_bad_lines='skip'
        )

        original_len = len(df)
        # 过滤掉没有元数据的课程交互
        df = df[df['course_id'].isin(valid_course_ids)]
        print(f"   过滤无效课程交互: {original_len} -> {len(df)}")

        if len(df) == 0:
            raise ValueError("错误: 交互数据中的 course_id 与 course.json 均不匹配！")

        # --- 3. 时间戳处理 ---
        # 尝试从 ID 解析
        print("3. 生成流式时间戳...")
        tqdm.pandas(desc="解析时间")
        df['timestamp'] = df['course_id'].progress_apply(self._extract_time_from_course_id)

        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)

        # --- 4. ID 编码 ---
        print("4. 编码 ID...")
        df['u_idx'] = self.user_enc.fit_transform(df['user_id'])
        df['i_idx'] = self.item_enc.fit_transform(df['course_id'])

        n_users = len(self.user_enc.classes_)
        n_items = len(self.item_enc.classes_)

        # --- 5. 生成 Content Embedding ---
        print("5. 生成 Content Embedding...")
        # 关键步骤：必须按 item_enc 编码后的 0, 1, 2... 顺序排列文本
        sorted_cids = self.item_enc.inverse_transform(range(n_items))
        sorted_texts = [course_texts[cid] for cid in sorted_cids]

        content_emb = self._extract_bert_features(sorted_texts)
        print(f"   Embedding Shape: {content_emb.shape}")  # 应该是 [n_items, 768]

        # --- 6. 计算流行度 ---
        print("6. 计算流式流行度...")
        df['popularity'] = df.groupby('i_idx').cumcount()

        # --- 7. 保存 ---
        print("7. 保存数据...")
        df.to_pickle(os.path.join(self.output_dir, 'stream_data.pkl'))
        torch.save(content_emb, os.path.join(self.output_dir, 'content_emb.pt'))

        meta = {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "content_dim": int(content_emb.shape[1])  # 768
        }
        with open(os.path.join(self.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        print(f"\n[完成] 数据处理完毕！输出目录: {self.output_dir}")


if __name__ == "__main__":
    # 请确认文件名是否正确
    INTER_FILE = "./MOOCCube/relations/user-course.json"
    COURSE_FILE = "./MOOCCube/entities/course.json"

    if os.path.exists(INTER_FILE) and os.path.exists(COURSE_FILE):
        processor = DataProcessor(INTER_FILE, COURSE_FILE)
        processor.process()
    else:
        print(f"错误: 请确保当前目录下存在 {INTER_FILE} 和 {COURSE_FILE}")
