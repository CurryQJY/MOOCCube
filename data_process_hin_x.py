"""
data_process_hin_x.py
---------------------
MOOCCubeX 版本的知识图谱增强数据预处理脚本。
适配 MOOCCubeX 数据格式，产出与 data_process_hin.py 完全兼容的输出。

与 MOOCCube 版本的主要差异:
  - user-course 交互直接从 user.json 的 course_order + enroll_time 提取
  - concept-course 关系从 concept-course.txt 反转读取
  - 无 teacher-course / school-course 关系文件，改用 course.json 的 field 字段增强
  - 同时生成 MOOCCubeX/relations/course-concept.json 供 build_course_artifacts() 使用

输出目录: ./processed_data_hin_x/
  - stream_data.pkl   (交互数据, 与原版格式完全一致)
  - content_emb.pt    (知识增强的 BERT [CLS] embedding, [N_items, 768])
  - meta.json         (元信息)
  - llm_scores.pkl    (空占位, 供模型加载)
"""

import pandas as pd
import numpy as np
import torch
import os
import re
import json
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ==============================
# 1. MOOCCubeX 数据加载器
# ==============================

def load_concept_course_reversed(filepath):
    """
    加载 MOOCCubeX 的 concept-course.txt (每行: concept_id\tcourse_id)。
    返回 course -> [concept1, concept2, ...] 的映射 (反转方向)。
    """
    mapping = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                concept_id, course_id = parts[0], parts[1]
                mapping[course_id].append(concept_id)
    return mapping


def load_concept_names(filepath):
    """
    加载 MOOCCubeX concept.json, 返回 {concept_id: name} 映射。
    """
    names = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cid = data.get('id', '')
                name = data.get('name', '') or ''
                if cid and name:
                    names[cid] = name
            except json.JSONDecodeError:
                continue
    return names


def generate_course_concept_tsv(concept_course_mapping, output_path):
    """
    从 course -> [concepts] 映射生成兼容 MOOCCube 格式的 course-concept.json (TSV)。
    每行: course_id\tconcept_id
    供 build_course_artifacts() 使用。
    """
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for course_id, concepts in concept_course_mapping.items():
            for concept_id in concepts:
                f.write(f"{course_id}\t{concept_id}\n")
                count += 1
    return count


# ==============================
# 2. MOOCCubeX HIN 数据处理器
# ==============================

class HINDataProcessorX:
    def __init__(self, base_dir='./MOOCCubeX', output_dir='./processed_data_hin_x'):
        self.base_dir = base_dir
        self.output_dir = output_dir

        # 文件路径
        self.user_file = os.path.join(base_dir, 'entities', 'user.json')
        self.course_file = os.path.join(base_dir, 'entities', 'course.json')
        self.concept_course_file = os.path.join(base_dir, 'relations', 'concept-course.txt')
        self.concept_entity_file = os.path.join(base_dir, 'entities', 'concept.json')

        # 兼容输出: 生成 course-concept.json 供模型使用
        self.compat_relation_dir = os.path.join(base_dir, 'relations')
        self.compat_course_concept_file = os.path.join(self.compat_relation_dir, 'course-concept.json')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _load_graph_features(self):
        """
        加载 MOOCCubeX 知识图谱数据。
        """
        print("  [HIN-X] 加载 concept-course 关系 (反转为 course->concepts)...")
        course_concepts = load_concept_course_reversed(self.concept_course_file)
        print(f"           覆盖 {len(course_concepts)} 门课程")

        # 生成兼容 MOOCCube 格式的 course-concept.json
        print(f"  [HIN-X] 生成兼容文件: {self.compat_course_concept_file}")
        n_pairs = generate_course_concept_tsv(course_concepts, self.compat_course_concept_file)
        print(f"           写入 {n_pairs} 对 course-concept 关系")

        print("  [HIN-X] 加载概念名称...")
        concept_names = load_concept_names(self.concept_entity_file)
        print(f"           概念数: {len(concept_names)}")

        return course_concepts, concept_names

    def _build_enriched_text(self, cid, name, about, fields, course_concepts, concept_names):
        """
        构建知识增强文本。与 MOOCCube 版本相比:
        - 用 course.json 自带的 field 替代 school/teacher
        - 保留知识概念增强

        模板:
            [学科领域] 计算机科学与技术, 数学
            [知识概念] 数据结构, 二叉树, 算法
            [课程] 数据结构基础
            [简介] 本课程主要讲解...
        """
        parts = []

        # 学科领域 (MOOCCubeX course.json 自带)
        if fields:
            parts.append(f"[学科领域] {'，'.join(fields)}")

        # 知识概念 (取前15个)
        concepts = course_concepts.get(cid, [])
        if concepts:
            c_names = []
            for c in concepts[:15]:
                # concept_names 里有就用名称, 否则从 ID 提取
                if c in concept_names:
                    c_names.append(concept_names[c])
                else:
                    c_clean = c.replace('K_', '')
                    c_parts = c_clean.split('_')
                    c_names.append(c_parts[0] if c_parts else c_clean)
            parts.append(f"[知识概念] {'，'.join(c_names)}")

        # 课程名称
        if name:
            parts.append(f"[课程] {name}")

        # 简介 (清洗 HTML)
        if about:
            about_clean = re.sub(r'<[^>]+>', '', about).strip()
            parts.append(f"[简介] {about_clean}")

        return ' '.join(parts) if parts else name or '未知课程'

    def _load_course_metadata_enriched(self, course_concepts, concept_names):
        """
        加载 MOOCCubeX course.json 并构建知识增强文本。
        同时收集 video_id -> course_id 映射。
        """
        print(f"1. 正在加载课程元数据 (知识增强): {self.course_file} ...")
        course_texts = {}

        with open(self.course_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析课程JSON"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cid = data.get('id')
                    name = data.get('name', '') or ''
                    about = data.get('about', '') or ''
                    fields = data.get('field', []) or []

                    text = self._build_enriched_text(
                        cid, name, about, fields,
                        course_concepts, concept_names
                    )

                    if cid and text:
                        course_texts[cid] = text
                except json.JSONDecodeError:
                    continue

        print(f"   成功加载 {len(course_texts)} 门课程的知识增强文本。")
        return course_texts

    def _load_user_course_interactions(self, valid_course_ids):
        """
        从 MOOCCubeX user.json 提取 user-course 交互。
        user.json 格式: {id, course_order: [cid_num, ...], enroll_time: ["2019-10-12 10:28:02", ...]}
        """
        print(f"2. 正在从 user.json 提取 user-course 交互: {self.user_file} ...")
        records = []
        skipped_users = 0
        total_interactions = 0

        with open(self.user_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析用户JSON"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    uid = data.get('id', '')
                    course_order = data.get('course_order', [])
                    enroll_time = data.get('enroll_time', [])

                    if not uid or not course_order:
                        skipped_users += 1
                        continue

                    for idx, cid_num in enumerate(course_order):
                        # course_order 中是数字 ID, 需要加 C_ 前缀
                        cid = f"C_{cid_num}"

                        if cid not in valid_course_ids:
                            continue

                        # 解析 enroll_time
                        ts = None
                        if idx < len(enroll_time) and enroll_time[idx]:
                            try:
                                ts = pd.Timestamp(enroll_time[idx]).timestamp()
                            except Exception:
                                pass

                        if ts is None:
                            # 兜底: 用 2019 年基准 + 序号偏移
                            ts = 1546300800 + idx * 86400 + np.random.randint(0, 3600)

                        records.append({
                            'user_id': uid,
                            'course_id': cid,
                            'timestamp': float(ts)
                        })
                        total_interactions += 1

                except json.JSONDecodeError:
                    continue

        print(f"   总交互数: {total_interactions}, 跳过空用户: {skipped_users}")
        return records

    def _extract_bert_features(self, texts, batch_size=32):
        """
        使用中文 BERT 提取特征, max_length=256。
        """
        print("   正在加载 BERT 模型 (bert-base-chinese)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            model = AutoModel.from_pretrained('bert-base-chinese')
        except Exception:
            print("   无法下载 bert-base-chinese，尝试使用 bert-base-uncased...")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        all_embs = []
        print(f"   正在提取特征 (Device: {device}, max_length=256)...")

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
                batch = texts[i: i + batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu()
                all_embs.append(emb)

        return torch.cat(all_embs, dim=0)

    def process(self):
        # --- 0. 加载知识图谱关系 ---
        print("0. [HIN-X] 加载 MOOCCubeX 知识图谱关系...")
        course_concepts, concept_names = self._load_graph_features()

        # --- 1. 加载知识增强的课程文本 ---
        course_texts = self._load_course_metadata_enriched(course_concepts, concept_names)
        valid_course_ids = set(course_texts.keys())

        # --- 2. 加载 user-course 交互 ---
        records = self._load_user_course_interactions(valid_course_ids)

        if len(records) == 0:
            raise ValueError("错误: 未提取到有效的 user-course 交互！")

        df = pd.DataFrame(records)

        # 去重: 同一用户同一课程取最早时间
        print("3. 聚合去重...")
        df = df.groupby(['user_id', 'course_id'])['timestamp'].min().reset_index()
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"   去重后交互数: {len(df)}")

        # 时间范围
        dt_min = pd.to_datetime(df['timestamp'].min(), unit='s')
        dt_max = pd.to_datetime(df['timestamp'].max(), unit='s')
        print(f"   时间跨度: {dt_min} 至 {dt_max}")

        # --- 4. ID 编码 ---
        print("4. 编码 ID...")
        df['u_idx'] = self.user_enc.fit_transform(df['user_id'])
        df['i_idx'] = self.item_enc.fit_transform(df['course_id'])

        n_users = len(self.user_enc.classes_)
        n_items = len(self.item_enc.classes_)

        # --- 5. 生成知识增强 Content Embedding ---
        print("5. 生成知识增强 Content Embedding...")
        sorted_cids = self.item_enc.inverse_transform(range(n_items))
        sorted_texts = [course_texts.get(cid, '未知课程') for cid in sorted_cids]

        # 打印前3个增强文本示例
        print("\n   === 增强文本示例 ===")
        for idx in range(min(3, len(sorted_texts))):
            print(f"   [{idx}] {sorted_texts[idx][:200]}...")
        print("   ===================\n")

        content_emb = self._extract_bert_features(sorted_texts)
        print(f"   Embedding Shape: {content_emb.shape}")

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
            "content_dim": int(content_emb.shape[1])
        }
        with open(os.path.join(self.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        # 生成空的 llm_scores.pkl 占位
        llm_scores = pd.DataFrame({'i_idx': range(n_items), 'score': [0.5] * n_items})
        llm_scores.to_pickle(os.path.join(self.output_dir, 'llm_scores.pkl'))

        print(f"\n[完成] MOOCCubeX 知识增强数据处理完毕！输出目录: {self.output_dir}")
        print(f"  用户数: {n_users}, 课程数: {n_items}, Embedding 维度: {content_emb.shape[1]}")

        # 统计覆盖率
        has_concept = sum(1 for cid in sorted_cids if cid in course_concepts)
        has_field = 0
        with open(self.course_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('id') in set(sorted_cids) and data.get('field'):
                        has_field += 1
                except:
                    continue

        print(f"  图谱覆盖率: 学科领域 {has_field}/{n_items} ({has_field/n_items*100:.1f}%), "
              f"知识概念 {has_concept}/{n_items} ({has_concept/n_items*100:.1f}%)")
        print(f"  兼容文件已生成: {self.compat_course_concept_file}")


if __name__ == "__main__":
    processor = HINDataProcessorX(
        base_dir='./MOOCCubeX',
        output_dir='./processed_data_hin_x'
    )
    processor.process()
