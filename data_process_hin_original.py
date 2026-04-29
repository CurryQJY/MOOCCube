"""
data_process_hin.py
-------------------
知识图谱增强的数据预处理脚本。
在原始 data_process.py 基础上，将 MOOCCube 的学校、教师、知识概念 (Concept)
三类关系数据注入课程文本，生成语义更丰富的 BERT Embedding。

输出目录: ./processed_data_hin/
  - stream_data.pkl   (交互数据, 与原版格式完全一致)
  - content_emb.pt    (知识增强的 BERT [CLS] embedding, [N_items, 768])
  - meta.json         (元信息)
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
# 1. 关系文件加载器
# ==============================

def load_tsv_relation(filepath, reverse=False):
    """
    加载 TSV 格式的关系文件 (每行: src\tdst)。
    返回 defaultdict(list): {src: [dst1, dst2, ...]}
    如果 reverse=True, 返回 {dst: [src1, src2, ...]}
    """
    mapping = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                src, dst = parts[0], parts[1]
                if reverse:
                    mapping[dst].append(src)
                else:
                    mapping[src].append(dst)
    return mapping


def load_entity_names(filepath):
    """
    加载 JSON Lines 格式的实体文件, 返回 {id: name} 映射。
    """
    names = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                eid = data.get('id', '')
                name = data.get('name', '') or ''
                if eid and name:
                    names[eid] = name
            except json.JSONDecodeError:
                continue
    return names


# ==============================
# 2. 知识增强数据处理器
# ==============================

class HINDataProcessor:
    def __init__(self, base_dir='./MOOCCube', output_dir='./processed_data_hin'):
        self.base_dir = base_dir
        self.output_dir = output_dir

        # 文件路径
        self.inter_file = os.path.join(base_dir, 'relations', 'user-course.json')
        self.course_file = os.path.join(base_dir, 'entities', 'course.json')
        self.school_course_file = os.path.join(base_dir, 'relations', 'school-course.json')
        self.teacher_course_file = os.path.join(base_dir, 'relations', 'teacher-course.json')
        self.course_concept_file = os.path.join(base_dir, 'relations', 'course-concept.json')
        self.teacher_entity_file = os.path.join(base_dir, 'entities', 'teacher.json')
        self.school_entity_file = os.path.join(base_dir, 'entities', 'school.json')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _load_graph_features(self):
        """
        加载知识图谱中的关系数据, 构建课程的多维特征映射。
        """
        print("  [HIN] 加载学校-课程关系...")
        school_course = load_tsv_relation(self.school_course_file)
        # 反转: Course -> School
        course_school = defaultdict(list)
        for school_id, courses in school_course.items():
            for cid in courses:
                course_school[cid].append(school_id)
        print(f"         覆盖 {len(course_school)} 门课程")

        print("  [HIN] 加载教师-课程关系...")
        teacher_course = load_tsv_relation(self.teacher_course_file)
        # 反转: Course -> Teachers
        course_teachers = defaultdict(list)
        for teacher_id, courses in teacher_course.items():
            for cid in courses:
                course_teachers[cid].append(teacher_id)
        print(f"         覆盖 {len(course_teachers)} 门课程")

        print("  [HIN] 加载课程-知识概念关系...")
        course_concepts = load_tsv_relation(self.course_concept_file)
        print(f"         覆盖 {len(course_concepts)} 门课程")

        print("  [HIN] 加载实体名称...")
        teacher_names = load_entity_names(self.teacher_entity_file)
        school_names = load_entity_names(self.school_entity_file)
        print(f"         教师: {len(teacher_names)}, 学校: {len(school_names)}")

        return course_school, course_teachers, course_concepts, teacher_names, school_names

    def _build_enriched_text(self, cid, name, about,
                              course_school, course_teachers, course_concepts,
                              teacher_names, school_names):
        """
        将原始课程文本 (name + about) 增强为带有学校、教师、知识概念的结构化文本。

        模板:
            [学校] 清华大学
            [教师] 李某某, 王某某
            [知识概念] 数据结构, 二叉树, 算法
            [课程] 数据结构基础
            [简介] 本课程主要讲解...

        BERT max_length=256, 所以概念截取前15个以防溢出。
        """
        parts = []

        # 学校
        schools = course_school.get(cid, [])
        if schools:
            s_names = [school_names.get(s, s.replace('S_', '')) for s in schools]
            parts.append(f"[学校] {'，'.join(s_names)}")

        # 教师
        teachers = course_teachers.get(cid, [])
        if teachers:
            t_names = [teacher_names.get(t, t.replace('T_', '')) for t in teachers]
            parts.append(f"[教师] {'，'.join(t_names[:5])}")  # 最多5位教师

        # 知识概念 (取前15个, 并提取可读名称)
        concepts = course_concepts.get(cid, [])
        if concepts:
            # 概念 ID 格式: "K_活性炭_化学" -> 取 "活性炭"
            c_names = []
            for c in concepts[:15]:
                c_clean = c.replace('K_', '')
                # 取第一个下划线之前的部分作为概念名
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

    def _load_course_metadata_enriched(self, course_school, course_teachers,
                                        course_concepts, teacher_names, school_names):
        """
        加载 course.json 并构建知识增强的课程文本。
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

                    text = self._build_enriched_text(
                        cid, name, about,
                        course_school, course_teachers, course_concepts,
                        teacher_names, school_names
                    )

                    if cid and text:
                        course_texts[cid] = text
                except json.JSONDecodeError:
                    continue

        print(f"   成功加载 {len(course_texts)} 门课程的知识增强文本。")
        return course_texts

    def _extract_bert_features(self, texts, batch_size=32):
        """
        使用中文 BERT 提取特征, max_length 提升到 256 以容纳更多结构化信息。
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
                    max_length=256,  # 提升以容纳结构化元数据
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu()
                all_embs.append(emb)

        return torch.cat(all_embs, dim=0)

    def _extract_time_from_course_id(self, course_id):
        """从课程 ID 解析时间戳"""
        cid_str = str(course_id)
        match = re.search(r'(20[0-2]\d)', cid_str)
        if match:
            year = int(match.group(1))
            base_time = (year - 1970) * 31536000
            return base_time + np.random.randint(0, 3600 * 24 * 30)
        else:
            return 1483228800 + np.random.randint(0, 31536000)

    def process(self):
        # --- 0. 加载知识图谱关系 ---
        print("0. [HIN] 加载知识图谱关系...")
        (course_school, course_teachers, course_concepts,
         teacher_names, school_names) = self._load_graph_features()

        # --- 1. 加载知识增强的课程文本 ---
        course_texts = self._load_course_metadata_enriched(
            course_school, course_teachers, course_concepts,
            teacher_names, school_names
        )
        valid_course_ids = set(course_texts.keys())

        # --- 2. 加载交互数据 ---
        print(f"2. 正在加载交互数据: {self.inter_file} ...")
        df = pd.read_csv(
            self.inter_file,
            sep='\t',
            header=None,
            names=['user_id', 'course_id', 'raw_time'],
            on_bad_lines='skip'
        )
        original_len = len(df)
        df = df[df['course_id'].isin(valid_course_ids)]
        print(f"   过滤无效课程交互: {original_len} -> {len(df)}")

        if len(df) == 0:
            raise ValueError("错误: 交互数据中的 course_id 与 course.json 均不匹配！")

        # --- 3. 时间戳处理 ---
        print("3. 生成流式时间戳...")
        tqdm.pandas(desc="解析时间")
        df['timestamp'] = df['course_id'].progress_apply(self._extract_time_from_course_id)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # --- 4. ID 编码 ---
        print("4. 编码 ID...")
        df['u_idx'] = self.user_enc.fit_transform(df['user_id'])
        df['i_idx'] = self.item_enc.fit_transform(df['course_id'])

        n_users = len(self.user_enc.classes_)
        n_items = len(self.item_enc.classes_)

        # --- 5. 生成知识增强 Content Embedding ---
        print("5. 生成知识增强 Content Embedding...")
        sorted_cids = self.item_enc.inverse_transform(range(n_items))
        sorted_texts = [course_texts[cid] for cid in sorted_cids]

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

        print(f"\n[完成] 知识增强数据处理完毕！输出目录: {self.output_dir}")
        print(f"  用户数: {n_users}, 课程数: {n_items}, Embedding 维度: {content_emb.shape[1]}")

        # 统计覆盖率
        has_school = sum(1 for cid in sorted_cids if cid in course_school)
        has_teacher = sum(1 for cid in sorted_cids if cid in course_teachers)
        has_concept = sum(1 for cid in sorted_cids if cid in course_concepts)
        print(f"  图谱覆盖率: 学校 {has_school}/{n_items} ({has_school/n_items*100:.1f}%), "
              f"教师 {has_teacher}/{n_items} ({has_teacher/n_items*100:.1f}%), "
              f"知识概念 {has_concept}/{n_items} ({has_concept/n_items*100:.1f}%)")


if __name__ == "__main__":
    processor = HINDataProcessor(
        base_dir='./MOOCCube',
        output_dir='./processed_data_hin'
    )
    processor.process()
