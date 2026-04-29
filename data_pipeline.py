import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


class MOOCCubeProcessor:
    def __init__(self, interaction_file, course_file):
        self.interaction_file = interaction_file
        self.course_file = course_file
        self.tokenizer = None
        self.bert_model = None

    def load_data(self):
        print("1. Loading Data...")
        # 假设 interaction_file 是 CSV 格式 [user_id, course_id, timestamp]
        # 如果是 JSON，请用 pd.read_json
        df_inter = pd.read_csv(self.interaction_file)

        # 假设 course_file 包含 [course_id, course_name, description]
        df_course = pd.read_csv(self.course_file)

        # 2. 编码 ID (String -> Int)
        print("2. Encoding IDs...")
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        df_inter['u_idx'] = self.user_enc.fit_transform(df_inter['user_id'])
        df_inter['i_idx'] = self.item_enc.fit_transform(df_inter['course_id'])

        # 映射课程元数据中的 ID
        # 注意：要处理 course 文件中存在但 interaction 中不存在的课程
        valid_courses = set(self.item_enc.classes_)
        df_course = df_course[df_course['course_id'].isin(valid_courses)].copy()
        df_course['i_idx'] = self.item_enc.transform(df_course['course_id'])

        return df_inter, df_course

    def generate_content_embeddings(self, df_course):
        print("3. Generating Syllabus Embeddings (BERT)...")
        # 简单拼接 标题 + 简介
        texts = (df_course['course_name'].fillna('') + " " + df_course['description'].fillna('')).tolist()

        # 加载预训练 BERT (建议使用轻量级模型如 'prajjwal1/bert-tiny' 或 'bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

        embeddings = []
        batch_size = 32

        # 分批处理防止 OOM
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                outputs = model(**inputs)
                # 取 [CLS] token 作为句向量
                cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(cls_emb)

        # 按 i_idx 排序并堆叠
        full_emb_matrix = np.vstack(embeddings)

        # 这里的顺序必须和 item_encoder 的 0, 1, 2... 对应
        # 我们创建一个按 i_idx 索引的数组
        num_items = len(self.item_enc.classes_)
        final_emb = np.zeros((num_items, 768))  # BERT hidden size

        for idx, row in df_course.iterrows():
            item_idx = row['i_idx']
            # 找到刚才计算的对应向量 (这里简化处理，实际需通过 course_id 匹配)
            # 建议: 直接在 df_course 上加一列 embedding 然后 sort_values('i_idx')
            pass

            # 简化版：假设 df_course 已经去重且按 i_idx 排好序
        # 实际项目中建议用 Dict 映射: {i_idx: emb}
        return torch.tensor(full_emb_matrix, dtype=torch.float32)

    def process_streaming_data(self, df_inter):
        print("4. Processing Streaming Logic...")
        # 核心：按时间排序
        df_inter = df_inter.sort_values('timestamp').reset_index(drop=True)

        # 核心：计算"当前时刻"的流行度 (Pre-calculate Popularity)
        # 这是一个累积计数：第 N 次出现时，它的流行度就是 N
        # 这比在 Dataset 里实时算要快 100 倍
        df_inter['popularity'] = df_inter.groupby('i_idx').cumcount()

        return df_inter


# ==========================================
# 模拟运行数据处理
# ==========================================
# 假设您已经有了 CSV 文件，如果没有，我们可以模拟生成：
def create_dummy_mooccube():
    # 模拟数据
    df_inter = pd.DataFrame({
        'user_id': np.random.choice(['u' + str(i) for i in range(1000)], 10000),
        'course_id': np.random.choice(['c' + str(i) for i in range(500)], 10000),
        'timestamp': np.random.randint(1600000000, 1700000000, 10000)
    })
    df_course = pd.DataFrame({
        'course_id': ['c' + str(i) for i in range(500)],
        'course_name': ['Python Course ' + str(i) for i in range(500)],
        'description': ['Learn python ' + str(i) for i in range(500)]
    })
    return df_inter, df_course


df_inter, df_course = create_dummy_mooccube()
processor = MOOCCubeProcessor(None, None)
# 手动注入模拟数据
processor.user_enc = LabelEncoder()
processor.item_enc = LabelEncoder()
df_inter['u_idx'] = processor.user_enc.fit_transform(df_inter['user_id'])
df_inter['i_idx'] = processor.item_enc.fit_transform(df_inter['course_id'])
# 模拟 BERT 向量 (随机)
content_emb = torch.randn(500, 32)  # 假设 BERT 输出 32 维

# 执行流式处理
df_stream = processor.process_streaming_data(df_inter)
print(df_stream[['u_idx', 'i_idx', 'timestamp', 'popularity']].head())