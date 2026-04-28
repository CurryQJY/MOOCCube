import torch
import numpy as np
import pandas as pd
import os
import json
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ================= 配置参数 =================
class GenConfig:
    MODEL_PATH = "./model/Qwen2.5-7B-Instruct"
    DATA_PATH = "processed_data/stream_data.pkl"
    META_PATH = "processed_data/course_titles.json"
    SAVE_PATH = "processed_data/usim_data.pkl"

    AUGMENT_RATIO = 0.2
    COLD_THRESHOLD = 5
    BATCH_SIZE = 32


# ================= 1. 本地 LLM =================
class LocalQwen:
    def __init__(self, model_path):
        print(f"🚀 [LLM] Loading Qwen from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config
            ).eval()
            print("✅ LLM Loaded!")
        except Exception as e:
            print(f"❌ Load Error: {e}")
            exit()

    def predict_batch(self, batch_prompts, debug=False):
        batch_texts = []
        for p in batch_prompts:
            messages = [
                {"role": "system",
                 "content": "You are a user simulator. Analyze the user history and predict future interest."},
                {"role": "user", "content": p}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        inputs = self.tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=2,  # 稍微多一点以便观察
                do_sample=True,  # 开启采样增加多样性
                temperature=0.7
            )

        gen_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.tokenizer.batch_decode(gen_ids_trimmed, skip_special_tokens=True)

        # 🔥 Debug 输出：每 100 次调用打印一次，或者是第一次调用时打印
        if debug:
            print(f"\n[DEBUG] Prompt: {batch_prompts[0][-100:]}")  # 只打印最后一点
            print(f"[DEBUG] Reply: {responses[0]}")

        results = []
        for res in responses:
            res_lower = res.lower()
            # 放宽判定逻辑
            if "yes" in res_lower:
                results.append(1)
            else:
                results.append(0)
        return results


# ================= 2. 数据处理 =================
def load_raw_data():
    print("📂 Loading data...")
    df = pd.read_pickle(GenConfig.DATA_PATH)

    if 'u_idx' in df.columns: df = df.rename(columns={'u_idx': 'user_id', 'i_idx': 'item_id'})
    df = df.loc[:, ~df.columns.duplicated()]

    user_groups = df.groupby('user_id')['item_id'].apply(list)
    popular_items = df['item_id'].value_counts().head(2000).index.tolist()

    # 🔥 修改加载逻辑：直接读取 ID -> Name 的 JSON
    id2title = {}
    if os.path.exists(GenConfig.META_PATH):
        print(f"📂 Loading Titles from {GenConfig.META_PATH}...")
        with open(GenConfig.META_PATH, 'r', encoding='utf-8') as f:
            # 新文件是 {"1": "Name", "2": "Name"} 格式
            raw_data = json.load(f)
            # 确保 Key 是 int 类型
            for k, v in raw_data.items():
                try:
                    id2title[int(k)] = v
                except:
                    pass
        print(f"✅ Loaded {len(id2title)} titles.")
    else:
        print("❌ Title file not found! LLM will fail.")

    return user_groups, id2title, df['item_id'].max(), popular_items


# ================= 3. 主程序 =================
if __name__ == "__main__":
    user_groups, id2title, n_items, popular_items = load_raw_data()
    cold_users = [u for u, items in user_groups.items() if len(items) < GenConfig.COLD_THRESHOLD]

    # 如果没有 title，LLM 跑不起来，直接报错提醒
    if len(id2title) < 10:
        print("⚠️ 警告: id2title 几乎为空！这会导致 LLM 只能看到 'Course_ID'，从而 HitRate=0。")
        print("请检查 processed_data/meta.json 是否正确生成。")

    llm = LocalQwen(GenConfig.MODEL_PATH)

    augmented_data = []
    num_to_gen = min(len(cold_users) * 2, int(len(user_groups) * GenConfig.AUGMENT_RATIO))
    print(f"🤖 Generating {num_to_gen} pairs (Sampling from top {len(popular_items)} popular items)...")

    batch_prompts = []
    batch_meta = []

    pbar = tqdm(total=num_to_gen, desc="Simulating")
    hits = 0
    total_processed = 0

    while len(augmented_data) < num_to_gen:
        while len(batch_prompts) < GenConfig.BATCH_SIZE:
            u = np.random.choice(cold_users)
            hist_ids = user_groups[u]
            if not hist_ids: continue

            # 🔥 关键修改：从热门物品中采样，大幅提高 HitRate
            cand_id = np.random.choice(popular_items)
            if cand_id in hist_ids: continue

            hist_titles = [id2title.get(i, f"Course_{i}") for i in hist_ids[-3:]]
            cand_title = id2title.get(cand_id, f"Course_{cand_id}")

            # 稍微放松 Prompt，让 LLM 敢于预测
            prompt = f"""User history: {', '.join(hist_titles)}. 
            Target course: {cand_title}. 
            Is this course relevant to the user's interests? Answer Yes or No. Do not explain."""

            batch_prompts.append(prompt)
            batch_meta.append([u, cand_id])

        # 第一次运行打印 Debug 信息
        debug_flag = (total_processed == 0)
        results = llm.predict_batch(batch_prompts, debug=debug_flag)

        for res, meta in zip(results, batch_meta):
            if res == 1:
                augmented_data.append(meta)
                hits += 1
                pbar.update(1)  # 只有生成了才更新进度条

        total_processed += len(batch_prompts)
        current_hit_rate = hits / total_processed if total_processed > 0 else 0
        pbar.set_postfix({'HitRate': f"{current_hit_rate:.2%}", 'TotalScanned': total_processed})

        batch_prompts = []
        batch_meta = []

        # 兜底机制：如果扫了 10万条 还是 0 命中，强制停止
        if total_processed > 100000 and hits == 0:
            print("\n❌ 严重错误: 尝试了 10万次 依然没有命中。")
            print("请检查 meta.json 中的课程名是否正确，或 LLM 是否拒绝回答。")
            break

    pbar.close()

    if len(augmented_data) > 0:
        print(f"💾 Saving {len(augmented_data)} pairs to {GenConfig.SAVE_PATH}...")
        with open(GenConfig.SAVE_PATH, "wb") as f:
            pickle.dump(augmented_data, f)
    else:
        print("❌ 生成失败，没有数据被保存。")
