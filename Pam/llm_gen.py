import pandas as pd
import pickle
import torch
import os
import re
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置区域 =================
MODEL_ID = "../model/Qwen2.5-7B-Instruct"
DATA_DIR = "./processed_data_video"
OUTPUT_FILE = os.path.join(DATA_DIR, "llm_scores.pkl")

# 【关键参数】显存够大就往大调 (8, 16, 32, 64)
BATCH_SIZE = 8
TARGET_POP = 20  # 只处理 Pop < 20 的视频


# ===========================================

def load_data():
    print("Loading Data...")
    stream_path = os.path.join(DATA_DIR, 'stream_data.pkl')
    meta_path = os.path.join(DATA_DIR, 'video_meta.pkl')

    if not os.path.exists(stream_path):
        raise FileNotFoundError(f"找不到 {stream_path}")

    df = pd.read_pickle(stream_path).sort_values('ts')
    with open(meta_path, 'rb') as f:
        v_meta = pickle.load(f)

    return df, v_meta


def prepare_tasks(df, v_meta, existing_scores):
    """ 预先筛选出所有需要打分的任务，不再边跑边筛 """
    print("Preprocessing tasks (筛选任务)...")
    tasks = []  # (u_idx, i_idx, prompt_text)
    user_hist = {}

    # 只需要遍历一遍 DataFrame
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        uid, vid = row['u'], row['i']
        u_idx, i_idx = row['u_idx'], row['i_idx']
        pop = row['popularity']

        # 1. 必须是冷启动
        # 2. 用户必须有历史
        # 3. 之前没算过
        if pop < TARGET_POP and len(user_hist.get(uid, [])) > 0:
            if (u_idx, i_idx) not in existing_scores:
                # 构造 Prompt
                hist_vids = user_hist[uid][-3:]  # 取最近3个
                # 确保元数据存在
                hist_txts = [v_meta[v]['text'] for v in hist_vids if v in v_meta]
                target_txt = v_meta.get(vid, {}).get('text', '')

                if hist_txts and target_txt:
                    h_str = "；".join(hist_txts)
                    prompt = (
                        f"根据用户观看历史预测是否会观看目标视频。\n"
                        f"历史：{h_str}\n"
                        f"目标：{target_txt}\n"
                        f"请仅输出一个0.0到1.0之间的概率小数："
                    )
                    tasks.append((u_idx, i_idx, prompt))

        # 更新历史
        if uid not in user_hist: user_hist[uid] = []
        user_hist[uid].append(vid)

    print(f"   -> 待处理任务总数: {len(tasks)}")
    return tasks


def main():
    # 1. 初始化
    scores = {}
    if os.path.exists(OUTPUT_FILE):
        print("📂 发现旧进度，加载中...")
        with open(OUTPUT_FILE, 'rb') as f:
            scores = pickle.load(f)
        print(f"   -> 已有分数: {len(scores)} 条")

    # 2. 准备数据
    df, v_meta = load_data()
    tasks = prepare_tasks(df, v_meta, scores)

    if len(tasks) == 0:
        print("🎉 所有任务已完成！无需运行。")
        return

    # 3. 加载模型
    print("📦 Loading Model (Batch Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # 【重要】Batch推理需要设置 padding_side='left' (Decoder-only模型)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 4. Batch 推理循环
    print(f"🚀 开始极速推理 (Batch Size = {BATCH_SIZE})...")
    total_batches = math.ceil(len(tasks) / BATCH_SIZE)

    # 使用 tqdm 显示进度
    new_scores = 0
    save_interval = 50  # 每50个Batch保存一次

    for i in tqdm(range(0, len(tasks), BATCH_SIZE), total=total_batches, desc="Inferencing"):
        batch_tasks = tasks[i: i + BATCH_SIZE]
        batch_prompts = [t[2] for t in batch_tasks]

        # 4.1 Tokenize (批量)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        # 4.2 Generate (批量)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.1,
                do_sample=False  # 确定性生成更快
            )

        # 4.3 Decode & Parse
        # 只解码新生成的 token
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_list = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 4.4 存入结果
        for j, text in enumerate(decoded_list):
            u_idx, i_idx, _ = batch_tasks[j]

            # 正则提取数字
            match = re.search(r"0\.\d+|1\.0|0|1", text)
            val = float(match.group()) if match else 0.5
            scores[(u_idx, i_idx)] = val
            new_scores += 1

        # 4.5 定期保存
        if (i // BATCH_SIZE) % save_interval == 0:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(scores, f)

    # 最终保存
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(scores, f)

    print(f"\n✅ 全部完成！")
    print(f"   - 总计保存: {len(scores)}")
    print(f"   - 本次新增: {new_scores}")
    print("➡️ 请运行 train_x_optimized.py 开始训练")


if __name__ == "__main__":
    main()