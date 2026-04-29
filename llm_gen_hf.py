import pandas as pd
import json
import os
import re
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 优化配置区域 =================
# 1. 模型路径
MODEL_ID = "./model/Qwen2.5-7B-Instruct"

# 2. 数据路径
DATA_FILE = "processed_data_x/stream_data.pkl"
COURSE_FILE = "MOOCCubeX/entities/course.json"
OUTPUT_FILE = "processed_data_x/llm_scores.pkl"

# 3. 【关键修改】提高打分门槛
# 配合训练脚本，涵盖更多“次冷门”课程 (Pop < 20)
TARGET_COLD_THRESHOLD = 20

# 4. 显存优化
USE_4BIT = True


# ===============================================

def load_json_flexible(file_path):
    """ 智能读取课程文件 (兼容 List/JSONL) """
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return {}

    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first = f.read(1)
            f.seek(0)
            if first == '[':
                data_list = json.load(f)
            else:
                data_list = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading json: {e}")
        return {}

    # 映射: Course_ID -> Text
    course_map = {}
    for item in data_list:
        cid = item.get('id')
        if cid:
            name = item.get('name', '')
            desc = item.get('about', '') or item.get('intro', '') or ''
            # 清洗文本
            desc = re.sub(r'<[^>]+>', '', desc).replace('\n', ' ').strip()
            full_text = f"{name}：{desc}"
            course_map[cid] = full_text[:1024]  # 截断

    return course_map


def get_score(model, tokenizer, hist_txt, target_txt):
    """ 调用 LLM 生成 0-1 分数 """
    prompt = (
        f"作为选课顾问，请基于学生的历史选课记录，预测他选修目标课程的概率。\n"
        f"历史课程：{hist_txt}\n"
        f"目标课程：{target_txt}\n"
        f"请仅输出一个0到1之间的小数（例如0.8），不要输出任何其他文字。"
    )

    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, temperature=0.1)

    res = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    match = re.search(r"0\.\d+|1\.0|0|1", res)
    return float(match.group()) if match else 0.5


def main():
    print("=== 🚀 启动 LLM 打分 (Optimized) ===")
    print(f"   -> 目标阈值: Popularity < {TARGET_COLD_THRESHOLD}")

    if not os.path.exists(DATA_FILE):
        print("Error: 找不到 stream_data.pkl，请先运行数据处理脚本。")
        return

    # 1. 强制重置 (防止使用旧索引)
    if os.path.exists(OUTPUT_FILE):
        print("⚠️ 检测到旧的评分文件。")
        choice = input("是否删除旧文件并重新生成? (y/n, 推荐y): ")
        if choice.lower() == 'y':
            os.remove(OUTPUT_FILE)
            scores = {}
            print("   -> 旧文件已删除，开始全新生成。")
        else:
            with open(OUTPUT_FILE, 'rb') as f:
                scores = pickle.load(f)
            print(f"   -> 继续使用旧文件 (已含 {len(scores)} 条)。警告：请确保索引未变！")
    else:
        scores = {}

    # 2. 加载模型
    print("📦 Loading Model...")
    try:
        bnb_config = None
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
            )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 加载课程文本
    print("📚 Loading Course Data...")
    c_map = load_json_flexible(COURSE_FILE)
    print(f"   -> 课程库: {len(c_map)} 门")

    # 4. 处理数据
    print("🔄 Processing Data...")
    df = pd.read_pickle(DATA_FILE)
    df = df.sort_values('timestamp')

    user_hist = {}
    new_count = 0
    save_interval = 50

    # 统计有多少样本符合冷启动条件
    total_cold_candidates = len(df[df['popularity'] < TARGET_COLD_THRESHOLD])
    print(f"   -> 预计处理任务数: {total_cold_candidates} (Pop < {TARGET_COLD_THRESHOLD})")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uid = row['user_id']
        cid_raw = row['course_id']  # C_xxx
        u_idx, i_idx = row['u_idx'], row['i_idx']
        pop = row['popularity']

        if uid not in user_hist: user_hist[uid] = []

        # 【核心逻辑】只要 Pop < 20 且有历史，就打分
        if pop < TARGET_COLD_THRESHOLD and len(user_hist[uid]) > 0:

            if (u_idx, i_idx) not in scores:
                # 获取最近 3 门课作为 Context
                hist_raw = user_hist[uid][-3:]
                hist_txt = "；".join([c_map.get(h, '') for h in hist_raw if c_map.get(h)])
                tgt_txt = c_map.get(cid_raw, '')

                if hist_txt and tgt_txt:
                    try:
                        s = get_score(model, tokenizer, hist_txt, tgt_txt)
                        scores[(u_idx, i_idx)] = s
                        new_count += 1

                        if new_count % save_interval == 0:
                            with open(OUTPUT_FILE, 'wb') as f: pickle.dump(scores, f)
                    except:
                        pass

        user_hist[uid].append(cid_raw)

    # 最终保存
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(scores, f)

    print("\n✅ 生成完成！")
    print(f"   - 本次新增评分: {new_count}")
    print(f"   - 总评分库大小: {len(scores)}")
    print("现在请运行 train_x_optimized.py 见证奇迹！")


if __name__ == "__main__":
    main()