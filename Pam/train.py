import torch
import pandas as pd
import numpy as np
import json
import os
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pam_model_llm import PAM_LLM, Config

# ================= 配置区域 =================
DATA_DIR = "processed_data_video"
LLM_SCORE_FILE = os.path.join(DATA_DIR, "llm_scores.pkl")
STREAM_FILE = os.path.join(DATA_DIR, "stream_data_5core.pkl")
META_FILE = os.path.join(DATA_DIR, "meta_5core.pkl")
COURSE_MAP_FILE = os.path.join(DATA_DIR, "vid2cid_map.pkl")

# 如果没跑5-core，回退(不建议)
if not os.path.exists(STREAM_FILE):
    print("⚠️ 未找到 5-core 数据，回退...")
    STREAM_FILE = os.path.join(DATA_DIR, "stream_data.pkl")
    META_FILE = os.path.join(DATA_DIR, "meta.json")

BATCH_SIZE = 512
NEW_COLD_THRESHOLD = 15
WARMUP_PERIODS = 2
NEGATIVE_SAMPLES = 100

NUM_ITEMS_GLOBAL = 0
C2I_MAP_GLOBAL = {}  # 全局课程倒排索引


# ===========================================

class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)
        if llm_map:
            keys = list(zip(df['u_idx'], df['i_idx']))
            vals = [llm_map.get(k, -1.0) for k in keys]
            self.llm_s = torch.tensor(vals, dtype=torch.float32)

    def __len__(self): return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm_s': self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm_s = torch.stack([item['llm_s'] for item in batch])
    # 训练时依然使用全局随机负采样 (让模型先学会区分不同课程)
    neg_i = torch.randint(0, NUM_ITEMS_GLOBAL, (len(batch),))
    return {'u': u, 'i': i, 'neg_i': neg_i}, pop, llm_s


def evaluate(model, loader, device, k_list=[10, 20]):
    """
    【Hard Mode】评估函数：使用同课程负采样
    """
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0

    # 获取 i2c 映射 (Tensor -> Numpy)
    i2c_np = model.i2c_map.cpu().numpy()

    with torch.no_grad():
        for batch, pop, _ in loader:
            mask = pop < model.cfg.cold_threshold
            n_cold = mask.sum().item()
            if n_cold == 0: continue

            u = batch['u'][mask].to(device)
            i_pos = batch['i'][mask].to(device)

            # === 用户意图 (注入课程信息) ===
            u_state = model.get_user_state(u, i_pos).unsqueeze(1)

            batch_size = u.size(0)

            # === 关键修改：构建 Hard Negatives ===
            # 对于每个正例，我们只从它所属的课程里选负例
            neg_samples_list = []

            # 转为 CPU 处理采样逻辑 (显存拷贝开销不大，因为 batch 只有几百)
            i_pos_cpu = i_pos.cpu().numpy()

            for target_vid in i_pos_cpu:
                # 1. 找到这门课的所有视频
                cid = i2c_np[target_vid]
                course_vids = C2I_MAP_GLOBAL.get(cid, [])

                # 2. 排除掉正例自己
                candidates_pool = [v for v in course_vids if v != target_vid]

                # 3. 采样
                if len(candidates_pool) == 0:
                    # 极端情况：这门课就这一个视频 -> 退化为随机采样
                    negs = np.random.randint(0, NUM_ITEMS_GLOBAL, NEGATIVE_SAMPLES)
                elif len(candidates_pool) >= NEGATIVE_SAMPLES:
                    # 候选够多 -> 随机采 100 个
                    negs = np.random.choice(candidates_pool, NEGATIVE_SAMPLES, replace=False)
                else:
                    # 候选不够 (例如只有 20 个) -> 全部选上，剩下的用全局随机补齐
                    # 这样既保证了难负例，又凑够了 100 个
                    n_need = NEGATIVE_SAMPLES - len(candidates_pool)
                    random_fill = np.random.randint(0, NUM_ITEMS_GLOBAL, n_need)
                    negs = np.concatenate([candidates_pool, random_fill])

                neg_samples_list.append(negs)

            # 转回 Tensor
            neg_samples = torch.tensor(np.array(neg_samples_list), dtype=torch.long).to(device)

            # 拼接: [Pos, Neg1, Neg2, ...]
            candidates = torch.cat([i_pos.view(-1, 1), neg_samples], dim=1)
            flat_candidates = candidates.view(-1)

            # 特征提取 (Hard Masking)
            raw_c = model.item_content(flat_candidates)
            proj_c = model.content_proj(raw_c)
            ids = model.item_id_emb(flat_candidates)

            proj_c = proj_c.view(batch_size, NEGATIVE_SAMPLES + 1, -1)
            ids = ids.view(batch_size, NEGATIVE_SAMPLES + 1, -1)

            # 强制抹除正例 ID
            ids[:, 0, :] = 0.0

            e_i = model.layernorm(proj_c + ids)

            # 推理
            z_u, _ = model.forward_mlp(u_state, None, False)
            z_i, _ = model.forward_mlp(e_i, None, True)

            scores = torch.bmm(z_u, z_i.transpose(1, 2)).squeeze(1)
            targets = torch.zeros(batch_size, dtype=torch.long).to(device).view(-1, 1)

            for k in k_list:
                _, topk_indices = torch.topk(scores, k, dim=1)
                hits = (topk_indices == targets).any(dim=1).float()
                metrics_sum[f'R@{k}'] += hits.sum().item()
                hit_ranks = torch.where(topk_indices == targets)
                if hit_ranks[1].numel() > 0:
                    ranks = hit_ranks[1].float()
                    dcg = 1.0 / torch.log2(ranks + 2.0)
                    metrics_sum[f'N@{k}'] += dcg.sum().item()
            total_cold_samples += n_cold

    if total_cold_samples == 0: return None, 0
    return {k: v / total_cold_samples for k, v in metrics_sum.items()}, total_cold_samples


def main():
    print(f"=== 🚀 启动 Hierarchical PAM 训练 (Hard Negative Eval) ===")

    if not os.path.exists(META_FILE): return
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    global NUM_ITEMS_GLOBAL, C2I_MAP_GLOBAL
    NUM_ITEMS_GLOBAL = meta['n_items']

    if not os.path.exists(COURSE_MAP_FILE): return
    with open(COURSE_MAP_FILE, 'rb') as f:
        course_data = pickle.load(f)

    # === 构建 C2I 倒排索引 (Course -> Video List) ===
    # course_data['i2c'] 是 {vid: cid}
    print("   -> 构建课程倒排索引 (Hard Negatives)...")
    temp_c2i = {}
    for vid, cid in course_data['i2c'].items():
        if cid not in temp_c2i: temp_c2i[cid] = []
        temp_c2i[cid].append(vid)
    C2I_MAP_GLOBAL = temp_c2i
    print(f"   -> 索引构建完成，覆盖 {len(C2I_MAP_GLOBAL)} 门课程")

    # 加载数据
    df = pd.read_pickle(STREAM_FILE)
    content_emb = torch.load(os.path.join(DATA_DIR, "content_emb.pt"))

    if 'timestamp' not in df.columns and 'ts' in df.columns:
        df.rename(columns={'ts': 'timestamp'}, inplace=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df['pid'] = df['dt'].dt.to_period('M')
    all_periods = sorted(df['pid'].dropna().unique())
    periods_data = [df[df['pid'] == p].reset_index(drop=True) for p in all_periods]

    cfg = Config(meta['n_users'], meta['n_items'], course_data['n_courses'], content_emb.shape[1])
    cfg.cold_threshold = NEW_COLD_THRESHOLD

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_LLM(cfg, content_emb, course_data['i2c']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    target_metrics = ['R@10', 'R@20', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}

    print("\n>>> 开始训练 (Hard Mode) <<<")
    print(f"{'Period':<6} | {'Samples':<8} | {'R@10':<8} | {'N@10':<8}")
    print("-" * 45)

    for t, p_df in enumerate(periods_data):
        if len(p_df) == 0: continue
        dataset = StreamDataset(p_df, None)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Eval
        if t >= WARMUP_PERIODS:
            met, n_cold = evaluate(model, loader, device)
            if met:
                print(f"{t:<6} | {n_cold:<8} | {met['R@10']:.4f}   | {met['N@10']:.4f}")
                for k in target_metrics: history[k].append(met[k])
            else:
                for k in target_metrics: history[k].append(0)
        else:
            for k in target_metrics: history[k].append(0)

        # Train
        model.train()
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        for batch, pop, llm_s in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, pop.to(device), llm_s.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("\n✅ 完成！")
    res_df = pd.DataFrame(history)
    res_df.to_csv(os.path.join(DATA_DIR, 'final_metrics_hard.csv'))


if __name__ == "__main__":
    main()