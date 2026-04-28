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

# ================= 优化配置区域 =================
DATA_DIR = "processed_data_x"
LLM_SCORE_FILE = os.path.join(DATA_DIR, "llm_scores.pkl")

# 【优化1】减小 Batch Size，增加更新频率
BATCH_SIZE = 1024

# 【优化2】大幅提高冷启动阈值 (解决热门逃逸问题)
NEW_COLD_THRESHOLD = 15

WARMUP_PERIODS = 3


# ===============================================

class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)

        if llm_map:
            keys = list(zip(df['u_idx'], df['i_idx']))
            # 增加鲁棒性：防止 LLM Map 是旧的导致 Key Error
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
    return {'u': u, 'i': i}, pop, llm_s


def compute_ranking_metrics(scores, k_list=[5, 10, 20]):
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = torch.arange(batch_size).to(scores.device).view(-1, 1)
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)

    results = {}
    for k in k_list:
        if k > num_candidates:
            results[f'R@{k}'] = 0.0;
            results[f'N@{k}'] = 0.0
            continue
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        results[f'R@{k}'] = hits.mean().item()
        hit_ranks = torch.where(preds == targets)
        ndcg_val = 0.0
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg_val = dcg.sum().item() / batch_size
        results[f'N@{k}'] = ndcg_val
    return results


def evaluate(model, loader, device, k_list=[10, 20]):
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0

    with torch.no_grad():
        for batch, pop, _ in loader:
            # 使用新的阈值
            mask = pop < model.cfg.cold_threshold
            n_cold = mask.sum().item()
            if n_cold < 2: continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)
            e_u = model.user_emb(u)
            e_i = model.get_item_features(i)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)
            z_i, _ = model.forward_mlp(e_i, model.vars, True)
            scores = torch.mm(z_u, z_i.t())

            res = compute_ranking_metrics(scores, k_list=k_list)
            for k, v in res.items(): metrics_sum[k] += v * n_cold
            total_cold_samples += n_cold

    if total_cold_samples == 0: return None, 0
    period_metrics = {k: v / total_cold_samples for k, v in metrics_sum.items()}
    return period_metrics, total_cold_samples


def main():
    print(f"=== 🚀 启动 PAM+LLM 优化版训练 ===")
    print(f"   -> Batch Size: {BATCH_SIZE}")
    print(f"   -> Cold Threshold: < {NEW_COLD_THRESHOLD} (已提高)")

    stream_file = os.path.join(DATA_DIR, "stream_data.pkl")
    meta_file = os.path.join(DATA_DIR, "meta.json")
    emb_file = os.path.join(DATA_DIR, "content_emb.pt")

    if not os.path.exists(stream_file): return

    with open(meta_file, "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(stream_file)
    content_emb = torch.load(emb_file)

    llm_map = None
    if os.path.exists(LLM_SCORE_FILE):
        print("   -> 加载 LLM 分数...")
        with open(LLM_SCORE_FILE, "rb") as f:
            llm_map = pickle.load(f)
        # 简单校验
        if len(llm_map) > 0:
            sample_key = list(llm_map.keys())[0]
            print(f"      (Sample Key: {sample_key}, Score: {llm_map[sample_key]})")
            print("      ⚠️ 请确认上面Key是否匹配当前数据集，否则请重跑 LLM 生成！")

    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df['pid'] = df['dt'].dt.to_period('M')
    all_periods = sorted(df['pid'].dropna().unique())
    periods_data = [df[df['pid'] == p].reset_index(drop=True) for p in all_periods]

    print(f"   -> 数据跨度: {all_periods[0]} 至 {all_periods[-1]}")

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    # 【强制覆盖配置】
    cfg.cold_threshold = NEW_COLD_THRESHOLD

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_LLM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    target_metrics = ['R@10', 'R@20', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}
    global_score_sum = {k: 0.0 for k in target_metrics}
    global_sample_count = 0
    valid_period_count = 0

    print("\n>>> 开始训练 (High Threshold) <<<")
    print(f"{'Period':<6} | {'Samples':<8} | {'R@10':<8} | {'N@10':<8}")
    print("-" * 45)

    for t, p_df in enumerate(periods_data):
        if len(p_df) == 0: continue
        dataset = StreamDataset(p_df, llm_map)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # --- Eval ---
        if t >= WARMUP_PERIODS:
            met, n_cold = evaluate(model, loader, device)
            if met:
                print(f"{t:<6} | {n_cold:<8} | {met['R@10']:.4f}   | {met['N@10']:.4f}")
                for k in target_metrics: history[k].append(met[k])
                for k in target_metrics: global_score_sum[k] += met[k] * n_cold
                global_sample_count += n_cold
                valid_period_count += 1
            else:
                for k in target_metrics: history[k].append(0)
        else:
            for k in target_metrics: history[k].append(0)

        # --- Train ---
        model.train()
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        for batch, pop, llm_s in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, pop.to(device), llm_s.to(device))

            if isinstance(loss, torch.Tensor):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    # ================= 结果 =================
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (Optimized)")
    print("=" * 60)
    print(f"总冷启动样本 : {global_sample_count}")
    print("-" * 60)
    print(f"{'Metric':<10} | {'Macro Avg':<15} | {'Micro Avg':<15}")
    print("-" * 60)

    for m in target_metrics:
        valid_vals = [v for i, v in enumerate(history[m]) if i >= WARMUP_PERIODS and v > 0]
        macro_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        micro_val = global_score_sum[m] / global_sample_count if global_sample_count > 0 else 0.0
        print(f"{m:<10} | {macro_val:.4f}          | {micro_val:.4f}")


if __name__ == "__main__":
    main()