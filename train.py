import torch
import pandas as pd
import numpy as np
import json
import os

# ==============================
# 【必需】强制使用非交互式后端，防止画图报错
# ==============================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ==============================

from torch.utils.data import Dataset, DataLoader
from pam_model import PAM, Config


# ==========================================
# 1. 工具函数
# ==========================================

class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


def collate_fn(batch):
    # 堆叠 Batch 数据
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])

    # 【核心修复】返回一个元组：(数据字典, 流行度张量)
    # 这样 for batch, pop in loader 才能正确解包
    return {'u': u, 'i': i}, pop


def split_dataframe_by_periods(df, period_type='M'):
    """ 按月(M)或周(W)切分数据 """
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['period_id'] = df['datetime'].dt.to_period(period_type)

    periods = []
    valid_periods = df['period_id'].dropna().unique()
    sorted_period_keys = sorted(valid_periods)

    print(f">> 数据已切分为 {len(sorted_period_keys)} 个周期 ({period_type})。")

    for p_key in sorted_period_keys:
        period_df = df[df['period_id'] == p_key].reset_index(drop=True)
        if len(period_df) > 0:
            periods.append(period_df)
    return periods


def compute_ranking_metrics(scores, k_list=[5, 10]):
    """ 鲁棒的指标计算，自动适应样本不足的情况 """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)

    targets = torch.arange(batch_size).to(scores.device).view(-1, 1)

    # 动态调整 K，防止越界
    max_k = max(k_list)
    actual_k = min(max_k, num_candidates)

    _, topk_indices = torch.topk(scores, actual_k, dim=1)

    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]  # Python切片自动处理越界
        hits = (preds == targets).any(dim=1).float()

        recall = hits.mean().item()

        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0

        results[f'R@{k}'] = recall
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg

    return results


# ==========================================
# 2. 评估函数 (支持 Cold 和 Hot 双重评估)
# ==========================================

def evaluate(model, loader, device):
    model.eval()

    # 分别记录冷、热指标
    metrics_sum = {
        'cold': {'R@10': 0.0, 'N@10': 0.0, 'count': 0},
        'hot': {'R@10': 0.0, 'N@10': 0.0, 'count': 0}
    }

    with torch.no_grad():
        for batch, pop in loader:
            # --- 1. 生成掩码 (Masks) ---
            mask_cold = pop < model.cfg.cold_threshold
            mask_hot = pop >= model.cfg.cold_threshold

            weights = model.vars

            # --- [分支 A] 测试冷启动 (新课) ---
            if mask_cold.sum() >= 2:
                u_test = batch['u'][mask_cold].to(device)
                i_test = batch['i'][mask_cold].to(device)

                e_u = model.user_emb(u_test)
                # 使用修正后的函数名 get_item_features
                e_i = model.get_item_features(i_test)

                # 使用修正后的函数名 forward_mlp
                z_u, _ = model.forward_mlp(e_u, weights, False)
                z_i, _ = model.forward_mlp(e_i, weights, True)

                scores = torch.mm(z_u, z_i.t())
                res = compute_ranking_metrics(scores, k_list=[10])

                metrics_sum['cold']['R@10'] += res['R@10']
                metrics_sum['cold']['N@10'] += res['N@10']
                metrics_sum['cold']['count'] += 1

            # --- [分支 B] 测试热门 (老课) ---
            if mask_hot.sum() >= 2:
                u_test = batch['u'][mask_hot].to(device)
                i_test = batch['i'][mask_hot].to(device)

                e_u = model.user_emb(u_test)
                e_i = model.get_item_features(i_test)

                z_u, _ = model.forward_mlp(e_u, weights, False)
                z_i, _ = model.forward_mlp(e_i, weights, True)

                scores = torch.mm(z_u, z_i.t())
                res = compute_ranking_metrics(scores, k_list=[10])

                metrics_sum['hot']['R@10'] += res['R@10']
                metrics_sum['hot']['N@10'] += res['N@10']
                metrics_sum['hot']['count'] += 1

    # --- 3. 汇总计算 ---
    final_results = {}

    # 计算 Cold 平均值
    if metrics_sum['cold']['count'] > 0:
        final_results['Cold_R@10'] = metrics_sum['cold']['R@10'] / metrics_sum['cold']['count']
        final_results['Cold_N@10'] = metrics_sum['cold']['N@10'] / metrics_sum['cold']['count']
    else:
        final_results['Cold_R@10'] = None

    # 计算 Hot 平均值
    if metrics_sum['hot']['count'] > 0:
        final_results['Hot_R@10'] = metrics_sum['hot']['R@10'] / metrics_sum['hot']['count']
        final_results['Hot_N@10'] = metrics_sum['hot']['N@10'] / metrics_sum['hot']['count']
    else:
        final_results['Hot_R@10'] = None

    return final_results


# ==========================================
# 3. 主程序
# ==========================================

def main():
    # --- Load Data ---
    print("Loading Data...")
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("错误: 找不到处理好的数据，请先运行 data_process.py")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    # --- Split Data ---
    periods = split_dataframe_by_periods(df, period_type='M')

    dataloaders = []
    for p_df in periods:
        ds = StreamDataset(p_df)
        dl = DataLoader(ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        dataloaders.append(dl)

    # --- Config & Model ---
    cfg = Config(meta['n_users'], meta['n_items'])

    # [关键] 维度自适应
    real_content_dim = content_emb.shape[1]
    cfg.content_dim = real_content_dim
    print(f">> 使用 Content Embedding 维度: {real_content_dim}")

    # [优化参数]
    cfg.gamma_s = 5.0
    cfg.lambda_cold = 2.0
    cfg.cold_threshold = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # --- Rolling Loop ---
    print(f"\n>>> 开始滚动评估 (Rolling Evaluation) - 共 {len(periods)} 个周期 <<<")

    history = {
        'Cold_R@10': [], 'Cold_N@10': [],
        'Hot_R@10': [], 'Hot_N@10': []
    }

    warmup = 3

    for t, loader in enumerate(dataloaders):
        n_samples = len(loader.dataset)
        print(f"\n>>> Period {t} (样本数: {n_samples}) <<<")

        # Phase 1: Test
        if t >= warmup:
            metrics = evaluate(model, loader, device)

            # 记录 Cold 指标
            if metrics['Cold_R@10'] is not None:
                c_r10 = metrics['Cold_R@10']
                c_n10 = metrics['Cold_N@10']
                history['Cold_R@10'].append(c_r10)
                history['Cold_N@10'].append(c_n10)
                print(f"  [★测试结果] ❄️ 新课 (Cold): Recall@10 = {c_r10:.4f} | NDCG@10 = {c_n10:.4f}")
            else:
                print("  [测试跳过] 本周期无足够的冷启动新课 (No Cold Items).")
                history['Cold_R@10'].append(0)
                history['Cold_N@10'].append(0)

            # 记录 Hot 指标
            if metrics['Hot_R@10'] is not None:
                h_r10 = metrics['Hot_R@10']
                h_n10 = metrics['Hot_N@10']
                history['Hot_R@10'].append(h_r10)
                history['Hot_N@10'].append(h_n10)
                print(f"             🔥 老课 (Hot ): Recall@10 = {h_r10:.4f} | NDCG@10 = {h_n10:.4f}")
            else:
                history['Hot_R@10'].append(0)
                history['Hot_N@10'].append(0)

        else:
            print("  [预热阶段] 仅训练 (Warm-up)...")
            for k in history: history[k].append(0)

        # Phase 2: Train
        model.train()
        total_loss = 0
        steps = 0

        # 这里的解包现在是安全的，因为 collate_fn 返回的是 (batch_dict, pop_tensor)
        for batch, pop in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pop = pop.to(device)

            # Forward 需要 batch 和 pop 两个参数
            loss, _ = model(batch, pop)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"  [训练完成] Avg Loss: {total_loss / steps:.4f}")

    # --- Plot Results ---
    print("\n>> 正在绘图...")
    plt.figure(figsize=(12, 6))
    x_axis = range(len(history['Cold_R@10']))

    plt.plot(x_axis, history['Cold_R@10'], marker='o', label='New Courses (Cold)', color='blue', linewidth=2)
    plt.plot(x_axis, history['Hot_R@10'], marker='s', label='Mature Courses (Hot)', color='red', linestyle='--',
             linewidth=2, alpha=0.7)

    plt.title('PAM Performance Comparison: Cold vs Hot')
    plt.xlabel('Periods (Months)')
    plt.ylabel('Recall@10')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.axvline(x=warmup - 0.5, color='gray', linestyle=':', label='End of Warmup')

    plt.savefig('mooc_comparison.png')
    print(">> 对比图已保存至: mooc_comparison.png")

    df_res = pd.DataFrame(history)
    df_res.index.name = 'Period'
    df_res.to_csv('mooc_metrics.csv')
    print(">> 数值结果已保存至: mooc_metrics.csv")


if __name__ == "__main__":
    main()
