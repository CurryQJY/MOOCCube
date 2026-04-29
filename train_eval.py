import torch
import pandas as pd
import numpy as np
import json
import os, random

# ==============================
# 1. 强制使用非交互式后端 (防止画图报错)
# ==============================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pam_model import PAM, Config
from processed_data_utils import load_processed_bundle


def setup_seed(seed=2025):
    """
    一键固定所有随机种子，确保实验可复现
    """
    # 1. Python 原生随机数
    random.seed(seed)
    # 2. Numpy 随机数 (用于 pandas 和 np 操作)
    np.random.seed(seed)
    # 3. PyTorch CPU 随机数
    torch.manual_seed(seed)
    # 4. PyTorch GPU 随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 5. 强制 CuDNN 使用确定性算法 (会慢一点，但结果唯一)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. 设置环境变量 (防止 Hash 随机化)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Seed fixed: {seed}")


# ==========================================
# 2. 数据集与切分工具
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
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    # 返回元组以便解包
    return {'u': u, 'i': i}, pop


def split_dataframe_by_periods(df, period_type='M'):
    """ 按月(M)或周(W)切分数据 """
    # 确保时间戳是 datetime 格式
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = df['timestamp']

    df['period_id'] = df['datetime'].dt.to_period(period_type)

    periods = []
    sorted_period_keys = sorted(df['period_id'].unique())
    print(f">> 数据已切分为 {len(sorted_period_keys)} 个周期 ({period_type})。")

    for p_key in sorted_period_keys:
        period_df = df[df['period_id'] == p_key].reset_index(drop=True)
        periods.append(period_df)
    return periods


def compute_ranking_metrics(scores, k_list=[5, 10, 20]):
    """ 计算 Batch 内的平均指标 """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)

    targets = torch.arange(batch_size).to(scores.device).view(-1, 1)

    # 动态调整 max_k
    max_k = max(k_list)
    actual_k = min(max_k, num_candidates)

    _, topk_indices = torch.topk(scores, actual_k, dim=1)

    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
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
# 3. 核心评估函数 (Evaluate) - 最终版
# ==========================================

def evaluate(model, loader, device):
    """
    计算当前周期的指标。
    返回: (metrics_dict, sample_count)
    """
    model.eval()

    # 使用累加器，而不是简单的平均，以处理不同 Batch 大小不一致的问题
    accum_metrics = {}
    total_cold_samples = 0

    target_k_list = [5, 10, 20]

    with torch.no_grad():
        for batch, pop in loader:
            # 1. 筛选冷启动样本
            mask_cold = pop < model.cfg.cold_threshold
            n_cold = mask_cold.sum().item()

            # 样本太少无法计算 Batch 内负采样，跳过该 Batch
            if n_cold < 2: continue

            u_test = batch['u'][mask_cold].to(device)
            i_test = batch['i'][mask_cold].to(device)

            # 2. 模型预测
            e_u = model.user_emb(u_test)
            # 使用临时参数 (Theta 近似 Omega)
            weights = model.vars
            # 注意：此处必须使用 correct function name
            e_i = model.get_item_features(i_test)

            z_u, _ = model.forward_mlp(e_u, weights, False)
            z_i, _ = model.forward_mlp(e_i, weights, True)

            # 3. 计算分数与指标
            scores = torch.mm(z_u, z_i.t())

            # batch_res 返回的是该 Batch 的平均值 (Mean)
            batch_res = compute_ranking_metrics(scores, k_list=target_k_list)

            # 4. 加权累加 (Mean * N = Sum)
            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_cold

            total_cold_samples += n_cold

    if total_cold_samples == 0:
        return None, 0

    # 计算该周期的加权平均值
    avg_metrics = {k: v / total_cold_samples for k, v in accum_metrics.items()}
    return avg_metrics, total_cold_samples


# ==========================================
# 4. 主训练循环
# ==========================================

def main():
    print("Loading Data...")
    try:
        data_dir, meta, df, content_emb = load_processed_bundle()
    except FileNotFoundError as exc:
        print(exc)
        return
    print(f">> Data Dir: {data_dir}")

    # 按月切分
    periods = split_dataframe_by_periods(df, period_type='M')

    dataloaders = []
    for p_df in periods:
        ds = StreamDataset(p_df)
        dl = DataLoader(ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        dataloaders.append(dl)

    cfg = Config(meta['n_users'], meta['n_items'])

    # [关键] 维度自适应覆盖
    cfg.content_dim = content_emb.shape[1]
    print(f">> Content Embedding Dim: {cfg.content_dim}")

    # [优化参数]
    cfg.gamma_s = 5.0
    cfg.lambda_cold = 2.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    print(f"\n>>> 开始滚动评估 - 共 {len(periods)} 个周期 <<<")

    # [修改点 1] 重新定义指标顺序：先Recall，再NDCG
    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]
    # 结果现在会是: ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']

    history = {'Period': [], 'Count': []}
    for k in metrics_keys: history[k] = []

    WARMUP_PERIODS = 3

    for t, loader in enumerate(dataloaders):
        n_total = len(loader.dataset)
        print(f"\n>>> Period {t} (Total Samples: {n_total}) <<<")

        # --- Phase 1: Test ---
        current_res = {k: 0.0 for k in metrics_keys}
        test_count = 0

        if t >= WARMUP_PERIODS:
            metrics, n_cold = evaluate(model, loader, device)

            if metrics:
                current_res = metrics
                test_count = n_cold
                print(f"  [TEST] (n={n_cold}) R@10: {metrics['R@10']:.4f} | N@10: {metrics['N@10']:.4f}")
            else:
                print("  [SKIP] No cold items to test.")
        else:
            print("  [WARMUP] Training only...")

        # 记录
        history['Period'].append(t)
        history['Count'].append(test_count)
        for k in metrics_keys:
            history[k].append(current_res.get(k, 0.0))

        # --- Phase 2: Train ---
        model.train()
        total_loss = 0
        steps = 0

        for batch, pop in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pop = pop.to(device)
            loss, _ = model(batch, pop)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        print(f"  [TRAIN] Avg Loss: {total_loss / steps:.4f}")

    # ==========================================
    # 5. 生成加权平均报告 (Final Weighted Report)
    # ==========================================
    print("\n" + "=" * 65)
    print("             FINAL EXPERIMENT REPORT (NeurIPS)             ")
    print("=" * 65)

    counts = np.array(history['Count'])
    # 有效周期：非预热期 且 有测试样本
    valid_mask = (np.array(history['Period']) >= WARMUP_PERIODS) & (counts > 0)

    if valid_mask.sum() > 0:
        valid_counts = counts[valid_mask]
        total_samples = valid_counts.sum()

        print(f"Valid Periods: {valid_mask.sum()}")
        print(f"Total Cold Samples Tested: {total_samples}")
        print("-" * 65)
        print(f"{'Metric':<10} | {'Weighted Avg (Official)':<25} | {'Simple Avg'}")
        print("-" * 65)

        # [修改点 2] 分组打印 Recall 和 NDCG
        print(">>> Recall Metrics:")
        recall_keys = [k for k in metrics_keys if k.startswith('R')]
        for k in recall_keys:
            scores = np.array(history[k])[valid_mask]
            weighted_avg = np.average(scores, weights=valid_counts)
            simple_avg = np.mean(scores)
            print(f"{k:<10} | {weighted_avg:.4f}                    | {simple_avg:.4f}")

        print("-" * 65)
        print(">>> NDCG Metrics:")
        ndcg_keys = [k for k in metrics_keys if k.startswith('N')]
        for k in ndcg_keys:
            scores = np.array(history[k])[valid_mask]
            weighted_avg = np.average(scores, weights=valid_counts)
            simple_avg = np.mean(scores)
            print(f"{k:<10} | {weighted_avg:.4f}                    | {simple_avg:.4f}")

        print("-" * 65)
    else:
        print("Warning: No valid test data found.")

    # --- 保存图表 ---
    plt.figure(figsize=(10, 6))
    plt.plot(history['Period'], history['R@10'], marker='o', label='Recall@10')
    plt.plot(history['Period'], history['N@10'], marker='s', label='NDCG@10')
    plt.axvline(x=WARMUP_PERIODS - 0.5, color='r', linestyle='--', label='Warmup End')
    plt.title('PAM Performance Trend')
    plt.xlabel('Period')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('mooc_result_final.png')
    print("\n>> Figure saved to mooc_result_final.png")

    # 保存 CSV
    # 此时 history 中的 key 顺序已经是：Period, Count, R@5, R@10, R@20, N@5, N@10, N@20
    pd.DataFrame(history).to_csv('mooc_metrics_final.csv', index=False)


if __name__ == "__main__":
    setup_seed(20)
    main()
