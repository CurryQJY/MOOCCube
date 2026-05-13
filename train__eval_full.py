import torch
import pandas as pd
import numpy as np
import json
import os
import pickle, random
import matplotlib

matplotlib.use('Agg')  # 防止无显示器报错
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pam_model_llm import PAM_LLM, Config


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

    print(f"✅ 随机种子已固定: {seed}")


# ================= 1. 数据集定义 =================
class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        # 初始化 LLM 分数 (-1 表示无)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)

        if llm_map:
            # 快速映射
            keys = list(zip(df['u_idx'], df['i_idx']))
            # 使用 get 稍微慢一点但安全
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
    # 返回: (数据字典, Pop, LLM分数)
    return {'u': u, 'i': i}, pop, llm_s


# ================= 2. 多指标评估函数 (含加权逻辑) =================
def compute_ranking_metrics(scores, k_list=[5, 10, 20]):
    """ 计算 Batch 内的平均指标 """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = torch.arange(batch_size).to(scores.device).view(-1, 1)

    # 动态调整最大 K
    max_k = max(k_list)
    actual_k = min(max_k, num_candidates)

    _, topk_indices = torch.topk(scores, actual_k, dim=1)

    results = {}
    for k in k_list:
        # 切片自动处理越界
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()

        # Recall
        results[f'R@{k}'] = hits.mean().item()

        # NDCG
        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg

    return results


def evaluate(model, loader, device, k_list=[5, 10, 20]):
    model.eval()

    # 初始化累加器
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0  # 记录本周期的冷启动总数

    with torch.no_grad():
        for batch, pop, _ in loader:
            # 筛选冷启动
            mask = pop < model.cfg.cold_threshold
            n_batch_cold = mask.sum().item()

            if n_batch_cold < 2: continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)

            # 使用模型当前的参数进行预测
            e_u = model.user_emb(u)
            e_i = model.get_item_features(i)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)
            z_i, _ = model.forward_mlp(e_i, model.vars, True)

            scores = torch.mm(z_u, z_i.t())

            # 计算 Batch 平均指标
            res = compute_ranking_metrics(scores, k_list=k_list)

            # 【核心修改】累加 Sum = Mean * Count
            # 这样可以在最后算出精确的加权平均
            for k, v in res.items():
                metrics_sum[k] += v * n_batch_cold

            total_cold_samples += n_batch_cold

    if total_cold_samples == 0: return None, 0

    # 计算本周期的平均值 (Macro for this period)
    period_metrics = {k: v / total_cold_samples for k, v in metrics_sum.items()}

    # 返回: (平均指标字典, 样本总数)
    return period_metrics, total_cold_samples


# ================= 3. 主流程 =================
def main():
    print("1. 加载数据...")
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("Error: 请先运行 data_process.py")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    llm_map = None
    if os.path.exists("processed_data/llm_scores.pkl"):
        print("   加载本地 LLM 分数...")
        with open("processed_data/llm_scores.pkl", "rb") as f: llm_map = pickle.load(f)

    # 切分周期
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=2048, collate_fn=collate_fn) for p in periods if
               len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_LLM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # 初始化记录器
    target_metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}

    # 全局累加器 (用于计算 Micro Average)
    global_accum = {k: 0.0 for k in target_metrics}
    global_cold_count = 0

    print(f"\n>>> 开始评估 (共 {len(loaders)} 个周期) <<<")
    WARMUP = 3

    for t, loader in enumerate(loaders):
        # --- 1. 评估 (Test Phase) ---
        if t >= WARMUP:
            # 接收 (指标, 样本数)
            met, n_samples = evaluate(model, loader, device, k_list=[5, 10, 20])

            if met:
                # 打印当前周期结果
                print(f"Period {t:<3} (n={n_samples:<4}): R@10={met['R@10']:.4f} | N@10={met['N@10']:.4f}")

                # 记录 Macro 历史 (用于画图)
                for k in target_metrics:
                    history[k].append(met[k])

                # 累加 Micro 全局数据 (指标 * 样本数)
                for k in target_metrics:
                    global_accum[k] += met[k] * n_samples
                global_cold_count += n_samples

            else:
                # 本周期无冷启动样本
                for k in target_metrics: history[k].append(0)
        else:
            # 预热期
            for k in target_metrics: history[k].append(0)

        # --- 2. 训练 (Train Phase) ---
        model.train()
        total_loss = 0
        steps = 0
        for batch, pop, llm_s in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, d = model(batch, pop.to(device), llm_s.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # Optional: print(f"   Train Loss: {total_loss/steps:.4f}")

    # ================= 4. 结果汇总与可视化 =================
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (Final Evaluation Report)")
    print("=" * 60)

    # 计算 Macro Average (去除预热期和0值)
    print(f"{'Metric':<10} | {'Macro Avg (Time)':<18} | {'Micro Avg (Weighted)':<18}")
    print("-" * 60)

    final_res = {}

    for m in target_metrics:
        # Macro: 只算有效周期
        valid_vals = [v for i, v in enumerate(history[m]) if i >= WARMUP and v > 0]
        macro_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

        # Micro: 总得分 / 总样本
        micro_val = global_accum[m] / global_cold_count if global_cold_count > 0 else 0.0

        print(f"{m:<10} | {macro_val:.4f}             | {micro_val:.4f}")
        final_res[m] = {'macro': macro_val, 'micro': micro_val}

    print("=" * 60)

    # 绘图 (只画 R@10, R@20, N@10 的 Macro 趋势)
    plt.figure(figsize=(12, 6))
    x_axis = range(len(history['R@10']))
    plt.plot(x_axis, history['R@10'], label='Recall@10', marker='o', markersize=4)
    plt.plot(x_axis, history['R@20'], label='Recall@20', linestyle='--')
    plt.plot(x_axis, history['N@10'], label='NDCG@10', alpha=0.7)

    plt.axvline(x=WARMUP - 0.5, color='gray', linestyle=':', label='Warmup End')
    plt.title(f'PAM+LLM Performance (Weighted R@10: {final_res["R@10"]["micro"]:.4f})')
    plt.xlabel('Period')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('final_metrics.png')

    # 保存详细数据
    pd.DataFrame(history).to_csv('final_metrics.csv')

    # 保存最终摘要到文本文件
    with open('final_summary.txt', 'w') as f:
        f.write("Metric,Macro_Avg,Micro_Avg\n")
        for m in target_metrics:
            f.write(f"{m},{final_res[m]['macro']:.4f},{final_res[m]['micro']:.4f}\n")

    print("完成！结果已保存至 final_metrics.csv, final_metrics.png 和 final_summary.txt")


if __name__ == "__main__":
    setup_seed(20)
    main()
