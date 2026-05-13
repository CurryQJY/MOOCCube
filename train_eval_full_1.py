import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import pickle, random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ================= 1. 基础设置 =================
def setup_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 随机种子已固定: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = 64
        self.content_dim = content_dim
        self.hidden_dim = 128
        self.cold_threshold = 5

        # --- 核心超参数 ---
        self.lr = 1e-3
        self.dropout_prob = 0.1  # [策略1] ID 丢弃率
        self.aux_weight = 0.1  # [策略3] 对比损失权重
        self.distill_weight = 0.85  # [策略4] LLM 蒸馏损失权重


# ================= 2. 数据集 (保持不变) =================
class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        # 初始化 LLM 分数 (-1 表示无)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)

        if llm_map:
            keys = list(zip(df['u_idx'], df['i_idx']))
            # 处理 map 中可能没有 key 的情况
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


def split_dataframe_by_periods(df, period_type='M'):
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = df['timestamp']
    df['period_id'] = df['datetime'].dt.to_period(period_type)
    periods = []
    sorted_period_keys = sorted(df['period_id'].unique())
    for p_key in sorted_period_keys:
        periods.append(df[df['period_id'] == p_key].reset_index(drop=True))
    return periods


def compute_ranking_metrics(scores, target_indices, k_list=[5, 10, 20]):
    """
    全库排名评估指标计算。
    Args:
        scores: [B, N_items] 每个用户对所有物品的打分
        target_indices: [B] 每个用户的真实物品索引 (在全库中的 item idx)
        k_list: 评估的 K 值列表
    """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)  # [B, 1]
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        results[f'R@{k}'] = hits.mean().item()
        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results


# ================= 3. 增强版 PAM_LLM 模型 =================

class PAM_LLM_Enhanced(nn.Module):
    def __init__(self, cfg, content_emb):
        super(PAM_LLM_Enhanced, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. User & Item ID Embedding
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        # 2. Content Encoder
        self.content_features = content_emb.to(self.device)
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim)
        )

        # 3. [新增] LLM Feature Encoder
        # 将标量 llm_score 映射为向量，以便与 Content 融合
        self.llm_proj = nn.Sequential(
            nn.Linear(1, cfg.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.emb_dim // 2, cfg.emb_dim)
        )

        # 4. [策略2] 门控融合 (Adaptive Gated Fusion)
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid()
        )

        self.temperature = 0.1

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        """
        生成物品向量：融合 ID + (Content + LLM)
        """
        # A. ID Embedding & [策略1] Input Perturbation
        id_e = self.item_id_emb(i_idx)
        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        # B. Content Embedding
        raw_content = self.content_features[i_idx]
        content_e = self.content_proj(raw_content)

        # C. [新增] LLM Embedding 注入
        # 处理 llm_s: 将 -1 (缺失) 视为 0，并扩展维度
        # mask: [B, 1], 1 if present, 0 if missing
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)  # [B, 1]

        llm_e = self.llm_proj(val_llm) * mask_llm  # 如果缺失，则 LLM 向量为 0

        # D. 语义组合: Content + LLM
        # 这里我们将 Content 和 LLM 视为“广义内容”
        semantic_e = content_e + llm_e

        # E. [策略2] 门控融合 ID vs Semantic
        concat = torch.cat([id_e, semantic_e], dim=-1)
        alpha = self.gate_net(concat)

        final_item_e = alpha * id_e + (1 - alpha) * semantic_e

        return final_item_e, id_e, semantic_e

    def forward(self, batch, pop, llm_s):
        u_idx = batch['u']
        i_idx = batch['i']

        # 1. Inference
        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)

        z_i, id_e_raw, semantic_e = self.get_item_vector(i_idx, llm_s, force_cold=False)
        z_i = F.normalize(z_i, dim=1)

        # 2. Main Loss (Ranking)
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        main_loss = F.cross_entropy(logits, labels)

        if self.training:
            # 3. [策略3] Contrastive Aux Loss (ID <-> Semantic)
            z_id = F.normalize(id_e_raw, dim=1)
            z_sem = F.normalize(semantic_e, dim=1)
            sim_matrix = torch.matmul(z_id, z_sem.t()) / 0.1
            aux_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2

            # 4. [策略4] LLM Distillation Loss (知识蒸馏)
            # 目的：让模型的预测得分 (logits 对角线) 逼近 LLM 的打分
            # 只对有 LLM 分数的样本计算
            mask_llm = (llm_s > -0.5)
            if mask_llm.sum() > 0:
                # 获取正例预测分数 (Batch 对角线)
                pos_scores = torch.sum(z_u * z_i, dim=1)  # [B]
                # LLM 分数通常是 0-1 或 1-5，这里假设归一化后的分数
                # 如果 llm_s 范围不一致，可以加 sigmoid 或 scale
                targets = llm_s[mask_llm]
                preds = pos_scores[mask_llm]

                # 使用 MSE 让预测值逼近 LLM 值
                distill_loss = F.mse_loss(preds, targets)
            else:
                distill_loss = 0.0

            total_loss = main_loss + self.cfg.aux_weight * aux_loss + self.cfg.distill_weight * distill_loss
            return total_loss, None

        return main_loss, None


# ================= 4. 评估函数 (适配 Enhanced) =================

def evaluate_enhanced(model, loader, device, k_list=[5, 10, 20], n_neg=999):
    """
    采样评估：对每个冷启动用户，从 1 个正样本 + n_neg 个随机负样本中排名。
    这是 NCF / LightGCN 等论文中广泛使用的标准评估协议。
    """
    model.eval()
    accum_metrics = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0

    with torch.no_grad():
        # ====== Step 1: 预计算全库物品向量 ======
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        all_llm_s = torch.full((n_items,), -1.0, device=device)

        ITEM_BATCH = 4096
        all_item_vecs = []
        for start in range(0, n_items, ITEM_BATCH):
            end = min(start + ITEM_BATCH, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]
            z_batch, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=True)
            z_batch = F.normalize(z_batch, dim=1)
            all_item_vecs.append(z_batch)
        all_item_vecs = torch.cat(all_item_vecs, dim=0)  # [N_items, emb_dim]

        # ====== Step 2: 采样评估 ======
        for batch, pop, llm_s in loader:
            mask = pop < model.cfg.cold_threshold
            n_batch_cold = mask.sum().item()
            if n_batch_cold < 1:
                continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)  # 正样本 item idx

            z_u = model.user_emb(u)
            z_u = F.normalize(z_u, dim=1)

            # 为每个用户采样 n_neg 个负样本
            neg_items = torch.randint(0, n_items, (n_batch_cold, n_neg), device=device)
            # 候选集: [正样本, neg_1, neg_2, ..., neg_999]
            # 正样本放在第 0 位，target 始终为 0
            cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)  # [B, 1+n_neg]
            cand_vecs = all_item_vecs[cand_idx]  # [B, 1+n_neg, dim]

            # 计算得分: [B, 1+n_neg]
            scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)

            # target 始终在第 0 位
            target_indices = torch.zeros(n_batch_cold, dtype=torch.long, device=device)
            res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)

            for k, v in res.items():
                accum_metrics[k] += v * n_batch_cold
            total_cold_samples += n_batch_cold

    if total_cold_samples == 0:
        return None, 0
    return {k: v / total_cold_samples for k, v in accum_metrics.items()}, total_cold_samples


# ================= 5. 主流程 =================

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

    periods = split_dataframe_by_periods(df)
    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=2048, collate_fn=collate_fn) for p in periods if
               len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    print(f">> Model: PAM_LLM_Enhanced | Drop: {cfg.dropout_prob} | Distill: {cfg.distill_weight}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_LLM_Enhanced(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    target_metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}
    global_accum = {k: 0.0 for k in target_metrics}
    global_cold_count = 0

    print(f"\n>>> 开始评估 (共 {len(loaders)} 个周期) <<<")
    WARMUP = 3

    for t, loader in enumerate(loaders):
        # --- Test ---
        if t >= WARMUP:
            met, n_samples = evaluate_enhanced(model, loader, device)
            if met:
                print(f"Period {t:<3} (n={n_samples:<4}): R@10={met['R@10']:.4f} | N@10={met['N@10']:.4f}")
                for k in target_metrics:
                    history[k].append(met[k])
                    global_accum[k] += met[k] * n_samples
                global_cold_count += n_samples
            else:
                for k in target_metrics: history[k].append(0)
        else:
            for k in target_metrics: history[k].append(0)

        # --- Train ---
        model.train()
        total_loss = 0
        steps = 0
        for batch, pop, llm_s in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward 内部包含 loss 计算
            loss, _ = model(batch, pop.to(device), llm_s.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # print(f"   Train Loss: {total_loss/steps:.4f}")

    # ================= 结果 =================
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (PAM_LLM Enhanced)")
    print("=" * 60)
    print(f"{'Metric':<10} | {'Macro Avg':<18} | {'Micro Avg':<18}")
    print("-" * 60)

    final_res = {}
    for m in target_metrics:
        valid_vals = [v for i, v in enumerate(history[m]) if i >= WARMUP and v > 0]
        macro = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        micro = global_accum[m] / global_cold_count if global_cold_count > 0 else 0.0
        print(f"{m:<10} | {macro:.4f}             | {micro:.4f}")
        final_res[m] = micro

    pd.DataFrame(history).to_csv('metrics_pam_llm_final.csv')

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(history['R@10'], label='Recall@10', marker='o')
    plt.plot(history['N@10'], label='NDCG@10', marker='s')
    plt.axvline(x=WARMUP - 0.5, color='red', linestyle='--', label='Warmup End')
    plt.title(f'PAM LLM Enhanced (Micro R@10: {final_res["R@10"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('result_pam_llm_final.png')
    print(">> 完成。结果保存至 result_pam_llm_final.png")


if __name__ == "__main__":
    setup_seed(2025)
    main()