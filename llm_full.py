import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import pickle, random

# ================= 1. 强制非交互后端 =================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ================= 2. 基础设置 =================

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


# ================= 3. 数据集 =================

class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)

        if llm_map:
            # 优化: 向量化处理或预处理 map
            # 这里保持简单循环
            keys = list(zip(df['u_idx'], df['i_idx']))
            vals = [llm_map.get(k, -1.0) for k in keys]
            self.llm_s = torch.tensor(vals, dtype=torch.float32)

    def __len__(self): return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm_s': self.llm_s[idx]}


# [新增] 全量物品遍历 Dataset
class SimpleItemDataset(Dataset):
    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


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


# ================= 4. 增强版 PAM_LLM 模型 =================

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

        # 3. LLM Feature Encoder
        self.llm_proj = nn.Sequential(
            nn.Linear(1, cfg.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.emb_dim // 2, cfg.emb_dim)
        )

        # 4. Adaptive Gated Fusion
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid()
        )

        self.temperature = 0.1

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        """
        生成物品向量：融合 ID + (Content + LLM)
        force_cold=True: 强制 Mask ID
        """
        # A. ID Embedding
        id_e = self.item_id_emb(i_idx)
        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        # B. Content Embedding
        raw_content = self.content_features[i_idx]
        content_e = self.content_proj(raw_content)

        # C. LLM Embedding 注入
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm

        # D. 语义组合
        semantic_e = content_e + llm_e

        # E. 门控融合
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

        # 2. Main Loss
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        main_loss = F.cross_entropy(logits, labels)

        if self.training:
            # 3. Aux Loss
            z_id = F.normalize(id_e_raw, dim=1)
            z_sem = F.normalize(semantic_e, dim=1)
            sim_matrix = torch.matmul(z_id, z_sem.t()) / 0.1
            aux_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2

            # 4. Distill Loss
            mask_llm = (llm_s > -0.5)
            if mask_llm.sum() > 0:
                pos_scores = torch.sum(z_u * z_i, dim=1)
                targets = llm_s[mask_llm]
                preds = pos_scores[mask_llm]
                distill_loss = F.mse_loss(preds, targets)
            else:
                distill_loss = 0.0

            total_loss = main_loss + self.cfg.aux_weight * aux_loss + self.cfg.distill_weight * distill_loss
            return total_loss, None

        return main_loss, None


# ================= 5. 全量排名相关函数 =================

def precompute_full_pool(model, num_items, batch_size=2048, device='cuda'):
    """
    [新增] 预计算全库物品向量 (背景库)
    策略:
    1. force_cold=True (模拟冷启动/纯内容检索)
    2. llm_s = -1.0 (假设背景库中的物品没有特定的 Prior Score)
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    # 构造一个全 -1 的 dummy LLM score
    # 只需要构造一次，batch 中复用

    print("⏳ Pre-computing Full Item Pool (Force Cold, No LLM)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)

            # 构造 dummy llm score (-1)
            dummy_llm = torch.full((i_batch.size(0),), -1.0, device=device)

            # 获取向量
            z_i, _, _ = model.get_item_vector(i_batch, dummy_llm, force_cold=True)
            z_i = F.normalize(z_i, dim=1)

            all_z_i.append(z_i.cpu())  # 存到 CPU

    return torch.cat(all_z_i, dim=0)


def evaluate_full_enhanced(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    """
    全量排名评估 (不对称评估)
    """
    model.eval()
    accum_metrics = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0

    # 尝试 GPU 加速
    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU OOM, ranking on CPU.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    with torch.no_grad():
        for batch, pop, llm_s in loader:
            # 1. 筛选
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            l_s = llm_s[mask].to(device)  # [关键] 获取真实的 LLM Score

            batch_size = u.size(0)

            # 2. User Vector
            z_u = model.user_emb(u)
            z_u = F.normalize(z_u, dim=1)

            # 3. Positive Item Vector (特异化)
            # 使用 真实 LLM Score + Force Cold
            z_i_pos, _, _ = model.get_item_vector(i_target, l_s, force_cold=True)
            z_i_pos = F.normalize(z_i_pos, dim=1)

            # 4. 计算全量分数
            if cpu_mode: z_u = z_u.cpu()

            # 背景分: User vs All Pool (Pool 是用 LLM=-1 算的)
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # 5. 替换逻辑 (Asymmetric Replacement)
            # 将正样本位置的分数，替换为用真实 LLM Score 算出来的“特异化分数”
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)

            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

            # 6. Metrics
            max_k = max(k_list)
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            target_cols = target_cols.view(-1, 1)

            for k in k_list:
                preds = topk_indices[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
                accum_metrics[f'R@{k}'] += hits.sum().item()

                hit_ranks = torch.where(preds == target_cols)
                if hit_ranks[1].numel() > 0:
                    dcg = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                    accum_metrics[f'N@{k}'] += dcg.sum().item()

            total_cold_samples += batch_size
            if cpu_mode: z_u = z_u.to(device)

    if total_cold_samples == 0: return None, 0
    return {k: v / total_cold_samples for k, v in accum_metrics.items()}, total_cold_samples


# ================= 6. 主流程 =================

def main():
    setup_seed(2025)
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

    # 调小 batch_size 适应全量评估
    periods = split_dataframe_by_periods(df)
    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=512, collate_fn=collate_fn) for p in periods if
               len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: PAM_LLM_Enhanced (Full Ranking) | Emb Dim: {cfg.emb_dim}")

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
        print(f"\n--- Period {t} (Samples: {len(loader.dataset)}) ---")

        # --- Test (Full Ranking) ---
        current_res = {k: 0.0 for k in target_metrics}

        if t >= WARMUP:
            # 1. 预计算全库
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            # 2. 全量评估
            met, n_samples = evaluate_full_enhanced(model, loader, all_z, device)

            if met:
                res_str = " | ".join([f"{k}={met[k]:.4f}" for k in target_metrics])
                print(f"📊 Eval: {res_str}")
                for k in target_metrics:
                    history[k].append(met[k])
                    global_accum[k] += met[k] * n_samples
                global_cold_count += n_samples
            else:
                print("  [SKIP] No cold items to test.")
                for k in target_metrics: history[k].append(0)
        else:
            print("  [WARMUP] Training only...")
            for k in target_metrics: history[k].append(0)

        # --- Train ---
        model.train()
        total_loss = 0
        steps = 0
        for batch, pop, llm_s in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, pop.to(device), llm_s.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"   Train Avg Loss: {total_loss / max(1, steps):.4f}")

    # ================= 结果 =================
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (PAM_LLM Enhanced - Full Ranking)")
    print("=" * 60)

    if global_cold_count > 0:
        print(f"{'Metric':<10} | {'Micro Avg':<18}")
        print("-" * 60)
        final_res = {}
        for m in target_metrics:
            micro = global_accum[m] / global_cold_count
            print(f"{m:<10} | {micro:.4f}")
            final_res[m] = micro

        # 保存 CSV
        pd.DataFrame(history).to_csv('metrics_pam_llm_full.csv', index=False)
        print("\n>> Saved metrics_pam_llm_full.csv")
    else:
        print("No evaluation performed.")


if __name__ == "__main__":
    main()