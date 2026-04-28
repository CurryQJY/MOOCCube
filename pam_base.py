import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import pickle
import random
import matplotlib

# 防止绘图报错
matplotlib.use('Agg')
from torch.utils.data import Dataset, DataLoader


# ================= 1. 基础配置与工具 =================

def setup_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Random Seed Fixed: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        # 维度配置
        self.num_users = n_users
        self.num_items = n_items
        self.user_dim = 64
        self.behavior_dim = 64
        self.content_dim = content_dim
        self.hidden_dims = [128, 64]  # MLP 结构

        # 训练参数
        self.cold_threshold = 5  # 冷启动阈值
        self.lambda_cold = 2.0  # Meta-Learning 冷启动权重
        self.lambda_hot = 0.5

        self.inner_lr = 0.001  # MAML 内层更新率
        self.outer_lr = 0.0005  # 全局更新率
        self.temp = 0.1  # Softmax 温度

        # PAM Base 特有配置
        self.dropout_prob = 0.1  # 训练时随机 Mask ID 的概率


# ================= 2. 辅助数据集 =================

class SimpleItemDataset(Dataset):
    """用于遍历所有物品 ID 的辅助 Dataset"""

    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


# ================= 3. 主模型: PAM (Base) =================

class PAM_Base(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Embeddings ---
        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        # 冻结 BERT 向量
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # --- Projections ---
        # 1. Content Projector
        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim, 256), nn.ReLU(),
            nn.Linear(256, config.behavior_dim), nn.LayerNorm(config.behavior_dim)
        )
        # 2. ID Projector
        self.id_proj = nn.Sequential(
            nn.Linear(config.behavior_dim, config.behavior_dim),
            nn.LayerNorm(config.behavior_dim)
        )
        # Base 模型没有 LLM Projector!

        # --- MAML Meta-Parameters (双塔 MLP) ---
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        dims = [config.user_dim] + config.hidden_dims

        # 初始化两组 MLP 参数 (User + Item)
        for _ in range(2):  # 0: User, 1: Item
            for i in range(len(dims) - 1):
                w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
                nn.init.xavier_normal_(w)
                b = nn.Parameter(torch.zeros(dims[i + 1]))
                self.vars.extend([w, b])
                self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                                  nn.Parameter(torch.ones_like(b) * config.inner_lr)])

    def forward_mlp(self, x, weights, is_item=False):
        """双塔 MLP 通用入口"""
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        num_layers = len(self.cfg.hidden_dims)
        for i in range(num_layers):
            w, b = weights[idx_start + 2 * i], weights[idx_start + 2 * i + 1]
            out = F.linear(out, w, b)
            if i < num_layers - 1:
                out = F.relu(out)
        return out, None

    def get_item_vector(self, i, force_cold=False):
        """
        特征融合逻辑: Content + ID (Base 模型没有 LLM)
        force_cold=True 时，强制 Mask 掉 ID，此时只剩 Content
        """
        # 1. Content Feature
        con_raw = self.item_con_emb(i)
        feat_con = self.con_proj(con_raw)

        # 2. ID Feature (Dropout or Force Cold)
        id_raw = self.item_beh_emb(i)

        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            # 强制冷启动或随机 Dropout -> ID = 0
            zeros = torch.zeros_like(id_raw)
            feat_id = self.id_proj(zeros)
        else:
            feat_id = self.id_proj(id_raw)

        # 融合: 只有 Content + ID
        return feat_con + feat_id

    def inner_loop(self, u, i):
        """MAML Inner Loop"""
        e_u = self.user_emb(u)

        # Base 模型 inner loop 不需要 llm_s
        feat_i = self.get_item_vector(i)

        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(feat_i, self.vars, True)

        # Loss
        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))

        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop, llm_s=None):
        # Base 模型实际上不需要 llm_s，但为了接口统一保留参数
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold

        total_loss = 0
        task_splits = {}
        if is_cold.sum() >= 4: task_splits['cold'] = is_cold
        if (~is_cold).sum() >= 4: task_splits['hot'] = ~is_cold

        for name, mask in task_splits.items():
            u_mask, i_mask = u[mask], i[mask]

            split = len(u_mask) // 2
            su, si = u_mask[:split], i_mask[:split]
            qu, qi = u_mask[split:], i_mask[split:]

            # 1. Inner Loop
            omega = self.inner_loop(su, si)

            # 2. Prediction
            e_u_q = self.user_emb(qu)
            feat_i_q = self.get_item_vector(qi)  # Query Set

            z_u, _ = self.forward_mlp(e_u_q, omega, False)
            z_i, _ = self.forward_mlp(feat_i_q, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        return total_loss


# ================= 4. 全量排名逻辑 (Full Ranking) =================

def precompute_pam_base(model, num_items, batch_size=1024, device='cuda'):
    """
    [PAM_Base] 预计算全量背景库
    逻辑：强制 Cold (ID=0), 仅使用 Content
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ [PAM_Base] Pre-computing Full Item Pool (Content Only)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)

            # force_cold=True 保证 ID 被 mask
            feat_i = model.get_item_vector(i_batch, force_cold=True)

            # 通过 MLP (使用全局 vars)
            z_i, _ = model.forward_mlp(feat_i, model.vars, True)

            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_full_pam_base(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    """
    全量评估
    因为没有 LLM，正样本向量和库里的向量其实是一样的。
    """
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU Memory Full, fallback to CPU.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    print("🚀 Running Full Ranking Eval (Symmetric)...")

    with torch.no_grad():
        for batch, pop, _ in loader:  # 不需要 llm_s
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)

            batch_size = u.size(0)

            # 1. User Vector
            e_u = model.user_emb(u)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)

            # 2. Positive Item Vector
            # 重新计算正样本 (逻辑上与 precompute 一致)
            feat_pos = model.get_item_vector(i_target, force_cold=True)
            z_i_pos, _ = model.forward_mlp(feat_pos, model.vars, True)

            # 3. 计算分数 & 替换
            if cpu_mode: z_u = z_u.cpu()

            # (A) Background Scores
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # (B) Positive Scores
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)

            # (C) Replace
            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

            # 4. Metrics
            max_k = max(k_list)
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            target_cols = target_cols.view(-1, 1)

            for k in k_list:
                preds = topk_indices[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
                metrics_sum[f'R@{k}'] += hits.sum().item()

                hit_ranks = torch.where(preds == target_cols)
                if hit_ranks[1].numel() > 0:
                    dcg = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                    metrics_sum[f'N@{k}'] += dcg.sum().item()

            total_samples += batch_size
            if cpu_mode: z_u = z_u.to(device)

    if total_samples == 0: return None, 0
    return {k: v / total_samples for k, v in metrics_sum.items()}, total_samples


# ================= 5. 数据处理与主程序 =================

class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        # Base 不需要 llm_s

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    return {'u': u, 'i': i}, pop, None  # llm_s is None


def main():
    setup_seed(2025)
    print(">>> 1. Loading Data...")

    if not os.path.exists("processed_data/stream_data.pkl"):
        print("❌ Error: Data not found.")
        return

    df = pd.read_pickle("processed_data/stream_data.pkl")
    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    content_emb = torch.load("processed_data/content_emb.pt")

    # Base Model 不需要 llm_scores.pkl

    # Split Periods
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['dt'] = df['timestamp']
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p), batch_size=512, collate_fn=collate_fn, shuffle=True)
               for p in periods if len(p) > 0]

    # Initialize Model
    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(">>> Initializing PAM (Base)...")
    model = PAM_Base(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # Metrics Setup
    target_k = [5, 10, 20]
    metric_names = [f'{m}@{k}' for k in target_k for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metric_names}
    global_count = 0

    print(f"\n>>> Start Training PAM_Base (Full Ranking Mode) <<<")
    WARMUP = 2

    for t, loader in enumerate(loaders):
        print(f"\n--- Period {t} (Samples: {len(loader.dataset)}) ---")

        # === Evaluation ===
        if t >= WARMUP:
            # 1. Precompute Full Pool
            all_item_z = precompute_pam_base(model, cfg.num_items, device=device)
            # 2. Evaluate
            met, n = evaluate_full_pam_base(model, loader, all_item_z, device, k_list=target_k)

            if met:
                res_str = " | ".join([f"{k}={met[k]:.4f}" for k in metric_names])
                print(f"📊 Eval: {res_str}")
                for k in metric_names:
                    global_accum[k] += met[k] * n
                global_count += n

        # === Training ===
        model.train()
        total_loss = 0
        for batch_idx, (batch, pop, _) in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # llm_s is None
            loss = model(batch, pop.to(device), llm_s=None)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"   Avg Loss: {total_loss / len(loader):.4f}")

    # === Final Report ===
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT (PAM_Base - Micro Avg)")
    print("-" * 60)
    if global_count > 0:
        for k in metric_names:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()