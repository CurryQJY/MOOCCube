import json
import os
import random

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制非交互式后端
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
        self.num_users = n_users
        self.num_items = n_items
        self.user_dim = 64
        self.behavior_dim = 64
        self.content_dim = content_dim
        self.hidden_dims = [128, 64]

        # 训练参数
        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5
        self.inner_lr = 0.001
        self.outer_lr = 0.0005
        self.temp = 0.1

        # Dropout 特有
        self.dropout_prob = 0.5


class SimpleItemDataset(Dataset):
    """辅助 Dataset，用于遍历所有物品 ID"""

    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


# ================= 2. 模型 A: PAM_Dropout =================
# 逻辑：在训练时随机 Drop 内容特征，强迫模型不过拟合

class PAM_Dropout(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim, 256), nn.ReLU(),
            nn.Linear(256, config.behavior_dim), nn.LayerNorm(config.behavior_dim)
        )
        self.id_proj = nn.Sequential(
            nn.Linear(config.behavior_dim, config.behavior_dim),
            nn.LayerNorm(config.behavior_dim)
        )

        # [核心] 显式 Dropout 层
        self.input_dropout = nn.Dropout(p=config.dropout_prob)

        # MAML Params
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()
        dims = [config.user_dim] + config.hidden_dims
        for _ in range(2):
            for i in range(len(dims) - 1):
                w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
                nn.init.xavier_normal_(w)
                b = nn.Parameter(torch.zeros(dims[i + 1]))
                self.vars.extend([w, b])
                self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                                  nn.Parameter(torch.ones_like(b) * config.inner_lr)])

    def forward_mlp(self, x, weights, is_item=False):
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
        全量排名核心逻辑:
        force_cold=True -> 关闭 Dropout (使用完整特征), Mask ID (全0)
        """
        con_raw = self.item_con_emb(i)

        # 训练模式下且非强制冷启动，才加 Noise
        if self.training and not force_cold:
            con_raw = self.input_dropout(con_raw)

        feat_con = self.con_proj(con_raw)

        # ID 处理
        id_raw = self.item_beh_emb(i)
        if force_cold or (self.training and random.random() < 0.5):
            zeros = torch.zeros_like(id_raw)
            feat_id = self.id_proj(zeros)
        else:
            feat_id = self.id_proj(id_raw)

        return feat_con + feat_id

    def inner_loop(self, u, i):
        e_u = self.user_emb(u)
        feat_i = self.get_item_vector(i)
        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(feat_i, self.vars, True)

        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))
        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop, _):
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

            omega = self.inner_loop(su, si)

            e_u_q = self.user_emb(qu)
            feat_i_q = self.get_item_vector(qi)
            z_u, _ = self.forward_mlp(e_u_q, omega, False)
            z_i, _ = self.forward_mlp(feat_i_q, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        return total_loss


# ================= 3. 模型 B: PAM_GAR (Generative Augmented) =================
# 逻辑：使用 Generator 从 Content 生成 Pseudo-ID

class PAM_GAR(PAM_Dropout):  # 继承大部分逻辑
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        # 覆盖掉 Dropout，GAR 不需要强 Input Dropout
        self.input_dropout = nn.Identity()

        # [核心] Generator: Content(64) -> Fake ID(64)
        self.generator = nn.Sequential(
            nn.Linear(config.behavior_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.behavior_dim),
            nn.Tanh()  # Tanh 限制在 -1~1 之间，防止生成值过大
        )

    def get_item_vector(self, i, force_cold=False):
        """
        全量排名核心逻辑:
        force_cold=True -> Content + Generated_ID
        """
        con_raw = self.item_con_emb(i)
        feat_con = self.con_proj(con_raw)

        # 生成伪 ID
        fake_id = self.generator(feat_con)

        # ID 处理
        id_raw = self.item_beh_emb(i)

        if force_cold or (self.training and random.random() < 0.5):
            # 冷启动: 使用 Content + Fake ID
            feat_final = feat_con + fake_id
        else:
            # 热门: 使用 Content + True ID (这里简化处理，不做 Distillation Loss)
            # 你也可以做 feat_con + feat_id + fake_id，但为了对比公平，通常是替代关系
            feat_id = self.id_proj(id_raw)
            feat_final = feat_con + feat_id

        return feat_final


# ================= 4. 全量评估函数 (两个模型通用) =================

def precompute_full_pool(model, num_items, batch_size=1024, device='cuda'):
    """
    预计算所有物品向量。
    model.eval() 会自动：
    1. 关闭 Dropout (对于 PAM_Dropout)
    2. Generator 保持确定性输出 (对于 PAM_GAR)
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print(f"⏳ Pre-computing Full Pool ({model.__class__.__name__})...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            # Force Cold -> 强制使用冷启动逻辑 (Mask ID / Generate ID)
            feat_i = model.get_item_vector(i_batch, force_cold=True)
            z_i, _ = model.forward_mlp(feat_i, model.vars, True)
            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_full(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    """
    全量排名评估。
    注意：对于这两个 Baseline，它是'对称'的，即正样本向量 == 库里的向量。
    但我们保留 'Replace' 逻辑以防未来扩展。
    """
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU Full, using CPU for ranking.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    with torch.no_grad():
        for batch, pop, _ in loader:
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            batch_size = u.size(0)

            # 1. User Vector
            e_u = model.user_emb(u)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)

            # 2. Positive Item Vector
            # 重新计算一遍正样本 (确保逻辑一致性)
            feat_pos = model.get_item_vector(i_target, force_cold=True)
            z_i_pos, _ = model.forward_mlp(feat_pos, model.vars, True)

            # 3. Ranking
            if cpu_mode: z_u = z_u.cpu()

            # (A) Background Scores
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # (B) Replace Target Score
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)

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


# ================= 5. 主程序 =================

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
    return {'u': u, 'i': i}, pop, None


def main():
    setup_seed(2025)

    # ================= SETTINGS =================
    # 👉 切换这里运行不同的 Baseline
    MODEL_TYPE = 'PAM_GAR'  # 可选: 'PAM_Dropout', 'PAM_GAR'
    # ============================================

    print(f">>> Running Baseline: {MODEL_TYPE}")

    if not os.path.exists("processed_data/stream_data.pkl"):
        print("❌ Error: processed_data/stream_data.pkl not found.")
        return

    df = pd.read_pickle("processed_data/stream_data.pkl")
    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    content_emb = torch.load("processed_data/content_emb.pt")

    # Data Splitting
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['dt'] = df['timestamp']
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p), batch_size=512, collate_fn=collate_fn, shuffle=True)
               for p in periods if len(p) > 0]

    # Init Model
    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODEL_TYPE == 'PAM_Dropout':
        model = PAM_Dropout(cfg, content_emb).to(device)
    elif MODEL_TYPE == 'PAM_GAR':
        model = PAM_GAR(cfg, content_emb).to(device)
    else:
        raise ValueError("Unknown Model Type")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    target_k = [5, 10, 20]
    metric_names = [f'{m}@{k}' for k in target_k for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metric_names}
    global_count = 0

    WARMUP = 2

    for t, loader in enumerate(loaders):
        print(f"\n--- Period {t} (Samples: {len(loader.dataset)}) ---")

        # === Eval ===
        if t >= WARMUP:
            all_z = precompute_full_pool(model, cfg.num_items, device=device)
            met, n = evaluate_full(model, loader, all_z, device, k_list=target_k)

            if met:
                res_str = " | ".join([f"{k}={met[k]:.4f}" for k in metric_names])
                print(f"📊 Eval: {res_str}")
                for k in metric_names:
                    global_accum[k] += met[k] * n
                global_count += n

        # === Train ===
        model.train()
        total_loss = 0
        for batch_idx, (batch, pop, _) in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch, pop.to(device), None)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"   Avg Loss: {total_loss / len(loader):.4f}")

    print("\n" + "=" * 60)
    print(f"🏆 FINAL RESULT ({MODEL_TYPE})")
    print("-" * 60)
    if global_count > 0:
        for k in metric_names:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()