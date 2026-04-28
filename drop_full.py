import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os, random

# ==============================
# 1. 强制非交互后端
# ==============================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ==============================
# 2. 基础设置
# ==============================

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
        self.lr = 1e-3
        # DropoutNet 特有参数
        self.dropout_prob = 0.5  # 训练时 Drop ID 的概率


class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


# 辅助 Dataset: 用于遍历全库物品
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
    return {'u': u, 'i': i}, pop


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


# ==============================
# 3. DropoutNet 模型定义 (Concat 版)
# ==============================

class DropoutNet(nn.Module):
    def __init__(self, cfg, content_emb):
        super(DropoutNet, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. User Tower
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)

        # 2. Item Preference Tower (ID Embedding)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        # 3. Item Content Tower
        self.content_features = content_emb.to(self.device)
        self.content_mlp = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.Tanh()
        )

        # 4. Fusion Layer (Transform)
        # [关键修改] 输入维度变成 emb_dim * 2，因为我们要拼接 ID 和 Content
        self.fusion_layer = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Tanh()
        )

        self.temperature = 0.1

    def get_item_vector(self, i_idx, force_dropout=False):
        """
        生成物品向量
        """
        # A. ID Embedding
        id_e = self.item_id_emb(i_idx)

        # B. Dropout 逻辑
        if force_dropout:
            # 全量排名评估时，强制 Mask 为 0
            id_e = torch.zeros_like(id_e)
        elif self.training:
            # 训练时随机 Dropout
            mask = torch.rand(id_e.size(0), 1, device=self.device) > self.cfg.dropout_prob
            id_e = id_e * mask.float()

        # C. Content Embedding
        content = self.content_features[i_idx]
        content_e = self.content_mlp(content)

        # D. [关键修改] 使用 Concat 而不是 Add
        # [B, 64] cat [B, 64] -> [B, 128]
        combined = torch.cat([id_e, content_e], dim=1)

        # [B, 128] -> [B, 64]
        fused = self.fusion_layer(combined)

        return fused

    def forward(self, batch):
        u_idx = batch['u']
        i_idx = batch['i']

        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)

        # 训练时 force_dropout=False
        z_i = self.get_item_vector(i_idx, force_dropout=False)
        z_i = F.normalize(z_i, dim=1)

        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        loss = F.cross_entropy(logits, labels)

        return loss


# ==============================
# 4. 全量排名相关函数
# ==============================

def precompute_full_pool(model, num_items, batch_size=2048, device='cuda'):
    """
    预计算全量物品池
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ Pre-computing Full Item Pool (Concat Mode / Force Dropout)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            # 强制 Mask ID
            z_i = model.get_item_vector(i_batch, force_dropout=True)
            z_i = F.normalize(z_i, dim=1)
            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_full_dropoutnet(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU Full, ranking on CPU.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    with torch.no_grad():
        for batch, pop in loader:
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            batch_size = u.size(0)

            # 1. User Vector
            z_u = model.user_emb(u)
            z_u = F.normalize(z_u, dim=1)

            # 2. Positive Item Vector (Target) - Force Cold
            z_i_pos = model.get_item_vector(i_target, force_dropout=True)
            z_i_pos = F.normalize(z_i_pos, dim=1)

            # 3. 全量分数
            if cpu_mode: z_u = z_u.cpu()

            # [B, N_items]
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # 4. 替换
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)
            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

            # 5. Metrics
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


# ==============================
# 5. 主训练循环
# ==============================

def main():
    setup_seed(2025)
    print("Loading Data for DropoutNet (Concat + Full Ranking)...")

    if not os.path.exists("processed_data/stream_data.pkl"):
        print("错误: 找不到数据文件")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    periods = split_dataframe_by_periods(df, period_type='M')

    # 调小 batch_size
    dataloaders = [DataLoader(StreamDataset(p), batch_size=512, shuffle=True, collate_fn=collate_fn) for p in periods]

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    cfg.dropout_prob = 0.5
    print(f">> Model: DropoutNet (Concat) | Full Ranking")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DropoutNet(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for k in k_list for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metrics_keys}
    global_count = 0

    WARMUP_PERIODS = 2

    for t, loader in enumerate(dataloaders):
        print(f"\n--- Period {t} (Total: {len(loader.dataset)}) ---")

        # --- Phase 1: Eval ---
        if t >= WARMUP_PERIODS:
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            metrics, n_cold = evaluate_full_dropoutnet(model, loader, all_z, device, k_list)

            if metrics:
                res_str = " | ".join([f"{k}={metrics[k]:.4f}" for k in metrics_keys])
                print(f"📊 Eval: {res_str}")
                for k in metrics_keys:
                    global_accum[k] += metrics[k] * n_cold
                global_count += n_cold
            else:
                print("  [SKIP] No cold items to test.")

        # --- Phase 2: Train ---
        model.train()
        total_loss = 0
        steps = 0
        for batch, pop in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"  [TRAIN] Avg Loss: {total_loss / max(1, steps):.4f}")

    # ==============================
    # 6. Final Report
    # ==============================
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT (DropoutNet Concat - Full Ranking)")
    print("-" * 60)
    if global_count > 0:
        for k in metrics_keys:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()