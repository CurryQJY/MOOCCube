import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import random

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

        # --- 关键超参数 ---
        self.lr = 1e-2
        self.dropout_prob = 0.1  # [策略1] 训练时 ID 丢弃率
        self.aux_weight = 0.1  # [策略3] 对比损失权重


class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


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
        period_df = df[df['period_id'] == p_key].reset_index(drop=True)
        periods.append(period_df)
    return periods


# ================= 3. 增强版 PAM 模型核心 =================

class PAM_Enhanced(nn.Module):
    def __init__(self, cfg, content_emb):
        super(PAM_Enhanced, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 基础 Embedding
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        # 2. 内容映射网络
        self.content_features = content_emb.to(self.device)
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim)
        )

        # 3. 自适应门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid()
        )

        self.temperature = 0.1

    def get_item_vector(self, i_idx, force_cold=False):
        """
        生成物品向量
        force_cold=True: 强制 Mask ID (全量排名核心逻辑)
        """
        # A. ID Embedding
        id_e = self.item_id_emb(i_idx)

        # B. Input Perturbation
        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        # C. Content Embedding
        raw_content = self.content_features[i_idx]
        content_e = self.content_proj(raw_content)

        # D. 门控融合
        concat = torch.cat([id_e, content_e], dim=-1)
        alpha = self.gate_net(concat)
        final_item_e = alpha * id_e + (1 - alpha) * content_e

        return final_item_e, id_e, content_e

    def forward(self, batch):
        u_idx = batch['u']
        i_idx = batch['i']

        # 1. Inference
        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)

        z_i, id_e_raw, content_e_processed = self.get_item_vector(i_idx, force_cold=False)
        z_i = F.normalize(z_i, dim=1)

        # 2. Main Loss
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        main_loss = F.cross_entropy(logits, labels)

        # 3. Aux Loss (Contrastive)
        if self.training:
            z_id = F.normalize(id_e_raw, dim=1)
            z_content = F.normalize(content_e_processed, dim=1)
            sim_matrix = torch.matmul(z_id, z_content.t()) / 0.1
            aux_loss_1 = F.cross_entropy(sim_matrix, labels)
            aux_loss_2 = F.cross_entropy(sim_matrix.t(), labels)
            aux_loss = (aux_loss_1 + aux_loss_2) / 2

            total_loss = main_loss + self.cfg.aux_weight * aux_loss
            return total_loss

        return main_loss


# ================= 4. 全量排名评估函数 =================

def precompute_full_pool(model, num_items, batch_size=2048, device='cuda'):
    """
    [新增] 预计算全库物品向量
    使用 force_cold=True，模拟只有 Content 信息的背景库
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ Pre-computing Full Item Pool (Force Cold)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            # 强制冷启动模式
            z_i, _, _ = model.get_item_vector(i_batch, force_cold=True)
            z_i = F.normalize(z_i, dim=1)
            all_z_i.append(z_i.cpu())  # 存入 CPU

    return torch.cat(all_z_i, dim=0)


def evaluate_full_pam(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    """
    [修改] 全量排名评估
    """
    model.eval()
    accum_metrics = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    # 尝试 GPU 加速
    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU OOM, ranking on CPU.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    with torch.no_grad():
        for batch, pop in loader:
            # 1. 筛选冷启动
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            batch_size = u.size(0)

            # 2. User Vector
            z_u = model.user_emb(u)
            z_u = F.normalize(z_u, dim=1)

            # 3. Positive Item Vector (Target)
            # 同样使用 Force Cold，保证与库里的一致性 (Symmetric)
            z_i_pos, _, _ = model.get_item_vector(i_target, force_cold=True)
            z_i_pos = F.normalize(z_i_pos, dim=1)

            # 4. 计算全量分数
            if cpu_mode: z_u = z_u.cpu()

            # [B, N_items]
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # 5. 替换正样本分数
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)

            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

            # 6. 计算 Metrics
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

            total_samples += batch_size
            if cpu_mode: z_u = z_u.to(device)

    if total_samples == 0: return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


# ================= 5. 主流程 =================

def main():
    setup_seed(2025)
    print("Loading Data for PAM Enhanced (Full Ranking)...")
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("错误: 请先运行 data_process.py")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    periods = split_dataframe_by_periods(df, period_type='M')

    # [优化] 调小 batch_size 以适应全量评估
    dataloaders = [DataLoader(StreamDataset(p), batch_size=512, shuffle=True, collate_fn=collate_fn) for p in periods]

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: PAM (Enhanced) Full Ranking | Dropout: {cfg.dropout_prob} | Aux: {cfg.aux_weight}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_Enhanced(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    target_metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}
    global_accum = {k: 0.0 for k in target_metrics}
    global_count = 0

    print(f"\n>>> 开始评估 (共 {len(dataloaders)} 个周期) <<<")
    WARMUP = 3

    for t, loader in enumerate(dataloaders):
        n_total = len(loader.dataset)
        print(f"\n>>> Period {t} (Samples: {n_total}) <<<")

        # --- Phase 1: Full Ranking Eval ---
        current_res = {k: 0.0 for k in target_metrics}

        if t >= WARMUP:
            # 1. 预计算
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            # 2. 评估
            metrics, n_cold = evaluate_full_pam(model, loader, all_z, device)

            if metrics:
                current_res = metrics
                res_str = " | ".join([f"{k}={metrics[k]:.4f}" for k in target_metrics])
                print(f"📊 Eval: {res_str}")

                for k in target_metrics:
                    history[k].append(metrics[k])
                    global_accum[k] += metrics[k] * n_cold
                global_count += n_cold
            else:
                print("  [SKIP] No cold items to test.")
                for k in target_metrics: history[k].append(0)
        else:
            print("  [WARMUP] Training only...")
            for k in target_metrics: history[k].append(0)

        # --- Phase 2: Train ---
        model.train()
        total_loss = 0
        steps = 0
        for batch, pop in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"  [TRAIN] Avg Loss: {total_loss / max(1, steps):.4f}")

    # ================= 结果报告 =================
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (PAM Enhanced - Full Ranking)")
    print("=" * 60)

    if global_count > 0:
        print(f"{'Metric':<10} | {'Micro Avg':<18}")
        print("-" * 60)
        for m in target_metrics:
            micro = global_accum[m] / global_count
            print(f"{m:<10} | {micro:.4f}")

        # 保存结果
        pd.DataFrame(history).to_csv('metrics_pam_enhanced_full.csv', index=False)
        print("\n>> Saved metrics_pam_enhanced_full.csv")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
