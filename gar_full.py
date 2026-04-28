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
from torch.utils.data import Dataset, DataLoader


# ==============================
# 2. 基础配置
# ==============================

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
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = 64
        self.content_dim = content_dim
        self.hidden_dim = 128
        self.cold_threshold = 5
        self.lr = 1e-3

        # --- GAFC (GAN) 特有参数 ---
        self.alpha = 1.0  # Recommender Loss 权重
        self.beta = 0.1  # Reconstruction Loss (MSE) 权重 (论文中通常保留以稳定训练)
        self.gamma = 0.5  # Adversarial Loss (Generator欺骗D) 权重
        self.d_steps = 1  # 判别器训练频次 (每训练1次G，训练几次D)
        self.noise_dim = 16  # (可选) 注入生成器的噪声维度，部分GAN变体使用


class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


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
# 3. GAFC 模型定义 (SIGIR '22)
# ==============================

class Generator(nn.Module):
    """
    生成器 G: Content -> Fake ID Embedding
    """

    def __init__(self, content_dim, hidden_dim, emb_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, emb_dim),
            nn.Tanh()  # 限制输出范围，匹配 ID Embedding 的分布
        )

    def forward(self, content):
        return self.net(content)


class Discriminator(nn.Module):
    """
    [新增] 判别器 D: Embedding -> Probability (Real or Fake)
    """

    def __init__(self, emb_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1] 概率
        )

    def forward(self, emb):
        return self.net(emb)


class GAFC(nn.Module):
    def __init__(self, cfg, content_emb):
        super(GAFC, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. User & Item ID (Real)
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        # 2. Generator & Discriminator
        self.content_features = content_emb.to(self.device)
        self.generator = Generator(cfg.content_dim, cfg.hidden_dim, cfg.emb_dim)
        self.discriminator = Discriminator(cfg.emb_dim, cfg.hidden_dim)

        self.temperature = 0.1

    def get_item_vector(self, i_idx, force_cold=False):
        """
        获取物品向量用于推荐
        force_cold=True (全量评估): 强制使用 Generator 生成的 Fake ID
        """
        # A. Real ID
        real_id = self.item_id_emb(i_idx)

        # B. Fake ID
        content = self.content_features[i_idx]
        fake_id = self.generator(content)

        if force_cold:
            return fake_id, real_id  # 评估时只用生成的

        # 训练时根据策略返回
        # 论文中推荐任务通常同时使用 generated augmentation 或 strict G output
        # 这里为了对抗训练稳定性，通常推荐任务使用 Real ID + Generated ID 的混合
        return fake_id, real_id

        # 注意：GAN 的 forward 逻辑比较复杂，通常拆开写在 train loop 里

    # 这里只保留推荐部分的 forward 计算
    def recommend_score(self, u_idx, i_emb):
        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)
        z_i = F.normalize(i_emb, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        return logits


# ==============================
# 4. 全量排名相关函数
# ==============================

def precompute_full_pool(model, num_items, batch_size=2048, device='cuda'):
    """
    预计算: 强制使用 Generator
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ Pre-computing Full Item Pool (Generator)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            content = model.content_features[i_batch]
            # GAFC 核心: 用 G(content) 替代 ID
            z_i = model.generator(content)
            z_i = F.normalize(z_i, dim=1)
            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_full_gafc(model, loader, all_item_z, device, k_list=[5, 10, 20]):
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

            z_u = model.user_emb(u)
            z_u = F.normalize(z_u, dim=1)

            # Target Item: Generator Output
            content = model.content_features[i_target]
            z_i_pos = model.generator(content)
            z_i_pos = F.normalize(z_i_pos, dim=1)

            if cpu_mode: z_u = z_u.cpu()
            scores = torch.matmul(z_u, all_item_emb_gpu.t())
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)
            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

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
# 5. 主训练循环 (对抗训练核心)
# ==============================

def main():
    setup_seed(2025)
    print("Loading Data for GAFC (SIGIR '22) - Full Ranking...")
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("错误: 请先运行 data_process.py")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    periods = split_dataframe_by_periods(df, period_type='M')
    dataloaders = [DataLoader(StreamDataset(p), batch_size=512, shuffle=True, collate_fn=collate_fn) for p in periods]

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: GAFC (GAN) | Alpha: {cfg.alpha} | Gamma (Adv): {cfg.gamma}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAFC(cfg, content_emb).to(device)

    # 优化器: G 和 D 分开优化
    # G 的优化器包含 Generator, UserEmbedding, ItemIDEmbedding
    opt_g = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.user_emb.parameters()},
        {'params': model.item_id_emb.parameters()}
    ], lr=cfg.lr)

    # D 的优化器只优化 Discriminator
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for k in k_list for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metrics_keys}
    global_count = 0
    WARMUP_PERIODS = 2

    # Binary Cross Entropy Loss for GAN
    criterion_gan = nn.BCELoss()

    for t, loader in enumerate(dataloaders):
        print(f"\n--- Period {t} (Total: {len(loader.dataset)}) ---")

        # --- Phase 1: Eval ---
        if t >= WARMUP_PERIODS:
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            metrics, n_cold = evaluate_full_gafc(model, loader, all_z, device, k_list)

            if metrics:
                res_str = " | ".join([f"{k}={metrics[k]:.4f}" for k in metrics_keys])
                print(f"📊 Eval: {res_str}")
                for k in metrics_keys:
                    global_accum[k] += metrics[k] * n_cold
                global_count += n_cold
            else:
                print("  [SKIP] No cold items to test.")

        # --- Phase 2: Train (Adversarial) ---
        model.train()
        total_g_loss = 0
        total_d_loss = 0
        steps = 0

        for batch, pop in loader:
            u_idx = batch['u'].to(device)
            i_idx = batch['i'].to(device)
            batch_size = u_idx.size(0)

            # Labels for GAN
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)

            # ============================================
            # 步骤 1: 训练 Discriminator (D)
            # ============================================
            opt_d.zero_grad()

            # 1.1 Train with Real
            real_emb = model.item_id_emb(i_idx).detach()  # Detach, 不更新 ID Emb
            prob_real = model.discriminator(real_emb)
            loss_d_real = criterion_gan(prob_real, real_label)

            # 1.2 Train with Fake
            content = model.content_features[i_idx]
            fake_emb = model.generator(content).detach()  # Detach G
            prob_fake = model.discriminator(fake_emb)
            loss_d_fake = criterion_gan(prob_fake, fake_label)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_d.step()

            # ============================================
            # 步骤 2: 训练 Generator (G) & Recommender
            # ============================================
            opt_g.zero_grad()

            # 2.1 Adversarial Loss (骗过 D)
            # 重新生成 fake_emb (带梯度)
            fake_emb_g = model.generator(content)
            prob_fake_g = model.discriminator(fake_emb_g)
            # 目标是让 D 认为是 Real (Label=1)
            loss_g_adv = criterion_gan(prob_fake_g, real_label)

            # 2.2 Recommendation Loss
            # 推荐任务通常使用 Fake ID 进行训练，或者混合
            # 原文策略: 生成的 embedding 应该对推荐有用
            logits = model.recommend_score(u_idx, fake_emb_g)
            rec_labels = torch.arange(batch_size).to(device)
            loss_rec = F.cross_entropy(logits, rec_labels)

            # 2.3 Reconstruction Loss (MSE 辅助)
            # 保持生成的 embedding 和真实的 ID 接近，不仅仅是骗过 D
            real_emb_g = model.item_id_emb(i_idx)
            loss_recon = F.mse_loss(fake_emb_g, real_emb_g)

            # 总 G Loss
            loss_g = cfg.alpha * loss_rec + cfg.gamma * loss_g_adv + cfg.beta * loss_recon

            loss_g.backward()
            opt_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            steps += 1

        print(f"  [TRAIN] G_Loss: {total_g_loss / steps:.4f} | D_Loss: {total_d_loss / steps:.4f}")

    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT (GAFC SIGIR'22)")
    print("-" * 60)
    if global_count > 0:
        for k in metrics_keys:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()