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
    print(f"Random Seed Fixed: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = 64
        self.content_dim = content_dim
        self.hidden_dim = 128
        self.cold_threshold = 1
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

    print("Pre-computing Full Item Pool (Generator)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            content = model.content_features[i_batch]
            # GAFC 核心: 用 G(content) 替代 ID
            z_i = model.generator(content)
            z_i = F.normalize(z_i, dim=1)
            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_dual_gafc(model, loader, all_item_z, device, k_list, user_seen_items=None):
    """同时计算 Cold 和 Hot 全库指标"""
    model.eval()
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total = 0
    h_total = 0
    seen_tensor_cache = {}
    
    try:
        all_emb = all_item_z.to(device)
        cpu_m = False
    except:
        all_emb = all_item_z
        cpu_m = True

    with torch.no_grad():
        for batch, pop in loader:
            pop_mask = pop < model.cfg.cold_threshold
            
            u = batch['u'].to(device)
            i_tgt = batch['i'].to(device)
            z_u = F.normalize(model.user_emb(u), dim=1)
            
            # Target Item: Generator Output
            content = model.content_features[i_tgt]
            z_i_pos = F.normalize(model.generator(content), dim=1)
            
            if cpu_m:
                scores = torch.matmul(z_u.cpu(), all_emb.t())
                pos_scores = (z_u.cpu() * z_i_pos.cpu()).sum(dim=1)
                t_cols = i_tgt.cpu()
            else:
                scores = torch.matmul(z_u, all_emb.t())
                pos_scores = (z_u * z_i_pos).sum(dim=1)
                t_cols = i_tgt
                
            rows = torch.arange(u.size(0), device=scores.device)
            scores[rows, t_cols] = pos_scores

            if user_seen_items:
                user_ids = u.detach().cpu().tolist()
                for row, user_id in enumerate(user_ids):
                    uid = int(user_id)
                    if uid not in seen_tensor_cache:
                        seen_items = user_seen_items.get(uid)
                        if seen_items:
                            seen_list = [it for it in seen_items if 0 <= it < model.cfg.n_items]
                            seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=scores.device) if seen_list else None
                        else:
                            seen_tensor_cache[uid] = None
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        continue
                    scores[row, seen_idx] = -1e9

                # Keep the target item score valid after masking seen items.
                scores[rows, t_cols] = pos_scores
            
            max_k = max(k_list)
            # Shuffle item columns before top-k to avoid deterministic index tie-bias.
            # This is important when many scores are tied (common under GAN collapse).
            perm = torch.randperm(scores.size(1), device=scores.device)
            shuffled_scores = scores[:, perm]
            _, topk_pos = torch.topk(shuffled_scores, k=max_k, dim=1)
            topk = perm[topk_pos]
            t_cols = t_cols.view(-1, 1)
            
            is_c = pop_mask.cpu() if cpu_m else pop_mask
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == t_cols).any(dim=1).float()
                # 修复后的 NDCG 逻辑
                rks = (preds == t_cols).nonzero(as_tuple=True)
                dcgs = torch.zeros(u.size(0), device=scores.device)
                if rks[0].numel() > 0:
                    dcgs[rks[0]] = 1.0 / torch.log2(rks[1].float() + 2.0)
                
                c_sum[f'R@{k}'] += hits[is_c].sum().item()
                c_sum[f'N@{k}'] += dcgs[is_c].sum().item()
                h_sum[f'R@{k}'] += hits[is_h].sum().item()
                h_sum[f'N@{k}'] += dcgs[is_h].sum().item()
                
            c_total += is_c.sum().item()
            h_total += is_h.sum().item()
            
    c_res = {k: v/c_total for k,v in c_sum.items()} if c_total > 0 else None
    h_res = {k: v/h_total for k,v in h_sum.items()} if h_total > 0 else None
    return c_res, c_total, h_res, h_total

def evaluate_sampled_gafc(model, loader, all_item_z, device, k_list, n_neg=999, user_seen_items=None):
    """计算基于 1正+999负 采样的 Cold 和 Hot 指标"""
    model.eval()
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total, h_total = 0, 0
    sampled_neg_total = 0
    sampled_user_total = 0
    
    n_items = model.cfg.n_items
    if n_items <= 1:
        return None, 0, None, 0
    n_neg_eff = min(n_neg, n_items - 1)
    all_items_np = np.arange(n_items, dtype=np.int64)
    try:
        all_emb = all_item_z.to(device)
        cpu_m = False
    except:
        all_emb = all_item_z
        cpu_m = True

    with torch.no_grad():
        for batch, pop in loader:
            pop_mask = pop < model.cfg.cold_threshold
            
            u = batch['u'].to(device)
            i_tgt = batch['i'].to(device)
            batch_size = u.size(0)
            
            z_u = F.normalize(model.user_emb(u), dim=1)
            
            # Build per-user candidate pools while excluding target and seen history.
            i_cpu = i_tgt.detach().cpu().numpy()
            u_cpu = u.detach().cpu().numpy()
            pools = []

            for row in range(batch_size):
                tgt = int(i_cpu[row])
                forbidden = {tgt}
                if user_seen_items:
                    forbidden.update(user_seen_items.get(int(u_cpu[row]), set()))

                forbidden = [x for x in forbidden if 0 <= x < n_items]
                if len(forbidden) >= n_items:
                    pool = all_items_np[all_items_np != tgt]
                else:
                    pool = np.setdiff1d(all_items_np, np.array(forbidden, dtype=np.int64), assume_unique=False)

                if pool.size == 0:
                    pool = all_items_np[all_items_np != tgt]
                pools.append(pool)

            # No-replacement sampled evaluation to avoid duplicated negatives.
            # Use the minimum pool size in this batch to keep tensor shapes aligned.
            min_pool_size = min(pool.size for pool in pools)
            batch_neg_eff = min(n_neg_eff, min_pool_size)
            if batch_neg_eff <= 0:
                continue

            neg_np = np.empty((batch_size, batch_neg_eff), dtype=np.int64)
            for row in range(batch_size):
                neg_np[row] = np.random.choice(pools[row], size=batch_neg_eff, replace=False)

            sampled_neg_total += batch_size * batch_neg_eff
            sampled_user_total += batch_size

            neg_items = torch.from_numpy(neg_np).to(device)
            cand_idx = torch.cat([i_tgt.unsqueeze(1), neg_items], dim=1)
            perm = torch.argsort(torch.rand(batch_size, cand_idx.size(1), device=cand_idx.device), dim=1)
            cand_idx = cand_idx.gather(1, perm)
            target_cols = (cand_idx == i_tgt.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1, 1)
            if cpu_m:
                target_cols = target_cols.cpu()
            
            if cpu_m:
                cand_idx_cpu = cand_idx.cpu()
                cand_vecs = all_emb[cand_idx_cpu]
                scores = torch.bmm(cand_vecs, z_u.cpu().unsqueeze(2)).squeeze(2)
            else:
                cand_vecs = all_emb[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
            
            # 目标永远是第 0 列 (即 cand_idx 中拼接的 i_tgt)
            
            max_k = min(max(k_list), scores.size(1))
            _, topk = torch.topk(scores, k=max_k, dim=1)
            
            is_c = pop_mask.cpu() if cpu_m else pop_mask
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
                # 修复后的 NDCG 逻辑
                rks = (preds == target_cols).nonzero(as_tuple=True)
                dcgs = torch.zeros(batch_size, device=scores.device)
                if rks[0].numel() > 0:
                    dcgs[rks[0]] = 1.0 / torch.log2(rks[1].float() + 2.0)
                
                c_sum[f'R@{k}'] += hits[is_c].sum().item()
                c_sum[f'N@{k}'] += dcgs[is_c].sum().item()
                h_sum[f'R@{k}'] += hits[is_h].sum().item()
                h_sum[f'N@{k}'] += dcgs[is_h].sum().item()
                
            c_total += is_c.sum().item()
            h_total += is_h.sum().item()
            
    c_res = {k: v/c_total for k,v in c_sum.items()} if c_total > 0 else None
    h_res = {k: v/h_total for k,v in h_sum.items()} if h_total > 0 else None
    return c_res, c_total, h_res, h_total



# ==============================
# 5. 主训练循环 (对抗训练核心)
# ==============================

def main():
    setup_seed(2025)
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for GAFC (SIGIR '22) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print(f"错误: {data_dir}/stream_data.pkl 未找到")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

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
    metrics_keys = [f'{m}@{k}' for m in ['R', 'N'] for k in k_list]
    
    accum_s = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_s = {'cold': 0, 'hot': 0}
    accum_f = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_f = {'cold': 0, 'hot': 0}
    WARMUP_PERIODS = 2
    user_seen_items = {}

    # Binary Cross Entropy Loss for GAN
    criterion_gan = nn.BCELoss()

    for t, loader in enumerate(dataloaders):

        # --- Phase 1: Eval ---
        if t >= WARMUP_PERIODS:
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_gafc(
                model, loader, all_z, device, k_list, user_seen_items=user_seen_items
            )
            c_m_s, n_c_s, h_m_s, n_h_s = evaluate_sampled_gafc(
                model, loader, all_z, device, k_list, n_neg=200, user_seen_items=user_seen_items
            )

            c_f_str = " | ".join([f"{k}={c_m_f[k]:.4f}" for k in metrics_keys]) if c_m_f else "N/A"
            h_f_str = " | ".join([f"{k}={h_m_f[k]:.4f}" for k in metrics_keys]) if h_m_f else "N/A"
            print(f"[{t}] Full Cold: {n_c_f} | " + c_f_str[:50] + "...")
            print(f"[{t}] Full Hot : {n_h_f} | " + h_f_str[:50] + "...")
            
            c_s_str = " | ".join([f"{k}={c_m_s[k]:.4f}" for k in metrics_keys]) if c_m_s else "N/A"
            h_s_str = " | ".join([f"{k}={h_m_s[k]:.4f}" for k in metrics_keys]) if h_m_s else "N/A"
            print(f"[{t}] Samp Cold: {n_c_s} | " + c_s_str[:50] + "...")
            print(f"[{t}] Samp Hot : {n_h_s} | " + h_s_str[:50] + "...")

            if c_m_f:
                counts_f['cold'] += n_c_f
                for k in metrics_keys: accum_f['cold'][k] += c_m_f[k] * n_c_f
            if h_m_f:
                counts_f['hot'] += n_h_f
                for k in metrics_keys: accum_f['hot'][k] += h_m_f[k] * n_h_f
            if c_m_s:
                counts_s['cold'] += n_c_s
                for k in metrics_keys: accum_s['cold'][k] += c_m_s[k] * n_c_s
            if h_m_s:
                counts_s['hot'] += n_h_s
                for k in metrics_keys: accum_s['hot'][k] += h_m_s[k] * n_h_s

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

        # Update seen interactions for next period's evaluation.
        p_df = periods[t]
        for u_idx, i_idx in zip(p_df['u_idx'].values, p_df['i_idx'].values):
            uid = int(u_idx)
            if uid not in user_seen_items:
                user_seen_items[uid] = set()
            user_seen_items[uid].add(int(i_idx))

    print("\n" + "=" * 90)
    print("         FINAL RESULT: Sampled (1+999) vs Full Ranking (GAFC HIN)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Samp Cold':<12} | {'Samp Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for k in metrics_keys:
        v_s_c = accum_s['cold'][k]/counts_s['cold'] if counts_s['cold'] > 0 else 0
        v_s_h = accum_s['hot'][k]/counts_s['hot'] if counts_s['hot'] > 0 else 0
        v_f_c = accum_f['cold'][k]/counts_f['cold'] if counts_f['cold'] > 0 else 0
        v_f_h = accum_f['hot'][k]/counts_f['hot'] if counts_f['hot'] > 0 else 0
        print(f"{k:<10} | {v_s_c:<12.4f} | {v_s_h:<12.4f} | {v_f_c:<12.4f} | {v_f_h:<12.4f}")
    print("=" * 90)

    out = {"model": "GAR", "protocol": "stream"}
    for k in metrics_keys:
        out[f"samp_cold_{k}"] = accum_s['cold'][k]/counts_s['cold'] if counts_s['cold'] > 0 else 0
        out[f"samp_hot_{k}"] = accum_s['hot'][k]/counts_s['hot'] if counts_s['hot'] > 0 else 0
        out[f"full_cold_{k}"] = accum_f['cold'][k]/counts_f['cold'] if counts_f['cold'] > 0 else 0
        out[f"full_hot_{k}"] = accum_f['hot'][k]/counts_f['hot'] if counts_f['hot'] > 0 else 0
    pd.DataFrame([out]).to_json("gar_full_result.json", orient="records", force_ascii=False)
    print("Saved: gar_full_result.json")


if __name__ == "__main__":
    main()
