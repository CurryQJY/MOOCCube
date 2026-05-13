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

def build_sparse_adj_and_svd(u_idx_list, i_idx_list, n_users, n_items, q=5, device='cuda'):
    edges = set(zip(u_idx_list, i_idx_list))
    if not edges: # handle empty graph
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float)
    else:
        u_idx = [e[0] for e in edges]
        i_idx = [e[1] for e in edges]
        indices = torch.tensor([u_idx, i_idx], dtype=torch.long)
        values = torch.ones(len(u_idx), dtype=torch.float)
        
    deg_u = torch.zeros(n_users)
    deg_i = torch.zeros(n_items)
    if indices.shape[1] > 0:
        deg_u.scatter_add_(0, indices[0], values)
        deg_i.scatter_add_(0, indices[1], values)
        
    deg_u = deg_u.clamp(min=1e-8).pow(-0.5)
    deg_i = deg_i.clamp(min=1e-8).pow(-0.5)
    
    val_norm = values * deg_u[indices[0]] * deg_i[indices[1]] if indices.shape[1] > 0 else values
    adj_norm = torch.sparse_coo_tensor(indices, val_norm, size=(n_users, n_items)).to(device)
    
    with torch.no_grad():
        if adj_norm._nnz() > q:
            U, S, V = torch.svd_lowrank(adj_norm, q=q, niter=2)
        else:
            U = torch.zeros(n_users, q, device=device)
            S = torch.zeros(q, device=device)
            V = torch.zeros(n_items, q, device=device)
    
    return adj_norm, U, S, V


class LightGCL(nn.Module):
    def __init__(self, cfg, content_emb):
        super(LightGCL, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        
        self.content_features = content_emb.to(self.device)
        self.content_mlp = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        )
        self.n_layers = 2
        self.temp = 0.2
        self.lambda_cl = 0.2

    def forward(self, adj_norm, svd_u, svd_s, svd_v, u_idx=None, i_idx=None):
        e_u = self.user_emb.weight
        e_i = self.item_emb.weight + self.content_mlp(self.content_features)
        
        u_embeds, i_embeds = [e_u], [e_i]
        g_u_embeds, g_i_embeds = [e_u], [e_i]
        
        for layer in range(self.n_layers):
            e_u_next = torch.sparse.mm(adj_norm, e_i)
            e_i_next = torch.sparse.mm(adj_norm.t(), e_u)
            e_u = e_u_next
            e_i = e_i_next
            u_embeds.append(e_u)
            i_embeds.append(e_i)
            
            vt_ei = torch.matmul(svd_v.t(), e_i)
            s_vt_ei = svd_s.unsqueeze(1) * vt_ei
            g_u_next = torch.matmul(svd_u, s_vt_ei)
            
            ut_eu = torch.matmul(svd_u.t(), e_u)
            s_ut_eu = svd_s.unsqueeze(1) * ut_eu
            g_i_next = torch.matmul(svd_v, s_ut_eu)
            
            g_u_embeds.append(g_u_next)
            g_i_embeds.append(g_i_next)
            
        final_e_u = torch.stack(u_embeds, dim=1).mean(dim=1)
        final_e_i = torch.stack(i_embeds, dim=1).mean(dim=1)
        final_g_u = torch.stack(g_u_embeds, dim=1).mean(dim=1)
        final_g_i = torch.stack(g_i_embeds, dim=1).mean(dim=1)
        
        if u_idx is None:
            return final_e_u, final_e_i
            
        return final_e_u[u_idx], final_e_i[i_idx], final_g_u[u_idx], final_g_i[i_idx]


# ==============================
# 4. 全量排名相关函数
# ==============================

def evaluate_dual_lightgcl(loader, all_z_u, all_z_i, device, k_list, cold_threshold, user_seen_items=None):
    """计算 LightGCL 的 Cold 和 Hot 全库指标"""
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total = 0
    h_total = 0
    seen_tensor_cache = {}
    
    try:
        all_emb = all_z_i.to(device)
        all_u_emb = all_z_u.to(device)
        cpu_m = False
    except:
        all_emb = all_z_i
        all_u_emb = all_z_u.cpu()
        cpu_m = True

    with torch.no_grad():
        for batch, pop in loader:
            pop_mask = pop < cold_threshold
            u = batch['u']
            i_tgt = batch['i']
            
            z_u = all_u_emb[u].to(device)
            z_i_pos = all_emb[i_tgt].to(device) if not cpu_m else all_emb[i_tgt]
            
            if cpu_m:
                scores = torch.matmul(z_u.cpu(), all_emb.t())
                pos_scores = (z_u.cpu() * z_i_pos).sum(dim=1)
                t_cols = i_tgt.cpu()
            else:
                scores = torch.matmul(z_u, all_emb.t())
                pos_scores = (z_u * z_i_pos).sum(dim=1)
                t_cols = i_tgt.to(device)
                
            rows = torch.arange(u.size(0), device=scores.device)
            scores[rows, t_cols] = pos_scores

            if user_seen_items:
                user_ids = u.detach().cpu().tolist()
                for row, user_id in enumerate(user_ids):
                    uid = int(user_id)
                    if uid not in seen_tensor_cache:
                        seen_items = user_seen_items.get(uid)
                        if seen_items:
                            seen_list = [it for it in seen_items if 0 <= it < all_emb.size(0)]
                            seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=scores.device) if seen_list else None
                        else:
                            seen_tensor_cache[uid] = None
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        continue
                    scores[row, seen_idx] = -1e9
                # Keep target score valid if masked.
                scores[rows, t_cols] = pos_scores
            
            max_k = max(k_list)
            _, topk = torch.topk(scores, k=max_k, dim=1)
            t_cols = t_cols.view(-1, 1)
            
            is_c = pop_mask.cpu() if cpu_m else pop_mask.to(scores.device)
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == t_cols).any(dim=1).float()
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

def evaluate_sampled_lightgcl(loader, all_z_u, all_z_i, device, k_list, cold_threshold, n_items, n_neg=999, user_seen_items=None):
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total, h_total = 0, 0
    sampled_neg_total = 0
    sampled_user_total = 0
    n_neg_eff = min(n_neg, max(1, n_items - 1))
    all_items_np = np.arange(n_items, dtype=np.int64)
    
    try:
        all_emb = all_z_i.to(device)
        all_u_emb = all_z_u.to(device)
        cpu_m = False
    except:
        all_emb = all_z_i
        all_u_emb = all_z_u.cpu()
        cpu_m = True

    with torch.no_grad():
        for batch, pop in loader:
            pop_mask = pop < cold_threshold
            u = batch['u']
            i_tgt = batch['i']
            batch_size = u.size(0)
            
            z_u = all_u_emb[u].to(device)

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
            cand_idx = torch.cat([i_tgt.unsqueeze(1).to(device), neg_items], dim=1)

            # Randomize candidate order to avoid positional tie-bias.
            perm = torch.argsort(torch.rand(batch_size, cand_idx.size(1), device=cand_idx.device), dim=1)
            cand_idx = cand_idx.gather(1, perm)
            target_cols = (cand_idx == i_tgt.unsqueeze(1).to(device)).nonzero(as_tuple=True)[1].view(-1, 1)
            
            if cpu_m:
                cand_idx_cpu = cand_idx.cpu()
                cand_vecs = all_emb[cand_idx_cpu]
                scores = torch.bmm(cand_vecs, z_u.cpu().unsqueeze(2)).squeeze(2)
                target_cols = target_cols.cpu()
            else:
                cand_vecs = all_emb[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)

            max_k = min(max(k_list), scores.size(1))
            _, topk = torch.topk(scores, k=max_k, dim=1)
            
            is_c = pop_mask.cpu() if cpu_m else pop_mask.to(scores.device)
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
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
    if sampled_user_total > 0:
        avg_neg = sampled_neg_total / sampled_user_total
        print(f"[Sample Eval] avg effective negatives per user: {avg_neg:.1f}")
    return c_res, c_total, h_res, h_total



# ==============================
# 5. 主训练循环 (对抗训练核心)
# ==============================

def main():
    setup_seed(2025)
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for LightGCL (SVD Contrastive) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print(f"错误: {data_dir}/stream_data.pkl 未找到")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    # [NEW] Static random split 8:1:1
    df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*0.8)]
    val_df = df.iloc[int(n*0.8):int(n*0.9)]
    test_df = df.iloc[int(n*0.9):]

    train_loader = DataLoader(StreamDataset(train_df), batch_size=2048, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(StreamDataset(val_df), batch_size=2048, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(StreamDataset(test_df), batch_size=2048, shuffle=False, collate_fn=collate_fn)

    # Seen caches for fair ranking:
    # val masks train seen; test masks train+val seen.
    train_seen_items = {}
    for u_idx, i_idx in zip(train_df['u_idx'].values, train_df['i_idx'].values):
        uid = int(u_idx)
        if uid not in train_seen_items:
            train_seen_items[uid] = set()
        train_seen_items[uid].add(int(i_idx))
    test_seen_items = {uid: set(items) for uid, items in train_seen_items.items()}
    for u_idx, i_idx in zip(val_df['u_idx'].values, val_df['i_idx'].values):
        uid = int(u_idx)
        if uid not in test_seen_items:
            test_seen_items[uid] = set()
        test_seen_items[uid].add(int(i_idx))

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: LightGCL (ICLR '23) | BatchSize: 2048 (InfoNCE)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightGCL(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for m in ['R', 'N'] for k in k_list]
    
    epochs = int(os.environ.get("LIGHTGCL_STATIC_EPOCHS", "40"))
    
    # [CRITICAL] Build and perform SVD only on TRAIN set bipartite graph to prevent data leakage!
    global_u_idx = train_df['u_idx'].tolist()
    global_i_idx = train_df['i_idx'].tolist()
    adj_norm, svd_u, svd_s, svd_v = build_sparse_adj_and_svd(
        global_u_idx, global_i_idx, cfg.n_users, cfg.n_items, q=5, device=device
    )

    for epoch in range(1, epochs + 1):
        # --- Phase 1: Train ---
        model.train()
        total_main_loss = 0
        total_cl_loss = 0
        steps = 0

        for batch, pop in train_loader:
            u_idx = batch['u'].to(device)
            i_idx = batch['i'].to(device)

            optimizer.zero_grad()
            batch_e_u, batch_e_i, batch_g_u, batch_g_i = model(
                adj_norm, svd_u, svd_s, svd_v, u_idx=u_idx, i_idx=i_idx
            )
            
            z_u = F.normalize(batch_e_u, dim=1)
            z_i = F.normalize(batch_e_i, dim=1)
            logits = torch.matmul(z_u, z_i.t()) / model.temp
            labels = torch.arange(logits.size(0)).to(device)
            loss_rec = F.cross_entropy(logits, labels)

            z_g_u = F.normalize(batch_g_u, dim=1)
            z_g_i = F.normalize(batch_g_i, dim=1)
            
            cl_u_logits = torch.matmul(z_u, z_g_u.t()) / model.temp
            cl_i_logits = torch.matmul(z_i, z_g_i.t()) / model.temp
            loss_cl_u = F.cross_entropy(cl_u_logits, labels)
            loss_cl_i = F.cross_entropy(cl_i_logits, labels)

            loss = loss_rec + model.lambda_cl * (loss_cl_u + loss_cl_i)

            loss.backward()
            optimizer.step()

            total_main_loss += loss_rec.item()
            total_cl_loss += (loss_cl_u + loss_cl_i).item()
            steps += 1

        print(f"Epoch [{epoch}/{epochs}] Train InfoNCE: {total_main_loss / steps:.4f} | SVD CL: {total_cl_loss / steps:.4f}")

        # --- Phase 2: Eval on Valid ---
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                all_z_u, all_z_i = model(adj_norm, svd_u, svd_s, svd_v, u_idx=None)
                all_z_u = F.normalize(all_z_u, dim=1)
                all_z_i = F.normalize(all_z_i, dim=1)
                
            # Quick valid ranking
            c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_lightgcl(
                val_loader, all_z_u.cpu(), all_z_i.cpu(), device, k_list, cfg.cold_threshold, user_seen_items=train_seen_items
            )
            h_f_str = " | ".join([f"{k}={h_m_f[k]:.4f}" for k in metrics_keys[:3]]) if h_m_f else "N/A"
            print(f"  --> Valid Hot Full: {h_f_str}")

    print("\n" + "=" * 90)
    print("         FINAL TEST REPORT: Sampled (1+999) vs Full Ranking (STATIC 8:1:1)")
    print("=" * 90)
    
    # --- Phase 3: Final Test ---
    model.eval()
    with torch.no_grad():
        all_z_u, all_z_i = model(adj_norm, svd_u, svd_s, svd_v, u_idx=None)
        all_z_u = F.normalize(all_z_u, dim=1)
        all_z_i = F.normalize(all_z_i, dim=1)
        
    c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_lightgcl(
        test_loader, all_z_u.cpu(), all_z_i.cpu(), device, k_list, cfg.cold_threshold, user_seen_items=test_seen_items
    )
    c_m_s, n_c_s, h_m_s, n_h_s = evaluate_sampled_lightgcl(
        test_loader, all_z_u.cpu(), all_z_i.cpu(), device, k_list, cfg.cold_threshold, cfg.n_items, user_seen_items=test_seen_items
    )

    print(f"{'Metric':<10} | {'Samp Cold':<12} | {'Samp Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for k in metrics_keys:
        v_s_c = c_m_s.get(k, 0.0) if c_m_s else 0.0
        v_s_h = h_m_s.get(k, 0.0) if h_m_s else 0.0
        v_f_c = c_m_f.get(k, 0.0) if c_m_f else 0.0
        v_f_h = h_m_f.get(k, 0.0) if h_m_f else 0.0
        print(f"{k:<10} | {v_s_c:<12.4f} | {v_s_h:<12.4f} | {v_f_c:<12.4f} | {v_f_h:<12.4f}")
    print("=" * 90)

    out = {"model": "LightGCL", "protocol": "static"}
    for k in metrics_keys:
        out[f"samp_cold_{k}"] = c_m_s.get(k, 0.0) if c_m_s else 0.0
        out[f"samp_hot_{k}"] = h_m_s.get(k, 0.0) if h_m_s else 0.0
        out[f"full_cold_{k}"] = c_m_f.get(k, 0.0) if c_m_f else 0.0
        out[f"full_hot_{k}"] = h_m_f.get(k, 0.0) if h_m_f else 0.0
    pd.DataFrame([out]).to_json("lightgcl_static_result.json", orient="records", force_ascii=False)
    print("Saved: lightgcl_static_result.json")


if __name__ == "__main__":
    main()
