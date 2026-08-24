import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os, random

# ==============================
# 2. 鍩虹璁剧疆
# ==============================
from torch.utils.data import Dataset, DataLoader

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
    print(f"Random Seed Set: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = 64
        self.content_dim = content_dim
        self.hidden_dim = 128
        self.cold_threshold = 5
        self.lr = 1e-3
        # DropoutNet 鐗规湁鍙傛暟
        self.dropout_prob = 0.5  # 璁粌鏃?Drop ID 鐨勬鐜?


class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


# 杈呭姪 Dataset: 鐢ㄤ簬閬嶅巻鍏ㄥ簱鐗╁搧
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
# 3. DropoutNet 妯″瀷瀹氫箟 (Concat 鐗?
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
        # [鍏抽敭淇敼] 杈撳叆缁村害鍙樻垚 emb_dim * 2锛屽洜涓烘垜浠鎷兼帴 ID 鍜?Content
        self.fusion_layer = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Tanh()
        )

        self.temperature = 0.1

    def get_item_vector(self, i_idx, force_dropout=False):
        """
        鐢熸垚鐗╁搧鍚戦噺
        """
        # A. ID Embedding
        id_e = self.item_id_emb(i_idx)

        # B. Dropout 閫昏緫
        if force_dropout:
            # 鍏ㄩ噺鎺掑悕璇勪及鏃讹紝寮哄埗 Mask 涓?0
            id_e = torch.zeros_like(id_e)
        elif self.training:
            # 璁粌鏃堕殢鏈?Dropout
            mask = torch.rand(id_e.size(0), 1, device=self.device) > self.cfg.dropout_prob
            id_e = id_e * mask.float()

        # C. Content Embedding
        content = self.content_features[i_idx]
        content_e = self.content_mlp(content)

        # D. [鍏抽敭淇敼] 浣跨敤 Concat 鑰屼笉鏄?Add
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

        # 璁粌鏃?force_dropout=False
        z_i = self.get_item_vector(i_idx, force_dropout=False)
        z_i = F.normalize(z_i, dim=1)

        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        loss = F.cross_entropy(logits, labels)

        return loss


# ==============================
# 4. 鍏ㄩ噺鎺掑悕鐩稿叧鍑芥暟
# ==============================

def precompute_full_pool(model, num_items, batch_size=2048, device='cuda'):
    """
    棰勮绠楀叏閲忕墿鍝佹睜
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("Pre-computing Full Item Pool (Concat Mode / Force Dropout)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            # 寮哄埗 Mask ID
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
        print("鈿狅笍 GPU Full, ranking on CPU.")
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

            # 3. 鍏ㄩ噺鍒嗘暟
            if cpu_mode: z_u = z_u.cpu()

            # [B, N_items]
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # 4. 鏇挎崲
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

def evaluate_dual_dropoutnet(model, loader, all_item_z, device, k_list, user_seen_items=None):
    """Compute cold/hot full-ranking metrics with optional seen-item masking."""
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
            z_i_pos = F.normalize(model.get_item_vector(i_tgt, force_dropout=True), dim=1)

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
                scores[rows, t_cols] = pos_scores

            max_k = max(k_list)
            _, topk = torch.topk(scores, k=max_k, dim=1)
            t_cols = t_cols.view(-1, 1)

            is_c = pop_mask.cpu() if cpu_m else pop_mask
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

    c_res = {k: v / c_total for k, v in c_sum.items()} if c_total > 0 else None
    h_res = {k: v / h_total for k, v in h_sum.items()} if h_total > 0 else None
    return c_res, c_total, h_res, h_total
    
def evaluate_sampled_dropoutnet(model, loader, all_item_z, device, k_list, n_neg=999, user_seen_items=None):
    """Compute cold/hot sampled metrics with seen-filtered negatives."""
    model.eval()
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total, h_total = 0, 0
    sampled_neg_total = 0
    sampled_user_total = 0
    seen_tensor_cache = {}

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
            z_i_pos = F.normalize(model.get_item_vector(i_tgt, force_dropout=True), dim=1)

            if cpu_m:
                scores_full = torch.matmul(z_u.cpu(), all_emb.t())
                pos_scores = (z_u.cpu() * z_i_pos.cpu()).sum(dim=1)
                t_cols_full = i_tgt.cpu()
            else:
                scores_full = torch.matmul(z_u, all_emb.t())
                pos_scores = (z_u * z_i_pos).sum(dim=1)
                t_cols_full = i_tgt

            rows = torch.arange(batch_size, device=scores_full.device)
            scores_full[rows, t_cols_full] = pos_scores

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

            if user_seen_items:
                for row, user_id in enumerate(u_cpu.tolist()):
                    uid = int(user_id)
                    if uid not in seen_tensor_cache:
                        seen_items = user_seen_items.get(uid)
                        if seen_items:
                            seen_list = [it for it in seen_items if 0 <= it < n_items]
                            seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=scores_full.device) if seen_list else None
                        else:
                            seen_tensor_cache[uid] = None
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        continue
                    scores_full[row, seen_idx] = -1e9
                scores_full[rows, t_cols_full] = pos_scores

            min_pool_size = min(pool.size for pool in pools)
            batch_neg_eff = min(n_neg_eff, min_pool_size)
            if batch_neg_eff <= 0:
                continue

            neg_np = np.empty((batch_size, batch_neg_eff), dtype=np.int64)
            for row in range(batch_size):
                neg_np[row] = np.random.choice(pools[row], size=batch_neg_eff, replace=False)

            sampled_neg_total += batch_size * batch_neg_eff
            sampled_user_total += batch_size

            score_device = scores_full.device
            t_cols_sample = t_cols_full.to(score_device)
            neg_items = torch.from_numpy(neg_np).to(score_device)
            cand_idx = torch.cat([t_cols_sample.unsqueeze(1), neg_items], dim=1)

            # Avoid fixed target-column bias in tie-heavy settings.
            perm = torch.argsort(torch.rand(batch_size, cand_idx.size(1), device=score_device), dim=1)
            cand_idx = cand_idx.gather(1, perm)
            target_cols = (cand_idx == t_cols_sample.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1, 1)
            scores = scores_full.gather(1, cand_idx)

            max_k = min(max(k_list), scores.size(1))
            _, topk = torch.topk(scores, k=max_k, dim=1)

            is_c = pop_mask.cpu() if cpu_m else pop_mask
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

    c_res = {k: v / c_total for k, v in c_sum.items()} if c_total > 0 else None
    h_res = {k: v / h_total for k, v in h_sum.items()} if h_total > 0 else None
    if sampled_user_total > 0:
        avg_neg = sampled_neg_total / sampled_user_total
        print(f"[Sample Eval] avg effective negatives per user: {avg_neg:.1f}")
    return c_res, c_total, h_res, h_total




# ==============================
# 5. 涓昏缁冨惊鐜?
# ==============================

def main():
    setup_seed(2025)
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for DropoutNet (Concat) from {data_dir}...")

    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print(f"Error: {data_dir}/stream_data.pkl not found")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    periods = split_dataframe_by_periods(df, period_type='M')

    # 璋冨皬 batch_size
    dataloaders = [DataLoader(StreamDataset(p), batch_size=512, shuffle=True, collate_fn=collate_fn) for p in periods]

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    cfg.dropout_prob = 0.5
    print(f">> Model: DropoutNet (Concat) | Full Ranking")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DropoutNet(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for m in ['R', 'N'] for k in k_list]
    
    # 鍖哄垎閲囨牱璇勪及 (samp) 涓?鍏ㄥ簱璇勪及 (full)
    accum_s = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_s = {'cold': 0, 'hot': 0}
    
    accum_f = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_f = {'cold': 0, 'hot': 0}

    WARMUP = 3
    user_seen_items = {}

    for t, loader in enumerate(dataloaders):
        
        # --- Eval ---
        if t >= WARMUP:
            all_z = precompute_full_pool(model, cfg.n_items, device=device)
            # 鍏ㄥ簱鎵撳垎
            c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_dropoutnet(
                model, loader, all_z, device, k_list, user_seen_items=user_seen_items
            )
            # 閲囨牱鎵撳垎
            c_m_s, n_c_s, h_m_s, n_h_s = evaluate_sampled_dropoutnet(
                model, loader, all_z, device, k_list, user_seen_items=user_seen_items
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

        # --- Train ---
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

        # Update seen interactions for the next period's evaluation.
        cur_u = loader.dataset.u.tolist()
        cur_i = loader.dataset.i.tolist()
        for u_idx, i_idx in zip(cur_u, cur_i):
            uid = int(u_idx)
            if uid not in user_seen_items:
                user_seen_items[uid] = set()
            user_seen_items[uid].add(int(i_idx))

    print("\n" + "=" * 90)
    print("         FINAL RESULT: Sampled (1+999) vs Full Ranking (DropoutNet HIN)")
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

    out = {"model": "DropoutNet", "protocol": "stream"}
    for k in metrics_keys:
        out[f"samp_cold_{k}"] = accum_s['cold'][k]/counts_s['cold'] if counts_s['cold'] > 0 else 0
        out[f"samp_hot_{k}"] = accum_s['hot'][k]/counts_s['hot'] if counts_s['hot'] > 0 else 0
        out[f"full_cold_{k}"] = accum_f['cold'][k]/counts_f['cold'] if counts_f['cold'] > 0 else 0
        out[f"full_hot_{k}"] = accum_f['hot'][k]/counts_f['hot'] if counts_f['hot'] > 0 else 0
    pd.DataFrame([out]).to_json("drop_full_result.json", orient="records", force_ascii=False)
    print("Saved: drop_full_result.json")


if __name__ == "__main__":
    main()

