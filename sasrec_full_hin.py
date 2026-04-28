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
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
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

        # --- SASRec 特有参数 ---
        self.max_seq_len = 50
        self.n_heads = 2
        self.n_blocks = 2
        self.dropout_rate = 0.2


# ==============================
# 3. 数据集与流式切分 (Sequence Build)
# ==============================
# SASRec 必须在流式数据中维护每个用户的时序交互上下文

def split_dataframe_by_periods(df, period_type='M'):
    """按时间划分 Period"""
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


class SASRecDataset(Dataset):
    def __init__(self, df, user_seq_dict, max_seq_len):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        
        self.max_seq_len = max_seq_len
        self.seqs = []
        
        # Build strict causal sequences (only events BEFORE current)
        for idx, row in df.iterrows():
            uid = row['u_idx']
            seq = user_seq_dict[uid].copy() # Get history BEFORE this point
            
            # Pad or truncate
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            else:
                seq = [0] * (max_seq_len - len(seq)) + seq # 0 is padding idx
                
            self.seqs.append(seq)
            
            # UPDATE global dict AFTER generating sequence to prevent causal cheating
            user_seq_dict[uid].append(row['i_idx'])

        self.seq_tensors = torch.tensor(self.seqs, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'seq': self.seq_tensors[idx], 'pop': self.pop[idx]}

def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    seq = torch.stack([item['seq'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    return {'u': u, 'i': i, 'seq': seq}, pop


# ==============================
# 4. SASRec 模型定义 (ICDM '18) 适配 Content + Graph
# ==============================

class SASRec(nn.Module):
    def __init__(self, cfg, content_emb):
        super(SASRec, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding: 0 is padding idx, so n_items+1
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.emb_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout_rate)

        # Content Incorporation (Optional for pure Cold but keeps parity)
        self.content_features = content_emb.to(self.device)
        self.content_mlp = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        )
        # Content padding
        self.content_pad = nn.Parameter(torch.zeros(1, cfg.emb_dim))

        self.attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.layer_norms_1 = nn.ModuleList()
        self.layer_norms_2 = nn.ModuleList()
        self.layer_norms_seq = nn.LayerNorm(cfg.emb_dim) # final norm

        for _ in range(cfg.n_blocks):
            new_attn_layer = nn.MultiheadAttention(cfg.emb_dim, cfg.n_heads, cfg.dropout_rate, batch_first=True)
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layer = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(cfg.hidden_dim, cfg.emb_dim),
                nn.Dropout(cfg.dropout_rate)
            )
            self.forward_layers.append(new_fwd_layer)

            self.layer_norms_1.append(nn.LayerNorm(cfg.emb_dim))
            self.layer_norms_2.append(nn.LayerNorm(cfg.emb_dim))

    def get_full_item_emb(self):
        """ Combines ID and Content for Items [1...N] """
        cnt = self.content_mlp(self.content_features)
        full_cnt = torch.cat([self.content_pad, cnt], dim=0)
        return self.item_emb.weight + full_cnt

    def forward(self, log_seqs, target_i=None):
        full_i_e = self.get_full_item_emb()
        
        seqs = full_i_e[log_seqs]
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.tensor(positions, device=self.device))
        seqs = self.emb_dropout(seqs)

        # Pytorch MHA mask needs to be square [T, T] causal mask
        seq_len = seqs.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        key_padding_mask = (log_seqs == 0)
        
        for i in range(self.cfg.n_blocks):
            Q = self.layer_norms_1[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=causal_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=False)
            seqs = Q + mha_outputs
            seqs = seqs + self.forward_layers[i](self.layer_norms_2[i](seqs))
            
        final_seqs = self.layer_norms_seq(seqs) # [B, T, D]
        # We only care about the last hidden state in the sequence to predict the next item
        user_emb = final_seqs[:, -1, :] # [B, D]

        if target_i is not None:
            item_emb = full_i_e[target_i] # [B, D]
            return user_emb, item_emb
        
        return user_emb # return state for eval


# ==============================
# 5. 评估工具
# ==============================

def evaluate_dual_sasrec(model, loader, device, k_list, cold_threshold, user_seen_items=None):
    """SASRec Dual Full-Ranking Metrics"""
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total = 0
    h_total = 0
    seen_tensor_cache = {}

    model.eval()
    with torch.no_grad():
        all_item_emb = model.get_full_item_emb()[1:] # Exclude padding 0
        
        for batch, pop in loader:
            pop_mask = pop < cold_threshold
            # Since dataset outputs shifted index explicitly:
            i_tgt = batch['i'] # Already 1-indexed
            seq = batch['seq'].to(device)
            u = batch['u']
            
            z_u = model(seq) # [B, D]
            
            # Full rank across 1 N (ignoring 0)
            scores = torch.matmul(z_u, all_item_emb.t())
            
            # target should match the position in all_item_emb (which is size N, containing idx 1..N)
            # So position 0 in all_item_emb corresponds to item 1.
            t_cols = (i_tgt.to(device) - 1)
            rows = torch.arange(scores.size(0), device=scores.device)
            pos_scores = scores[rows, t_cols]

            if user_seen_items:
                user_ids = u.detach().cpu().tolist()
                for row, user_id in enumerate(user_ids):
                    uid = int(user_id)
                    if uid not in seen_tensor_cache:
                        seen_items = user_seen_items.get(uid)
                        if seen_items:
                            seen_list = [it - 1 for it in seen_items if 1 <= it <= model.cfg.n_items]
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
            
            is_c = pop_mask.to(scores.device)
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == t_cols).any(dim=1).float()
                rks = (preds == t_cols).nonzero(as_tuple=True)
                dcgs = torch.zeros(z_u.size(0), device=scores.device)
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

def evaluate_sampled_sasrec(model, loader, device, k_list, cold_threshold, n_items, n_neg=999, user_seen_items=None):
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total, h_total = 0, 0
    sampled_neg_total = 0
    sampled_user_total = 0

    model.eval()
    with torch.no_grad():
        all_item_emb = model.get_full_item_emb() # Includes 0 padding
        # all_item_emb[1] -> original item 0
        all_items_np = np.arange(1, n_items + 1, dtype=np.int64)
        n_neg_eff = min(n_neg, max(1, n_items - 1))
        
        for batch, pop in loader:
            pop_mask = pop < cold_threshold
            i_tgt = batch['i'].to(device) # Already 1-indexed
            u = batch['u'].to(device)
            seq = batch['seq'].to(device)
            batch_size = seq.size(0)
            
            z_u = model(seq) # [B, D]

            # Sample negatives while excluding target and seen history.
            i_cpu = i_tgt.detach().cpu().numpy()
            u_cpu = u.detach().cpu().numpy()
            pools = []
            for row in range(batch_size):
                tgt = int(i_cpu[row])
                forbidden = {tgt}
                if user_seen_items:
                    forbidden.update(user_seen_items.get(int(u_cpu[row]), set()))
                forbidden = [x for x in forbidden if 1 <= x <= n_items]
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
            cand_idx = torch.cat([i_tgt.unsqueeze(1), neg_items], dim=1)

            # Randomize candidate order to avoid positional tie-bias.
            perm = torch.argsort(torch.rand(batch_size, cand_idx.size(1), device=cand_idx.device), dim=1)
            cand_idx = cand_idx.gather(1, perm)
            target_cols = (cand_idx == i_tgt.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1, 1)
            
            cand_vecs = all_item_emb[cand_idx]
            scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)

            max_k = min(max(k_list), scores.size(1))
            _, topk = torch.topk(scores, k=max_k, dim=1)
            
            is_c = pop_mask.to(scores.device)
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
# 6. 主训练循环
# ==============================

def main():
    setup_seed(2025)
    print("Loading Data for SASRec (Self-Attention Sequence) from processed_data_hin...")

    with open("processed_data_hin/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data_hin/stream_data.pkl")
    content_emb = torch.load("processed_data_hin/content_emb.pt")

    # Shift df items by 1 to make room for padding 0 natively in the seq logic? 
    # Actually, the dataset loader logic simply uses +1 when looking up targets in evaluation.
    # But for SEQ generation, we MUST ensure items are +1 so 0 is reserved for padding.
    df['i_idx_shifted'] = df['i_idx'] + 1
    # df we use for target labels 'i' will still contain original '0-N', 
    # the Dataset collator handles 'seq' which relies on 'i_idx_shifted'.
    df['i_idx'] = df['i_idx_shifted']

    periods = split_dataframe_by_periods(df, period_type='M')
    
    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: SASRec (ICDM '18) | BatchSize: 2048 | MaxSeq: 50")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for m in ['R', 'N'] for k in k_list]
    
    accum_s = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_s = {'cold': 0, 'hot': 0}
    accum_f = {'cold': {k: 0.0 for k in metrics_keys}, 'hot': {k: 0.0 for k in metrics_keys}}
    counts_f = {'cold': 0, 'hot': 0}
    WARMUP_PERIODS = 2

    # Sequence Dictionary mapping UID -> List of interactive items (chronological)
    from collections import defaultdict
    user_seq_dict_global = defaultdict(list)
    user_seen_items = {}

    for t, p_df in enumerate(periods):
        
        # NOTE: SASRecDataset updates `user_seq_dict_global` in-place,
        # so historical sequences build up period-by-period without leaking the future.
        dataset = SASRecDataset(p_df, user_seq_dict_global, cfg.max_seq_len)
        loader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn)
        
        eval_ds = SASRecDataset(p_df, defaultdict(list), cfg.max_seq_len) # Empty seq generation
        # Override eval seqs with the real global sequences valid UP TO exactly this event
        eval_ds.seq_tensors = dataset.seq_tensors 
        
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        # --- Phase 1: Eval ---
        if t >= WARMUP_PERIODS:
            c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_sasrec(
                model, eval_loader, device, k_list, cfg.cold_threshold, user_seen_items=user_seen_items
            )
            c_m_s, n_c_s, h_m_s, n_h_s = evaluate_sampled_sasrec(
                model, eval_loader, device, k_list, cfg.cold_threshold, cfg.n_items, user_seen_items=user_seen_items
            )

            c_f_str = " | ".join([f"{k}={c_m_f[k]:.4f}" for k in metrics_keys[:3]]) if c_m_f else "N/A"
            h_f_str = " | ".join([f"{k}={h_m_f[k]:.4f}" for k in metrics_keys[:3]]) if h_m_f else "N/A"
            print(f"[{t}] Full Cold: {n_c_f} | " + c_f_str[:50] + "...")
            print(f"[{t}] Full Hot : {n_h_f} | " + h_f_str[:50] + "...")
            
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

        # --- Phase 2: Train ---
        model.train()
        total_main_loss = 0
        steps = 0

        # We must align training to batch loop, using InfoNCE against all items in batch (In-Batch Negatives)
        for batch, pop in loader:
            seq = batch['seq'].to(device)
            i_idx = batch['i'].to(device)

            optimizer.zero_grad()
            z_u, z_i = model(seq, i_idx)
            
            z_u = F.normalize(z_u, dim=1)
            z_i = F.normalize(z_i, dim=1)
            
            # InfoNCE (Temperature scaled)
            logits = torch.matmul(z_u, z_i.t()) / 0.1
            labels = torch.arange(logits.size(0)).to(device)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            total_main_loss += loss.item()
            steps += 1

        print(f"  [TRAIN] InfoNCE: {total_main_loss / steps:.4f}")

        # Update seen interactions after finishing the period to avoid look-ahead in eval.
        for u_idx, i_idx in zip(p_df['u_idx'].values, p_df['i_idx'].values):
            uid = int(u_idx)
            if uid not in user_seen_items:
                user_seen_items[uid] = set()
            user_seen_items[uid].add(int(i_idx))

    print("\n" + "=" * 90)
    print("         FINAL RESULT: Sampled (1+999) vs Full Ranking (SASRec Streaming)")
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


if __name__ == "__main__":
    main()
