import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os, random
import copy

# ==============================
# 1. 寮哄埗闈炰氦浜掑悗绔?
# ==============================
import matplotlib

matplotlib.use('Agg')
from hin_data_common import static_result_path, static_split_df
from baseline_checkpoint import checkpoint_config, maybe_resume_checkpoint, save_checkpoint
from torch.utils.data import Dataset, DataLoader


# ==============================
# 2. 鍩虹閰嶇疆
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
        self.cold_threshold = int(os.environ.get("GAR_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("GAR_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("GAR_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("GAR_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("GAR_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("GAR_STATIC_VAL_RATIO", "0.1"))
        self.batch_size = int(os.environ.get("GAR_BATCH_SIZE", "512"))
        self.n_epochs = int(os.environ.get("GAR_STATIC_EPOCHS", "40"))
        self.eval_interval = int(os.environ.get("GAR_EVAL_INTERVAL", "5"))
        self.eval_item_mode = os.environ.get("GAR_EVAL_ITEM_MODE", "mixed").strip().lower()
        self.rec_mode = os.environ.get("GAR_REC_MODE", "real_fake").strip().lower()
        self.lr = 1e-3
        self.ckpt = checkpoint_config("GAR")

        # --- GAFC (GAN) 鐗规湁鍙傛暟 ---
        self.alpha = 1.0  # Recommender Loss 鏉冮噸
        self.beta = 0.1  # Reconstruction Loss (MSE) 鏉冮噸 (璁烘枃涓€氬父淇濈暀浠ョǔ瀹氳缁?
        self.gamma = 0.5  # Adversarial Loss (Generator娆洪獥D) 鏉冮噸
        self.d_steps = 1  # 鍒ゅ埆鍣ㄨ缁冮娆?(姣忚缁?娆锛岃缁冨嚑娆)
        self.noise_dim = 16  # (鍙€? 娉ㄥ叆鐢熸垚鍣ㄧ殑鍣０缁村害锛岄儴鍒咷AN鍙樹綋浣跨敤


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
# 3. GAFC 妯″瀷瀹氫箟 (SIGIR '22)
# ==============================

class Generator(nn.Module):
    """
    鐢熸垚鍣?G: Content -> Fake ID Embedding
    """

    def __init__(self, content_dim, hidden_dim, emb_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, emb_dim),
            nn.Tanh()  # 闄愬埗杈撳嚭鑼冨洿锛屽尮閰?ID Embedding 鐨勫垎甯?
        )

    def forward(self, content):
        return self.net(content)


class Discriminator(nn.Module):
    """
    [鏂板] 鍒ゅ埆鍣?D: Embedding -> Probability (Real or Fake)
    """

    def __init__(self, emb_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 杈撳嚭 [0, 1] 姒傜巼
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
        鑾峰彇鐗╁搧鍚戦噺鐢ㄤ簬鎺ㄨ崘
        force_cold=True (鍏ㄩ噺璇勪及): 寮哄埗浣跨敤 Generator 鐢熸垚鐨?Fake ID
        """
        # A. Real ID
        real_id = self.item_id_emb(i_idx)

        # B. Fake ID
        content = self.content_features[i_idx]
        fake_id = self.generator(content)

        if force_cold:
            return fake_id, real_id  # 璇勪及鏃跺彧鐢ㄧ敓鎴愮殑

        # 璁粌鏃舵牴鎹瓥鐣ヨ繑鍥?
        # 璁烘枃涓帹鑽愪换鍔￠€氬父鍚屾椂浣跨敤 generated augmentation 鎴?strict G output
        # 杩欓噷涓轰簡瀵规姉璁粌绋冲畾鎬э紝閫氬父鎺ㄨ崘浠诲姟浣跨敤 Real ID + Generated ID 鐨勬贩鍚?
        return fake_id, real_id

        # 娉ㄦ剰锛欸AN 鐨?forward 閫昏緫姣旇緝澶嶆潅锛岄€氬父鎷嗗紑鍐欏湪 train loop 閲?

    # 杩欓噷鍙繚鐣欐帹鑽愰儴鍒嗙殑 forward 璁＄畻
    def recommend_score(self, u_idx, i_emb):
        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)
        z_i = F.normalize(i_emb, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        return logits


# ==============================
# 4. 鍏ㄩ噺鎺掑悕鐩稿叧鍑芥暟
# ==============================

def _mixed_item_vector(model, i_idx, cold_mask):
    content = model.content_features[i_idx]
    fake_id = model.generator(content)
    mode = getattr(model.cfg, "eval_item_mode", "mixed")
    if mode in {"generator", "generator_all", "fake", "fake_all"}:
        return fake_id
    if mode in {"real", "real_all", "id", "id_all"}:
        return model.item_id_emb(i_idx)

    cold_mask = cold_mask.to(device=i_idx.device).bool().view(-1, 1)
    if cold_mask.all():
        return fake_id
    real_id = model.item_id_emb(i_idx)
    if (~cold_mask).all():
        return real_id
    return torch.where(cold_mask, fake_id, real_id)


def precompute_full_pool(model, num_items, batch_size=2048, device='cuda', item_popularity=None):
    """
    棰勮绠? 寮哄埗浣跨敤 Generator
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print(f"Pre-computing Full Item Pool (GAFC eval_item_mode={model.cfg.eval_item_mode})...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            if item_popularity is None:
                cold_mask = torch.ones_like(i_batch, dtype=torch.bool)
            else:
                cold_mask = item_popularity[i_batch.detach().cpu()].to(device) < model.cfg.cold_threshold
            z_i = _mixed_item_vector(model, i_batch, cold_mask)
            z_i = F.normalize(z_i, dim=1)
            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_dual_gafc(model, loader, all_item_z, device, k_list, user_seen_items=None, average_mode="interaction"):
    """鍚屾椂璁＄畻 Cold 鍜?Hot 鍏ㄥ簱鎸囨爣"""
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    model.eval()
    c_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    h_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    c_total = 0
    h_total = 0
    c_item_sum = {f'{m}@{k}': {} for m in ['R', 'N'] for k in k_list}
    h_item_sum = {f'{m}@{k}': {} for m in ['R', 'N'] for k in k_list}
    c_item_count = {}
    h_item_count = {}
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
            
            z_i_pos = F.normalize(_mixed_item_vector(model, i_tgt, pop_mask.to(device)), dim=1)
            
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
            
            is_c = pop_mask.to(scores.device)
            is_h = ~is_c
            if average_mode == "item_macro":
                item_ids = [int(x) for x in i_tgt.detach().cpu().tolist()]
                cold_flags = [bool(x) for x in is_c.detach().cpu().tolist()]
                for item_id, is_cold_item in zip(item_ids, cold_flags):
                    counts = c_item_count if is_cold_item else h_item_count
                    counts[item_id] = counts.get(item_id, 0) + 1
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == t_cols).any(dim=1).float()
                # 淇鍚庣殑 NDCG 閫昏緫
                rks = (preds == t_cols).nonzero(as_tuple=True)
                dcgs = torch.zeros(u.size(0), device=scores.device)
                if rks[0].numel() > 0:
                    dcgs[rks[0]] = 1.0 / torch.log2(rks[1].float() + 2.0)
                
                if average_mode == "item_macro":
                    hit_vals = [float(x) for x in hits.detach().cpu().tolist()]
                    ndcg_vals = [float(x) for x in dcgs.detach().cpu().tolist()]
                    for row, item_id in enumerate(item_ids):
                        sums = c_item_sum if cold_flags[row] else h_item_sum
                        sums[f'R@{k}'][item_id] = sums[f'R@{k}'].get(item_id, 0.0) + hit_vals[row]
                        sums[f'N@{k}'][item_id] = sums[f'N@{k}'].get(item_id, 0.0) + ndcg_vals[row]
                else:
                    c_sum[f'R@{k}'] += hits[is_c].sum().item()
                    c_sum[f'N@{k}'] += dcgs[is_c].sum().item()
                    h_sum[f'R@{k}'] += hits[is_h].sum().item()
                    h_sum[f'N@{k}'] += dcgs[is_h].sum().item()

            if average_mode == "interaction":
                c_total += is_c.sum().item()
                h_total += is_h.sum().item()
            
    if average_mode == "item_macro":
        def macro_result(item_sum, item_count):
            if not item_count:
                return None, 0
            res = {}
            for key, per_item in item_sum.items():
                vals = [
                    per_item.get(item_id, 0.0) / count
                    for item_id, count in item_count.items()
                    if count > 0
                ]
                res[key] = sum(vals) / max(1, len(vals))
            return res, len(item_count)

        c_res, c_count = macro_result(c_item_sum, c_item_count)
        h_res, h_count = macro_result(h_item_sum, h_item_count)
        return c_res, c_count, h_res, h_count

    c_res = {k: v/c_total for k,v in c_sum.items()} if c_total > 0 else None
    h_res = {k: v/h_total for k,v in h_sum.items()} if h_total > 0 else None
    return c_res, c_total, h_res, h_total

def evaluate_sampled_gafc(model, loader, all_item_z, device, k_list, n_neg=999, user_seen_items=None):
    """璁＄畻鍩轰簬 1姝?999璐?閲囨牱鐨?Cold 鍜?Hot 鎸囨爣"""
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
            z_i_pos = F.normalize(_mixed_item_vector(model, i_tgt, pop_mask.to(device)), dim=1)

            # Compute full scores first, then gather sampled candidates from the same
            # score matrix to keep full/sample numerically consistent.
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

            score_device = scores_full.device
            t_cols_sample = i_tgt.to(score_device)
            neg_items = torch.from_numpy(neg_np).to(score_device)
            cand_idx = torch.cat([t_cols_sample.unsqueeze(1), neg_items], dim=1)

            # Randomize candidate order to avoid positional tie-bias
            # (target fixed at col 0 can inflate sampled metrics under many ties).
            perm = torch.argsort(torch.rand(batch_size, cand_idx.size(1), device=cand_idx.device), dim=1)
            cand_idx = cand_idx.gather(1, perm)
            target_cols = (cand_idx == t_cols_sample.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1, 1)
            scores = scores_full.gather(1, cand_idx)
            
            max_k = min(max(k_list), scores.size(1))
            _, topk = torch.topk(scores, k=max_k, dim=1)
            
            is_c = pop_mask.to(scores.device)
            is_h = ~is_c
            
            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
                # 淇鍚庣殑 NDCG 閫昏緫
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
# 5. 涓昏缁冨惊鐜?(瀵规姉璁粌鏍稿績)
# ==============================

def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for GAFC (SIGIR '22) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print(f"Error: {data_dir}/stream_data.pkl not found")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    setup_seed(cfg.seed)
    train_df, val_df, test_df = static_split_df(
        df, seed=cfg.static_seed, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio
    )

    train_loader = DataLoader(StreamDataset(train_df), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(StreamDataset(val_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(StreamDataset(test_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    item_popularity = torch.zeros(cfg.n_items, dtype=torch.long)
    train_counts = train_df["i_idx"].astype(int).value_counts()
    for item_id, count in train_counts.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_popularity[idx] = int(count)

    print(
        f">> Model: GAFC (GAN) | Alpha: {cfg.alpha} | Gamma (Adv): {cfg.gamma} | "
        f"STATIC | eval_n_neg={cfg.eval_n_neg} | eval_item_mode={cfg.eval_item_mode} | rec_mode={cfg.rec_mode}"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAFC(cfg, content_emb).to(device)

    opt_g = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.user_emb.parameters()},
        {'params': model.item_id_emb.parameters()}
    ], lr=cfg.lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr)

    k_list = [5, 10, 20]
    metrics_keys = [f'{m}@{k}' for m in ['R', 'N'] for k in k_list]

    # Seen cache: validation masks train seen; test uses train-only unless configured otherwise.
    train_seen_items = {}
    for u_idx, i_idx in zip(train_df['u_idx'].values, train_df['i_idx'].values):
        uid = int(u_idx)
        if uid not in train_seen_items:
            train_seen_items[uid] = set()
        train_seen_items[uid].add(int(i_idx))

    test_seen_items = {uid: set(items) for uid, items in train_seen_items.items()}
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        for u_idx, i_idx in zip(val_df['u_idx'].values, val_df['i_idx'].values):
            uid = int(u_idx)
            if uid not in test_seen_items:
                test_seen_items[uid] = set()
            test_seen_items[uid].add(int(i_idx))

    criterion_gan = nn.BCELoss()
    epochs = cfg.n_epochs
    best_val = -1.0
    best_epoch = -1
    best_state = None
    start_epoch, ckpt_state = maybe_resume_checkpoint(
        cfg.ckpt,
        model,
        {"opt_g": opt_g, "opt_d": opt_d},
        device,
    )
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        steps = 0

        for batch, pop in train_loader:
            u_idx = batch['u'].to(device)
            i_idx = batch['i'].to(device)
            batch_size = u_idx.size(0)

            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)

            # 1) Train D
            opt_d.zero_grad()
            real_emb = model.item_id_emb(i_idx).detach()
            prob_real = model.discriminator(real_emb)
            loss_d_real = criterion_gan(prob_real, real_label)

            content = model.content_features[i_idx]
            fake_emb = model.generator(content).detach()
            prob_fake = model.discriminator(fake_emb)
            loss_d_fake = criterion_gan(prob_fake, fake_label)

            loss_d = (loss_d_real + loss_d_fake) / 2.0
            loss_d.backward()
            opt_d.step()

            # 2) Train G + recommender
            opt_g.zero_grad()
            fake_emb_g = model.generator(content)
            real_emb_g = model.item_id_emb(i_idx)
            prob_fake_g = model.discriminator(fake_emb_g)
            loss_g_adv = criterion_gan(prob_fake_g, real_label)

            rec_labels = torch.arange(batch_size, device=device)
            logits_fake = model.recommend_score(u_idx, fake_emb_g)
            loss_rec_fake = F.cross_entropy(logits_fake, rec_labels)
            if cfg.rec_mode in {"real_fake", "fake_real", "both", "mixed"}:
                logits_real = model.recommend_score(u_idx, real_emb_g)
                loss_rec_real = F.cross_entropy(logits_real, rec_labels)
                loss_rec = 0.5 * (loss_rec_fake + loss_rec_real)
            elif cfg.rec_mode in {"real", "id", "real_only"}:
                logits_real = model.recommend_score(u_idx, real_emb_g)
                loss_rec = F.cross_entropy(logits_real, rec_labels)
            else:
                loss_rec = loss_rec_fake

            loss_recon = F.mse_loss(fake_emb_g, real_emb_g)

            loss_g = cfg.alpha * loss_rec + cfg.gamma * loss_g_adv + cfg.beta * loss_recon
            loss_g.backward()
            opt_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            steps += 1

        print(f"Epoch [{epoch}/{epochs}] Train G_Loss: {total_g_loss / max(1, steps):.4f} | D_Loss: {total_d_loss / max(1, steps):.4f}")

        if epoch % cfg.eval_interval == 0 or epoch == epochs:
            improved = False
            all_z = precompute_full_pool(model, cfg.n_items, device=device, item_popularity=item_popularity)
            c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_gafc(
                model, val_loader, all_z, device, k_list, user_seen_items=train_seen_items
            )
            val_key = c_m_f.get("N@10", 0.0) if c_m_f else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                improved = True
            if cfg.ckpt.save and improved:
                save_checkpoint(
                    cfg.ckpt,
                    "best.pt",
                    epoch,
                    model,
                    {"opt_g": opt_g, "opt_d": opt_d},
                    best_state=best_state,
                    extra={"best_val": best_val, "best_epoch": best_epoch},
                )
            c_f_str = " | ".join([f"{k}={c_m_f[k]:.4f}" for k in metrics_keys[:3]]) if c_m_f else "N/A"
            h_f_str = " | ".join([f"{k}={h_m_f[k]:.4f}" for k in metrics_keys[:3]]) if h_m_f else "N/A"
            print(f"  --> Valid Cold Full: {c_f_str} | Hot Full: {h_f_str}")
        if cfg.ckpt.save:
            save_checkpoint(
                cfg.ckpt,
                "latest.pt",
                epoch,
                model,
                {"opt_g": opt_g, "opt_d": opt_d},
                best_state=best_state,
                extra={"best_val": best_val, "best_epoch": best_epoch},
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restore best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    print("\n" + "=" * 90)
    print(f"         FINAL TEST REPORT: Sampled (1+{cfg.eval_n_neg}) vs Full Ranking (STATIC)")
    print("=" * 90)

    all_z = precompute_full_pool(model, cfg.n_items, device=device, item_popularity=item_popularity)
    c_m_f, n_c_f, h_m_f, n_h_f = evaluate_dual_gafc(
        model, test_loader, all_z, device, k_list, user_seen_items=test_seen_items
    )
    c_m_f_item_macro, n_c_f_item_macro, h_m_f_item_macro, n_h_f_item_macro = evaluate_dual_gafc(
        model,
        test_loader,
        all_z,
        device,
        k_list,
        user_seen_items=test_seen_items,
        average_mode="item_macro",
    )
    c_m_s, n_c_s, h_m_s, n_h_s = evaluate_sampled_gafc(
        model, test_loader, all_z, device, k_list, n_neg=cfg.eval_n_neg, user_seen_items=test_seen_items
    )

    print(f"{'Metric':<10} | {'Samp Cold':<12} | {'Samp Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for k in metrics_keys:
        v_s_c = c_m_s.get(k, 0.0) if c_m_s else 0.0
        v_s_h = h_m_s.get(k, 0.0) if h_m_s else 0.0
        v_f_c = c_m_f.get(k, 0.0) if c_m_f else 0.0
        v_f_h = h_m_f.get(k, 0.0) if h_m_f else 0.0
        print(f"{k:<10} | {v_s_c:<12.4f} | {v_s_h:<12.4f} | {v_f_c:<12.4f} | {v_f_h:<12.4f}")
    print("-" * 90)
    print(f"采样 Samples: Cold={n_c_s}, Hot={n_h_s}")
    print(f"全库 Samples: Cold={n_c_f}, Hot={n_h_f}")
    print("=" * 90)

    out = {
        "model": "GAR",
        "protocol": "static_item_cold",
        "sample_cold": c_m_s or {},
        "sample_hot": h_m_s or {},
        "full_cold": c_m_f or {},
        "full_hot": h_m_f or {},
        "full_cold_item_macro": c_m_f_item_macro or {},
        "full_hot_item_macro": h_m_f_item_macro or {},
    }
    for k in metrics_keys:
        out[f"samp_cold_{k}"] = c_m_s.get(k, 0.0) if c_m_s else 0.0
        out[f"samp_hot_{k}"] = h_m_s.get(k, 0.0) if h_m_s else 0.0
        out[f"full_cold_{k}"] = c_m_f.get(k, 0.0) if c_m_f else 0.0
        out[f"full_hot_{k}"] = h_m_f.get(k, 0.0) if h_m_f else 0.0
    out.update({
        "count_sample_cold": n_c_s,
        "count_sample_hot": n_h_s,
        "count_full_cold": n_c_f,
        "count_full_hot": n_h_f,
        "count_full_cold_item_macro": n_c_f_item_macro,
        "count_full_hot_item_macro": n_h_f_item_macro,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_metric": "cold",
        "eval_n_neg": cfg.eval_n_neg,
        "eval_item_mode": cfg.eval_item_mode,
        "rec_mode": cfg.rec_mode,
        "checkpoint_dir": cfg.ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
    })
    result_path = static_result_path("gar_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
