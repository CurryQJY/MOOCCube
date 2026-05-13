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
# 2. 基础工具 (保持不变以确保公平)
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
        self.emb_dim = 128           # [优化] 增大 embedding 维度
        self.content_dim = content_dim
        self.hidden_dim = 256        # [优化] 增大隐藏层
        self.cold_threshold = 5

        self.lr = 5e-4
        self.dropout_prob = 0.35     # [优化] 增大丢弃率 (Capacity & Temp Tuning)
        self.aux_weight = 0.3        # [优化] 增大对比损失权重
        self.n_epochs = 3            # [新增] 每个 period 训练多轮
        self.margin = 0.15           # [优化] Phase 2: Additive Margin


class StreamDataset(Dataset):
    def __init__(self, df, llm_scores):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.tensor([llm_scores.get(int(idx), -1.0) for idx in self.i], dtype=torch.float)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm': self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm = torch.stack([item['llm'] for item in batch])
    return {'u': u, 'i': i, 'llm': llm}, pop


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


def compute_ranking_metrics(scores, target_indices, k_list=[5, 10, 20]):
    """
    全库排名评估指标计算。
    Args:
        scores: [B, N_items] 每个用户对所有物品的打分
        target_indices: [B] 每个用户的真实物品索引 (在全库中的 item idx)
        k_list: 评估的 K 值列表
    """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)  # [B, 1]
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        recall = hits.mean().item()
        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0
        results[f'R@{k}'] = recall
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results


# ==============================
# 3. 增强版 PAM 模型核心 (The Winning Part)
# ==============================

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

        # 2. [优化] 更深的内容编码器 (3层 + Dropout + 残差)
        self.content_features = content_emb.to(self.device)
        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim)
        )

        # 3. [优化] User 也加投影层，增强表达能力
        self.user_proj = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim)
        )

        # [新增] LLM 投影层
        self.llm_proj = nn.Linear(1, cfg.emb_dim)

        # 4. 门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid()
        )

        self.temperature = 0.07  # [优化] Phase 1: 降低温度，增强困难负样本惩罚

    def get_item_vector(self, i_idx, llm_s, force_cold=False):
        id_e = self.item_id_emb(i_idx)

        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        raw_content = self.content_features[i_idx]
        content_e = self.content_proj(raw_content)

        # [新增] 将 LLM 特征融入内容编码
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        content_e = content_e + llm_e

        concat = torch.cat([id_e, content_e], dim=-1)
        alpha = self.gate_net(concat)
        final_item_e = alpha * id_e + (1 - alpha) * content_e

        return final_item_e, id_e, content_e

    def forward(self, batch):
        u_idx = batch['u']
        i_idx = batch['i']
        llm_s = batch['llm']

        # 1. User Vector (with projection)
        z_u = self.user_proj(self.user_emb(u_idx))
        z_u = F.normalize(z_u, dim=1)

        # 2. Item Vector
        z_i, id_e_raw, content_e = self.get_item_vector(i_idx, llm_s, force_cold=False)
        z_i = F.normalize(z_i, dim=1)

        # 3. 主任务: InfoNCE
        logits = torch.matmul(z_u, z_i.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # [优化] Phase 2: Additive Margin for Hard Negatives
        # 仅对正样本（对角线）减去 margin
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.temperature
        main_loss = F.cross_entropy(logits_margin, labels)

        if self.training:
            # 辅助对比损失: Content <-> ID 对齐
            z_id = F.normalize(id_e_raw, dim=1)
            z_con = F.normalize(content_e, dim=1)
            sim = torch.matmul(z_id, z_con.t()) / self.temperature
            aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

            total_loss = main_loss + self.cfg.aux_weight * aux_loss
            return total_loss

        return main_loss


# ==============================
# 4. 评估函数 (PAM Enhanced)
# ==============================

def evaluate_pam(model, loader, device, llm_scores, k_list=[5, 10, 20], n_neg=999, eval_type='cold', full_ranking=False):
    """
    评估函数。
    full_ranking=False: 采样评估 (1 正 + n_neg 负)
    full_ranking=True:  全库排名 (对所有 N_items 排名)
    eval_type: 'cold' | 'hot' | 'all'
    """
    model.eval()
    accum_metrics = {}
    total_samples = 0

    with torch.no_grad():
        # ====== Step 1: 预计算全库物品向量 ======
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        all_llm_s = torch.tensor([llm_scores.get(int(idx), -1.0) for idx in all_item_idx], dtype=torch.float, device=device)

        use_cold = (eval_type == 'cold')

        ITEM_BATCH = 4096
        all_item_vecs = []
        for start in range(0, n_items, ITEM_BATCH):
            end = min(start + ITEM_BATCH, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]
            z_batch, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=use_cold)
            z_batch = F.normalize(z_batch, dim=1)
            all_item_vecs.append(z_batch)
        all_item_vecs = torch.cat(all_item_vecs, dim=0)  # [N_items, emb_dim]

        # ====== Step 2: 评估 ======
        for batch, pop in loader:
            if eval_type == 'cold':
                mask = pop < model.cfg.cold_threshold
            elif eval_type == 'hot':
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u_test = batch['u'][mask].to(device)
            i_test = batch['i'][mask].to(device)

            z_u = model.user_proj(model.user_emb(u_test))
            z_u = F.normalize(z_u, dim=1)

            if full_ranking:
                # 全库排名: [B, N_items]
                scores = torch.mm(z_u, all_item_vecs.t())
                target_indices = i_test  # 全局 item idx
            else:
                # 采样评估: [B, 1+n_neg]
                neg_items = torch.randint(0, n_items, (n_sel, n_neg), device=device)
                cand_idx = torch.cat([i_test.unsqueeze(1), neg_items], dim=1)
                cand_vecs = all_item_vecs[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)

            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0

    avg_metrics = {k: v / total_samples for k, v in accum_metrics.items()}
    return avg_metrics, total_samples


# ==============================
# 5. 主训练循环
# ==============================

def main():
    DATA_DIR = "processed_data_hin"
    print(f"Loading Data for PAM Enhanced (HIN) from {DATA_DIR}...")
    if not os.path.exists(f"{DATA_DIR}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{DATA_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{DATA_DIR}/stream_data.pkl")
    with open(f"{DATA_DIR}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{DATA_DIR}/content_emb.pt")

    periods = split_dataframe_by_periods(df, period_type='M')

    dataloaders = []
    for p_df in periods:
        ds = StreamDataset(p_df, llm_scores)
        dl = DataLoader(ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        dataloaders.append(dl)

    # 配置
    cfg = Config(meta['n_users'], meta['n_items'], content_dim=content_emb.shape[1])
    print(f">> Model: PAM (Enhanced) | Dropout Prob: {cfg.dropout_prob} | Aux Weight: {cfg.aux_weight}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_Enhanced(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"\n>>> 开始累积训练评估 (PAM Enhanced) - 共 {len(periods)} 个周期 <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]
    history = {'Period': [], 'Count_cold': [], 'Count_hot': []}
    for prefix in ['cold_', 'hot_']:
        for k in metrics_keys:
            history[prefix + k] = []

    # 三类评估累加器 (采样)
    accum_cold = {k: 0.0 for k in metrics_keys}
    accum_hot = {k: 0.0 for k in metrics_keys}
    accum_all = {k: 0.0 for k in metrics_keys}
    count_cold, count_hot, count_all = 0, 0, 0

    # 全库排名累加器
    full_cold = {k: 0.0 for k in metrics_keys}
    full_hot = {k: 0.0 for k in metrics_keys}
    full_all = {k: 0.0 for k in metrics_keys}
    fc_cold, fc_hot, fc_all = 0, 0, 0

    WARMUP_PERIODS = 3
    accumulated_dfs = []

    for t in range(len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= WARMUP_PERIODS:
            # 采样评估
            met_cold, n_cold_t = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='cold')
            met_hot, n_hot_t = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='hot')
            met_all, n_all_t = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='all')
            # 全库排名评估
            fmet_cold, fn_c = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='cold', full_ranking=True)
            fmet_hot, fn_h = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='hot', full_ranking=True)
            fmet_all, fn_a = evaluate_pam(model, eval_loader, device, llm_scores, k_list, eval_type='all', full_ranking=True)

            if met_cold:
                cold_res = met_cold
                for k in metrics_keys: accum_cold[k] += met_cold[k] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for k in metrics_keys: accum_hot[k] += met_hot[k] * n_hot_t
                count_hot += n_hot_t
            if met_all:
                for k in metrics_keys: accum_all[k] += met_all[k] * n_all_t
                count_all += n_all_t
            if fmet_cold:
                for k in metrics_keys: full_cold[k] += fmet_cold[k] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for k in metrics_keys: full_hot[k] += fmet_hot[k] * fn_h
                fc_hot += fn_h
            if fmet_all:
                for k in metrics_keys: full_all[k] += fmet_all[k] * fn_a
                fc_all += fn_a

            c_s = met_cold['R@10'] if met_cold else 0
            h_s = met_hot['R@10'] if met_hot else 0
            c_f = fmet_cold['R@10'] if fmet_cold else 0
            h_f = fmet_hot['R@10'] if fmet_hot else 0
            print(f"  采样 Cold={c_s:.4f} Hot={h_s:.4f} | 全库 Cold={c_f:.4f} Hot={h_f:.4f}")
        else:
            print("  [WARMUP] Training only...")

        history['Period'].append(t)
        history['Count_cold'].append(n_cold_t)
        history['Count_hot'].append(n_hot_t)
        for k in metrics_keys:
            history['cold_' + k].append(cold_res.get(k, 0.0))
            history['hot_' + k].append(hot_res.get(k, 0.0))

        # --- Phase 2: 累积训练 ---
        accumulated_dfs.append(p_df)
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, collate_fn=collate_fn)

        model.train()
        for epoch in range(cfg.n_epochs):
            total_loss = 0
            steps = 0
            for batch, pop in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            if epoch == cfg.n_epochs - 1:
                print(f"  [TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | 累积: {len(combined_df)} | Loss: {total_loss / steps:.4f}")

    # ==============================
    # 6. 最终报告: 采样 vs 全库排名
    # ==============================
    print("\n" + "=" * 90)
    print("         FINAL REPORT: 采样评估 (1+999) vs 全库排名")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)

    for m in metrics_keys:
        sc = accum_cold[m] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[m] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[m] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[m] / fc_hot if fc_hot > 0 else 0.0
        print(f"{m:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv('mooc_metrics_pam_hin.csv', index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history['Period'], history['cold_R@10'], marker='o', label='Cold R@10')
    plt.plot(history['Period'], history['hot_R@10'], marker='s', label='Hot R@10')
    plt.axvline(x=WARMUP_PERIODS - 0.5, color='r', linestyle='--', label='Warmup End')
    plt.title('PAM Enhanced: Cumulative Training')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('mooc_result_pam_hin.png')
    print(">> Saved mooc_result_pam_hin.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()