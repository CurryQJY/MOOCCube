import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os, random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==========================================
# 0. 基础设置
# ==========================================
def setup_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 随机种子已固定: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.num_users = n_users
        self.num_items = n_items
        self.user_dim = 64
        self.content_dim = content_dim
        self.behavior_dim = 64
        self.hidden_dims = [128, 64]

        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5
        self.gamma_s = 5.0  # Syllabus Enhancer weight

        self.inner_lr = 0.001
        self.outer_lr = 0.001
        self.temp = 0.1


# ==========================================
# 1. 模型定义 (PAM + MAML + Predict_All)
# ==========================================
class PAM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config

        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        if config.content_dim != config.behavior_dim:
            self.con_proj = nn.Linear(config.content_dim, config.behavior_dim)
        else:
            self.con_proj = nn.Identity()

        # Meta-Parameters
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        dims_u = [config.user_dim] + config.hidden_dims
        dims_i = [config.behavior_dim * 2] + config.hidden_dims

        for dims in [dims_u, dims_i]:
            for i in range(len(dims) - 1):
                w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
                nn.init.xavier_normal_(w)
                b = nn.Parameter(torch.zeros(dims[i + 1]))
                self.vars.extend([w, b])
                self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                                  nn.Parameter(torch.ones_like(b) * config.inner_lr)])

        # self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-2]))
        self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-1]))
        self.sup_b = nn.Parameter(torch.zeros(config.behavior_dim))

    def get_item_features(self, i_idx):
        beh = self.item_beh_emb(i_idx)
        con = self.item_con_emb(i_idx)
        con = self.con_proj(con)
        return torch.cat([beh, con], dim=1)

    def forward_mlp(self, x, weights, is_item=False):
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        for i in range(len(self.cfg.hidden_dims)):
            w, b = weights[idx_start + 2 * i], weights[idx_start + 2 * i + 1]
            out = F.linear(out, w, b)
            if i < len(self.cfg.hidden_dims) - 1:
                out = F.relu(out)
        return out

    # --- 新增: 全量预测函数 ---
    def predict_all(self, u_idx, device):
        """
        计算 User Batch 对 所有物品 的打分
        Returns: [Batch, N_items]
        """
        # 1. User Representation
        e_u = self.user_emb(u_idx)
        z_u = self.forward_mlp(e_u, self.vars, False)  # [B, Dim]

        # 2. Item Representation (All Items)
        all_items = torch.arange(self.cfg.num_items).to(device)
        e_i = self.get_item_features(all_items)
        z_i = self.forward_mlp(e_i, self.vars, True)  # [N_items, Dim]

        # 3. Dot Product
        scores = torch.mm(z_u, z_i.t())  # [B, N_items]
        return scores

    # --- 训练逻辑 (Meta Inner Loop) ---
    def inner_loop(self, u, i):
        e_u = self.user_emb(u)
        e_i = self.get_item_features(i)
        z_u = self.forward_mlp(e_u, self.vars, False)
        z_i = self.forward_mlp(e_i, self.vars, True)

        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))

        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold
        total_loss = 0

        # Meta-Learning Splits
        task_splits = {}
        if is_cold.sum() >= 2: task_splits['cold'] = {'u': u[is_cold], 'i': i[is_cold]}
        if (~is_cold).sum() >= 2: task_splits['hot'] = {'u': u[~is_cold], 'i': i[~is_cold]}

        for name, data in task_splits.items():
            split = len(data['u']) // 2
            if split < 1: continue
            su, si = data['u'][:split], data['i'][:split]
            qu, qi = data['u'][split:], data['i'][split:]

            omega = self.inner_loop(su, si)

            e_u = self.user_emb(qu)
            e_i = self.get_item_features(qi)
            z_u = self.forward_mlp(e_u, omega, False)
            z_i = self.forward_mlp(e_i, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        # Enhancer Loss (Hot Items)
        if (~is_cold).sum() > 0:
            hi = i[~is_cold]
            e_i_hot = self.get_item_features(hi)
            z_i_hot = self.forward_mlp(e_i_hot, self.vars, True)
            pred_id = F.linear(z_i_hot, self.sup_w, self.sup_b)
            real_id = self.item_beh_emb(hi).detach()
            total_loss += self.cfg.gamma_s * F.mse_loss(pred_id, real_id)

        return total_loss


# ==========================================
# 2. 数据与工具
# ==========================================
class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self): return len(self.u)

    def __getitem__(self, idx): return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}


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
    periods = [df[df['period_id'] == p].reset_index(drop=True) for p in sorted(df['period_id'].unique())]
    return periods


# ==========================================
# 3. 核心评估函数 (Full Ranking)
# ==========================================
def evaluate_full_ranking(model, loader, device, global_history, n_items, k_list=[5, 10, 20]):
    """
    全量排序评估:
    1. 计算 User vs All Items 分数
    2. 屏蔽历史交互 (global_history)
    3. 计算 Recall/NDCG
    """
    model.eval()

    metrics = {f'R@{k}': 0.0 for k in k_list}
    metrics.update({f'N@{k}': 0.0 for k in k_list})
    total_cold = 0

    with torch.no_grad():
        for batch, pop in loader:
            # 只评估冷启动用户
            mask = pop < model.cfg.cold_threshold
            n_batch_cold = mask.sum().item()
            if n_batch_cold == 0: continue

            u_test = batch['u'][mask].to(device)
            target_items = batch['i'][mask].to(device)  # Ground Truth

            # 1. 全量打分 [B, N_items]
            scores = model.predict_all(u_test, device)

            # 2. History Masking (关键步骤)
            # 将用户以前看过的物品分数设为 -inf
            u_cpu = u_test.cpu().numpy()
            for idx, uid in enumerate(u_cpu):
                visited_items = global_history.get(uid, [])
                if visited_items:
                    # 只有当索引在范围内时才 mask
                    valid_visited = [x for x in visited_items if x < n_items]
                    scores[idx, valid_visited] = -float('inf')

            # 3. Top-K Metrics
            # 我们只关心 target_item 排在哪里
            # 为了速度，我们直接看 target_item 的分数，算出比它大的有多少个

            # 方法: 将 target_item 的分数取出来，和所有分数比
            # 这种方法比 argsort 快很多

            # 获取每个用户对应的 Target Item 的分数
            # scores: [B, N], target_items: [B]
            target_scores = scores.gather(1, target_items.view(-1, 1))  # [B, 1]

            # 计算排名: 有多少个物品的分数 > target_score
            # rank = (scores > target_scores).sum(dim=1) + 1
            # 这里的 compare 会包含 target 自身(如果不小心 > 的话)，严谨用 >
            ranks = (scores > target_scores).sum(dim=1).float() + 1

            for k in k_list:
                # Recall: 排名 <= K 即为命中
                hits = (ranks <= k).float()
                metrics[f'R@{k}'] += hits.sum().item()

                # NDCG: 1 / log2(rank + 1)
                ndcg = (1.0 / torch.log2(ranks + 1.0)) * hits
                metrics[f'N@{k}'] += ndcg.sum().item()

            total_cold += n_batch_cold

    if total_cold == 0: return None, 0

    # 平均化
    avg_metrics = {k: v / total_cold for k, v in metrics.items()}
    return avg_metrics, total_cold


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print("🚀 启动流式全量评估训练...")

    # 1. 数据准备
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("❌ Error: processed_data/stream_data.pkl 不存在")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")

    if os.path.exists("processed_data/content_emb.pt"):
        print("📥 加载 BERT 向量...")
        content_emb = torch.load("processed_data/content_emb.pt", map_location='cpu')
    else:
        print("⚠️ 警告: 使用随机 BERT 向量 (Run gen_bert_emb.py first!)")
        content_emb = torch.randn(meta['n_items'] + 1, 384)

    # 2. 周期切分
    periods = split_dataframe_by_periods(df, period_type='M')

    # 3. 模型初始化
    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")

    model = PAM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # 4. 全局历史记录器 (用于 Masking)
    global_history = {}

    # 5. 记录器
    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]
    history_log = {k: [] for k in metrics_keys}
    history_log['Period'] = []
    history_log['Count'] = []

    WARMUP = 3

    print(f"\n>>> 开始流式训练 (Periods: {len(periods)}) <<<")

    for t, p_df in enumerate(periods):
        # 构造 DataLoader
        loader = DataLoader(StreamDataset(p_df), batch_size=2048, shuffle=False, collate_fn=collate_fn)
        n_samples = len(p_df)

        print(f"\n📅 Period {t} (Samples: {n_samples})")

        # --- Phase 1: Test (Next-Period Prediction) ---
        # 测试当前周期的数据 (在模型没见过之前)
        current_res = {}
        test_cnt = 0

        if t >= WARMUP:
            # 全量评估
            met, n_cold = evaluate_full_ranking(model, loader, device, global_history, cfg.num_items, k_list)

            if met:
                current_res = met
                test_cnt = n_cold
                print(f"  [TEST] (Cold={n_cold}) R@10: {met['R@10']:.4f} | N@10: {met['N@10']:.4f}")
            else:
                print("  [TEST] No cold items.")
        else:
            print("  [WARMUP] Skipping evaluation.")

        # 记录日志
        history_log['Period'].append(t)
        history_log['Count'].append(test_cnt)
        for k in metrics_keys: history_log[k].append(current_res.get(k, 0.0))

        # --- Phase 2: Train ---
        model.train()
        total_loss = 0

        # 使用 tqdm 显示训练进度
        pbar = tqdm(loader, desc="  [TRAIN]", leave=False, ncols=80)
        for batch, pop in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pop = pop.to(device)

            loss = model(batch, pop)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # --- Phase 3: Update History (On-the-fly) ---
            # 训练过的交互，加入历史记录，供下个周期的评估 Mask 使用
            u_np = batch['u'].cpu().numpy()
            i_np = batch['i'].cpu().numpy()
            for uid, iid in zip(u_np, i_np):
                if uid not in global_history: global_history[uid] = []
                global_history[uid].append(iid)

    # ==========================================
    # 5. 最终报告
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 FINAL STREAMING REPORT (Full Ranking)")
    print("=" * 60)

    counts = np.array(history_log['Count'])
    valid_mask = (np.array(history_log['Period']) >= WARMUP) & (counts > 0)

    if valid_mask.sum() > 0:
        valid_counts = counts[valid_mask]
        print(f"Total Cold Samples: {valid_counts.sum()}")
        print("-" * 60)
        print(f"{'Metric':<10} | {'Weighted Avg':<15} | {'Simple Avg'}")
        print("-" * 60)

        for k in metrics_keys:
            vals = np.array(history_log[k])[valid_mask]
            w_avg = np.average(vals, weights=valid_counts)
            s_avg = np.mean(vals)
            print(f"{k:<10} | {w_avg:.4f}          | {s_avg:.4f}")

        # 保存 CSV
        pd.DataFrame(history_log).to_csv('pam_full_ranking_results.csv', index=False)
        print("\n✅ 结果已保存至 pam_full_ranking_results.csv")
    else:
        print("No valid test periods.")


if __name__ == "__main__":
    setup_seed(0)
    main()