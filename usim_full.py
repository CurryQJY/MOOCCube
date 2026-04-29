import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import json
import os
import pickle
import random
import matplotlib

# 强制使用非交互式后端
matplotlib.use('Agg')
from torch.utils.data import Dataset, DataLoader


# ================= 1. 基础配置与工具 =================

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
        # --- 基础维度 ---
        self.num_users = n_users
        self.num_items = n_items
        self.user_dim = 64
        self.behavior_dim = 64
        self.content_dim = content_dim
        self.hidden_dims = [128, 64]  # MLP 隐藏层

        # --- 阈值与权重 ---
        self.cold_threshold = 5  # <5 交互视为冷启动
        self.lambda_cold = 2.0  # Meta-Learning 冷启动权重
        self.lambda_hot = 0.5

        # --- 优化器 ---
        self.inner_lr = 0.001  # MAML 内层更新步长
        self.outer_lr = 0.0005  # 全局 Adam 步长
        self.temp = 0.1  # Softmax 温度

        # --- RL (User Simulation) ---
        self.usim_steps = 9  # 想象迭代步数 (T=0,1,2)
        self.n_candidates = 20  # 动作空间大小 (随机候选人数)
        self.usim_lr = 0.8  # 梯度上升步长 (Eta)

        # PPO 超参
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_coeffs = {'value': 0.5, 'entropy': 0.01}


# ================= 2. PPO Agent (Actor-Critic) =================

class SimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=4):
        super(SimpleAC, self).__init__()
        input_dim = item_dim + time_dim

        # 共享感知层
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Actor: 输出查询向量，用于 Attention 选人
        self.actor_head = nn.Linear(128, 128)
        # Critic: 预测状态价值 V(s)
        self.critic_head = nn.Linear(128, 1)
        # User Projection: 将候选用户映射到同一空间
        self.user_proj = nn.Linear(item_dim, 128)

    def get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
        # 1. 状态构造: Concat[Item, Time_OneHot]
        # time_step: [B, 1] -> OneHot [B, 4]
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=10)[:, :4].float()
        state = torch.cat([item_state, t_emb], dim=1)

        feat = self.common(state)
        value = self.critic_head(feat)

        # 2. Actor 策略 (Attention over Candidates)
        query = self.actor_head(feat).unsqueeze(1)  # [B, 1, 128]
        keys = self.user_proj(candidates_emb)  # [B, N_cand, 128]
        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)  # [B, N_cand]

        dist = Categorical(logits=logits)

        # 3. 采样动作
        if action_idx is None:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        return action_idx, log_prob, value, entropy


# ================= 3. 主模型: PAM_RL_USIM =================

class PAM_RL_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === Embeddings ===
        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        # item_beh_emb: 真实 ID Embedding (仅作为 RL 的 Target/Teacher，不作为 Input)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        # item_con_emb: 冻结的 BERT 向量
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # === Projections ===
        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim, 256), nn.ReLU(),
            nn.Linear(256, config.behavior_dim), nn.LayerNorm(config.behavior_dim)
        )
        self.llm_proj = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, config.behavior_dim)
        )

        # === RL Agent ===
        self.agent = SimpleAC(config.behavior_dim, time_dim=4)

        # === MAML Meta-Parameters (The Two-Tower MLP) ===
        # 使用 ParameterList 以便在 Inner Loop 中手动更新梯度
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()  # Layer-specific Learning Rates

        # 定义双塔参数结构: [User_Dim, 128, 64]
        dims = [config.user_dim] + config.hidden_dims

        # 1. 初始化 User Tower 参数 (Indices 0 ~ 3)
        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
            nn.init.xavier_normal_(w)
            b = nn.Parameter(torch.zeros(dims[i + 1]))
            self.vars.extend([w, b])
            self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                              nn.Parameter(torch.ones_like(b) * config.inner_lr)])

        # 2. 初始化 Item Tower 参数 (Indices 4 ~ 7)
        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
            nn.init.xavier_normal_(w)
            b = nn.Parameter(torch.zeros(dims[i + 1]))
            self.vars.extend([w, b])
            self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                              nn.Parameter(torch.ones_like(b) * config.inner_lr)])

    def get_candidates(self, item_emb):
        """随机采样候选用户池 (Random Sampling)"""
        B = item_emb.size(0)
        N_cand = self.cfg.n_candidates
        # 全局随机采样
        rand_idx = torch.randint(0, self.cfg.num_users, (B, N_cand), device=self.device)
        cand_emb = self.user_emb(rand_idx).detach()  # Detach! 这里的用户只是参考点
        return cand_emb

    def run_usim_episode(self, init_item_emb, target_id_emb=None):
        """
        RL 想象循环 (The Core Loop)
        :param init_item_emb: 初始状态 (Content + LLM)
        :param target_id_emb: 真实 ID Embedding (Ground Truth)
        """
        current_h = init_item_emb.clone()
        trajectory = {'log_probs': [], 'values': [], 'rewards': [], 'entropies': []}

        for t in range(self.cfg.usim_steps):
            # 1. 构造时间步 (0, 1, 2...)
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)

            # 2. Agent 决策
            candidates = self.get_candidates(current_h)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            # 3. 执行动作 (选择虚拟用户)
            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            # 4. 梯度上升 (Gradient Ascent)
            # 目标: 让 current_h 更接近 selected_user (Score 变大)
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad  # Note: Plus sign for Ascent

            # 5. 计算 Dense Reward (每一步都计算与真实 ID 的距离)
            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_id_emb is not None:
                # Reward = -MSE (距离越小，奖励越大)
                dist = F.mse_loss(current_h, target_id_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0

                # 如果 Target 是全0 (真正的冷启动物品，无 ID)，Mask 掉 Reward
                is_target_zero = (target_id_emb.abs().sum(dim=1, keepdim=True) < 1e-6)
                reward = reward.masked_fill(is_target_zero, 0.0)

            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['entropies'].append(entropy)

        return current_h, trajectory

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory['rewards']).squeeze(-1)
        values = torch.stack(trajectory['values']).squeeze(-1)
        log_probs = torch.stack(trajectory['log_probs'])
        entropies = torch.stack(trajectory['entropies'])

        # 计算 Returns (GAE 简化版)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.cfg.ppo_gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        advantage = returns - values

        # PPO Clip Loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantage.detach()
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage.detach()

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()

        return actor_loss + self.cfg.ppo_coeffs['value'] * critic_loss + self.cfg.ppo_coeffs['entropy'] * entropy_loss

    def forward_mlp(self, x, weights, is_item=False):
        """双塔 MLP 通用入口"""
        # is_item=True -> 取后一半参数; is_item=False -> 取前一半参数
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        num_layers = len(self.cfg.hidden_dims)
        for i in range(num_layers):
            w, b = weights[idx_start + 2 * i], weights[idx_start + 2 * i + 1]
            out = F.linear(out, w, b)
            if i < num_layers - 1:
                out = F.relu(out)
        return out, None

    def inner_loop(self, u, i, llm_s, item_emb_cache):
        """MAML Inner Loop: Theta -> Theta'"""
        e_u = self.user_emb(u)
        e_i = item_emb_cache  # 这里使用的是 Support Set 的 h_T

        # 前向传播 (使用原始 vars)
        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(e_i, self.vars, True)

        # 计算 Support Set Loss
        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))

        # 计算梯度
        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)

        # 更新参数得到 Fast Weights (Omega)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop, llm_s):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold

        # 1. 初始状态 (Content + LLM) - 强制无 ID
        con_raw = self.item_con_emb(i)
        con_base = self.con_proj(con_raw)

        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm

        init_state = con_base + llm_e

        # 2. 准备 RL Reward Target (仅热门物品有 ID)
        target_id = torch.zeros_like(init_state)
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_id[hot_mask] = self.item_beh_emb(i[hot_mask]).detach()

        # 3. 运行 RL -> 得到最终 h_T
        final_h, trajectory = self.run_usim_episode(init_state, target_id)
        ppo_loss = self.compute_ppo_loss(trajectory)

        # 4. Meta-Learning (Ranking Loss)
        total_ranking_loss = 0
        task_splits = {}
        if is_cold.sum() >= 4: task_splits['cold'] = is_cold
        if (~is_cold).sum() >= 4: task_splits['hot'] = ~is_cold

        for name, mask in task_splits.items():
            u_mask, i_mask, l_mask = u[mask], i[mask], llm_s[mask]
            h_mask = final_h[mask]

            # Support / Query Split
            split = len(u_mask) // 2
            su, si, sl = u_mask[:split], i_mask[:split], l_mask[:split]
            h_support = h_mask[:split]

            qu, qi = u_mask[split:], i_mask[split:]
            h_query = h_mask[split:]

            # Inner Loop (Support)
            omega = self.inner_loop(su, si, sl, h_support)

            # Prediction (Query) using Omega
            e_u_q = self.user_emb(qu)
            z_u, _ = self.forward_mlp(e_u_q, omega, False)
            z_i, _ = self.forward_mlp(h_query, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_ranking_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        total_loss = total_ranking_loss + ppo_loss
        return total_loss, None


# ================= 4. 全量排名 & 评估逻辑 =================

class SimpleItemDataset(Dataset):
    """辅助 Dataset，用于快速遍历所有物品"""

    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


def precompute_all_items(model, num_items, batch_size=1024, device='cuda'):
    """
    预计算全量物品向量 (Base Pool)。
    对于库中的所有物品，假设 LLM Score = -1 (Unknown)，计算基础向量。
    """
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ [Evaluation] Pre-computing Full Item Pool (Background)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            # 仅使用 Content，无 LLM 注入
            con_raw = model.item_con_emb(i_batch)
            init_state = model.con_proj(con_raw)

            # 运行 RL
            final_h, _ = model.run_usim_episode(init_state, None)
            # 运行 MLP (使用全局 vars)
            z_i, _ = model.forward_mlp(final_h, model.vars, True)

            all_z_i.append(z_i.cpu())  # 存到 CPU

    return torch.cat(all_z_i, dim=0)


def evaluate_full(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    """
    全量排名评估函数 (Asymmetric)
    正样本: 使用真实 LLM Score 计算高精度向量
    负样本: 使用预计算的 Base 向量
    """
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    # 尝试将库放入 GPU 加速
    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU Memory Full, fallback to CPU for ranking.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    print("🚀 Running Full Ranking Eval...")

    with torch.no_grad():
        for batch, pop, llm_s in loader:
            # 筛选冷启动用户进行评估
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            l_s = llm_s[mask].to(device)

            batch_size = u.size(0)

            # 1. 计算用户向量 z_u
            e_u = model.user_emb(u)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)

            # 2. 正样本特异化 (Hero Item)
            # 重新计算正样本，这次带上真实的 LLM Score
            con_raw = model.item_con_emb(i_target)
            con_base = model.con_proj(con_raw)

            mask_llm = (l_s > -0.5).float().unsqueeze(1)
            val_llm = torch.clamp(l_s, min=0.0).unsqueeze(1)
            llm_e = model.llm_proj(val_llm) * mask_llm
            init_state = con_base + llm_e

            final_h, _ = model.run_usim_episode(init_state, None)
            z_i_pos, _ = model.forward_mlp(final_h, model.vars, True)

            # 3. 替换与排名
            if cpu_mode: z_u = z_u.cpu()

            # (A) 计算与全库(Base)的分数 [B, N_items]
            scores = torch.matmul(z_u, all_item_emb_gpu.t())

            # (B) 计算与特异化正样本的分数 [B]
            pos_scores = (z_u * (z_i_pos.cpu() if cpu_mode else z_i_pos)).sum(dim=1)

            # (C) 替换: 把 i_target 位置的分数换成精确分
            rows = torch.arange(batch_size, device=scores.device)
            target_cols = i_target.cpu() if cpu_mode else i_target
            scores[rows, target_cols] = pos_scores

            # 4. Top-K 指标计算
            max_k = max(k_list)
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)

            target_cols = target_cols.view(-1, 1)

            for k in k_list:
                preds = topk_indices[:, :k]
                hits = (preds == target_cols).any(dim=1).float()
                metrics_sum[f'R@{k}'] += hits.sum().item()

                hit_ranks = torch.where(preds == target_cols)
                if hit_ranks[1].numel() > 0:
                    # NDCG = 1 / log2(rank + 2)
                    dcg = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                    metrics_sum[f'N@{k}'] += dcg.sum().item()

            total_samples += batch_size
            if cpu_mode: z_u = z_u.to(device)

    if total_samples == 0: return None, 0
    return {k: v / total_samples for k, v in metrics_sum.items()}, total_samples


# ================= 5. 数据流处理与主程序 =================

class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)
        if llm_map:
            # 实际部署时建议优化此处的循环
            vals = [llm_map.get((u, i), -1.0) for u, i in zip(df['u_idx'], df['i_idx'])]
            self.llm_s = torch.tensor(vals, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm_s': self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm_s = torch.stack([item['llm_s'] for item in batch])
    return {'u': u, 'i': i}, pop, llm_s


def main():
    setup_seed(20)
    print(">>> 1. Loading Data...")

    # ⚠️ 数据路径检查
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("❌ Error: 'processed_data/stream_data.pkl' not found.")
        return

    df = pd.read_pickle("processed_data/stream_data.pkl")
    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    content_emb = torch.load("processed_data/content_emb.pt")

    llm_map = {}
    if os.path.exists("processed_data/llm_scores.pkl"):
        with open("processed_data/llm_scores.pkl", "rb") as f: llm_map = pickle.load(f)

    # 按月切分数据
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['dt'] = df['timestamp']
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=512, collate_fn=collate_fn, shuffle=True)
               for p in periods if len(p) > 0]

    # 初始化
    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_RL_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # 统计指标定义
    target_k = [5, 10, 20]
    metric_names = [f'{m}@{k}' for k in target_k for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metric_names}
    global_count = 0

    print(f"\n>>> Start Training (Metrics: {metric_names}) <<<")
    WARMUP = 2

    for t, loader in enumerate(loaders):
        print(f"\n--- Period {t} (Samples: {len(loader.dataset)}) ---")

        # === 评估阶段 ===
        if t >= WARMUP:
            # 1. 预计算背景库
            all_item_z = precompute_all_items(model, cfg.num_items, device=device)
            # 2. 全量评估
            met, n = evaluate_full(model, loader, all_item_z, device, k_list=target_k)

            if met:
                # 格式化输出
                res_str = " | ".join([f"{k}={met[k]:.4f}" for k in metric_names])
                print(f"📊 Eval: {res_str}")
                for k in metric_names:
                    global_accum[k] += met[k] * n
                global_count += n

        # === 训练阶段 ===
        model.train()
        total_loss = 0
        for batch_idx, (batch, pop, llm_s) in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, pop.to(device), llm_s.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"   Avg Loss: {total_loss / len(loader):.4f}")

    # === 最终报告 ===
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT (Micro Average)")
    print("-" * 60)
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 60)
    if global_count > 0:
        for k in metric_names:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
