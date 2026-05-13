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
        self.hidden_dims = [128, 64]

        # --- 阈值与权重 ---
        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5

        # --- 优化器 ---
        self.inner_lr = 0.001
        self.outer_lr = 0.0005
        self.temp = 0.1

        # --- RL (User Simulation) ---
        self.usim_steps = 9  # 想象步数
        self.n_candidates = 15  # [核心] Top-K 候选池大小
        self.usim_lr = 0.8  # [建议] 配合 Top-K 时步长调小，防止过拟合

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

        # Actor: 输出查询向量
        self.actor_head = nn.Linear(128, 128)
        # Critic: 预测价值
        self.critic_head = nn.Linear(128, 1)
        # User Projection
        self.user_proj = nn.Linear(item_dim, 128)

    def get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
        # 1. State
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=10)[:, :4].float()
        state = torch.cat([item_state, t_emb], dim=1)

        feat = self.common(state)
        value = self.critic_head(feat)

        # 2. Actor Policy (Attention)
        query = self.actor_head(feat).unsqueeze(1)  # [B, 1, 128]
        keys = self.user_proj(candidates_emb)  # [B, N_cand, 128]
        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)  # [B, N_cand]

        dist = Categorical(logits=logits)

        # 3. Sample
        if action_idx is None:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        return action_idx, log_prob, value, entropy


# ================= 3. 主模型: PAM_RL_USIM (Top-K Ver) =================

class PAM_RL_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === Embeddings ===
        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
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

        # [新增] 检索投影层: 将 Item State 映射到 User 空间以进行 Top-K 检索
        self.retrieval_proj = nn.Linear(config.behavior_dim, config.user_dim)

        # === RL Agent ===
        self.agent = SimpleAC(config.behavior_dim, time_dim=4)

        # === MAML Meta-Parameters ===
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        dims = [config.user_dim] + config.hidden_dims
        for _ in range(2):  # User Tower + Item Tower
            for i in range(len(dims) - 1):
                w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
                nn.init.xavier_normal_(w)
                b = nn.Parameter(torch.zeros(dims[i + 1]))
                self.vars.extend([w, b])
                self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                                  nn.Parameter(torch.ones_like(b) * config.inner_lr)])

    def get_candidates(self, item_emb):
        """
        [修改] Top-K 候选池检索
        不是随机采样，而是计算当前 item_emb 与所有用户的相似度，取 Top-K
        """
        # 1. 将 Item 映射到 User 空间
        item_query = self.retrieval_proj(item_emb)  # [B, User_Dim]

        # 2. 获取所有用户的 Embedding (detach 防止更新用户)
        all_users = self.user_emb.weight.detach()  # [N_Users, User_Dim]

        # 3. 计算相似度 (Dot Product)
        # [B, User_Dim] @ [User_Dim, N_Users] -> [B, N_Users]
        scores = torch.matmul(item_query, all_users.t())

        # 4. 取 Top-K
        # indices: [B, K]
        _, topk_indices = torch.topk(scores, k=self.cfg.n_candidates, dim=1)

        # 5. 获取对应的 User Embeddings
        # [B, K, User_Dim]
        candidates = self.user_emb(topk_indices).detach()

        return candidates

    def run_usim_episode(self, init_item_emb, target_id_emb=None):
        """RL 想象循环"""
        # [关键] 保留原始状态用于 Residual Connection
        original_state = init_item_emb.clone()
        current_h = init_item_emb.clone()

        trajectory = {'log_probs': [], 'values': [], 'rewards': [], 'entropies': []}

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)

            # 1. [修改] 使用 Top-K 检索获取候选人
            candidates = self.get_candidates(current_h)

            # 2. Agent 从 Top-K 中精选
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            # 3. 执行动作
            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            # 4. 梯度上升
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            # 5. Dense Reward
            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_id_emb is not None:
                dist = F.mse_loss(current_h, target_id_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0
                is_target_zero = (target_id_emb.abs().sum(dim=1, keepdim=True) < 1e-6)
                reward = reward.masked_fill(is_target_zero, 0.0)

            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['entropies'].append(entropy)

        # [关键] Residual Connection: 保证不比初始状态差
        final_h = current_h + original_state

        return final_h, trajectory

    # ... (compute_ppo_loss, forward_mlp, inner_loop 保持不变) ...

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory['rewards']).squeeze(-1)
        values = torch.stack(trajectory['values']).squeeze(-1)
        log_probs = torch.stack(trajectory['log_probs'])
        entropies = torch.stack(trajectory['entropies'])

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.cfg.ppo_gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        advantage = returns - values
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantage.detach()
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage.detach()

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()

        return actor_loss + self.cfg.ppo_coeffs['value'] * critic_loss + self.cfg.ppo_coeffs['entropy'] * entropy_loss

    def forward_mlp(self, x, weights, is_item=False):
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
        e_u = self.user_emb(u)
        e_i = item_emb_cache
        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(e_i, self.vars, True)
        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))
        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop, llm_s):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold

        con_raw = self.item_con_emb(i)
        con_base = self.con_proj(con_raw)

        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm

        init_state = con_base + llm_e

        target_id = torch.zeros_like(init_state)
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_id[hot_mask] = self.item_beh_emb(i[hot_mask]).detach()

        final_h, trajectory = self.run_usim_episode(init_state, target_id)
        ppo_loss = self.compute_ppo_loss(trajectory)

        total_ranking_loss = 0
        task_splits = {}
        if is_cold.sum() >= 4: task_splits['cold'] = is_cold
        if (~is_cold).sum() >= 4: task_splits['hot'] = ~is_cold

        for name, mask in task_splits.items():
            u_mask, i_mask, l_mask = u[mask], i[mask], llm_s[mask]
            h_mask = final_h[mask]
            split = len(u_mask) // 2
            su, si, sl = u_mask[:split], i_mask[:split], l_mask[:split]
            h_support = h_mask[:split]
            qu, qi = u_mask[split:], i_mask[split:]
            h_query = h_mask[split:]

            omega = self.inner_loop(su, si, sl, h_support)

            e_u_q = self.user_emb(qu)
            z_u, _ = self.forward_mlp(e_u_q, omega, False)
            z_i, _ = self.forward_mlp(h_query, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_ranking_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        total_loss = total_ranking_loss + ppo_loss
        return total_loss, None


# ================= 4. 全量排名 & 评估逻辑 =================

class SimpleItemDataset(Dataset):
    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


def precompute_all_items(model, num_items, batch_size=1024, device='cuda'):
    model.eval()
    item_loader = DataLoader(SimpleItemDataset(num_items), batch_size=batch_size, shuffle=False)
    all_z_i = []

    print("⏳ Pre-computing Full Item Pool (Top-K RL)...")
    with torch.no_grad():
        for i_batch in item_loader:
            i_batch = i_batch.to(device)
            con_raw = model.item_con_emb(i_batch)
            init_state = model.con_proj(con_raw)

            # 使用 Top-K 检索的 RL 过程
            final_h, _ = model.run_usim_episode(init_state, None)
            z_i, _ = model.forward_mlp(final_h, model.vars, True)

            all_z_i.append(z_i.cpu())

    return torch.cat(all_z_i, dim=0)


def evaluate_full(model, loader, all_item_z, device, k_list=[5, 10, 20]):
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    try:
        all_item_emb_gpu = all_item_z.to(device)
        cpu_mode = False
    except RuntimeError:
        print("⚠️ GPU Memory Full, fallback to CPU.")
        all_item_emb_gpu = all_item_z
        cpu_mode = True

    with torch.no_grad():
        for batch, pop, llm_s in loader:
            mask = pop < model.cfg.cold_threshold
            if mask.sum() < 1: continue

            u = batch['u'][mask].to(device)
            i_target = batch['i'][mask].to(device)
            l_s = llm_s[mask].to(device)

            batch_size = u.size(0)

            e_u = model.user_emb(u)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)

            con_raw = model.item_con_emb(i_target)
            con_base = model.con_proj(con_raw)
            mask_llm = (l_s > -0.5).float().unsqueeze(1)
            val_llm = torch.clamp(l_s, min=0.0).unsqueeze(1)
            llm_e = model.llm_proj(val_llm) * mask_llm
            init_state = con_base + llm_e

            final_h, _ = model.run_usim_episode(init_state, None)
            z_i_pos, _ = model.forward_mlp(final_h, model.vars, True)

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


# ================= 5. 主程序 =================

class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)
        if llm_map:
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
    setup_seed(2025)
    print(">>> 1. Loading Data...")

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

    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['dt'] = df['timestamp']
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=512, collate_fn=collate_fn, shuffle=True)
               for p in periods if len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>> Initializing PAM_RL_USIM (Top-K Candidates)...")
    model = PAM_RL_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    target_k = [5, 10, 20]
    metric_names = [f'{m}@{k}' for k in target_k for m in ['R', 'N']]
    global_accum = {name: 0.0 for name in metric_names}
    global_count = 0

    print(f"\n>>> Start Training (Top-K Retrieval Mode) <<<")
    WARMUP = 2

    for t, loader in enumerate(loaders):
        print(f"\n--- Period {t} (Samples: {len(loader.dataset)}) ---")

        if t >= WARMUP:
            all_item_z = precompute_all_items(model, cfg.num_items, device=device)
            met, n = evaluate_full(model, loader, all_item_z, device, k_list=target_k)

            if met:
                res_str = " | ".join([f"{k}={met[k]:.4f}" for k in metric_names])
                print(f"📊 Eval: {res_str}")
                for k in metric_names:
                    global_accum[k] += met[k] * n
                global_count += n

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

    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT (PAM_RL Top-K)")
    print("-" * 60)
    if global_count > 0:
        for k in metric_names:
            print(f"{k:<10} | {global_accum[k] / global_count:.4f}")
    else:
        print("No evaluation performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()