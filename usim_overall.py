import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import json
import os
import pickle, random
import matplotlib

# 强制使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ================= 1. 基础设置 =================
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
        self.num_users = n_users
        self.num_items = n_items

        self.user_dim = 64
        self.behavior_dim = 64
        self.content_dim = content_dim
        self.hidden_dims = [128, 64]

        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5

        # 优化参数
        self.inner_lr = 0.001
        self.outer_lr = 0.0005
        self.temp = 0.1

        # === RL 配置 ===
        self.dropout_prob = 0.5
        self.gamma_llm = 0.5

        # RL Hyperparams
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5  # [新增] PPO 多轮更新次数
        self.ppo_coeffs = {'value': 0.5, 'entropy': 0.01}

        self.usim_steps = 9  # 想象步数
        self.n_candidates = 20  # 动作空间大小
        self.usim_lr = 0.8  # 想象更新步长


# ================= 2. PPO Agent =================

class SimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=4):
        super(SimpleAC, self).__init__()
        input_dim = item_dim + time_dim
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(128, 128)
        self.critic_head = nn.Linear(128, 1)
        self.user_proj = nn.Linear(item_dim, 128)

    def get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=10)[:, :4].float()
        state = torch.cat([item_state, t_emb], dim=1)

        feat = self.common(state)
        value = self.critic_head(feat)

        query = self.actor_head(feat).unsqueeze(1)
        keys = self.user_proj(candidates_emb)

        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        dist = Categorical(logits=logits)

        if action_idx is None:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        return action_idx, log_prob, value, entropy


# ================= 3. PAM + RL-USIM 主模型 =================

class PAM_RL_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embeddings
        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # Projections
        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.behavior_dim),
            nn.LayerNorm(config.behavior_dim)
        )
        self.llm_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, config.behavior_dim)
        )

        # Agent
        self.agent = SimpleAC(config.behavior_dim, time_dim=4)

        # MAML Params
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()
        dims_u = [config.user_dim] + config.hidden_dims
        dims_i = [config.behavior_dim] + config.hidden_dims
        for dims in [dims_u, dims_i]:
            for i in range(len(dims) - 1):
                w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
                nn.init.xavier_normal_(w)
                b = nn.Parameter(torch.zeros(dims[i + 1]))
                self.vars.extend([w, b])
                self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                                  nn.Parameter(torch.ones_like(b) * config.inner_lr)])

    def get_candidates(self, item_emb):
        """
        Top-K 候选用户检索（确定性）
        相同的 item_emb 总是返回相同的候选用户
        """
        # 1. 计算物品与所有用户的相似度
        all_users = self.user_emb.weight.detach()  # [N_users, user_dim]
        scores = torch.matmul(item_emb, all_users.t())  # [B, N_users]

        # 2. 取 Top-K 相似用户
        _, topk_idx = torch.topk(scores, k=self.cfg.n_candidates, dim=1)  # [B, K]

        # 3. 获取用户嵌入
        cand_emb = self.user_emb(topk_idx).detach()  # [B, K, user_dim]

        return cand_emb

    def run_usim_episode(self, init_item_emb, target_id_emb=None):
        current_h = init_item_emb.clone()
        trajectory = {
            'log_probs': [], 'values': [], 'rewards': [], 'entropies': [],
            'states': [], 'time_steps': [], 'candidates': [], 'actions': []  # [新增] 存储用于 PPO 重放
        }

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates = self.get_candidates(current_h)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            # [新增] 存储状态用于 PPO 重放
            trajectory['states'].append(current_h.detach().clone())
            trajectory['time_steps'].append(time_step.detach().clone())
            trajectory['candidates'].append(candidates.detach().clone())
            trajectory['actions'].append(action_idx.detach().clone())

            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            # [关键修复] 显式开启梯度计算
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_id_emb is not None:
                dist = F.mse_loss(current_h, target_id_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0
                is_target_zero = (target_id_emb.abs().sum(dim=1, keepdim=True) < 1e-6)
                reward = reward.masked_fill(is_target_zero, 0.0)

            trajectory['log_probs'].append(log_prob.detach())  # [修改] detach 作为 old_log_probs
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['entropies'].append(entropy)

        return current_h, trajectory

    def compute_ppo_loss(self, trajectory):
        """
        [完整 PPO] 多轮更新，使用存储的 states/candidates/actions 重新计算 log_probs
        """
        rewards = torch.stack(trajectory['rewards']).squeeze(-1)
        old_log_probs = torch.stack(trajectory['log_probs'])  # 旧策略的 log_probs
        states = trajectory['states']  # List of [B, D]
        time_steps = trajectory['time_steps']  # List of [B, 1]
        candidates = trajectory['candidates']  # List of [B, N_cand, D]
        actions = trajectory['actions']  # List of [B]

        # 计算 returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.cfg.ppo_gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        total_ppo_loss = 0

        # [关键] 多轮 PPO 更新
        for _ in range(self.cfg.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            new_entropies_list = []

            # 重新前向传播，获取新策略的 log_probs
            for t in range(len(states)):
                state_t = states[t]
                time_t = time_steps[t]
                cand_t = candidates[t]
                action_t = actions[t]

                # 使用相同的 action_idx 重新计算 log_prob
                _, new_log_prob, new_value, new_entropy = self.agent.get_action_value(
                    state_t, time_t, cand_t, action_idx=action_t
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                new_entropies_list.append(new_entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze(-1)
            new_entropies = torch.stack(new_entropies_list)

            # PPO Clip
            advantage = (returns - new_values).detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - new_values).pow(2).mean()
            entropy_loss = -new_entropies.mean()

            total_ppo_loss += actor_loss + self.cfg.ppo_coeffs['value'] * critic_loss + self.cfg.ppo_coeffs[
                'entropy'] * entropy_loss

        return total_ppo_loss / self.cfg.ppo_epochs

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

        # 1. State
        con_raw = self.item_con_emb(i)
        con_base = self.con_proj(con_raw)
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        init_state = con_base + llm_e

        # 2. Target (Only Hot)
        target_id = torch.zeros_like(init_state)
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_id[hot_mask] = self.item_beh_emb(i[hot_mask]).detach()

        # 3. RL USIM
        final_h, trajectory = self.run_usim_episode(init_state, target_id)
        ppo_loss = self.compute_ppo_loss(trajectory)

        # 4. Meta-Learning
        total_ranking_loss = 0
        task_splits = {}
        if is_cold.sum() >= 2: task_splits['cold'] = is_cold
        if (~is_cold).sum() >= 2: task_splits['hot'] = ~is_cold

        for name, mask in task_splits.items():
            u_mask, i_mask, l_mask = u[mask], i[mask], llm_s[mask]
            h_mask = final_h[mask]

            split = len(u_mask) // 2
            if split < 1: continue

            su, si, sl = u_mask[:split], i_mask[:split], l_mask[:split]
            qu, qi = u_mask[split:], i_mask[split:]

            h_support = h_mask[:split]
            h_query = h_mask[split:]

            omega = self.inner_loop(su, si, sl, h_support)

            e_u_q = self.user_emb(qu)
            z_u, _ = self.forward_mlp(e_u_q, omega, False)
            z_i, _ = self.forward_mlp(h_query, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_ranking_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss

        total_loss = total_ranking_loss + ppo_loss
        return total_loss, None


# ================= 4. 评估与主流程 =================

def precompute_all_items(model, num_items, device, batch_size=1024):
    """
    预计算所有物品的表示 (用于全量排名)
    """
    model.eval()
    all_z = []

    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            item_ids = torch.arange(start, end, device=device)

            # Content embedding
            con_raw = model.item_con_emb(item_ids)
            con_base = model.con_proj(con_raw)

            # 无 LLM score，直接用 content
            init_state = con_base

            # USIM 增强
            final_h, _ = model.run_usim_episode(init_state, None)

            # MLP 投影
            z_i, _ = model.forward_mlp(final_h, model.vars, True)
            all_z.append(z_i)

    return torch.cat(all_z, dim=0)  # [N_items, dim]


def evaluate_batch(model, loader, device, k_list=[5, 10, 20], eval_type='cold'):
    """
    Batch 内排名评估
    eval_type: 'cold' | 'hot' | 'all'
    """
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_samples = 0

    with torch.no_grad():
        for batch, pop, llm_s in loader:
            # 根据 eval_type 筛选样本
            if eval_type == 'cold':
                mask = pop < model.cfg.cold_threshold
            elif eval_type == 'hot':
                mask = pop >= model.cfg.cold_threshold
            else:  # 'all'
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_batch = mask.sum().item()
            if n_batch < 2: continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)
            l_s = llm_s[mask].to(device)

            # 用户表示
            e_u = model.user_emb(u)

            # 物品表示 (实时计算 USIM)
            con_raw = model.item_con_emb(i)
            con_base = model.con_proj(con_raw)
            mask_llm = (l_s > -0.5).float().unsqueeze(1)
            val_llm = torch.clamp(l_s, min=0.0).unsqueeze(1)
            llm_e = model.llm_proj(val_llm) * mask_llm
            init_state = con_base + llm_e

            final_h, _ = model.run_usim_episode(init_state, None)

            z_u, _ = model.forward_mlp(e_u, model.vars, False)
            z_i, _ = model.forward_mlp(final_h, model.vars, True)

            # Batch 内排名
            scores = torch.mm(z_u, z_i.t())  # [B, B]

            batch_size = scores.size(0)
            targets = torch.arange(batch_size).to(device).view(-1, 1)
            actual_k = min(max(k_list), batch_size)
            _, topk = torch.topk(scores, actual_k, dim=1)

            for k in k_list:
                preds = topk[:, :k]
                hits = (preds == targets).any(dim=1).float()
                metrics_sum[f'R@{k}'] += hits.sum().item()

                hit_ranks = torch.where(preds == targets)
                if hit_ranks[1].numel() > 0:
                    dcg = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                    metrics_sum[f'N@{k}'] += dcg.sum().item()

            total_samples += n_batch

    if total_samples == 0: return None, 0
    return {k: v / total_samples for k, v in metrics_sum.items()}, total_samples


class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)
        if llm_map:
            vals = []
            for u, i in zip(df['u_idx'], df['i_idx']):
                vals.append(llm_map.get((u, i), -1.0))
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
    print("1. 加载数据 (RL-USIM Fixed Version)...")
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("Error: 找不到数据文件")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
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

    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=512, collate_fn=collate_fn) for p in periods if
               len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PAM_RL_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    target_metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}

    # 三类评估的累加器
    accum_cold = {k: 0.0 for k in target_metrics}
    accum_hot = {k: 0.0 for k in target_metrics}
    accum_all = {k: 0.0 for k in target_metrics}
    count_cold, count_hot, count_all = 0, 0, 0

    print(f"\n>>> 开始 RL 训练 (Steps={cfg.usim_steps}) <<<")
    WARMUP = 3

    for t, loader in enumerate(loaders):
        if t >= WARMUP:
            # [Batch 内排名] 评估三类
            met_cold, n_cold = evaluate_batch(model, loader, device, eval_type='cold')
            met_hot, n_hot = evaluate_batch(model, loader, device, eval_type='hot')
            met_all, n_all = evaluate_batch(model, loader, device, eval_type='all')

            if met_cold:
                for k in target_metrics:
                    accum_cold[k] += met_cold[k] * n_cold
                count_cold += n_cold

            if met_hot:
                for k in target_metrics:
                    accum_hot[k] += met_hot[k] * n_hot
                count_hot += n_hot

            if met_all:
                for k in target_metrics:
                    accum_all[k] += met_all[k] * n_all
                    history[k].append(met_all[k])
                count_all += n_all
                print(
                    f"Period {t:<3} | Cold(n={n_cold:<4}) R@10={met_cold['R@10'] if met_cold else 0:.4f} | Hot(n={n_hot:<4}) R@10={met_hot['R@10'] if met_hot else 0:.4f} | All(n={n_all:<4}) R@10={met_all['R@10']:.4f}")
            else:
                for k in target_metrics: history[k].append(0)
        else:
            for k in target_metrics: history[k].append(0)

        model.train()
        for batch_idx, (batch, pop, llm_s) in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch, pop.to(device), llm_s.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"   Step {batch_idx}: Loss={loss.item():.4f}")

    # --- Final Report ---
    print("\n" + "=" * 80)
    print("🏆 FINAL RESULT (RL-USIM)")
    print("=" * 80)
    print(f"{'Metric':<10} | {'Cold':<12} | {'Hot':<12} | {'All':<12}")
    print("-" * 80)

    for m in target_metrics:
        cold_val = accum_cold[m] / count_cold if count_cold > 0 else 0.0
        hot_val = accum_hot[m] / count_hot if count_hot > 0 else 0.0
        all_val = accum_all[m] / count_all if count_all > 0 else 0.0
        print(f"{m:<10} | {cold_val:<12.4f} | {hot_val:<12.4f} | {all_val:<12.4f}")

    print("=" * 80)
    print(f"Sample Count: Cold={count_cold}, Hot={count_hot}, All={count_all}")

    pd.DataFrame(history).to_csv('metrics_rl_usim.csv')
    print("Done.")


if __name__ == "__main__":
    setup_seed(2025)
    main()
