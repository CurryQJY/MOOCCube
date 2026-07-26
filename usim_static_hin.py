import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import json
import os
import random
import matplotlib

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
    print(f"Random Seed Fixed: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items

        self.emb_dim = 128            # 统一维度
        self.hidden_dim = 256
        self.content_dim = content_dim

        self.cold_threshold = 5

        # 优化参数
        self.lr = 0.0005              # 统一学习率
        self.temp = 0.07              # 对齐 PAM Enhanced
        self.margin = 0.15            # Additive Margin

        # Dropout 与 辅助损失
        self.dropout_prob = 0.35      # ID Dropout
        self.aux_weight = 0.3         # 辅助对比损失权重

        # RL Hyperparams
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {'value': 0.5, 'entropy': 0.01}

        self.usim_steps = 5           # USIM 步数
        self.n_candidates = 20
        self.usim_lr = 0.3

        self.n_epochs = 3
        # 【去 MAML 后显存大解放】 
        self.batch_size = 2048        # 真实的大 Batch Size!
        self.accum_steps = 1          # 不再需要累积


# ================= 2. PPO Agent =================

class SimpleAC(nn.Module):
    def __init__(self, item_dim, time_dim=4):
        super(SimpleAC, self).__init__()
        input_dim = item_dim + time_dim
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        self.actor_head = nn.Linear(256, 128)
        self.critic_head = nn.Linear(256, 1)
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


# ================= 3. Pure RL-USIM + PAM Enhanced 主模型 =================

class PAM_RL_Pure_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 统一维度 Embeddings
        self.user_emb = nn.Embedding(config.n_users, config.emb_dim)
        self.item_id_emb = nn.Embedding(config.n_items, config.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # 2. 3层 GELU 内容编码器
        self.content_proj = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim)
        )

        # 3. 用户投影层 (对齐 PAM Enhanced)
        self.user_proj = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim)
        )

        # 3. 门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid()
        )

        # 5. RL Agent
        self.agent = SimpleAC(config.emb_dim, time_dim=4)

    def get_candidates(self, item_emb):
        B = item_emb.size(0)
        N_cand = self.cfg.n_candidates
        rand_idx = torch.randint(0, self.cfg.n_users, (B, N_cand), device=self.device)
        cand_emb = self.user_proj(self.user_emb(rand_idx)).detach()  # 用投影后的向量供 PPO attention
        return cand_emb

    def get_item_vector(self, i_idx, force_cold=False):
        id_e = self.item_id_emb(i_idx)
        
        # ID Dropout
        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)
            
        content_e = self.content_proj(self.item_con_emb(i_idx))
        
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        
        item_fused = alpha * id_e + (1 - alpha) * content_e
        return item_fused, id_e, content_e

    def run_usim_episode(self, init_item_emb, target_emb=None):
        current_h = init_item_emb.clone()
        trajectory = {
            'log_probs': [], 'values': [], 'rewards': [], 'entropies': [],
            'states': [], 'time_steps': [], 'candidates': [], 'actions': []
        }

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates = self.get_candidates(current_h)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            trajectory['states'].append(current_h.detach().clone())
            trajectory['time_steps'].append(time_step.detach().clone())
            trajectory['candidates'].append(candidates.detach().clone())
            trajectory['actions'].append(action_idx.detach().clone())

            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]

            # 模拟用户点击产生的表征偏移
            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            # Reward 信号
            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_emb is not None:
                dist = F.mse_loss(current_h, target_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0

            trajectory['log_probs'].append(log_prob.detach())
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['entropies'].append(entropy)

        return current_h, trajectory

    def compute_ppo_loss(self, trajectory):
        rewards = torch.stack(trajectory['rewards']).squeeze(-1)
        old_log_probs = torch.stack(trajectory['log_probs'])
        states = trajectory['states']
        time_steps = trajectory['time_steps']
        candidates = trajectory['candidates']
        actions = trajectory['actions']

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.cfg.ppo_gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        total_ppo_loss = 0

        for _ in range(self.cfg.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            new_entropies_list = []

            for t in range(len(states)):
                _, new_log_prob, new_value, new_entropy = self.agent.get_action_value(
                    states[t], time_steps[t], candidates[t], action_idx=actions[t]
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                new_entropies_list.append(new_entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze(-1)
            new_entropies = torch.stack(new_entropies_list)

            advantage = (returns - new_values).detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - new_values).pow(2).mean()
            entropy_loss = -new_entropies.mean()

            total_ppo_loss += actor_loss + self.cfg.ppo_coeffs['value'] * critic_loss + \
                              self.cfg.ppo_coeffs['entropy'] * entropy_loss

        return total_ppo_loss / self.cfg.ppo_epochs

    def forward(self, batch, pop):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold

        # 1. 基础表征 (PAM 优势 + LLM 增益)
        z_u_base = self.user_proj(self.user_emb(u))
        z_i_base, id_e_raw, content_e = self.get_item_vector(i, force_cold=False)

        # 2. RL USIM 序列模拟
        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

        # 对于 Cold 物品，我们利用 PPO 在训练中生成"想象中"的交互轨迹
        final_h, trajectory = self.run_usim_episode(z_i_base, target_emb)
        ppo_loss = self.compute_ppo_loss(trajectory)

        # 3. 排名损失计算 ———【去 MAML，大 Batch 直接 InfoNCE!】———
        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)  # 使用强化学习优化后的最终表征进入对比

        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0)).to(self.device)

        # Additive Margin for Hard Negatives
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp
        
        main_loss = F.cross_entropy(logits_margin, labels)

        # 4. 辅助损失 (Content <-> ID)
        z_id = F.normalize(id_e_raw, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        total_loss = main_loss + self.cfg.aux_weight * aux_loss + ppo_loss
        return total_loss, None


# ================= 4. 评估工具 =================

def compute_ranking_metrics(scores, target_indices, k_list=[5, 10, 20]):
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)
    actual_k = min(max(k_list), num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        recall = hits.mean().item()
        
        # 统一和对齐基线的安全 NDCG
        rks = (preds == targets).nonzero(as_tuple=True)
        dcg_tensor = torch.zeros(batch_size, device=scores.device)
        if rks[0].numel() > 0:
            dcg_tensor[rks[0]] = 1.0 / torch.log2(rks[1].float() + 2.0)
        ndcg = dcg_tensor.mean()
        results[f'R@{k}'] = recall
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results

def split_dataframe_by_periods(df, period_type='M'):
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = df['timestamp']
    df['period_id'] = df['datetime'].dt.to_period(period_type)
    periods = []
    for p_key in sorted(df['period_id'].unique()):
        periods.append(df[df['period_id'] == p_key].reset_index(drop=True))
    return periods


# ================= 5. 评估函数 =================

def evaluate_usim(model, loader, device, k_list=[5, 10, 20], n_neg=999,
                  eval_type='cold', full_ranking=False):
    model.eval()
    accum_metrics = {}
    total_samples = 0

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        # llm_scores is not used in get_item_vector anymore, but kept for compatibility with the main loop's call signature
        # all_llm_s = torch.tensor([llm_scores.get(int(idx), -1.0) for idx in all_item_idx], dtype=torch.float, device=device)

        ITEM_BATCH = 1024
        all_item_vecs = []
        for start in range(0, n_items, ITEM_BATCH):
            end = min(start + ITEM_BATCH, n_items)
            idx_batch = all_item_idx[start:end]
            z_i, _, _ = model.get_item_vector(idx_batch, force_cold=True) 
            all_item_vecs.append(F.normalize(z_i, dim=1))

        all_item_vecs = torch.cat(all_item_vecs, dim=0)

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

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)

            z_u = F.normalize(model.user_proj(model.user_emb(u)), dim=1)

            if full_ranking:
                scores = torch.mm(z_u, all_item_vecs.t())
                target_indices = i
            else:
                neg_items = torch.randint(0, n_items, (n_sel, n_neg), device=device)
                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                cand_vecs = all_item_vecs[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)

            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0

    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


# ================= 6. 数据集 =================

class StreamDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx]}

def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    return {'u': u, 'i': i}, pop




# ================= 7. 主训练循环 =================

def main():
    DATA_DIR = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Pure RL-USIM (No LLM) from {DATA_DIR}...")
    if not os.path.exists(f"{DATA_DIR}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{DATA_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{DATA_DIR}/stream_data.pkl")
    content_emb = torch.load(f"{DATA_DIR}/content_emb.pt")

    # [NEW] Static random split 8:1:1
    df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*0.8)]
    val_df = df.iloc[int(n*0.8):int(n*0.9)]
    test_df = df.iloc[int(n*0.9):]

    train_loader = DataLoader(StreamDataset(train_df), batch_size=2048, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(StreamDataset(val_df), batch_size=2048, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(StreamDataset(test_df), batch_size=2048, shuffle=False, collate_fn=collate_fn)

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PAM_RL_Pure_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    epochs = 40
    print(f">> 架构: Pure RL-USIM + InfoNCE (Batch Size=2048) | Static Evaluation (8:1:1)")
    
    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        steps = 0
        for batch_idx, (batch, pop) in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, _ = model(batch, pop.to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {total_loss / steps:.4f}")

        if epoch % 5 == 0 or epoch == epochs:
            fmet_hot, fn_h = evaluate_usim(model, val_loader, device, k_list=k_list, eval_type='hot', full_ranking=True)
            h_f_str = " | ".join([f"{k}={fmet_hot[k]:.4f}" for k in metrics_keys[:3]]) if fmet_hot else "N/A"
            print(f"  --> Valid Hot Full: {h_f_str}")

    print("\n" + "=" * 90)
    print("         FINAL TEST REPORT: Sampled (1+999) vs Full Ranking (STATIC 8:1:1)")
    print("=" * 90)
    
    met_cold, n_cold_t = evaluate_usim(model, test_loader, device, k_list=k_list, eval_type='cold')
    met_hot, n_hot_t = evaluate_usim(model, test_loader, device, k_list=k_list, eval_type='hot')
    fmet_cold, fn_c = evaluate_usim(model, test_loader, device, k_list=k_list, eval_type='cold', full_ranking=True)
    fmet_hot, fn_h = evaluate_usim(model, test_loader, device, k_list=k_list, eval_type='hot', full_ranking=True)

    print(f"{'Metric':<10} | {'Samp Cold':<12} | {'Samp Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for m in metrics_keys:
        sc = met_cold.get(m, 0.0) if met_cold else 0.0
        sh = met_hot.get(m, 0.0) if met_hot else 0.0
        fc = fmet_cold.get(m, 0.0) if fmet_cold else 0.0
        fh = fmet_hot.get(m, 0.0) if fmet_hot else 0.0
        print(f"{m:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
    print("=" * 90)


if __name__ == "__main__":
    setup_seed(2025)
    main()
