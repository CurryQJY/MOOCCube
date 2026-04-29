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
import pickle
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
    print(f"✅ 随机种子已固定: {seed}")


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items
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

        # RL 配置
        self.dropout_prob = 0.5
        self.gamma_llm = 0.5

        # RL Hyperparams
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {'value': 0.5, 'entropy': 0.01}

        # [优化3] 减少步数 + 降低学习率
        self.usim_steps = 5         # 9 → 5
        self.n_candidates = 20
        self.usim_lr = 0.3           # 0.8 → 0.3

        # 累积训练
        self.n_epochs = 3

        # [优化5] 梯度累积
        self.accum_steps = 4         # 有效 batch = 512 × 4 = 2048

        # [核心优化] 直接 InfoNCE 损失权重
        self.direct_weight = 1.0

        # [领域专属配置]
        self.n_concepts = 25201 # 从 concept_meta.json 得到
        self.concept_dim = 32
        self.prereq_penalty_weight = 1.0


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


# ================= 3. PAM + RL-USIM 主模型 (原生优化版) =================

class PAM_RL_USIM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embeddings (保持原始维度 64, 不动)
        self.user_emb = nn.Embedding(config.n_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.n_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # [优化2] 3层 GELU 编码器 (不加门控/ID dropout, 保持 USIM 初始状态稳定)
        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim + config.concept_dim, 256), # Include concept_dim
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.behavior_dim),
            nn.LayerNorm(config.behavior_dim)
        )

        self.llm_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, config.behavior_dim)
        )

        # [领域专属特征抽取]
        # Multi-hot concept encoders (using an Embedding layer to project multihot vector)
        self.concept_emb = nn.Embedding(config.n_concepts + 1, config.concept_dim, padding_idx=0)
        self.concept_proj_u = nn.Linear(config.concept_dim, config.user_dim)
        
        # [优化1] 冷启动内容相似度: 预计算全库内容嵌入用于 reward
        # (在 forward 中动态使用 con_proj 的输出)

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
        B = item_emb.size(0)
        N_cand = self.cfg.n_candidates
        rand_idx = torch.randint(0, self.cfg.n_users, (B, N_cand), device=self.device)
        cand_emb = self.user_emb(rand_idx).detach()
        # Not including dynamic concepts here yet for simplicity in candidate generation
        return cand_emb, rand_idx

    def run_usim_episode(self, init_item_emb, target_emb=None, item_req_concepts=None, u_mastery_padded=None):
        """
        target_emb: 对 hot 物品是 item_beh_emb, 对 cold 物品是内容嵌入 (优化1)
        item_req_concepts: [B, num_concepts] multi-hot vector for required concepts
        """
        current_h = init_item_emb.clone()
        trajectory = {
            'log_probs': [], 'values': [], 'rewards': [], 'entropies': [],
            'states': [], 'time_steps': [], 'candidates': [], 'actions': []
        }

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, cand_idx = self.get_candidates(current_h)
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            trajectory['states'].append(current_h.detach().clone())
            trajectory['time_steps'].append(time_step.detach().clone())
            trajectory['candidates'].append(candidates.detach().clone())
            trajectory['actions'].append(action_idx.detach().clone())

            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]
            selected_user_idx = cand_idx[batch_indices, action_idx]

            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            # [优化1] 所有物品都有 reward 信号 (hot → beh_emb, cold → content_emb)
            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            if target_emb is not None:
                dist = F.mse_loss(current_h, target_emb, reduction='none').mean(dim=1, keepdim=True)
                reward = -dist * 10.0
                
            # [领域专属] 先决条件惩罚 - 如果模拟用户缺乏课程要求概念，给予负反馈
            if item_req_concepts is not None and u_mastery_padded is not None:
                 sel_user_con = u_mastery_padded[selected_user_idx] # [B, max_u_concepts]
                 sel_user_mastery = torch.zeros((selected_user_idx.size(0), self.cfg.n_concepts + 1), device=self.device)
                 sel_user_mastery.scatter_(1, sel_user_con, 1.0)
                 
                 # Unmet requirements = Requirement AND (NOT Mastery)
                 unmet = torch.relu(item_req_concepts - sel_user_mastery) 
                 # sum over concepts to find total unmet requirements
                 penalty = unmet.sum(dim=1, keepdim=True) 
                 # apply penalty to reward (scale it appropriately)
                 reward = reward - self.cfg.prereq_penalty_weight * penalty

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

    # [优化4] FOMAML: create_graph=False
    def inner_loop(self, u, i, llm_s, item_emb_cache):
        e_u = self.user_emb(u)
        e_i = item_emb_cache
        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(e_i, self.vars, True)
        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))
        # [优化4] 一阶近似: create_graph=False, 不做二阶反传
        grads = torch.autograd.grad(loss, self.vars, create_graph=False, allow_unused=True)
        return [w - a * g.detach() if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def get_user_rep(self, u, u_mastery):
        e_u = self.user_emb(u)
        # Pool multi-hot embeddings for mastery
        # e_u_concept = self.concept_proj_u(self.concept_emb(u_mastery).mean(dim=1)) 
        # e_u_final = e_u + e_u_concept # Simplest combination
        # For now, just return e_u, as we test prerequisite penalty mainly
        return e_u

    def forward(self, batch, pop, llm_s, u_mastery_padded=None):
        u, i = batch['u'], batch['i']
        i_con_multihot = batch['i_con'] # [B, L]
        i_req_multihot = batch['i_req'] # [B, num_concepts] multi-hot vector directly
        is_cold = pop < self.cfg.cold_threshold

        # 1. State: 内容编码
        con_raw = self.item_con_emb(i)
        
        # [领域专属] Embed current item concepts into a pool and concat to raw bert
        mask = (i_con_multihot != 0).float().unsqueeze(-1)
        # Embedding ignores 0 padding
        c_emb = (self.concept_emb(i_con_multihot) * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        con_combined = torch.cat([con_raw, c_emb], dim=1)
        con_base = self.con_proj(con_combined)
        
        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        init_state = con_base + llm_e

        # 2. [优化1] Target: hot → beh_emb, cold → content_emb (所有物品都有引导)
        target_emb = con_base.detach().clone()  # 默认: 冷启动用内容嵌入作为 target
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_beh_emb(i[hot_mask]).detach()

        # 3. RL USIM
        final_h, trajectory = self.run_usim_episode(
            init_state, 
            target_emb,
            item_req_concepts=i_req_multihot,
            u_mastery_padded=u_mastery_padded
        )
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

        # 5. [核心优化] 直接 InfoNCE 损失 (不经过 MAML, 给 con_proj 和 user_emb 直接梯度)
        z_u_direct = F.normalize(self.user_emb(u), dim=1)
        z_i_direct = F.normalize(con_base, dim=1)
        direct_logits = torch.mm(z_u_direct, z_i_direct.t()) / self.cfg.temp
        direct_labels = torch.arange(len(u)).to(u.device)
        direct_loss = F.cross_entropy(direct_logits, direct_labels)

        total_loss = total_ranking_loss + ppo_loss + self.cfg.direct_weight * direct_loss
        return total_loss, None


# ================= 4. 评估工具 (对齐 train_eval_hin.py) =================

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
        all_llm_s = torch.full((n_items,), -1.0, device=device)

        ITEM_BATCH = 512
        all_item_vecs = []
        for start in range(0, n_items, ITEM_BATCH):
            end = min(start + ITEM_BATCH, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]

            # [修复] 推理时直接用训练好的 con_proj → MLP，不跑 USIM episode
            con_raw = model.item_con_emb(idx_batch)
            con_base = model.con_proj(con_raw)
            z_i_batch, _ = model.forward_mlp(con_base, model.vars, True)
            all_item_vecs.append(z_i_batch)

        all_item_vecs = torch.cat(all_item_vecs, dim=0)

        for batch, pop, llm_s in loader:
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

            e_u = model.user_emb(u)
            z_u, _ = model.forward_mlp(e_u, model.vars, False)

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
    def __init__(self, df, llm_map=None, u_mastery_map=None, i_concepts_map=None, i_req_concepts_map=None, max_concepts=50):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)
        if llm_map:
            vals = []
            for u, i in zip(df['u_idx'], df['i_idx']):
                vals.append(llm_map.get((u, i), -1.0))
            self.llm_s = torch.tensor(vals, dtype=torch.float32)
            
        self.max_concepts = max_concepts
        self.i_con_map = i_concepts_map or {}
        self.i_req_map = i_req_concepts_map or {}
        
        # We process multihot for requirements statically for speed
        n_concepts = 25201 # HARDCODED for MOOCCube Domain
        
        self.i_req_multihots = []
        self.i_con_padded = []
        
        for i_idx in df['i_idx'].values:
            # Padded concept sequence for embedding pooling
            cons = self.i_con_map.get(i_idx, [])
            if len(cons) > self.max_concepts:
                cons = cons[:self.max_concepts]
            padded_cons = cons + [0] * (self.max_concepts - len(cons))
            self.i_con_padded.append(padded_cons)
            
            # Multi-hot vector for requirements
            reqs = self.i_req_map.get(i_idx, [])
            m_hot = torch.zeros(n_concepts + 1, dtype=torch.float32)
            if reqs:
                m_hot[reqs] = 1.0
            self.i_req_multihots.append(m_hot)
            
        self.i_con_padded = torch.tensor(self.i_con_padded, dtype=torch.long)
        self.i_req_multihots = torch.stack(self.i_req_multihots)


    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            'u': self.u[idx], 
            'i': self.i[idx], 
            'pop': self.pop[idx], 
            'llm_s': self.llm_s[idx],
            'i_con': self.i_con_padded[idx],
            'i_req': self.i_req_multihots[idx]
        }


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm_s = torch.stack([item['llm_s'] for item in batch])
    i_con = torch.stack([item['i_con'] for item in batch])
    i_req = torch.stack([item['i_req'] for item in batch])
    return {'u': u, 'i': i, 'i_con': i_con, 'i_req': i_req}, pop, llm_s


# ================= 7. 主训练循环 =================

def main():
    DATA_DIR = "processed_data_hin"
    print(f"Loading Data for PAM RL-USIM (Native Opt) from {DATA_DIR}...")
    if not os.path.exists(f"{DATA_DIR}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{DATA_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{DATA_DIR}/stream_data.pkl")
    content_emb = torch.load(f"{DATA_DIR}/content_emb.pt")

    llm_map = {}
    llm_path = f"{DATA_DIR}/llm_scores.pkl"
    if os.path.exists(llm_path):
        with open(llm_path, "rb") as f:
            llm_map = pickle.load(f)
        print(f"  已加载 LLM 分数: {len(llm_map)} 条")

    # Load domain multi-hots
    DOM_DIR = "processed_data_domain"
    print(f"Loading Domain Multi-Hots from {DOM_DIR}...")
    with open(f"{DOM_DIR}/user_mastery.pkl", "rb") as f:
        u_mastery_map = pickle.load(f)
    with open(f"{DOM_DIR}/course_concepts.pkl", "rb") as f:
        i_concepts_map = pickle.load(f)
    with open(f"{DOM_DIR}/course_req_concepts.pkl", "rb") as f:
        i_req_concepts_map = pickle.load(f)

    # Pre-build user mastery padded tensor to save memory
    n_users = meta['n_users']
    n_concepts = 25201 # From concept_meta.json
    max_u_concepts = 300
    # pad with n_concepts which corresponds to an ignored index in scatter_
    u_mastery_padded = torch.full((n_users, max_u_concepts), n_concepts, dtype=torch.long)
    for uid, mastery_list in u_mastery_map.items():
        if mastery_list and uid < n_users:
            m_list = mastery_list[:max_u_concepts]
            u_mastery_padded[uid, :len(m_list)] = torch.tensor(m_list, dtype=torch.long)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u_mastery_padded = u_mastery_padded.to(device)

    periods = split_dataframe_by_periods(df, period_type='M')

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])

    model = PAM_RL_USIM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    print(f">> 优化配置: USIM steps={cfg.usim_steps}, lr={cfg.usim_lr}, PrereqPenalty={cfg.prereq_penalty_weight}")
    print(f"\n>>> 开始累积训练评估 - 共 {len(periods)} 个周期 <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f'R@{k}' for k in k_list] + [f'N@{k}' for k in k_list]
    history = {'Period': [], 'Count_cold': [], 'Count_hot': []}
    for prefix in ['cold_', 'hot_']:
        for k in metrics_keys:
            history[prefix + k] = []

    # 采样评估累加器
    accum_cold = {k: 0.0 for k in metrics_keys}
    accum_hot = {k: 0.0 for k in metrics_keys}
    count_cold, count_hot = 0, 0

    # 全库排名累加器
    full_cold = {k: 0.0 for k in metrics_keys}
    full_hot = {k: 0.0 for k in metrics_keys}
    fc_cold, fc_hot = 0, 0

    WARMUP_PERIODS = 3
    accumulated_dfs = []

    for t in range(len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_map, u_mastery_map, i_concepts_map, i_req_concepts_map)
        eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= WARMUP_PERIODS:
            met_cold, n_cold_t = evaluate_usim(model, eval_loader, device, k_list, eval_type='cold')
            met_hot, n_hot_t = evaluate_usim(model, eval_loader, device, k_list, eval_type='hot')
            fmet_cold, fn_c = evaluate_usim(model, eval_loader, device, k_list, eval_type='cold', full_ranking=True)
            fmet_hot, fn_h = evaluate_usim(model, eval_loader, device, k_list, eval_type='hot', full_ranking=True)

            if met_cold:
                cold_res = met_cold
                for k in metrics_keys: accum_cold[k] += met_cold[k] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for k in metrics_keys: accum_hot[k] += met_hot[k] * n_hot_t
                count_hot += n_hot_t
            if fmet_cold:
                for k in metrics_keys: full_cold[k] += fmet_cold[k] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for k in metrics_keys: full_hot[k] += fmet_hot[k] * fn_h
                fc_hot += fn_h

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

        # --- 累积训练 (带梯度累积) ---
        accumulated_dfs.append(p_df)
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_map, u_mastery_map, i_concepts_map, i_req_concepts_map)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_fn)

        model.train()
        for epoch in range(cfg.n_epochs):
            total_loss = 0
            steps = 0
            optimizer.zero_grad()

            for batch_idx, (batch, pop, llm_s) in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = model(batch, pop.to(device), llm_s.to(device), u_mastery_padded)

                # [优化5] 梯度累积
                loss = loss / cfg.accum_steps
                loss.backward()

                if (batch_idx + 1) % cfg.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * cfg.accum_steps
                steps += 1

            # 处理末尾剩余梯度
            if (batch_idx + 1) % cfg.accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if epoch == cfg.n_epochs - 1:
                print(f"  [TRAIN] Epoch {epoch+1}/{cfg.n_epochs} | 累积: {len(combined_df)} | Loss: {total_loss / steps:.4f}")

    # ==============================
    # 最终报告
    # ==============================
    print("\n" + "=" * 90)
    print("         FINAL REPORT: 采样评估 (1+999) vs 全库排名 (PAM RL-USIM DOMAIN)")
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

    pd.DataFrame(history).to_csv('mooc_metrics_usim_domain.csv', index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history['Period'], history['cold_R@10'], marker='o', label='Cold R@10')
    plt.plot(history['Period'], history['hot_R@10'], marker='s', label='Hot R@10')
    plt.axvline(x=WARMUP_PERIODS - 0.5, color='r', linestyle='--', label='Warmup End')
    plt.title('PAM RL-USIM (Domain Aware): Cumulative Training')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('mooc_result_usim_domain.png')
    print(">> Saved mooc_result_usim_domain.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
