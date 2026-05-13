import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import pickle, random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def setup_seed(seed=2024):
    """
    一键固定所有随机种子，确保实验可复现
    """
    # 1. Python 原生随机数
    random.seed(seed)
    # 2. Numpy 随机数 (用于 pandas 和 np 操作)
    np.random.seed(seed)
    # 3. PyTorch CPU 随机数
    torch.manual_seed(seed)
    # 4. PyTorch GPU 随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 5. 强制 CuDNN 使用确定性算法 (会慢一点，但结果唯一)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. 设置环境变量 (防止 Hash 随机化)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ 随机种子已固定: {seed}")


# ================= 0. 配置类 (增加 USIM 参数) =================
class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.num_users = n_users
        self.num_items = n_items

        # 维度配置
        self.user_dim = 64
        self.content_dim = content_dim
        self.behavior_dim = 64
        self.hidden_dims = [128, 64]  # MLP 结构

        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5

        # Loss Weights
        self.gamma_s = 5.0  # Syllabus Enhancer
        self.gamma_llm = 0.5  # LLM Distillation
        self.gamma_usim = 1.0  # [新增] USIM 对齐权重

        self.inner_lr = 0.001
        self.outer_lr = 0.001
        self.temp = 0.1


# ================= 1. 模型定义 (集成 USIM) =================

class USIM_Module(nn.Module):
    """
    [USIM 核心组件] 想象模块
    模拟论文逻辑: h_{t+1} = h_t + Delta
    输入: Content Projection
    输出: Imagination Delta (特征偏移量)
    """

    def __init__(self, input_dim):
        super(USIM_Module, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Tanh()  # 限制幅度，防止初始阶段各种漂移
        )
        # 【关键】强制初始化最后一层为 0
        nn.init.zeros_(self.net[-2].weight)  # -2 是最后一个 Linear
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        return self.net(x)


class PAM_LLM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config

        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        # 冻结原始 BERT，符合 USIM/Adapter 思路
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # Content Projector (Adapter)
        # 如果维度不一致则投影，一致则保留 Identity 或也加一层非线性
        if config.content_dim != config.behavior_dim:
            self.con_proj = nn.Sequential(
                nn.Linear(config.content_dim, 256),
                nn.ReLU(),
                nn.Linear(256, config.behavior_dim)
            )
        else:
            self.con_proj = nn.Identity()

        # [新增] USIM 模块
        self.usim = USIM_Module(config.behavior_dim)

        # 新增一个门控层
        self.usim_gate = nn.Sequential(
            nn.Linear(config.behavior_dim, 1),
            nn.Sigmoid()
        )

        # Meta-Parameters
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        # 输入维度: User部分=user_dim, Item部分=behavior_dim * 2 (因为是 cat)
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

        # Syllabus Enhancer (监督信号映射层)
        self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-2]))
        self.sup_b = nn.Parameter(torch.zeros(config.behavior_dim))

    def get_item_features(self, i_idx, return_refined=False):
        """
        获取物品特征 (USIM Enhanced)
        """
        beh = self.item_beh_emb(i_idx)

        # 1. 基础内容特征
        con_raw = self.item_con_emb(i_idx)
        con_base = self.con_proj(con_raw)

        # 2. [USIM] 想象交互
        # 即使是冷物品，也加上这个 Delta，模拟"如果它被交互了会怎样"
        delta = self.usim(con_base)
        # 2. 计算置信度 (Gate)
        # 输入可以是 con_base，也可以是 delta
        alpha = self.usim_gate(con_base)

        con_refined = con_base + alpha * delta

        # 3. 拼接 (保留原代码结构)
        # 使用 Refined Content 替代原始 Content
        features = torch.cat([beh, con_refined], dim=1)

        if return_refined:
            return features, con_refined
        return features

    def forward_mlp(self, x, weights, is_item=False):
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        prev_out = x
        num_layers = len(self.cfg.hidden_dims)

        for i in range(num_layers):
            w, b = weights[idx_start + 2 * i], weights[idx_start + 2 * i + 1]
            if i == num_layers - 1:
                prev_out = out
            out = F.linear(out, w, b)
            if i < num_layers - 1:
                out = F.relu(out)
        return out, prev_out

    def inner_loop(self, u, i):
        e_u = self.user_emb(u)
        e_i = self.get_item_features(i)
        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(e_i, self.vars, True)

        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        loss = F.cross_entropy(logits, torch.arange(len(u)).to(u.device))

        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)
        return [w - a * g if g is not None else w for w, g, a in zip(self.vars, grads, self.lslr)]

    def forward(self, batch, pop, llm_scores, usim_weight=1.0):
        u, i = batch['u'], batch['i']
        is_cold = pop < self.cfg.cold_threshold
        total_loss = 0
        loss_dict = {}

        # 1. Meta-Learning
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
            z_u, _ = self.forward_mlp(e_u, omega, False)
            z_i, _ = self.forward_mlp(e_i, omega, True)

            loss = F.cross_entropy(torch.mm(z_u, z_i.t()) / self.cfg.temp, torch.arange(len(qu)).to(qu.device))
            total_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss
            loss_dict[name] = loss.item()

        # 2. Auxiliary Tasks (Hot Items)
        if (~is_cold).sum() > 0:
            hi_idx = i[~is_cold]
            # 获取 Refined Content 用于对齐
            hi_feat, hi_con_refined = self.get_item_features(hi_idx, return_refined=True)
            target_beh = self.item_beh_emb(hi_idx).detach()

            # A. Syllabus Enhancer (原代码保留)
            _, feat = self.forward_mlp(hi_feat, self.vars, True)
            loss_sup = F.mse_loss(F.linear(feat, self.sup_w, self.sup_b), target_beh)
            total_loss += self.cfg.gamma_s * loss_sup
            loss_dict['sup'] = loss_sup.item()

            # B. [新增] USIM Alignment Loss
            # 强制 "想象优化后" 的 Content 逼近真实 Behavior
            # loss_usim = F.mse_loss(hi_con_refined, target_beh)

            # 在训练循环里动态调整 gamma

            loss_usim = F.mse_loss(hi_con_refined, target_beh.detach())
            # total_loss += self.cfg.gamma_usim * loss_usim * usim_weight
            total_loss += self.cfg.gamma_usim * loss_usim
            loss_dict['usim'] = loss_usim.item()

        # 3. LLM Distillation
        mask_llm = llm_scores > -0.5
        if mask_llm.sum() > 0:
            u_l, i_l = u[mask_llm], i[mask_llm]
            t_score = llm_scores[mask_llm]
            z_u, _ = self.forward_mlp(self.user_emb(u_l), self.vars, False)
            z_i, _ = self.forward_mlp(self.get_item_features(i_l), self.vars, True)
            s_score = torch.sigmoid((z_u * z_i).sum(dim=1))
            # loss_llm = F.mse_loss(s_score, t_score)

            # 新代码：根据热度(pop)动态调整权重
            # pop 越小，weight 越大；pop > 0 时，weight 迅速衰减
            # 这里的 1.0 / (pop + 1.0) 意味着：
            # pop=0 (纯冷) -> weight=1.0
            # pop=4 (微冷) -> weight=0.2

            dynamic_weight = 1.0 / (pop[mask_llm] + 1.0)

            # 手动计算加权 MSE
            loss_llm = (dynamic_weight * (s_score - t_score) ** 2).mean()

            total_loss += self.cfg.gamma_llm * loss_llm
            loss_dict['llm'] = loss_llm.item()

        return total_loss, loss_dict


# ================= 2. 数据集定义 =================
class StreamDataset(Dataset):
    def __init__(self, df, llm_map=None):
        self.u = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.i = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.pop = torch.tensor(df['popularity'].values, dtype=torch.long)
        self.llm_s = torch.full((len(df),), -1.0, dtype=torch.float32)

        if llm_map:
            keys = list(zip(df['u_idx'], df['i_idx']))
            vals = [llm_map.get(k, -1.0) for k in keys]
            self.llm_s = torch.tensor(vals, dtype=torch.float32)

    def __len__(self): return len(self.u)

    def __getitem__(self, idx):
        return {'u': self.u[idx], 'i': self.i[idx], 'pop': self.pop[idx], 'llm_s': self.llm_s[idx]}


def collate_fn(batch):
    u = torch.stack([item['u'] for item in batch])
    i = torch.stack([item['i'] for item in batch])
    pop = torch.stack([item['pop'] for item in batch])
    llm_s = torch.stack([item['llm_s'] for item in batch])
    return {'u': u, 'i': i}, pop, llm_s


# ================= 3. 评估函数 (包含 R@5,10,20 和 N@5,10,20) =================
def compute_ranking_metrics(scores, k_list=[5, 10, 20]):
    """ 计算 Batch 内的平均指标 """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = torch.arange(batch_size).to(scores.device).view(-1, 1)

    max_k = max(k_list)
    actual_k = min(max_k, num_candidates)
    _, topk_indices = torch.topk(scores, actual_k, dim=1)

    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        results[f'R@{k}'] = hits.mean().item()

        hit_ranks = torch.where(preds == targets)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            dcg = 1.0 / torch.log2(ranks + 2.0)
            ndcg = dcg.sum() / batch_size
        else:
            ndcg = 0.0
        results[f'N@{k}'] = ndcg.item() if isinstance(ndcg, torch.Tensor) else ndcg
    return results


def evaluate(model, loader, device, k_list=[5, 10, 20]):
    model.eval()
    metrics_sum = {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
    total_cold_samples = 0

    with torch.no_grad():
        for batch, pop, _ in loader:
            mask = pop < model.cfg.cold_threshold
            n_batch_cold = mask.sum().item()
            if n_batch_cold < 2: continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)

            e_u = model.user_emb(u)
            # USIM 增强后的特征
            e_i = model.get_item_features(i)

            z_u, _ = model.forward_mlp(e_u, model.vars, False)
            z_i, _ = model.forward_mlp(e_i, model.vars, True)

            scores = torch.mm(z_u, z_i.t())

            # 计算 R@5,10,20 和 N@5,10,20
            res = compute_ranking_metrics(scores, k_list=k_list)

            for k, v in res.items():
                metrics_sum[k] += v * n_batch_cold
            total_cold_samples += n_batch_cold

    if total_cold_samples == 0: return None, 0
    period_metrics = {k: v / total_cold_samples for k, v in metrics_sum.items()}
    return period_metrics, total_cold_samples


# ================= 4. 主流程 =================
def main():
    print("1. 加载数据...")
    # 请确保路径正确，如果不一致请自行修改
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("Error: 请先运行 data_process.py")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")

    llm_map = None
    if os.path.exists("processed_data/llm_scores.pkl"):
        print("   加载本地 LLM 分数...")
        with open("processed_data/llm_scores.pkl", "rb") as f: llm_map = pickle.load(f)

    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df['pid'] = df['dt'].dt.to_period('M')
    periods = [df[df['pid'] == p].reset_index(drop=True) for p in sorted(df['pid'].dropna().unique())]

    loaders = [DataLoader(StreamDataset(p, llm_map), batch_size=2048, collate_fn=collate_fn) for p in periods if
               len(p) > 0]

    cfg = Config(meta['n_users'], meta['n_items'], content_emb.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM_LLM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # 评估指标设置
    target_metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    history = {k: [] for k in target_metrics}
    global_accum = {k: 0.0 for k in target_metrics}
    global_cold_count = 0

    print(f"\n>>> 开始评估 (USIM Enhanced) <<<")
    WARMUP = 3

    for t, loader in enumerate(loaders):
        # --- Eval ---
        if t >= WARMUP:
            met, n_samples = evaluate(model, loader, device, k_list=[5, 10, 20])
            if met:
                # 打印 R@10 和 N@10 作为参考
                print(f"Period {t:<3} (n={n_samples:<4}): R@10={met['R@10']:.4f} | N@10={met['N@10']:.4f}")
                for k in target_metrics:
                    history[k].append(met[k])
                    global_accum[k] += met[k] * n_samples
                global_cold_count += n_samples
            else:
                for k in target_metrics: history[k].append(0)
        else:
            for k in target_metrics: history[k].append(0)

        # --- Train ---
        model.train()
        total_loss = 0
        steps = 0
        # === 动态权重策略 ===
        # 前 3 个周期 (0, 1, 2) 不加 USIM Loss，让 Content_Proj 先飞一会儿
        # 第 3 个周期开始，权重设为 1.0 (或 0.1，如果您觉得 1.0 太大)
        current_usim_weight = 0.0 if t < 3 else 1.0

        for batch, pop, llm_s in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, d = model(batch, pop.to(device), llm_s.to(device), usim_weight=current_usim_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    # --- 结果汇总 ---
    print("\n" + "=" * 60)
    print("🏆 最终评估报告 (Final Evaluation Report)")
    print("=" * 60)
    print(f"{'Metric':<10} | {'Macro Avg':<18} | {'Micro Avg':<18}")
    print("-" * 60)

    for m in target_metrics:
        valid_vals = [v for i, v in enumerate(history[m]) if i >= WARMUP and v > 0]
        macro_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        micro_val = global_accum[m] / global_cold_count if global_cold_count > 0 else 0.0
        print(f"{m:<10} | {macro_val:.4f}             | {micro_val:.4f}")

    # 保存结果
    pd.DataFrame(history).to_csv('final_metrics.csv')
    print("\n完成！")


if __name__ == "__main__":
    setup_seed(20)
    main()
