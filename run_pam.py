import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# 1. 配置参数 (Configuration)
# ============================================================================
class Config:
    DATA_PATH = "processed_data/stream_data.pkl"
    META_PATH = "processed_data/meta.json"
    BERT_PATH = "processed_data/content_emb.pt"
    LLM_PATH = "processed_data/llm_scores.pkl"

    BATCH_SIZE = 2048  # 大 Batch 适合双塔
    LR = 0.001
    EPOCHS = 100  # 设置较大，依赖早停
    PATIENCE = 8  # 耐心值调大一点，等待 NDCG 收敛

    EMB_DIM = 64
    HIDDEN_DIM = 128
    COLD_THRES = 5

    # 蒸馏权重 (调整此参数控制 LLM 的影响力)
    GAMMA_LLM = 0.5

    K_LIST = [5, 10, 20]


# ============================================================================
# 2. PAM + BERT 模型
# ============================================================================
class PAM_LLM(nn.Module):
    def __init__(self, n_users, n_items, pretrained_emb=None, emb_dim=64, hidden_dim=128):
        super(PAM_LLM, self).__init__()

        # --- User Tower ---
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        # --- Item Tower ---
        self.item_id_emb = nn.Embedding(n_items + 1, emb_dim)

        # BERT Embedding
        bert_dim = 768
        if pretrained_emb is not None:
            bert_dim = pretrained_emb.shape[1]
            self.bert_content = nn.Embedding(n_items + 1, bert_dim, padding_idx=0)
            # 复制权重并冻结
            valid_rows = min(pretrained_emb.shape[0], n_items + 1)
            self.bert_content.weight.data[:valid_rows] = pretrained_emb[:valid_rows]
            self.bert_content.weight.requires_grad = False
            print(f"✅ Model: BERT Embedding loaded & frozen (Dim: {bert_dim})")
        else:
            self.bert_content = nn.Embedding(n_items + 1, bert_dim)
            print("⚠️ Model: BERT Randomized (Not Recommended)")

        # Adapter
        self.content_adapter = nn.Sequential(
            nn.Linear(bert_dim, emb_dim),
            nn.Tanh()
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Embedding):
                if m != self.bert_content:
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u_e = self.user_emb(user_ids)
        z_u = self.user_mlp(u_e)

        i_id_e = self.item_id_emb(item_ids)
        bert_v = self.bert_content(item_ids)
        i_con_e = self.content_adapter(bert_v)

        i_fused = i_id_e + i_con_e
        z_i = self.item_mlp(i_fused)

        return z_u, z_i

    def predict_all(self, user_ids, all_item_ids):
        """ 全量推理用于评估 """
        u_e = self.user_emb(user_ids)
        z_u = self.user_mlp(u_e)

        i_id_e = self.item_id_emb(all_item_ids)
        bert_v = self.bert_content(all_item_ids)
        i_con_e = self.content_adapter(bert_v)
        i_fused = i_id_e + i_con_e
        z_i = self.item_mlp(i_fused)

        # Dot Product [B, N_items]
        return torch.matmul(z_u, z_i.t())


# ============================================================================
# 3. 数据集 (支持 LLM Score)
# ============================================================================
class PAMDataset(Dataset):
    def __init__(self, pairs, n_items, llm_scores=None):
        self.pairs = pairs
        self.n_items = n_items
        self.llm_scores = llm_scores

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]

        # 简单负采样
        j = np.random.randint(1, self.n_items + 1)
        while j == i: j = np.random.randint(1, self.n_items + 1)

        # 获取 LLM 分数 (如果不存在则为 -1)
        score_pos = -1.0

        if self.llm_scores:
            # 尝试获取 (u, i) 的分数，如果是 Mock 数据或真实数据
            score_pos = self.llm_scores.get((u, i), -1.0)

        return torch.tensor(u), torch.tensor(i), torch.tensor(j), torch.tensor(score_pos, dtype=torch.float)


def load_data_and_flatten(data_path):
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_pickle(data_path)

    # 清洗列名
    if 'u_idx' in df.columns and 'user_id' in df.columns: df = df.drop(columns=['user_id'])
    if 'i_idx' in df.columns and 'item_id' in df.columns: df = df.drop(columns=['item_id'])
    rename = {}
    if 'i_idx' in df.columns: rename['i_idx'] = 'item_id'
    if 'u_idx' in df.columns: rename['u_idx'] = 'user_id'
    if 'raw_time' in df.columns and 'timestamp' not in df.columns: rename['raw_time'] = 'timestamp'
    if rename: df = df.rename(columns=rename)

    if not pd.api.types.is_integer_dtype(df['item_id']): df['item_id'] = df['item_id'].astype(int)

    n_items = df['item_id'].max()
    if df['item_id'].min() == 0:
        df['item_id'] = df['item_id'] + 1
        n_items += 1
    n_items = max(n_items, df['item_id'].max())
    n_users = df['user_id'].max() + 1

    df = df.sort_values(['user_id', 'timestamp'])
    user_groups = df.groupby('user_id')['item_id'].apply(list)

    train_pairs = []
    test_data = []
    for user_id, items in user_groups.items():
        if len(items) < 3: continue
        test_data.append({'user_id': user_id, 'history': items[:-1], 'target': items[-1], 'hist_len': len(items) - 1})
        for item in items[:-1]:
            train_pairs.append([user_id, item])

    print(f"✅ Data Ready! Train Pairs: {len(train_pairs)}")
    return train_pairs, test_data, n_users, n_items


def mock_llm_scores(train_pairs):
    """ 生成模拟 LLM 分数以跑通代码 """
    print("⚠️ 未找到真实 LLM 分数，正在生成模拟数据 (Mocking)...")
    scores = {}
    # 随机采样 30% 的训练数据给予高分指导
    sample_indices = np.random.choice(len(train_pairs), int(len(train_pairs) * 0.3), replace=False)
    for idx in tqdm(sample_indices, desc="Mocking LLM"):
        u, i = train_pairs[idx]
        # 模拟 LLM 认为这是好课 (0.8 ~ 0.99)
        scores[(u, i)] = np.random.uniform(0.80, 0.99)
    return scores


# ============================================================================
# 4. 全量评估 (返回详细字典)
# ============================================================================
def evaluate_pam_advanced(model, test_data, n_items, device, k_list=[5, 10, 20], cold_threshold=5):
    model.eval()

    metrics = {g: {'cnt': 0} for g in ['All', 'Cold', 'Warm']}
    for g in metrics:
        for k in k_list:
            metrics[g][f'R@{k}'] = 0.0
            metrics[g][f'N@{k}'] = 0.0

    all_candidates = torch.arange(1, n_items + 1).to(device)
    max_k = max(k_list)

    with torch.no_grad():
        test_loader = tqdm(test_data, desc="🔍 Eval", leave=False, ncols=80)
        for u_data in test_loader:
            user_id = u_data['user_id']
            history = u_data['history']
            target = u_data['target']

            # 全量打分
            u_tensor = torch.tensor([user_id]).to(device)
            scores = model.predict_all(u_tensor, all_candidates).squeeze()

            # Mask History
            for visited in history:
                if 1 <= visited <= n_items: scores[visited - 1] = -float('inf')

            # Top-K
            _, topk = torch.topk(scores, max_k)
            target_idx = target - 1

            hit_group = 'Cold' if u_data['hist_len'] <= cold_threshold else 'Warm'

            if target_idx in topk:
                rank = (topk == target_idx).nonzero(as_tuple=True)[0].item()
                for k in k_list:
                    if rank < k:
                        hit = 1.0
                        ndcg = 1.0 / np.log2(rank + 2)
                        for g in [hit_group, 'All']:
                            metrics[g][f'R@{k}'] += hit
                            metrics[g][f'N@{k}'] += ndcg

            metrics[hit_group]['cnt'] += 1
            metrics['All']['cnt'] += 1

    # 打印报表
    print(f"\n{'=' * 25} Evaluation Report {'=' * 25}")
    headers = ["Group", "Users"] + [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    header_fmt = "{:<6} | {:<6} | " + " | ".join(["{:<7}"] * len(k_list) * 2)

    print(header_fmt.format(*headers))
    print("-" * 100)

    results = {}  # 存储返回结果

    for g in ['All', 'Cold', 'Warm']:
        cnt = metrics[g]['cnt']
        if cnt == 0: continue

        # 计算平均值
        vals = []
        for k in k_list: vals.append(metrics[g][f'R@{k}'] / cnt)
        for k in k_list: vals.append(metrics[g][f'N@{k}'] / cnt)

        row = [g, str(cnt)] + [f"{v:.4f}" for v in vals]
        print(header_fmt.format(*row))

        if g == 'All':
            for i, k in enumerate(k_list):
                results[f'R@{k}'] = vals[i]
                results[f'N@{k}'] = vals[len(k_list) + i]

    print("=" * 100 + "\n")
    return results  # 返回字典 {'R@10': 0.25, 'N@10': 0.15 ...}


# ============================================================================
# 5. 主程序 (综合早停)
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    # 1. 加载数据
    train_pairs, test_data, n_users, n_items = load_data_and_flatten(Config.DATA_PATH)

    # 2. 加载 BERT
    pretrained_emb = None
    if os.path.exists(Config.BERT_PATH):
        print(f"📥 Loading BERT: {Config.BERT_PATH}")
        pretrained_emb = torch.load(Config.BERT_PATH, map_location='cpu')
    else:
        print("⚠️ Warning: BERT file not found.")

    # 3. 加载 LLM 分数
    llm_scores = None
    if os.path.exists(Config.LLM_PATH):
        print(f"📥 Loading LLM Scores: {Config.LLM_PATH}")
        with open(Config.LLM_PATH, "rb") as f:
            llm_scores = pickle.load(f)
    else:
        llm_scores = mock_llm_scores(train_pairs)

    # 4. 初始化
    model = PAM_LLM(n_users, n_items, pretrained_emb=pretrained_emb,
                    emb_dim=Config.EMB_DIM, hidden_dim=Config.HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    dataset = PAMDataset(train_pairs, n_items, llm_scores=llm_scores)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 5. 训练循环
    print(f"🚀 Start Training with Distillation (Gamma={Config.GAMMA_LLM})...")
    print(f"🛑 Early Stopping Metric: Recall@10 + NDCG@10")

    best_score = 0.0
    patience_cnt = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0
        mse_loss_acc = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)

        for u, i, j, s_pos in pbar:
            u, i, j = u.to(device), i.to(device), j.to(device)
            s_pos = s_pos.to(device)

            z_u, z_i = model(u, i)
            _, z_j = model(u, j)

            # A. BPR Loss (Collaborative Filtering)
            score_pos = (z_u * z_i).sum(1)
            score_neg = (z_u * z_j).sum(1)
            loss_bpr = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()

            # B. Distillation Loss (Semantic Guidance)
            loss_distill = 0.0
            mask = s_pos > 0  # 只在有 LLM 分数的地方计算
            if mask.sum() > 0:
                # 学生输出经过 Sigmoid 映射到 0-1
                pred = torch.sigmoid(score_pos[mask])
                target = s_pos[mask]
                loss_distill = nn.MSELoss()(pred, target)
                mse_loss_acc += loss_distill.item()

            loss = loss_bpr + Config.GAMMA_LLM * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 评估
        res = evaluate_pam_advanced(model, test_data, n_items, device, k_list=Config.K_LIST,
                                    cold_threshold=Config.COLD_THRES)

        # === 核心修改：综合指标早停 ===
        current_score = res['R@10'] + res['N@10']

        if current_score > best_score:
            best_score = current_score
            patience_cnt = 0
            torch.save(model.state_dict(), "pam_final_best.pth")
            print(f"🌟 New Best! Score: {best_score:.4f} (R@10: {res['R@10']:.4f}, N@10: {res['N@10']:.4f})")
        else:
            patience_cnt += 1
            print(f"⏳ Patience: {patience_cnt}/{Config.PATIENCE} (Best Score: {best_score:.4f})")

            if patience_cnt >= Config.PATIENCE:
                print(f"🛑 Early Stopping triggered at Epoch {epoch}!")
                break