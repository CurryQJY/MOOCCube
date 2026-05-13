import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# 1. 配置参数
# ============================================================================
class Config:
    DATA_PATH = "processed_data/stream_data.pkl"
    META_PATH = "processed_data/meta.json"
    BERT_PATH = "processed_data/content_emb.pt"

    BATCH_SIZE = 2048
    LR = 0.001
    EPOCHS = 50  # DropoutNet 收敛较慢，建议多跑几轮
    PATIENCE = 8

    EMB_DIM = 64
    HIDDEN_DIM = 128
    COLD_THRES = 5  # 用于评估区分 Cold User

    # === DropoutNet 核心超参 ===
    DROPOUT_RATE = 0.5  # 训练时丢弃 ID 的概率 (论文推荐 0.5)

    K_LIST = [5, 10, 20]


# ============================================================================
# 2. DropoutNet 模型定义
# ============================================================================
class DropoutNet(nn.Module):
    def __init__(self, n_users, n_items, pretrained_emb=None, emb_dim=64, hidden_dim=128, dropout_prob=0.5):
        super(DropoutNet, self).__init__()
        self.dropout_prob = dropout_prob

        # --- User Tower ---
        # DropoutNet 原文主要针对 Item Cold Start，User 侧保持简单
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        # --- Item Tower ---
        self.item_id_emb = nn.Embedding(n_items + 1, emb_dim)

        # BERT Content
        bert_dim = 768
        if pretrained_emb is not None:
            bert_dim = pretrained_emb.shape[1]
            self.bert_content = nn.Embedding(n_items + 1, bert_dim, padding_idx=0)
            valid_rows = min(pretrained_emb.shape[0], n_items + 1)
            self.bert_content.weight.data[:valid_rows] = pretrained_emb[:valid_rows]
            self.bert_content.weight.requires_grad = False  # 冻结 BERT
            print(f"✅ DropoutNet: BERT Loaded (Dim: {bert_dim})")
        else:
            self.bert_content = nn.Embedding(n_items + 1, bert_dim)

        # Adapter: 把 BERT 降维到和 ID 一样，方便处理
        self.content_adapter = nn.Sequential(
            nn.Linear(bert_dim, emb_dim),
            nn.Tanh()
        )

        # 融合后的 MLP
        # 输入维度是 emb_dim * 2 (ID + Content)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
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
        # 1. User Representation
        u_e = self.user_emb(user_ids)
        z_u = self.user_mlp(u_e)

        # 2. Item Representation (Raw)
        i_id_e = self.item_id_emb(item_ids)  # [B, Dim]
        bert_v = self.bert_content(item_ids)  # [B, 768]
        i_con_e = self.content_adapter(bert_v)  # [B, Dim]

        # === 核心逻辑: Input Dropout ===
        # 仅在训练模式下生效
        if self.training:
            # 生成 Mask: 0 的概率为 dropout_prob
            # 形状 [B, 1] 广播到 Embedding 维度
            mask = (torch.rand(i_id_e.shape[0], 1, device=i_id_e.device) > self.dropout_prob).float()

            # 关键：Mask 掉 ID Embedding
            # 注意：DropoutNet 通常不对 Content 做 dropout，因为它是冷启动的唯一救命稻草
            i_id_e = i_id_e * mask

        # 3. Concatenation & Transform
        # 把 ID 和 Content 拼起来，让 MLP 去学它们的关系
        # 当 ID 被 Mask 成 0 时，MLP 被迫只利用 Content 部分
        i_combined = torch.cat([i_id_e, i_con_e], dim=1)  # [B, Dim*2]
        z_i = self.item_mlp(i_combined)

        return z_u, z_i

    def predict_all(self, user_ids, all_item_ids):
        """ 全量推理 """
        u_e = self.user_emb(user_ids)
        z_u = self.user_mlp(u_e)

        i_id_e = self.item_id_emb(all_item_ids)
        bert_v = self.bert_content(all_item_ids)
        i_con_e = self.content_adapter(bert_v)

        # 推理时：可以手动模拟 Cold Start
        # 如果想测试纯冷启动效果，可以强制把 i_id_e 设为 0
        # 这里为了公平对比 Standard Setting，我们保留 ID
        # i_id_e = torch.zeros_like(i_id_e) # <--- 如果要测试极寒启动，解开这行注释

        i_combined = torch.cat([i_id_e, i_con_e], dim=1)
        z_i = self.item_mlp(i_combined)

        return torch.matmul(z_u, z_i.t())


# ============================================================================
# 3. 数据加载 (复用之前的 Robust 版本)
# ============================================================================
# ... (此处复用 run_pam_final.py 中的 PAMDataset 和 load_data_and_flatten) ...
# 为了代码简洁，这里假设您已经把这两个类/函数复制过来了
# 或者直接 import 之前的模块

class PAMDataset(Dataset):
    def __init__(self, pairs, n_items):
        self.pairs = pairs
        self.n_items = n_items

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        j = np.random.randint(1, self.n_items + 1)
        while j == i: j = np.random.randint(1, self.n_items + 1)
        return torch.tensor(u), torch.tensor(i), torch.tensor(j)


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


# ============================================================================
# 4. 全量评估函数 (保持一致)
# ============================================================================
def evaluate_full(model, test_data, n_items, device, k_list=[5, 10, 20], cold_threshold=5):
    model.eval()
    metrics = {g: {'cnt': 0} for g in ['All', 'Cold', 'Warm']}
    for g in metrics:
        for k in k_list: metrics[g][f'R@{k}'] = 0.0; metrics[g][f'N@{k}'] = 0.0
    all_candidates = torch.arange(1, n_items + 1).to(device)
    max_k = max(k_list)

    with torch.no_grad():
        test_loader = tqdm(test_data, desc="🔍 Evaluating", leave=False, ncols=80)
        for u_data in test_loader:
            user_id = u_data['user_id']
            history = u_data['history']
            target = u_data['target']

            u_tensor = torch.tensor([user_id]).to(device)
            scores = model.predict_all(u_tensor, all_candidates).squeeze()

            for visited in history:
                if 1 <= visited <= n_items: scores[visited - 1] = -float('inf')

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
            metrics[hit_group]['cnt'] += 1;
            metrics['All']['cnt'] += 1

    # 打印报表
    print(f"\n{'=' * 25} DropoutNet Evaluation {'=' * 25}")
    headers = ["Group", "Users"] + [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    header_fmt = "{:<6} | {:<6} | " + " | ".join(["{:<7}"] * len(k_list) * 2)
    print(header_fmt.format(*headers))
    print("-" * 100)

    res = {}
    for g in ['All', 'Cold', 'Warm']:
        cnt = metrics[g]['cnt']
        if cnt == 0: continue
        vals = []
        for k in k_list: vals.append(metrics[g][f'R@{k}'] / cnt)
        for k in k_list: vals.append(metrics[g][f'N@{k}'] / cnt)
        row = [g, str(cnt)] + [f"{v:.4f}" for v in vals]
        print(header_fmt.format(*row))
        if g == 'All':
            res['R@10'] = vals[1];
            res['N@10'] = vals[4]  # Index for 10

    print("=" * 100 + "\n")
    return res['R@10'] + res['N@10']


# ============================================================================
# 5. 主程序
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    # 1. Load Data
    train_pairs, test_data, n_users, n_items = load_data_and_flatten(Config.DATA_PATH)

    # 2. Load BERT
    pretrained_emb = None
    if os.path.exists(Config.BERT_PATH):
        pretrained_emb = torch.load(Config.BERT_PATH, map_location='cpu')
        print("✅ BERT Embedding Loaded.")
    else:
        print("⚠️ Warning: BERT not found, using random init.")

    # 3. Initialize DropoutNet
    model = DropoutNet(n_users, n_items, pretrained_emb=pretrained_emb,
                       emb_dim=Config.EMB_DIM, hidden_dim=Config.HIDDEN_DIM,
                       dropout_prob=Config.DROPOUT_RATE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    dataset = PAMDataset(train_pairs, n_items)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 4. Training Loop
    print(f"🚀 Start DropoutNet Training (Prob={Config.DROPOUT_RATE})...")
    best_score = 0.0
    patience = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
        for u, i, j in pbar:
            u, i, j = u.to(device), i.to(device), j.to(device)

            z_u, z_i = model(u, i)
            _, z_j = model(u, j)  # User tower shared

            # Standard BPR Loss
            loss = -torch.log(torch.sigmoid((z_u * z_i).sum(1) - (z_u * z_j).sum(1)) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Eval
        score = evaluate_full(model, test_data, n_items, device, k_list=Config.K_LIST, cold_threshold=Config.COLD_THRES)

        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), "dropoutnet_best.pth")
            print(f"🌟 New Best DropoutNet! Score: {best_score:.4f}")
        else:
            patience += 1
            print(f"⏳ Patience: {patience}/{Config.PATIENCE}")
            if patience >= Config.PATIENCE: break

    print("✅ DropoutNet Baseline Training Completed.")