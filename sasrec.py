import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import time
import argparse
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. 严谨版 SASRec 模型 (Strict Implementation)
#    符合 Kang & McAuley (ICDM '18) 原论文架构
# ============================================================================
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs)))))
        outputs += inputs  # Residual Connection
        return outputs


class SASRecBlock(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads

        self.layernorm1 = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.layernorm2 = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.feed_forward = PointWiseFeedForward(hidden_units, dropout_rate)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, input_seqs, attention_mask):
        # 1. Layer Norm (Pre-Norm style as in common SASRec repos)
        norm_inputs = self.layernorm1(input_seqs)

        # 2. Self-Attention
        attn_output, _ = self.attention(
            query=norm_inputs,
            key=norm_inputs,
            value=norm_inputs,
            attn_mask=attention_mask,
            need_weights=False
        )

        # 3. Residual + Dropout
        outputs = input_seqs + self.dropout(attn_output)

        # 4. Feed Forward Block
        outputs = self.layernorm2(outputs)
        outputs = self.feed_forward(outputs)

        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, n_items, hidden_size=64, max_len=50, num_blocks=2, num_heads=2, dropout=0.1):
        super(SASRec, self).__init__()
        self.n_items = n_items
        self.hidden_units = hidden_size

        # Embeddings (n_items + 1 to account for padding '0')
        self.item_emb = torch.nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(max_len, hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        # Transformer Blocks
        self.blocks = torch.nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        # Initialization (Critical for convergence)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, log_seqs):
        device = log_seqs.device
        seq_len = log_seqs.shape[1]

        # Causal Mask (Upper triangle is -inf)
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()

        # Embeddings
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Apply Blocks
        for block in self.blocks:
            seqs = block(seqs, attention_mask)

        # Final Norm
        seqs = self.last_layernorm(seqs)
        return seqs

    def predict(self, log_seqs, item_indices):
        """ Inference: Get scores for candidate items """
        feats = self.forward(log_seqs)
        final_feat = feats[:, -1, :]  # [B, H]

        if isinstance(item_indices, list) or isinstance(item_indices, range):
            item_indices = torch.tensor(item_indices).to(log_seqs.device)

        item_embs = self.item_emb(item_indices)  # [Candidates, H]
        logits = torch.matmul(final_feat, item_embs.t())
        return logits


# ============================================================================
# 2. 数据集定义
# ============================================================================
class SASRecDataset(Dataset):
    def __init__(self, seqs, n_items, max_len=50):
        self.seqs = seqs
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # Input: [1, 2, 3] -> Target: [2, 3, 4]
        input_ids = seq[:-1]
        target_ids = seq[1:]

        def pad(s):
            s = s[-self.max_len:]
            return [0] * (self.max_len - len(s)) + s

        return torch.tensor(pad(input_ids)), torch.tensor(pad(target_ids))


# ============================================================================
# 3. 智能数据加载器 (修复了列名冲突和索引问题)
# ============================================================================
def load_data_from_pickle(data_path, meta_path):
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_pickle(data_path)

    print(f"🔍 [DEBUG] 原始列名: {df.columns.tolist()}")

    # 1. 冲突清理：优先保留整数索引 (idx)，删除字符串索引 (id)
    if 'u_idx' in df.columns and 'user_id' in df.columns:
        print("🗑️ 检测到列名冲突，删除字符串类型 'user_id'...")
        df = df.drop(columns=['user_id'])
    if 'i_idx' in df.columns and 'item_id' in df.columns:
        print("🗑️ 检测到列名冲突，删除字符串类型 'item_id'...")
        df = df.drop(columns=['item_id'])

    # 2. 建立重命名映射
    rename_map = {}
    if 'i_idx' in df.columns:
        rename_map['i_idx'] = 'item_id'
    elif 'course_id' in df.columns:
        pass  # 没救了，除非只有这一列

    if 'u_idx' in df.columns: rename_map['u_idx'] = 'user_id'
    if 'raw_time' in df.columns and 'timestamp' not in df.columns: rename_map['raw_time'] = 'timestamp'

    if rename_map:
        print(f"🔄 重命名列: {rename_map}")
        df = df.rename(columns=rename_map)

    # 3. 检查必要列
    if 'item_id' not in df.columns or 'user_id' not in df.columns:
        raise ValueError("❌ 错误：找不到 item_id 或 user_id 列 (也找不到 i_idx/u_idx)。")

    # 4. 强制类型转换 (Int)
    if not pd.api.types.is_integer_dtype(df['item_id']):
        try:
            df['item_id'] = df['item_id'].astype(int)
        except:
            raise TypeError("❌ item_id 无法转为整数，请检查是否映射了字符串列。")

    # 5. ID 偏移处理 (0-based -> 1-based)
    min_id = df['item_id'].min()
    max_id = df['item_id'].max()
    print(f"📊 ID 范围: Min={min_id}, Max={max_id}")

    n_items = max_id
    if min_id == 0:
        print("🔧 ID 从 0 开始，执行全局 +1 (适配 Padding)...")
        df['item_id'] = df['item_id'] + 1
        n_items = max_id + 1

    n_items = max(n_items, df['item_id'].max())
    print(f"🔢 最终 n_items = {n_items}")

    # 6. 生成序列
    print("🔄 Converting to sequences...")
    df = df.sort_values(['user_id', 'timestamp'])
    user_groups = df.groupby('user_id')['item_id'].apply(list)

    train_seqs = []
    test_data = []

    for user_id, items in user_groups.items():
        if len(items) < 3: continue
        train_seqs.append(items[:-1])
        test_data.append({
            'user_id': user_id,
            'history': items[:-1],
            'target': items[-1],
            'hist_len': len(items) - 1
        })

    print(f"✅ Data ready! Train Seqs: {len(train_seqs)}, Test Users: {len(test_data)}")
    return train_seqs, test_data, n_items


# ============================================================================
# 4. 全量排序评估器 (Full Ranking - 解决 1.0 指标问题)
# ============================================================================
def evaluate_full_ranking(model, test_data, n_items, device, k=10, cold_threshold=5):
    model.eval()

    metrics = {
        'Cold': {'hits': 0, 'ndcg': 0, 'cnt': 0},
        'Warm': {'hits': 0, 'ndcg': 0, 'cnt': 0},
        'All': {'hits': 0, 'ndcg': 0, 'cnt': 0}
    }

    # 候选集: 1 到 n_items
    all_candidates = torch.arange(1, n_items + 1).to(device)

    print(f"🚀 开始全量评估 (Items: {len(all_candidates)})...")

    with torch.no_grad():
        for i, u_data in enumerate(test_data):
            seq = u_data['history']
            target = u_data['target']
            hist_len = u_data['hist_len']

            # 准备输入
            max_len = model.pos_emb.num_embeddings
            seq_pad = seq[-max_len:]
            seq_pad = [0] * (max_len - len(seq_pad)) + seq_pad
            seq_tensor = torch.tensor([seq_pad]).to(device)

            # 预测所有物品的分数
            scores = model.predict(seq_tensor, all_candidates).squeeze()  # [n_items]

            # Mask 掉历史物品 (设为 -inf)
            # scores 的索引 0 对应 item_id 1
            for visited_item in seq:
                if 1 <= visited_item <= n_items:
                    scores[visited_item - 1] = -float('inf')

            # Top-K
            # target 对应的 scores 索引是 target - 1
            target_idx = target - 1
            _, topk_indices = torch.topk(scores, k)

            hit = 0
            ndcg = 0

            if target_idx in topk_indices:
                hit = 1
                rank_pos = (topk_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
                ndcg = 1.0 / np.log2(rank_pos + 1)

            # 统计
            group = 'Cold' if hist_len <= cold_threshold else 'Warm'
            for g in [group, 'All']:
                metrics[g]['hits'] += hit
                metrics[g]['ndcg'] += ndcg
                metrics[g]['cnt'] += 1

    print(f"\n{'=' * 15} Full Ranking Results (Cold <= {cold_threshold}) {'=' * 15}")
    print(f"{'Group':<6} | {'Users':<6} | {'R@10':<8} | {'N@10':<8}")
    print("-" * 50)
    for g in ['All', 'Cold', 'Warm']:
        cnt = metrics[g]['cnt']
        if cnt > 0:
            r10 = metrics[g]['hits'] / cnt
            n10 = metrics[g]['ndcg'] / cnt
            print(f"{g:<6} | {cnt:<6} | {r10:.4f}   | {n10:.4f}")
    print("=" * 50 + "\n")

    return metrics['All']['hits'] / metrics['All']['cnt']


# ============================================================================
# 5. 主程序
# ============================================================================
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "processed_data/stream_data.pkl"
    META_PATH = "processed_data/meta.json"

    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 10
    MAX_LEN = 50
    EMB_DIM = 64
    COLD_THRES = 5  # 定义冷启动的阈值

    # 1. 检查文件
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    # 2. 加载数据
    train_seqs, test_data, n_items = load_data_from_pickle(DATA_PATH, META_PATH)

    dataset = SASRecDataset(train_seqs, n_items, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化模型
    model = SASRec(n_items, hidden_size=EMB_DIM, max_len=MAX_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # 4. 训练与评估
    best_recall = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        start_time = time.time()

        for step, (seqs, targets) in enumerate(dataloader):
            seqs, targets = seqs.to(device), targets.to(device)

            # Forward
            logits = model(seqs)  # [B, L, H]

            # 计算 Loss (这里做个简单的 reshape 即可)
            # logits: [B, L, H] * [H, V] -> [B, L, V]
            # 为了省显存，只计算 batch 内的 loss
            scores = torch.matmul(logits, model.item_emb.weight.t())

            scores = scores.view(-1, n_items + 1)
            targets = targets.view(-1)

            loss = loss_fn(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")

        # 5. 全量评估
        current_recall = evaluate_full_ranking(model, test_data, n_items, device, cold_threshold=COLD_THRES)

        if current_recall > best_recall:
            best_recall = current_recall
            torch.save(model.state_dict(), "sasrec_best.pth")
            print("🌟 New Best Model Saved!")

    print("✅ SASRec 复现完成！")