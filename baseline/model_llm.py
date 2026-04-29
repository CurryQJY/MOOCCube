import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PAM_LLM_Enhanced(nn.Module):
    def __init__(self, cfg, content_features):
        super().__init__()
        self.cfg = cfg
        self.emb_dim = cfg.emb_dim
        self.temperature = cfg.temperature  # 例如 0.1

        # --- 1. Embedding Layers ---
        # 用户 ID Embedding (随机初始化，可学习)
        self.user_emb = nn.Embedding(cfg.n_users, self.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)

        # 物品 ID Embedding (随机初始化，可学习)
        self.item_id_emb = nn.Embedding(cfg.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        # --- 2. Content & LLM Processing ---
        # 冻结的 BERT 特征 (来自 stream_data.pkl / content_emb.pt)
        # 注册为 buffer，不参与梯度更新
        self.register_buffer("content_features", content_features)

        # 将 768维 BERT 投影到 64维
        self.content_proj = nn.Sequential(
            nn.Linear(768, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )

        # 将 1维 LLM Score 投影到 64维
        self.llm_proj = nn.Sequential(
            nn.Linear(1, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )

        # --- 3. Gate Net (核心决策层) ---
        # 输入是 [ID; Semantic] 拼接，输出是 1维权重 alpha
        self.gate_net = nn.Linear(self.emb_dim * 2, 1)

    def get_item_vector(self, i_idx, llm_scores, force_cold=False):
        """
        生成最终物品向量的核心逻辑
        """
        # ================= Path A: ID Embedding =================
        id_e = self.item_id_emb(i_idx)  # [Batch, 64]

        # [关键点]: Force Cold / Dropout 机制
        # 训练时随机丢弃 ID，或者测试时强制冷启动
        if force_cold or (self.training and np.random.rand() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)  # ID 变为空，模拟冷启动/防止过拟合

        # ================= Path B: Content Embedding =================
        # 从冻结的表中查出 768维特征，投影到 64维
        raw_content = self.content_features[i_idx]
        content_e = self.content_proj(raw_content)

        # ================= Path C: LLM Embedding =================
        # [关键点]: Mask & Clamp
        # 1. Clamp: 把 -1 (缺失) 变成 0.0，防止负数输入 MLP
        val_llm = torch.clamp(llm_scores, min=0.0).unsqueeze(-1)  # [Batch, 1]

        # 2. Mask: 只有 > -0.5 (即有效分数) 的才保留，否则置 0
        mask_llm = (llm_scores > -0.5).float().unsqueeze(-1)

        # 3. Project & Apply Mask: 缺失值的向量会被物理切断 (全0)
        llm_e = self.llm_proj(val_llm) * mask_llm

        # ================= Fusion 1: Semantic Addition =================
        # 语义 = 客观内容 + 主观 LLM 评价
        semantic_e = content_e + llm_e

        # ================= Fusion 2: Gating & Weighted Sum =================
        # [关键点]: 拼接 ID 和 语义，输入 Gate Net
        gate_input = torch.cat([id_e, semantic_e], dim=-1)  # [Batch, 128]

        # 计算 alpha (语义的权重), Sigmoid 压缩到 [0, 1]
        alpha = torch.sigmoid(self.gate_net(gate_input))

        # [关键点]: 加权求和
        # alpha 越大，越听语义的；alpha 越小，越听 ID 的
        final_item_e = (1 - alpha) * id_e + alpha * semantic_e

        # ================= Normalization =================
        # [关键点]: L2 归一化，把向量拉到单位球面上
        final_item_e = F.normalize(final_item_e, dim=1)
        semantic_e = F.normalize(semantic_e, dim=1)  # 辅助用的语义向量也归一化
        id_e = F.normalize(id_e, dim=1) if not force_cold else id_e  # ID 也归一化

        return final_item_e, semantic_e, id_e, alpha

    def forward(self, batch, force_cold=False):
        u_idx = batch['u_idx']
        i_idx = batch['i_idx']
        llm_s = batch['llm_score']

        # 1. 获取 User Vector
        z_u = self.user_emb(u_idx)
        z_u = F.normalize(z_u, dim=1)  # L2 归一化

        # 2. 获取 Item Vector (包含复杂的融合逻辑)
        z_i, semantic_e, id_e, alpha = self.get_item_vector(i_idx, llm_s, force_cold)

        # 3. Main Logic (主预测)
        # 计算点积 (Dot Product) / Temperature
        logits = torch.sum(z_u * z_i, dim=1) / self.temperature

        # 4. Aux Logic (辅助对齐)
        # [关键点]: 对齐损失
        # 计算 masked_id_e 和 semantic_e 的点积
        # 如果 id_e 被 dropout 成了 0，这里 aux_sim 就是 0，梯度截断 (Circuit Breaker)
        aux_sim = torch.sum(id_e * semantic_e, dim=1)

        return logits, aux_sim, alpha