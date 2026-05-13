import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, n_users, n_items, n_courses, emb_dim):
        self.n_users = n_users
        self.num_items = n_items
        self.n_courses = n_courses
        self.emb_dim = emb_dim
        self.hidden_dim = 256
        self.n_layers = 2

        # 保持猛药配置
        self.outer_lr = 5e-3
        self.inner_lr = 1e-2
        self.cold_threshold = 15
        self.alpha = 10.0


class ImaginationModule(nn.Module):
    """
    USIM 核心模块：用户序列想象
    功能：输入当前的 Content Embedding，预测一个“虚拟用户交互”带来的梯度更新量。
    论文对应：模拟公式 h_{t+1} = h_t + lambda * e_{user}
    """

    def __init__(self, emb_dim):
        super(ImaginationModule, self).__init__()
        # 这是一个轻量级的生成器，用于生成“想象中的用户反馈”
        self.generator = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()  # 限制更新幅度，防止想象过度导致数值不稳定
        )

    def forward(self, content_emb):
        # 生成虚拟的用户反馈 (Imagined User Feedback)
        delta = self.generator(content_emb)
        return delta


class PAM_LLM(nn.Module):
    def __init__(self, cfg, content_emb, i2c_map_dict):
        super(PAM_LLM, self).__init__()
        self.cfg = cfg
        self.num_items = cfg.num_items

        # 1. 注册 i2c 映射
        i2c_tensor = torch.zeros(cfg.num_items, dtype=torch.long)
        for i, c in i2c_map_dict.items():
            if i < cfg.num_items:
                i2c_tensor[i] = c
        self.register_buffer("i2c_map", i2c_tensor)

        # 2. Embeddings
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.course_emb = nn.Embedding(cfg.n_courses, cfg.emb_dim)

        # 冻结 BERT，改用 Adapter/Projection 模式 (符合 PEFT 趋势)
        self.item_content = nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.item_id_emb = nn.Embedding(cfg.num_items, cfg.emb_dim)

        # 3. Projections
        self.content_proj = nn.Sequential(nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.Tanh())
        self.course_proj = nn.Sequential(nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.Tanh())

        # === 创新点融入：USIM 想象模块 ===
        self.imagination_net = ImaginationModule(cfg.emb_dim)

        self.layernorm = nn.LayerNorm(cfg.emb_dim)

        # MLP
        self.input_dim = cfg.emb_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layer = cfg.n_layers
        self.user_mlp = self._build_mlp(self.input_dim, self.hidden_dim)
        self.item_mlp = self._build_mlp(self.input_dim, self.hidden_dim)

        self.vars = nn.ParameterList()
        for p in self.user_mlp.parameters(): self.vars.append(p)
        for p in self.item_mlp.parameters(): self.vars.append(p)

        self.idx_user = 0
        self.idx_item = len(list(self.user_mlp.parameters()))

    def _build_mlp(self, in_dim, out_dim):
        layers = []
        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def get_user_state(self, u_idx, target_i_idx):
        u = self.user_emb(u_idx)
        c_idx = self.i2c_map[target_i_idx]
        c = self.course_emb(c_idx)
        return u + self.course_proj(c)

    def get_item_embeddings(self, i_idx):
        """
        获取物品特征，融入 USIM 想象机制
        """
        # 1. 基础 BERT 内容
        raw_content = self.item_content(i_idx)
        # 投影到推荐空间 (Initial State h_0)
        h_0 = self.content_proj(raw_content)

        # === 创新：USIM 想象微调过程 ===
        # 论文逻辑：先根据内容想象用户交互，再利用反馈更新 Embedding
        # h_{t+1} = h_t + lambda * imagined_feedback
        imagined_feedback = self.imagination_net(h_0)

        # 获得“想象优化后”的内容嵌入
        # 这里的 0.1 相当于论文公式中的 lambda (学习率)
        refined_content = h_0 + 0.1 * imagined_feedback

        # 2. 获取 ID
        id_emb = self.item_id_emb(i_idx)

        return refined_content, id_emb

    def forward_mlp(self, x, weights, is_item):
        if weights is None:
            if is_item:
                return self.item_mlp(x), None
            else:
                return self.user_mlp(x), None

        idx_start = self.idx_user if not is_item else self.idx_item
        x = x.float()
        for i in range(self.n_layer):
            w = weights[idx_start + 2 * i]
            b = weights[idx_start + 2 * i + 1]
            x = F.linear(x, w, b)
            if i < self.n_layer - 1:
                # Manual Norm
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)
                x = (x - mean) / (std + 1e-5)
                x = F.relu(x)
        return x, None

    def forward(self, batch, pop, llm_scores=None, vars=None):
        if vars is None: vars = self.vars

        # User State
        u_state = self.get_user_state(batch['u'], batch['i'])

        # Item Embeddings (经过想象微调)
        pos_content, pos_id = self.get_item_embeddings(batch['i'])
        neg_content, neg_id = self.get_item_embeddings(batch['neg_i'])

        # Hard Masking
        mask_cold = (pop < self.cfg.cold_threshold).float().unsqueeze(1)

        # 训练时：冷物品强制只使用“想象微调后”的 Content
        pos_final = pos_content + (1.0 - mask_cold) * pos_id
        pos_final = self.layernorm(pos_final)

        neg_final = neg_content + neg_id
        neg_final = self.layernorm(neg_final)

        # MLP
        z_u, _ = self.forward_mlp(u_state, vars, False)
        z_i_pos, _ = self.forward_mlp(pos_final, vars, True)
        z_i_neg, _ = self.forward_mlp(neg_final, vars, True)

        pos_scores = (z_u * z_i_pos).sum(dim=1)
        neg_scores = (z_u * z_i_neg).sum(dim=1)

        # Loss
        loss_task = F.margin_ranking_loss(pos_scores, neg_scores, torch.ones_like(pos_scores), margin=1.0)

        loss_distill = torch.tensor(0.0, device=loss_task.device)
        if llm_scores is not None and mask_cold.any():
            pred_cold = torch.sigmoid(pos_scores[mask_cold.bool().squeeze(1)])
            target_cold = llm_scores[mask_cold.bool().squeeze(1)]
            valid_mask = target_cold >= 0
            if valid_mask.any():
                loss_distill = F.mse_loss(pred_cold[valid_mask], target_cold[valid_mask])

        return loss_task + self.cfg.alpha * loss_distill, pos_scores
