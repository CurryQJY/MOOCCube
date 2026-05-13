import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Config:
    def __init__(self, n_users, n_items, n_courses, content_dim=768):
        self.num_users = n_users
        self.num_items = n_items
        self.n_courses = n_courses

        # 维度配置
        self.user_dim = 256
        self.content_dim = content_dim
        self.behavior_dim = 256
        self.hidden_dims = [512, 256]

        # 阈值与权重
        self.cold_threshold = 15
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5
        self.gamma_s = 5.0  # Syllabus Enhancer (MLP层监督)
        self.gamma_llm = 10.0  # LLM Distillation
        self.gamma_usim = 1.0  # [新增] USIM Alignment Loss 权重

        # 学习率
        self.inner_lr = 0.001
        self.outer_lr = 0.001
        self.temp = 0.1


class USIM_Module(nn.Module):
    """
    [USIM 核心组件] 用户序列想象模块
    模拟论文中的: h_{t+1} = h_t + lambda * gradient(imagined_user)
    这里我们用一个残差网络来模拟这个 'gradient update' 过程
    """

    def __init__(self, input_dim):
        super(USIM_Module, self).__init__()
        self.imagination_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Tanh()  # Tanh 用于限制"想象"的幅度，防止特征漂移过远
        )

    def forward(self, content_emb):
        # 预测 "Delta"，即经过虚拟用户交互后的特征偏移量
        delta = self.imagination_net(content_emb)
        return delta


class PAM_LLM(nn.Module):
    def __init__(self, config, content_emb, i2c_map_dict):
        super(PAM_LLM, self).__init__()
        self.cfg = config

        # === 1. Hierarchy & Embeddings ===
        i2c_tensor = torch.zeros(config.num_items, dtype=torch.long)
        for i, c in i2c_map_dict.items():
            if i < config.num_items: i2c_tensor[i] = c
        self.register_buffer("i2c_map", i2c_tensor)

        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.course_emb = nn.Embedding(config.n_courses, config.user_dim)

        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # Content Adapter (D -> 256)
        self.con_proj = nn.Sequential(
            nn.Linear(config.content_dim, config.behavior_dim),
            nn.Tanh()
        )
        self.course_proj = nn.Sequential(
            nn.Linear(config.user_dim, config.user_dim),
            nn.Tanh()
        )

        # === [新增] USIM 想象模块 ===
        self.usim = USIM_Module(config.behavior_dim)

        self.layernorm = nn.LayerNorm(config.behavior_dim)

        # === 2. Meta-Parameters (MAML) ===
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        dims = [config.behavior_dim] + config.hidden_dims

        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.empty(dims[i + 1], dims[i]))
            nn.init.xavier_normal_(w)
            b = nn.Parameter(torch.zeros(dims[i + 1]))
            self.vars.extend([w, b])
            self.lslr.extend([nn.Parameter(torch.ones_like(w) * config.inner_lr),
                              nn.Parameter(torch.ones_like(b) * config.inner_lr)])

        # === 3. Syllabus Enhancer (监督信号) ===
        self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-2]))
        nn.init.xavier_normal_(self.sup_w)
        self.sup_b = nn.Parameter(torch.zeros(config.behavior_dim))

    def get_user_state(self, u_idx, target_i_idx):
        u = self.user_emb(u_idx)
        c_idx = self.i2c_map[target_i_idx]
        c = self.course_emb(c_idx)
        return u + self.course_proj(c)

    def get_item_features(self, i_idx, force_mask=False):
        """
        获取物品特征 (USIM Enhanced)
        """
        # 1. 基础 Content 投影
        con_raw = self.item_con_emb(i_idx)
        con_base = self.con_proj(con_raw)  # h_0

        # === [新增] USIM Imagination Step ===
        # 模拟: h_{t+1} = h_t + imagination
        delta = self.usim(con_base)
        con_refined = con_base + delta  # h_{refined}

        # 2. 获取真实 ID
        beh = self.item_beh_emb(i_idx)

        # 3. Hard Masking Logic
        if force_mask:
            # 冷启动：只使用"想象后"的内容特征
            return self.layernorm(con_refined), con_refined  # 返回 refined 用于 Loss 计算

        # 热启动：融合 (Content + Delta + ID)
        return self.layernorm(con_refined + beh), con_refined

    def forward_mlp(self, x, weights):
        out = x
        prev_out = x
        num_layers = len(self.cfg.hidden_dims)

        for i in range(num_layers):
            w = weights[2 * i]
            b = weights[2 * i + 1]
            if i == num_layers - 1:
                prev_out = out
            out = F.linear(out, w, b)
            if i < num_layers - 1:
                out = F.relu(out)
        return out, prev_out

    def inner_loop(self, u_emb, i_emb):
        """ MAML Inner Loop """
        z_u, _ = self.forward_mlp(u_emb, self.vars)
        z_i, _ = self.forward_mlp(i_emb, self.vars)

        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(len(u_emb)).to(u_emb.device)
        loss = F.cross_entropy(logits, labels)

        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)

        fast_weights = []
        for w, g, lr in zip(self.vars, grads, self.lslr):
            if g is not None:
                fast_weights.append(w - lr * g)
            else:
                fast_weights.append(w)
        return fast_weights

    def forward(self, batch, pop, llm_scores):
        u, i = batch['u'], batch['i']

        # 1. User State
        u_state = self.get_user_state(u, i)

        # 2. Item Features (USIM Logic)
        is_cold = pop < self.cfg.cold_threshold

        # 获取特征 (同时拿回 refined_content 用于对齐训练)
        # Hot Item: feat_full (Con+ID), con_refined_hot
        feat_full, con_refined_hot = self.get_item_features(i, force_mask=False)
        # Cold Item: feat_cold (Con Only), con_refined_cold
        feat_cold, _ = self.get_item_features(i, force_mask=True)

        # 组合特征供 MLP 使用
        i_features = torch.where(is_cold.unsqueeze(1), feat_cold, feat_full)

        total_loss = 0
        loss_dict = {}

        # === Phase 1: Meta-Learning (MAML) ===
        task_splits = {}
        if is_cold.sum() >= 4: task_splits['cold'] = {'u': u_state[is_cold], 'i': i_features[is_cold]}
        if (~is_cold).sum() >= 4: task_splits['hot'] = {'u': u_state[~is_cold], 'i': i_features[~is_cold]}

        for name, data in task_splits.items():
            split = len(data['u']) // 2
            su, si = data['u'][:split], data['i'][:split]
            qu, qi = data['u'][split:], data['i'][split:]

            omega = self.inner_loop(su, si)

            z_u, _ = self.forward_mlp(qu, omega)
            z_i, _ = self.forward_mlp(qi, omega)

            logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
            loss = F.cross_entropy(logits, torch.arange(len(qu)).to(qu.device))

            total_loss += (self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot) * loss
            loss_dict[name] = loss.item()

        # === Phase 2: Auxiliary Tasks (Hot Items) ===
        if (~is_cold).sum() > 0:
            hi_idx = i[~is_cold]
            hi_feat = i_features[~is_cold]

            # Target ID Embedding (Ground Truth)
            target_beh = self.item_beh_emb(hi_idx).detach()

            # A. Syllabus Enhancer (MLP深层对齐)
            _, feat_penultimate = self.forward_mlp(hi_feat, self.vars)
            pred_beh_syll = F.linear(feat_penultimate, self.sup_w, self.sup_b)
            loss_sup = F.mse_loss(pred_beh_syll, target_beh)
            total_loss += self.cfg.gamma_s * loss_sup
            loss_dict['sup'] = loss_sup.item()

            # B. [新增] USIM Alignment Loss (浅层对齐)
            # 强迫 USIM 想象出来的 Refined Content 能够逼近真实的 ID Embedding
            # 只有对齐了，想象才有意义 (Embedding Alignment Reward [cite: 180])
            hi_con_refined = con_refined_hot[~is_cold]
            loss_usim = F.mse_loss(hi_con_refined, target_beh)
            total_loss += self.cfg.gamma_usim * loss_usim
            loss_dict['usim'] = loss_usim.item()

        # === Phase 3: LLM Distillation ===
        mask_llm = (llm_scores > -0.5) & is_cold
        if mask_llm.sum() > 0:
            u_l = u_state[mask_llm]
            i_l = i_features[mask_llm]
            t_score = llm_scores[mask_llm]

            z_u, _ = self.forward_mlp(u_l, self.vars)
            z_i, _ = self.forward_mlp(i_l, self.vars)

            s_score = torch.sigmoid((z_u * z_i).sum(dim=1))
            loss_llm = F.mse_loss(s_score, t_score)

            total_loss += self.cfg.gamma_llm * loss_llm
            loss_dict['llm'] = loss_llm.item()

        return total_loss, loss_dict