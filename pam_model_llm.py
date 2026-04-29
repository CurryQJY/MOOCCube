import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.num_users = n_users
        self.num_items = n_items
        self.user_dim = 64
        self.content_dim = content_dim
        self.behavior_dim = 64
        self.hidden_dims = [128, 64]

        self.cold_threshold = 5
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5

        # Loss Weights
        self.gamma_s = 5.0  # Syllabus -> ID
        self.gamma_llm = 0.5  # LLM Distillation

        self.inner_lr = 0.001
        self.outer_lr = 0.001
        self.temp = 0.1


class PAM_LLM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config

        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # Content Projector
        if config.content_dim != config.behavior_dim:
            self.con_proj = nn.Sequential(
                nn.Linear(config.content_dim, 256),
                nn.ReLU(),
                nn.Linear(256, config.behavior_dim)
            )
        else:
            self.con_proj = nn.Identity()

        # Meta-Parameters
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()
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

        # 监督信号映射层 (128 -> 64)
        # 这里的输入维度必须对应 hidden_dims[-2] 即 128
        self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-2]))
        self.sup_b = nn.Parameter(torch.zeros(config.behavior_dim))

    def get_item_features(self, i_idx):
        beh = self.item_beh_emb(i_idx)
        con = self.con_proj(self.item_con_emb(i_idx))
        return torch.cat([beh, con], dim=1)

    def forward_mlp(self, x, weights, is_item=False):
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        prev_out = x  # 初始化

        num_layers = len(self.cfg.hidden_dims)

        for i in range(num_layers):
            w, b = weights[idx_start + 2 * i], weights[idx_start + 2 * i + 1]

            # 【修复点】在通过最后一层线性层之前，保存当前的 out
            # 这就是"倒数第二层"的特征，维度是 128
            if i == num_layers - 1:
                prev_out = out

            out = F.linear(out, w, b)

            if i < num_layers - 1:
                out = F.relu(out)

        # 返回: (最终输出 64维, 倒数第二层输出 128维)
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

    def forward(self, batch, pop, llm_scores):
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

        # 2. Syllabus Enhancer
        if 'hot' in task_splits:
            hi = task_splits['hot']['i'][len(task_splits['hot']['i']) // 2:]

            # 这里调用 forward_mlp 会返回 (z_i, feat)
            # feat 就是 128 维的中间特征
            _, feat = self.forward_mlp(self.get_item_features(hi), self.vars, True)

            # 现在 feat 是 128维, sup_w 是 (64, 128)，可以相乘了
            loss_sup = F.mse_loss(F.linear(feat, self.sup_w, self.sup_b), self.item_beh_emb(hi).detach())
            total_loss += self.cfg.gamma_s * loss_sup
            loss_dict['sup'] = loss_sup.item()

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
