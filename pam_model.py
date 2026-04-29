import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, n_users, n_items, content_dim=768):
        self.num_users = n_users
        self.num_items = n_items

        # 维度配置
        self.user_dim = 64
        self.content_dim = content_dim
        self.behavior_dim = 64
        self.hidden_dims = [128, 64]

        # 任务划分 (MOOCCube 建议阈值 5)
        self.cold_threshold = 5

        # Loss 权重
        self.lambda_cold = 2.0
        self.lambda_hot = 0.5
        self.gamma_m = 1.0
        self.gamma_s = 2.0
        self.gamma_a = 1.0

        # 训练配置
        self.inner_lr = 0.001
        self.outer_lr = 0.001
        self.temp = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PAM(nn.Module):
    def __init__(self, config, content_emb):
        super().__init__()
        self.cfg = config

        # 1. Embeddings
        self.user_emb = nn.Embedding(config.num_users, config.user_dim)
        self.item_beh_emb = nn.Embedding(config.num_items, config.behavior_dim)
        # 固定内容特征
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)

        # 投影层 (如果 BERT 维度不是 64，需要投影)
        # 如果 content_dim (768) != behavior_dim (64)，加一层 Linear
        if config.content_dim != config.behavior_dim:
            self.con_proj = nn.Linear(config.content_dim, config.behavior_dim)
        else:
            self.con_proj = nn.Identity()

        # 2. Network Parameters (Theta) - 手动定义以支持 Inner Loop
        self.vars = nn.ParameterList()
        self.lslr = nn.ParameterList()

        # User Tower & Item Tower
        # Item 输入 = Behavior + Content (Projected)
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

        # 3. Instructor (f_Sup)
        self.sup_w = nn.Parameter(torch.randn(config.behavior_dim, config.hidden_dims[-2]))
        self.sup_b = nn.Parameter(torch.zeros(config.behavior_dim))

        self.cold_memory = {}

    def get_item_features(self, i_idx, beh_override=None):
        beh = beh_override if beh_override is not None else self.item_beh_emb(i_idx)
        con = self.item_con_emb(i_idx)
        con = self.con_proj(con)  # 投影到 64 维
        return torch.cat([beh, con], dim=1)

    def forward_mlp(self, x, weights, is_item=False):
        idx_start = len(self.vars) // 2 if is_item else 0
        out = x
        prev_out = x
        num_layers = len(self.cfg.hidden_dims)

        for i in range(num_layers):
            w = weights[idx_start + 2 * i]
            b = weights[idx_start + 2 * i + 1]
            prev_out = out
            out = F.linear(out, w, b)
            if i < num_layers - 1:
                out = F.relu(out)
        return out, prev_out

    def compute_loss(self, z_u, z_i):
        logits = torch.mm(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(z_u.size(0)).to(z_u.device)
        return F.cross_entropy(logits, labels)

    def inner_loop(self, u, i):
        # 接收 u, i 两个 Tensor 参数
        e_u = self.user_emb(u)
        e_i = self.get_item_features(i)

        z_u, _ = self.forward_mlp(e_u, self.vars, False)
        z_i, _ = self.forward_mlp(e_i, self.vars, True)

        loss = self.compute_loss(z_u, z_i)

        grads = torch.autograd.grad(loss, self.vars, create_graph=True, allow_unused=True)

        omega = []
        for w, g, alpha in zip(self.vars, grads, self.lslr):
            if g is not None:
                omega.append(w - alpha * g)
            else:
                omega.append(w)
        return omega

    def forward(self, batch, pop):
        # 接收 batch, pop 两个参数
        u = batch['u']
        i = batch['i']

        is_cold = pop < self.cfg.cold_threshold

        total_loss = 0
        loss_dict = {}

        task_splits = {}

        # 任务划分逻辑
        if is_cold.sum() >= 2:
            task_splits['cold'] = {'u': u[is_cold], 'i': i[is_cold]}
        if (~is_cold).sum() >= 2:
            task_splits['hot'] = {'u': u[~is_cold], 'i': i[~is_cold]}

        # Meta-Learning Loop
        for name, data in task_splits.items():
            split = len(data['u']) // 2
            if split < 1: continue

            su, si = data['u'][:split], data['i'][:split]
            qu, qi = data['u'][split:], data['i'][split:]

            # Inner Loop (Call with u, i)
            omega = self.inner_loop(su, si)

            # Outer Loop
            e_u = self.user_emb(qu)
            e_i = self.get_item_features(qi)
            z_u, _ = self.forward_mlp(e_u, omega, False)
            z_i, _ = self.forward_mlp(e_i, omega, True)

            loss = self.compute_loss(z_u, z_i)
            weight = self.cfg.lambda_cold if name == 'cold' else self.cfg.lambda_hot
            total_loss += weight * loss
            loss_dict[name] = loss.item()

            if name == 'cold':
                with torch.no_grad():
                    beh = self.item_beh_emb(qi).detach()
                    for idx, item_id in enumerate(qi):
                        self.cold_memory[item_id.item()] = beh[idx]

        # Enhancer
        if 'hot' in task_splits:
            # 这里我们取后半部分 query 数据做增强
            h_data = task_splits['hot']
            split = len(h_data['u']) // 2
            hi = h_data['i'][split:]  # Query Items

            e_i_hot = self.get_item_features(hi)
            _, feat_prev = self.forward_mlp(e_i_hot, self.vars, True)

            pred_id = F.linear(feat_prev, self.sup_w, self.sup_b)
            real_id = self.item_beh_emb(hi).detach()

            loss_sup = F.mse_loss(pred_id, real_id)
            total_loss += self.cfg.gamma_s * loss_sup
            loss_dict['sup'] = loss_sup.item()

        return total_loss, loss_dict
