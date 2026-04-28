import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


def setup_seed(seed=2025):
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


# ============================================================================
# 1. 配置参数 (Configuration)
# ============================================================================
class Config:
    DATA_PATH = "processed_data/stream_data.pkl"
    BERT_PATH = "processed_data/content_emb.pt"

    BATCH_SIZE = 2048

    # Stage 1: Pre-training Rec
    LR_REC = 0.001
    EPOCHS_STAGE1 = 20

    # Stage 2: GAN Training (Multi-Task)
    LR_G = 0.00005  # 极低学习率，微调生成器
    LR_D = 0.0001
    EPOCHS_STAGE2 = 100  # 增加轮数，让它跑久一点

    # 🔥 核心修改：增加耐心值
    PATIENCE = 15  # GAN 收敛慢，给它更多机会

    # Loss 权重平衡
    W_BPR = 1.0
    W_MSE = 0.5
    W_GAN = 0.01

    EMB_DIM = 64
    HIDDEN_DIM = 128

    K_LIST = [5, 10, 20]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 2. 模型定义 (Spectral Norm + LayerNorm)
# ============================================================================
class Generator(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # 输入归一化
        x = F.normalize(x, p=2, dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(64, 1)),
            nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)


class GAR_System(nn.Module):
    def __init__(self, n_users, n_items, pretrained_emb=None):
        super(GAR_System, self).__init__()

        # Rec Components
        self.user_emb = nn.Embedding(n_users + 1, Config.EMB_DIM)
        self.user_mlp = nn.Sequential(
            nn.Linear(Config.EMB_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.EMB_DIM)
        )

        self.item_id_emb = nn.Embedding(n_items + 1, Config.EMB_DIM)
        self.item_mlp = nn.Sequential(
            nn.Linear(Config.EMB_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.EMB_DIM)
        )

        # BERT (Frozen)
        self.bert = nn.Embedding(n_items + 1, 768, padding_idx=0)
        if pretrained_emb is not None:
            valid_rows = min(pretrained_emb.shape[0], n_items + 1)
            self.bert.weight.data[:valid_rows] = pretrained_emb[:valid_rows]
            self.bert.weight.requires_grad = False

        # GAN
        self.G = Generator(input_dim=768, output_dim=Config.EMB_DIM)
        self.D = Discriminator(input_dim=Config.EMB_DIM)
        self._init_rec_weights()

    def _init_rec_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)

    def predict_all(self, u, all_items, mode='real'):
        u_e = self.user_mlp(self.user_emb(u))
        if mode == 'real':
            raw_emb = self.item_id_emb(all_items)
        else:
            # GAN Mode
            bert_v = self.bert(all_items)
            raw_emb = self.G(bert_v)

        i_e = self.item_mlp(raw_emb)
        return torch.matmul(u_e, i_e.t())


# ============================================================================
# 3. 数据处理
# ============================================================================
class PAMDataset(Dataset):
    def __init__(self, pairs, n_items):
        self.pairs = pairs;
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

    if 'u_idx' in df.columns: df = df.rename(columns={'u_idx': 'user_id', 'i_idx': 'item_id', 'raw_time': 'timestamp'})
    df = df.loc[:, ~df.columns.duplicated()]

    if df['user_id'].dtype == 'object': df['user_id'] = LabelEncoder().fit_transform(df['user_id'].astype(str))
    if df['item_id'].dtype == 'object': df['item_id'] = LabelEncoder().fit_transform(df['item_id'].astype(str))

    df['user_id'] = df['user_id'].astype(int) + 1
    df['item_id'] = df['item_id'].astype(int) + 1

    n_items, n_users = df['item_id'].max(), df['user_id'].max()
    df = df.sort_values(['user_id', 'timestamp'])

    train_pairs, test_data = [], []
    for u, items in tqdm(df.groupby('user_id')['item_id'].apply(list).items(), desc="Building"):
        if len(items) < 3: continue
        test_data.append({'user_id': u, 'history': items[:-1], 'target': items[-1], 'hist_len': len(items) - 1})
        for i in items[:-1]: train_pairs.append([u, i])

    return train_pairs, test_data, n_users, n_items


# ============================================================================
# 4. 评估函数
# ============================================================================
def evaluate_full(model, test_data, n_items, device, mode='real'):
    model.eval()

    metrics = {g: {} for g in ['All', 'Cold', 'Warm']}
    for g in metrics:
        metrics[g]['cnt'] = 0
        for k in Config.K_LIST:
            metrics[g][f'R@{k}'] = 0.0
            metrics[g][f'N@{k}'] = 0.0

    all_items = torch.arange(1, n_items + 1).to(device)
    max_k = max(Config.K_LIST)

    with torch.no_grad():
        loader = DataLoader(test_data, batch_size=64, collate_fn=lambda x: x, num_workers=0)

        for batch in tqdm(loader, desc=f"Eval ({mode})", leave=False, ncols=80):
            valid_batch = [d for d in batch if d['user_id'] < model.user_emb.num_embeddings]
            if not valid_batch: continue

            u_ids = torch.tensor([d['user_id'] for d in valid_batch]).to(device)
            scores = model.predict_all(u_ids, all_items, mode)

            for idx, d in enumerate(valid_batch):
                s = scores[idx]

                # Mask History
                hist_tensor = torch.tensor(d['history'], device=device, dtype=torch.long)
                hist_tensor = hist_tensor[(hist_tensor > 0) & (hist_tensor <= n_items)]
                if len(hist_tensor) > 0:
                    s[hist_tensor - 1] = -1e9

                _, topk_indices = torch.topk(s, max_k)
                topk_items = topk_indices.cpu().numpy() + 1

                target = d['target']
                grp = 'Cold' if d['hist_len'] <= 5 else 'Warm'
                groups = ['All', grp]

                if target in topk_items:
                    rank = np.where(topk_items == target)[0][0]
                    for k in Config.K_LIST:
                        if rank < k:
                            hit = 1.0
                            ndcg = 1.0 / np.log2(rank + 2)
                            for g in groups:
                                metrics[g][f'R@{k}'] += hit
                                metrics[g][f'N@{k}'] += ndcg

                for g in groups:
                    metrics[g]['cnt'] += 1

    print(f"\n{'=' * 40} GAR Evaluation ({mode}) {'=' * 40}")
    headers = ["Group", "Users"] + [f"R@{k}" for k in Config.K_LIST] + [f"N@{k}" for k in Config.K_LIST]
    header_fmt = "{:<6} | {:<6} | " + " | ".join(["{:<7}"] * len(Config.K_LIST) * 2)
    print(header_fmt.format(*headers))
    print("-" * 115)

    final_res = {}

    for g in ['All', 'Cold', 'Warm']:
        cnt = max(1, metrics[g]['cnt'])

        vals = []
        for k in Config.K_LIST:
            val = metrics[g][f'R@{k}'] / cnt
            vals.append(val)
            if g == 'Cold': final_res[f'Cold-R@{k}'] = val

        for k in Config.K_LIST:
            val = metrics[g][f'N@{k}'] / cnt
            vals.append(val)
            if g == 'Cold': final_res[f'Cold-N@{k}'] = val

        row = [g, str(cnt)] + [f"{v:.4f}" for v in vals]
        print(header_fmt.format(*row))

    print("=" * 115 + "\n")
    return final_res


# ============================================================================
# 5. 主程序
# ============================================================================
if __name__ == "__main__":
    setup_seed(20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    train_pairs, test_data, n_users, n_items = load_data_and_flatten(Config.DATA_PATH)
    pretrained_emb = torch.load(Config.BERT_PATH, map_location='cpu') if os.path.exists(Config.BERT_PATH) else None

    model = GAR_System(n_users, n_items, pretrained_emb).to(device)
    dataset = PAMDataset(train_pairs, n_items)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Stage 1: Pre-train Rec ---
    print("\n🚀 [Stage 1] Pre-training Rec...")

    best_s1_score = 0.0
    model_loaded = False

    # 尝试加载旧模型
    if os.path.exists("gar_stage1_best.pth"):
        try:
            model.load_state_dict(torch.load("gar_stage1_best.pth"), strict=False)
            print("✅ Found and Loaded Stage 1 model.")
            model_loaded = True
        except Exception as e:
            print(f"⚠️ Load failed ({e}). Re-training...")

    # 如果没加载成功，则开始训练
    if not model_loaded:
        opt_rec = torch.optim.Adam(list(model.user_emb.parameters()) + list(model.item_id_emb.parameters()) +
                                   list(model.user_mlp.parameters()) + list(model.item_mlp.parameters()),
                                   lr=Config.LR_REC)

        for ep in range(1, Config.EPOCHS_STAGE1 + 1):
            model.train()
            for u, i, j in tqdm(dataloader, desc=f"S1-Ep {ep}", leave=False):
                u, i, j = u.to(device), i.to(device), j.to(device)
                u_e = model.user_mlp(model.user_emb(u))
                i_e = model.item_mlp(model.item_id_emb(i))
                j_e = model.item_mlp(model.item_id_emb(j))
                loss = -torch.log(torch.sigmoid((u_e * i_e).sum(1) - (u_e * j_e).sum(1)) + 1e-8).mean()
                opt_rec.zero_grad();
                loss.backward();
                opt_rec.step()

            res = evaluate_full(model, test_data, n_items, device, 'real')
            curr_score = res.get('Cold-R@10', 0) + res.get('Cold-N@10', 0)

            if curr_score > best_s1_score:
                best_s1_score = curr_score
                torch.save(model.state_dict(), "gar_stage1_best.pth")
                print(f"   🌟 S1 New Best! Cold Score: {best_s1_score:.4f}")

        if os.path.exists("gar_stage1_best.pth"):
            model.load_state_dict(torch.load("gar_stage1_best.pth"))

    # 再次确保 best_s1_score 存在（用于日志对比，不影响Stage 2）
    if best_s1_score == 0.0:
        print("   📊 Evaluating loaded Stage 1 model...")
        res = evaluate_full(model, test_data, n_items, device, 'real')
        best_s1_score = res.get('Cold-R@10', 0) + res.get('Cold-N@10', 0)

    print(f"✅ Stage 1 Final Score: {best_s1_score:.4f}")

    # --- Stage 2: GAN (BPR + MSE + GAN) ---
    print("\n🚀 [Stage 2] Training GAN (Multi-Task: BPR+MSE+GAN)...")

    # 冻结 Rec 参数
    for p in model.user_emb.parameters(): p.requires_grad = False
    for p in model.user_mlp.parameters(): p.requires_grad = False
    for p in model.item_id_emb.parameters(): p.requires_grad = False
    for p in model.item_mlp.parameters(): p.requires_grad = False

    opt_G = torch.optim.Adam(model.G.parameters(), lr=Config.LR_G)
    opt_D = torch.optim.Adam(model.D.parameters(), lr=Config.LR_D)

    # 🔥🔥🔥 核心修改：重置最佳分数，让 Stage 2 独立记录 🔥🔥🔥
    best_s2_score = 0.0
    patience = 0

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    for ep in range(1, Config.EPOCHS_STAGE2 + 1):
        model.G.train();
        model.D.train()

        pbar = tqdm(dataloader, desc=f"S2-Ep {ep}", ncols=100)

        for u, i, j in pbar:
            u, i, j = u.to(device), i.to(device), j.to(device)

            # 1. 准备数据
            real_emb = model.item_id_emb(i).detach()  # Target
            bert_i = model.bert(i)
            bert_j = model.bert(j)

            # 2. Train D
            opt_D.zero_grad()
            fake_emb_detach = model.G(bert_i).detach()

            loss_d = bce(model.D(real_emb), torch.full((len(u), 1), 0.9, device=device)) + \
                     bce(model.D(fake_emb_detach), torch.zeros((len(u), 1), device=device))
            loss_d.backward();
            opt_D.step()

            # 3. Train G (Multi-Task)
            opt_G.zero_grad()

            fake_emb_i = model.G(bert_i)
            fake_emb_j = model.G(bert_j)

            loss_mse = mse(fake_emb_i, real_emb)
            loss_gan = bce(model.D(fake_emb_i), torch.ones((len(u), 1), device=device))

            u_e = model.user_mlp(model.user_emb(u))
            i_e_fake = model.item_mlp(fake_emb_i)
            j_e_fake = model.item_mlp(fake_emb_j)

            pos_score = (u_e * i_e_fake).sum(1)
            neg_score = (u_e * j_e_fake).sum(1)
            loss_bpr = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            loss_total = Config.W_BPR * loss_bpr + Config.W_MSE * loss_mse + Config.W_GAN * loss_gan

            loss_total.backward()
            opt_G.step()

            pbar.set_postfix({'BPR': f"{loss_bpr.item():.3f}", 'MSE': f"{loss_mse.item():.3f}"})

        # Eval (mode='gan')
        res = evaluate_full(model, test_data, n_items, device, 'gan')
        # curr_score = res.get('Cold-R@10', 0) + res.get('Cold-N@10', 0)

        # 指标总和早停
        curr_score = 0.0
        for k in Config.K_LIST:
            curr_score += res.get(f'Cold-R@{k}', 0.0)
            curr_score += res.get(f'Cold-N@{k}', 0.0)

        if curr_score > best_s2_score:
            best_s2_score = curr_score
            patience = 0
            torch.save(model.state_dict(), "gar_final_best.pth")
            print(f"   🌟 New S2 Best! Score: {best_s2_score:.4f}")
        else:
            patience += 1
            print(f"   ⏳ Patience: {patience}/{Config.PATIENCE}")
            if patience >= Config.PATIENCE:
                print("🛑 Stage 2 Early Stopping.");
                break

    print(f"✅ Final Best Cold Score (Stage 2): {best_s2_score:.4f}")
