import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pam_model import PAM, Config
from train_eval import StreamDataset, collate_fn, split_dataframe_by_periods, evaluate


def run_experiment(exp_name, override_cfg, periods, meta, content_emb):
    print(f"\n{'=' * 30}\nRunning: {exp_name}\n{'=' * 30}")

    cfg = Config(meta['n_users'], meta['n_items'])
    cfg.content_dim = content_emb.shape[1]

    # 覆盖参数
    for k, v in override_cfg.items():
        setattr(cfg, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAM(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    # === Stage 0: Warm-up (对齐语义空间) ===
    if exp_name != "Random":
        print("  >> Stage 0: Warming up Projector...")
        proj_opt = torch.optim.Adam(model.con_proj.parameters(), lr=0.001)
        model.train()
        for ep in range(5):
            idx = torch.randperm(cfg.num_items)[:1024].to(device)
            target = model.item_beh_emb(idx).detach()
            txt = model.item_con_emb(idx)
            pred = model.con_proj(txt)
            loss = F.mse_loss(pred, target)
            proj_opt.zero_grad()
            loss.backward()
            proj_opt.step()
        print("  >> Warmup Done.")

    target_periods = periods
    dataloaders = [DataLoader(StreamDataset(p), batch_size=2048, shuffle=False, collate_fn=collate_fn) for p in
                   target_periods]

    WARMUP = 3

    # 定义所有要记录的指标
    metrics_keys = ['R@5', 'N@5', 'R@10', 'N@10', 'R@20', 'N@20']
    weighted_sums = {k: 0.0 for k in metrics_keys}
    total_cold_samples = 0

    for t, loader in enumerate(dataloaders):
        # --- Test ---
        if t >= WARMUP:
            if exp_name == "Random":
                n_cold = 0
                for _, pop in loader: n_cold += (pop < cfg.cold_threshold).sum().item()
                if n_cold > 0:
                    # 随机基线估算 (假设 total_items ≈ 3000)
                    # Recall ≈ K / N
                    rand_res = {
                        'R@5': 0.0017, 'N@5': 0.0008,
                        'R@10': 0.0033, 'N@10': 0.0015,
                        'R@20': 0.0067, 'N@20': 0.0025
                    }
                    for k in metrics_keys:
                        weighted_sums[k] += rand_res[k] * n_cold
                    total_cold_samples += n_cold
            else:
                # 模型测试 (evaluate 函数已支持返回所有 K 的指标)
                metrics, n = evaluate(model, loader, device)
                if metrics:
                    for k in metrics_keys:
                        weighted_sums[k] += metrics.get(k, 0.0) * n
                    total_cold_samples += n

        # --- Train ---
        if exp_name != "Random":
            model.train()
            for batch, pop in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pop = pop.to(device)

                loss, _ = model(batch, pop)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    # 计算加权平均
    final_results = {}
    if total_cold_samples > 0:
        for k in metrics_keys:
            final_results[k] = weighted_sums[k] / total_cold_samples
    else:
        for k in metrics_keys: final_results[k] = 0.0

    print(f"  >> Result: {final_results}")
    return final_results


def main():
    if not os.path.exists("processed_data/stream_data.pkl"):
        print("Run data_process.py first.")
        return

    with open("processed_data/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle("processed_data/stream_data.pkl")
    content_emb = torch.load("processed_data/content_emb.pt")
    periods = split_dataframe_by_periods(df, period_type='M')

    # === 对比实验配置 ===
    experiments = {
        "Random": {},

        # 1. Standard DL
        "Standard DL": {
            "inner_lr": 0.0,
            "gamma_s": 0.0,
            "lambda_cold": 1.0,
        },

        # 2. MeLU
        "MeLU": {
            "inner_lr": 0.01,
            "gamma_s": 0.0,
            "lambda_cold": 2.0,
        },

        # 3. PAM (Ours) - 最佳参数
        "PAM (Ours)": {
            "inner_lr": 0.01,
            "gamma_s": 0.1,  # 有 Warmup 后，低权重辅助效果最佳
            "lambda_cold": 2.0,
        }
    }

    all_results = {}
    for name, cfg in experiments.items():
        all_results[name] = run_experiment(name, cfg, periods, meta, content_emb)

    print("\n" + "=" * 60)
    print("             FINAL FULL METRICS COMPARISON             ")
    print("=" * 60)

    # 转为 DataFrame 并显示
    df_res = pd.DataFrame(all_results).T
    # 调整列顺序
    cols = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    df_res = df_res[cols]

    print(df_res)
    df_res.to_csv("final_comparison_full.csv")
    print("\n>> Table saved to final_comparison_full.csv")

    # --- 画图 (Recall 柱状图) ---
    df_res[['R@5', 'R@10', 'R@20']].plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title("Recall Performance Comparison (Top 5/10/20)")
    plt.ylabel("Recall")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("comparison_recall.png")

    # --- 画图 (NDCG 柱状图) ---
    plt.figure()
    df_res[['N@5', 'N@10', 'N@20']].plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title("NDCG Performance Comparison (Top 5/10/20)")
    plt.ylabel("NDCG")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("comparison_ndcg.png")

    print(">> Charts saved to comparison_recall.png and comparison_ndcg.png")


if __name__ == "__main__":
    main()