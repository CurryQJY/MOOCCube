import torch
import torch.optim as optim
import os
import argparse
import pickle
import numpy as np
from warm_model.bprmf import BPRMF
from dataloader.dataloader_1 import BPRDataLoader
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MOOCCubeX')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--factor_num', type=int, default=200)  # 必须与 USIM 一致
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.device = device  # 修正类型兼容性

    # 加载字典
    data_path = f'data/{args.dataset}'
    with open(os.path.join(data_path, 'convert_dict.pkl'), 'rb') as f:
        para_dict = pickle.load(f)

    # 初始化模型
    model = BPRMF(para_dict['user_num'], para_dict['item_num'], args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载数据
    train_loader = BPRDataLoader(args, 'warm_emb', para_dict)

    print("Start Pre-training MF Backbone...")
    for epoch in range(20):  # 跑20轮差不多了
        total_loss = 0
        cnt = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            loss = model.calculate_loss(batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

        print(f"Epoch {epoch}: Loss = {total_loss / cnt:.4f}")
        # 每轮都要重新负采样
        train_loader.negative_sampling()

    # 保存
    save_path = os.path.join(data_path, 'MF_backbone.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Saved backbone to {save_path}")


if __name__ == '__main__':
    main()