import logging
from turtledemo.forest import start
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import argparse
import random
import torch.optim as optim
from dgl.nn.pytorch import GATConv
import matplotlib.pyplot as plt
import time
import csv
import json


from model import SemanticAttention,HANLayer,HAN,BprMF,HybridModel
from utils import mrr, generate_data, hr,ndcg

def parse_args():
    parser = argparse.ArgumentParser(description="Run HAM")
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')
    parser.add_argument("--device",type=str,default="cuda:0"  if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (e.g., 'cpu' or 'cuda:0')"
    )
    parser.add_argument("--in_size", type=int, default=128, help="Input size for the model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size for the model")
    parser.add_argument("--num_heads", type=int, nargs='+', default=[8,8], help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("--reg", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha parameter for model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--graph_path", type=str, default='./graph/train_heterograph.bin',
                        help="Path to the graph file")
    parser.add_argument("--user_course_features_path", type=str, default='./data/user_course_features.npy',
                        help="Path to the user-course features file")
    parser.add_argument("--user_course_matrix_path", type=str, default='./data/train_uc.npy',
                        help="Path to the user-course matrix file")

    parser.add_argument("--val_user_course_matrix_path", type=str, default='./data/val_uc.npy', help="Path to the validation user-course matrix file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    #seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    in_size = args.in_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    dropout = args.dropout
    reg = args.reg
    alpha = args.alpha
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    graphs = args.graph_path
    user_course_features_path = args.user_course_features_path
    user_course_matrix_path = args.user_course_matrix_path
    val_user_course_matrix_path = args.val_user_course_matrix_path

    dim = in_size * num_heads[0]
    graphs,_ =dgl.load_graphs(graphs) 
    g = graphs[0]
    g = g.to(device)
    print("device:", g.device)
    user_features = torch.tensor(np.load(user_course_features_path), dtype=torch.float32).to(device)
    user_course_matrix = np.load(user_course_matrix_path)
    n_user = user_course_matrix.shape[0]
    m_item = user_course_matrix.shape[1]
    meta_paths=[['uc', 'cu'], ['uv', 'vu'], ['uc', 'ck', 'kc', 'cu'], ['uv', 'vk', 'kv', 'vu']]
    allPos = [np.where(user_course_matrix[user] == 1)[0] for user in range(n_user)]
    testPos = [np.random.choice(pos) for pos in allPos if len(pos) > 0]
    
    # Load validation data
    val_user_course_matrix = np.load(val_user_course_matrix_path)
    n_val_user = val_user_course_matrix.shape[0]
    allValPos = [np.where(val_user_course_matrix[user] == 1)[0] for user in range(n_val_user)]
    valTestPos = [np.random.choice(pos) for pos in allValPos if len(pos) > 0]
    
    han_model = HAN(meta_paths, in_size, hidden_size, num_heads, dropout).to(device)
    bprmf_model = BprMF(n_user, m_item, dim, reg).to(device)
    hybrid_model = HybridModel(han_model, bprmf_model, alpha).to(device)
    
    optimizer = optim.Adam(hybrid_model.parameters(), lr, betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 调整学习率调度器

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_epoch = -1
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        hybrid_model.train()
        total_loss = 0
        for users, pos_items, neg_items in generate_data(n_user, m_item, allPos, batch_size):
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss = hybrid_model(g, user_features, users, pos_items, neg_items)
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        avg_loss = total_loss / (n_user / batch_size)
        train_losses.append(avg_loss)

        # ---------------- Validation ----------------
        hybrid_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for users, pos_items, neg_items in generate_data(n_val_user, m_item, allValPos, batch_size):
                users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
                val_loss = hybrid_model(g, user_features, users, pos_items, neg_items)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / (n_val_user / batch_size)
        val_losses.append(avg_val_loss)

        end_time = time.time()
        epoch_time = end_time - start_time

        
        mrr10 = mrr(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=20, batch_size=batch_size)
        hr1 = hr(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=1)
        hr5 = hr(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=5)
        hr10 = hr(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=10)
        hr20 = hr(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=20)
        ndcg5 = ndcg(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=5)
        ndcg10 = ndcg(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=10)
        ndcg20 = ndcg(n_val_user, m_item, allValPos, valTestPos, hybrid_model, device, k=20)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, MRR@10: {mrr10:.4f}, "
              f"HR@1: {hr1:.4f}, HR@5: {hr5:.4f}, HR@10: {hr10:.4f}, HR@20: {hr20:.4f}, "
              f"NDCG@5: {ndcg5:.4f}, NDCG@10: {ndcg10:.4f}, NDCG@20: {ndcg20:.4f}")
        print("val_loss:", avg_val_loss)
        print("time:", epoch_time)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
            reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
            print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, MRR@10: {mrr10:.4f}, "
                     f"HR@1: {hr1:.4f}, HR@5: {hr5:.4f}, HR@10: {hr10:.4f}, HR@20: {hr20:.4f}, "
                     f"NDCG@5: {ndcg5:.4f}, NDCG@10: {ndcg10:.4f}, NDCG@20: {ndcg20:.4f}")

        
        if epoch > 40:
            torch.save(hybrid_model, './model/hybrid_model_complete{}.pth'.format(epoch + 1))
            torch.save(hybrid_model.state_dict(), './state/hybrid_model_state_dict_epoch{}.pth'.format(epoch + 1))

    print(f"Best Epoch: {best_epoch}, Best Training Loss: {best_train_loss:.4f}, Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
