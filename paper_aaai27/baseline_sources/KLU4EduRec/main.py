import os
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model.GNNRec import LLM4EduRec
from model.dataset import EduDataset
from model.utils import set_seed, draw_trajectory
from model.early_stops import EarlyStopping

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', '--cuda', default=1, type=int)
    parser.add_argument('-seed', '--seed', default=2026, type=int)
    parser.add_argument('-lr', '--lr', default=1e-2, type=float)
    parser.add_argument('-wd', '--wd', default=1e-4, type=float)
    parser.add_argument('-ep', '--epochs', default=1000, type=int)
    parser.add_argument('-pat', '--patiences', default=20, type=int)
    
    parser.add_argument('-d', '--dataset', default='mooccubex', type=str)
    parser.add_argument('--train_batch_size', default=8192, type=int)
    parser.add_argument('--eval_step', default=1, type=int)
    
    parser.add_argument('--embed_size', default=64, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--node_drop', default=0, type=float)
    parser.add_argument('--edge_drop', default=0, type=float)
    parser.add_argument('--add_self_loops', default=True, type=bool)

    parser.add_argument('--mode', type=str, default='both', choices=['base', 'item_se', 'user_se', 'both'])

    parser.add_argument('-iff', '--item_fusion_func', type=str, default='gating', choices=['add', 'gating', 'concat', 'attention'])
    parser.add_argument('--item_temp', default=0.5, type=float)
    parser.add_argument('--item_loss_reg', default=1e-2, type=float)

    parser.add_argument('--user_segments_type', type=str, default='knowseg', choices=['allseq', 'knowseg'])
    parser.add_argument('--seg_know_agg', type=str, default='att', choices=['mean', 'att'])
    parser.add_argument('--seg_agg', type=str, default='rnn', choices=['mean', 'rnn'])
    parser.add_argument('-uff', '--user_fusion_func', type=str, default='gating', choices=['add', 'gating', 'concat', 'attention'])
    parser.add_argument('--user_temp', default=0.5, type=float)
    parser.add_argument('--user_loss_reg', default=5e-4, type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    config_file = f'./config/{args.dataset}.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_params = json.load(f)
            for key, value in config_params.items():
                setattr(args, key, value)
    print(f'Arguments: {args}')
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}'if torch.cuda.is_available() else 'cpu')
    base_param_str = f'lr{args.lr}_wd{args.wd}_d{args.embed_size}_nl{args.n_layers}'
    if args.mode == 'base':
        args.saved_dir = f'./data/model_results/{args.dataset}/base_{base_param_str}'
    elif args.mode == 'item_se':
        item_param_str = f'i-summary-{args.item_fusion_func}-reg{args.item_loss_reg}-t{args.item_temp}'
        args.saved_dir = f'./data/model_results/{args.dataset}/item_{base_param_str}_{item_param_str}'
    elif args.mode == 'user_se':
        if args.user_segments_type == 'allseq':
            user_param_str = f'u-allseq-{args.user_fusion_func}-reg{args.user_loss_reg}-t{args.user_temp}'
        else:
            user_param_str = f'u-knowseg-{args.seg_know_agg}-{args.seg_agg}-{args.user_fusion_func}-reg{args.user_loss_reg}-t{args.user_temp}'
        args.saved_dir = f'./data/model_results/{args.dataset}/user_{base_param_str}_{user_param_str}'
    else:
        item_param_str = f'i-summary-{args.item_fusion_func}-reg{args.item_loss_reg}-t{args.item_temp}'
        if args.user_segments_type == 'allseq':
            user_param_str = f'u-allseq-{args.user_fusion_func}-reg{args.user_loss_reg}-t{args.user_temp}'
        else:
            user_param_str = f'u-knowseg-{args.seg_know_agg}-{args.seg_agg}-{args.user_fusion_func}-reg{args.user_loss_reg}-t{args.user_temp}'
        args.saved_dir = f'./data/model_results/{args.dataset}/both_{base_param_str}_{item_param_str}_{user_param_str}'
    os.makedirs(args.saved_dir, exist_ok=True)
    args.saved_model_file = f'{args.saved_dir}/model_{args.seed}.pth'
    args.saved_result_file = f'{args.saved_dir}/result.json'
    print(f'Model and results will be saved to {args.saved_dir}')

   
    input_path = f'./data/{args.dataset}'
    dataset = EduDataset(input_path, args.dataset)
    TrainDataLoader = dataset.get_train_loader(batch_size=args.train_batch_size)
    args.n_users = dataset.n_users
    args.m_items = dataset.m_items

    graph = dataset.get_graph().to(device)


    if args.mode in ['user_se', 'item_se', 'both']:
        item_embed_file = f'{input_path}/resource_summary_embeddings_qwen3.pt'
        args.pretrained_item_embeddings = torch.load(item_embed_file, weights_only=True)[1:].to(device)
        args.item_LLM_emb_dim = args.pretrained_item_embeddings.size(1)
        print(f'Loaded item summary embeddings, shape: {args.pretrained_item_embeddings.size()}')
    
    if args.mode in ['user_se', 'both']:
        user_embed_file = f'{input_path}/user_summary_embeddings_qwen3_{args.user_segments_type}.pt'
        args.pretrained_user_embeddings = torch.load(user_embed_file, weights_only=True).to(device)
        args.user_LLM_emb_dim = args.pretrained_user_embeddings.size(1)
        print(f'Loaded user summary embeddings, type: {args.user_segments_type}, shape: {args.pretrained_user_embeddings.size()}')

        if args.user_segments_type == 'knowseg':
            all_user_ids, all_item_ids, all_seg_ids, seg_user_ids, seg_ptr, user_ptr = dataset.get_user_segments()
            args.segment_index_dict = {"user_ids": all_user_ids.to(device), "item_ids": all_item_ids.to(device), "seg_ids": all_seg_ids.to(device), 
                                        "seg_users": seg_user_ids.to(device), "seg_ptr": seg_ptr.to(device), "user_ptr": user_ptr.to(device)}


    model = LLM4EduRec(args, graph.edge_index, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patiences, higher_is_better=True, saved_model_file=args.saved_model_file)
    

    epoch_trajectory = []
    epoch_bar = tqdm(range(0, args.epochs), desc="Training", leave=True)
    for epoch_idx in epoch_bar:
        model.train()
        batch_loss = 0
        batch_bar = tqdm(enumerate(TrainDataLoader), total=len(TrainDataLoader), desc=f"Batch", leave=False)
        for batch_idx, (batch_user, batch_pos, batch_neg) in batch_bar:
            batch_user, batch_pos, batch_neg = batch_user.to(device), batch_pos.to(device), batch_neg.to(device)
            rec_loss = model.calculate_loss(batch_user, batch_pos, batch_neg)
            
            optimizer.zero_grad()
            rec_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss += rec_loss.item()
        
        if epoch_idx % args.eval_step == 0:
            model.eval()
            with torch.no_grad():
                test_user, test_pos, test_negatives = dataset.get_test_data()
                test_user = test_user.to(device)
                rec, pre, ndcg, mrr, _ = model.compute_all_ranking(test_user, test_pos, test_negatives)
            should_stop, cur_patience = early_stopping.check(ndcg[10], model)
            if should_stop:
                print(f'Early stopping at epoch {epoch_idx} with best NDCG@10: {early_stopping.best_score:.4f}')
                break
        epoch_trajectory.append([batch_loss, rec[10], pre[10], ndcg[10], mrr[10]])
        epoch_bar.set_description(f'Epoch {epoch_idx}, patience={cur_patience}, Loss={batch_loss}, Recall@10={rec[10]:.4f}, Precision@10={pre[10]:.4f}, NDCG@10={ndcg[10]:.4f}, MRR@10={mrr[10]:.4f}')
    
    checkpoint = torch.load(args.saved_model_file, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])        
    model.eval()
    with torch.no_grad():
        test_user, test_pos, test_negatives = dataset.get_test_data()
        rec, pre, ndcg, mrr, all_user_metrics = model.compute_all_ranking(test_user, test_pos, test_negatives)
    print(f'\n{args.dataset} metric: Recall@5: {rec[5]:.4f}, Recall@10: {rec[10]:.4f}, NDCG@5: {ndcg[5]:.4f}, NDCG@10: {ndcg[10]:.4f}, MRR@10: {mrr[10]:.4f}')
    
    test_result = [rec[5], rec[10], rec[20], pre[5], pre[10], pre[20], ndcg[5], ndcg[10], ndcg[20], mrr[5], mrr[10], mrr[20]]
    if not os.path.exists(args.saved_result_file):
        result = {}
    else:
        with open(args.saved_result_file, 'r', encoding='utf-8') as f:
            try:
                result = json.load(f)
            except:
                result = {}
    result[str(args.seed)] = str([round(x, 4) for x in test_result])
    with open(args.saved_result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


    loss_file = f'{args.saved_dir}/train_trajectory_seed{args.seed}.txt'
    with open(loss_file, 'w') as f:
        for row in epoch_trajectory[:-1]:
            f.write(','.join([str(val) for val in row]) + '\n')
    trajectory_fig = f'{args.saved_dir}/train_trajectory_fig.jpg'
    draw_trajectory(epoch_trajectory, trajectory_fig)

    user_metric_file = f'{args.saved_dir}/user_metrics_seed{args.seed}.txt'
    with open(user_metric_file, 'w', encoding='utf-8') as f:
        for user_id in range(dataset.n_users):
            metrics = all_user_metrics[user_id]
            f.write(f"{user_id},{metrics[0]},{metrics[1]},{metrics[2]:.4f},{metrics[3]:.4f},{metrics[4]:.4f},{metrics[5]:.4f},{metrics[6]:.4f}\n")
    print(f'Training and evaluation complete. saved to {args.saved_dir}\n')
