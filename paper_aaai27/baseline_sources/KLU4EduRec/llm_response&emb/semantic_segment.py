import torch
from tqdm import tqdm
import argparse
import numpy as np
import json

def semantic_aware_motivation_segments(dataset, mode='train', percent='75%', threshold=None):
    embed_file = f'../data/{dataset}/resource_summary_embeddings_qwen3.pt'
    seqinfo_file = f'../data/{dataset}/user_behavior_seq_info_{mode}.txt'
    segment_file = f'../data/{dataset}/user_behavior_seq_info_{mode}_segment_{percent}.json'
    alpha=0.3
    omega=5

    user_seqinfo = {}
    with open(seqinfo_file, 'r') as f:
        for uid, line in enumerate(f):
            items = eval(line.strip())
            user_seqinfo[uid] = items
    print(f'Loaded {len(user_seqinfo)} user sequences.')

    embeddings = torch.load(embed_file, weights_only=True).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f'Loaded embeddings shape: {embeddings.shape}')

    if mode=='train' or threshold is None:
        all_deltas, all_Deltas = [], []
        for user_id, seq in tqdm(user_seqinfo.items(), desc='Calculating semantic deltas'):
            smoothed_deltas = 0.0
            queue = []
            for t in range(1, len(seq)):
                e_cur = embeddings[seq[t][0]]
                e_prev = embeddings[seq[t - 1][0]]
                delta = 1 - np.dot(e_cur, e_prev)
                smoothed_deltas = (1 - alpha) * smoothed_deltas + alpha * delta
                queue.append(smoothed_deltas)
                if len(queue) > omega:
                    queue.pop(0)
                Delta_t = sum(queue)
                all_deltas.append(delta)
                all_Deltas.append(Delta_t)
                
        all_deltas = np.array(all_deltas)
        delta_stats = {
            'mean': float(np.mean(all_deltas)),
            'std': float(np.std(all_deltas)),
            'min': float(np.min(all_deltas)),
            '25%': float(np.percentile(all_deltas, 25)),
            '50%': float(np.percentile(all_deltas, 50)),
            '75%': float(np.percentile(all_deltas, 75)),
            'max': float(np.max(all_deltas)),
        }

        print('\nInstant semantic changes δ_t statistics:')
        for k, v in delta_stats.items():
            print(f'{k}: {v:.4f}', end=', ')
        print()

        all_Deltas = np.array(all_Deltas)
        Delta_stats = {
            'mean': float(np.mean(all_Deltas)),
            'std': float(np.std(all_Deltas)),
            'min': float(np.min(all_Deltas)),
            '25%': float(np.percentile(all_Deltas, 25)),
            '50%': float(np.percentile(all_Deltas, 50)),
            '75%': float(np.percentile(all_Deltas, 75)),
            'max': float(np.max(all_Deltas)),
        }
        print('\nCumulative semantic changes Δ_t statistics:')
        for k, v in Delta_stats.items():
            print(f'{k}: {v:.4f}', end=', ')
        print()
        threshold = Delta_stats[percent]


    if mode=='valid':
        user_max_seg_id = {}
        train_file = f'../data/{dataset}/user_behavior_seq_info_train_segment_{percent}.json'
        with open(train_file, 'r') as f:
            train_segments = json.load(f)
        for uid, segments in train_segments.items():
            user_max_seg_id[int(uid)] = max([int(seg_id) for seg_id in segments.keys()])

    segment_counts, segment_results = [], {}
    all_llm_calls = 0
    for uid, seq in tqdm(user_seqinfo.items(), desc='Calculating semantic segments'):
        segments = {}
        seg_id = 0 if mode=='train' else user_max_seg_id[uid] + 1
        segments[seg_id] = [seq[0]]

        smoothed = 0.0
        queue = []
        for t in range(1, len(seq)):
            e_cur = embeddings[seq[t][0]]
            e_prev = embeddings[seq[t - 1][0]]
            delta = 1 - np.dot(e_cur, e_prev)
            smoothed = (1 - alpha) * smoothed + alpha * delta
            queue.append(smoothed)
            if len(queue) > omega:
                queue.pop(0)
            Delta_t = sum(queue)
            if Delta_t > threshold:
                segments[seg_id] = str(segments[seg_id])
                seg_id += 1
                segments[seg_id] = []
                queue = []
                smoothed = 0.0
            segments[seg_id].append(seq[t])
        segments[seg_id] = str(segments[seg_id])
        segment_results[uid] = segments
        segment_counts.append(len(segments))
        all_llm_calls += len(segments)
    values = np.array(segment_counts)
    segment_stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        '25%': np.percentile(values, 25),
        '50%': np.percentile(values, 50),
        '75%': np.percentile(values, 75),
        'max': np.max(values)
    }
    print('\nSemantic segment count statistics based on cumulative drift:')
    for k, v in segment_stats.items():
        print(f'{k}: {v:.2f}', end=', ')
    print(f'\nTotal LLM calls needed for semantic-aware segmentation: {all_llm_calls}')

    with open(segment_file, 'w') as f:
        json.dump(segment_results, f, ensure_ascii=False, indent=2)

def get_pure_segments(dataset, mode='train', percent='75%'):
    segment_file = f'../data/{dataset}/user_behavior_seq_info_{mode}_segment_{percent}.json'
    with open(segment_file, 'r') as f:
        segment_results = json.load(f)

    segment_dict = {}
    for uid, segments in segment_results.items():
        segment_dict[uid] = {}
        for seg_id, segs in segments.items():
            segment_dict[uid][int(seg_id)] = []
            segs = eval(segs)
            for row in segs:
                item_id = int(row[0]) - 1
                segment_dict[uid][int(seg_id)].append(item_id)
    output_file = f'../data/{dataset}/user_segments_{mode}_{percent}.json'
    with open(output_file, 'w') as f:
        json.dump(segment_dict, f, ensure_ascii=False, indent=2)
    print(f'Pure segments saved to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mooccubex', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--percent', type=str, default='75%', help='Percentile for cumulative drift threshold')
    args = parser.parse_args()

    threshold = {
        'mooccubex': {'75%': 2.0230, '50%': 1.7314},
        'mooccube': {'75%': 2.4187, '50%': 2.1086},
    }

    semantic_aware_motivation_segments(args.dataset, mode='train', percent=args.percent, threshold=None)
    get_pure_segments(args.dataset, mode='train', percent=args.percent)