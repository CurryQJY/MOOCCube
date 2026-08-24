import csv
import torch
import os

def calc(data):
    c_count = sum(float(r['Count_cold']) for r in data)
    h_count = sum(float(r['Count_hot']) for r in data)
    res = {}
    metrics = ['R@5','R@10','R@20','N@5','N@10','N@20']
    for m in metrics:
        if c_count > 0:
            res['cold_'+m] = sum(float(r['cold_'+m])*float(r['Count_cold']) for r in data) / c_count
        if h_count > 0:
            res['hot_'+m] = sum(float(r['hot_'+m])*float(r['Count_hot']) for r in data) / h_count
    return res

csv_file = 'mooc_metrics_usim_feedback.csv'
if os.path.exists(csv_file):
    reader = list(csv.DictReader(open(csv_file)))
    res = calc(reader)
    print("USIM_FEEDBACK (Sampled) Cold:")
    for m in ['R@5','R@10','R@20','N@5','N@10','N@20']:
        print(f"  {m}: {res.get('cold_'+m, 0.0):.4f}")
    print("\nUSIM_FEEDBACK (Sampled) Hot:")
    for m in ['R@5','R@10','R@20','N@5','N@10','N@20']:
        print(f"  {m}: {res.get('hot_'+m, 0.0):.4f}")
else:
    print(f"{csv_file} not found.")

ckpt_file = 'checkpoints/usim_feedback/finished.pt'
if os.path.exists(ckpt_file):
    try:
        cp = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        fc = cp.get('fc_cold', 0)
        fh = cp.get('fc_hot', 0)
        print("\nUSIM_FEEDBACK (Full Rank) Cold:")
        if fc > 0 and 'full_cold' in cp:
            for m in ['R@5','R@10','R@20','N@5','N@10','N@20']:
                print(f"  {m}: {cp['full_cold'][m]/fc:.4f}")
        print("\nUSIM_FEEDBACK (Full Rank) Hot:")
        if fh > 0 and 'full_hot' in cp:
            for m in ['R@5','R@10','R@20','N@5','N@10','N@20']:
                print(f"  {m}: {cp['full_hot'][m]/fh:.4f}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
else:
    print(f"{ckpt_file} not found.")
