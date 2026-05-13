import csv

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

f2 = list(csv.DictReader(open('mooc_metrics_usim_feedback_fast2.csv')))
f3 = list(csv.DictReader(open('mooc_metrics_usim_feedback_fast3.csv')))

res2 = calc(f2)
res3 = calc(f3)

print("FAST2 Results:")
for k, v in res2.items():
    print(f"  {k}: {v:.4f}")

print("\nFAST3 Results:")
for k, v in res3.items():
    print(f"  {k}: {v:.4f}")
