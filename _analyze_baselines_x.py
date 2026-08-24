"""汇总 MOOCCubeX 所有基线实验结果"""
import json, os, glob
import pandas as pd

# --- 文件名到 (model, protocol) 的映射 ---
NAME_MAP = {
    "drop_static": ("DropoutNet", "static"),
    "drop_full":   ("DropoutNet", "stream"),
    "gar_static":  ("GAR", "static"),
    "gar_full":    ("GAR", "stream"),
    "lightgcl_static": ("LightGCL", "static"),
    "lightgcl_full":   ("LightGCL", "stream"),
    "lightgcn_static": ("LightGCN", "static"),
    "lightgcn_full":   ("LightGCN", "stream"),
    "sasrec_static":   ("SASRec", "static"),
    "sasrec_full":     ("SASRec", "stream"),
    "hhcor_static":    ("HHCoR", "static"),
    "hhcor_full":      ("HHCoR", "stream"),
    "light_path_static": ("LightPath", "static"),
    "light_path_full":   ("LightPath", "stream"),
}

def normalize_record(fp, data):
    """统一不同 JSON 格式为 flat dict"""
    if isinstance(data, list):
        data = data[0]

    stem = os.path.splitext(os.path.basename(fp))[0].replace("_result", "")
    model, protocol = NAME_MAP.get(stem, (data.get("model", "?"), data.get("protocol", "?")))

    rec = {"model": model, "protocol": protocol, "_file": fp}

    # 格式 A: flat keys like samp_cold_R@10
    if "samp_cold_R@10" in data or "samp_cold_R@5" in data:
        for k, v in data.items():
            if k not in ("model", "protocol", "_file"):
                rec[k] = v
    # 格式 B: nested dicts like {"sample_cold": {"R@5": ...}, "full_cold": {...}}
    elif "sample_cold" in data or "full_cold" in data:
        mapping = {"sample_cold": "samp_cold", "sample_hot": "samp_hot",
                   "full_cold": "full_cold", "full_hot": "full_hot"}
        for src_key, dst_prefix in mapping.items():
            sub = data.get(src_key, {})
            if isinstance(sub, dict):
                for mk, mv in sub.items():
                    rec[f"{dst_prefix}_{mk}"] = mv
    return rec


# 收集所有 JSON 结果
results = []
for fp in sorted(glob.glob("*_result.json")):
    with open(fp, "r") as f:
        data = json.load(f)
    results.append(normalize_record(fp, data))

if not results:
    print("No result JSON files found!")
    exit()

df = pd.DataFrame(results)
print(f"Found {len(df)} result files:\n")

# 检测 GAN collapse: 所有指标几乎相同
for idx, row in df.iterrows():
    vals = [row.get(c, 0) for c in ["full_cold_R@5", "full_cold_R@10", "full_cold_R@20"] if c in row]
    if len(vals) >= 2 and vals[0] > 0.5 and abs(vals[0] - vals[1]) < 0.001:
        print(f"  [!] {row['model']}-{row['protocol']}: GAN collapse detected (指标异常), 标记为无效")
        for c in df.columns:
            if c.startswith("samp_") or c.startswith("full_"):
                df.at[idx, c] = float('nan')

# 核心指标: Full Ranking (论文常用)
core_metrics = ["full_cold_R@10", "full_cold_R@20", "full_cold_N@10",
                "full_hot_R@10",  "full_hot_R@20",  "full_hot_N@10"]

print("\n" + "=" * 100)
print("  MOOCCubeX Baseline Results — Full Ranking (主指标)")
print("=" * 100)

header = f"{'Model':<12} {'Protocol':<8}"
for m in core_metrics:
    short = m.replace("full_", "").replace("cold_", "C-").replace("hot_", "H-")
    header += f" {short:>8}"
print(header)
print("-" * 100)

for _, row in df.iterrows():
    line = f"{row['model']:<12} {row['protocol']:<8}"
    for m in core_metrics:
        v = row.get(m, 0)
        if pd.isna(v):
            line += f" {'N/A':>8}"
        else:
            line += f" {v:>8.4f}"
    print(line)

print("-" * 100)

# 采样指标
samp_metrics = ["samp_cold_R@10", "samp_cold_R@20", "samp_cold_N@10",
                "samp_hot_R@10",  "samp_hot_R@20",  "samp_hot_N@10"]

print("\n" + "=" * 100)
print("  MOOCCubeX Baseline Results — Sampled Ranking")
print("=" * 100)

header = f"{'Model':<12} {'Protocol':<8}"
for m in samp_metrics:
    short = m.replace("samp_", "").replace("cold_", "C-").replace("hot_", "H-")
    header += f" {short:>8}"
print(header)
print("-" * 100)

for _, row in df.iterrows():
    line = f"{row['model']:<12} {row['protocol']:<8}"
    for m in samp_metrics:
        v = row.get(m, 0)
        if pd.isna(v):
            line += f" {'N/A':>8}"
        else:
            line += f" {v:>8.4f}"
    print(line)

print("-" * 100)

# 排名分析 (排除 NaN)
print("\n" + "=" * 60)
print("  Full Ranking R@10 排名 (排除无效结果)")
print("=" * 60)

valid = df.dropna(subset=["full_cold_R@10", "full_hot_R@10"])
for group_name, col in [("Cold-Start", "full_cold_R@10"), ("Hot (Popular)", "full_hot_R@10")]:
    print(f"\n>> {group_name}:")
    sub = valid[["model", "protocol", col]].copy().sort_values(col, ascending=False)
    for rank, (_, row) in enumerate(sub.iterrows(), 1):
        print(f"  {rank}. {row['model']:<12} {row['protocol']:<8}  R@10={row[col]:.4f}")
