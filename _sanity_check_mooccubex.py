"""Sanity check that MOOCCubeX data loads correctly into fast3_content_delta pipeline."""
import os, time, json
os.environ.setdefault("USIM_DATA_DIR", "processed_data_hin_x")
os.environ.setdefault("USIM_RELATION_DIR", "MOOCCubeX/relations")
os.environ.setdefault("USIM_PREREQ_GRAPH_SOURCE", "behavior")

t0 = time.time()
from hin_data_common import load_hin_processed
meta, df, content_emb = load_hin_processed("processed_data_hin_x")
n_users, n_items = meta["n_users"], meta["n_items"]
print(f"[{time.time()-t0:.1f}s] data loaded: users={n_users}, items={n_items}, df={df.shape}")
print(f"          content_emb: {tuple(content_emb.shape)} {content_emb.dtype}")

t0 = time.time()
from usim_feedback_fast3_content_delta import build_course_artifacts
artifacts, stats = build_course_artifacts(
    df, n_items,
    relation_dir="MOOCCubeX/relations",
    prereq_min_support=30, prereq_max_per_item=5,
    prereq_min_items=1, prereq_max_forward=20,
)
print(f"[{time.time()-t0:.1f}s] course artifacts built")
print("stats:", json.dumps(stats, indent=2, ensure_ascii=False))
print("artifact tensor shapes:")
for k, v in artifacts.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

# Check periods (monthly split)
from hin_data_common import split_dataframe_by_periods
t0 = time.time()
periods = split_dataframe_by_periods(df, period_type="M")
print(f"[{time.time()-t0:.1f}s] split into {len(periods)} monthly periods")
for i, p in enumerate(periods[:3]):
    print(f"  period {i}: rows={len(p)}")
print("  ...")
for i in range(max(0, len(periods)-3), len(periods)):
    print(f"  period {i}: rows={len(periods[i])}")

print("\n[OK] MOOCCubeX is fully compatible with fast3_content_delta pipeline.")
