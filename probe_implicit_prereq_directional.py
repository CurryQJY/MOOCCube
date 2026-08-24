"""数据探针 2：给隐式行为先修加"方向性过滤"，区分真先修 vs 纯共现。

只读数据、不碰模型。回答：滤掉纯共现噪声后，隐式信号还剩多少新增覆盖 + 质量如何。

核心思想（LIGHT 式）：
  真先修 A->B  应该是"方向不对称"的：学 A 后学 B 的概率，明显高于学 B 后学 A。
  纯共现（同主题并列/流行课）：A->B 和 B->A 差不多，对称，应滤掉。

方向性得分（非对称性）：
  给有序对 (a,b)：n_ab = #(a 在 b 之前), n_ba = #(b 在 a 之前)
  条件概率  p(a before b | 两者共现) = n_ab / (n_ab + n_ba)
  不对称度  asym = n_ab / (n_ab + n_ba)   （>0.5 表示 a 更像 b 的先修）
  同时要求最小支持度 n_ab >= MIN_SUP，避免低频噪声。
  保留 asym >= ASYM_TH 的 a 作为 b 的隐式先修候选。
"""
from __future__ import annotations
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path("processed_data_hin_clean_pop5")
REL_DIR = Path("MOOCCube/relations")
TOPK = 10
MIN_SUP = 5          # a->b 最小共现次数
ASYM_TH = 0.60       # 方向不对称阈值，>0.5 才算有方向；0.60 滤掉近似对称的纯共现


def load_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t")
            if len(p) >= 2:
                out.append((p[0], p[1]))
    return out


# ---------- 0. id 映射 + 受课日志 ----------
with open(DATA_DIR / "stream_data.pkl", "rb") as f:
    stream = pickle.load(f)
df = stream
c2i = dict(zip(df["course_id"], df["i_idx"].astype(int)))
n_courses = len(c2i)
print(f"n_courses={n_courses}, n_interactions={len(df)}")

# ---------- 1. 人工先修（同 build_prereq_index）----------
course_concepts = defaultdict(set)
concept_courses = defaultdict(set)
for c, k in load_pairs(REL_DIR / "course-concept.json"):
    if c in c2i:
        course_concepts[c].add(k)
        concept_courses[k].add(c)
concept_prereqs = defaultdict(set)
for a, b in load_pairs(REL_DIR / "prerequisite-dependency.json"):
    concept_prereqs[b].add(a)
concept_df = {k: len(cs) for k, cs in concept_courses.items()}


def idf(p):
    dfp = concept_df.get(p, 0)
    return math.log(1.0 + n_courses / dfp) if dfp > 0 else 0.0


manual_prereq = {}
for c in c2i:
    my = course_concepts.get(c, set())
    if not my:
        continue
    pre_con = set()
    for k in my:
        pre_con |= concept_prereqs.get(k, set())
    pre_con -= my
    if not pre_con:
        continue
    score = defaultdict(float)
    for p in pre_con:
        w = idf(p)
        if w <= 0:
            continue
        for c2 in concept_courses.get(p, set()):
            if c2 != c:
                score[c2] += w
    if score:
        topk = [c2 for c2, _ in sorted(score.items(), key=lambda x: -x[1])[:TOPK]]
        manual_prereq[c] = set(topk)
n_manual = len(manual_prereq)

# ---------- 2. 有序共现计数 ----------
seq = df.sort_values(["u_idx", "timestamp"])
before_counts = defaultdict(lambda: defaultdict(int))  # b -> {a: n(a before b)}
grouped = seq.groupby("u_idx")["course_id"].apply(list)
for courses in grouped:
    seen = set()
    order = []
    for c in courses:
        if c not in seen:
            order.append(c)
            seen.add(c)
    for j, b in enumerate(order):
        for a in order[:j]:
            if a != b:
                before_counts[b][a] += 1

# ---------- 3a. 无方向过滤(旧口径, 频次topk) ----------
implicit_raw = {}
for b, amap in before_counts.items():
    if b not in c2i:
        continue
    topk = [a for a, _ in sorted(amap.items(), key=lambda x: -x[1])[:TOPK] if a in c2i]
    if topk:
        implicit_raw[b] = set(topk)

# ---------- 3b. 方向性过滤 ----------
implicit_dir = {}
edge_kept = 0
edge_total = 0
for b, amap in before_counts.items():
    if b not in c2i:
        continue
    cand = []
    for a, n_ab in amap.items():
        if a not in c2i:
            continue
        n_ba = before_counts.get(a, {}).get(b, 0)
        denom = n_ab + n_ba
        if denom == 0:
            continue
        edge_total += 1
        if n_ab < MIN_SUP:
            continue
        asym = n_ab / denom
        if asym >= ASYM_TH:
            cand.append((a, n_ab, asym))
            edge_kept += 1
    if cand:
        # 按支持度排序取 topk
        cand.sort(key=lambda x: -x[1])
        implicit_dir[b] = set(a for a, _, _ in cand[:TOPK])

# ---------- 4. 覆盖对比 ----------
def cov(d):
    return len(d), 100 * len(d) / n_courses

print("\n=== 覆盖率对比 ===")
print(f"人工先修:            {cov(manual_prereq)[0]}/{n_courses} = {cov(manual_prereq)[1]:.1f}%")
print(f"隐式(无方向过滤):    {cov(implicit_raw)[0]}/{n_courses} = {cov(implicit_raw)[1]:.1f}%")
print(f"隐式(方向过滤后):    {cov(implicit_dir)[0]}/{n_courses} = {cov(implicit_dir)[1]:.1f}%")
print(f"方向过滤保留边比例:  {edge_kept}/{edge_total} = {100*edge_kept/max(1,edge_total):.1f}%")

# 方向过滤后隐式对人工的净增覆盖
ms = set(manual_prereq)
only_dir = set(implicit_dir) - ms
union_dir = ms | set(implicit_dir)
print(f"\n方向过滤后隐式净增覆盖: +{len(only_dir)} 门课 = +{100*len(only_dir)/n_courses:.1f} 个百分点")
print(f"人工 ∪ 方向隐式 并集覆盖: {len(union_dir)}/{n_courses} = {100*len(union_dir)/n_courses:.1f}%")

# ---------- 5. 信号质量：方向隐式与人工的一致性 ----------
both = ms & set(implicit_dir)
if both:
    jac, prec = [], []
    for c in both:
        m, im = manual_prereq[c], implicit_dir[c]
        u = m | im
        jac.append(len(m & im) / len(u) if u else 0.0)
        prec.append(len(m & im) / len(im) if im else 0.0)
    print(f"\n重叠课程({len(both)}门)上:")
    print(f"  方向隐式 vs 人工 Jaccard: mean={np.mean(jac):.3f} median={np.median(jac):.3f}")
    print(f"  方向隐式∩人工/方向隐式 (隐式命中人工的比例): mean={np.mean(prec):.3f}")
    print("  (该比例上升=方向过滤后隐式更接近人工判定的真先修)")

# ---------- 6. 用人工先修当弱标签，评方向过滤是否提高精度 ----------
# 对所有有人工先修的课，看隐式候选(过滤前/后)命中人工先修的比例
def hit_rate(impl):
    hits, tot = 0, 0
    for c in ms:
        if c not in impl:
            continue
        m = manual_prereq[c]
        for a in impl[c]:
            tot += 1
            if a in m:
                hits += 1
    return hits, tot, (100 * hits / tot if tot else 0.0)

hr_raw = hit_rate(implicit_raw)
hr_dir = hit_rate(implicit_dir)
print("\n=== 以人工先修为弱标签，隐式候选命中率(precision proxy) ===")
print(f"  无方向过滤: {hr_raw[0]}/{hr_raw[1]} = {hr_raw[2]:.1f}%")
print(f"  方向过滤后: {hr_dir[0]}/{hr_dir[1]} = {hr_dir[2]:.1f}%")
print("  (方向过滤后命中率上升 => 过滤确实提高了信号质量)")
