"""数据探针：隐式行为先修信号 vs 现行人工先修信号，是否有增量覆盖。

只读数据、不碰模型、不重跑训练。回答一个问题：
  从受课日志(时间序)统计课程转移，能给"无人工先修"的课程补多少覆盖？

链路对照 build_prereq_index.py：
  人工: course->concepts->prereq concepts->covering courses (topk)
  隐式: 受课日志中，在目标课 c 之前被同一学习者学过的课程(时间序) -> 频次/条件概率 topk
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

# stream 是 DataFrame: user_id, course_id, timestamp, u_idx, i_idx, popularity
df = stream
c2i = dict(zip(df["course_id"], df["i_idx"].astype(int)))
i2c = {i: c for c, i in c2i.items()}
n_items = int(df["i_idx"].max() + 1)
print(f"n_items={n_items}, n_interactions={len(df)}")

# ---------- 1. 现行人工先修覆盖率 ----------
course_concepts = defaultdict(set)
concept_courses = defaultdict(set)
for c, k in load_pairs(REL_DIR / "course-concept.json"):
    if c in c2i:
        course_concepts[c].add(k)
        concept_courses[k].add(c)

concept_prereqs = defaultdict(set)
for a, b in load_pairs(REL_DIR / "prerequisite-dependency.json"):
    concept_prereqs[b].add(a)

n_courses = len(c2i)
concept_df = {k: len(cs) for k, cs in concept_courses.items()}


def idf(p):
    dfp = concept_df.get(p, 0)
    return math.log(1.0 + n_courses / dfp) if dfp > 0 else 0.0


manual_prereq = {}  # course_id -> set(prereq course_id)
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
print("\n=== [1] 现行人工先修覆盖 ===")
print(f"有人工先修的课程: {n_manual}/{n_courses} = {100*n_manual/n_courses:.1f}%")
sizes = [len(v) for v in manual_prereq.values()]
if sizes:
    print(f"先修集大小: median={int(np.median(sizes))} mean={np.mean(sizes):.1f}")

# ---------- 2. 隐式行为先修：受课日志时间序转移 ----------
# 每个学习者按时间排序其课程序列；对 (a 在 b 之前) 记一次 a->b
seq = df.sort_values(["u_idx", "timestamp"])
before_counts = defaultdict(lambda: defaultdict(int))  # b -> {a: count(a before b)}
target_totals = defaultdict(int)  # b -> 出现次数(有前驱的)

grouped = seq.groupby("u_idx")["course_id"].apply(list)
for courses in grouped:
    seen = []
    for c in courses:
        for a in seen:
            if a != c:
                before_counts[c][a] += 1
        if seen:
            target_totals[c] += 1
        seen.append(c)

implicit_prereq = {}  # course_id -> set topk predecessors by count
for b, amap in before_counts.items():
    if b not in c2i:
        continue
    topk = [a for a, _ in sorted(amap.items(), key=lambda x: -x[1])[:TOPK] if a in c2i]
    if topk:
        implicit_prereq[b] = set(topk)

n_impl = len(implicit_prereq)
print("\n=== [2] 隐式行为先修覆盖 ===")
print(f"有隐式先修的课程: {n_impl}/{n_courses} = {100*n_impl/n_courses:.1f}%")

# ---------- 3. 增量分析 ----------
manual_set = set(manual_prereq)
impl_set = set(implicit_prereq)
only_manual = manual_set - impl_set
only_impl = impl_set - manual_set
both = manual_set & impl_set
union = manual_set | impl_set

print("\n=== [3] 覆盖增量 ===")
print(f"仅人工:   {len(only_manual)}")
print(f"仅隐式:   {len(only_impl)}  <- 隐式新增覆盖的课程数")
print(f"两者都有: {len(both)}")
print(f"并集覆盖: {len(union)}/{n_courses} = {100*len(union)/n_courses:.1f}%  (人工单独 {100*n_manual/n_courses:.1f}%)")
print(f"隐式带来的净增覆盖: +{100*len(only_impl)/n_courses:.1f} 个百分点")

# 重叠课程上，两种信号推荐的先修课重合度(Jaccard)
if both:
    jac = []
    for c in both:
        m, im = manual_prereq[c], implicit_prereq[c]
        jac.append(len(m & im) / len(m | im) if (m | im) else 0.0)
    print(f"\n重叠课程上人工vs隐式先修集 Jaccard: mean={np.mean(jac):.3f} median={np.median(jac):.3f}")
    print("(Jaccard 低 => 两种信号互补，融合有意义；高 => 隐式只是重复人工)")
