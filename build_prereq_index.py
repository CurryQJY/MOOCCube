"""路A实现地基：构造 checkpoint 无关的先修课程索引映射。

链路：course -> concepts(course-concept) -> prereq concepts(prerequisite-dependency)
      -> 覆盖这些先修概念的课程(按IDF加权共享先修概念得分) -> top-k 课程索引

产出 [n_items, topk] 的 item_id 索引 + [n_items] 有效掩码。纯数据，不依赖任何模型。
训练时用它动态查表：mu_pre(c) = mean(item_id_emb[prereq_idx[c]])，零循环依赖。

输出：outputs/prereq_target/prereq_index_topk{K}.pt
"""
from __future__ import annotations
import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load_jsonl_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    ap.add_argument("--relation-dir", default="MOOCCube/relations")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs/prereq_target")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rel_dir = Path(args.relation_dir)

    # 1. i_idx <-> course_id 映射
    with open(data_dir / "stream_data.pkl", "rb") as f:
        stream = pickle.load(f)
    i2c = dict(zip(stream["i_idx"].astype(int), stream["course_id"]))
    c2i = {c: i for i, c in i2c.items()}
    n_items = int(max(i2c) + 1)

    # 2. course -> concepts
    course_concepts = defaultdict(set)
    concept_courses = defaultdict(set)
    for c, k in load_jsonl_pairs(rel_dir / "course-concept.json"):
        if c in c2i:
            course_concepts[c].add(k)
            concept_courses[k].add(c)

    # 3. concept 先修：'a\tb' 表示 a 是 b 的先修 => b 的先修集含 a
    concept_prereqs = defaultdict(set)
    for a, b in load_jsonl_pairs(rel_dir / "prerequisite-dependency.json"):
        concept_prereqs[b].add(a)

    # 4. IDF 权重：先修概念 p 越稀有权重越高
    n_courses = len(c2i)
    concept_df = {k: len(cs) for k, cs in concept_courses.items()}
    def idf(p):
        df = concept_df.get(p, 0)
        return math.log(1.0 + n_courses / df) if df > 0 else 0.0

    # 5. 每门课的先修课程 top-k 索引
    prereq_idx = torch.full((n_items, args.topk), -1, dtype=torch.long)
    has_prereq = torch.zeros(n_items, dtype=torch.bool)
    sizes = []

    for c, i in c2i.items():
        my_concepts = course_concepts.get(c, set())
        if not my_concepts:
            continue
        my_prereq_concepts = set()
        for k in my_concepts:
            my_prereq_concepts |= concept_prereqs.get(k, set())
        my_prereq_concepts -= my_concepts
        if not my_prereq_concepts:
            continue
        cand_score = defaultdict(float)
        for p in my_prereq_concepts:
            w = idf(p)
            if w <= 0:
                continue
            for c2 in concept_courses.get(p, set()):
                if c2 != c:
                    cand_score[c2] += w
        if not cand_score:
            continue
        topk = sorted(cand_score.items(), key=lambda x: -x[1])[: args.topk]
        idxs = [c2i[c2] for c2, _ in topk if c2 in c2i]
        if not idxs:
            continue
        for j, idx in enumerate(idxs):
            prereq_idx[i, j] = idx
        has_prereq[i] = True
        sizes.append(len(idxs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prereq_index_topk{args.topk}.pt"
    torch.save(
        {"prereq_idx": prereq_idx, "has_prereq": has_prereq, "topk": args.topk, "n_items": n_items},
        out_path,
    )
    print(f"=== 先修索引映射 (topk={args.topk}) ===")
    print(f"n_items={n_items}, 有先修的课程={int(has_prereq.sum())}/{n_items}")
    print(f"先修集合大小: median={int(np.median(sizes))} mean={np.mean(sizes):.1f}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
