"""路A地基：构造每门课的"先修课程行为嵌入质心" mu_pre，并检验区分度。

链路：course -> concepts(course-concept) -> prereq concepts(prerequisite-dependency)
      -> 覆盖这些先修概念的课程(按IDF加权共享先修概念得分) -> top-k -> id_e_true质心

mu_pre 在 item_id_emb 的 128 维行为空间，可直接替换 forward 里的 target_emb。

关键 gate：mu_pre 之间必须有区分度（不能都塌缩到全局质心），否则先修 target 无意义。
输出：outputs/prereq_target/mu_pre_seed{seed}.pt + 区分度报告。
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load_jsonl_pairs(path):
    """每行 'A\\tB' -> (A,B)。"""
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs/prereq_target")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rel_dir = Path(args.relation_dir)

    # 1. i_idx <-> course_id 映射（来自 stream_data）
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

    # 3. concept 先修：prerequisite-dependency  A 是 B 的先修 (A -> B 表示先学A再学B? 取决于语义)
    #    文件格式：每行 "prereq_concept\ttarget_concept"（前者是后者的先修）
    concept_prereqs = defaultdict(set)  # concept -> 它的先修concepts
    prereq_pairs = load_jsonl_pairs(rel_dir / "prerequisite-dependency.json")
    for a, b in prereq_pairs:
        # a 是 b 的先修 => b 的先修集合含 a
        concept_prereqs[b].add(a)

    # 4. IDF 权重：先修概念 p 的稀有度 = log(n_courses / df(p))
    n_courses = len(c2i)
    concept_df = {k: len(cs) for k, cs in concept_courses.items()}
    def idf(p):
        df = concept_df.get(p, 0)
        if df <= 0:
            return 0.0
        return math.log(1.0 + n_courses / df)

    # 5. 载入 id_e_true (item_id_emb)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ms = ck.get("model_state", ck)
    id_emb = ms["item_id_emb.weight"].float()  # (n_items,128)
    global_centroid = id_emb.mean(dim=0)

    # 6. 为每门课构造先修课程集合 + mu_pre
    mu_pre = torch.zeros(n_items, id_emb.size(1))
    has_prereq = torch.zeros(n_items, dtype=torch.bool)
    prereq_set_sizes = []

    for c, i in c2i.items():
        my_concepts = course_concepts.get(c, set())
        if not my_concepts:
            continue
        # 我的先修概念集合
        my_prereq_concepts = set()
        for k in my_concepts:
            my_prereq_concepts |= concept_prereqs.get(k, set())
        my_prereq_concepts -= my_concepts  # 去掉自身也教的概念
        if not my_prereq_concepts:
            continue
        # 候选先修课程：覆盖了我的先修概念的其他课程，按 IDF 加权共享数打分
        cand_score = defaultdict(float)
        for p in my_prereq_concepts:
            w = idf(p)
            if w <= 0:
                continue
            for c2 in concept_courses.get(p, set()):
                if c2 == c:
                    continue
                cand_score[c2] += w
        if not cand_score:
            continue
        topk = sorted(cand_score.items(), key=lambda x: -x[1])[: args.topk]
        idxs = [c2i[c2] for c2, _ in topk if c2 in c2i]
        if not idxs:
            continue
        mu_pre[i] = id_emb[idxs].mean(dim=0)
        has_prereq[i] = True
        prereq_set_sizes.append(len(idxs))

    # 7. 区分度检验（gate）
    valid_idx = has_prereq.nonzero(as_tuple=False).view(-1)
    mu_valid = mu_pre[valid_idx]
    mu_norm = torch.nn.functional.normalize(mu_valid, dim=1)
    # 两两 cosine
    sim = mu_norm @ mu_norm.t()
    n = sim.size(0)
    off = sim[~torch.eye(n, dtype=torch.bool)]
    # 到全局质心的 cosine
    gc = torch.nn.functional.normalize(global_centroid, dim=0)
    sim_to_global = (mu_norm @ gc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mu_pre": mu_pre, "has_prereq": has_prereq, "topk": args.topk, "seed": args.seed},
        out_dir / f"mu_pre_seed{args.seed}.pt",
    )

    print(f"=== mu_pre 构造 (seed{args.seed}, topk={args.topk}) ===")
    print(f"有先修的课程: {int(has_prereq.sum())}/{n_items}")
    print(f"先修课程集合大小: median={int(np.median(prereq_set_sizes))} mean={np.mean(prereq_set_sizes):.1f}")
    print(f"\n=== 区分度 gate ===")
    print(f"mu_pre 两两 cosine: mean={off.mean():.4f} median={off.median():.4f} (越低越有区分度)")
    print(f"mu_pre 到全局质心 cosine: mean={sim_to_global.mean():.4f} median={sim_to_global.median():.4f}")
    print(f"  (若普遍接近1，说明塌缩到全局均值，无区分度)")
    # 冷课单独看
    # 冷课 = per_item_full_cold 里出现的 item_id
    print(f"\n判据：两两cosine均值 < 0.9 且到全局cosine有分散 => 有区分度，路A可继续")


if __name__ == "__main__":
    main()
