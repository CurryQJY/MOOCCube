import copy
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hin_data_common import (
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    load_hin_processed,
    setup_seed,
    static_split_df,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = 128
        self.hidden_dim = 256
        self.batch_size = 2048
        self.n_epochs = int(os.environ.get("HHCOR_STATIC_EPOCHS", "8"))
        self.lr = 5e-4
        self.temperature = 0.10
        self.cold_threshold = int(os.environ.get("HHCOR_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("HHCOR_EVAL_N_NEG", "200"))

        self.static_seed = int(os.environ.get("HHCOR_STATIC_SEED", "2025"))
        self.train_ratio = float(os.environ.get("HHCOR_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("HHCOR_STATIC_VAL_RATIO", "0.1"))

        self.user_hist_len = int(os.environ.get("HHCOR_USER_HIST_LEN", "30"))
        self.graph_topk = int(os.environ.get("HHCOR_GRAPH_TOPK", "20"))
        self.graph_mix_weight = float(os.environ.get("HHCOR_GRAPH_MIX_WEIGHT", "0.35"))
        self.cluster_aux_weight = float(os.environ.get("HHCOR_CLUSTER_AUX_WEIGHT", "0.20"))

        self.prereq_min_support = int(os.environ.get("HHCOR_PREREQ_MIN_SUPPORT", "10"))
        self.prereq_max_per_item = int(os.environ.get("HHCOR_PREREQ_MAX_PER_ITEM", "8"))
        self.prereq_max_forward = int(os.environ.get("HHCOR_PREREQ_MAX_FORWARD", "20"))


def _parse_subject_from_course_id(course_id: str) -> str:
    cid = str(course_id)
    if cid.startswith("C_"):
        cid = cid[2:]
    m = re.search(r"course-v1:([^+]+)\+", cid)
    return m.group(1) if m else "UNK"


def _read_relation_pairs(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def build_item_course_mapping(df: pd.DataFrame, n_items: int):
    idx_course = df[["i_idx", "course_id"]].drop_duplicates(subset=["i_idx"])
    idx_to_course = [None] * n_items
    for row in idx_course.itertuples(index=False):
        idx = int(row.i_idx)
        if 0 <= idx < n_items:
            idx_to_course[idx] = str(row.course_id)
    return idx_to_course


def build_item_cluster_ids(idx_to_course):
    subjects = []
    for cid in idx_to_course:
        if cid is None:
            subjects.append("UNK")
        else:
            subjects.append(_parse_subject_from_course_id(cid))
    uniq = sorted(set(subjects))
    sub_to_idx = {s: i for i, s in enumerate(uniq)}
    cluster_ids = torch.tensor([sub_to_idx[s] for s in subjects], dtype=torch.long)
    return cluster_ids, uniq


def topk_row_normalize(mat: np.ndarray, topk: int) -> torch.Tensor:
    n = mat.shape[0]
    out = np.zeros_like(mat, dtype=np.float32)
    for i in range(n):
        row = mat[i].copy()
        row[i] = 0.0
        pos_idx = np.where(row > 0)[0]
        if pos_idx.size == 0:
            continue
        if topk > 0 and pos_idx.size > topk:
            top_idx = np.argpartition(row, -topk)[-topk:]
            top_idx = top_idx[row[top_idx] > 0]
            out[i, top_idx] = row[top_idx]
        else:
            out[i, pos_idx] = row[pos_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    out = out / row_sum
    return torch.tensor(out, dtype=torch.float32)


def build_concept_adj(idx_to_course, relation_dir: str, topk: int) -> torch.Tensor:
    n_items = len(idx_to_course)
    course_to_idx = {cid: idx for idx, cid in enumerate(idx_to_course) if cid is not None}
    concept_sets = [set() for _ in range(n_items)]
    concept_file = os.path.join(relation_dir, "course-concept.json")
    for cid, concept in _read_relation_pairs(concept_file):
        idx = course_to_idx.get(cid)
        if idx is not None and concept:
            concept_sets[idx].add(concept)

    sim = np.zeros((n_items, n_items), dtype=np.float32)
    for i in range(n_items):
        ci = concept_sets[i]
        if not ci:
            continue
        for j in range(i + 1, n_items):
            cj = concept_sets[j]
            if not cj:
                continue
            inter = len(ci.intersection(cj))
            if inter <= 0:
                continue
            score = inter / np.sqrt(float(len(ci) * len(cj)))
            sim[i, j] = score
            sim[j, i] = score
    return topk_row_normalize(sim, topk=topk)


def build_prereq_adj(
    train_df: pd.DataFrame,
    n_items: int,
    min_support: int,
    max_per_item: int,
    max_forward: int,
    topk: int
) -> torch.Tensor:
    ordered = train_df.sort_values("timestamp") if "timestamp" in train_df.columns else train_df
    user_seq = defaultdict(list)
    user_seen = defaultdict(set)
    for u, i in zip(ordered["u_idx"].values, ordered["i_idx"].values):
        uid = int(u)
        iid = int(i)
        if iid not in user_seen[uid]:
            user_seen[uid].add(iid)
            user_seq[uid].append(iid)

    support = defaultdict(int)
    for seq in user_seq.values():
        for pos, dst in enumerate(seq):
            start = max(0, pos - max_forward)
            for src in seq[start:pos]:
                if src != dst:
                    support[(src, dst)] += 1

    incoming = defaultdict(list)
    for (src, dst), cnt in support.items():
        if cnt >= min_support:
            incoming[dst].append((src, cnt))

    mat = np.zeros((n_items, n_items), dtype=np.float32)
    for dst, src_list in incoming.items():
        src_list.sort(key=lambda x: x[1], reverse=True)
        for src, cnt in src_list[:max_per_item]:
            if 0 <= src < n_items and 0 <= dst < n_items:
                mat[dst, src] = float(cnt)
    return topk_row_normalize(mat, topk=topk)


def build_cooc_adj(train_df: pd.DataFrame, n_items: int, topk: int) -> torch.Tensor:
    user_items = defaultdict(set)
    for u, i in zip(train_df["u_idx"].values, train_df["i_idx"].values):
        user_items[int(u)].add(int(i))

    counts = defaultdict(int)
    for items in user_items.values():
        if len(items) < 2:
            continue
        arr = sorted(items)
        m = len(arr)
        for a in range(m):
            ia = arr[a]
            if ia < 0 or ia >= n_items:
                continue
            for b in range(a + 1, m):
                ib = arr[b]
                if ib < 0 or ib >= n_items:
                    continue
                counts[(ia, ib)] += 1
                counts[(ib, ia)] += 1

    mat = np.zeros((n_items, n_items), dtype=np.float32)
    for (i, j), c in counts.items():
        mat[i, j] = float(c)
    return topk_row_normalize(mat, topk=topk)


def _clone_user_histories(user_histories):
    return {uid: list(items) for uid, items in user_histories.items()}


def _update_histories_from_df(user_histories, src_df: pd.DataFrame):
    ordered = src_df.sort_values("timestamp") if "timestamp" in src_df.columns else src_df
    for u, i in zip(ordered["u_idx"].values, ordered["i_idx"].values):
        uid = int(u)
        if uid not in user_histories:
            user_histories[uid] = []
        user_histories[uid].append(int(i))
    return user_histories


def build_history_tensor(
    df: pd.DataFrame,
    base_histories,
    max_len: int,
    update_histories: bool
):
    hist = np.full((len(df), max_len), -1, dtype=np.int64)
    work = _clone_user_histories(base_histories)

    ordered = df.reset_index(drop=True).copy()
    ordered["pos_idx"] = np.arange(len(ordered), dtype=np.int64)
    if "timestamp" in ordered.columns:
        ordered = ordered.sort_values("timestamp")

    pos_idx = ordered["pos_idx"].to_numpy()
    u_arr = ordered["u_idx"].to_numpy()
    i_arr = ordered["i_idx"].to_numpy()

    for row_idx, uid_raw, iid_raw in zip(pos_idx, u_arr, i_arr):
        uid = int(uid_raw)
        iid = int(iid_raw)
        seq = work.get(uid, [])
        if seq:
            tail = seq[-max_len:]
            hist[row_idx, -len(tail):] = tail
        if update_histories:
            if uid not in work:
                work[uid] = []
            work[uid].append(iid)

    return torch.tensor(hist, dtype=torch.long), work


class HHCoRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, history_tensor: torch.Tensor):
        self.u = torch.tensor(df["u_idx"].values, dtype=torch.long)
        self.i = torch.tensor(df["i_idx"].values, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)
        self.hist = history_tensor

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            "u": self.u[idx],
            "i": self.i[idx],
            "hist": self.hist[idx],
            "pop": self.pop[idx]
        }


def collate_hhcor(batch):
    u = torch.stack([x["u"] for x in batch])
    i = torch.stack([x["i"] for x in batch])
    pop = torch.stack([x["pop"] for x in batch])
    hist = torch.stack([x["hist"] for x in batch])
    return {"u": u, "i": i, "hist": hist}, pop


class HHCoRStaticModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor, n_clusters: int):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_id_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.cluster_emb = nn.Embedding(n_clusters, cfg.emb_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)
        nn.init.xavier_normal_(self.cluster_emb.weight)

        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim),
        )
        self.user_gate = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid(),
        )
        self.user_fuse = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.emb_dim),
        )
        self.channel_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        self.register_buffer("concept_adj", torch.zeros(cfg.n_items, cfg.n_items))
        self.register_buffer("prereq_adj", torch.zeros(cfg.n_items, cfg.n_items))
        self.register_buffer("cooc_adj", torch.zeros(cfg.n_items, cfg.n_items))
        self.register_buffer("item_cluster_ids", torch.zeros(cfg.n_items, dtype=torch.long))

    def set_graph_buffers(
        self,
        concept_adj: torch.Tensor,
        prereq_adj: torch.Tensor,
        cooc_adj: torch.Tensor,
        item_cluster_ids: torch.Tensor
    ):
        self.concept_adj = concept_adj.to(self.user_emb.weight.device)
        self.prereq_adj = prereq_adj.to(self.user_emb.weight.device)
        self.cooc_adj = cooc_adj.to(self.user_emb.weight.device)
        self.item_cluster_ids = item_cluster_ids.to(self.user_emb.weight.device)

    def get_item_bank(self):
        item_id = self.item_id_emb.weight
        item_con = self.content_proj(self.item_con_emb.weight)
        base = F.normalize(item_id + item_con, dim=1)

        concept_msg = torch.matmul(self.concept_adj, base)
        prereq_msg = torch.matmul(self.prereq_adj, base)
        cooc_msg = torch.matmul(self.cooc_adj, base)

        w = F.softmax(self.channel_logits, dim=0)
        graph_msg = w[0] * concept_msg + w[1] * prereq_msg + w[2] * cooc_msg
        item_bank = F.normalize(base + self.cfg.graph_mix_weight * graph_msg, dim=1)
        return item_bank

    def encode_users(self, u_idx: torch.Tensor, hist_idx: torch.Tensor, item_bank: torch.Tensor):
        u_id = self.user_emb(u_idx)
        mask = (hist_idx >= 0).float().unsqueeze(-1)
        safe_idx = hist_idx.clamp(min=0)
        hist_item_vec = item_bank[safe_idx]
        hist_den = mask.sum(dim=1).clamp_min(1.0)
        hist_vec = (hist_item_vec * mask).sum(dim=1) / hist_den

        gate = self.user_gate(torch.cat([u_id, hist_vec], dim=1))
        mixed = gate * u_id + (1.0 - gate) * hist_vec
        user_vec = self.user_fuse(torch.cat([mixed, hist_vec], dim=1))
        return F.normalize(user_vec, dim=1)

    def forward(self, batch):
        item_bank = self.get_item_bank()
        z_u = self.encode_users(batch["u"], batch["hist"], item_bank)
        z_i = item_bank[batch["i"]]

        logits_item = torch.matmul(z_u, z_i.t()) / self.cfg.temperature
        labels = torch.arange(logits_item.size(0), device=logits_item.device)
        loss_item = F.cross_entropy(logits_item, labels)

        cluster_targets = self.item_cluster_ids[batch["i"]]
        cluster_bank = F.normalize(self.cluster_emb.weight, dim=1)
        cluster_logits = torch.matmul(z_u, cluster_bank.t()) / self.cfg.temperature
        loss_cluster = F.cross_entropy(cluster_logits, cluster_targets)

        return loss_item + self.cfg.cluster_aux_weight * loss_cluster


def main():
    setup_seed(2025)
    print("Loading data from processed_data_hin ...")
    meta, df, content_emb = load_hin_processed("processed_data_hin")
    cfg = Config(meta["n_users"], meta["n_items"], content_dim=content_emb.shape[1])

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio
    )
    print(
        f"Static split done: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}"
    )

    idx_to_course = build_item_course_mapping(df, cfg.n_items)
    item_cluster_ids, subjects = build_item_cluster_ids(idx_to_course)
    print(f"Cluster count(subject): {len(subjects)}")

    relation_dir = os.path.join("MOOCCube", "relations")
    concept_adj = build_concept_adj(idx_to_course, relation_dir, topk=cfg.graph_topk)
    prereq_adj = build_prereq_adj(
        train_df,
        n_items=cfg.n_items,
        min_support=cfg.prereq_min_support,
        max_per_item=cfg.prereq_max_per_item,
        max_forward=cfg.prereq_max_forward,
        topk=cfg.graph_topk
    )
    cooc_adj = build_cooc_adj(train_df, n_items=cfg.n_items, topk=cfg.graph_topk)
    print("Graph channels ready: concept/prereq/cooc")

    train_hist, train_histories = build_history_tensor(
        train_df, base_histories={}, max_len=cfg.user_hist_len, update_histories=True
    )
    val_hist, _ = build_history_tensor(
        val_df, base_histories=train_histories, max_len=cfg.user_hist_len, update_histories=False
    )
    train_val_histories = _clone_user_histories(train_histories)
    _update_histories_from_df(train_val_histories, val_df)
    test_hist, _ = build_history_tensor(
        test_df, base_histories=train_val_histories, max_len=cfg.user_hist_len, update_histories=False
    )

    train_ds = HHCoRDataset(train_df, train_hist)
    val_ds = HHCoRDataset(val_df, val_hist)
    test_ds = HHCoRDataset(test_df, test_hist)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_hhcor)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hhcor)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hhcor)

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    add_user_seen_from_df(test_seen, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HHCoRStaticModel(cfg, content_emb, n_clusters=len(subjects)).to(device)
    model.set_graph_buffers(concept_adj, prereq_adj, cooc_adj, item_cluster_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: HHCoR-static | device={device} | epochs={cfg.n_epochs}")

    best_val = -1.0
    best_epoch = -1
    best_state = None
    k_list = [5, 10, 20]

    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch, _ in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)

        model.eval()
        with torch.no_grad():
            item_bank = model.get_item_bank().detach()
            get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)
            val_full_cold, _ = evaluate_embedding_ranker(
                val_loader,
                device=device,
                n_items=cfg.n_items,
                cold_threshold=cfg.cold_threshold,
                get_user_vectors_fn=get_user_fn,
                all_item_vectors=item_bank,
                k_list=k_list,
                n_neg=cfg.eval_n_neg,
                eval_type="cold",
                full_ranking=True,
                user_seen_items=train_seen
            )
            val_key = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f} | val_full_cold_N@10={val_key:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restore best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    model.eval()
    with torch.no_grad():
        item_bank = model.get_item_bank().detach()
        get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)

        sample_cold, n_sc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
            user_seen_items=test_seen
        )
        sample_hot, n_sh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
            user_seen_items=test_seen
        )
        full_cold, n_fc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
            user_seen_items=test_seen
        )
        full_hot, n_fh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
            user_seen_items=test_seen
        )

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=sample_cold,
        sample_hot=sample_hot,
        full_cold=full_cold,
        full_hot=full_hot,
        count_sample_cold=n_sc,
        count_sample_hot=n_sh,
        count_full_cold=n_fc,
        count_full_hot=n_fh,
        title="HHCoR-Style Static HIN"
    )

    out = {
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
    }
    pd.DataFrame([out]).to_json("hhcor_static_result.json", orient="records", force_ascii=False)
    print("Saved: hhcor_static_result.json")


if __name__ == "__main__":
    main()
