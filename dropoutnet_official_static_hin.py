"""Official-protocol DropoutNet adaptation for the shared static item-cold split.

This script keeps the local baseline separate from ``drop_static_hin.py``.
It follows the original DropoutNet recipe more closely:

1. Train a warm collaborative teacher on train interactions only.
2. Freeze the teacher user/item latent factors.
3. Train DropoutNet towers from ``[preference latent, side feature]`` inputs,
   randomly dropping the item preference latent during training.
4. At item-cold evaluation time, cold items are scored with their preference
   latent forced to zero, so ranking depends on content features.

The official repository is https://github.com/layer6ai-labs/DropoutNet.  If a
local checkout exists at ``third_party/DropoutNet``, the output records it; the
training code here is a protocol adaptation for the MOOCCube static split.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hin_data_common import load_hin_processed, static_result_path, static_split_df
from baseline_checkpoint import CheckpointConfig, checkpoint_config, maybe_resume_checkpoint, save_checkpoint


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
K_LIST = (5, 10, 20)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed fixed: {seed}")


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim
        self.emb_dim = int(os.environ.get("DROPOUT_OFFICIAL_EMB_DIM", "64"))
        self.hidden_dim = int(os.environ.get("DROPOUT_OFFICIAL_HIDDEN_DIM", "128"))
        self.batch_size = int(os.environ.get("DROPOUT_OFFICIAL_BATCH_SIZE", "4096"))
        self.teacher_epochs = int(os.environ.get("DROPOUT_OFFICIAL_TEACHER_EPOCHS", "80"))
        self.student_epochs = int(
            os.environ.get(
                "DROPOUT_OFFICIAL_STATIC_EPOCHS",
                os.environ.get("DROPOUT_STATIC_EPOCHS", "80"),
            )
        )
        self.eval_interval = int(os.environ.get("DROPOUT_OFFICIAL_EVAL_INTERVAL", "5"))
        self.teacher_lr = float(os.environ.get("DROPOUT_OFFICIAL_TEACHER_LR", "1e-3"))
        self.student_lr = float(os.environ.get("DROPOUT_OFFICIAL_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("DROPOUT_OFFICIAL_WEIGHT_DECAY", "1e-6"))
        self.item_pref_dropout = float(os.environ.get("DROPOUT_OFFICIAL_ITEM_DROPOUT", "0.5"))
        self.user_pref_dropout = float(os.environ.get("DROPOUT_OFFICIAL_USER_DROPOUT", "0.0"))
        self.temperature = float(os.environ.get("DROPOUT_OFFICIAL_TEMPERATURE", "0.1"))
        self.cold_threshold = int(os.environ.get("DROPOUT_OFFICIAL_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1")))
        self.eval_n_neg = int(os.environ.get("DROPOUT_OFFICIAL_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("DROPOUT_OFFICIAL_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("DROPOUT_OFFICIAL_SEED", os.environ.get("DROPOUT_SEED", str(self.static_seed))))
        self.train_ratio = float(os.environ.get("DROPOUT_OFFICIAL_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("DROPOUT_OFFICIAL_STATIC_VAL_RATIO", "0.1"))
        self.early_stop_average_mode = os.environ.get(
            "DROPOUT_OFFICIAL_EARLY_STOP_AVG_MODE",
            os.environ.get("USIM_EARLY_STOP_AVG_MODE", "item_macro"),
        ).strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("early stop average mode must be 'interaction' or 'item_macro'")
        self.ckpt = checkpoint_config("DROPOUT_OFFICIAL")
        teacher_ckpt_dir = os.environ.get("DROPOUT_OFFICIAL_TEACHER_CKPT_DIR", "").strip()
        if not teacher_ckpt_dir and self.ckpt.dir:
            teacher_ckpt_dir = os.path.join(self.ckpt.dir, "teacher")
        self.teacher_ckpt = CheckpointConfig(
            dir=teacher_ckpt_dir,
            save=self.ckpt.save,
            resume=self.ckpt.resume,
            force_fresh=self.ckpt.force_fresh,
            save_opt=self.ckpt.save_opt,
        )


class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.u = torch.tensor(df["u_idx"].to_numpy(), dtype=torch.long)
        self.i = torch.tensor(df["i_idx"].to_numpy(), dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.u.numel())

    def __getitem__(self, idx: int):
        return {"u": self.u[idx], "i": self.i[idx], "pop": self.pop[idx]}


def collate(batch):
    return (
        {
            "u": torch.stack([x["u"] for x in batch]),
            "i": torch.stack([x["i"] for x in batch]),
        },
        torch.stack([x["pop"] for x in batch]),
    )


def build_user_seen(df: pd.DataFrame) -> Dict[int, set[int]]:
    seen: Dict[int, set[int]] = {}
    for u_idx, i_idx in zip(df["u_idx"].to_numpy(), df["i_idx"].to_numpy()):
        seen.setdefault(int(u_idx), set()).add(int(i_idx))
    return seen


def clone_seen(seen: Dict[int, set[int]]) -> Dict[int, set[int]]:
    return {u: set(items) for u, items in seen.items()}


class TeacherMF(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        user = F.normalize(self.user_emb(u), dim=-1)
        item = F.normalize(self.item_emb(i), dim=-1)
        return (user * item).sum(dim=-1)


class OfficialStyleDropoutNet(nn.Module):
    def __init__(
        self,
        cfg: Config,
        content_emb: torch.Tensor,
        teacher_user: torch.Tensor,
        teacher_item: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("content_features", content_emb.float())
        self.register_buffer("teacher_user", teacher_user.float())
        self.register_buffer("teacher_item", teacher_item.float())
        self.user_net = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )
        self.item_net = nn.Sequential(
            nn.Linear(cfg.emb_dim + cfg.content_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )

    def _drop_preference(self, pref: torch.Tensor, p: float, force_zero: bool = False) -> torch.Tensor:
        if force_zero:
            return torch.zeros_like(pref)
        if self.training and p > 0.0:
            keep = (torch.rand(pref.size(0), 1, device=pref.device) >= p).float()
            return pref * keep
        return pref

    def user_vector(self, u_idx: torch.Tensor, force_zero_pref: bool = False) -> torch.Tensor:
        pref = self.teacher_user[u_idx]
        pref = self._drop_preference(pref, self.cfg.user_pref_dropout, force_zero_pref)
        return self.user_net(pref)

    def item_vector(self, i_idx: torch.Tensor, force_zero_pref: bool = False) -> torch.Tensor:
        pref = self.teacher_item[i_idx]
        pref = self._drop_preference(pref, self.cfg.item_pref_dropout, force_zero_pref)
        content = self.content_features[i_idx]
        return self.item_net(torch.cat([pref, content], dim=-1))


def sample_negative_items(item_pool: torch.Tensor, shape: torch.Size, device: torch.device) -> torch.Tensor:
    pool = item_pool.to(device)
    idx = torch.randint(0, pool.numel(), shape, device=device)
    return pool[idx]


def train_teacher(
    cfg: Config,
    train_loader: DataLoader,
    negative_item_pool: torch.Tensor,
    device: torch.device,
) -> TeacherMF:
    teacher = TeacherMF(cfg).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=cfg.teacher_lr, weight_decay=cfg.weight_decay)
    start_epoch, _ = maybe_resume_checkpoint(cfg.teacher_ckpt, teacher, opt, device)
    for epoch in range(start_epoch + 1, cfg.teacher_epochs + 1):
        teacher.train()
        total = 0.0
        steps = 0
        for batch, _ in train_loader:
            u = batch["u"].to(device)
            pos = batch["i"].to(device)
            neg = sample_negative_items(negative_item_pool, pos.shape, device)
            pos_s = teacher.score(u, pos)
            neg_s = teacher.score(u, neg)
            loss = -F.logsigmoid(pos_s - neg_s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if epoch == 1 or epoch == cfg.teacher_epochs or epoch % max(1, cfg.eval_interval) == 0:
            print(f"[Teacher] Epoch {epoch}/{cfg.teacher_epochs} loss={total / max(1, steps):.4f}")
        if cfg.teacher_ckpt.save:
            save_checkpoint(
                cfg.teacher_ckpt,
                "latest.pt",
                epoch,
                teacher,
                opt,
                extra={"best_val": -1.0, "best_epoch": epoch},
            )
    return teacher


def mixed_item_vectors(model: OfficialStyleDropoutNet, i_idx: torch.Tensor, cold_mask: torch.Tensor) -> torch.Tensor:
    cold_mask = cold_mask.to(i_idx.device).bool().view(-1, 1)
    if bool(cold_mask.all()):
        return model.item_vector(i_idx, force_zero_pref=True)
    if bool((~cold_mask).all()):
        return model.item_vector(i_idx, force_zero_pref=False)
    cold_vec = model.item_vector(i_idx, force_zero_pref=True)
    hot_vec = model.item_vector(i_idx, force_zero_pref=False)
    return torch.where(cold_mask, cold_vec, hot_vec)


def precompute_item_bank(
    cfg: Config,
    model: OfficialStyleDropoutNet,
    item_counts: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, cfg.n_items, batch_size):
            idx = torch.arange(start, min(cfg.n_items, start + batch_size), dtype=torch.long, device=device)
            cold_mask = item_counts[idx.detach().cpu()].to(device) < cfg.cold_threshold
            vec = mixed_item_vectors(model, idx, cold_mask)
            out.append(F.normalize(vec, dim=-1).cpu())
    return torch.cat(out, dim=0)


def _empty_metric_sum() -> Dict[str, float]:
    return {key: 0.0 for key in METRICS}


def _macro_result(
    item_sum: Dict[str, Dict[int, float]],
    item_count: Dict[int, int],
    export_path: str | None = None,
):
    if not item_count:
        return None, 0
    out = {}
    for key, values in item_sum.items():
        vals = [values.get(item_id, 0.0) / count for item_id, count in item_count.items() if count > 0]
        out[key] = float(sum(vals) / max(1, len(vals)))
    if export_path:
        rows = []
        for item_id in sorted(item_count):
            count = max(1, int(item_count[item_id]))
            row = {"item_id": int(item_id), "count": int(item_count[item_id])}
            for key, values in item_sum.items():
                row[key] = float(values.get(item_id, 0.0) / count)
            rows.append(row)
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(export_path, index=False)
    return out, len(item_count)


def evaluate_full(
    cfg: Config,
    model: OfficialStyleDropoutNet,
    loader: DataLoader,
    all_item_z: torch.Tensor,
    item_counts: torch.Tensor,
    device: torch.device,
    user_seen_items: Dict[int, set[int]] | None,
    average_mode: str = "interaction",
    export_cold_item_metrics_path: str | None = None,
    export_hot_item_metrics_path: str | None = None,
):
    model.eval()
    average_mode = average_mode.strip().lower()
    cold_sum = _empty_metric_sum()
    hot_sum = _empty_metric_sum()
    cold_total = 0
    hot_total = 0
    cold_item_sum = {key: {} for key in METRICS}
    hot_item_sum = {key: {} for key in METRICS}
    cold_item_count: Dict[int, int] = {}
    hot_item_count: Dict[int, int] = {}
    seen_tensor_cache = {}
    all_item_z = all_item_z.to(device)
    with torch.no_grad():
        for batch, pop in loader:
            u = batch["u"].to(device)
            i = batch["i"].to(device)
            pop = pop.to(device)
            cold_mask = pop < cfg.cold_threshold
            z_u = F.normalize(model.user_vector(u), dim=-1)
            z_pos = F.normalize(mixed_item_vectors(model, i, cold_mask), dim=-1)
            scores = torch.matmul(z_u, all_item_z.t())
            pos_scores = (z_u * z_pos).sum(dim=-1)
            rows = torch.arange(u.size(0), device=device)
            scores[rows, i] = pos_scores

            if user_seen_items:
                user_ids = u.detach().cpu().tolist()
                for row, uid_raw in enumerate(user_ids):
                    uid = int(uid_raw)
                    if uid not in seen_tensor_cache:
                        seen = user_seen_items.get(uid)
                        if seen:
                            seen_list = [x for x in seen if 0 <= x < cfg.n_items]
                            seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=device) if seen_list else None
                        else:
                            seen_tensor_cache[uid] = None
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is not None:
                        scores[row, seen_idx] = -1e9
                scores[rows, i] = pos_scores

            _, topk = torch.topk(scores, k=max(K_LIST), dim=1)
            target = i.view(-1, 1)
            item_ids = [int(x) for x in i.detach().cpu().tolist()]
            cold_cpu = cold_mask.detach().cpu()
            for k in K_LIST:
                preds = topk[:, :k]
                hits = (preds == target).any(dim=1).float()
                hit_ranks = (preds == target).nonzero(as_tuple=True)
                dcg = torch.zeros(u.size(0), device=device)
                if hit_ranks[0].numel() > 0:
                    dcg[hit_ranks[0]] = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                key_r = f"R@{k}"
                key_n = f"N@{k}"
                if average_mode == "item_macro":
                    hits_cpu = hits.detach().cpu()
                    dcg_cpu = dcg.detach().cpu()
                    for row, item_id in enumerate(item_ids):
                        if bool(cold_cpu[row].item()):
                            if k == K_LIST[0]:
                                cold_item_count[item_id] = cold_item_count.get(item_id, 0) + 1
                            cold_item_sum[key_r][item_id] = cold_item_sum[key_r].get(item_id, 0.0) + float(hits_cpu[row].item())
                            cold_item_sum[key_n][item_id] = cold_item_sum[key_n].get(item_id, 0.0) + float(dcg_cpu[row].item())
                        else:
                            if k == K_LIST[0]:
                                hot_item_count[item_id] = hot_item_count.get(item_id, 0) + 1
                            hot_item_sum[key_r][item_id] = hot_item_sum[key_r].get(item_id, 0.0) + float(hits_cpu[row].item())
                            hot_item_sum[key_n][item_id] = hot_item_sum[key_n].get(item_id, 0.0) + float(dcg_cpu[row].item())
                else:
                    hot_mask = ~cold_mask
                    cold_sum[key_r] += float(hits[cold_mask].sum().item())
                    cold_sum[key_n] += float(dcg[cold_mask].sum().item())
                    hot_sum[key_r] += float(hits[hot_mask].sum().item())
                    hot_sum[key_n] += float(dcg[hot_mask].sum().item())

            cold_total += int(cold_mask.sum().item())
            hot_total += int((~cold_mask).sum().item())

    if average_mode == "item_macro":
        cold_res, cold_count = _macro_result(
            cold_item_sum,
            cold_item_count,
            export_cold_item_metrics_path,
        )
        hot_res, hot_count = _macro_result(
            hot_item_sum,
            hot_item_count,
            export_hot_item_metrics_path,
        )
        return cold_res, cold_count, hot_res, hot_count
    cold_res = {k: v / max(1, cold_total) for k, v in cold_sum.items()} if cold_total else None
    hot_res = {k: v / max(1, hot_total) for k, v in hot_sum.items()} if hot_total else None
    return cold_res, cold_total, hot_res, hot_total


def evaluate_sampled(
    cfg: Config,
    model: OfficialStyleDropoutNet,
    loader: DataLoader,
    all_item_z: torch.Tensor,
    item_counts: torch.Tensor,
    device: torch.device,
    user_seen_items: Dict[int, set[int]] | None,
):
    model.eval()
    cold_sum = _empty_metric_sum()
    hot_sum = _empty_metric_sum()
    cold_total = 0
    hot_total = 0
    all_items_np = np.arange(cfg.n_items, dtype=np.int64)
    all_item_z = all_item_z.to(device)
    with torch.no_grad():
        for batch, pop in loader:
            u = batch["u"].to(device)
            i = batch["i"].to(device)
            pop = pop.to(device)
            bsz = u.size(0)
            z_u = F.normalize(model.user_vector(u), dim=-1)
            cold_mask = pop < cfg.cold_threshold
            z_pos = F.normalize(mixed_item_vectors(model, i, cold_mask), dim=-1)
            scores_full = torch.matmul(z_u, all_item_z.t())
            rows = torch.arange(bsz, device=device)
            scores_full[rows, i] = (z_u * z_pos).sum(dim=-1)
            neg_np = np.empty((bsz, min(cfg.eval_n_neg, cfg.n_items - 1)), dtype=np.int64)
            u_cpu = u.detach().cpu().numpy()
            i_cpu = i.detach().cpu().numpy()
            for row in range(bsz):
                forbidden = {int(i_cpu[row])}
                if user_seen_items:
                    forbidden.update(user_seen_items.get(int(u_cpu[row]), set()))
                pool = np.setdiff1d(all_items_np, np.array(list(forbidden), dtype=np.int64), assume_unique=False)
                if pool.size < neg_np.shape[1]:
                    pool = all_items_np[all_items_np != int(i_cpu[row])]
                neg_np[row] = np.random.choice(pool, size=neg_np.shape[1], replace=False)
            neg = torch.from_numpy(neg_np).to(device)
            cand = torch.cat([i.view(-1, 1), neg], dim=1)
            perm = torch.argsort(torch.rand(cand.size(0), cand.size(1), device=device), dim=1)
            cand = cand.gather(1, perm)
            target_cols = (cand == i.view(-1, 1)).nonzero(as_tuple=True)[1].view(-1, 1)
            scores = scores_full.gather(1, cand)
            _, topk = torch.topk(scores, k=min(max(K_LIST), scores.size(1)), dim=1)
            for k in K_LIST:
                preds = topk[:, : min(k, topk.size(1))]
                hits = (preds == target_cols).any(dim=1).float()
                hit_ranks = (preds == target_cols).nonzero(as_tuple=True)
                dcg = torch.zeros(bsz, device=device)
                if hit_ranks[0].numel() > 0:
                    dcg[hit_ranks[0]] = 1.0 / torch.log2(hit_ranks[1].float() + 2.0)
                hot_mask = ~cold_mask
                cold_sum[f"R@{k}"] += float(hits[cold_mask].sum().item())
                cold_sum[f"N@{k}"] += float(dcg[cold_mask].sum().item())
                hot_sum[f"R@{k}"] += float(hits[hot_mask].sum().item())
                hot_sum[f"N@{k}"] += float(dcg[hot_mask].sum().item())
            cold_total += int(cold_mask.sum().item())
            hot_total += int((~cold_mask).sum().item())
    cold_res = {k: v / max(1, cold_total) for k, v in cold_sum.items()} if cold_total else None
    hot_res = {k: v / max(1, hot_total) for k, v in hot_sum.items()} if hot_total else None
    return cold_res, cold_total, hot_res, hot_total


def print_report(title: str, sample_cold, sample_hot, full_cold, full_hot, n_sc, n_sh, n_fc, n_fh) -> None:
    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{os.environ.get('USIM_EVAL_N_NEG', '200')}) vs full ranking ({title})")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for key in METRICS:
        sc = sample_cold.get(key, 0.0) if sample_cold else 0.0
        sh = sample_hot.get(key, 0.0) if sample_hot else 0.0
        fc = full_cold.get(key, 0.0) if full_cold else 0.0
        fh = full_hot.get(key, 0.0) if full_hot else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
    print("-" * 90)
    print(f"Sampled Samples: Cold={n_sc}, Hot={n_sh}")
    print(f"Full Samples: Cold={n_fc}, Hot={n_fh}")
    print("=" * 90)


def main() -> None:
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
    setup_seed(cfg.seed)
    train_df, val_df, test_df = static_split_df(df, seed=cfg.static_seed, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(InteractionDataset(train_df), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(InteractionDataset(val_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(InteractionDataset(test_df), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    item_counts = torch.zeros(cfg.n_items, dtype=torch.long)
    for item_id, count in train_df["i_idx"].astype(int).value_counts().items():
        if 0 <= int(item_id) < cfg.n_items:
            item_counts[int(item_id)] = int(count)
    negative_item_pool = torch.where(item_counts >= cfg.cold_threshold)[0].long()
    if negative_item_pool.numel() == 0:
        negative_item_pool = torch.arange(cfg.n_items, dtype=torch.long)

    print(
        ">> Model: DropoutNet official-protocol adapted | "
        f"device={device} | teacher_epochs={cfg.teacher_epochs} | "
        f"student_epochs={cfg.student_epochs} | item_dropout={cfg.item_pref_dropout:.2f} | "
        f"best_avg={cfg.early_stop_average_mode} | neg_pool={negative_item_pool.numel()} train-warm items"
    )

    teacher = train_teacher(cfg, train_loader, negative_item_pool, device)
    teacher.eval()
    with torch.no_grad():
        teacher_user = F.normalize(teacher.user_emb.weight.detach().cpu(), dim=-1)
        teacher_item = F.normalize(teacher.item_emb.weight.detach().cpu(), dim=-1)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = OfficialStyleDropoutNet(cfg, content_emb, teacher_user, teacher_item).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.student_lr, weight_decay=cfg.weight_decay)
    train_seen = build_user_seen(train_df)
    test_seen = clone_seen(train_seen)
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        for uid, items in build_user_seen(val_df).items():
            test_seen.setdefault(uid, set()).update(items)

    best_val = -math.inf
    best_epoch = -1
    best_state = None
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.ckpt, model, opt, device)
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)
    for epoch in range(start_epoch + 1, cfg.student_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch, _ in train_loader:
            u = batch["u"].to(device)
            pos = batch["i"].to(device)
            neg = sample_negative_items(negative_item_pool, pos.shape, device)
            z_u = F.normalize(model.user_vector(u), dim=-1)
            z_pos = F.normalize(model.item_vector(pos), dim=-1)
            neg_cold = item_counts[neg.detach().cpu()].to(device) < cfg.cold_threshold
            z_neg = F.normalize(mixed_item_vectors(model, neg, neg_cold), dim=-1)
            pos_s = (z_u * z_pos).sum(dim=-1) / cfg.temperature
            neg_s = (z_u * z_neg).sum(dim=-1) / cfg.temperature
            loss = -F.logsigmoid(pos_s - neg_s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"[DropoutNet] Epoch {epoch}/{cfg.student_epochs} loss={total / max(1, steps):.4f}")

        if epoch % cfg.eval_interval == 0 or epoch == cfg.student_epochs:
            improved = False
            all_z = precompute_item_bank(cfg, model, item_counts, device)
            val_cold, _, val_hot, _ = evaluate_full(
                cfg,
                model,
                val_loader,
                all_z,
                item_counts,
                device,
                train_seen,
                average_mode=cfg.early_stop_average_mode,
            )
            val_key = val_cold.get("N@10", 0.0) if val_cold else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                tag = "best"
                improved = True
            else:
                tag = "keep"
            if cfg.ckpt.save and improved:
                save_checkpoint(
                    cfg.ckpt,
                    "best.pt",
                    epoch,
                    model,
                    opt,
                    best_state=best_state,
                    extra={"best_val": best_val, "best_epoch": best_epoch},
                )
            hot_key = val_hot.get("N@10", 0.0) if val_hot else 0.0
            print(f"  [VAL] cold_N@10={val_key:.4f} hot_N@10={hot_key:.4f} | {tag}")
        if cfg.ckpt.save:
            save_checkpoint(
                cfg.ckpt,
                "latest.pt",
                epoch,
                model,
                opt,
                best_state=best_state,
                extra={"best_val": best_val, "best_epoch": best_epoch},
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restore best epoch={best_epoch}, val_full_cold_N@10({cfg.early_stop_average_mode})={best_val:.4f}")

    all_z = precompute_item_bank(cfg, model, item_counts, device)
    full_cold, n_fc, full_hot, n_fh = evaluate_full(cfg, model, test_loader, all_z, item_counts, device, test_seen)
    full_cold_item, n_fc_item, full_hot_item, n_fh_item = evaluate_full(
        cfg,
        model,
        test_loader,
        all_z,
        item_counts,
        device,
        test_seen,
        average_mode="item_macro",
        export_cold_item_metrics_path=static_result_path("per_item_full_cold_dropoutnet_official_static.csv"),
        export_hot_item_metrics_path=static_result_path("per_item_full_hot_dropoutnet_official_static.csv"),
    )
    sample_cold, n_sc, sample_hot, n_sh = evaluate_sampled(cfg, model, test_loader, all_z, item_counts, device, test_seen)
    print_report(
        "DropoutNet Official-Protocol Adapted",
        sample_cold,
        sample_hot,
        full_cold,
        full_hot,
        n_sc,
        n_sh,
        n_fc,
        n_fh,
    )

    official_dir = Path("third_party/DropoutNet")
    out = {
        "model": "DropoutNet (official-protocol adapted)",
        "model_display": "DropoutNet",
        "protocol": "static_item_cold",
        "epoch_tag": os.environ.get(
            "DROPOUT_OFFICIAL_EPOCH_TAG",
            f"teacher{cfg.teacher_epochs}_student{cfg.student_epochs}",
        ),
        "output_dir": os.environ.get("USIM_BASELINE_OUTPUT_DIR", ""),
        "sample_cold": sample_cold or {},
        "sample_hot": sample_hot or {},
        "full_cold": full_cold or {},
        "full_hot": full_hot or {},
        "full_cold_item_macro": full_cold_item or {},
        "full_hot_item_macro": full_hot_item or {},
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item,
        "count_full_hot_item_macro": n_fh_item,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_average_mode": cfg.early_stop_average_mode,
        "best_metric": f"cold_{cfg.early_stop_average_mode}_N@10",
        "eval_interval": cfg.eval_interval,
        "eval_n_neg": cfg.eval_n_neg,
        "teacher_epochs": cfg.teacher_epochs,
        "student_epochs": cfg.student_epochs,
        "item_pref_dropout": cfg.item_pref_dropout,
        "user_pref_dropout": cfg.user_pref_dropout,
        "negative_sampling": "train_warm_items_only",
        "negative_item_pool_size": int(negative_item_pool.numel()),
        "checkpoint_dir": cfg.ckpt.dir or None,
        "teacher_checkpoint_dir": cfg.teacher_ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
        "per_item_full_cold_path": static_result_path("per_item_full_cold_dropoutnet_official_static.csv"),
        "per_item_full_hot_path": static_result_path("per_item_full_hot_dropoutnet_official_static.csv"),
        "official_repo": "https://github.com/layer6ai-labs/DropoutNet",
        "official_source_dir": str(official_dir),
        "official_source_present": official_dir.exists(),
        "note": (
            "Protocol adaptation of official DropoutNet: warm latent teacher + latent-input dropout + "
            "content side features. The upstream repository could not be cloned automatically in this environment "
            "unless official_source_present is true."
        ),
    }
    for key in METRICS:
        out[f"samp_cold_{key}"] = sample_cold.get(key, 0.0) if sample_cold else 0.0
        out[f"samp_hot_{key}"] = sample_hot.get(key, 0.0) if sample_hot else 0.0
        out[f"full_cold_{key}"] = full_cold.get(key, 0.0) if full_cold else 0.0
        out[f"full_hot_{key}"] = full_hot.get(key, 0.0) if full_hot else 0.0

    result_path = static_result_path("dropoutnet_official_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
