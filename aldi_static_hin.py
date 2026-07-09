"""
ALDI static item-cold adaptation for the shared HIN split.

This is a PyTorch adaptation of the official ALDI implementation pulled under
third_party/ALDI. The protocol and evaluator follow the local static item-cold
main-table setup:
  - shared static split via USIM_STATIC_SPLIT_DIR
  - train-only history mask by default
  - best checkpoint selected by validation full cold NDCG@10

ALDI uses a warm BPR teacher and trains content/user mapping networks by
supervised BPR plus ranking, identification, and rating distillation losses.
At ranking time, warm items are scored with the teacher user embedding, while
cold items are scored with the transformed user embedding, matching the
official implementation's get_user_rating behavior.
"""

import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    static_result_path,
    static_split_df,
)
from hin_eval_common import compute_ranking_metric_values, compute_ranking_metrics, print_final_report
from lightgcn_static_hin import compute_bpr_loss, prepare_train_cache, sample_negatives
from baseline_checkpoint import CheckpointConfig, checkpoint_config, maybe_resume_checkpoint, save_checkpoint


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("ALDI_EMB_DIM", "200"))
        self.hidden_dim = int(os.environ.get("ALDI_HIDDEN_DIM", str(self.emb_dim)))
        self.batch_size = int(os.environ.get("ALDI_BATCH_SIZE", "4096"))
        self.eval_batch_size = int(os.environ.get("ALDI_EVAL_BATCH_SIZE", str(self.batch_size)))
        self.bank_batch_size = int(os.environ.get("ALDI_BANK_BATCH_SIZE", "32768"))

        self.teacher_epochs = int(os.environ.get("ALDI_TEACHER_EPOCHS", "200"))
        self.teacher_lr = float(os.environ.get("ALDI_TEACHER_LR", "1e-3"))
        self.teacher_reg = float(os.environ.get("ALDI_TEACHER_REG", "1e-4"))
        self.teacher_eval_interval = int(os.environ.get("ALDI_TEACHER_EVAL_INTERVAL", "20"))

        self.n_epochs = int(os.environ.get("ALDI_STATIC_EPOCHS", "100"))
        self.lr = float(os.environ.get("ALDI_LR", "1e-3"))
        self.reg = float(os.environ.get("ALDI_REG", "1e-4"))
        self.eval_interval = int(os.environ.get("ALDI_EVAL_INTERVAL", "5"))

        self.alpha = float(os.environ.get("ALDI_ALPHA", "0.9"))
        self.beta = float(os.environ.get("ALDI_BETA", "0.05"))
        self.gamma = float(os.environ.get("ALDI_GAMMA", "0.1"))
        self.tws = os.environ.get("ALDI_TWS", "0").strip().lower() in {"1", "true", "yes"}
        self.freq_coef_m = float(os.environ.get("ALDI_FREQ_COEF_M", "4"))

        self.cold_threshold = int(os.environ.get("ALDI_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "5")))
        self.eval_n_neg = int(os.environ.get("ALDI_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")))
        self.static_seed = int(os.environ.get("ALDI_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.seed = int(os.environ.get("ALDI_SEED", str(self.static_seed)))
        self.train_ratio = float(os.environ.get("ALDI_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("ALDI_STATIC_VAL_RATIO", "0.1"))
        self.early_stop_average_mode = os.environ.get(
            "ALDI_EARLY_STOP_AVG_MODE",
            os.environ.get(
                "BASELINE_EARLY_STOP_AVG_MODE",
                os.environ.get("USIM_EARLY_STOP_AVG_MODE", "interaction"),
            ),
        ).strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError(
                "ALDI_EARLY_STOP_AVG_MODE/BASELINE_EARLY_STOP_AVG_MODE/"
                "USIM_EARLY_STOP_AVG_MODE must be interaction or item_macro"
            )
        self.ckpt = checkpoint_config("ALDI")
        teacher_ckpt_dir = os.environ.get("ALDI_TEACHER_CKPT_DIR", "").strip()
        if not teacher_ckpt_dir and self.ckpt.dir:
            teacher_ckpt_dir = os.path.join(self.ckpt.dir, "teacher")
        self.teacher_ckpt = CheckpointConfig(
            dir=teacher_ckpt_dir,
            save=self.ckpt.save,
            resume=self.ckpt.resume,
            force_fresh=self.ckpt.force_fresh,
            save_opt=self.ckpt.save_opt,
        )


class BPRTeacher(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.emb_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)


class ALDIStudent(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.item_map = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )
        self.user_map = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )

    def map_items(self, item_content: torch.Tensor) -> torch.Tensor:
        return self.item_map(item_content)

    def map_users(self, teacher_user_emb: torch.Tensor) -> torch.Tensor:
        return self.user_map(teacher_user_emb)

    def distill_loss(
        self,
        pos_content: torch.Tensor,
        pos_teacher_item: torch.Tensor,
        neg_content: torch.Tensor,
        neg_teacher_item: torch.Tensor,
        teacher_user: torch.Tensor,
        pos_freq: torch.Tensor,
        neg_freq: torch.Tensor,
        freq_coef_a: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        pos_gen = self.map_items(pos_content)
        neg_gen = self.map_items(neg_content)
        mapped_user = self.map_users(teacher_user)

        student_pos = (mapped_user * pos_gen).sum(dim=1)
        student_neg = (mapped_user * neg_gen).sum(dim=1)
        student_rank = student_pos - student_neg
        supervised = F.binary_cross_entropy_with_logits(student_rank, torch.ones_like(student_rank))

        with torch.no_grad():
            teacher_pos = (teacher_user * pos_teacher_item).sum(dim=1)
            teacher_neg = (teacher_user * neg_teacher_item).sum(dim=1)
            teacher_rank = teacher_pos - teacher_neg
            rank_target = torch.sigmoid(teacher_rank)

        if cfg.tws:
            item_weight = torch.tanh(freq_coef_a * torch.cat([pos_freq, neg_freq], dim=0))
            item_weight = item_weight.clamp(min=0.0, max=float(np.tanh(cfg.freq_coef_m)))
            pos_weight = item_weight[:pos_freq.size(0)]
        else:
            pos_weight = torch.ones_like(student_rank)

        rank_distill = cfg.alpha * (
            pos_weight * F.binary_cross_entropy_with_logits(student_rank, rank_target, reduction="none")
        ).mean()

        student_ii = (pos_gen * pos_gen).sum(dim=1)
        student_ij = torch.matmul(pos_gen, neg_gen.t()).mean(dim=1)
        student_iden = student_ii - student_ij
        with torch.no_grad():
            teacher_ii = (pos_teacher_item * pos_teacher_item).sum(dim=1)
            teacher_ij = torch.matmul(pos_teacher_item, neg_teacher_item.t()).mean(dim=1)
            iden_target = torch.sigmoid(teacher_ii - teacher_ij)
        iden_distill = cfg.beta * (
            pos_weight * F.binary_cross_entropy_with_logits(student_iden, iden_target, reduction="none")
        ).mean()

        rating_distill = cfg.gamma * (
            torch.abs(teacher_pos.detach() - student_pos) + torch.abs(teacher_neg.detach() - student_neg)
        ).mean()

        loss = supervised + rank_distill + iden_distill + rating_distill
        parts = {
            "supervised": float(supervised.detach().cpu().item()),
            "rank": float(rank_distill.detach().cpu().item()),
            "iden": float(iden_distill.detach().cpu().item()),
            "rating": float(rating_distill.detach().cpu().item()),
        }
        return loss, parts


def build_item_train_counts(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    counts = torch.zeros(n_items, dtype=torch.long)
    value_counts = train_df["i_idx"].astype(int).value_counts()
    for item_id, count in value_counts.items():
        idx = int(item_id)
        if 0 <= idx < n_items:
            counts[idx] = int(count)
    return counts


def compute_aldi_item_frequency(train_df: pd.DataFrame, n_users: int, n_items: int) -> torch.Tensor:
    user_degree = train_df.groupby("u_idx").size().to_dict()
    item_users = train_df.groupby("i_idx")["u_idx"].unique()
    freq = torch.ones(n_items, dtype=torch.float32)
    for item_id, users in item_users.items():
        total = 0.0
        for user_id in users:
            degree = int(user_degree.get(int(user_id), 0))
            if degree > 0:
                total += 1.0 / degree
        idx = int(item_id)
        if 0 <= idx < n_items:
            freq[idx] = max(total, 1e-6)
    return freq


def train_teacher(
    cfg: Config,
    teacher: BPRTeacher,
    train_df: pd.DataFrame,
    val_loader: DataLoader,
    train_seen: Dict[int, set],
    device: torch.device,
) -> Tuple[int, float]:
    train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=cfg.teacher_lr)
    n_train = train_users_t.numel()

    best_epoch = -1
    best_hot = -1.0
    best_state = None
    k_list = [5, 10, 20]
    print(
        f"Teacher BPR: epochs={cfg.teacher_epochs}, emb_dim={cfg.emb_dim}, "
        f"eval_interval={cfg.teacher_eval_interval}"
    )
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.teacher_ckpt, teacher, optimizer, device)
    best_hot = float(ckpt_state.get("best_val", best_hot))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)

    for epoch in range(start_epoch + 1, cfg.teacher_epochs + 1):
        teacher.train()
        neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
        neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = compute_bpr_loss(
                teacher.user_emb.weight,
                teacher.item_emb.weight,
                train_users_t[idx],
                train_pos_t[idx],
                neg_t[idx],
                reg_weight=cfg.teacher_reg,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        do_eval = (epoch % cfg.teacher_eval_interval == 0) or (epoch == cfg.teacher_epochs)
        if do_eval:
            improved = False
            teacher.eval()
            with torch.no_grad():
                user_bank = F.normalize(teacher.user_emb.weight, dim=1)
                item_bank = F.normalize(teacher.item_emb.weight, dim=1)
                hot_metrics, hot_count = evaluate_teacher_ranker(
                    val_loader,
                    device=device,
                    cfg=cfg,
                    user_bank=user_bank,
                    item_bank=item_bank,
                    eval_type="hot",
                    full_ranking=True,
                    user_seen_items=train_seen,
                )
                hot_n10 = hot_metrics.get("N@10", 0.0) if hot_metrics else 0.0
            if hot_n10 > best_hot:
                best_hot = hot_n10
                best_epoch = epoch
                best_state = copy.deepcopy(teacher.state_dict())
                improved = True
            if cfg.teacher_ckpt.save and improved:
                save_checkpoint(
                    cfg.teacher_ckpt,
                    "best.pt",
                    epoch,
                    teacher,
                    optimizer,
                    best_state=best_state,
                    extra={"best_val": best_hot, "best_epoch": best_epoch},
                )
            print(
                f"Teacher Epoch [{epoch}/{cfg.teacher_epochs}] "
                f"loss={epoch_loss / max(1, n_batches):.4f} | val_full_hot_N@10={hot_n10:.4f} "
                f"| hot_count={hot_count}"
            )
        else:
            print(f"Teacher Epoch [{epoch}/{cfg.teacher_epochs}] loss={epoch_loss / max(1, n_batches):.4f}")
        if cfg.teacher_ckpt.save:
            save_checkpoint(
                cfg.teacher_ckpt,
                "latest.pt",
                epoch,
                teacher,
                optimizer,
                best_state=best_state,
                extra={"best_val": best_hot, "best_epoch": best_epoch},
            )

    if best_state is not None:
        teacher.load_state_dict(best_state)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    print(f"Restore teacher best epoch={best_epoch}, val_full_hot_N@10={best_hot:.4f}")
    return best_epoch, best_hot


def evaluate_teacher_ranker(
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    user_bank: torch.Tensor,
    item_bank: torch.Tensor,
    eval_type: str,
    full_ranking: bool,
    user_seen_items: Optional[Dict[int, set]],
) -> Tuple[Optional[Dict[str, float]], int]:
    accum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in [5, 10, 20]}
    total = 0
    seen_cache: Dict[int, Optional[torch.Tensor]] = {}
    all_item_idx = torch.arange(cfg.n_items, device=device, dtype=torch.long)
    with torch.no_grad():
        for batch, pop in loader:
            if eval_type == "cold":
                mask = pop < cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)
            if int(mask.sum().item()) < 1:
                continue
            u = batch["u"][mask].to(device)
            i = batch["i"][mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            z_u = user_bank[u]
            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                if user_seen_items:
                    rows = torch.arange(u.size(0), device=device)
                    target_scores = scores[rows, i].clone()
                    for row, uid in enumerate(user_ids):
                        if uid not in seen_cache:
                            seen = user_seen_items.get(uid)
                            seen_cache[uid] = (
                                torch.tensor([x for x in seen if 0 <= x < cfg.n_items], device=device, dtype=torch.long)
                                if seen else None
                            )
                        seen_idx = seen_cache[uid]
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                    scores[rows, i] = target_scores
                target = i
            else:
                n_neg = min(cfg.eval_n_neg, cfg.n_items - 1)
                neg_items = torch.empty((u.size(0), n_neg), dtype=torch.long, device=device)
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(cfg.n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    if user_seen_items:
                        seen = user_seen_items.get(uid)
                        if seen:
                            seen_idx = torch.tensor(
                                [x for x in seen if 0 <= x < cfg.n_items],
                                device=device,
                                dtype=torch.long,
                            )
                            forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg]
                    neg_items[row] = candidates[pick]
                cand = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                perm = torch.argsort(torch.rand(cand.size(0), cand.size(1), device=device), dim=1)
                cand = cand.gather(1, perm)
                scores = torch.bmm(item_bank[cand], z_u.unsqueeze(2)).squeeze(2)
                target = (cand == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)
            res = compute_ranking_metrics(scores, target, k_list=(5, 10, 20))
            n = int(u.size(0))
            for key, val in res.items():
                accum[key] += val * n
            total += n
    if total < 1:
        return None, 0
    return {k: v / total for k, v in accum.items()}, total


def precompute_aldi_banks(
    cfg: Config,
    teacher: BPRTeacher,
    student: ALDIStudent,
    content_emb: torch.Tensor,
    item_counts: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    student.eval()
    teacher.eval()
    item_is_cold = (item_counts < cfg.cold_threshold).to(device)
    content_emb = content_emb.to(device)
    with torch.no_grad():
        teacher_user = F.normalize(teacher.user_emb.weight, dim=1)
        teacher_item = teacher.item_emb.weight.detach().clone()
        gen_item = teacher_item.clone()
        cold_idx = torch.where(item_is_cold)[0]
        for start in range(0, cold_idx.numel(), cfg.bank_batch_size):
            idx = cold_idx[start:start + cfg.bank_batch_size]
            if idx.numel() > 0:
                gen_item[idx] = student.map_items(content_emb[idx])
        item_bank = F.normalize(gen_item, dim=1)

        mapped_chunks = []
        user_weight = teacher.user_emb.weight.detach()
        for start in range(0, cfg.n_users, cfg.bank_batch_size):
            mapped = student.map_users(user_weight[start:start + cfg.bank_batch_size])
            mapped_chunks.append(mapped)
        mapped_user = F.normalize(torch.cat(mapped_chunks, dim=0), dim=1)
    return teacher_user, mapped_user, item_bank, item_is_cold


def _score_candidates(
    teacher_user: torch.Tensor,
    mapped_user: torch.Tensor,
    item_bank: torch.Tensor,
    item_is_cold: torch.Tensor,
    users: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    z_teacher = teacher_user[users]
    z_mapped = mapped_user[users]
    if candidates.dim() == 1:
        score_hot = torch.mm(z_teacher, item_bank.t())
        score_cold = torch.mm(z_mapped, item_bank.t())
        return torch.where(item_is_cold.view(1, -1), score_cold, score_hot)
    cand_vec = item_bank[candidates]
    score_hot = torch.bmm(cand_vec, z_teacher.unsqueeze(2)).squeeze(2)
    score_cold = torch.bmm(cand_vec, z_mapped.unsqueeze(2)).squeeze(2)
    return torch.where(item_is_cold[candidates], score_cold, score_hot)


def evaluate_aldi_ranker(
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    teacher_user: torch.Tensor,
    mapped_user: torch.Tensor,
    item_bank: torch.Tensor,
    item_is_cold: torch.Tensor,
    full_ranking: bool,
    user_seen_items: Optional[Dict[int, set]],
    average_mode: str = "interaction",
    export_cold_item_metrics_path: Optional[str] = None,
    export_hot_item_metrics_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, float]], int, Optional[Dict[str, float]], int]:
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    k_list = [5, 10, 20]
    cold_sum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in k_list}
    hot_sum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in k_list}
    cold_total = 0
    hot_total = 0
    cold_item_sum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    hot_item_sum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    cold_item_count: Dict[int, int] = {}
    hot_item_count: Dict[int, int] = {}
    seen_cache: Dict[int, Optional[torch.Tensor]] = {}
    all_item_idx = torch.arange(cfg.n_items, device=device, dtype=torch.long)

    with torch.no_grad():
        for batch, pop in loader:
            u = batch["u"].to(device)
            i = batch["i"].to(device)
            pop = pop.to(device)
            group_is_cold = pop < cfg.cold_threshold
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

            if full_ranking:
                scores = _score_candidates(teacher_user, mapped_user, item_bank, item_is_cold, u, all_item_idx)
                rows = torch.arange(u.size(0), device=device)
                target_scores = scores[rows, i].clone()
                if user_seen_items:
                    for row, uid in enumerate(user_ids):
                        if uid not in seen_cache:
                            seen = user_seen_items.get(uid)
                            seen_cache[uid] = (
                                torch.tensor([x for x in seen if 0 <= x < cfg.n_items], device=device, dtype=torch.long)
                                if seen else None
                            )
                        seen_idx = seen_cache[uid]
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                scores[rows, i] = target_scores
                target = i
            else:
                n_neg_eff = min(cfg.eval_n_neg, max(1, cfg.n_items - 1))
                pools = []
                min_pool = cfg.n_items - 1
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(cfg.n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    if user_seen_items:
                        seen = user_seen_items.get(uid)
                        if seen:
                            seen_idx = torch.tensor(
                                [x for x in seen if 0 <= x < cfg.n_items],
                                device=device,
                                dtype=torch.long,
                            )
                            forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pools.append(candidates)
                    min_pool = min(min_pool, int(candidates.numel()))
                n_neg_batch = min(n_neg_eff, max(1, min_pool))
                neg = torch.empty((u.size(0), n_neg_batch), dtype=torch.long, device=device)
                for row, candidates in enumerate(pools):
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg[row] = candidates[pick]
                cand = torch.cat([i.unsqueeze(1), neg], dim=1)
                perm = torch.argsort(torch.rand(cand.size(0), cand.size(1), device=device), dim=1)
                cand = cand.gather(1, perm)
                scores = _score_candidates(teacher_user, mapped_user, item_bank, item_is_cold, u, cand)
                target = (cand == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            n_cold = int(group_is_cold.sum().item())
            n_hot = int((~group_is_cold).sum().item())
            if average_mode == "item_macro":
                values = compute_ranking_metric_values(scores, target, k_list)
                item_ids = [int(x) for x in i.detach().cpu().tolist()]
                cold_flags = [bool(x) for x in group_is_cold.detach().cpu().tolist()]
                for item_id, is_cold_item in zip(item_ids, cold_flags):
                    counts = cold_item_count if is_cold_item else hot_item_count
                    counts[item_id] = counts.get(item_id, 0) + 1
                for key, row_values in values.items():
                    vals = [float(x) for x in row_values.detach().cpu().tolist()]
                    for row, item_id in enumerate(item_ids):
                        sums = cold_item_sum if cold_flags[row] else hot_item_sum
                        sums[key][item_id] = sums[key].get(item_id, 0.0) + vals[row]
            else:
                accumulate_group_metrics(scores, target, group_is_cold, k_list, cold_sum, hot_sum)
                cold_total += n_cold
                hot_total += n_hot

    if average_mode == "item_macro":
        def macro_result(item_sum, item_count, export_path: Optional[str] = None):
            if not item_count:
                return None, 0
            res = {}
            for key, per_item in item_sum.items():
                vals = [
                    per_item.get(item_id, 0.0) / count
                    for item_id, count in item_count.items()
                    if count > 0
                ]
                res[key] = sum(vals) / max(1, len(vals))
            if export_path:
                rows = []
                for item_id in sorted(item_count):
                    count = max(1, int(item_count[item_id]))
                    row = {"item_id": int(item_id), "count": int(item_count[item_id])}
                    for key, per_item in item_sum.items():
                        row[key] = float(per_item.get(item_id, 0.0) / count)
                    rows.append(row)
                os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
                pd.DataFrame(rows).to_csv(export_path, index=False)
            return res, len(item_count)

        cold_res, cold_count = macro_result(cold_item_sum, cold_item_count, export_cold_item_metrics_path)
        hot_res, hot_count = macro_result(hot_item_sum, hot_item_count, export_hot_item_metrics_path)
        return cold_res, cold_count, hot_res, hot_count

    cold_res = {k: v / cold_total for k, v in cold_sum.items()} if cold_total > 0 else None
    hot_res = {k: v / hot_total for k, v in hot_sum.items()} if hot_total > 0 else None
    return cold_res, cold_total, hot_res, hot_total


def accumulate_group_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    is_cold: torch.Tensor,
    k_list,
    cold_sum: Dict[str, float],
    hot_sum: Dict[str, float],
) -> None:
    max_k = min(max(k_list), scores.size(1))
    _, topk = torch.topk(scores, k=max_k, dim=1)
    target_view = target.view(-1, 1)
    hot_mask = ~is_cold
    for k in k_list:
        preds = topk[:, :k]
        hits = (preds == target_view).any(dim=1).float()
        cold_sum[f"R@{k}"] += hits[is_cold].sum().item() if int(is_cold.sum().item()) > 0 else 0.0
        hot_sum[f"R@{k}"] += hits[hot_mask].sum().item() if int(hot_mask.sum().item()) > 0 else 0.0

        vals = torch.zeros(scores.size(0), device=scores.device)
        rks = (preds == target_view).nonzero(as_tuple=True)
        if rks[0].numel() > 0:
            vals[rks[0]] = 1.0 / torch.log2(rks[1].float() + 2.0)
        cold_sum[f"N@{k}"] += vals[is_cold].sum().item() if int(is_cold.sum().item()) > 0 else 0.0
        hot_sum[f"N@{k}"] += vals[hot_mask].sum().item() if int(hot_mask.sum().item()) > 0 else 0.0


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading data from {data_dir} ...")
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
    setup_seed(cfg.seed)

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    print(
        f"Static split done: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}"
    )

    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    if os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower() == "train_val":
        add_user_seen_from_df(test_seen, val_df)

    item_counts = build_item_train_counts(train_df, cfg.n_items)
    item_freq = compute_aldi_item_frequency(train_df, cfg.n_users, cfg.n_items)
    x_expect = (len(train_df) / max(1, cfg.n_items)) * (1.0 / max(1e-12, len(train_df) / max(1, cfg.n_users)))
    freq_coef_a = cfg.freq_coef_m / max(x_expect, 1e-12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_emb = content_emb.float().to(device)
    item_freq = item_freq.to(device)
    teacher = BPRTeacher(cfg).to(device)
    student = ALDIStudent(cfg).to(device)
    print(
        f"Model: ALDI official-adapted | device={device} | student_epochs={cfg.n_epochs} | "
        f"alpha={cfg.alpha}, beta={cfg.beta}, gamma={cfg.gamma}, tws={int(cfg.tws)}"
    )

    teacher_epoch, teacher_hot = train_teacher(cfg, teacher, train_df, val_loader, train_seen, device)

    train_users_np, train_pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    train_users_t = torch.tensor(train_users_np, dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=device)
    n_train = train_users_t.numel()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.reg)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    k_list = [5, 10, 20]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]
    start_epoch, ckpt_state = maybe_resume_checkpoint(cfg.ckpt, student, optimizer, device)
    best_val = float(ckpt_state.get("best_val", best_val))
    best_epoch = int(ckpt_state.get("best_epoch", best_epoch))
    best_state = ckpt_state.get("best_state", best_state)

    for epoch in range(start_epoch + 1, cfg.n_epochs + 1):
        student.train()
        neg_np = sample_negatives(train_pos_np, user_rows, user_neg_pool, cfg.n_items)
        neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        loss_parts = {"supervised": 0.0, "rank": 0.0, "iden": 0.0, "rating": 0.0}
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            u = train_users_t[idx]
            pos = train_pos_t[idx]
            neg = neg_t[idx]
            with torch.no_grad():
                true_user = teacher.user_emb.weight[u]
                pos_item = teacher.item_emb.weight[pos]
                neg_item = teacher.item_emb.weight[neg]
            loss, parts = student.distill_loss(
                content_emb[pos],
                pos_item,
                content_emb[neg],
                neg_item,
                true_user,
                item_freq[pos],
                item_freq[neg],
                freq_coef_a=freq_coef_a,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            for key in loss_parts:
                loss_parts[key] += parts[key]
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        part_msg = " ".join([f"{k}={v / max(1, n_batches):.4f}" for k, v in loss_parts.items()])
        do_eval = (epoch % cfg.eval_interval == 0) or (epoch == cfg.n_epochs)
        if do_eval:
            improved = False
            teacher_user, mapped_user, item_bank, item_is_cold = precompute_aldi_banks(
                cfg, teacher, student, content_emb, item_counts, device
            )
            val_cold, n_vc, _, _ = evaluate_aldi_ranker(
                val_loader,
                device,
                cfg,
                teacher_user,
                mapped_user,
                item_bank,
                item_is_cold,
                full_ranking=True,
                user_seen_items=train_seen,
                average_mode=cfg.early_stop_average_mode,
            )
            val_key = val_cold.get("N@10", 0.0) if val_cold else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_state = copy.deepcopy(student.state_dict())
                improved = True
            if cfg.ckpt.save and improved:
                save_checkpoint(
                    cfg.ckpt,
                    "best.pt",
                    epoch,
                    student,
                    optimizer,
                    best_state=best_state,
                    extra={"best_val": best_val, "best_epoch": best_epoch},
                )
            print(
                f"ALDI Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg} | "
                f"val_full_cold_N@10({cfg.early_stop_average_mode})={val_key:.4f} | "
                f"val_cold_count={n_vc}"
            )
        else:
            print(f"ALDI Epoch [{epoch}/{cfg.n_epochs}] loss={avg_loss:.4f} | {part_msg}")
        if cfg.ckpt.save:
            save_checkpoint(
                cfg.ckpt,
                "latest.pt",
                epoch,
                student,
                optimizer,
                best_state=best_state,
                extra={"best_val": best_val, "best_epoch": best_epoch},
            )

    if best_state is not None:
        student.load_state_dict(best_state)
    print(
        f"Restore ALDI best epoch={best_epoch}, "
        f"val_full_cold_N@10({cfg.early_stop_average_mode})={best_val:.4f}"
    )

    teacher_user, mapped_user, item_bank, item_is_cold = precompute_aldi_banks(
        cfg, teacher, student, content_emb, item_counts, device
    )
    sample_cold, n_sc, sample_hot, n_sh = evaluate_aldi_ranker(
        test_loader,
        device,
        cfg,
        teacher_user,
        mapped_user,
        item_bank,
        item_is_cold,
        full_ranking=False,
        user_seen_items=test_seen,
    )
    full_cold, n_fc, full_hot, n_fh = evaluate_aldi_ranker(
        test_loader,
        device,
        cfg,
        teacher_user,
        mapped_user,
        item_bank,
        item_is_cold,
        full_ranking=True,
        user_seen_items=test_seen,
    )
    full_cold_item_macro, n_fc_item_macro, full_hot_item_macro, n_fh_item_macro = evaluate_aldi_ranker(
        test_loader,
        device,
        cfg,
        teacher_user,
        mapped_user,
        item_bank,
        item_is_cold,
        full_ranking=True,
        user_seen_items=test_seen,
        average_mode="item_macro",
        export_cold_item_metrics_path=static_result_path("per_item_full_cold_aldi_static.csv"),
        export_hot_item_metrics_path=static_result_path("per_item_full_hot_aldi_static.csv"),
    )

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
    full_cold_item_macro = full_cold_item_macro or {}
    full_hot_item_macro = full_hot_item_macro or {}
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
        title="ALDI Static HIN (official-adapted)",
    )

    out = {
        "model": "ALDI",
        "model_display": "ALDI (official-adapted)",
        "source": "Official ALDI source pulled under third_party/ALDI; PyTorch static-HIN adaptation.",
        "protocol": "static_item_cold",
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item_macro,
        "full_hot_item_macro": full_hot_item_macro,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item_macro,
        "count_full_hot_item_macro": n_fh_item_macro,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_metric": "cold",
        "best_average_mode": cfg.early_stop_average_mode,
        "teacher_best_epoch": teacher_epoch,
        "teacher_best_val_full_hot_n10": teacher_hot,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "gamma": cfg.gamma,
        "tws": int(cfg.tws),
        "checkpoint_dir": cfg.ckpt.dir or None,
        "teacher_checkpoint_dir": cfg.teacher_ckpt.dir or None,
        "resumed_from_epoch": start_epoch,
        "per_item_full_cold_path": static_result_path("per_item_full_cold_aldi_static.csv"),
        "per_item_full_hot_path": static_result_path("per_item_full_hot_aldi_static.csv"),
        "note": (
            "Warm BPR teacher is trained on the shared static train split; "
            f"student checkpoint selected by validation full cold N@10 ({cfg.early_stop_average_mode})."
        ),
    }
    result_path = static_result_path("aldi_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
