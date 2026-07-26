"""
usim_feedback_fast2.py — P1 改进版
在 fast 版基础上增加:
  P1-1. 用户行为序列建模: 注意力池化最近交互课程，增强用户表征
  P1-2. PPO epochs 从 5 降至 2: 减少 60% PPO 计算开销
"""
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from usim import (
    StreamDataset,
    _add_user_seen_from_df,
    build_eval_item_vecs,
    _build_eval_pos_item_vecs,
    _build_llm_score_tensor,
    _select_eval_item_bank,
    compute_ranking_metrics,
    build_course_artifacts,
    collate_fn,
    setup_seed,
    split_dataframe_by_periods,
)
from usim_feedback import (
    FeedbackConfig,
    _format_eta,
    _should_log_train_progress,
    _feedback_ckpt_enabled,
    _feedback_ckpt_auto_resume,
    _feedback_ckpt_force_fresh,
    _deserialize_user_seen_items,
    _save_feedback_checkpoint,
    _load_feedback_checkpoint,
    _move_state_to_cpu,
    _optimizer_state_to_device,
    _maybe_clear_cuda_cache,
    _build_feedback_ckpt_state,
)
from usim_feedback_fast import FastFeedbackUSIM


# ---- 配置 ----
class Fast2Config(FeedbackConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", "2"))
        self.stream_train_window = int(os.environ.get("USIM_TRAIN_WINDOW", "0"))
        # 用户序列参数
        self.user_seq_len = int(os.environ.get("USIM_SEQ_LEN", "20"))


# ---- P1-1: 用户行为序列建模 ----
class SeqFeedbackUSIM(FastFeedbackUSIM):
    """在 FastFeedbackUSIM 基础上加入用户行为序列建模。"""

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        dim = config.emb_dim  # 128
        self.seq_len = config.user_seq_len

        # 注意力池化: 对最近 K 个交互 item 做加权平均
        self.seq_attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        # 门控融合: ID embedding + 序列特征 → 最终用户向量
        self.user_seq_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.user_seq_proj = nn.Linear(dim * 2, dim)

    def get_user_vector(self, u, user_seen_items=None):
        """计算序列增强的用户向量。"""
        z_u_id = self.user_proj(self.user_emb(u))  # [B, 128]

        if user_seen_items is None or self.seq_len <= 0:
            return z_u_id

        batch_size = u.size(0)
        dim = z_u_id.size(1)
        device = u.device

        # 收集每个用户最近 seq_len 个交互 item
        seq_indices = []
        seq_lengths = []
        for uid in u.detach().cpu().tolist():
            seen = user_seen_items.get(int(uid))
            if seen:
                recent = list(seen)[-self.seq_len:]
                recent = [it for it in recent if 0 <= it < self.cfg.n_items]
            else:
                recent = []
            seq_lengths.append(len(recent))
            # 填充到 seq_len
            padded = recent + [0] * (self.seq_len - len(recent))
            seq_indices.append(padded)

        seq_idx = torch.tensor(seq_indices, dtype=torch.long, device=device)  # [B, K]
        seq_lens = torch.tensor(seq_lengths, dtype=torch.long, device=device)  # [B]

        has_seq = seq_lens > 0
        if not has_seq.any():
            return z_u_id

        # 获取 item embeddings (使用 ID embedding，不是 content)
        with torch.no_grad():
            seq_embs = self.item_id_emb(seq_idx)  # [B, K, 128]

        # 注意力权重
        attn_scores = self.seq_attn(seq_embs).squeeze(-1)  # [B, K]

        # Mask padding positions
        mask = torch.arange(self.seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)  # [B, K]
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, K]

        # 加权池化
        z_u_seq = (attn_weights.unsqueeze(-1) * seq_embs).sum(dim=1)  # [B, 128]

        # 没有序列的用户回退到纯 ID
        z_u_seq[~has_seq] = 0.0

        # 门控融合
        combined = torch.cat([z_u_id, z_u_seq], dim=1)  # [B, 256]
        gate = self.user_seq_gate(combined)  # [B, 128], sigmoid 输出
        z_u_fused = gate * z_u_id + (1.0 - gate) * self.user_seq_proj(combined)

        return z_u_fused

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold

        # P1-1: 使用序列增强的用户向量
        z_u_base = self.get_user_vector(u, user_seen_items)
        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)

        target_emb = z_i_base.detach().clone()

        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base, target_emb, user_bank_raw=user_bank_raw,
            item_idx=i, target_pop=pop, user_seen_items=user_seen_items,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0), device=self.device)

        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp

        if self.training and self.cfg.use_mixed_hard_neg and logits_margin.size(0) > 1:
            batch_size = logits_margin.size(0)
            max_neg = batch_size - 1
            n_total_neg = min(self.cfg.train_num_negs, max_neg)
            if n_total_neg > 0:
                n_hard = int(n_total_neg * self.cfg.hard_neg_ratio)
                n_hard = max(0, min(n_hard, n_total_neg))
                n_rand = n_total_neg - n_hard
                neg_logits = logits_margin.clone()
                neg_logits[pos_mask] = -1e9
                hard_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                rand_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                if n_hard > 0:
                    if self.cfg.use_structured_hard_neg and self.item_hard_adj is not None:
                        hard_mask = self.item_hard_adj[i][:, i]
                        hard_mask = hard_mask & (~pos_mask)
                        hard_logits = neg_logits.masked_fill(~hard_mask, -1e9)
                        hard_scores, hard_idx = torch.topk(hard_logits, k=n_hard, dim=1)
                        valid_mask = hard_scores > -1e8
                        if (~valid_mask).any():
                            bad_rows = torch.nonzero((~valid_mask).any(dim=1), as_tuple=False).view(-1).tolist()
                            for row in bad_rows:
                                need = int((~valid_mask[row]).sum().item())
                                if need < 1:
                                    continue
                                fallback = neg_logits[row].clone()
                                if valid_mask[row].any():
                                    fallback[hard_idx[row, valid_mask[row]]] = -1e9
                                _, fill_idx = torch.topk(fallback, k=need, dim=0)
                                hard_idx[row, ~valid_mask[row]] = fill_idx
                    else:
                        _, hard_idx = torch.topk(neg_logits, k=n_hard, dim=1)
                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(rand_scores, k=n_rand, dim=1)
                cand_idx = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                main_loss = F.cross_entropy(cand_logits, main_targets)
            else:
                main_loss = F.cross_entropy(logits_margin, labels)
        else:
            main_loss = F.cross_entropy(logits_margin, labels)

        z_id = F.normalize(id_e_true, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            batch_size_p = logits.size(0)
            user_ids_p = [int(x) for x in u.detach().cpu().tolist()]
            seen_mat, _ = self._build_seen_mat(user_ids_p, user_seen_items)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(seen_mat, prereq_mat_batch.t())
            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0 - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch)
            ).clamp(0.0, 1.0)
            valid_rows = seen_mat.sum(dim=1) >= float(self.cfg.prereq_aux_min_seen)
            if self.cfg.prereq_aux_only_cold:
                valid_rows = valid_rows & is_cold
            unmet_mask = violation_batch > float(self.cfg.prereq_aux_violation_thr)
            unmet_mask = unmet_mask & (~pos_mask)
            candidate_mask = unmet_mask & valid_rows.unsqueeze(1)
            if candidate_mask.any():
                neg_scores = logits.masked_fill(~candidate_mask, -1e9)
                neg_vals, _ = neg_scores.max(dim=1)
                has_neg = neg_vals > -1e8
                if has_neg.any():
                    pos_vals = logits[torch.arange(batch_size_p, device=self.device), labels]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss
        )
        return total_loss, candidate_stats


# ---- 序列感知评估函数 ----
def evaluate_usim_seq(model, loader, device, llm_scores, k_list=[5, 10, 20], n_neg=200,
                      eval_type='cold', full_ranking=False, user_seen_items=None, all_item_vecs=None):
    """与 usim.evaluate_usim 相同，但调用 model.get_user_vector 获取序列增强用户向量。"""
    model.eval()
    accum_metrics = {}
    total_samples = 0
    seen_tensor_cache = {}

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        if all_item_vecs is None:
            all_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
        item_bank = _select_eval_item_bank(all_item_vecs, eval_type)

        for batch, pop, llm in loader:
            if eval_type == 'cold':
                mask = pop < model.cfg.cold_threshold
            elif eval_type == 'hot':
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)
            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            item_ids = [int(x) for x in i.detach().cpu().tolist()]

            for uid in user_ids:
                if uid in seen_tensor_cache:
                    continue
                seen_items = user_seen_items.get(uid) if user_seen_items else None
                if seen_items:
                    seen_list = [it for it in seen_items if 0 <= it < n_items]
                    seen_tensor_cache[uid] = torch.tensor(seen_list, dtype=torch.long, device=device) if seen_list else None
                else:
                    seen_tensor_cache[uid] = None

            # 核心区别: 使用序列增强的用户向量
            z_u = F.normalize(model.get_user_vector(u, user_seen_items), dim=1)

            pos_llm = _build_llm_score_tensor(llm_scores, user_ids, item_ids, device=device)
            pos_vec = _build_eval_pos_item_vecs(model, i, pos_llm, pop_sel, eval_type)
            pos_scores = (z_u * pos_vec).sum(dim=1)

            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                row_idx = torch.arange(n_sel, device=device)
                target_scores = pos_scores.clone()
                if user_seen_items:
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache[uid]
                        if seen_idx is None:
                            continue
                        scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                else:
                    scores[row_idx, i] = target_scores
                scores = model.apply_course_rerank(
                    scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=pop_sel
                )
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache[uid]
                    avail = n_items - 1 - (int((seen_idx != i[row]).sum().item()) if seen_idx is not None else 0)
                    avail_counts.append(max(1, avail))
                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row, user_id in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache[int(user_id)]
                    if seen_idx is not None:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]
                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                cand_vecs = item_bank[cand_idx].clone()
                cand_vecs[:, 0, :] = pos_vec
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                scores = model.apply_course_rerank(
                    scores, user_ids, seen_tensor_cache, cand_idx=cand_idx, target_pop=pop_sel
                )
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)

            batch_res = compute_ranking_metrics(scores, target_indices=target_indices, k_list=k_list)
            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


# ---- Checkpoint 隔离 ----
def _feedback_ckpt_dir():
    return os.environ.get("USIM_FB_CKPT_DIR", os.path.join("checkpoints", "usim_feedback_fast2"))


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Feedback-Aware USIM (FAST2) from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("错误: 请先运行 data_process_hin.py")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    # ---- P1-2: 使用 Fast2Config ----
    cfg = Fast2Config(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    course_artifacts, course_stats = build_course_artifacts(
        df, cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    pop_stats = df.groupby("i_idx")["popularity"].max()
    for item_id, pop_value in pop_stats.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_final_pop[idx] = int(pop_value)

    model = SeqFeedbackUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> 架构: Feedback-Aware RL-USIM + InfoNCE [FAST2] (Batch Size={cfg.batch_size})")
    print(f">> P1-1 用户序列建模: seq_len={cfg.user_seq_len} (0=禁用)")
    print(f">> P1-2 PPO epochs: {cfg.ppo_epochs} (原版=5)")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {"Period": [], "Count_cold": [], "Count_hot": []}
    for prefix in ["cold_", "hot_"]:
        for key in metrics_keys:
            history[prefix + key] = []

    accum_cold = {key: 0.0 for key in metrics_keys}
    accum_hot = {key: 0.0 for key in metrics_keys}
    count_cold, count_hot = 0, 0
    full_cold = {key: 0.0 for key in metrics_keys}
    full_hot = {key: 0.0 for key in metrics_keys}
    fc_cold, fc_hot = 0, 0

    warmup_periods = 3
    accumulated_dfs = []
    user_seen_items = {}
    ckpt_dir = _feedback_ckpt_dir()
    ckpt_enabled = _feedback_ckpt_enabled()
    auto_resume = _feedback_ckpt_auto_resume()
    force_fresh = _feedback_ckpt_force_fresh()
    print(
        f">> Checkpoint: enabled={ckpt_enabled} | auto_resume={auto_resume} | "
        f"force_fresh={force_fresh} | dir={ckpt_dir}"
    )

    start_period = 0
    resume_current_period = None
    resume_next_epoch = 0
    resume_accumulated_periods = 0
    resume_es_best = None
    resume_es_best_state = None
    resume_es_best_opt_state = None
    resume_es_no_improve = 0

    if ckpt_enabled and auto_resume and not force_fresh:
        resume_state = _load_feedback_checkpoint(ckpt_dir)
        if resume_state is not None:
            status = resume_state.get("status", "between_periods")
            if status == "finished":
                print(">> Resume: found finished checkpoint. Set USIM_FB_FORCE_FRESH=1 to start over.")
                return
            total_periods_saved = int(resume_state.get("total_periods", len(periods)))
            if total_periods_saved != len(periods):
                print(f">> Resume skipped: checkpoint total_periods={total_periods_saved}, current={len(periods)}")
            else:
                model.load_state_dict(resume_state["model_state"])
                optimizer.load_state_dict(resume_state["optimizer_state"])
                _optimizer_state_to_device(optimizer, device)
                history = copy.deepcopy(resume_state.get("history", history))
                accum_cold = copy.deepcopy(resume_state.get("accum_cold", accum_cold))
                accum_hot = copy.deepcopy(resume_state.get("accum_hot", accum_hot))
                count_cold = int(resume_state.get("count_cold", count_cold))
                count_hot = int(resume_state.get("count_hot", count_hot))
                full_cold = copy.deepcopy(resume_state.get("full_cold", full_cold))
                full_hot = copy.deepcopy(resume_state.get("full_hot", full_hot))
                fc_cold = int(resume_state.get("fc_cold", fc_cold))
                fc_hot = int(resume_state.get("fc_hot", fc_hot))
                user_seen_items = _deserialize_user_seen_items(resume_state.get("user_seen_items"))
                warmup_periods = int(resume_state.get("warmup_periods", warmup_periods))
                resume_accumulated_periods = int(resume_state.get("accumulated_periods", 0))
                accumulated_dfs = periods[:resume_accumulated_periods]
                start_period = int(resume_state.get("next_period", 0))
                resume_current_period = resume_state.get("current_period")
                if resume_current_period is not None:
                    resume_current_period = int(resume_current_period)
                    start_period = resume_current_period
                resume_next_epoch = int(resume_state.get("next_epoch", 0))
                resume_es_best = copy.deepcopy(resume_state.get("es_best"))
                resume_es_best_state = _move_state_to_cpu(resume_state.get("es_best_state"))
                resume_es_best_opt_state = _move_state_to_cpu(resume_state.get("es_best_opt_state"))
                resume_es_no_improve = int(resume_state.get("es_no_improve", 0))
                print(
                    f">> Resume: status={status} | start_period={start_period} | "
                    f"resume_current_period={resume_current_period} | next_epoch={resume_next_epoch} | "
                    f"accumulated_periods={resume_accumulated_periods}"
                )

    for t in range(start_period, len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
        n_total = len(eval_ds)
        print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0
        resume_this_period = (resume_current_period is not None and t == resume_current_period)

        if resume_this_period:
            print(f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs}")
        elif t >= warmup_periods:
            print("  [EVAL-START] Build eval item bank and run sampled/full ranking...")
            all_item_vecs_eval = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, n_cold_t = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="hot", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            if met_cold:
                cold_res = met_cold
                for key in metrics_keys:
                    accum_cold[key] += met_cold[key] * n_cold_t
                count_cold += n_cold_t
            if met_hot:
                hot_res = met_hot
                for key in metrics_keys:
                    accum_hot[key] += met_hot[key] * n_hot_t
                count_hot += n_hot_t
            if fmet_cold:
                for key in metrics_keys:
                    full_cold[key] += fmet_cold[key] * fn_c
                fc_cold += fn_c
            if fmet_hot:
                for key in metrics_keys:
                    full_hot[key] += fmet_hot[key] * fn_h
                fc_hot += fn_h
            c_s = met_cold["R@10"] if met_cold else 0.0
            h_s = met_hot["R@10"] if met_hot else 0.0
            c_f = fmet_cold["R@10"] if fmet_cold else 0.0
            h_f = fmet_hot["R@10"] if fmet_hot else 0.0
            print(f"  采样 Cold={c_s:.4f} Hot={h_s:.4f} | 全库 Cold={c_f:.4f} Hot={h_f:.4f}")
            del all_item_vecs_eval
            _maybe_clear_cuda_cache()
        else:
            print("  [WARMUP] Training only...")

        if not resume_this_period:
            history["Period"].append(t)
            history["Count_cold"].append(n_cold_t)
            history["Count_hot"].append(n_hot_t)
            for key in metrics_keys:
                history["cold_" + key].append(cold_res.get(key, 0.0))
                history["hot_" + key].append(hot_res.get(key, 0.0))
            _add_user_seen_from_df(user_seen_items, p_df)
            accumulated_dfs.append(p_df)

        # ===================== P1-1: 滑动窗口 =====================
        # 用全部 accumulated_dfs 保留完整历史（用于 user_seen_items），
        # 但训练时只取最近 window 个 period 的数据
        window = cfg.stream_train_window
        if window > 0 and len(accumulated_dfs) > window:
            train_dfs = accumulated_dfs[-window:]
            print(f"  [WINDOW] 使用最近 {window}/{len(accumulated_dfs)} periods 训练 "
                  f"({sum(len(d) for d in train_dfs)} samples)")
        else:
            train_dfs = accumulated_dfs

        combined_df = pd.concat(train_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = (t >= warmup_periods) and cfg.use_epoch_early_stop and cfg.n_epochs > 1
        es_best = copy.deepcopy(resume_es_best) if resume_this_period else None
        es_best_state = copy.deepcopy(resume_es_best_state) if resume_this_period else None
        es_best_opt_state = copy.deepcopy(resume_es_best_opt_state) if resume_this_period else None
        es_no_improve = int(resume_es_no_improve) if resume_this_period else 0
        epoch_start_idx = resume_next_epoch if resume_this_period else 0

        if ckpt_enabled and not resume_this_period:
            period_start_state = _build_feedback_ckpt_state(
                model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                accumulated_periods=t + 1, warmup_periods=warmup_periods,
                total_periods=len(periods), status="in_period",
                next_period=t, current_period=t, next_epoch=0,
            )
            _save_feedback_checkpoint(ckpt_dir, period_start_state)

        for epoch in range(epoch_start_idx, cfg.n_epochs):
            epoch_start = time.time()
            num_batches = max(1, len(train_loader))
            total_loss = 0.0
            steps = 0
            cand_dup_sum = 0.0
            cand_cov_sum = 0.0
            cand_gain_sum = 0.0
            cand_pen_sum = 0.0
            course_sample_fit_sum = 0.0
            course_prereq_sum = 0.0
            course_concept_sum = 0.0
            course_diff_sum = 0.0
            course_redundant_sum = 0.0
            cand_batches = 0
            optimizer.zero_grad()

            cached_user_bank = None
            if cfg.candidate_strategy == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()

            print(
                f"  [TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Period {t + 1}/{len(periods)} | samples={len(combined_df)} | batches={num_batches}"
            )
            last_progress_log = epoch_start

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
                if (
                    cached_user_bank is not None and
                    cfg.user_bank_refresh_steps > 0 and
                    batch_idx > 0 and
                    (batch_idx % cfg.user_bank_refresh_steps == 0)
                ):
                    cached_user_bank = model._build_user_bank_raw()

                batch = {k: v.to(device) for k, v in batch.items()}
                loss, cand_info = model(
                    batch, pop.to(device), llm.to(device),
                    user_bank_raw=cached_user_bank, user_seen_items=user_seen_items,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += float(loss.item())
                steps += 1
                if cand_info and cand_info.get("steps", 0) > 0:
                    cand_dup_sum += cand_info["dup_rate"]
                    cand_cov_sum += cand_info["topm_coverage"]
                    cand_gain_sum += cand_info.get("step_gain", 0.0)
                    cand_pen_sum += cand_info.get("collapse_penalty", 0.0)
                    course_sample_fit_sum += cand_info.get("course_sample_fit", 0.0)
                    course_prereq_sum += cand_info.get("course_prereq_gap", 0.0)
                    course_concept_sum += cand_info.get("course_concept_bonus", 0.0)
                    course_diff_sum += cand_info.get("course_difficulty_gap", 0.0)
                    course_redundant_sum += cand_info.get("course_redundant", 0.0)
                    cand_batches += 1

                now_ts = time.time()
                if _should_log_train_progress(batch_idx, num_batches, cfg, last_progress_log, now_ts):
                    done = batch_idx + 1
                    elapsed = now_ts - epoch_start
                    avg_batch_sec = elapsed / max(1, done)
                    eta = avg_batch_sec * max(0, num_batches - done)
                    pct = 100.0 * done / max(1, num_batches)
                    print(
                        f"    [TRAIN-PROGRESS] {done}/{num_batches} ({pct:.0f}%) | "
                        f"avg_loss={total_loss / max(1, steps):.4f} | "
                        f"elapsed={_format_eta(elapsed)} | eta={_format_eta(eta)}"
                    )
                    last_progress_log = now_ts

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                avg_gain = cand_gain_sum / cand_batches
                avg_pen = cand_pen_sum / cand_batches
                avg_csf = course_sample_fit_sum / cand_batches
                avg_cp = course_prereq_sum / cand_batches
                avg_cc = course_concept_sum / cand_batches
                avg_cd = course_diff_sum / cand_batches
                avg_cr = course_redundant_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 训练: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"SampleFit: {avg_csf:.4f} | "
                    f"Course[p={avg_cp:.4f}, c={avg_cc:.4f}, d={avg_cd:.4f}, r={avg_cr:.4f}]"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 训练: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if ckpt_enabled:
                epoch_state = _build_feedback_ckpt_state(
                    model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                    full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                    accumulated_periods=t + 1, warmup_periods=warmup_periods,
                    total_periods=len(periods), status="in_period",
                    next_period=t, current_period=t, next_epoch=epoch + 1,
                    es_best=es_best, es_best_state=es_best_state,
                    es_best_opt_state=es_best_opt_state, es_no_improve=es_no_improve,
                )
                _save_feedback_checkpoint(ckpt_dir, epoch_state)

            if do_early_stop:
                print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
                all_item_vecs_es = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim_seq(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="cold", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )
                es_hot, _ = evaluate_usim_seq(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="hot", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )
                key_n = f"N@{cfg.early_stop_k}"
                key_r = f"R@{cfg.early_stop_k}"
                cur_n = es_cold.get(key_n, 0.0) if es_cold else 0.0
                cur_cr = es_cold.get(key_r, 0.0) if es_cold else 0.0
                cur_hr = es_hot.get(key_r, 0.0) if es_hot else 0.0

                if es_best is None:
                    is_better = True
                else:
                    hot_floor = es_best["hot_r"] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
                    hot_ok = cur_hr >= hot_floor
                    n_improve = cur_n > es_best["cold_n"] + cfg.early_stop_min_delta
                    n_tie = abs(cur_n - es_best["cold_n"]) <= cfg.early_stop_min_delta
                    r_tie_break = cur_cr > es_best["cold_r"] + 1e-12
                    is_better = hot_ok and (n_improve or (n_tie and r_tie_break))

                if is_better:
                    es_best = {"epoch": epoch + 1, "cold_n": float(cur_n), "cold_r": float(cur_cr), "hot_r": float(cur_hr)}
                    es_best_state = _move_state_to_cpu(model.state_dict())
                    es_best_opt_state = _move_state_to_cpu(optimizer.state_dict())
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                print(
                    f"  [EARLYSTOP] Epoch {epoch + 1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
                )
                del all_item_vecs_es, es_cold, es_hot
                _maybe_clear_cuda_cache()

                if ckpt_enabled:
                    _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
                        model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                        full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                        accumulated_periods=t + 1, warmup_periods=warmup_periods,
                        total_periods=len(periods), status="in_period",
                        next_period=t, current_period=t, next_epoch=epoch + 1,
                        es_best=es_best, es_best_state=es_best_state,
                        es_best_opt_state=es_best_opt_state, es_no_improve=es_no_improve,
                    ))

                if es_no_improve >= cfg.early_stop_patience:
                    print(f"  [EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    break

        if do_early_stop and es_best_state is not None:
            model.load_state_dict(es_best_state)
            if es_best_opt_state is not None:
                optimizer.load_state_dict(es_best_opt_state)
                _optimizer_state_to_device(optimizer, device)
            print(
                f"  [EARLYSTOP] Restore best epoch={es_best['epoch']} "
                f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
                f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
                f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
            )
            _maybe_clear_cuda_cache()

        if ckpt_enabled:
            _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
                model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
                accumulated_periods=t + 1, warmup_periods=warmup_periods,
                total_periods=len(periods), status="between_periods", next_period=t + 1,
            ))

        if resume_this_period:
            resume_current_period = None
            resume_next_epoch = 0
            resume_es_best = None
            resume_es_best_state = None
            resume_es_best_opt_state = None
            resume_es_no_improve = 0

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (RL-USIM FAST2)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv("mooc_metrics_usim_feedback_fast2.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("RL-USIM [FAST2]: P1 Window + PPO Reduction")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_usim_feedback_fast2.png")
    print(">> Saved mooc_result_usim_feedback_fast2.png and csv")

    if ckpt_enabled:
        _save_feedback_checkpoint(ckpt_dir, _build_feedback_ckpt_state(
            model, optimizer, history, accum_cold, accum_hot, count_cold, count_hot,
            full_cold, full_hot, fc_cold, fc_hot, user_seen_items,
            accumulated_periods=len(periods), warmup_periods=warmup_periods,
            total_periods=len(periods), status="finished", next_period=len(periods),
        ), snapshot_name="finished.pt")


if __name__ == "__main__":
    setup_seed(2025)
    main()
