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

from hhcor_static_hin import build_history_tensor, _update_histories_from_df
from usim import _add_user_seen_from_df, _clone_user_seen, build_course_artifacts, setup_seed, split_dataframe_by_periods
from usim_course import (
    CourseAwareUSIM,
    CourseConfig,
    CourseSeqDataset,
    build_all_item_vecs_course,
    collate_course,
    evaluate_course_usim,
    train_one_epoch,
)


FEEDBACK_TYPE_NAMES = (
    "good_fit",
    "prereq_unmet",
    "difficulty_too_high",
    "topic_drift",
    "redundant_recommendation",
)


class FeedbackLiteCourseConfig(CourseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_lite_accept_weight = float(os.environ.get("USIM_FEEDBACK_LITE_ACCEPT_WEIGHT", "0.08"))
        self.feedback_lite_type_weight = float(os.environ.get("USIM_FEEDBACK_LITE_TYPE_WEIGHT", "0.05"))
        self.feedback_lite_top_l = int(os.environ.get("USIM_FEEDBACK_LITE_TOPL", "30"))
        self.feedback_lite_train = os.environ.get("USIM_FEEDBACK_LITE_TRAIN", "0") == "1"
        self.feedback_lite_only_cold = os.environ.get("USIM_FEEDBACK_LITE_ONLY_COLD", "1") == "1"
        self.feedback_lite_aux_only_cold = os.environ.get("USIM_FEEDBACK_LITE_AUX_ONLY_COLD", "1") == "1"
        self.feedback_lite_warm_seen = int(os.environ.get("USIM_FEEDBACK_LITE_WARM_SEEN", str(self.course_score_warm_seen)))

        self.feedback_lite_good_alpha = float(os.environ.get("USIM_FEEDBACK_LITE_GOOD_ALPHA", "0.06"))
        self.feedback_lite_accept_alpha = float(os.environ.get("USIM_FEEDBACK_LITE_ACCEPT_ALPHA", "0.03"))
        self.feedback_lite_prereq_penalty = float(os.environ.get("USIM_FEEDBACK_LITE_PREREQ_PENALTY", "0.05"))
        self.feedback_lite_diff_penalty = float(os.environ.get("USIM_FEEDBACK_LITE_DIFF_PENALTY", "0.03"))
        self.feedback_lite_topic_penalty = float(os.environ.get("USIM_FEEDBACK_LITE_TOPIC_PENALTY", "0.03"))
        self.feedback_lite_redundant_penalty = float(os.environ.get("USIM_FEEDBACK_LITE_REDUNDANT_PENALTY", "0.02"))

        self.feedback_lite_prereq_thr = float(os.environ.get("USIM_FEEDBACK_LITE_PREREQ_THR", "0.55"))
        self.feedback_lite_diff_thr = float(os.environ.get("USIM_FEEDBACK_LITE_DIFF_THR", "0.25"))
        self.feedback_lite_concept_thr = float(os.environ.get("USIM_FEEDBACK_LITE_CONCEPT_THR", "0.10"))
        self.feedback_lite_redundant_thr = float(os.environ.get("USIM_FEEDBACK_LITE_REDUNDANT_THR", "0.70"))
        self.feedback_lite_hard_neg_lambda = float(os.environ.get("USIM_FEEDBACK_LITE_HARD_NEG_LAMBDA", "0.20"))
        self.feedback_lite_hard_preselect = int(os.environ.get("USIM_FEEDBACK_LITE_HARD_PRESELECT", "24"))
        self.feedback_lite_hard_neg_only_cold = os.environ.get("USIM_FEEDBACK_LITE_HARD_NEG_ONLY_COLD", "1") == "1"
        self.feedback_lite_hard_accept_alpha = float(os.environ.get("USIM_FEEDBACK_LITE_HARD_ACCEPT_ALPHA", "0.50"))
        self.feedback_lite_hard_good_alpha = float(os.environ.get("USIM_FEEDBACK_LITE_HARD_GOOD_ALPHA", "0.50"))
        self.feedback_lite_hard_bad_penalty = float(os.environ.get("USIM_FEEDBACK_LITE_HARD_BAD_PENALTY", "0.25"))
        self.feedback_lite_margin_weight = float(os.environ.get("USIM_FEEDBACK_LITE_MARGIN_WEIGHT", "0.03"))
        self.feedback_lite_margin = float(os.environ.get("USIM_FEEDBACK_LITE_MARGIN", "0.05"))
        self.feedback_lite_margin_topk = int(os.environ.get("USIM_FEEDBACK_LITE_MARGIN_TOPK", "4"))
        self.feedback_lite_margin_only_cold = os.environ.get("USIM_FEEDBACK_LITE_MARGIN_ONLY_COLD", "1") == "1"


class FeedbackLiteCourseUSIM(CourseAwareUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.feedback_context_proj = nn.Sequential(
            nn.Linear(config.emb_dim * 2 + 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.LayerNorm(config.emb_dim),
        )
        self.feedback_accept_head = nn.Sequential(
            nn.Linear(config.emb_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.feedback_type_head = nn.Sequential(
            nn.Linear(config.emb_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(FEEDBACK_TYPE_NAMES)),
        )
        self.item_popularity = None
        self.item_difficulty = None
        self._feedback_cached_item_bank = None

    def set_feedback_artifacts(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_difficulty = None
            return

        pop = item_popularity.to(self.device).float()
        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    def build_course_item_bank(self, force_cold=True, item_batch=1024, deterministic=False):
        bank = super().build_course_item_bank(
            force_cold=force_cold,
            item_batch=item_batch,
            deterministic=deterministic,
        )
        self._feedback_cached_item_bank = bank.detach()
        return bank

    def _get_feedback_item_bank(self):
        if self._feedback_cached_item_bank is None or self._feedback_cached_item_bank.size(0) != self.cfg.n_items:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                self._feedback_cached_item_bank = super().build_course_item_bank(force_cold=True, item_batch=1024).detach()
            if was_training:
                self.train()
        return self._feedback_cached_item_bank

    def _build_feedback_context(self, user_ids, seen_mat, seen_cnt_raw):
        item_bank = self._get_feedback_item_bank()
        hist_mean = torch.matmul(seen_mat, item_bank) / seen_cnt_raw.clamp_min(1.0)
        uid_t = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        u_base = self.user_proj(self.user_emb(uid_t))
        gate = self.course_gate(torch.cat([u_base, hist_mean], dim=1))
        mixed = gate * u_base + (1.0 - gate) * hist_mean
        fused = self.course_fuse(torch.cat([mixed, hist_mean], dim=1))
        hist_norm = (seen_cnt_raw / max(1.0, float(self.cfg.course_hist_len))).clamp(0.0, 1.0)
        seen_weight = self._seen_weight(seen_cnt_raw, self.cfg.feedback_lite_warm_seen)
        context = self.feedback_context_proj(torch.cat([fused, hist_mean, hist_norm, seen_weight], dim=1))
        return F.normalize(context, dim=1)

    def _compute_feedback_targets(self, item_idx, seen_mat, seen_cnt_raw):
        batch_idx = torch.arange(item_idx.size(0), device=self.device)
        zero = torch.zeros((item_idx.size(0), 1), dtype=torch.float32, device=self.device)

        prereq_gap = zero
        concept_match = zero
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            prereq_gap = violation_full[batch_idx, item_idx].unsqueeze(1)

        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0)

        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[item_idx].unsqueeze(1)
        else:
            item_difficulty = zero

        user_readiness = (
            seen_cnt_raw / max(1.0, float(self.cfg.course_hist_warm_seen))
        ).clamp(0.0, 1.0)
        difficulty_gap = F.relu(item_difficulty - user_readiness)

        redundant_mask = (
            (concept_match >= float(self.cfg.feedback_lite_redundant_thr)) &
            (seen_cnt_raw >= float(self.cfg.course_min_seen))
        )
        topic_drift_mask = (
            (concept_match < float(self.cfg.feedback_lite_concept_thr)) &
            (seen_cnt_raw > 0)
        )
        prereq_mask = prereq_gap >= float(self.cfg.feedback_lite_prereq_thr)
        difficulty_mask = difficulty_gap >= float(self.cfg.feedback_lite_diff_thr)

        feedback_label = torch.zeros(item_idx.size(0), dtype=torch.long, device=self.device)
        feedback_label = torch.where(redundant_mask.squeeze(1), torch.full_like(feedback_label, 4), feedback_label)
        feedback_label = torch.where(topic_drift_mask.squeeze(1), torch.full_like(feedback_label, 3), feedback_label)
        feedback_label = torch.where(difficulty_mask.squeeze(1), torch.full_like(feedback_label, 2), feedback_label)
        feedback_label = torch.where(prereq_mask.squeeze(1), torch.full_like(feedback_label, 1), feedback_label)
        accept_target = (feedback_label == 0).float().unsqueeze(1)
        return accept_target, feedback_label

    def _feedback_pair_logits(self, context_vec, candidate_idx):
        item_bank = self._get_feedback_item_bank()
        cand_vec = item_bank[candidate_idx]
        ctx = context_vec.unsqueeze(1).expand(-1, candidate_idx.size(1), -1)
        pair_feat = torch.cat([ctx, cand_vec, ctx * cand_vec, torch.abs(ctx - cand_vec)], dim=-1)
        accept_logits = self.feedback_accept_head(pair_feat).squeeze(-1)
        type_logits = self.feedback_type_head(pair_feat)
        return accept_logits, type_logits

    def _feedback_pair_probs(self, context_vec, candidate_idx):
        accept_logits, type_logits = self._feedback_pair_logits(context_vec, candidate_idx)
        accept_prob = torch.sigmoid(accept_logits)
        type_probs = F.softmax(type_logits, dim=-1)
        return accept_logits, type_logits, accept_prob, type_probs

    def _compose_feedback_hard_score(self, accept_prob, type_probs):
        return (
            float(self.cfg.feedback_lite_hard_accept_alpha) * accept_prob +
            float(self.cfg.feedback_lite_hard_good_alpha) * type_probs[..., 0] -
            float(self.cfg.feedback_lite_hard_bad_penalty) * type_probs[..., 1:].sum(dim=-1)
        )

    def _select_feedback_hard_negatives(self, neg_logits, candidate_item_idx, context_vec, is_cold, n_hard):
        batch_size = neg_logits.size(0)
        if n_hard < 1 or context_vec is None or candidate_item_idx is None:
            return None

        lambda_fb = float(self.cfg.feedback_lite_hard_neg_lambda)
        if lambda_fb <= 0:
            return None

        max_neg = neg_logits.size(1) - 1
        if max_neg < 1:
            return None

        pre_k = min(max(n_hard, int(self.cfg.feedback_lite_hard_preselect)), max_neg)
        selected_rows = torch.arange(batch_size, device=self.device)
        if self.cfg.feedback_lite_hard_neg_only_cold:
            selected_rows = torch.nonzero(is_cold, as_tuple=False).view(-1)
        if selected_rows.numel() < 1:
            return None

        base_scores, base_pos = torch.topk(neg_logits[selected_rows], k=pre_k, dim=1)
        base_item_idx = candidate_item_idx[selected_rows].gather(1, base_pos)

        with torch.no_grad():
            _, _, accept_prob, type_probs = self._feedback_pair_probs(context_vec[selected_rows], base_item_idx)
            fb_score = self._compose_feedback_hard_score(accept_prob, type_probs)

        combined = base_scores + lambda_fb * fb_score
        _, local_pos = torch.topk(combined, k=n_hard, dim=1)
        chosen_pos = base_pos.gather(1, local_pos)

        hard_idx = None
        if selected_rows.numel() == batch_size:
            hard_idx = chosen_pos
        else:
            hard_idx = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
            base_default = torch.topk(neg_logits, k=n_hard, dim=1).indices
            hard_idx = base_default
            hard_idx[selected_rows] = chosen_pos
        return hard_idx

    def compute_feedback_lite_adjust(
        self,
        user_ids,
        user_seen_items,
        candidate_idx,
        target_pop=None,
        exclude_items=None,
        seen_mat=None,
        seen_cnt_raw=None,
    ):
        batch_size = len(user_ids)
        if batch_size < 1:
            return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)

        if seen_mat is None or seen_cnt_raw is None:
            if user_seen_items is None:
                return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)
            seen_mat, seen_cnt_raw = self._build_seen_mat(user_ids, user_seen_items, exclude_items=exclude_items)
        if seen_cnt_raw.max().item() < 1:
            return torch.zeros(candidate_idx.size(0), candidate_idx.size(1), device=self.device)

        context_vec = self._build_feedback_context(user_ids, seen_mat, seen_cnt_raw)
        with torch.set_grad_enabled(self.training and self.cfg.feedback_lite_train):
            _, _, accept_prob, type_probs = self._feedback_pair_probs(context_vec, candidate_idx)

        active = self._seen_weight(seen_cnt_raw, self.cfg.feedback_lite_warm_seen)
        if self.cfg.feedback_lite_only_cold and target_pop is not None:
            active = active * (target_pop.view(-1, 1) < float(self.cfg.cold_threshold)).float()

        adjust = (
            float(self.cfg.feedback_lite_accept_alpha) * accept_prob +
            float(self.cfg.feedback_lite_good_alpha) * type_probs[..., 0] -
            float(self.cfg.feedback_lite_prereq_penalty) * type_probs[..., 1] -
            float(self.cfg.feedback_lite_diff_penalty) * type_probs[..., 2] -
            float(self.cfg.feedback_lite_topic_penalty) * type_probs[..., 3] -
            float(self.cfg.feedback_lite_redundant_penalty) * type_probs[..., 4]
        )
        return adjust * active

    def maybe_add_course_score(
        self,
        base_scores,
        user_ids,
        candidate_idx,
        user_seen_items=None,
        target_pop=None,
        exclude_items=None,
        seen_mat=None,
        seen_cnt_raw=None,
        concept_seen_mat=None,
        concept_seen_cnt_raw=None,
        training=False,
    ):
        scores = super().maybe_add_course_score(
            base_scores,
            user_ids,
            candidate_idx,
            user_seen_items=user_seen_items,
            target_pop=target_pop,
            exclude_items=exclude_items,
            seen_mat=seen_mat,
            seen_cnt_raw=seen_cnt_raw,
            concept_seen_mat=concept_seen_mat,
            concept_seen_cnt_raw=concept_seen_cnt_raw,
            training=training,
        )
        if training and not self.cfg.feedback_lite_train:
            return scores

        top_l = int(self.cfg.feedback_lite_top_l)
        if top_l <= 0 or top_l >= candidate_idx.size(1):
            adjust = self.compute_feedback_lite_adjust(
                user_ids,
                user_seen_items,
                candidate_idx,
                target_pop=target_pop,
                exclude_items=exclude_items,
                seen_mat=seen_mat,
                seen_cnt_raw=seen_cnt_raw,
            )
            return scores + adjust

        k = min(top_l, candidate_idx.size(1))
        top_pos = torch.topk(scores, k=k, dim=1).indices
        top_items = candidate_idx.gather(1, top_pos)
        top_adjust = self.compute_feedback_lite_adjust(
            user_ids,
            user_seen_items,
            top_items,
            target_pop=target_pop,
            exclude_items=exclude_items,
            seen_mat=seen_mat,
            seen_cnt_raw=seen_cnt_raw,
        )
        delta = torch.zeros_like(scores)
        delta.scatter_(1, top_pos, top_adjust)
        return scores + delta

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i, hist = batch["u"], batch["i"], batch["hist"]
        is_cold = pop < self.cfg.cold_threshold
        user_ids = [int(x) for x in u.detach().cpu().tolist()]

        item_bank = self.build_course_item_bank(force_cold=True)
        z_u_base, _, _ = self.encode_course_user(u, hist, item_bank)
        hist_seen_mat, hist_seen_cnt = self._hist_to_seen_mat(hist)
        context_vec = self._build_feedback_context(user_ids, hist_seen_mat, hist_seen_cnt)
        z_i_base, id_e_raw, content_e = self.get_item_vector(i, llm_s, force_cold=False)

        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp

        cand_idx_full = i.view(1, -1).expand(logits.size(0), -1)
        logits = self.maybe_add_course_score(
            logits,
            user_ids,
            cand_idx_full,
            user_seen_items=user_seen_items,
            target_pop=pop,
            exclude_items=i,
            seen_mat=hist_seen_mat,
            seen_cnt_raw=hist_seen_cnt,
            training=self.training,
        )

        labels = torch.arange(logits.size(0), device=self.device)
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp
        row_weights = torch.where(
            is_cold,
            torch.full_like(pop, float(self.cfg.cold_loss_weight), dtype=torch.float32),
            torch.ones_like(pop, dtype=torch.float32),
        )
        selected_hard_item_idx = None

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
                    if (not self.cfg.use_structured_hard_neg):
                        feedback_hard_idx = self._select_feedback_hard_negatives(
                            neg_logits,
                            cand_idx_full,
                            context_vec,
                            is_cold,
                            n_hard,
                        )
                        if feedback_hard_idx is not None:
                            hard_idx = feedback_hard_idx
                        else:
                            _, hard_idx = torch.topk(neg_logits, k=n_hard, dim=1)
                    elif self.item_hard_adj is not None:
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
                    selected_hard_item_idx = cand_idx_full.gather(1, hard_idx)

                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(rand_scores, k=n_rand, dim=1)

                cand_idx = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                per_row_loss = F.cross_entropy(cand_logits, main_targets, reduction="none")
                main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)
            else:
                per_row_loss = F.cross_entropy(logits_margin, labels, reduction="none")
                main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)
        else:
            per_row_loss = F.cross_entropy(logits_margin, labels, reduction="none")
            main_loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-12)

        z_id = F.normalize(id_e_raw, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            seen_mat = hist_seen_mat
            seen_cnt = hist_seen_cnt.squeeze(1)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(seen_mat, prereq_mat_batch.t())
            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0 - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch),
            ).clamp(0.0, 1.0)

            valid_rows = seen_cnt >= float(self.cfg.prereq_aux_min_seen)
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
                    pos_vals = logits[torch.arange(logits.size(0), device=self.device), labels]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        accept_target, feedback_label = self._compute_feedback_targets(i, hist_seen_mat, hist_seen_cnt)
        pos_accept_logits, pos_type_logits, pos_accept_prob, pos_type_probs = self._feedback_pair_probs(
            context_vec,
            i.view(-1, 1),
        )
        pos_accept_logits = pos_accept_logits.squeeze(1)
        pos_type_logits = pos_type_logits.squeeze(1)

        feedback_mask = torch.ones_like(is_cold, dtype=torch.bool)
        if self.cfg.feedback_lite_aux_only_cold:
            feedback_mask = is_cold

        accept_loss = torch.tensor(0.0, device=self.device)
        type_loss = torch.tensor(0.0, device=self.device)
        feedback_margin_loss = torch.tensor(0.0, device=self.device)
        if feedback_mask.any():
            accept_loss = F.binary_cross_entropy_with_logits(
                pos_accept_logits[feedback_mask],
                accept_target.squeeze(1)[feedback_mask],
            )
            type_loss = F.cross_entropy(
                pos_type_logits[feedback_mask],
                feedback_label[feedback_mask],
            )

        if (
            selected_hard_item_idx is not None and selected_hard_item_idx.numel() > 0 and
            float(self.cfg.feedback_lite_margin_weight) > 0
        ):
            margin_rows = torch.ones_like(is_cold, dtype=torch.bool)
            if self.cfg.feedback_lite_margin_only_cold:
                margin_rows = is_cold
            margin_k = min(int(self.cfg.feedback_lite_margin_topk), selected_hard_item_idx.size(1))
            if margin_rows.any() and margin_k > 0:
                neg_item_idx = selected_hard_item_idx[:, :margin_k]
                _, _, neg_accept_prob, neg_type_probs = self._feedback_pair_probs(context_vec, neg_item_idx)
                pos_fb_score = self._compose_feedback_hard_score(pos_accept_prob, pos_type_probs).squeeze(1)
                neg_fb_score = self._compose_feedback_hard_score(neg_accept_prob, neg_type_probs)
                hardest_neg = neg_fb_score.max(dim=1).values
                feedback_margin_loss = F.relu(
                    float(self.cfg.feedback_lite_margin) - pos_fb_score[margin_rows] + hardest_neg[margin_rows]
                ).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss +
            float(self.cfg.feedback_lite_accept_weight) * accept_loss +
            float(self.cfg.feedback_lite_type_weight) * type_loss +
            float(self.cfg.feedback_lite_margin_weight) * feedback_margin_loss
        )
        return total_loss, candidate_stats


def build_item_popularity(df, n_items):
    counts = torch.zeros(n_items, dtype=torch.float32)
    vc = df["i_idx"].value_counts()
    for item_idx, count in vc.items():
        idx = int(item_idx)
        if 0 <= idx < n_items:
            counts[idx] = float(count)
    return counts


def run_static_experiment_feedback_lite(df, cfg, device, model, optimizer, llm_scores):
    static_seed = int(os.environ.get("USIM_STATIC_SEED", "2025"))
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))

    df_static = df.sample(frac=1.0, random_state=static_seed).reset_index(drop=True)
    n_total = len(df_static)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_df = df_static.iloc[:n_train]
    val_df = df_static.iloc[n_train:n_train + n_val]
    test_df = df_static.iloc[n_train + n_val:]

    train_hist, train_histories = build_history_tensor(
        train_df, base_histories={}, max_len=cfg.course_hist_len, update_histories=True
    )
    val_hist, _ = build_history_tensor(
        val_df, base_histories=train_histories, max_len=cfg.course_hist_len, update_histories=False
    )
    train_val_histories = copy.deepcopy(train_histories)
    _update_histories_from_df(train_val_histories, val_df)
    test_hist, _ = build_history_tensor(
        test_df, base_histories=train_val_histories, max_len=cfg.course_hist_len, update_histories=False
    )

    train_loader = DataLoader(
        CourseSeqDataset(train_df, llm_scores, train_hist),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_course,
    )
    val_loader = DataLoader(
        CourseSeqDataset(val_df, llm_scores, val_hist),
        batch_size=2048,
        shuffle=False,
        collate_fn=collate_course,
    )
    test_loader = DataLoader(
        CourseSeqDataset(test_df, llm_scores, test_hist),
        batch_size=2048,
        shuffle=False,
        collate_fn=collate_course,
    )

    train_seen = {}
    _add_user_seen_from_df(train_seen, train_df)
    test_seen = _clone_user_seen(train_seen)
    _add_user_seen_from_df(test_seen, val_df)

    print(
        f"\n>>> Start STATIC feedback-lite train/eval | split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} "
        f"| train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    best_val = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        avg_loss, avg_dup, avg_cov = train_one_epoch(model, train_loader, optimizer, device, cfg, train_seen)
        epoch_sec = time.time() - epoch_start

        all_item_vecs_val = build_all_item_vecs_course(model)
        val_cold, _ = evaluate_course_usim(
            model,
            val_loader,
            device,
            k_list=k_list,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=all_item_vecs_val,
        )
        val_key = val_cold.get("N@10", 0.0) if val_cold else 0.0
        if val_key > best_val:
            best_val = val_key
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        tag = (
            f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
            if avg_dup is not None else ""
        )
        print(
            f"  [STATIC-FEEDBACK-LITE] Epoch {epoch + 1}/{cfg.n_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_sec:.1f}s | {tag}Val Full Cold N@10: {val_key:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [STATIC-FEEDBACK-LITE] Restore best epoch={best_epoch} | Full Cold N@10={best_val:.4f}")

    all_item_vecs_test = build_all_item_vecs_course(model)
    met_cold, n_cold_t = evaluate_course_usim(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    met_hot, n_hot_t = evaluate_course_usim(
        model, test_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_cold, fn_c = evaluate_course_usim(
        model, test_loader, device, k_list, eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_hot, fn_h = evaluate_course_usim(
        model, test_loader, device, k_list, eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT (STATIC FEEDBACK-LITE): sampled (1+{cfg.eval_n_neg}) vs full ranking")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for m in metrics_keys:
        print(
            f"{m:<10} | {met_cold.get(m, 0.0) if met_cold else 0.0:<12.4f} | "
            f"{met_hot.get(m, 0.0) if met_hot else 0.0:<12.4f} | "
            f"{fmet_cold.get(m, 0.0) if fmet_cold else 0.0:<12.4f} | "
            f"{fmet_hot.get(m, 0.0) if fmet_hot else 0.0:<12.4f}"
        )
    print("-" * 90)
    print(f"Sampled Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"Full Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)


def main():
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")
    print(f"Loading Data for Course Feedback-Lite USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FeedbackLiteCourseConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    course_artifacts, course_stats = build_course_artifacts(
        df,
        cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    item_popularity = build_item_popularity(df, cfg.n_items)

    model = FeedbackLiteCourseUSIM(cfg, content_emb).to(device)
    model.set_course_artifacts(course_artifacts)
    model.set_feedback_artifacts(item_popularity)
    model.set_global_llm_scores(llm_scores)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f">> Architecture: Course Feedback-Lite USIM (Batch Size={cfg.batch_size})")
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | TopM={cfg.retrieve_top_m} | "
        f"Temp={cfg.candidate_temp:.2f} | Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates}"
    )
    print(
        f">> Course Priors: concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}, "
        f"prereq_edges={course_stats['prereq_edges_kept']}"
    )
    print(
        f">> Feedback-Lite: topL={cfg.feedback_lite_top_l} | only_cold={cfg.feedback_lite_only_cold} | "
        f"aux_only_cold={cfg.feedback_lite_aux_only_cold} | train_rerank={cfg.feedback_lite_train} | "
        f"hard_neg_lambda={cfg.feedback_lite_hard_neg_lambda:.2f} | margin_w={cfg.feedback_lite_margin_weight:.2f}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        run_static_experiment_feedback_lite(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    print(f"\n>>> Start cumulative feedback-lite train/eval - total {len(periods)} periods <<<")

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]
    history = {"Period": [], "Count_cold": [], "Count_hot": []}
    for prefix in ["cold_", "hot_"]:
        for key in metrics_keys:
            history[prefix + key] = []

    accum_cold = {k: 0.0 for k in metrics_keys}
    accum_hot = {k: 0.0 for k in metrics_keys}
    count_cold, count_hot = 0, 0
    full_cold = {k: 0.0 for k in metrics_keys}
    full_hot = {k: 0.0 for k in metrics_keys}
    fc_cold, fc_hot = 0, 0

    warmup_periods = 3
    accumulated_dfs = []
    user_seen_items = {}
    user_histories = {}

    for t, p_df in enumerate(periods):
        eval_hist, _ = build_history_tensor(
            p_df, base_histories=user_histories, max_len=cfg.course_hist_len, update_histories=False
        )
        eval_ds = CourseSeqDataset(p_df, llm_scores, eval_hist)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_course)
        print(f"\n>>> Period {t} (current: {len(eval_ds)}, cumulative: {sum(len(d) for d in accumulated_dfs) + len(eval_ds)}) <<<")

        cold_res = {k: 0.0 for k in metrics_keys}
        hot_res = {k: 0.0 for k in metrics_keys}
        n_cold_t, n_hot_t = 0, 0

        if t >= warmup_periods:
            all_item_vecs_eval = build_all_item_vecs_course(model)
            met_cold, n_cold_t = evaluate_course_usim(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_course_usim(
                model, eval_loader, device, k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_course_usim(
                model, eval_loader, device, k_list, eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_course_usim(
                model, eval_loader, device, k_list, eval_type="hot", full_ranking=True,
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

            print(
                f"  Sampled Cold={met_cold.get('R@10', 0.0) if met_cold else 0.0:.4f} "
                f"Hot={met_hot.get('R@10', 0.0) if met_hot else 0.0:.4f} | "
                f"Full Cold={fmet_cold.get('R@10', 0.0) if fmet_cold else 0.0:.4f} "
                f"Hot={fmet_hot.get('R@10', 0.0) if fmet_hot else 0.0:.4f}"
            )
        else:
            print("  [WARMUP] Training only...")

        history["Period"].append(t)
        history["Count_cold"].append(n_cold_t)
        history["Count_hot"].append(n_hot_t)
        for key in metrics_keys:
            history["cold_" + key].append(cold_res.get(key, 0.0))
            history["hot_" + key].append(hot_res.get(key, 0.0))

        accumulated_dfs.append(p_df)
        combined_df = pd.concat(accumulated_dfs, ignore_index=True)
        train_hist, _ = build_history_tensor(
            combined_df, base_histories={}, max_len=cfg.course_hist_len, update_histories=True
        )
        train_ds = CourseSeqDataset(combined_df, llm_scores, train_hist)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_course)

        for epoch in range(cfg.n_epochs):
            epoch_start = time.time()
            avg_loss, avg_dup, avg_cov = train_one_epoch(model, train_loader, optimizer, device, cfg, user_seen_items)
            epoch_sec = time.time() - epoch_start
            if avg_dup is not None:
                print(
                    f"  [TRAIN-FEEDBACK-LITE] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f}"
                )
            else:
                print(
                    f"  [TRAIN-FEEDBACK-LITE] Epoch {epoch + 1}/{cfg.n_epochs} | cumulative: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

        _add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking (Course Feedback-Lite USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"Sampled Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"Full Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    pd.DataFrame(history).to_csv("mooc_metrics_course_usim_feedback_lite.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Course Feedback-Lite USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig("mooc_result_course_usim_feedback_lite.png")
    print(">> Saved mooc_result_course_usim_feedback_lite.png and csv")


if __name__ == "__main__":
    setup_seed(2025)
    main()
