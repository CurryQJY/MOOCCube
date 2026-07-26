"""
usim_feedback_fast3_seq.py - FAST3 + User Behavior Sequence Modeling

Extends FAST3 standalone with Transformer-based user sequence encoding.
The user's historical course interaction sequence is encoded via a causal
Transformer and gated-fused with the static user embedding, producing a
richer, dynamic user representation that captures interest evolution.

Key additions over fast3_standalone:
  1. UserSequenceEncoder  - causal Transformer encoder for user history
  2. UserHistoryTracker   - maintains temporally ordered user histories
  3. SeqFast3FeedbackUSIM - model with sequence-aware user representations
  4. SeqStreamDataset     - dataset that provides padded history sequences

Reference papers:
  - CMCLRec (SIGIR 2024): Cross-modal Contrastive Learning for User
    Cold-start Sequential Recommendation
  - LLM-ESR (NeurIPS 2024): Large Language Models Enhancement for
    Long-tailed Sequential Recommendation
"""

import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Import base classes and utilities from the standalone script ──────────
import usim_feedback_fast3_standalone as base

# Re-export setup_seed for __main__
setup_seed = base.setup_seed


# =========================================================================
#  Configuration
# =========================================================================

class SeqConfig(base.Fast3Config):
    """Extends Fast3Config with user-sequence hyperparameters."""

    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.seq_max_len = int(os.environ.get("USIM_SEQ_MAX_LEN", "50"))
        self.seq_n_heads = int(os.environ.get("USIM_SEQ_N_HEADS", "4"))
        self.seq_n_layers = int(os.environ.get("USIM_SEQ_N_LAYERS", "2"))
        self.seq_dropout = float(os.environ.get("USIM_SEQ_DROPOUT", "0.1"))
        self.seq_pool = os.environ.get("USIM_SEQ_POOL", "last").strip().lower()
        # Cold-hot split: only use seq encoder for users with short history
        self.seq_cold_only = os.environ.get("USIM_SEQ_COLD_ONLY", "1").strip() == "1"
        self.seq_user_cold_thr = int(os.environ.get("USIM_SEQ_USER_COLD_THR", "20"))


# =========================================================================
#  User Sequence Encoder
# =========================================================================

class UserSequenceEncoder(nn.Module):
    """
    Causal Transformer encoder that converts a user's historical item
    interaction sequence into a dynamic user representation.

    Architecture:
      item_id_emb(seq) + positional_emb → Transformer (causal mask) → pool
      static_user_emb ⊕ dynamic_emb → gate → fused user representation
    """

    def __init__(self, emb_dim, n_heads=4, n_layers=2, max_seq_len=50,
                 dropout=0.1, pool="last"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pool = pool  # "last" or "mean"
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.ln = nn.LayerNorm(emb_dim)
        # Gating network to fuse static and dynamic representations
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid(),
        )

    def forward(self, static_user_emb, seq_item_embs, seq_mask):
        """
        Parameters
        ----------
        static_user_emb : (B, D)  static user embedding
        seq_item_embs   : (B, L, D)  item embeddings for user history
        seq_mask        : (B, L)  True for PAD positions

        Returns
        -------
        fused : (B, D)  sequence-aware user representation
        """
        B, L, D = seq_item_embs.shape

        # If no history at all, fall back to static embedding
        valid_counts = (~seq_mask).sum(dim=1)  # (B,)
        if valid_counts.max().item() < 1:
            return static_user_emb

        # Positional encoding (clamped to max_seq_len)
        positions = torch.arange(L, device=seq_item_embs.device).clamp(
            max=self.max_seq_len - 1
        )
        pos_emb = self.pos_emb(positions)  # (L, D)
        seq_input = seq_item_embs + pos_emb.unsqueeze(0)

        # Causal attention mask (upper triangular = masked)
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq_item_embs.device), diagonal=1
        ).bool()

        # Transformer encode
        seq_out = self.transformer(
            seq_input,
            mask=causal_mask,
            src_key_padding_mask=seq_mask,
        )  # (B, L, D)

        # Pooling
        if self.pool == "mean":
            # Masked mean pooling
            valid_mask = (~seq_mask).unsqueeze(-1).float()  # (B, L, 1)
            dynamic_emb = (seq_out * valid_mask).sum(dim=1) / valid_mask.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            # Take the last valid (non-pad) position
            seq_lengths = valid_counts.clamp(min=1)
            last_idx = (seq_lengths - 1).long()
            batch_idx = torch.arange(B, device=seq_out.device)
            dynamic_emb = seq_out[batch_idx, last_idx]

        dynamic_emb = self.ln(dynamic_emb)  # (B, D)

        # Gate fusion
        gate = self.gate(
            torch.cat([static_user_emb, dynamic_emb], dim=-1)
        )
        fused = gate * static_user_emb + (1.0 - gate) * dynamic_emb

        # For users with zero history, keep static embedding
        no_history = (valid_counts < 1).unsqueeze(1).float()
        fused = no_history * static_user_emb + (1.0 - no_history) * fused

        return fused


# =========================================================================
#  User History Tracker
# =========================================================================

class UserHistoryTracker:
    """
    Maintains temporally-ordered user interaction histories.

    Provides both:
      - ordered sequences (for the sequence encoder)
      - set-based seen items (for reward / reranking / eval filtering)
    """

    def __init__(self, max_seq_len=50):
        self.max_seq_len = max_seq_len
        self._ordered = {}   # uid -> list[int]  ordered by timestamp
        self._sets = {}      # uid -> set[int]   for fast membership check

    # ── Mutation ──────────────────────────────────────────────────────

    def add_from_df(self, df):
        """Ingest interactions from a dataframe, preserving temporal order."""
        sorted_df = df.sort_values("timestamp")
        for u_idx, i_idx in zip(
            sorted_df["u_idx"].values, sorted_df["i_idx"].values
        ):
            uid, iid = int(u_idx), int(i_idx)
            if uid not in self._ordered:
                self._ordered[uid] = []
                self._sets[uid] = set()
            if iid not in self._sets[uid]:
                self._ordered[uid].append(iid)
                self._sets[uid].add(iid)

    # ── Queries ───────────────────────────────────────────────────────

    @property
    def seen_items(self):
        """Dict[int, Set[int]] compatible with user_seen_items."""
        return self._sets

    def get_sequence(self, uid, max_len=None):
        """Return the most recent items for a user as a list."""
        max_len = max_len or self.max_seq_len
        seq = self._ordered.get(int(uid), [])
        return seq[-max_len:]

    # ── Serialization ─────────────────────────────────────────────────

    def serialize(self):
        return {
            int(uid): [int(it) for it in items]
            for uid, items in self._ordered.items()
        }

    @classmethod
    def deserialize(cls, payload, max_seq_len=50):
        tracker = cls(max_seq_len)
        if not payload:
            return tracker
        for uid, items in payload.items():
            uid = int(uid)
            item_list = [int(it) for it in items]
            tracker._ordered[uid] = item_list
            tracker._sets[uid] = set(item_list)
        return tracker


# =========================================================================
#  Model: Sequence-Aware FAST3
# =========================================================================

class SeqFast3FeedbackUSIM(base.Fast3FeedbackUSIM):
    """
    Extends Fast3FeedbackUSIM with a Transformer-based sequence encoder.

    Changes from parent:
      - __init__: adds UserSequenceEncoder
      - forward(): z_u_base computed via sequence-aware path
      - get_user_vector(): for evaluation with sequences
    """

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.seq_encoder = UserSequenceEncoder(
            emb_dim=config.emb_dim,
            n_heads=config.seq_n_heads,
            n_layers=config.seq_n_layers,
            max_seq_len=config.seq_max_len,
            dropout=config.seq_dropout,
            pool=config.seq_pool,
        )

    def _encode_user_with_seq(self, u, seq_items, seq_mask):
        """
        Compute user representation with cold-hot split.

        If seq_cold_only is enabled, only users with history shorter than
        seq_user_cold_thr get sequence encoding; others keep static embedding.
        This preserves well-trained hot-user embeddings while boosting cold users.
        """
        static_emb = self.user_emb(u)                                  # (B, D)
        seq_items = seq_items.to(self.device)
        seq_mask = seq_mask.to(self.device)

        if not self.cfg.seq_cold_only:
            # All users get sequence encoding
            seq_item_embs = self.item_id_emb(seq_items)                # (B, L, D)
            fused = self.seq_encoder(static_emb, seq_item_embs, seq_mask)
            return self.user_proj(fused)

        # Cold-hot split: determine which users are "cold" by history length
        hist_len = (~seq_mask).sum(dim=1)                              # (B,)
        is_cold_user = hist_len < self.cfg.seq_user_cold_thr           # (B,)

        if not is_cold_user.any():
            # All hot → skip sequence encoder entirely
            return self.user_proj(static_emb)

        if is_cold_user.all():
            # All cold → encode all
            seq_item_embs = self.item_id_emb(seq_items)
            fused = self.seq_encoder(static_emb, seq_item_embs, seq_mask)
            return self.user_proj(fused)

        # Mixed batch: encode only cold users, keep static for hot
        cold_idx = is_cold_user.nonzero(as_tuple=True)[0]
        fused = static_emb.clone()
        cold_embs = self.item_id_emb(seq_items[cold_idx])
        cold_fused = self.seq_encoder(
            static_emb[cold_idx], cold_embs, seq_mask[cold_idx]
        )
        fused[cold_idx] = cold_fused
        return self.user_proj(fused)

    def get_user_vector(self, u, user_history_tracker=None):
        """
        Compute user representation for evaluation.

        If a UserHistoryTracker is provided and seq_cold_only is enabled,
        only cold users (short history) get sequence-aware encoding.
        """
        if user_history_tracker is not None:
            device = u.device
            max_len = self.cfg.seq_max_len
            seqs = []
            masks = []
            for uid in u.detach().cpu().tolist():
                seq = user_history_tracker.get_sequence(int(uid), max_len)
                seq_len = len(seq)
                padded = seq + [0] * (max_len - seq_len)
                mask = [False] * seq_len + [True] * (max_len - seq_len)
                seqs.append(padded)
                masks.append(mask)
            seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
            mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
            return self._encode_user_with_seq(u, seq_t, mask_t)
        return self.user_proj(self.user_emb(u))

    # ── Override forward to inject sequence-aware user repr ───────────

    def forward(self, batch, pop, llm_s, user_bank_raw=None,
                user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold

        # ▸▸▸ CHANGED: sequence-aware user representation ◂◂◂
        seq_items = batch.get("seq")
        seq_mask = batch.get("seq_mask")
        if seq_items is not None and seq_mask is not None:
            z_u_base = self._encode_user_with_seq(
                u, seq_items.to(self.device), seq_mask.to(self.device)
            )
        else:
            z_u_base = self.user_proj(self.user_emb(u))
        # ▸▸▸ END CHANGE ◂◂◂

        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(
            i, llm_s, force_cold=force_cold_mask
        )
        target_emb = z_i_base.detach().clone()
        final_h, trajectory, candidate_stats = self.run_usim_episode(
            z_i_base,
            target_emb,
            user_bank_raw=user_bank_raw,
            item_idx=i,
            target_pop=pop,
            user_seen_items=user_seen_items,
        )
        ppo_loss = self.compute_ppo_loss(trajectory)

        z_u = F.normalize(z_u_base, dim=1)
        z_i = F.normalize(final_h, dim=1)
        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
        labels = torch.arange(logits.size(0), device=self.device)
        pos_mask = torch.eye(logits.size(0), device=self.device).bool()
        logits_margin = logits.clone()
        logits_margin[pos_mask] -= self.cfg.margin / self.cfg.temp

        if (
            self.training
            and self.cfg.use_mixed_hard_neg
            and logits_margin.size(0) > 1
        ):
            batch_size = logits_margin.size(0)
            max_neg = batch_size - 1
            n_total_neg = min(self.cfg.train_num_negs, max_neg)
            if n_total_neg > 0:
                n_hard = int(n_total_neg * self.cfg.hard_neg_ratio)
                n_hard = max(0, min(n_hard, n_total_neg))
                n_rand = n_total_neg - n_hard
                neg_logits = logits_margin.clone()
                neg_logits[pos_mask] = -1e9
                hard_idx = torch.empty(
                    batch_size, 0, dtype=torch.long, device=self.device
                )
                rand_idx = torch.empty(
                    batch_size, 0, dtype=torch.long, device=self.device
                )
                if n_hard > 0:
                    if (
                        self.cfg.use_structured_hard_neg
                        and self.item_hard_adj is not None
                    ):
                        hard_mask = self.item_hard_adj[i][:, i]
                        hard_mask = hard_mask & (~pos_mask)
                        hard_logits = neg_logits.masked_fill(
                            ~hard_mask, -1e9
                        )
                        hard_scores, hard_idx = torch.topk(
                            hard_logits, k=n_hard, dim=1
                        )
                        valid_mask = hard_scores > -1e8
                        if (~valid_mask).any():
                            bad_rows = (
                                torch.nonzero(
                                    (~valid_mask).any(dim=1), as_tuple=False
                                )
                                .view(-1)
                                .tolist()
                            )
                            for row in bad_rows:
                                need = int(
                                    (~valid_mask[row]).sum().item()
                                )
                                if need < 1:
                                    continue
                                fallback = neg_logits[row].clone()
                                if valid_mask[row].any():
                                    fallback[
                                        hard_idx[row, valid_mask[row]]
                                    ] = -1e9
                                _, fill_idx = torch.topk(
                                    fallback, k=need, dim=0
                                )
                                hard_idx[
                                    row, ~valid_mask[row]
                                ] = fill_idx
                    else:
                        _, hard_idx = torch.topk(
                            neg_logits, k=n_hard, dim=1
                        )
                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(
                        rand_scores, k=n_rand, dim=1
                    )
                cand_idx = torch.cat(
                    [labels.view(-1, 1), hard_idx, rand_idx], dim=1
                )
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
                main_loss = F.cross_entropy(cand_logits, main_targets)
            else:
                main_loss = F.cross_entropy(logits_margin, labels)
        else:
            main_loss = F.cross_entropy(logits_margin, labels)

        z_id = F.normalize(id_e_true, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (
            F.cross_entropy(sim, labels)
            + F.cross_entropy(sim.t(), labels)
        ) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training
            and self.cfg.use_prereq_aux_loss
            and user_seen_items is not None
            and self.item_prereq_item_mat is not None
            and self.item_prereq_item_cnt is not None
            and logits.size(0) > 1
        ):
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            seen_mat, seen_cnt_raw = self._build_seen_mat(
                user_ids, user_seen_items
            )
            seen_cnt = seen_cnt_raw.squeeze(1)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(
                seen_mat, prereq_mat_batch.t()
            )
            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0
                - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch),
            ).clamp(0.0, 1.0)
            valid_rows = seen_cnt >= float(self.cfg.prereq_aux_min_seen)
            if self.cfg.prereq_aux_only_cold:
                valid_rows = valid_rows & is_cold
            unmet_mask = violation_batch > float(
                self.cfg.prereq_aux_violation_thr
            )
            unmet_mask = unmet_mask & (~pos_mask)
            candidate_mask = unmet_mask & valid_rows.unsqueeze(1)
            if candidate_mask.any():
                neg_scores = logits.masked_fill(~candidate_mask, -1e9)
                neg_vals, _ = neg_scores.max(dim=1)
                has_neg = neg_vals > -1e8
                if has_neg.any():
                    pos_vals = logits[
                        torch.arange(logits.size(0), device=self.device),
                        labels,
                    ]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(
                        margin - pos_vals[has_neg] + neg_vals[has_neg]
                    ).mean()

        total_loss = (
            main_loss
            + self.cfg.aux_weight * aux_loss
            + ppo_loss
            + self.cfg.prereq_aux_weight * prereq_aux_loss
        )
        return total_loss, candidate_stats


# =========================================================================
#  Dataset with user history sequences
# =========================================================================

class SeqStreamDataset(Dataset):
    """
    Extends StreamDataset to include padded user history sequences.

    Each sample contains:
      u, i, pop, llm_s  (same as StreamDataset)
      seq       : (max_seq_len,)   padded item ID sequence
      seq_mask  : (max_seq_len,)   True for PAD positions
    """

    def __init__(self, df, llm_scores, user_history_tracker, max_seq_len=50):
        user_ids = [int(x) for x in df["u_idx"].values]
        item_ids = [int(x) for x in df["i_idx"].values]
        self.u = torch.tensor(user_ids, dtype=torch.long)
        self.i = torch.tensor(item_ids, dtype=torch.long)
        self.pop = torch.tensor(df["popularity"].values, dtype=torch.long)
        self.llm_s = base._build_llm_score_tensor(
            llm_scores, user_ids, item_ids
        )
        self.max_seq_len = max_seq_len

        # Pre-build padded sequences for each sample
        self.seq_data = []
        self.seq_masks = []
        for uid in user_ids:
            seq = user_history_tracker.get_sequence(uid, max_seq_len)
            seq_len = len(seq)
            padded = seq + [0] * (max_seq_len - seq_len)
            mask = [False] * seq_len + [True] * (max_seq_len - seq_len)
            self.seq_data.append(padded)
            self.seq_masks.append(mask)

        # Convert to tensors for fast access
        self.seq_tensor = torch.tensor(self.seq_data, dtype=torch.long)
        self.mask_tensor = torch.tensor(self.seq_masks, dtype=torch.bool)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            "u": self.u[idx],
            "i": self.i[idx],
            "pop": self.pop[idx],
            "llm": self.llm_s[idx],
            "seq": self.seq_tensor[idx],
            "seq_mask": self.mask_tensor[idx],
        }


def seq_collate_fn(batch):
    """Collate function that includes sequence data."""
    u = torch.stack([item["u"] for item in batch])
    i = torch.stack([item["i"] for item in batch])
    pop = torch.stack([item["pop"] for item in batch])
    llm = torch.stack([item["llm"] for item in batch])
    seq = torch.stack([item["seq"] for item in batch])
    seq_mask = torch.stack([item["seq_mask"] for item in batch])
    return {"u": u, "i": i, "seq": seq, "seq_mask": seq_mask}, pop, llm


# =========================================================================
#  Evaluation with sequence-aware user representations
# =========================================================================

def evaluate_usim_seq(
    model,
    loader,
    device,
    llm_scores,
    user_history_tracker,
    k_list=[5, 10, 20],
    n_neg=200,
    eval_type="cold",
    full_ranking=False,
    all_item_vecs=None,
):
    """
    Same as base.evaluate_usim but uses model.get_user_vector() with
    the user_history_tracker for sequence-aware user representations.
    """
    model.eval()
    accum_metrics = {}
    total_samples = 0
    user_seen_items = user_history_tracker.seen_items
    seen_tensor_cache = {}

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        if all_item_vecs is None:
            all_item_vecs = base.build_eval_item_vecs(
                model, device, llm_scores, item_batch=1024
            )
        item_bank = base._select_eval_item_bank(all_item_vecs, eval_type)

        for batch, pop, llm in loader:
            if eval_type == "cold":
                mask = pop < model.cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)
            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u = batch["u"][mask].to(device)
            i = batch["i"][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            item_ids = [int(x) for x in i.detach().cpu().tolist()]

            for uid in user_ids:
                if uid in seen_tensor_cache:
                    continue
                seen_items = user_seen_items.get(uid)
                if seen_items:
                    seen_list = [
                        it for it in seen_items if 0 <= it < n_items
                    ]
                    seen_tensor_cache[uid] = (
                        torch.tensor(
                            seen_list, dtype=torch.long, device=device
                        )
                        if seen_list
                        else None
                    )
                else:
                    seen_tensor_cache[uid] = None

            # ▸▸▸ CHANGED: sequence-aware user vectors ◂◂◂
            z_u = F.normalize(
                model.get_user_vector(u, user_history_tracker), dim=1
            )
            # ▸▸▸ END CHANGE ◂◂◂

            pos_llm = base._build_llm_score_tensor(
                llm_scores, user_ids, item_ids, device=device
            )
            pos_vec = base._build_eval_pos_item_vecs(
                model, i, pos_llm, pop_sel, eval_type
            )
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
                    scores,
                    user_ids,
                    seen_tensor_cache,
                    cand_idx=None,
                    target_pop=pop_sel,
                )
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        avail = n_items - 1 - int(
                            (seen_idx != i[row]).sum().item()
                        )
                    avail_counts.append(max(1, avail))
                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty(
                    (n_sel, n_neg_batch), dtype=torch.long, device=device
                )
                for row, user_id in enumerate(user_ids):
                    forbidden = torch.zeros(
                        n_items, dtype=torch.bool, device=device
                    )
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache[int(user_id)]
                    if seen_idx is not None:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[
                        :n_neg_batch
                    ]
                    neg_items[row] = candidates[pick]
                cand_idx = torch.cat(
                    [i.unsqueeze(1), neg_items], dim=1
                )
                cand_vecs = item_bank[cand_idx].clone()
                cand_vecs[:, 0, :] = pos_vec
                scores = torch.bmm(
                    cand_vecs, z_u.unsqueeze(2)
                ).squeeze(2)
                scores = model.apply_course_rerank(
                    scores,
                    user_ids,
                    seen_tensor_cache,
                    cand_idx=cand_idx,
                    target_pop=pop_sel,
                )
                target_indices = torch.zeros(
                    n_sel, dtype=torch.long, device=device
                )

            batch_res = base.compute_ranking_metrics(
                scores, target_indices=target_indices, k_list=k_list
            )
            for k, v in batch_res.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * n_sel
            total_samples += n_sel

    if total_samples == 0:
        return None, 0
    return {k: v / total_samples for k, v in accum_metrics.items()}, total_samples


# =========================================================================
#  Checkpoint helpers
# =========================================================================

def _seq_ckpt_dir():
    return os.environ.get(
        "USIM_FB_CKPT_DIR",
        os.path.join("checkpoints", "usim_feedback_fast3_seq"),
    )


def _seq_output_dir():
    explicit = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    tag = os.environ.get("USIM_FB_OUTPUT_TAG", "").strip()
    if tag:
        path = os.path.join("outputs", "usim_feedback_fast3_seq", tag)
        os.makedirs(path, exist_ok=True)
        return path
    return "."


def _seq_output_path(filename):
    return os.path.join(_seq_output_dir(), filename)


def _build_seq_ckpt_state(
    model, optimizer, history,
    accum_cold, accum_hot, count_cold, count_hot,
    full_cold, full_hot, fc_cold, fc_hot,
    user_history_tracker,
    accumulated_periods, warmup_periods, total_periods,
    status, next_period,
    current_period=None, next_epoch=0,
    es_best=None, es_best_state=None, es_best_opt_state=None,
    es_no_improve=0,
):
    """Build checkpoint state including ordered user history."""
    state = base._build_feedback_ckpt_state(
        model, optimizer, history,
        accum_cold, accum_hot, count_cold, count_hot,
        full_cold, full_hot, fc_cold, fc_hot,
        user_history_tracker.seen_items,
        accumulated_periods, warmup_periods, total_periods,
        status, next_period,
        current_period=current_period,
        next_epoch=next_epoch,
        es_best=es_best,
        es_best_state=es_best_state,
        es_best_opt_state=es_best_opt_state,
        es_no_improve=es_no_improve,
    )
    state["user_history_ordered"] = user_history_tracker.serialize()
    return state


# =========================================================================
#  Main
# =========================================================================

MODEL_NAME = "USIM-Feedback-FAST3-Seq"


def main():
    data_dir = "processed_data_hin"
    print(f"Loading Data for {MODEL_NAME} from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        print("Error: please run data_process_hin.py first")
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    llm_scores, llm_score_path, _ = base.load_llm_scores_for_stream(
        data_dir, df,
        cold_threshold=5,
        n_users=meta.get("n_users"),
        n_items=meta.get("n_items"),
        fallback_data_dirs=["processed_data"],
    )
    content_emb = torch.load(f"{data_dir}/content_emb.pt")
    if llm_score_path:
        print(f"   LLM scores loaded from {llm_score_path}")

    # ── Config & Model ────────────────────────────────────────────────
    cfg = SeqConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = base._resolve_torch_device()

    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = base.build_course_artifacts(
            df, cfg.n_items,
            relation_dir="MOOCCube/relations",
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, course_stats = None, base._empty_course_stats(cfg.n_items)

    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    pop_stats = df.groupby("i_idx")["popularity"].max()
    for item_id, pop_value in pop_stats.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_final_pop[idx] = int(pop_value)

    model = SeqFast3FeedbackUSIM(cfg, content_emb).to(device)
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── Print config ──────────────────────────────────────────────────
    print(f">> Architecture: {MODEL_NAME} (Batch Size={cfg.batch_size})")
    print(f">> Device: {device}")
    print(
        f">> Sequence Encoder: max_len={cfg.seq_max_len} | n_heads={cfg.seq_n_heads} | "
        f"n_layers={cfg.seq_n_layers} | dropout={cfg.seq_dropout} | pool={cfg.seq_pool} | "
        f"cold_only={cfg.seq_cold_only} | user_cold_thr={cfg.seq_user_cold_thr}"
    )
    print(
        f">> Window={cfg.stream_train_window} | PPO epochs={cfg.ppo_epochs} | "
        f"lambda={cfg.ppo_lambda:.2f} | value_clip={cfg.ppo_value_clip:.2f} | "
        f"adv_norm={cfg.ppo_adv_norm}"
    )
    print(
        f">> Adaptive Mix: cold={cfg.fast3_target_alpha_cold:.2f} | "
        f"hot={cfg.fast3_target_alpha_hot:.2f} | step_gain={cfg.fast3_target_alpha_step:.2f} | "
        f"entropy_pen={cfg.fast3_target_alpha_entropy:.2f}"
    )
    print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    print(
        f">> Course Artifacts: enabled={cfg.feedback_load_course_artifacts} | "
        f"prereq_aux={cfg.use_prereq_aux_loss} | "
        f"rerank={cfg.use_course_rerank} | "
        f"struct_hard_neg={cfg.use_structured_hard_neg}"
    )
    print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

    # ── Streaming periods ─────────────────────────────────────────────
    periods = base.split_dataframe_by_periods(df, period_type="M")
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

    # ▸▸▸ CHANGED: use UserHistoryTracker instead of plain dict ◂◂◂
    user_history = UserHistoryTracker(max_seq_len=cfg.seq_max_len)

    ckpt_dir = _seq_ckpt_dir()
    ckpt_enabled = base._feedback_ckpt_enabled()
    auto_resume = base._feedback_ckpt_auto_resume()
    force_fresh = base._feedback_ckpt_force_fresh()
    print(
        f">> Checkpoint: enabled={ckpt_enabled} | auto_resume={auto_resume} | "
        f"force_fresh={force_fresh} | save_opt={base._feedback_ckpt_save_optimizer_state()} | "
        f"dir={ckpt_dir}"
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
        resume_state = base._load_feedback_checkpoint(ckpt_dir)
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
                if resume_state.get("optimizer_state") is not None:
                    optimizer.load_state_dict(resume_state["optimizer_state"])
                    base._optimizer_state_to_device(optimizer, device)
                history = copy.deepcopy(resume_state.get("history", history))
                accum_cold = copy.deepcopy(resume_state.get("accum_cold", accum_cold))
                accum_hot = copy.deepcopy(resume_state.get("accum_hot", accum_hot))
                count_cold = int(resume_state.get("count_cold", count_cold))
                count_hot = int(resume_state.get("count_hot", count_hot))
                full_cold = copy.deepcopy(resume_state.get("full_cold", full_cold))
                full_hot = copy.deepcopy(resume_state.get("full_hot", full_hot))
                fc_cold = int(resume_state.get("fc_cold", fc_cold))
                fc_hot = int(resume_state.get("fc_hot", fc_hot))
                warmup_periods = int(resume_state.get("warmup_periods", warmup_periods))
                resume_accumulated_periods = int(resume_state.get("accumulated_periods", 0))
                accumulated_dfs = periods[:resume_accumulated_periods]

                # Restore ordered user history
                if resume_state.get("user_history_ordered"):
                    user_history = UserHistoryTracker.deserialize(
                        resume_state["user_history_ordered"],
                        max_seq_len=cfg.seq_max_len,
                    )
                else:
                    # Fallback: reconstruct from accumulated DFs
                    user_history = UserHistoryTracker(max_seq_len=cfg.seq_max_len)
                    for adf in accumulated_dfs:
                        user_history.add_from_df(adf)

                start_period = int(resume_state.get("next_period", 0))
                resume_current_period = resume_state.get("current_period")
                if resume_current_period is not None:
                    resume_current_period = int(resume_current_period)
                    start_period = resume_current_period
                resume_next_epoch = int(resume_state.get("next_epoch", 0))
                resume_es_best = copy.deepcopy(resume_state.get("es_best"))
                resume_es_best_state = base._move_state_to_cpu(resume_state.get("es_best_state"))
                resume_es_best_opt_state = base._move_state_to_cpu(resume_state.get("es_best_opt_state"))
                resume_es_no_improve = int(resume_state.get("es_no_improve", 0))
                print(
                    f">> Resume: status={status} | start_period={start_period} | "
                    f"resume_current_period={resume_current_period} | "
                    f"next_epoch={resume_next_epoch} | "
                    f"accumulated_periods={resume_accumulated_periods}"
                )

    # ── Main loop ─────────────────────────────────────────────────────

    for t in range(start_period, len(periods)):
        p_df = periods[t]
        # Use base collate for eval (no seq needed in eval loader)
        eval_ds = base.StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(
            eval_ds, batch_size=2048, shuffle=False, collate_fn=base.collate_fn
        )
        n_total = len(eval_ds)
        print(
            f"\n>>> Period {t} (current={n_total}, "
            f"accumulated={sum(len(d) for d in accumulated_dfs) + n_total}) <<<"
        )

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0
        resume_this_period = (
            resume_current_period is not None and t == resume_current_period
        )

        if resume_this_period:
            print(f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs}")
        elif t >= warmup_periods:
            print("  [EVAL-START] Build eval item bank and run sampled/full ranking...")
            all_item_vecs_eval = base.build_eval_item_vecs(
                model, device, llm_scores, item_batch=1024
            )
            met_cold, n_cold_t = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, user_history,
                k_list, n_neg=cfg.eval_n_neg, eval_type="cold",
                all_item_vecs=all_item_vecs_eval,
            )
            met_hot, n_hot_t = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, user_history,
                k_list, n_neg=cfg.eval_n_neg, eval_type="hot",
                all_item_vecs=all_item_vecs_eval,
            )
            fmet_cold, fn_c = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, user_history,
                k_list, eval_type="cold", full_ranking=True,
                all_item_vecs=all_item_vecs_eval,
            )
            fmet_hot, fn_h = evaluate_usim_seq(
                model, eval_loader, device, llm_scores, user_history,
                k_list, eval_type="hot", full_ranking=True,
                all_item_vecs=all_item_vecs_eval,
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
            print(f"  Sampled Cold={c_s:.4f} Hot={h_s:.4f} | Full Cold={c_f:.4f} Hot={h_f:.4f}")
            del all_item_vecs_eval
            base._maybe_clear_cuda_cache()
        else:
            print("  [WARMUP] Training only...")

        if not resume_this_period:
            history["Period"].append(t)
            history["Count_cold"].append(n_cold_t)
            history["Count_hot"].append(n_hot_t)
            for key in metrics_keys:
                history["cold_" + key].append(cold_res.get(key, 0.0))
                history["hot_" + key].append(hot_res.get(key, 0.0))

            # ▸▸▸ CHANGED: update both ordered history and seen items ◂◂◂
            user_history.add_from_df(p_df)
            accumulated_dfs.append(p_df)

        # ── Training ──────────────────────────────────────────────────
        window = cfg.stream_train_window
        if window > 0 and len(accumulated_dfs) > window:
            train_dfs = accumulated_dfs[-window:]
            print(
                f"  [WINDOW] Use latest {window}/{len(accumulated_dfs)} periods "
                f"for training ({sum(len(d) for d in train_dfs)} samples)"
            )
        else:
            train_dfs = accumulated_dfs

        combined_df = pd.concat(train_dfs, ignore_index=True)

        # ▸▸▸ CHANGED: use SeqStreamDataset with history sequences ◂◂◂
        train_ds = SeqStreamDataset(
            combined_df, llm_scores, user_history,
            max_seq_len=cfg.seq_max_len,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=seq_collate_fn,
        )

        model.train()
        do_early_stop = (
            t >= warmup_periods
            and cfg.use_epoch_early_stop
            and cfg.n_epochs > 1
        )
        es_best = copy.deepcopy(resume_es_best) if resume_this_period else None
        es_best_state = copy.deepcopy(resume_es_best_state) if resume_this_period else None
        es_best_opt_state = copy.deepcopy(resume_es_best_opt_state) if resume_this_period else None
        es_no_improve = int(resume_es_no_improve) if resume_this_period else 0
        epoch_start_idx = resume_next_epoch if resume_this_period else 0

        if ckpt_enabled and not resume_this_period:
            _save_ckpt(
                ckpt_dir, model, optimizer, history,
                accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot,
                user_history, t + 1, warmup_periods, len(periods),
                "in_period", t, current_period=t, next_epoch=0,
            )

        for epoch in range(epoch_start_idx, cfg.n_epochs):
            epoch_start = time.time()
            num_batches = max(1, len(train_loader))
            total_loss = 0.0
            steps = 0
            cand_dup_sum = 0.0
            cand_cov_sum = 0.0
            cand_gain_sum = 0.0
            cand_pen_sum = 0.0
            cand_mix_sum = 0.0
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
                f"Period {t + 1}/{len(periods)} | samples={len(combined_df)} | "
                f"batches={num_batches}"
            )
            last_progress_log = epoch_start

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
                if (
                    cached_user_bank is not None
                    and cfg.user_bank_refresh_steps > 0
                    and batch_idx > 0
                    and batch_idx % cfg.user_bank_refresh_steps == 0
                ):
                    cached_user_bank = model._build_user_bank_raw()

                batch = {k: v.to(device) for k, v in batch.items()}
                loss, cand_info = model(
                    batch,
                    pop.to(device),
                    llm.to(device),
                    user_bank_raw=cached_user_bank,
                    user_seen_items=user_history.seen_items,
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
                    cand_mix_sum += cand_info.get("target_alpha", 0.0)
                    course_sample_fit_sum += cand_info.get("course_sample_fit", 0.0)
                    course_prereq_sum += cand_info.get("course_prereq_gap", 0.0)
                    course_concept_sum += cand_info.get("course_concept_bonus", 0.0)
                    course_diff_sum += cand_info.get("course_difficulty_gap", 0.0)
                    course_redundant_sum += cand_info.get("course_redundant", 0.0)
                    cand_batches += 1

                now_ts = time.time()
                if base._should_log_train_progress(
                    batch_idx, num_batches, cfg, last_progress_log, now_ts
                ):
                    done = batch_idx + 1
                    elapsed = now_ts - epoch_start
                    avg_batch_sec = elapsed / max(1, done)
                    eta = avg_batch_sec * max(0, num_batches - done)
                    pct = 100.0 * done / max(1, num_batches)
                    print(
                        f"    [TRAIN-PROGRESS] {done}/{num_batches} ({pct:.0f}%) | "
                        f"avg_loss={total_loss / max(1, steps):.4f} | "
                        f"elapsed={base._format_eta(elapsed)} | "
                        f"eta={base._format_eta(eta)}"
                    )
                    last_progress_log = now_ts

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                avg_gain = cand_gain_sum / cand_batches
                avg_pen = cand_pen_sum / cand_batches
                avg_mix = cand_mix_sum / cand_batches
                avg_csf = course_sample_fit_sum / cand_batches
                avg_cp = course_prereq_sum / cand_batches
                avg_cc = course_concept_sum / cand_batches
                avg_cd = course_diff_sum / cand_batches
                avg_cr = course_redundant_sum / cand_batches
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                    f"train={len(combined_df)} | Loss: {avg_loss:.4f} | "
                    f"Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"MixAlpha: {avg_mix:.4f} | SampleFit: {avg_csf:.4f} | "
                    f"Course[p={avg_cp:.4f}, c={avg_cc:.4f}, "
                    f"d={avg_cd:.4f}, r={avg_cr:.4f}]"
                )
            else:
                print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                    f"train={len(combined_df)} | Loss: {avg_loss:.4f} | "
                    f"Time: {epoch_sec:.1f}s"
                )

            if ckpt_enabled:
                _save_ckpt(
                    ckpt_dir, model, optimizer, history,
                    accum_cold, accum_hot, count_cold, count_hot,
                    full_cold, full_hot, fc_cold, fc_hot,
                    user_history, t + 1, warmup_periods, len(periods),
                    "in_period", t, current_period=t, next_epoch=epoch + 1,
                    es_best=es_best, es_best_state=es_best_state,
                    es_best_opt_state=es_best_opt_state,
                    es_no_improve=es_no_improve,
                )

            # ── Early stopping ────────────────────────────────────────
            if do_early_stop:
                print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
                all_item_vecs_es = base.build_eval_item_vecs(
                    model, device, llm_scores, item_batch=1024
                )
                es_cold, _ = evaluate_usim_seq(
                    model, eval_loader, device, llm_scores, user_history,
                    k_list, eval_type="cold", full_ranking=True,
                    all_item_vecs=all_item_vecs_es,
                )
                es_hot, _ = evaluate_usim_seq(
                    model, eval_loader, device, llm_scores, user_history,
                    k_list, eval_type="hot", full_ranking=True,
                    all_item_vecs=all_item_vecs_es,
                )
                key_n = f"N@{cfg.early_stop_k}"
                key_r = f"R@{cfg.early_stop_k}"
                cur_n = es_cold.get(key_n, 0.0) if es_cold else 0.0
                cur_cr = es_cold.get(key_r, 0.0) if es_cold else 0.0
                cur_hr = es_hot.get(key_r, 0.0) if es_hot else 0.0

                if es_best is None:
                    is_better = True
                else:
                    hot_floor = es_best["hot_r"] * (
                        1.0 - cfg.early_stop_hot_r10_drop_tol
                    )
                    hot_ok = cur_hr >= hot_floor
                    n_improve = cur_n > es_best["cold_n"] + cfg.early_stop_min_delta
                    n_tie = abs(cur_n - es_best["cold_n"]) <= cfg.early_stop_min_delta
                    r_tie_break = cur_cr > es_best["cold_r"] + 1e-12
                    is_better = hot_ok and (n_improve or (n_tie and r_tie_break))

                if is_better:
                    es_best = {
                        "epoch": epoch + 1,
                        "cold_n": float(cur_n),
                        "cold_r": float(cur_cr),
                        "hot_r": float(cur_hr),
                    }
                    es_best_state = base._move_state_to_cpu(model.state_dict())
                    es_best_opt_state = base._move_state_to_cpu(
                        optimizer.state_dict()
                    )
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                print(
                    f"  [EARLYSTOP] Epoch {epoch + 1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | "
                    f"{es_tag}"
                )
                del all_item_vecs_es, es_cold, es_hot
                base._maybe_clear_cuda_cache()

                if ckpt_enabled:
                    _save_ckpt(
                        ckpt_dir, model, optimizer, history,
                        accum_cold, accum_hot, count_cold, count_hot,
                        full_cold, full_hot, fc_cold, fc_hot,
                        user_history, t + 1, warmup_periods, len(periods),
                        "in_period", t, current_period=t,
                        next_epoch=epoch + 1,
                        es_best=es_best, es_best_state=es_best_state,
                        es_best_opt_state=es_best_opt_state,
                        es_no_improve=es_no_improve,
                    )

                if es_no_improve >= cfg.early_stop_patience:
                    print(f"  [EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    break

        # ── Restore best (early stop) ─────────────────────────────────
        if do_early_stop and es_best_state is not None:
            model.load_state_dict(es_best_state)
            if es_best_opt_state is not None:
                optimizer.load_state_dict(es_best_opt_state)
                base._optimizer_state_to_device(optimizer, device)
            print(
                f"  [EARLYSTOP] Restore best epoch={es_best['epoch']} "
                f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
                f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
                f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
            )
            base._maybe_clear_cuda_cache()

        if ckpt_enabled:
            _save_ckpt(
                ckpt_dir, model, optimizer, history,
                accum_cold, accum_hot, count_cold, count_hot,
                full_cold, full_hot, fc_cold, fc_hot,
                user_history, t + 1, warmup_periods, len(periods),
                "between_periods", t + 1,
            )

        if resume_this_period:
            resume_current_period = None
            resume_next_epoch = 0
            resume_es_best = None
            resume_es_best_state = None
            resume_es_best_opt_state = None
            resume_es_no_improve = 0

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: sampled (1+{cfg.eval_n_neg}) vs full ranking ({MODEL_NAME})")
    print("=" * 90)
    print(
        f"{'Metric':<10} | {'Sampled Cold':<12} | {'Sampled Hot':<12} | "
        f"{'Full Cold':<12} | {'Full Hot':<12}"
    )
    print("-" * 90)
    summary_rows = []
    sampled_row = {
        "Model": MODEL_NAME, "Eval": "sampled",
        "ColdSamples": count_cold, "HotSamples": count_hot,
    }
    full_row = {
        "Model": MODEL_NAME, "Eval": "full_rank",
        "ColdSamples": fc_cold, "HotSamples": fc_hot,
    }
    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
        sampled_row[f"Cold_{key}"] = sc
        sampled_row[f"Hot_{key}"] = sh
        full_row[f"Cold_{key}"] = fc
        full_row[f"Hot_{key}"] = fh
    print("-" * 90)
    print(f"Sampled Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"Full Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)
    summary_rows.extend([sampled_row, full_row])

    final_sampled_cold = {
        key: (accum_cold[key] / count_cold if count_cold > 0 else 0.0)
        for key in metrics_keys
    }
    final_sampled_hot = {
        key: (accum_hot[key] / count_hot if count_hot > 0 else 0.0)
        for key in metrics_keys
    }
    final_full_cold = {
        key: (full_cold[key] / fc_cold if fc_cold > 0 else 0.0)
        for key in metrics_keys
    }
    final_full_hot = {
        key: (full_hot[key] / fc_hot if fc_hot > 0 else 0.0)
        for key in metrics_keys
    }
    detail_path, fullrank_path = base._save_final_report_exports(
        protocol="stream",
        metrics_keys=metrics_keys,
        sampled_cold=final_sampled_cold,
        sampled_hot=final_sampled_hot,
        full_cold=final_full_cold,
        full_hot=final_full_hot,
        sampled_cold_count=count_cold,
        sampled_hot_count=count_hot,
        full_cold_count=fc_cold,
        full_hot_count=fc_hot,
        model_name=MODEL_NAME,
    )

    suffix = "fast3_seq"
    metrics_path = _seq_output_path(f"mooc_metrics_usim_feedback_{suffix}.csv")
    summary_path = _seq_output_path(f"mooc_metrics_usim_feedback_{suffix}_summary.csv")
    plot_path = _seq_output_path(f"mooc_result_usim_feedback_{suffix}.png")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title(f"RL-USIM [{MODEL_NAME}]: FAST3 + Transformer User Sequence")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(plot_path)
    print(
        f">> Saved {plot_path}, {metrics_path}, {summary_path}, "
        f"{detail_path}, and {fullrank_path}"
    )

    if ckpt_enabled:
        _save_ckpt(
            ckpt_dir, model, optimizer, history,
            accum_cold, accum_hot, count_cold, count_hot,
            full_cold, full_hot, fc_cold, fc_hot,
            user_history, len(periods), warmup_periods, len(periods),
            "finished", len(periods),
            snapshot_name="finished.pt",
        )


# ── Checkpoint save shorthand ─────────────────────────────────────────

def _save_ckpt(
    ckpt_dir, model, optimizer, history,
    accum_cold, accum_hot, count_cold, count_hot,
    full_cold, full_hot, fc_cold, fc_hot,
    user_history, accumulated_periods, warmup_periods, total_periods,
    status, next_period,
    current_period=None, next_epoch=0,
    es_best=None, es_best_state=None, es_best_opt_state=None,
    es_no_improve=0, snapshot_name=None,
):
    state = _build_seq_ckpt_state(
        model, optimizer, history,
        accum_cold, accum_hot, count_cold, count_hot,
        full_cold, full_hot, fc_cold, fc_hot,
        user_history,
        accumulated_periods, warmup_periods, total_periods,
        status, next_period,
        current_period=current_period,
        next_epoch=next_epoch,
        es_best=es_best,
        es_best_state=es_best_state,
        es_best_opt_state=es_best_opt_state,
        es_no_improve=es_no_improve,
    )
    base._save_feedback_checkpoint(ckpt_dir, state, snapshot_name=snapshot_name)


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_seed(2025)
    main()
