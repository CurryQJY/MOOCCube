"""C3K's isolated, evidence-consistent course encoder and ranking model.

This module deliberately reuses only the legacy encoder and course-artifact
containers.  It does not invoke the legacy simulator, PPO objective, reward,
or refined-inference path.
"""

from __future__ import annotations

import os
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


class C3KFeedbackUSIM(Fast3FeedbackUSIM):
    """A strict cold-start ranker with an explicit course-ID evidence mask."""

    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.c3k_gate_max = float(os.environ.get("C3K_GATE_MAX", "0.20"))
        if self.c3k_gate_max <= 0.0:
            raise ValueError("C3K_GATE_MAX must be positive")
        self.c3k_consistency_weight = float(
            os.environ.get("C3K_CONSISTENCY_WEIGHT", "0.10")
        )
        self.c3k_gate_weight = float(os.environ.get("C3K_GATE_WEIGHT", "0.001"))
        self.c3k_train_negatives = max(1, int(os.environ.get("C3K_TRAIN_NEGATIVES", "16")))
        self.c3k_residual = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.c3k_gate = nn.Sequential(
            nn.Linear(config.emb_dim * 2 + 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 4),
        )
        self.c3k_adapter = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim),
            nn.GELU(),
            nn.Linear(config.emb_dim, config.emb_dim),
        )
        for module in (self.c3k_residual, self.c3k_gate, self.c3k_adapter):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.c3k_residual[-1].weight)
        nn.init.zeros_(self.c3k_residual[-1].bias)
        nn.init.zeros_(self.c3k_gate[-1].weight)
        nn.init.constant_(self.c3k_gate[-1].bias, -4.0)

    def _empty_llm(self, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (int(item_ids.numel()),),
            -1.0,
            dtype=torch.float32,
            device=item_ids.device,
        )

    def _as_cold_mask(self, mask: torch.Tensor | bool, item_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(mask, bool):
            return torch.full(
                (int(item_ids.numel()),), mask, dtype=torch.bool, device=item_ids.device
            )
        mask_t = torch.as_tensor(mask, device=item_ids.device, dtype=torch.bool).view(-1)
        if int(mask_t.numel()) != int(item_ids.numel()):
            raise ValueError("cold_style_mask must contain exactly one value per item")
        return mask_t

    def _course_content(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute one shared content realization for the full/masked pair."""
        return F.normalize(self._content_base_embedding(item_ids), dim=1)

    def _fuse_course_evidence(
        self,
        item_ids: torch.Tensor,
        content_e: torch.Tensor,
        cold_style_mask: torch.Tensor,
    ) -> torch.Tensor:
        id_e = self.item_id_emb(item_ids)
        id_e = torch.where(cold_style_mask[:, None], torch.zeros_like(id_e), id_e)
        alpha = self.gate_net(torch.cat([id_e, content_e], dim=1))
        return F.normalize(alpha * id_e + (1.0 - alpha) * content_e, dim=1)

    def paired_item_views(
        self,
        item_ids: torch.Tensor,
        llm_scores: torch.Tensor | None,
        pseudo_mask: torch.Tensor | bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full and cold-style views with identical content evidence.

        ``llm_scores`` is accepted for the runner-compatible signature but is
        intentionally unused: C3K's frozen launch disables LLM score features.
        """
        del llm_scores
        item_ids = item_ids.to(device=self.device, dtype=torch.long).view(-1)
        mask = self._as_cold_mask(pseudo_mask, item_ids)
        content_e = self._course_content(item_ids)
        full = self._fuse_course_evidence(
            item_ids, content_e, torch.zeros_like(mask, dtype=torch.bool)
        )
        masked = self._fuse_course_evidence(item_ids, content_e, mask)
        return full, torch.where(mask[:, None], masked, full)

    def item_view(
        self,
        item_ids: torch.Tensor,
        llm_scores: torch.Tensor | None,
        cold_style_mask: torch.Tensor | bool,
    ) -> torch.Tensor:
        """Build an item representation without stochastic ID dropout."""
        _, masked_or_full = self.paired_item_views(item_ids, llm_scores, cold_style_mask)
        return masked_or_full

    def build_item_bank(self, strict_cold_mask: torch.Tensor) -> torch.Tensor:
        item_ids = torch.arange(self.cfg.n_items, dtype=torch.long, device=self.device)
        mask = self._as_cold_mask(strict_cold_mask, item_ids)
        return self.item_view(item_ids, self._empty_llm(item_ids), mask)

    def structural_feature_grid(
        self,
        user_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
    ) -> torch.Tensor:
        """Vectorize C3K features over an item block without duplicating history.

        For each candidate, target removal is performed algebraically from the
        user's one shared train-history vector.  This is exactly equivalent to
        physically zeroing that candidate column for every pair, while keeping
        full ranking practical on the MOOCcube catalog.
        """
        user_ids = user_ids.to(device=self.device, dtype=torch.long).view(-1)
        candidate_item_ids = candidate_item_ids.to(
            device=self.device, dtype=torch.long
        )
        if candidate_item_ids.dim() != 2:
            raise ValueError("candidate_item_ids must have shape (n_users, n_candidates)")
        if int(candidate_item_ids.size(0)) != int(user_ids.numel()):
            raise ValueError("candidate_item_ids must have one row per user")
        if bool(((candidate_item_ids < 0) | (candidate_item_ids >= self.cfg.n_items)).any()):
            raise ValueError("candidate_item_ids contains an invalid catalog index")

        seen, _ = self._build_seen_mat(user_ids, user_history)
        seen = seen.float()
        seen_count = seen.sum(dim=1, keepdim=True)
        target_seen = seen.gather(1, candidate_item_ids)
        effective_seen_count = (seen_count - target_seen).clamp_min(0.0)
        shape = candidate_item_ids.shape
        zero = torch.zeros(shape, dtype=torch.float32, device=self.device)

        concept = zero
        if self.item_concept_overlap is not None:
            overlap = self.item_concept_overlap.to(device=self.device, dtype=torch.float32)
            all_concept_sum = torch.matmul(seen, overlap.t())
            concept_sum = all_concept_sum.gather(1, candidate_item_ids)
            overlap_diagonal = torch.diagonal(overlap, offset=0)
            concept_sum = concept_sum - target_seen * overlap_diagonal[candidate_item_ids]
            concept = (concept_sum / effective_seen_count.clamp_min(1.0)).clamp(0.0, 1.0)

        prereq_gap = zero
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_matrix = self.item_prereq_item_mat.to(
                device=self.device, dtype=torch.float32
            )
            all_prereq_seen = torch.matmul(seen, prereq_matrix.t())
            satisfied = all_prereq_seen.gather(1, candidate_item_ids)
            prereq_diagonal = torch.diagonal(prereq_matrix, offset=0)
            satisfied = satisfied - target_seen * prereq_diagonal[candidate_item_ids]
            prereq_count = self.item_prereq_item_cnt.to(
                device=self.device, dtype=torch.float32
            )[candidate_item_ids]
            prereq_gap = torch.where(
                prereq_count > 0.0,
                1.0 - satisfied / prereq_count.clamp_min(1.0),
                zero,
            ).clamp(0.0, 1.0)

        difficulty_gap = zero
        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty.to(
                device=self.device, dtype=torch.float32
            )[candidate_item_ids]
            warm_seen = max(1.0, float(os.environ.get("C3K_WARM_SEEN", "5")))
            readiness = (effective_seen_count / warm_seen).clamp(0.0, 1.0)
            difficulty_gap = F.relu(item_difficulty - readiness).clamp(0.0, 1.0)

        redundant_threshold = float(os.environ.get("C3K_REDUNDANCY_THRESHOLD", "0.70"))
        redundant_threshold = min(0.99, max(0.0, redundant_threshold))
        redundancy = ((concept - redundant_threshold) / (1.0 - redundant_threshold)).clamp(
            0.0, 1.0
        )
        return torch.stack([concept, prereq_gap, difficulty_gap, redundancy], dim=2)

    def structural_features(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
    ) -> torch.Tensor:
        """Return C3K's four target-excluded pair features in [0, 1]."""
        item_ids = item_ids.to(device=self.device, dtype=torch.long).view(-1)
        return self.structural_feature_grid(user_ids, item_ids[:, None], user_history)[:, 0, :]

    def knowledge_coefficients(
        self,
        user_vectors: torch.Tensor,
        item_vectors: torch.Tensor,
        structural_features: torch.Tensor,
    ) -> torch.Tensor:
        """Bound and sign-constrain the interpretable calibration coefficients."""
        if user_vectors.shape != item_vectors.shape:
            raise ValueError("user_vectors and item_vectors must have identical shapes")
        if structural_features.dim() != 2 or structural_features.size(1) != 4:
            raise ValueError("structural_features must have shape (n_pairs, 4)")
        if structural_features.size(0) != user_vectors.size(0):
            raise ValueError("structural_features must have one row per user-item pair")
        raw = self.c3k_gate(
            torch.cat([user_vectors, item_vectors, structural_features], dim=1)
        )
        magnitude = self.c3k_gate_max * torch.sigmoid(raw)
        return torch.cat([magnitude[:, :1], -magnitude[:, 1:]], dim=1)

    def _score_from_embeddings(
        self,
        user_vectors: torch.Tensor,
        item_vectors: torch.Tensor,
        features: torch.Tensor,
        calibration: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vectors = F.normalize(user_vectors, dim=1)
        item_vectors = F.normalize(item_vectors, dim=1)
        base = (user_vectors * item_vectors).sum(dim=1, keepdim=True) / float(self.cfg.temp)
        residual = self.c3k_residual(torch.cat([user_vectors, item_vectors], dim=1))
        coefficients = self.knowledge_coefficients(user_vectors, item_vectors, features)
        calibration_term = (coefficients * features).sum(dim=1, keepdim=True)
        if not calibration:
            calibration_term = torch.zeros_like(calibration_term)
        return base + residual + calibration_term, coefficients

    def _score_candidate_grid(
        self,
        user_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        candidate_vectors: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
        *,
        calibration: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the same pair score over a user-by-candidate grid."""
        user_ids = user_ids.to(device=self.device, dtype=torch.long).view(-1)
        candidate_item_ids = candidate_item_ids.to(
            device=self.device, dtype=torch.long
        )
        candidate_vectors = candidate_vectors.to(device=self.device, dtype=torch.float32)
        if candidate_item_ids.dim() != 2 or candidate_vectors.dim() != 3:
            raise ValueError("candidate IDs/vectors must have shapes (users, candidates) and (users, candidates, dim)")
        if candidate_item_ids.shape != candidate_vectors.shape[:2]:
            raise ValueError("candidate IDs and vectors must share user/candidate dimensions")
        if int(candidate_item_ids.size(0)) != int(user_ids.numel()):
            raise ValueError("candidate rows must match user IDs")
        n_users, n_candidates = candidate_item_ids.shape
        user_vectors = self.user_proj(self.user_emb(user_ids))
        features = self.structural_feature_grid(user_ids, candidate_item_ids, user_history)
        flat_users = user_vectors[:, None, :].expand(-1, n_candidates, -1).reshape(-1, user_vectors.size(1))
        flat_items = candidate_vectors.reshape(-1, candidate_vectors.size(2))
        flat_features = features.reshape(-1, 4)
        flat_scores, flat_coefficients = self._score_from_embeddings(
            flat_users, flat_items, flat_features, calibration
        )
        return (
            flat_scores.view(n_users, n_candidates),
            flat_coefficients.view(n_users, n_candidates, 4),
        )

    def score_pairs(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_vectors: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
        *,
        calibration: bool = True,
        return_coefficients: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score aligned user/item pairs through C3K's only ranking function."""
        user_ids = user_ids.to(device=self.device, dtype=torch.long).view(-1)
        item_ids = item_ids.to(device=self.device, dtype=torch.long).view(-1)
        item_vectors = item_vectors.to(device=self.device, dtype=torch.float32)
        if item_vectors.dim() != 2 or int(item_vectors.size(0)) != int(item_ids.numel()):
            raise ValueError("item_vectors must be a two-dimensional aligned pair tensor")
        scores, coefficients = self._score_candidate_grid(
            user_ids,
            item_ids[:, None],
            item_vectors[:, None, :],
            user_history,
            calibration=calibration,
        )
        if return_coefficients:
            return scores[:, 0:1], coefficients[:, 0, :]
        return scores[:, 0:1]

    def score_catalog(
        self,
        user_ids: torch.Tensor,
        item_bank: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
        *,
        item_block: int = 128,
        calibration: bool = True,
    ) -> torch.Tensor:
        """Score the whole catalog through the same vectorized pair score."""
        user_ids = user_ids.to(device=self.device, dtype=torch.long).view(-1)
        item_bank = item_bank.to(device=self.device, dtype=torch.float32)
        if item_bank.dim() != 2 or int(item_bank.size(0)) != int(self.cfg.n_items):
            raise ValueError("item_bank must contain one vector per catalog item")
        block_size = max(1, int(item_block))
        n_users = int(user_ids.numel())
        item_ids = torch.arange(self.cfg.n_items, device=self.device, dtype=torch.long)
        score_blocks = []
        for start in range(0, self.cfg.n_items, block_size):
            candidate_ids = item_ids[start : start + block_size]
            n_candidates = int(candidate_ids.numel())
            candidate_grid = candidate_ids[None, :].expand(n_users, n_candidates)
            vector_grid = item_bank.index_select(0, candidate_ids)[None, :, :].expand(
                n_users, n_candidates, -1
            )
            pair_scores, _ = self._score_candidate_grid(
                user_ids,
                candidate_grid,
                vector_grid,
                user_history,
                calibration=calibration,
            )
            score_blocks.append(pair_scores)
        return torch.cat(score_blocks, dim=1)

    def _sampled_rank_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_vectors: torch.Tensor,
        user_history: Mapping[int, set[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use C3K score on a positive plus in-batch sampled negatives."""
        batch_size = int(user_ids.numel())
        zero = item_vectors.sum() * 0.0
        if batch_size < 2:
            return zero, item_vectors.new_zeros((0, 4))
        n_negatives = min(self.c3k_train_negatives, batch_size - 1)
        positive_positions = torch.arange(batch_size, device=self.device).view(-1, 1)
        offsets = torch.randint(
            1, batch_size, (batch_size, n_negatives), device=self.device
        )
        negative_positions = (positive_positions + offsets) % batch_size
        candidate_positions = torch.cat([positive_positions, negative_positions], dim=1)
        candidates_per_row = int(candidate_positions.size(1))

        candidate_item_ids = item_ids.index_select(0, candidate_positions.reshape(-1)).view(
            batch_size, candidates_per_row
        )
        candidate_vectors = item_vectors.index_select(0, candidate_positions.reshape(-1)).view(
            batch_size, candidates_per_row, item_vectors.size(1)
        )
        logits, coefficient_grid = self._score_candidate_grid(
            user_ids,
            candidate_item_ids,
            candidate_vectors,
            user_history,
            calibration=True,
        )

        known_positive = self._build_known_positive_batch_mask(
            user_ids, item_ids, user_history
        )
        invalid = candidate_item_ids.eq(item_ids[:, None])
        if known_positive is not None:
            invalid = invalid | known_positive.gather(1, candidate_positions)
        invalid[:, 0] = False
        logits = logits.masked_fill(invalid, -1e9)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels), coefficient_grid

    def forward(
        self,
        batch,
        pop: torch.Tensor,
        llm_scores: torch.Tensor,
        user_bank_raw=None,
        user_seen_items: Mapping[int, set[int]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Train C3K without any simulator/RL computation."""
        del user_bank_raw
        user_ids = batch["u"].to(device=self.device, dtype=torch.long).view(-1)
        item_ids = batch["i"].to(device=self.device, dtype=torch.long).view(-1)
        pop = pop.to(device=self.device, dtype=torch.float32).view(-1)
        llm_scores = llm_scores.to(device=self.device, dtype=torch.float32).view(-1)
        if int(user_ids.numel()) != int(item_ids.numel()):
            raise ValueError("batch users and items must have the same length")

        pseudo_mask = self._pseudo_cold_mask_for_items(item_ids)
        strict_cold = self._cold_mask_from_pop(pop)
        if strict_cold is None:
            strict_cold = torch.zeros_like(pseudo_mask)
        cold_style_mask = pseudo_mask | strict_cold
        full_view, cold_style_view = self.paired_item_views(
            item_ids, llm_scores, cold_style_mask
        )
        rank_view = torch.where(cold_style_mask[:, None], cold_style_view, full_view)
        rank_loss, coefficient_grid = self._sampled_rank_loss(
            user_ids, item_ids, rank_view, user_seen_items
        )

        zero = rank_view.sum() * 0.0
        pseudo_count = int(pseudo_mask.sum().detach().item())
        consistency_loss = zero
        gate_smoothness = zero
        if pseudo_count > 0:
            masked_adapter = self.c3k_adapter(cold_style_view[pseudo_mask])
            with torch.no_grad():
                full_adapter = self.c3k_adapter(full_view[pseudo_mask])
            consistency_loss = (
                1.0 - F.cosine_similarity(masked_adapter, full_adapter, dim=1)
            ).mean()
            _, full_coefficients = self.score_pairs(
                user_ids[pseudo_mask],
                item_ids[pseudo_mask],
                full_view[pseudo_mask],
                user_seen_items,
                return_coefficients=True,
            )
            if coefficient_grid.numel() > 0:
                masked_coefficients = coefficient_grid[:, 0, :][pseudo_mask]
                gate_smoothness = F.mse_loss(masked_coefficients, full_coefficients)

        gate_regularization = zero
        if coefficient_grid.numel() > 0:
            gate_regularization = coefficient_grid.pow(2).mean() + 0.1 * gate_smoothness
        loss = (
            rank_loss
            + self.c3k_consistency_weight * consistency_loss
            + self.c3k_gate_weight * gate_regularization
        )
        max_gate_abs = (
            float(coefficient_grid.detach().abs().max().item())
            if coefficient_grid.numel() > 0
            else 0.0
        )
        diagnostics = {
            "rank_loss": float(rank_loss.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "gate_regularization": float(gate_regularization.detach().item()),
            "gate_abs_max": max_gate_abs,
            "pseudo_cold_count": pseudo_count,
            "pseudo_cold_ratio": float(pseudo_mask.float().mean().detach().item()),
            "strict_cold_count": int(strict_cold.sum().detach().item()),
        }
        return loss, diagnostics


__all__ = ["C3KFeedbackUSIM"]
