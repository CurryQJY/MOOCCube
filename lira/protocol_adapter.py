from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from .config import LIRAConfig
from .model import LIRAModel
from .refinement import learner_fit


class LIRAProtocolAdapter(LIRAModel):
    """Expose the shared static protocol interface without inheriting an old model."""

    def __init__(self, protocol_config, content_embeddings: torch.Tensor):
        model_config = LIRAConfig(
            n_users=protocol_config.n_users,
            n_items=protocol_config.n_items,
            content_dim=content_embeddings.shape[1],
            embedding_dim=protocol_config.emb_dim,
            hidden_dim=protocol_config.hidden_dim,
            dropout=protocol_config.dropout_prob,
            temperature=protocol_config.temp,
            margin=protocol_config.margin,
            steps=int(getattr(protocol_config, "usim_steps", 3)),
            update_lr=float(os.environ.get("LIRA_UPDATE_LR", "0.10")),
            min_fit=float(os.environ.get("LIRA_MIN_FIT", "0.05")),
            step_cap=float(os.environ.get("LIRA_STEP_CAP", "0.05")),
            total_cap=float(os.environ.get("LIRA_TOTAL_CAP", "0.10")),
            min_gain=float(os.environ.get("LIRA_MIN_GAIN", "0.001")),
            refinement_loss_weight=float(os.environ.get("LIRA_REFINEMENT_LOSS_WEIGHT", "0.5")),
            stability_loss_weight=float(os.environ.get("LIRA_STABILITY_LOSS_WEIGHT", "0.01")),
            pseudo_cold_ratio=float(getattr(protocol_config, "pseudo_cold_ratio", 0.30)),
            pseudo_cold_min_popularity=int(getattr(protocol_config, "pseudo_cold_min_pop", 5)),
        )
        super().__init__(model_config, content_embeddings)
        self.cfg = protocol_config
        self.device = torch.device("cpu")
        self.user_seen_index = None
        self.item_popularity = None
        self.item_difficulty = None
        self.fixed_pseudo_cold_mask = None
        self.item_prereq_item_mat = None
        self.item_prereq_item_cnt = None
        self.item_concept_overlap = None
        self.item_same_family = None
        self.item_video_contain = None

    @property
    def user_emb(self):
        return self.user_embedding

    @property
    def user_proj(self):
        return self.user_projection

    def set_course_artifacts(self, artifacts) -> None:
        if not artifacts:
            return
        for name in (
            "item_prereq_item_mat", "item_prereq_item_cnt", "item_concept_overlap",
            "item_same_family", "item_video_contain",
        ):
            value = artifacts.get(name)
            setattr(self, name, value.to(self.device) if value is not None else None)

    def set_feedback_item_stats(self, item_popularity: torch.Tensor) -> None:
        pop = item_popularity.to(self.device).float().view(-1)
        self.item_popularity = pop
        self.item_difficulty = torch.log1p(pop) / torch.log1p(pop.max().clamp_min(1.0))
        eligible = pop >= float(self.config.pseudo_cold_min_popularity)
        rows = sorted(
            ((float(pop[idx].item()), int(idx)) for idx in eligible.nonzero().view(-1).cpu()),
            key=lambda pair: (pair[0], pair[1]),
        )
        target_mass = self.config.pseudo_cold_ratio * sum(value for value, _ in rows)
        selected, cumulative = [], 0.0
        for value, idx in rows:
            selected.append(idx)
            cumulative += value
            if cumulative >= target_mass:
                break
        mask = torch.zeros(self.config.n_items, dtype=torch.bool, device=self.device)
        if selected:
            mask[torch.tensor(selected, device=self.device)] = True
        self.fixed_pseudo_cold_mask = mask

    @torch.no_grad()
    def set_user_seen_index(self, user_seen_items) -> None:
        if user_seen_items is None:
            self.user_seen_index = None
            return
        matrix = torch.zeros(
            (self.config.n_users, self.config.n_items), dtype=torch.bool, device=self.device
        )
        for user_id, items in user_seen_items.items():
            valid = [int(item) for item in items if 0 <= int(item) < self.config.n_items]
            if valid and 0 <= int(user_id) < self.config.n_users:
                matrix[int(user_id), torch.tensor(valid, device=self.device)] = True
        self.user_seen_index = matrix

    def _effective_cold(self, popularity: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        true_cold = popularity.to(self.device).view(-1) < float(self.cfg.cold_threshold)
        if self.training and self.fixed_pseudo_cold_mask is not None:
            return true_cold | self.fixed_pseudo_cold_mask[item_ids]
        return true_cold

    def _user_bank(self) -> torch.Tensor:
        user_ids = torch.arange(self.config.n_users, device=self.device)
        return self.encode_users(user_ids)

    def _candidate_pool(self, item_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bank = self._user_bank()
        scores = F.normalize(item_vectors, dim=1) @ F.normalize(bank, dim=1).t()
        count = min(int(getattr(self.cfg, "num_candidates", 20)), self.config.n_users)
        candidate_ids = torch.topk(scores, k=count, dim=1).indices
        return bank[candidate_ids].detach(), candidate_ids

    def _candidate_fit(
        self,
        item_vectors: torch.Tensor,
        item_ids: torch.Tensor,
        candidates: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, candidate_count = candidate_ids.shape
        similarity = (F.normalize(item_vectors, dim=1).unsqueeze(1) * F.normalize(candidates, dim=2)).sum(2)
        flat_users = candidate_ids.reshape(-1)
        flat_items = item_ids.view(-1, 1).expand(-1, candidate_count).reshape(-1)
        if self.user_seen_index is None:
            seen = torch.zeros((flat_users.numel(), self.config.n_items), device=self.device)
        else:
            seen = self.user_seen_index[flat_users].float().clone()
        if self.fixed_pseudo_cold_mask is not None:
            seen[:, self.fixed_pseudo_cold_mask] = 0.0
        seen.scatter_(1, flat_items.view(-1, 1), 0.0)
        seen_count = seen.sum(1, keepdim=True)
        concept = torch.zeros_like(seen_count)
        if self.item_concept_overlap is not None:
            concept = (
                self.item_concept_overlap[flat_items].float() * seen
            ).sum(1, keepdim=True) / seen_count.clamp_min(1.0)
        prerequisite_gap = torch.zeros_like(seen_count)
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            completed = (self.item_prereq_item_mat[flat_items].float() * seen).sum(1, keepdim=True)
            required = self.item_prereq_item_cnt[flat_items].float().view(-1, 1)
            prerequisite_gap = torch.where(
                required > 0, 1.0 - completed / required.clamp_min(1.0), torch.zeros_like(required)
            ).clamp(0.0, 1.0)
        readiness = (seen_count / 30.0).clamp(0.0, 1.0)
        difficulty_gap = torch.zeros_like(seen_count)
        if self.item_difficulty is not None:
            difficulty_gap = F.relu(self.item_difficulty[flat_items].view(-1, 1) - readiness)
        redundancy = torch.zeros_like(seen_count)
        if self.item_same_family is not None:
            redundancy = (self.item_same_family[flat_items].float() * seen).amax(1, keepdim=True)
        shape = (batch_size, candidate_count)
        return learner_fit(
            similarity,
            concept.view(shape),
            prerequisite_gap.view(shape),
            difficulty_gap.view(shape),
            redundancy.view(shape),
            concept_weight=self.config.concept_weight,
            prerequisite_beta=self.config.prerequisite_beta,
            difficulty_beta=self.config.difficulty_beta,
        ).detach()

    def forward(self, batch, popularity, llm_scores=None, user_bank_raw=None, user_seen_items=None):
        user_ids, item_ids = batch["u"], batch["i"]
        effective_cold = self._effective_cold(popularity, item_ids)
        base = self.encode_items(item_ids, effective_cold)
        if self.config.steps > 0:
            def candidate_provider(current, selected_history):
                candidates, candidate_ids = self._candidate_pool(current)
                fit = self._candidate_fit(current, item_ids, candidates, candidate_ids)
                return candidates, candidate_ids, fit
            candidates = candidate_ids = fit = None
        else:
            candidate_provider = None
            candidates = candidate_ids = fit = None
        output = super().forward(
            user_ids,
            item_ids,
            effective_cold,
            candidate_vectors=candidates,
            candidate_user_ids=candidate_ids,
            candidate_fit=fit,
            candidate_provider=candidate_provider,
        )
        pseudo = effective_cold & ~(popularity.to(self.device).view(-1) < float(self.cfg.cold_threshold))
        diagnostics: dict[str, Any] = {
            **output.diagnostics,
            "main_loss": float(output.loss.detach().item()),
            "base_loss": float(output.base_loss.detach().item()),
            "refinement_loss": float(output.refinement_loss.detach().item()),
            "stability_loss": float(output.stability_loss.detach().item()),
            "total_loss": float(output.loss.detach().item()),
            "aux_loss": 0.0,
            "ppo_loss": 0.0,
            "ppo_loss_raw": 0.0,
            "prereq_aux_loss": 0.0,
            "delta_reg_loss": 0.0,
            "course_sample_fit": float(output.diagnostics.get("fit_mean", 0.0)),
            "pseudo_cold_count": int(pseudo.sum().item()),
            "pseudo_cold_ratio": float(pseudo.float().mean().item()),
            "effective_cold_ratio": float(effective_cold.float().mean().item()),
            "steps": self.config.steps,
        }
        return output.loss, diagnostics

    def get_item_vector(self, item_ids, llm_scores=None, force_cold=False, disable_id_dropout=False):
        if isinstance(force_cold, torch.Tensor):
            cold = force_cold.to(self.device, dtype=torch.bool).view(-1)
        else:
            cold = torch.full((item_ids.numel(),), bool(force_cold), device=self.device)
        fused = self.encode_items(item_ids, cold)
        content = F.normalize(self.content_projection(self.item_content_embedding(item_ids)), dim=1)
        return fused, self.item_id_embedding(item_ids), content

    @torch.no_grad()
    def infer_refined_item_vectors(
        self,
        item_ids,
        llm_s=None,
        item_batch=1024,
        force_cold=True,
        item_popularity=None,
        user_seen_items=None,
        user_bank_raw=None,
        **kwargs,
    ):
        popularity = item_popularity if item_popularity is not None else self.item_popularity[item_ids]
        cold = popularity.to(self.device).view(-1) < float(self.cfg.cold_threshold)
        base = self.encode_items(item_ids, cold)
        if self.config.steps <= 0 or not cold.any():
            return base
        def candidate_provider(current, selected_history):
            candidates, candidate_ids = self._candidate_pool(current)
            fit = self._candidate_fit(current, item_ids, candidates, candidate_ids)
            return candidates, candidate_ids, fit
        return super().forward(
            torch.zeros_like(item_ids), item_ids, cold,
            candidate_provider=candidate_provider,
        ).refined_items

    def apply_course_rerank(self, scores, *args, **kwargs):
        return scores

    def content_delta_trainable_parameters(self):
        return []

    def content_delta_stats(self):
        return None

    def clip_content_delta_(self):
        return None

    def enable_delta_only_training_(self):
        return []

    def apply_sage_two_expert_score_fusion(self, scores, *args, **kwargs):
        return scores

    def _build_user_bank_raw(self):
        bank = self._user_bank()
        return bank, F.normalize(bank, dim=1)
