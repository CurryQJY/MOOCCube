from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import LIRAConfig
from .refinement import bounded_refinement, dynamic_bounded_refinement


@dataclass
class LIRAOutput:
    loss: torch.Tensor
    base_loss: torch.Tensor
    refinement_loss: torch.Tensor
    stability_loss: torch.Tensor
    logits: torch.Tensor
    base_items: torch.Tensor
    refined_items: torch.Tensor
    diagnostics: dict[str, Any]


class LIRAModel(nn.Module):
    def __init__(self, config: LIRAConfig, content_embeddings: torch.Tensor):
        super().__init__()
        if content_embeddings.shape != (config.n_items, config.content_dim):
            raise ValueError("content embedding shape does not match configuration")
        self.config = config
        self.user_embedding = nn.Embedding(config.n_users, config.embedding_dim)
        self.item_id_embedding = nn.Embedding(config.n_items, config.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)
        self.item_content_embedding = nn.Embedding.from_pretrained(content_embeddings.float(), freeze=True)
        self.content_projection = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.user_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * config.embedding_dim, config.embedding_dim),
            nn.Sigmoid(),
        )

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_projection(self.user_embedding(user_ids))

    def encode_items(self, item_ids: torch.Tensor, cold_mask: torch.Tensor) -> torch.Tensor:
        item_id = self.item_id_embedding(item_ids)
        cold = cold_mask.to(device=item_ids.device, dtype=torch.bool).view(-1, 1)
        visible_id = torch.where(cold, torch.zeros_like(item_id), item_id)
        content = F.normalize(
            self.content_projection(self.item_content_embedding(item_ids)), dim=1
        )
        gate = self.fusion_gate(torch.cat([visible_id, content], dim=1))
        return gate * visible_id + (1.0 - gate) * content

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        effective_cold: torch.Tensor,
        *,
        candidate_vectors: torch.Tensor | None = None,
        candidate_user_ids: torch.Tensor | None = None,
        candidate_fit: torch.Tensor | None = None,
        candidate_provider=None,
    ) -> LIRAOutput:
        users = self.encode_users(user_ids)
        base_items = self.encode_items(item_ids, effective_cold)
        if self.config.steps == 0:
            refined = base_items
            diagnostics = {"repeated_user_rate": 0.0, "update_active_ratio": 0.0}
        else:
            if candidate_provider is not None:
                refined, diagnostics = dynamic_bounded_refinement(
                    base_items,
                    effective_cold,
                    candidate_provider,
                    steps=self.config.steps,
                    update_lr=self.config.update_lr,
                    min_fit=self.config.min_fit,
                    min_gain=self.config.min_gain,
                    step_cap=self.config.step_cap,
                    total_cap=self.config.total_cap,
                )
            elif candidate_vectors is None or candidate_user_ids is None or candidate_fit is None:
                raise ValueError("refinement candidates are required when steps > 0")
            else:
                refined, diagnostics = bounded_refinement(
                    base_items,
                    candidate_vectors,
                    candidate_user_ids,
                    candidate_fit,
                    effective_cold,
                    steps=self.config.steps,
                    update_lr=self.config.update_lr,
                    min_fit=self.config.min_fit,
                    step_cap=self.config.step_cap,
                    total_cap=self.config.total_cap,
                )
        base_logits = F.normalize(users, dim=1) @ F.normalize(base_items, dim=1).t()
        base_logits = base_logits / self.config.temperature
        logits = F.normalize(users, dim=1) @ F.normalize(refined, dim=1).t()
        logits = logits / self.config.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        base_margin_logits = base_logits.clone()
        base_margin_logits[labels, labels] -= self.config.margin / self.config.temperature
        refined_margin_logits = logits.clone()
        refined_margin_logits[labels, labels] -= self.config.margin / self.config.temperature
        if logits.size(0) > 1:
            negative_count = min(200, logits.size(0) - 1)
            negative_logits = base_margin_logits.masked_fill(
                torch.eye(logits.size(0), dtype=torch.bool, device=logits.device),
                float("-inf"),
            )
            hard_negative = torch.topk(negative_logits, k=negative_count, dim=1).values
            sampled_logits = torch.cat(
                [base_margin_logits.diagonal().view(-1, 1), hard_negative], dim=1
            )
            base_loss = F.cross_entropy(
                sampled_logits,
                torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
            )
        else:
            base_loss = base_margin_logits.sum() * 0.0
        cold_rows = effective_cold.to(logits.device, dtype=torch.bool).view(-1)
        if self.config.steps > 0 and cold_rows.any():
            cold_logits = refined_margin_logits[cold_rows]
            cold_targets = labels[cold_rows]
            refinement_loss = F.cross_entropy(cold_logits, cold_targets)
        else:
            refinement_loss = base_loss * 0.0
        stability_loss = (refined - base_items).pow(2).sum(dim=1)
        stability_loss = stability_loss[cold_rows].mean() if cold_rows.any() else base_loss * 0.0
        loss = (
            base_loss
            + self.config.refinement_loss_weight * refinement_loss
            + self.config.stability_loss_weight * stability_loss
        )
        return LIRAOutput(
            loss,
            base_loss,
            refinement_loss,
            stability_loss,
            logits,
            base_items,
            refined,
            diagnostics,
        )
