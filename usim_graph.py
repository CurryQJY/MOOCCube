import os
import random

import torch
import torch.nn as nn

from course_graph import CourseGraphEncoder
from usim import Config, PAM_RL_Pure_USIM


class GraphConfig(Config):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.graph_topk_prereq = int(os.environ.get("USIM_GRAPH_TOPK_PREREQ", "5"))
        self.graph_topk_semantic = int(os.environ.get("USIM_GRAPH_TOPK_SEMANTIC", "0"))
        self.graph_hidden_dim = int(os.environ.get("USIM_GRAPH_HIDDEN_DIM", str(self.hidden_dim)))
        self.graph_prereq_weight = float(os.environ.get("USIM_GRAPH_PREREQ_WEIGHT", "1.0"))
        self.graph_semantic_weight = float(os.environ.get("USIM_GRAPH_SEMANTIC_WEIGHT", "0.8"))
        self.graph_mix_dropout = float(os.environ.get("USIM_GRAPH_MIX_DROPOUT", "0.05"))
        self.graph_residual_scale = float(os.environ.get("USIM_GRAPH_RESIDUAL_SCALE", "0.15"))
        self.graph_apply_cold_only = os.environ.get("USIM_GRAPH_APPLY_COLD_ONLY", "1") == "1"


class GraphItemMixin:
    def init_graph_modules(self, config):
        self.graph_encoder = CourseGraphEncoder(
            config.emb_dim,
            getattr(config, "graph_hidden_dim", config.hidden_dim),
            prereq_weight=getattr(config, "graph_prereq_weight", 1.0),
            semantic_weight=getattr(config, "graph_semantic_weight", 0.8),
        )
        self.graph_mix_dropout = nn.Dropout(getattr(config, "graph_mix_dropout", 0.05))
        self.graph_residual_proj = nn.Sequential(
            nn.Linear(config.emb_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emb_dim),
        )
        self.graph_residual_norm = nn.LayerNorm(config.emb_dim)
        self.item_popularity = None

    def set_course_artifacts(self, artifacts):
        super().set_course_artifacts(artifacts)
        self.graph_encoder.set_artifacts(artifacts, self.device)
        self.item_popularity = None
        if artifacts and artifacts.get("item_popularity") is not None:
            self.item_popularity = artifacts["item_popularity"].to(self.device)

    def _graph_seed_lookup(self, i_idx):
        return self.content_proj(self.item_con_emb(i_idx))

    def _get_graph_mask(self, i_idx, force_cold=False, apply_graph_mask=None):
        if apply_graph_mask is not None:
            return apply_graph_mask.float().view(-1, 1)

        if not getattr(self.cfg, "graph_apply_cold_only", True):
            return torch.ones((i_idx.size(0), 1), device=i_idx.device)

        if self.item_popularity is None:
            return torch.ones((i_idx.size(0), 1), device=i_idx.device) if force_cold else torch.zeros((i_idx.size(0), 1), device=i_idx.device)

        cold_mask = (self.item_popularity[i_idx] < float(self.cfg.cold_threshold)).float().unsqueeze(1)
        return cold_mask

    def get_item_vector(self, i_idx, llm_s, force_cold=False, apply_graph_mask=None):
        id_e = self.item_id_emb(i_idx)

        if force_cold or (self.training and random.random() < self.cfg.dropout_prob):
            id_e = torch.zeros_like(id_e)

        content_e = self.content_proj(self.item_con_emb(i_idx))

        mask_llm = (llm_s > -0.5).float().unsqueeze(1)
        val_llm = torch.clamp(llm_s, min=0.0).unsqueeze(1)
        llm_e = self.llm_proj(val_llm) * mask_llm
        content_e = content_e + llm_e

        graph_mask = self._get_graph_mask(i_idx, force_cold=force_cold, apply_graph_mask=apply_graph_mask)
        if graph_mask.max().item() > 0:
            graph_e = self.graph_encoder(i_idx, content_e, self._graph_seed_lookup)
            graph_delta = self.graph_residual_proj(
                self.graph_mix_dropout(torch.cat([content_e, graph_e - content_e, llm_e], dim=-1))
            )
            graph_content = self.graph_residual_norm(
                content_e + float(getattr(self.cfg, "graph_residual_scale", 0.15)) * graph_delta
            )
            graph_content = graph_mask * graph_content + (1.0 - graph_mask) * content_e
        else:
            graph_content = content_e

        alpha = self.gate_net(torch.cat([id_e, graph_content], dim=-1))
        item_fused = alpha * id_e + (1.0 - alpha) * graph_content
        return item_fused, id_e, graph_content


class GraphEnhancedUSIM(GraphItemMixin, PAM_RL_Pure_USIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.init_graph_modules(config)
