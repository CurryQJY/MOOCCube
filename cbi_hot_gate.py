"""TDInit soft-anchor simulator with an explicit Hot-only ID/content gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cbi_anchor_sim import CBIAnchorFast3FeedbackUSIM


class CBIHotGateFast3FeedbackUSIM(CBIAnchorFast3FeedbackUSIM):
    """Keep Cold content-only while restoring ID/content fusion for Hot items."""

    def get_item_vector(self, i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        id_e_true = self.item_id_emb(i_idx)
        batch_size = id_e_true.size(0)
        device = id_e_true.device

        if isinstance(force_cold, torch.Tensor):
            cold_mask = force_cold.to(device=device)
            if cold_mask.dtype != torch.bool:
                cold_mask = cold_mask > 0
            cold_mask = cold_mask.view(-1)
        else:
            cold_mask = torch.full(
                (batch_size,), bool(force_cold), dtype=torch.bool, device=device
            )

        mask_id = cold_mask.view(-1, 1).clone()
        if self.training and self.cfg.dropout_prob > 0 and not disable_id_dropout:
            dropout_mask = torch.rand((batch_size, 1), device=device) < float(
                self.cfg.dropout_prob
            )
            mask_id = mask_id | dropout_mask
        id_e = torch.where(mask_id, torch.zeros_like(id_e_true), id_e_true)

        content_base_e = self._content_base_embedding(i_idx)
        delta_force_cold = force_cold
        if self.training and getattr(self.cfg, "content_delta_train_on_id_dropout", False):
            delta_force_cold = mask_id.view(-1)
        content_e = self._apply_content_delta(
            content_base_e, i_idx, force_cold=delta_force_cold
        )

        llm_weight = float(getattr(self.cfg, "llm_weight", 0.0))
        if not getattr(self.cfg, "disable_llm_score", False) and llm_weight > 0.0:
            mask_llm = (llm_s > -0.5).float().unsqueeze(1)
            if getattr(self.cfg, "llm_cold_only", False) or getattr(
                self.cfg, "llm_hot_only", False
            ):
                cold_float = cold_mask.float().view(-1, 1)
                mask_llm = mask_llm * (
                    1.0 - cold_float
                    if getattr(self.cfg, "llm_hot_only", False)
                    else cold_float
                )
            llm_e = self.llm_proj(torch.clamp(llm_s, min=0.0).unsqueeze(1))
            content_e = content_e + llm_weight * llm_e * mask_llm

        gate = self.gate_net(torch.cat([id_e, content_e], dim=-1))
        hot_fused = F.normalize(gate * id_e + (1.0 - gate) * content_e, dim=1)
        item_fused = torch.where(cold_mask.view(-1, 1), content_e, hot_fused)

        aux_mode = str(getattr(self.cfg, "content_delta_aux_mode", "base")).strip().lower()
        aux_content_e = content_base_e if aux_mode in {"base", "raw", "no_delta"} else content_e
        return item_fused, id_e_true, aux_content_e
