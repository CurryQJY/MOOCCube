"""Evaluation-only A/B probe for main-table Actor inference."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class InferenceAudit:
    actor_calls: int = 0
    episode_calls: int = 0
    refined_items: int = 0
    cosine_sum: float = 0.0
    l2_sum: float = 0.0


AUDIT = InferenceAudit()


def deterministic_get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
    """Select the highest-logit Actor action while retaining PPO diagnostics."""
    AUDIT.actor_calls += 1
    t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=self.time_dim).float()
    feat = self.common(torch.cat([item_state, t_emb], dim=1))
    value = self.critic_head(feat)
    query = self.actor_head(feat).unsqueeze(1)
    keys = self.user_proj(candidates_emb)
    logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
    dist = Categorical(logits=logits)
    if action_idx is None:
        action_idx = logits.argmax(dim=-1)
    return action_idx, dist.log_prob(action_idx), value, dist.entropy()
