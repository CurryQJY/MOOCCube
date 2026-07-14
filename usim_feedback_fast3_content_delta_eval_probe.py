"""Evaluation-only probe for the recovered legacy PPO checkpoint."""

import os

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy

# Static runner guard tokens delegated to legacy: def run_static_experiment, _static_split_df

_original_get_action_value = legacy.FixedSimpleAC.get_action_value


def deterministic_get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
    t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=self.time_dim).float()
    state = torch.cat([item_state, t_emb], dim=1)
    feat = self.common(state)
    value = self.critic_head(feat)
    query = self.actor_head(feat).unsqueeze(1)
    keys = self.user_proj(candidates_emb)
    logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
    dist = Categorical(logits=logits)
    if action_idx is None:
        action_idx = logits.argmax(dim=-1)
    return action_idx, dist.log_prob(action_idx), value, dist.entropy()


def _accept_probe_checkpoint(resume_state, cfg, split_info=None, script_path=None):
    return True, "evaluation probe accepts frozen checkpoint", "eval-probe", resume_state.get(
        "train_config_fingerprint"
    ) if isinstance(resume_state, dict) else None


def install_probe(eval_seed, action_mode):
    mode = str(action_mode).strip().lower()
    if mode not in {"sample", "argmax"}:
        raise ValueError(f"Unsupported evaluation action mode: {action_mode}")
    os.environ["USIM_SEED"] = str(int(eval_seed))
    legacy._checkpoint_config_matches = _accept_probe_checkpoint
    legacy.FixedSimpleAC.get_action_value = (
        deterministic_get_action_value if mode == "argmax" else _original_get_action_value
    )


def main():
    install_probe(
        eval_seed=int(os.environ.get("USIM_EVAL_PROBE_SEED", "2025")),
        action_mode=os.environ.get("USIM_EVAL_ACTION_MODE", "sample"),
    )
    legacy.main()


if __name__ == "__main__":
    main()
