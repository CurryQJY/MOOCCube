"""Evaluation-only A/B probe for main-table Actor inference."""

from dataclasses import asdict, dataclass
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import fast3_delta.eval as eval_core
import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


@dataclass
class InferenceAudit:
    actor_calls: int = 0
    episode_calls: int = 0
    refined_items: int = 0
    cosine_sum: float = 0.0
    l2_sum: float = 0.0


AUDIT = InferenceAudit()
EVAL_SEED = 7001
ACTIVE_ITEM_BANK = None
ORIGINAL_EVALUATE = legacy.evaluate_usim
ORIGINAL_BUILD_POS = eval_core.build_eval_pos_item_vecs
ORIGINAL_GET_ACTION_VALUE = legacy.FixedSimpleAC.get_action_value


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


def reset_audit():
    global AUDIT
    AUDIT = InferenceAudit()


def set_eval_seed(seed):
    global EVAL_SEED
    EVAL_SEED = int(seed)
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_SEED)


def infer_actor_refined_item_vectors(
    self,
    item_idx,
    llm_s=None,
    item_batch=1024,
    force_cold=True,
    user_bank_raw=None,
    user_seen_items=None,
):
    """Refine cold item vectors with a deterministic Actor rollout."""
    set_eval_seed(EVAL_SEED)
    item_idx = torch.as_tensor(item_idx, dtype=torch.long, device=self.device).view(-1)
    if item_idx.numel() == 0:
        return torch.empty(
            (0, self.cfg.emb_dim),
            dtype=self.item_id_emb.weight.dtype,
            device=self.device,
        )
    if llm_s is None:
        llm_s = torch.full((item_idx.numel(),), -1.0, dtype=torch.float32, device=self.device)
    else:
        llm_s = torch.as_tensor(llm_s, dtype=torch.float32, device=self.device).view(-1)
        if llm_s.numel() != item_idx.numel():
            raise ValueError("llm_s must have the same length as item_idx")

    was_training = self.training
    self.eval()
    outputs = []
    bank = user_bank_raw if user_bank_raw is not None else self._build_user_bank_raw()
    history_context = user_seen_items
    if history_context is None and getattr(self, "user_seen_index", None) is not None:
        # The model's dense user_seen_index is the actual data source.  A non-None
        # sentinel enables the existing course-fit helpers' fast path.
        history_context = {}
    batch_size = max(1, int(item_batch))
    try:
        with torch.no_grad():
            for start in range(0, item_idx.numel(), batch_size):
                end = min(start + batch_size, item_idx.numel())
                idx = item_idx[start:end]
                score = llm_s[start:end]
                base, _, _ = self.get_item_vector(
                    idx,
                    score,
                    force_cold=force_cold,
                    disable_id_dropout=True,
                )
                pop = None
                if self.item_popularity is not None:
                    pop = self.item_popularity.to(self.device).index_select(0, idx).float()
                final, _, _ = self.run_usim_episode(
                    base,
                    target_emb=None,
                    user_bank_raw=bank,
                    item_idx=idx,
                    target_pop=pop,
                    user_seen_items=history_context,
                )
                refined = self._blend_rl_episode_output(base, final)
                cosine = F.cosine_similarity(base, refined, dim=1)
                l2 = torch.linalg.vector_norm(refined - base, dim=1)
                AUDIT.episode_calls += 1
                AUDIT.refined_items += int(idx.numel())
                AUDIT.cosine_sum += float(cosine.sum().item())
                AUDIT.l2_sum += float(l2.sum().item())
                outputs.append(refined.detach())
    finally:
        self.train(was_training)
    return torch.cat(outputs, dim=0)


def bank_aligned_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    """Use the exact cached item-bank vector when restoring a target score."""
    if ACTIVE_ITEM_BANK is not None:
        return ACTIVE_ITEM_BANK.index_select(0, item_idx)
    return ORIGINAL_BUILD_POS(model, item_idx, llm_s, pop_sel, eval_type)


def evaluate_with_bank_targets(*args, **kwargs):
    """Run the existing evaluator while aligning positive and catalog vectors."""
    global ACTIVE_ITEM_BANK
    previous = ACTIVE_ITEM_BANK
    banks = kwargs.get("all_item_vecs")
    eval_type = kwargs.get("eval_type", "cold")
    ACTIVE_ITEM_BANK = eval_core.select_eval_item_bank(banks, eval_type) if banks is not None else None
    try:
        return ORIGINAL_EVALUATE(*args, **kwargs)
    finally:
        ACTIVE_ITEM_BANK = previous


def install_mode(mode, eval_seed):
    """Install either static control or deterministic Actor inference."""
    reset_audit()
    set_eval_seed(eval_seed)
    mode = str(mode).strip().lower()
    if mode not in {"static", "actor"}:
        raise ValueError("USIM_ACTOR_INFERENCE_MODE must be static or actor")

    legacy.evaluate_usim = evaluate_with_bank_targets
    eval_core.build_eval_pos_item_vecs = bank_aligned_pos_item_vecs
    legacy.FixedSimpleAC.get_action_value = ORIGINAL_GET_ACTION_VALUE

    current_refiner = getattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", None)
    if mode == "static":
        if current_refiner is infer_actor_refined_item_vectors:
            delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")
        return

    legacy.FixedSimpleAC.get_action_value = deterministic_get_action_value
    legacy.Fast3FeedbackUSIM.infer_refined_item_vectors = infer_actor_refined_item_vectors


def write_audit(path, mode, eval_seed):
    """Write inference call counts and representation displacement statistics."""
    payload = asdict(AUDIT)
    count = max(1, int(AUDIT.refined_items))
    payload.update(
        {
            "mode": str(mode),
            "eval_seed": int(eval_seed),
            "mean_cosine": float(AUDIT.cosine_sum) / count,
            "mean_l2": float(AUDIT.l2_sum) / count,
        }
    )
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_read_only_torch_save(checkpoint_dir, real_save):
    """Block writes inside the source checkpoint tree while allowing other saves."""
    root = os.path.normcase(os.path.abspath(os.fspath(checkpoint_dir)))

    def guarded_save(obj, path, *args, **kwargs):
        try:
            target = os.path.normcase(os.path.abspath(os.fspath(path)))
        except TypeError:
            return real_save(obj, path, *args, **kwargs)
        try:
            inside = os.path.commonpath([root, target]) == root
        except ValueError:
            inside = False
        if inside:
            print(f">> EVAL READ-ONLY: blocked checkpoint write {target}")
            return None
        return real_save(obj, path, *args, **kwargs)

    return guarded_save


def main():
    mode = os.environ.get("USIM_ACTOR_INFERENCE_MODE", "static")
    eval_seed = int(os.environ.get("USIM_ACTOR_INFERENCE_SEED", "7001"))
    checkpoint_dir = os.environ.get("USIM_FB_CKPT_DIR", "").strip()
    if not checkpoint_dir:
        raise RuntimeError("USIM_FB_CKPT_DIR is required for checkpoint replay")

    torch.save = make_read_only_torch_save(checkpoint_dir, real_save=torch.save)
    install_mode(mode, eval_seed)
    legacy.main()

    output_path = legacy._feedback_output_path("actor_inference_audit.json")
    write_audit(output_path, mode=mode, eval_seed=eval_seed)
    print(
        ">> ACTOR INFERENCE AUDIT: "
        f"mode={mode} actor_calls={AUDIT.actor_calls} "
        f"episode_calls={AUDIT.episode_calls} refined_items={AUDIT.refined_items}"
    )


if __name__ == "__main__":
    main()
