"""Evaluation-only A/B probe for main-table Actor inference."""

# Static runner guard tokens delegated to the recovered implementation:
# def run_static_experiment, _static_split_df

from dataclasses import asdict, dataclass, field
from collections import Counter
import hashlib
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
    split_train_rows: int = 0
    split_validation_rows: int = 0
    split_test_rows: int = 0
    course_fit_calls: int = 0
    course_fit_candidate_pairs: int = 0
    target_seen_candidate_pairs: int = 0
    course_fit_target_rows: int = 0
    target_rows_with_seen_candidate: int = 0
    history_set_calls: int = 0
    history_sources: list = field(default_factory=list)
    refined_item_ids: list = field(default_factory=list)
    behavior_target_none_calls: int = 0
    behavior_target_non_null_calls: int = 0
    course_match_exclude_target_values: list = field(default_factory=list)


@dataclass
class InferenceAuditContext:
    train_items: set = field(default_factory=set)
    validation_items: set = field(default_factory=set)
    test_items: set = field(default_factory=set)
    train_history: dict = field(default_factory=dict)
    train_plus_validation_history: dict = field(default_factory=dict)


AUDIT = InferenceAudit()
AUDIT_CONTEXT = InferenceAuditContext()
EVAL_SEED = 7001
ACTIVE_ITEM_BANK = None
INFERENCE_ROLLOUT_POLICY = "ppo"
COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE = None
POLICY_MODES = {
    "actor": "ppo",
    "ppo": "ppo",
    "greedy_similarity": "greedy_similarity",
    "course_fit": "course_fit",
    "random": "random",
}
ORIGINAL_EVALUATE = legacy.evaluate_usim
ORIGINAL_BUILD_POS = eval_core.build_eval_pos_item_vecs
ORIGINAL_GET_ACTION_VALUE = legacy.FixedSimpleAC.get_action_value
ORIGINAL_STATIC_SPLIT = legacy._static_split_df
ORIGINAL_COMPUTE_COURSE_FIT = legacy.Fast3FeedbackUSIM._compute_candidate_course_fit
ORIGINAL_SET_USER_SEEN_INDEX = legacy.Fast3FeedbackUSIM.set_user_seen_index


def history_fingerprint(user_seen_items):
    """Return an order-independent digest and cardinalities for a history map."""
    digest = hashlib.sha256()
    users = 0
    pairs = 0
    for user_id in sorted(int(uid) for uid in user_seen_items):
        items = sorted(int(item_id) for item_id in user_seen_items[user_id])
        if items:
            users += 1
        for item_id in items:
            digest.update(f"{user_id}:{item_id}\n".encode("ascii"))
            pairs += 1
    return {"sha256": digest.hexdigest(), "users": users, "pairs": pairs}


def parse_optional_bool(raw):
    """Parse an optional environment boolean without inventing a default."""
    if raw is None or str(raw).strip() == "":
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid optional boolean: {raw!r}")


def classify_history_source(user_seen_items):
    """Classify an installed history mapping against audited split histories."""
    if user_seen_items is None:
        return "cleared"
    summary = history_fingerprint(user_seen_items)
    if summary == AUDIT_CONTEXT.train_history:
        return "train_only"
    if summary == AUDIT_CONTEXT.train_plus_validation_history:
        return "train_plus_validation"
    return "unknown"


def audited_set_user_seen_index(self, user_seen_items):
    """Record the exact history source before building the dense index."""
    AUDIT.history_set_calls += 1
    AUDIT.history_sources.append(classify_history_source(user_seen_items))
    return ORIGINAL_SET_USER_SEEN_INDEX(self, user_seen_items)


def make_audited_split(split_fn):
    """Observe split/history provenance without changing returned dataframes."""
    def audited_split(df):
        train, validation, test, info = split_fn(df)
        AUDIT.split_train_rows = int(len(train))
        AUDIT.split_validation_rows = int(len(validation))
        AUDIT.split_test_rows = int(len(test))
        AUDIT_CONTEXT.train_items = set(int(x) for x in train["i_idx"].unique())
        AUDIT_CONTEXT.validation_items = set(int(x) for x in validation["i_idx"].unique())
        AUDIT_CONTEXT.test_items = set(int(x) for x in test["i_idx"].unique())

        train_seen = legacy._add_user_seen_from_df({}, train)
        train_plus_validation = legacy._clone_user_seen(train_seen)
        legacy._add_user_seen_from_df(train_plus_validation, validation)
        AUDIT_CONTEXT.train_history = history_fingerprint(train_seen)
        AUDIT_CONTEXT.train_plus_validation_history = history_fingerprint(train_plus_validation)
        return train, validation, test, info

    return audited_split


def audited_compute_candidate_course_fit(
    self,
    candidate_user_idx,
    item_idx,
    target_pop=None,
    user_seen_items=None,
):
    """Count whether candidate train histories contain their strict-cold target."""
    seen_index = getattr(self, "user_seen_index", None)
    if seen_index is not None and candidate_user_idx is not None and item_idx is not None:
        candidates = torch.as_tensor(candidate_user_idx, device=self.user_seen_index.device).long()
        targets = torch.as_tensor(item_idx, device=self.user_seen_index.device).long().view(-1)
        if candidates.ndim == 2 and candidates.size(0) == targets.numel():
            expanded_targets = targets.unsqueeze(1).expand_as(candidates)
            seen = self.user_seen_index[candidates, expanded_targets]
            AUDIT.course_fit_calls += 1
            AUDIT.course_fit_candidate_pairs += int(seen.numel())
            AUDIT.target_seen_candidate_pairs += int(seen.sum().item())
            AUDIT.course_fit_target_rows += int(seen.size(0))
            AUDIT.target_rows_with_seen_candidate += int(seen.any(dim=1).sum().item())
    return ORIGINAL_COMPUTE_COURSE_FIT(
        self,
        candidate_user_idx,
        item_idx,
        target_pop=target_pop,
        user_seen_items=user_seen_items,
    )


def install_integrity_audit():
    """Install evaluation-only provenance wrappers after target routing."""
    legacy._static_split_df = make_audited_split(legacy._static_split_df)
    legacy.Fast3FeedbackUSIM.set_user_seen_index = audited_set_user_seen_index
    legacy.Fast3FeedbackUSIM._compute_candidate_course_fit = audited_compute_candidate_course_fit


def make_validation_target_split(split_fn):
    """Route the original validation dataframe to the final evaluator slot."""
    def validation_target(df):
        train, val, test, info = split_fn(df)
        del test
        return train, val, val.copy(deep=True), info

    return validation_target


def install_evaluation_target(target):
    """Select test or validation evaluation without changing split metadata."""
    target = str(target).strip().lower()
    if target == "test":
        legacy._static_split_df = ORIGINAL_STATIC_SPLIT
    elif target == "validation":
        legacy._static_split_df = make_validation_target_split(ORIGINAL_STATIC_SPLIT)
    else:
        raise ValueError("USIM_ACTOR_EVAL_TARGET must be test or validation")


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
    global AUDIT, AUDIT_CONTEXT
    AUDIT = InferenceAudit()
    AUDIT_CONTEXT = InferenceAuditContext()


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
    previous_rollout_policy = str(getattr(self.cfg, "rollout_policy", "ppo"))
    previous_exclude_target = bool(
        getattr(self.cfg, "feedback_course_match_exclude_target", False)
    )
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
        self.cfg.rollout_policy = INFERENCE_ROLLOUT_POLICY
        if COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE is not None:
            self.cfg.feedback_course_match_exclude_target = bool(
                COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE
            )
        effective_exclude = bool(
            getattr(self.cfg, "feedback_course_match_exclude_target", False)
        )
        if effective_exclude not in AUDIT.course_match_exclude_target_values:
            AUDIT.course_match_exclude_target_values.append(effective_exclude)
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
                target_emb = None
                if target_emb is None:
                    AUDIT.behavior_target_none_calls += 1
                else:
                    AUDIT.behavior_target_non_null_calls += 1
                final, _, _ = self.run_usim_episode(
                    base,
                    target_emb=target_emb,
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
                AUDIT.refined_item_ids.extend(int(x) for x in idx.detach().cpu().tolist())
                outputs.append(refined.detach())
    finally:
        self.cfg.rollout_policy = previous_rollout_policy
        self.cfg.feedback_course_match_exclude_target = previous_exclude_target
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
    """Install static or one of the frozen-checkpoint rollout policies."""
    global INFERENCE_ROLLOUT_POLICY
    reset_audit()
    set_eval_seed(eval_seed)
    mode = str(mode).strip().lower()
    if mode != "static" and mode not in POLICY_MODES:
        raise ValueError(f"Unsupported inference policy: {mode}")

    legacy.evaluate_usim = evaluate_with_bank_targets
    eval_core.build_eval_pos_item_vecs = bank_aligned_pos_item_vecs
    legacy.FixedSimpleAC.get_action_value = ORIGINAL_GET_ACTION_VALUE

    current_refiner = getattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", None)
    if mode == "static":
        INFERENCE_ROLLOUT_POLICY = "ppo"
        if current_refiner is infer_actor_refined_item_vectors:
            delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")
        return

    INFERENCE_ROLLOUT_POLICY = POLICY_MODES[mode]
    if INFERENCE_ROLLOUT_POLICY == "ppo":
        legacy.FixedSimpleAC.get_action_value = deterministic_get_action_value
    legacy.Fast3FeedbackUSIM.infer_refined_item_vectors = infer_actor_refined_item_vectors


def write_audit(path, mode, eval_seed):
    """Write inference call counts and representation displacement statistics."""
    payload = asdict(AUDIT)
    count = max(1, int(AUDIT.refined_items))
    candidate_pairs = max(1, int(AUDIT.course_fit_candidate_pairs))
    target_rows = max(1, int(AUDIT.course_fit_target_rows))
    refined = set(int(x) for x in AUDIT.refined_item_ids)
    train_items = AUDIT_CONTEXT.train_items
    validation_items = AUDIT_CONTEXT.validation_items
    test_items = AUDIT_CONTEXT.test_items
    history_counts = dict(Counter(AUDIT.history_sources))
    payload.update(
        {
            "mode": str(mode),
            "eval_seed": int(eval_seed),
            "evaluation_target": os.environ.get("USIM_ACTOR_EVAL_TARGET", "test").strip().lower(),
            "mean_cosine": float(AUDIT.cosine_sum) / count,
            "mean_l2": float(AUDIT.l2_sum) / count,
            "history_source_counts": history_counts,
            "history_all_train_only": bool(AUDIT.history_sources)
            and all(source == "train_only" for source in AUDIT.history_sources),
            "target_seen_candidate_rate": float(AUDIT.target_seen_candidate_pairs)
            / candidate_pairs,
            "target_rows_with_seen_candidate_rate": float(
                AUDIT.target_rows_with_seen_candidate
            )
            / target_rows,
            "refined_item_composition": {
                "total_unique": len(refined),
                "train_present": len(refined & train_items),
                "validation_only": len((refined & validation_items) - test_items),
                "test_only": len((refined & test_items) - validation_items),
                "validation_and_test": len(refined & validation_items & test_items),
                "neither_validation_nor_test": len(
                    refined - validation_items - test_items
                ),
            },
            "effective_course_match_exclude_target": list(
                AUDIT.course_match_exclude_target_values
            ),
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
    global COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE
    mode = os.environ.get("USIM_ACTOR_INFERENCE_MODE", "static")
    eval_seed = int(os.environ.get("USIM_ACTOR_INFERENCE_SEED", "7001"))
    evaluation_target = os.environ.get("USIM_ACTOR_EVAL_TARGET", "test")
    checkpoint_dir = os.environ.get("USIM_FB_CKPT_DIR", "").strip()
    if not checkpoint_dir:
        raise RuntimeError("USIM_FB_CKPT_DIR is required for checkpoint replay")

    torch.save = make_read_only_torch_save(checkpoint_dir, real_save=torch.save)
    COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE = parse_optional_bool(
        os.environ.get("USIM_COURSE_MATCH_EXCLUDE_TARGET")
    )
    install_evaluation_target(evaluation_target)
    install_mode(mode, eval_seed)
    install_integrity_audit()
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
