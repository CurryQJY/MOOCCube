import copy
import json
import math
import os
import time
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from usim import (
    Config as BaseConfig,
    PAM_RL_Pure_USIM,
    StreamDataset,
    _add_user_seen_from_df,
    _clone_user_seen,
    build_eval_item_vecs,
    build_course_artifacts,
    collate_fn,
    evaluate_usim,
    setup_seed,
    split_dataframe_by_periods,
)


class FeedbackConfig(BaseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_load_course_artifacts = os.environ.get("USIM_FB_LOAD_COURSE_ARTIFACTS", "1") == "1"
        self.reward_terminal_weight = float(os.environ.get("USIM_FB_REWARD_TERM_W", "10.0"))
        self.reward_gain_weight = float(os.environ.get("USIM_FB_REWARD_GAIN_W", "5.0"))
        self.reward_gain_clip = float(os.environ.get("USIM_FB_REWARD_GAIN_CLIP", "0.05"))
        self.reward_dup_penalty_weight = float(os.environ.get("USIM_FB_REWARD_DUP_W", "0.50"))
        self.reward_cov_bonus_weight = float(os.environ.get("USIM_FB_REWARD_COV_W", "0.00"))
        self.feedback_course_only_cold = os.environ.get("USIM_FB_COURSE_ONLY_COLD", "1") == "1"
        self.feedback_course_warm_seen = int(os.environ.get("USIM_FB_COURSE_WARM_SEEN", "5"))
        self.feedback_course_concept_min = float(os.environ.get("USIM_FB_COURSE_CONCEPT_MIN", "0.12"))
        self.feedback_course_redundant_thr = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_THR", "0.70"))
        self.feedback_course_prereq_gate = float(os.environ.get("USIM_FB_COURSE_PREREQ_GATE", "0.20"))
        self.feedback_course_prereq_weight = float(os.environ.get("USIM_FB_COURSE_PREREQ_W", "0.08"))
        self.feedback_course_concept_weight = float(os.environ.get("USIM_FB_COURSE_CONCEPT_W", "0.04"))
        self.feedback_course_difficulty_weight = float(os.environ.get("USIM_FB_COURSE_DIFF_W", "0.03"))
        self.feedback_course_redundant_weight = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_W", "0.02"))
        self.feedback_course_sample_beta = float(os.environ.get("USIM_FB_COURSE_SAMPLE_BETA", "0.20"))
        self.feedback_course_sample_only_cold = os.environ.get("USIM_FB_COURSE_SAMPLE_ONLY_COLD", "1") == "1"
        self.feedback_course_sample_topk = int(os.environ.get("USIM_FB_COURSE_SAMPLE_TOPK", "32"))
        self.feedback_course_sample_top_l = int(
            os.environ.get("USIM_FB_COURSE_SAMPLE_TOPL", str(self.feedback_course_sample_topk))
        )
        self.use_prereq_aux_loss = os.environ.get(
            "USIM_USE_PREREQ_AUX_LOSS",
            "1" if self.use_prereq_aux_loss else "0",
        ) == "1"
        self.prereq_aux_weight = float(os.environ.get("USIM_PREREQ_AUX_WEIGHT", str(self.prereq_aux_weight)))
        self.prereq_aux_margin = float(os.environ.get("USIM_PREREQ_AUX_MARGIN", str(self.prereq_aux_margin)))
        self.prereq_aux_violation_thr = float(
            os.environ.get("USIM_PREREQ_AUX_VIOLATION_THR", str(self.prereq_aux_violation_thr))
        )
        self.prereq_aux_min_seen = int(os.environ.get("USIM_PREREQ_AUX_MIN_SEEN", str(self.prereq_aux_min_seen)))
        self.prereq_aux_only_cold = os.environ.get(
            "USIM_PREREQ_AUX_ONLY_COLD",
            "1" if self.prereq_aux_only_cold else "0",
        ) == "1"
        self.use_course_rerank = os.environ.get(
            "USIM_USE_COURSE_RERANK",
            "1" if self.use_course_rerank else "0",
        ) == "1"
        self.rerank_alpha = float(os.environ.get("USIM_COURSE_RERANK_ALPHA", str(self.rerank_alpha)))
        self.rerank_lambda = float(os.environ.get("USIM_COURSE_RERANK_LAMBDA", str(self.rerank_lambda)))
        self.rerank_min_seen = int(os.environ.get("USIM_COURSE_RERANK_MIN_SEEN", str(self.rerank_min_seen)))
        self.rerank_top_l = int(os.environ.get("USIM_COURSE_RERANK_TOPL", str(self.rerank_top_l)))
        self.rerank_penalty_cap = float(
            os.environ.get("USIM_COURSE_RERANK_PENALTY_CAP", str(self.rerank_penalty_cap))
        )
        self.rerank_only_cold = os.environ.get(
            "USIM_COURSE_RERANK_ONLY_COLD",
            "1" if self.rerank_only_cold else "0",
        ) == "1"
        self.use_structured_hard_neg = os.environ.get(
            "USIM_USE_STRUCTURED_HARD_NEG",
            "1" if self.use_structured_hard_neg else "0",
        ) == "1"
        self.train_log_interval = int(os.environ.get("USIM_FB_TRAIN_LOG_INTERVAL", "25"))
        self.train_log_first = int(os.environ.get("USIM_FB_TRAIN_LOG_FIRST", "1"))
        self.train_log_time_sec = float(os.environ.get("USIM_FB_TRAIN_LOG_TIME_SEC", "60"))
        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", str(self.ppo_epochs)))
        self.ppo_lambda = float(os.environ.get("USIM_PPO_LAMBDA", "0.95"))
        self.ppo_value_clip = float(os.environ.get("USIM_PPO_VALUE_CLIP", "0.20"))
        self.ppo_adv_norm = os.environ.get("USIM_PPO_ADV_NORM", "1") == "1"
        self.ablation_fast3_target_mix = os.environ.get("USIM_ABL_FAST3_TARGET_MIX", "0") == "1"
        self.ablation_fast3_soft_sampling = os.environ.get("USIM_ABL_FAST3_SOFT_SAMPLING", "0") == "1"
        self.ablation_fast3_ppo = os.environ.get("USIM_ABL_FAST3_PPO", "0") == "1"
        self.ablation_train_window = int(os.environ.get("USIM_ABL_TRAIN_WINDOW", "0"))
        self.fast3_target_alpha_cold = float(os.environ.get("USIM_FAST3_TGT_ALPHA_COLD", "0.35"))
        self.fast3_target_alpha_hot = float(os.environ.get("USIM_FAST3_TGT_ALPHA_HOT", "0.60"))
        self.fast3_target_alpha_step = float(os.environ.get("USIM_FAST3_TGT_ALPHA_STEP", "0.20"))
        self.fast3_target_alpha_entropy = float(os.environ.get("USIM_FAST3_TGT_ALPHA_ENT", "0.20"))
        self.fast3_target_alpha_min = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MIN", "0.15"))
        self.fast3_target_alpha_max = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MAX", "0.85"))


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _console_compact():
    return os.environ.get("USIM_FB_CONSOLE_COMPACT", "1") == "1"


def _console_print(*args, force=False, **kwargs):
    if force or not _console_compact():
        print(*args, **kwargs)


def _should_log_train_progress(batch_idx, num_batches, cfg, last_log_ts, now_ts):
    step = batch_idx + 1
    if step <= max(0, int(cfg.train_log_first)):
        return True
    interval = max(1, int(cfg.train_log_interval))
    if step % interval == 0:
        return True
    if step >= num_batches:
        return True
    if (now_ts - last_log_ts) >= float(cfg.train_log_time_sec):
        return True
    return False


def _feedback_ckpt_dir():
    explicit = os.environ.get("USIM_FB_CKPT_DIR")
    if explicit:
        return explicit
    tag = os.environ.get("USIM_FB_OUTPUT_TAG", "").strip()
    base = os.path.join("checkpoints", "usim_feedback")
    return os.path.join(base, tag) if tag else base


def _feedback_ckpt_enabled():
    return os.environ.get("USIM_FB_SAVE_CKPT", "1") == "1"


def _feedback_ckpt_auto_resume():
    return os.environ.get("USIM_FB_AUTO_RESUME", "1") == "1"


def _feedback_ckpt_force_fresh():
    return os.environ.get("USIM_FB_FORCE_FRESH", "0") == "1"


def _serialize_user_seen_items(user_seen_items):
    return {
        int(uid): sorted(int(it) for it in items)
        for uid, items in user_seen_items.items()
    }


def _deserialize_user_seen_items(payload):
    if not payload:
        return {}
    return {
        int(uid): set(int(it) for it in items)
        for uid, items in payload.items()
    }


def _latest_feedback_ckpt_path(ckpt_dir):
    return os.path.join(ckpt_dir, "latest.pt")


def _save_feedback_checkpoint(ckpt_dir, state, snapshot_name=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    tmp_path = latest_path + ".tmp"
    state = copy.deepcopy(state)
    state["saved_at"] = time.time()
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    if snapshot_name:
        snapshot_path = os.path.join(ckpt_dir, snapshot_name)
        torch.save(state, snapshot_path)
    return latest_path


def _load_feedback_checkpoint(ckpt_dir):
    latest_path = _latest_feedback_ckpt_path(ckpt_dir)
    if not os.path.exists(latest_path):
        return None
    return torch.load(latest_path, map_location="cpu")


def _move_state_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _maybe_clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_feedback_ckpt_state(
    model,
    optimizer,
    history,
    accum_cold,
    accum_hot,
    count_cold,
    count_hot,
    full_cold,
    full_hot,
    fc_cold,
    fc_hot,
    user_seen_items,
    accumulated_periods,
    warmup_periods,
    total_periods,
    status,
    next_period,
    current_period=None,
    next_epoch=0,
    es_best=None,
    es_best_state=None,
    es_best_opt_state=None,
    es_no_improve=0,
):
    return {
        "version": 1,
        "status": status,
        "next_period": int(next_period),
        "current_period": None if current_period is None else int(current_period),
        "next_epoch": int(next_epoch),
        "accumulated_periods": int(accumulated_periods),
        "warmup_periods": int(warmup_periods),
        "total_periods": int(total_periods),
        "history": copy.deepcopy(history),
        "accum_cold": copy.deepcopy(accum_cold),
        "accum_hot": copy.deepcopy(accum_hot),
        "count_cold": int(count_cold),
        "count_hot": int(count_hot),
        "full_cold": copy.deepcopy(full_cold),
        "full_hot": copy.deepcopy(full_hot),
        "fc_cold": int(fc_cold),
        "fc_hot": int(fc_hot),
        "user_seen_items": _serialize_user_seen_items(user_seen_items),
        "model_state": _move_state_to_cpu(model.state_dict()),
        "optimizer_state": _move_state_to_cpu(optimizer.state_dict()),
        "es_best": copy.deepcopy(es_best),
        "es_best_state": _move_state_to_cpu(es_best_state),
        "es_best_opt_state": _move_state_to_cpu(es_best_opt_state),
        "es_no_improve": int(es_no_improve),
    }


def _feedback_output_dir():
    explicit = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    tag = os.environ.get("USIM_FB_OUTPUT_TAG", "").strip()
    if tag:
        path = os.path.join("outputs", "usim_feedback", tag)
        os.makedirs(path, exist_ok=True)
        return path
    return "."


def _feedback_output_path(filename):
    return os.path.join(_feedback_output_dir(), filename)


def _save_final_report_exports(
    protocol,
    metrics_keys,
    sampled_cold,
    sampled_hot,
    full_cold,
    full_hot,
    sampled_cold_count,
    sampled_hot_count,
    full_cold_count,
    full_hot_count,
    model_name="USIM-Feedback",
):
    suffix = "" if protocol == "stream" else f"_{protocol}"
    detail_path = _feedback_output_path(f"final_report_usim_feedback{suffix}.csv")
    fullrank_path = _feedback_output_path(f"final_fullrank_usim_feedback{suffix}.csv")

    detail_rows = []
    for key in metrics_keys:
        detail_rows.append(
            {
                "metric": key,
                "sampled_cold": float(sampled_cold.get(key, 0.0)),
                "sampled_hot": float(sampled_hot.get(key, 0.0)),
                "full_cold": float(full_cold.get(key, 0.0)),
                "full_hot": float(full_hot.get(key, 0.0)),
            }
        )

    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    fullrank_row = {
        "model": model_name,
        "protocol": protocol,
        "full_cold_r5": float(full_cold.get("R@5", 0.0)),
        "full_cold_r10": float(full_cold.get("R@10", 0.0)),
        "full_cold_r20": float(full_cold.get("R@20", 0.0)),
        "full_cold_n5": float(full_cold.get("N@5", 0.0)),
        "full_cold_n10": float(full_cold.get("N@10", 0.0)),
        "full_cold_n20": float(full_cold.get("N@20", 0.0)),
        "full_hot_r5": float(full_hot.get("R@5", 0.0)),
        "full_hot_r10": float(full_hot.get("R@10", 0.0)),
        "full_hot_r20": float(full_hot.get("R@20", 0.0)),
        "full_hot_n5": float(full_hot.get("N@5", 0.0)),
        "full_hot_n10": float(full_hot.get("N@10", 0.0)),
        "full_hot_n20": float(full_hot.get("N@20", 0.0)),
        "sampled_cold_count": int(sampled_cold_count),
        "sampled_hot_count": int(sampled_hot_count),
        "full_cold_count": int(full_cold_count),
        "full_hot_count": int(full_hot_count),
        "notes": f"auto-exported from {model_name} ({protocol})",
    }
    pd.DataFrame([fullrank_row]).to_csv(fullrank_path, index=False)
    return detail_path, fullrank_path


def _empty_course_stats(n_items):
    return {
        "items_with_concept": 0,
        "items_with_prereq": 0,
        "hard_density": 0.0,
        "prereq_edges_kept": 0,
        "prereq_edges_raw": 0,
        "prereq_users": 0,
        "n_items": int(n_items),
    }


class FeedbackAwareUSIM(PAM_RL_Pure_USIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.item_popularity = None
        self.item_difficulty = None
        self._seen_index_cache = OrderedDict()
        self._seen_cache_owner = None
        self._seen_cache_max = int(os.environ.get("USIM_FB_SEEN_CACHE_MAX", "20000"))

    def clear_seen_cache(self):
        self._seen_index_cache.clear()
        self._seen_cache_owner = None

    def set_feedback_item_stats(self, item_popularity):
        if item_popularity is None:
            self.item_popularity = None
            self.item_difficulty = None
            return

        pop = torch.as_tensor(item_popularity, dtype=torch.float32, device=self.device)
        if pop.numel() != self.cfg.n_items:
            raise ValueError(f"item_popularity size mismatch: expect {self.cfg.n_items}, got {pop.numel()}")

        max_log = torch.log1p(pop.max()).clamp_min(1.0)
        difficulty = 1.0 - torch.log1p(pop) / max_log
        self.item_popularity = pop
        self.item_difficulty = difficulty.clamp(0.0, 1.0)

    def _compute_target_alpha(self, target_pop, step_idx, entropy, num_candidates, batch_size):
        if target_pop is not None:
            cold_mask = (target_pop.view(-1, 1).to(self.device) < float(self.cfg.cold_threshold)).float()
        else:
            cold_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        alpha = (
            cold_mask * float(self.cfg.fast3_target_alpha_cold)
            + (1.0 - cold_mask) * float(self.cfg.fast3_target_alpha_hot)
        )

        if self.cfg.usim_steps > 1:
            progress = float(step_idx) / float(max(1, self.cfg.usim_steps - 1))
            alpha = alpha + float(self.cfg.fast3_target_alpha_step) * progress

        if entropy is not None and num_candidates > 1:
            max_entropy = max(1e-6, math.log(float(num_candidates)))
            entropy_norm = (entropy.detach().unsqueeze(1) / max_entropy).clamp(0.0, 1.0)
            alpha = alpha - float(self.cfg.fast3_target_alpha_entropy) * entropy_norm

        return alpha.clamp(
            min=float(self.cfg.fast3_target_alpha_min),
            max=float(self.cfg.fast3_target_alpha_max),
        )

    def _get_seen_idx_tensor(self, uid, user_seen_items):
        uid = int(uid)
        cached = self._seen_index_cache.get(uid)
        if cached is not None:
            self._seen_index_cache.move_to_end(uid)
            return cached

        seen_items = user_seen_items.get(uid) if user_seen_items is not None else None
        if not seen_items:
            seen_idx = torch.empty(0, dtype=torch.long)
        else:
            seen_list = [it for it in seen_items if 0 <= it < self.cfg.n_items]
            seen_idx = torch.tensor(seen_list, dtype=torch.long) if seen_list else torch.empty(0, dtype=torch.long)

        self._seen_index_cache[uid] = seen_idx
        if len(self._seen_index_cache) > max(1, self._seen_cache_max):
            self._seen_index_cache.popitem(last=False)
        return seen_idx

    def _build_seen_mat(self, user_ids, user_seen_items):
        if isinstance(user_ids, torch.Tensor):
            user_ids = [int(x) for x in user_ids.detach().cpu().tolist()]
        else:
            user_ids = [int(x) for x in user_ids]

        batch_size = len(user_ids)
        seen_mat = torch.zeros((batch_size, self.cfg.n_items), dtype=torch.float32, device=self.device)
        if user_seen_items is None:
            return seen_mat, seen_mat.sum(dim=1, keepdim=True)
        if self._seen_cache_owner is not user_seen_items:
            self.clear_seen_cache()
            self._seen_cache_owner = user_seen_items

        uid_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        unique_uids, inverse = torch.unique(uid_tensor, sorted=False, return_inverse=True)
        unique_rows = torch.zeros((unique_uids.numel(), self.cfg.n_items), dtype=torch.float32, device=self.device)
        unique_counts = torch.zeros((unique_uids.numel(), 1), dtype=torch.float32, device=self.device)

        for row, uid in enumerate(unique_uids.detach().cpu().tolist()):
            seen_idx_cpu = self._get_seen_idx_tensor(uid, user_seen_items)
            if seen_idx_cpu.numel() < 1:
                continue
            seen_idx = seen_idx_cpu.to(device=self.device)
            unique_rows[row, seen_idx] = 1.0
            unique_counts[row, 0] = float(seen_idx.numel())

        seen_mat = unique_rows[inverse]
        seen_cnt = unique_counts[inverse]
        return seen_mat, seen_cnt

    def _compute_course_profile(
        self,
        user_ids,
        item_idx,
        user_seen_items=None,
        target_pop=None,
        only_cold=False,
    ):
        if user_ids is None or user_seen_items is None:
            return None

        if not torch.is_tensor(item_idx):
            item_idx = torch.tensor(item_idx, dtype=torch.long, device=self.device)
        else:
            item_idx = item_idx.to(self.device)

        batch_size = int(item_idx.numel())
        zero = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        seen_mat, seen_cnt_raw = self._build_seen_mat(user_ids, user_seen_items)
        if seen_cnt_raw.max().item() < 1:
            return {
                "active": zero,
                "prereq_gap": zero,
                "concept_bonus": zero,
                "difficulty_gap": zero,
                "redundant": zero,
            }

        active = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        if only_cold and target_pop is not None:
            active = (target_pop.view(-1, 1).to(self.device) < float(self.cfg.cold_threshold)).float()

        batch_idx = torch.arange(batch_size, device=self.device)
        seen_active = (seen_cnt_raw >= 1.0).float()
        warm_seen = max(1.0, float(self.cfg.feedback_course_warm_seen))
        user_readiness = (seen_cnt_raw / warm_seen).clamp(0.0, 1.0)

        prereq_gate = float(min(1.0, max(0.0, self.cfg.feedback_course_prereq_gate)))
        prereq_gap = zero.clone()
        prereq_safe = torch.ones_like(zero)
        if self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None:
            prereq_seen = torch.matmul(seen_mat, self.item_prereq_item_mat.t())
            prereq_cnt = self.item_prereq_item_cnt.unsqueeze(0)
            violation_full = torch.where(
                prereq_cnt > 0,
                1.0 - prereq_seen / prereq_cnt.clamp_min(1.0),
                torch.zeros_like(prereq_seen),
            ).clamp(0.0, 1.0)
            prereq_gap = violation_full[batch_idx, item_idx].unsqueeze(1)
            prereq_safe = (prereq_gap <= prereq_gate).float()

        concept_bonus = zero.clone()
        redundant = zero.clone()
        if self.item_concept_overlap is not None:
            concept_full = torch.matmul(seen_mat, self.item_concept_overlap.t()) / seen_cnt_raw.clamp_min(1.0)
            concept_match = concept_full[batch_idx, item_idx].unsqueeze(1).clamp(0.0, 1.0)
            redundant_thr = float(min(0.99, max(0.0, self.cfg.feedback_course_redundant_thr)))
            concept_min = float(min(redundant_thr - 1e-3, max(0.0, self.cfg.feedback_course_concept_min)))
            concept_band = max(1e-6, redundant_thr - concept_min)
            concept_bonus = ((concept_match - concept_min) / concept_band).clamp(0.0, 1.0)
            redundant = ((concept_match - redundant_thr) / max(1e-6, 1.0 - redundant_thr)).clamp(0.0, 1.0)
            concept_bonus = concept_bonus * prereq_safe * seen_active * (1.0 - redundant)

        difficulty_gap = zero.clone()
        if self.item_difficulty is not None:
            item_difficulty = self.item_difficulty[item_idx].unsqueeze(1)
            difficulty_gap = F.relu(item_difficulty - user_readiness)

        return {
            "active": active,
            "prereq_gap": prereq_gap,
            "concept_bonus": concept_bonus,
            "difficulty_gap": difficulty_gap,
            "redundant": redundant,
        }

    def _compose_course_fit(self, profile):
        if profile is None:
            return None
        return (
            float(self.cfg.feedback_course_concept_weight) * profile["concept_bonus"]
            - float(self.cfg.feedback_course_prereq_weight) * profile["prereq_gap"]
            - float(self.cfg.feedback_course_difficulty_weight) * profile["difficulty_gap"]
            - float(self.cfg.feedback_course_redundant_weight) * profile["redundant"]
        ) * profile["active"]

    def _compute_course_reward_terms(self, selected_user_ids, item_idx, target_pop=None, user_seen_items=None):
        batch_size = int(item_idx.size(0))
        zero = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        if selected_user_ids is None or user_seen_items is None:
            return {
                "prereq_gap": zero,
                "concept_bonus": zero,
                "difficulty_gap": zero,
                "redundant": zero,
            }
        profile = self._compute_course_profile(
            selected_user_ids,
            item_idx,
            user_seen_items=user_seen_items,
            target_pop=target_pop,
            only_cold=self.cfg.feedback_course_only_cold,
        )
        return {
            "prereq_gap": profile["prereq_gap"] * profile["active"],
            "concept_bonus": profile["concept_bonus"] * profile["active"],
            "difficulty_gap": profile["difficulty_gap"] * profile["active"],
            "redundant": profile["redundant"] * profile["active"],
        }

    def _compute_candidate_course_fit(self, candidate_user_idx, item_idx, target_pop=None, user_seen_items=None):
        batch_size, n_cand = candidate_user_idx.shape
        zero = torch.zeros((batch_size, n_cand), dtype=torch.float32, device=self.device)
        if user_seen_items is None or candidate_user_idx is None:
            return zero

        flat_user_idx = candidate_user_idx.reshape(-1)
        flat_item_idx = item_idx.view(-1, 1).expand(-1, n_cand).reshape(-1)
        flat_target_pop = None
        if target_pop is not None:
            flat_target_pop = target_pop.view(-1, 1).expand(-1, n_cand).reshape(-1)

        profile = self._compute_course_profile(
            flat_user_idx,
            flat_item_idx,
            user_seen_items=user_seen_items,
            target_pop=flat_target_pop,
            only_cold=self.cfg.feedback_course_sample_only_cold,
        )
        fit = self._compose_course_fit(profile)
        return fit.view(batch_size, n_cand)

    def _apply_course_sampling_bias(
        self,
        candidates,
        cand_user_idx,
        item_idx,
        target_pop=None,
        user_seen_items=None,
        state_emb=None,
    ):
        if (
            candidates is None or
            cand_user_idx is None or
            float(self.cfg.feedback_course_sample_beta) <= 0.0
        ):
            return candidates, cand_user_idx, None

        batch_size, n_cand = cand_user_idx.shape
        if getattr(self.cfg, "ablation_fast3_soft_sampling", False) and state_emb is not None:
            fit_score = self._compute_candidate_course_fit(
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            if not torch.isfinite(fit_score).all():
                fit_score = torch.nan_to_num(fit_score, nan=0.0, posinf=0.0, neginf=0.0)

            top_l_cfg = int(getattr(self.cfg, "feedback_course_sample_top_l", 0))
            if top_l_cfg <= 0 or top_l_cfg >= n_cand:
                top_l = n_cand
            else:
                top_l = max(1, top_l_cfg)

            retrieval_score = (
                F.normalize(state_emb, dim=1).unsqueeze(1) * F.normalize(candidates, dim=2)
            ).sum(dim=2)
            fit_scale = fit_score.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
            fit_norm = fit_score / fit_scale
            combined_score = retrieval_score + float(self.cfg.feedback_course_sample_beta) * fit_norm

            base_order = torch.argsort(retrieval_score, dim=1, descending=True)
            top_idx = base_order[:, :top_l]
            rest_idx = base_order[:, top_l:]
            top_combined = combined_score.gather(1, top_idx)
            top_reorder = torch.argsort(top_combined, dim=1, descending=True)
            top_idx = top_idx.gather(1, top_reorder)
            final_order = torch.cat([top_idx, rest_idx], dim=1)

            candidates = candidates.gather(1, final_order.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
            cand_user_idx = cand_user_idx.gather(1, final_order)
            fit_score = fit_score.gather(1, final_order)
            return candidates, cand_user_idx, fit_score

        topk_cfg = int(getattr(self.cfg, "feedback_course_sample_topk", 0))
        if topk_cfg <= 0 or topk_cfg >= n_cand:
            fit_score = self._compute_candidate_course_fit(
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            if not torch.isfinite(fit_score).all():
                fit_score = torch.nan_to_num(fit_score, nan=0.0, posinf=0.0, neginf=0.0)

            order = torch.argsort(fit_score, dim=1, descending=True)
            candidates = candidates.gather(1, order.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
            cand_user_idx = cand_user_idx.gather(1, order)
            fit_score = fit_score.gather(1, order)
            return candidates, cand_user_idx, fit_score

        topk = min(max(1, topk_cfg), n_cand)
        top_candidates = candidates[:, :topk, :]
        top_user_idx = cand_user_idx[:, :topk]
        fit_top = self._compute_candidate_course_fit(
            top_user_idx,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )
        if not torch.isfinite(fit_top).all():
            fit_top = torch.nan_to_num(fit_top, nan=0.0, posinf=0.0, neginf=0.0)

        top_order = torch.argsort(fit_top, dim=1, descending=True)
        top_candidates = top_candidates.gather(1, top_order.unsqueeze(-1).expand(-1, -1, top_candidates.size(-1)))
        top_user_idx = top_user_idx.gather(1, top_order)
        fit_top = fit_top.gather(1, top_order)

        if topk < n_cand:
            rest_candidates = candidates[:, topk:, :]
            rest_user_idx = cand_user_idx[:, topk:]
            rest_fit = torch.zeros((batch_size, n_cand - topk), dtype=fit_top.dtype, device=self.device)
            candidates = torch.cat([top_candidates, rest_candidates], dim=1)
            cand_user_idx = torch.cat([top_user_idx, rest_user_idx], dim=1)
            fit_score = torch.cat([fit_top, rest_fit], dim=1)
        else:
            candidates = top_candidates
            cand_user_idx = top_user_idx
            fit_score = fit_top
        return candidates, cand_user_idx, fit_score

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
    ):
        current_h = init_item_emb.clone()
        trajectory = {
            "log_probs": [],
            "values": [],
            "rewards": [],
            "entropies": [],
            "states": [],
            "time_steps": [],
            "candidates": [],
            "actions": [],
        }
        candidate_stats = {
            "dup_rate": 0.0,
            "topm_coverage": 0.0,
            "steps": 0,
            "step_gain": 0.0,
            "collapse_penalty": 0.0,
            "course_sample_fit": 0.0,
            "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0,
            "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
            "target_alpha": 0.0,
        }

        if user_bank_raw is None and self.training and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw = self._build_user_bank_raw()

        for t in range(self.cfg.usim_steps):
            time_step = torch.full((current_h.size(0), 1), t, device=self.device)
            candidates, cand_user_idx, cand_stats = self.get_candidates(current_h, user_bank_raw=user_bank_raw)
            candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                candidates,
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
                state_emb=current_h,
            )
            action_idx, log_prob, value, entropy = self.agent.get_action_value(current_h, time_step, candidates)

            if cand_stats is not None:
                candidate_stats["dup_rate"] += cand_stats["dup_rate"]
                candidate_stats["topm_coverage"] += cand_stats["topm_coverage"]
                candidate_stats["steps"] += 1
            if fit_score is not None:
                candidate_stats["course_sample_fit"] += float(fit_score.mean().item())

            trajectory["states"].append(current_h.detach().clone())
            trajectory["time_steps"].append(time_step.detach().clone())
            trajectory["candidates"].append(candidates.detach().clone())
            trajectory["actions"].append(action_idx.detach().clone())

            prev_h = current_h
            batch_indices = torch.arange(current_h.size(0), device=self.device)
            selected_user = candidates[batch_indices, action_idx]
            selected_user_ids = None
            if cand_user_idx is not None:
                selected_user_ids = cand_user_idx[batch_indices, action_idx]

            with torch.enable_grad():
                h_detached = current_h.detach().requires_grad_(True)
                if getattr(self.cfg, "ablation_fast3_target_mix", False) and target_emb is not None:
                    user_align = (h_detached * selected_user.detach()).sum(dim=1, keepdim=True)
                    target_align = (h_detached * target_emb.detach()).sum(dim=1, keepdim=True)
                    target_alpha = self._compute_target_alpha(
                        target_pop=target_pop,
                        step_idx=t,
                        entropy=entropy,
                        num_candidates=candidates.size(1),
                        batch_size=current_h.size(0),
                    )
                    candidate_stats["target_alpha"] += float(target_alpha.mean().item())
                    score = (((1.0 - target_alpha) * user_align) + (target_alpha * target_align)).mean()
                else:
                    score = (h_detached * selected_user.detach()).sum(dim=1).mean()
                grad = torch.autograd.grad(score, h_detached)[0]

            current_h = current_h + self.cfg.usim_lr * grad

            reward = torch.zeros(current_h.size(0), 1, device=self.device)
            step_gain_mean = 0.0
            collapse_penalty = 0.0
            if target_emb is not None:
                prev_dist = F.mse_loss(prev_h, target_emb, reduction="none").mean(dim=1, keepdim=True)
                new_dist = F.mse_loss(current_h, target_emb, reduction="none").mean(dim=1, keepdim=True)

                terminal_reward = -new_dist * float(self.cfg.reward_terminal_weight)
                step_gain = (prev_dist - new_dist).clamp(
                    min=-float(self.cfg.reward_gain_clip),
                    max=float(self.cfg.reward_gain_clip),
                )
                reward = terminal_reward + float(self.cfg.reward_gain_weight) * step_gain
                step_gain_mean = float(step_gain.mean().item())

                if cand_stats is not None:
                    collapse_penalty = float(self.cfg.reward_dup_penalty_weight) * float(cand_stats["dup_rate"])
                    reward = reward - collapse_penalty
                    if float(self.cfg.reward_cov_bonus_weight) > 0.0:
                        reward = reward + float(self.cfg.reward_cov_bonus_weight) * float(cand_stats["topm_coverage"])

            course_terms = self._compute_course_reward_terms(
                selected_user_ids,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            reward = (
                reward
                + float(self.cfg.feedback_course_concept_weight) * course_terms["concept_bonus"]
                - float(self.cfg.feedback_course_prereq_weight) * course_terms["prereq_gap"]
                - float(self.cfg.feedback_course_difficulty_weight) * course_terms["difficulty_gap"]
                - float(self.cfg.feedback_course_redundant_weight) * course_terms["redundant"]
            )

            candidate_stats["step_gain"] += step_gain_mean
            candidate_stats["collapse_penalty"] += collapse_penalty
            candidate_stats["course_prereq_gap"] += float(course_terms["prereq_gap"].mean().item())
            candidate_stats["course_concept_bonus"] += float(course_terms["concept_bonus"].mean().item())
            candidate_stats["course_difficulty_gap"] += float(course_terms["difficulty_gap"].mean().item())
            candidate_stats["course_redundant"] += float(course_terms["redundant"].mean().item())
            trajectory["log_probs"].append(log_prob.detach())
            trajectory["values"].append(value.detach() if getattr(self.cfg, "ablation_fast3_ppo", False) else value)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

        if candidate_stats["steps"] > 0:
            candidate_stats["dup_rate"] /= candidate_stats["steps"]
            candidate_stats["topm_coverage"] /= candidate_stats["steps"]
            candidate_stats["step_gain"] /= candidate_stats["steps"]
            candidate_stats["collapse_penalty"] /= candidate_stats["steps"]
            candidate_stats["course_sample_fit"] /= candidate_stats["steps"]
            candidate_stats["course_prereq_gap"] /= candidate_stats["steps"]
            candidate_stats["course_concept_bonus"] /= candidate_stats["steps"]
            candidate_stats["course_difficulty_gap"] /= candidate_stats["steps"]
            candidate_stats["course_redundant"] /= candidate_stats["steps"]
            candidate_stats["target_alpha"] /= candidate_stats["steps"]
        return current_h, trajectory, candidate_stats

    def compute_ppo_loss(self, trajectory):
        if not getattr(self.cfg, "ablation_fast3_ppo", False):
            return super().compute_ppo_loss(trajectory)

        rewards = torch.stack(trajectory["rewards"]).squeeze(-1)
        old_log_probs = torch.stack(trajectory["log_probs"])
        old_values = torch.stack(trajectory["values"]).squeeze(-1)
        states = trajectory["states"]
        time_steps = trajectory["time_steps"]
        candidates = trajectory["candidates"]
        actions = trajectory["actions"]

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        next_value = torch.zeros_like(old_values[0])
        gamma = float(self.cfg.ppo_gamma)
        lam = float(getattr(self.cfg, "ppo_lambda", 0.95))

        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + gamma * next_value - old_values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            next_value = old_values[t]

        returns = advantages + old_values
        if getattr(self.cfg, "ppo_adv_norm", False):
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
            advantages = (advantages - adv_mean) / adv_std

        total_ppo_loss = 0.0
        value_clip = float(getattr(self.cfg, "ppo_value_clip", 0.0))

        for _ in range(self.cfg.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            new_entropies_list = []

            for t in range(len(states)):
                _, new_log_prob, new_value, new_entropy = self.agent.get_action_value(
                    states[t],
                    time_steps[t],
                    candidates[t],
                    action_idx=actions[t],
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                new_entropies_list.append(new_entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze(-1)
            new_entropies = torch.stack(new_entropies_list)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            if value_clip > 0.0:
                value_delta = (new_values - old_values).clamp(-value_clip, value_clip)
                value_pred_clipped = old_values + value_delta
                critic_unclipped = (new_values - returns.detach()).pow(2)
                critic_clipped = (value_pred_clipped - returns.detach()).pow(2)
                critic_loss = 0.5 * torch.max(critic_unclipped, critic_clipped).mean()
            else:
                critic_loss = 0.5 * (new_values - returns.detach()).pow(2).mean()

            entropy_loss = -new_entropies.mean()
            total_ppo_loss += (
                actor_loss
                + self.cfg.ppo_coeffs["value"] * critic_loss
                + self.cfg.ppo_coeffs["entropy"] * entropy_loss
            )

        return total_ppo_loss / self.cfg.ppo_epochs

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        u, i = batch["u"], batch["i"]
        is_cold = pop < self.cfg.cold_threshold

        z_u_base = self.user_proj(self.user_emb(u))
        force_cold_mask = is_cold if self.cfg.train_force_cold else False
        z_i_base, id_e_true, content_e = self.get_item_vector(i, llm_s, force_cold=force_cold_mask)

        target_emb = z_i_base.detach().clone()
        hot_mask = ~is_cold
        if hot_mask.sum() > 0:
            target_emb[hot_mask] = self.item_id_emb(i[hot_mask]).detach()

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

        if self.training and self.cfg.use_mixed_hard_neg and logits_margin.size(0) > 1:
            batch_size = logits_margin.size(0)
            max_neg = batch_size - 1
            n_total_neg = min(self.cfg.train_num_negs, max_neg)

            if n_total_neg > 0:
                n_hard = int(n_total_neg * self.cfg.hard_neg_ratio)
                n_hard = max(0, min(n_hard, n_total_neg))
                n_rand = n_total_neg - n_hard

                neg_logits = logits_margin.clone()
                neg_logits[pos_mask] = -1e9

                hard_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                rand_idx = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

                if n_hard > 0:
                    if self.cfg.use_structured_hard_neg and self.item_hard_adj is not None:
                        hard_mask = self.item_hard_adj[i][:, i]
                        hard_mask = hard_mask & (~pos_mask)
                        hard_logits = neg_logits.masked_fill(~hard_mask, -1e9)
                        hard_scores, hard_idx = torch.topk(hard_logits, k=n_hard, dim=1)
                        valid_mask = hard_scores > -1e8

                        if (~valid_mask).any():
                            bad_rows = torch.nonzero((~valid_mask).any(dim=1), as_tuple=False).view(-1).tolist()
                            for row in bad_rows:
                                need = int((~valid_mask[row]).sum().item())
                                if need < 1:
                                    continue
                                fallback = neg_logits[row].clone()
                                if valid_mask[row].any():
                                    fallback[hard_idx[row, valid_mask[row]]] = -1e9
                                _, fill_idx = torch.topk(fallback, k=need, dim=0)
                                hard_idx[row, ~valid_mask[row]] = fill_idx
                    else:
                        _, hard_idx = torch.topk(neg_logits, k=n_hard, dim=1)

                if n_rand > 0:
                    rand_scores = torch.rand_like(neg_logits)
                    rand_scores[pos_mask] = -1e9
                    if n_hard > 0:
                        rand_scores.scatter_(1, hard_idx, -1e9)
                    _, rand_idx = torch.topk(rand_scores, k=n_rand, dim=1)

                cand_idx = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
                cand_logits = logits_margin.gather(1, cand_idx)
                main_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                main_loss = F.cross_entropy(cand_logits, main_targets)
            else:
                main_loss = F.cross_entropy(logits_margin, labels)
        else:
            main_loss = F.cross_entropy(logits_margin, labels)

        z_id = F.normalize(id_e_true, dim=1)
        z_con = F.normalize(content_e, dim=1)
        sim = torch.matmul(z_id, z_con.t()) / self.cfg.temp
        aux_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        prereq_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            self.training and self.cfg.use_prereq_aux_loss and user_seen_items is not None and
            self.item_prereq_item_mat is not None and self.item_prereq_item_cnt is not None and
            logits.size(0) > 1
        ):
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            seen_mat, seen_cnt_raw = self._build_seen_mat(user_ids, user_seen_items)
            seen_cnt = seen_cnt_raw.squeeze(1)
            prereq_mat_batch = self.item_prereq_item_mat[i]
            prereq_cnt_batch = self.item_prereq_item_cnt[i].unsqueeze(0)
            prereq_seen_batch = torch.matmul(seen_mat, prereq_mat_batch.t())

            violation_batch = torch.where(
                prereq_cnt_batch > 0,
                1.0 - prereq_seen_batch / prereq_cnt_batch.clamp_min(1.0),
                torch.zeros_like(prereq_seen_batch),
            ).clamp(0.0, 1.0)

            valid_rows = seen_cnt >= float(self.cfg.prereq_aux_min_seen)
            if self.cfg.prereq_aux_only_cold:
                valid_rows = valid_rows & is_cold

            unmet_mask = violation_batch > float(self.cfg.prereq_aux_violation_thr)
            unmet_mask = unmet_mask & (~pos_mask)
            candidate_mask = unmet_mask & valid_rows.unsqueeze(1)

            if candidate_mask.any():
                neg_scores = logits.masked_fill(~candidate_mask, -1e9)
                neg_vals, _ = neg_scores.max(dim=1)
                has_neg = neg_vals > -1e8
                if has_neg.any():
                    pos_vals = logits[torch.arange(logits.size(0), device=self.device), labels]
                    margin = float(self.cfg.prereq_aux_margin)
                    prereq_aux_loss = F.relu(margin - pos_vals[has_neg] + neg_vals[has_neg]).mean()

        total_loss = (
            main_loss +
            self.cfg.aux_weight * aux_loss +
            ppo_loss +
            self.cfg.prereq_aux_weight * prereq_aux_loss
        )
        return total_loss, candidate_stats


def run_static_experiment_feedback(df, cfg, device, model, optimizer, llm_scores):
    static_seed = int(os.environ.get("USIM_STATIC_SEED", "2025"))
    train_ratio = float(os.environ.get("USIM_STATIC_TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("USIM_STATIC_VAL_RATIO", "0.1"))
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        _console_print("[STATIC] ratio invalid, fallback to 0.8/0.1/0.1", force=True)
        train_ratio, val_ratio = 0.8, 0.1

    df_static = df.sample(frac=1.0, random_state=static_seed).reset_index(drop=True)
    n_total = len(df_static)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        _console_print(f"[STATIC] split failed: total={n_total}, train={n_train}, val={n_val}, test={n_test}", force=True)
        return

    train_df = df_static.iloc[:n_train]
    val_df = df_static.iloc[n_train:n_train + n_val]
    test_df = df_static.iloc[n_train + n_val:]

    train_ds = StreamDataset(train_df, llm_scores)
    val_ds = StreamDataset(val_df, llm_scores)
    test_ds = StreamDataset(test_df, llm_scores)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

    train_seen = {}
    _add_user_seen_from_df(train_seen, train_df)
    test_seen = _clone_user_seen(train_seen)
    _add_user_seen_from_df(test_seen, val_df)
    model.clear_seen_cache()

    _console_print(
        f"\n>>> Start STATIC train/eval (seed={static_seed}) | "
        f"split={train_ratio:.2f}/{val_ratio:.2f}/{1.0 - train_ratio - val_ratio:.2f} | "
        f"train={n_train}, val={n_val}, test={n_test}"
    )

    k_list = [5, 10, 20]
    metrics_keys = [f"R@{k}" for k in k_list] + [f"N@{k}" for k in k_list]

    do_early_stop = cfg.use_epoch_early_stop and cfg.n_epochs > 1
    es_best = None
    es_best_state = None
    es_best_opt_state = None
    es_no_improve = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_start = time.time()
        num_batches = max(1, len(train_loader))
        total_loss = 0.0
        steps = 0
        cand_dup_sum = 0.0
        cand_cov_sum = 0.0
        cand_gain_sum = 0.0
        cand_pen_sum = 0.0
        cand_mix_sum = 0.0
        course_prereq_sum = 0.0
        course_concept_sum = 0.0
        course_diff_sum = 0.0
        course_redundant_sum = 0.0
        cand_batches = 0

        optimizer.zero_grad()
        cached_user_bank = None
        if cfg.candidate_strategy == "retrieve_sample":
            cached_user_bank = model._build_user_bank_raw()
        _console_print(
            f"  [STATIC-TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | "
            f"samples={len(train_ds)} | batches={num_batches}"
        )
        last_progress_log = epoch_start

        for batch_idx, (batch, pop, llm) in enumerate(train_loader):
            if (
                cached_user_bank is not None and
                cfg.user_bank_refresh_steps > 0 and
                batch_idx > 0 and
                (batch_idx % cfg.user_bank_refresh_steps == 0)
            ):
                cached_user_bank = model._build_user_bank_raw()

            batch = {k: v.to(device) for k, v in batch.items()}
            loss, cand_info = model(
                batch,
                pop.to(device),
                llm.to(device),
                user_bank_raw=cached_user_bank,
                user_seen_items=train_seen,
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
                course_prereq_sum += cand_info.get("course_prereq_gap", 0.0)
                course_concept_sum += cand_info.get("course_concept_bonus", 0.0)
                course_diff_sum += cand_info.get("course_difficulty_gap", 0.0)
                course_redundant_sum += cand_info.get("course_redundant", 0.0)
                cand_batches += 1

            now_ts = time.time()
            if _should_log_train_progress(batch_idx, num_batches, cfg, last_progress_log, now_ts):
                done = batch_idx + 1
                elapsed = now_ts - epoch_start
                avg_batch_sec = elapsed / max(1, done)
                eta = avg_batch_sec * max(0, num_batches - done)
                pct = 100.0 * done / max(1, num_batches)
                msg = (
                    f"    [ETA] Static E{epoch + 1}/{cfg.n_epochs} | "
                    f"{done}/{num_batches} ({pct:.0f}%) | eta={_format_eta(eta)}"
                )
                print(msg)
                last_progress_log = now_ts

        epoch_sec = time.time() - epoch_start
        avg_loss = total_loss / max(1, steps)
        if cand_batches > 0:
            avg_dup = cand_dup_sum / cand_batches
            avg_cov = cand_cov_sum / cand_batches
            avg_gain = cand_gain_sum / cand_batches
            avg_pen = cand_pen_sum / cand_batches
            avg_mix = cand_mix_sum / cand_batches
            avg_course_prereq = course_prereq_sum / cand_batches
            avg_course_concept = course_concept_sum / cand_batches
            avg_course_diff = course_diff_sum / cand_batches
            avg_course_redundant = course_redundant_sum / cand_batches
            _console_print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                f"TargetMix: {avg_mix:.4f} | "
                f"Course[p={avg_course_prereq:.4f}, c={avg_course_concept:.4f}, "
                f"d={avg_course_diff:.4f}, r={avg_course_redundant:.4f}]"
            )
        else:
            _console_print(
                f"  [STATIC-TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
            )

        if do_early_stop:
            _console_print("  [STATIC-EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
            all_item_vecs_val = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            val_cold, _ = evaluate_usim(
                model, val_loader, device, llm_scores, k_list,
                eval_type="cold", full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=all_item_vecs_val
            )
            val_hot, _ = evaluate_usim(
                model, val_loader, device, llm_scores, k_list,
                eval_type="hot", full_ranking=True,
                user_seen_items=train_seen, all_item_vecs=all_item_vecs_val
            )

            key_n = f"N@{cfg.early_stop_k}"
            key_r = f"R@{cfg.early_stop_k}"
            cur_n = val_cold.get(key_n, 0.0) if val_cold else 0.0
            cur_cr = val_cold.get(key_r, 0.0) if val_cold else 0.0
            cur_hr = val_hot.get(key_r, 0.0) if val_hot else 0.0

            if es_best is None:
                is_better = True
            else:
                hot_floor = es_best["hot_r"] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
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
                es_best_state = _move_state_to_cpu(model.state_dict())
                es_best_opt_state = _move_state_to_cpu(optimizer.state_dict())
                es_no_improve = 0
                es_tag = "update"
            else:
                es_no_improve += 1
                es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

            _console_print(
                f"  [STATIC-EARLYSTOP] Epoch {epoch + 1}: "
                f"Full Cold {key_n}={cur_n:.4f}, Full Cold {key_r}={cur_cr:.4f}, "
                f"Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
            )

            del all_item_vecs_val, val_cold, val_hot
            _maybe_clear_cuda_cache()

            if es_no_improve >= cfg.early_stop_patience:
                _console_print(f"  [STATIC-EARLYSTOP] Triggered at epoch {epoch + 1}.")
                break

    if do_early_stop and es_best_state is not None:
        model.load_state_dict(es_best_state)
        if es_best_opt_state is not None:
            optimizer.load_state_dict(es_best_opt_state)
            _optimizer_state_to_device(optimizer, device)
        _console_print(
            f"  [STATIC-EARLYSTOP] Restore best epoch={es_best['epoch']} "
            f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
            f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
            f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
        )

    _console_print("  [STATIC-TEST-EVAL] Evaluate test split with sampled and full ranking...")
    all_item_vecs_test = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
    met_cold, n_cold_t = evaluate_usim(
        model, test_loader, device, llm_scores, k_list,
        n_neg=cfg.eval_n_neg, eval_type="cold",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    met_hot, n_hot_t = evaluate_usim(
        model, test_loader, device, llm_scores, k_list,
        n_neg=cfg.eval_n_neg, eval_type="hot",
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_cold, fn_c = evaluate_usim(
        model, test_loader, device, llm_scores, k_list,
        eval_type="cold", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )
    fmet_hot, fn_h = evaluate_usim(
        model, test_loader, device, llm_scores, k_list,
        eval_type="hot", full_ranking=True,
        user_seen_items=test_seen, all_item_vecs=all_item_vecs_test
    )

    print("\n" + "=" * 90)
    print("         FINAL REPORT (STATIC): 采样评估 (1+200) vs 全库排名 (Feedback-Aware RL-USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for key in metrics_keys:
        sc = met_cold.get(key, 0.0) if met_cold else 0.0
        sh = met_hot.get(key, 0.0) if met_hot else 0.0
        fc = fmet_cold.get(key, 0.0) if fmet_cold else 0.0
        fh = fmet_hot.get(key, 0.0) if fmet_hot else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
    print("-" * 90)
    print(f"采样 Samples: Cold={n_cold_t}, Hot={n_hot_t}")
    print(f"全库 Samples: Cold={fn_c}, Hot={fn_h}")
    print("=" * 90)

    detail_path, fullrank_path = _save_final_report_exports(
        protocol="static",
        metrics_keys=metrics_keys,
        sampled_cold=met_cold or {},
        sampled_hot=met_hot or {},
        full_cold=fmet_cold or {},
        full_hot=fmet_hot or {},
        sampled_cold_count=n_cold_t,
        sampled_hot_count=n_hot_t,
        full_cold_count=fn_c,
        full_hot_count=fn_h,
    )
    _console_print(f">> Saved {detail_path} and {fullrank_path}")


def main():
    data_dir = "processed_data_hin"
    _console_print(f"Loading Data for Feedback-Aware USIM from {data_dir}...")
    if not os.path.exists(f"{data_dir}/stream_data.pkl"):
        _console_print("错误: 请先运行 data_process_hin.py", force=True)
        return

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    df = pd.read_pickle(f"{data_dir}/stream_data.pkl")
    with open(f"{data_dir}/llm_scores.pkl", "rb") as f:
        llm_scores = pd.read_pickle(f)
    content_emb = torch.load(f"{data_dir}/content_emb.pt")

    cfg = FeedbackConfig(meta["n_users"], meta["n_items"], content_emb.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = build_course_artifacts(
            df,
            cfg.n_items,
            relation_dir="MOOCCube/relations",
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, course_stats = None, _empty_course_stats(cfg.n_items)
    item_final_pop = torch.zeros(cfg.n_items, dtype=torch.long)
    pop_stats = df.groupby("i_idx")["popularity"].max()
    for item_id, pop_value in pop_stats.items():
        idx = int(item_id)
        if 0 <= idx < cfg.n_items:
            item_final_pop[idx] = int(pop_value)

    model = FeedbackAwareUSIM(cfg, content_emb).to(device)
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_final_pop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    _console_print(f">> 架构: Feedback-Aware RL-USIM + InfoNCE (Batch Size={cfg.batch_size})")
    _console_print(
        f">> Candidate Strategy: {cfg.candidate_strategy} | "
        f"TopM={cfg.retrieve_top_m} | Temp={cfg.candidate_temp:.2f} | "
        f"Eps={cfg.candidate_epsilon:.2f} | Ncand={cfg.n_candidates} | "
        f"BankRefresh={cfg.user_bank_refresh_steps}"
    )
    _console_print(
        f">> Reward Loop: term_w={cfg.reward_terminal_weight:.2f} | "
        f"gain_w={cfg.reward_gain_weight:.2f} | "
        f"gain_clip={cfg.reward_gain_clip:.3f} | "
        f"dup_w={cfg.reward_dup_penalty_weight:.2f} | "
        f"cov_w={cfg.reward_cov_bonus_weight:.2f}"
    )
    _console_print(
        f">> Course Feedback: only_cold={cfg.feedback_course_only_cold} | "
        f"warm_seen={cfg.feedback_course_warm_seen} | "
        f"concept_min={cfg.feedback_course_concept_min:.2f} | "
        f"redundant_thr={cfg.feedback_course_redundant_thr:.2f} | "
        f"prereq_gate={cfg.feedback_course_prereq_gate:.2f} | "
        f"w[p={cfg.feedback_course_prereq_weight:.2f}, "
        f"c={cfg.feedback_course_concept_weight:.2f}, "
        f"d={cfg.feedback_course_difficulty_weight:.2f}, "
        f"r={cfg.feedback_course_redundant_weight:.2f}]"
    )
    _console_print(
        f">> Course Artifacts: enabled={cfg.feedback_load_course_artifacts}"
    )
    _console_print(
        f">> Course Sampling: beta={cfg.feedback_course_sample_beta:.2f} | "
        f"only_cold={cfg.feedback_course_sample_only_cold} | "
        f"topk={cfg.feedback_course_sample_topk} | "
        f"topL={cfg.feedback_course_sample_top_l}"
    )
    _console_print(
        f">> Ablation: target_mix={cfg.ablation_fast3_target_mix} | "
        f"soft_topL={cfg.ablation_fast3_soft_sampling} | "
        f"fast3_ppo={cfg.ablation_fast3_ppo} | "
        f"train_window={cfg.ablation_train_window}"
    )
    _console_print(
        f">> Ablation Params: target_alpha[cold={cfg.fast3_target_alpha_cold:.2f}, "
        f"hot={cfg.fast3_target_alpha_hot:.2f}, step={cfg.fast3_target_alpha_step:.2f}, "
        f"entropy={cfg.fast3_target_alpha_entropy:.2f}, "
        f"range=({cfg.fast3_target_alpha_min:.2f},{cfg.fast3_target_alpha_max:.2f})] | "
        f"ppo[epochs={cfg.ppo_epochs}, lambda={cfg.ppo_lambda:.2f}, "
        f"value_clip={cfg.ppo_value_clip:.2f}, adv_norm={cfg.ppo_adv_norm}]"
    )
    _console_print(
        f">> Cold Train: force_cold={cfg.train_force_cold} | "
        f"id_dropout={cfg.dropout_prob:.2f}"
    )
    _console_print(
        f">> Course Priors: source={course_stats.get('prereq_source', 'behavior')} | "
        f"concept={course_stats['items_with_concept']}/{cfg.n_items}, "
        f"prereq={course_stats['items_with_prereq']}/{cfg.n_items}, "
        f"hard_density={course_stats['hard_density']:.3f}, "
        f"prereq_raw={course_stats.get('prereq_edges_raw', 0)} | "
        f"prereq_edges={course_stats['prereq_edges_kept']} "
        f"(raw={course_stats['prereq_edges_raw']}, users={course_stats['prereq_users']})"
    )
    _console_print(
        f">> Course Mode: rerank={cfg.use_course_rerank} "
        f"(alpha={cfg.rerank_alpha:.2f}, lambda={cfg.rerank_lambda:.2f}) | "
        f"min_seen={cfg.rerank_min_seen} | topL={cfg.rerank_top_l} | "
        f"cap={cfg.rerank_penalty_cap:.2f} | only_cold={cfg.rerank_only_cold} | "
        f"prereq[min_sup={cfg.prereq_min_support}, max_per_item={cfg.prereq_max_per_item}] | "
        f"prereq_aux={cfg.use_prereq_aux_loss} (w={cfg.prereq_aux_weight:.2f}) | "
        f"structured_hard_neg={cfg.use_structured_hard_neg}"
    )
    _console_print(
        f">> EarlyStop: enabled={cfg.use_epoch_early_stop} | monitor=Full Cold N@{cfg.early_stop_k} | "
        f"tie=Full Cold R@{cfg.early_stop_k} | hot_drop_tol={cfg.early_stop_hot_r10_drop_tol:.2%} | "
        f"patience={cfg.early_stop_patience} | min_delta={cfg.early_stop_min_delta:.1e}"
    )

    if os.environ.get("USIM_STATIC", "0") == "1":
        _console_print(">> Static mode: use feedback-aware static train/eval with explicit progress logs.")
        run_static_experiment_feedback(df, cfg, device, model, optimizer, llm_scores)
        return

    if os.environ.get("USIM_STATIC", "0") == "1":
        _console_print(">> Static 模式复用原版 static 训练流程，仅更换模型与 reward。")
        run_static_experiment(df, cfg, device, model, optimizer, llm_scores)
        return

    periods = split_dataframe_by_periods(df, period_type="M")
    _console_print(f"\n>>> Start cumulative train/eval - total {len(periods)} periods <<<")

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
    user_seen_items = {}
    ckpt_dir = _feedback_ckpt_dir()
    ckpt_enabled = _feedback_ckpt_enabled()
    auto_resume = _feedback_ckpt_auto_resume()
    force_fresh = _feedback_ckpt_force_fresh()
    _console_print(
        f">> Checkpoint: enabled={ckpt_enabled} | auto_resume={auto_resume} | "
        f"force_fresh={force_fresh} | dir={ckpt_dir}"
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
        resume_state = _load_feedback_checkpoint(ckpt_dir)
        if resume_state is not None:
            status = resume_state.get("status", "between_periods")
            if status == "finished":
                _console_print(">> Resume: found finished checkpoint. Nothing to resume. Set USIM_FB_FORCE_FRESH=1 to start over.", force=True)
                return

            total_periods_saved = int(resume_state.get("total_periods", len(periods)))
            if total_periods_saved != len(periods):
                _console_print(
                    f">> Resume skipped: checkpoint total_periods={total_periods_saved}, "
                    f"current={len(periods)}"
                )
            else:
                model.load_state_dict(resume_state["model_state"])
                optimizer.load_state_dict(resume_state["optimizer_state"])
                _optimizer_state_to_device(optimizer, device)
                history = copy.deepcopy(resume_state.get("history", history))
                accum_cold = copy.deepcopy(resume_state.get("accum_cold", accum_cold))
                accum_hot = copy.deepcopy(resume_state.get("accum_hot", accum_hot))
                count_cold = int(resume_state.get("count_cold", count_cold))
                count_hot = int(resume_state.get("count_hot", count_hot))
                full_cold = copy.deepcopy(resume_state.get("full_cold", full_cold))
                full_hot = copy.deepcopy(resume_state.get("full_hot", full_hot))
                fc_cold = int(resume_state.get("fc_cold", fc_cold))
                fc_hot = int(resume_state.get("fc_hot", fc_hot))
                user_seen_items = _deserialize_user_seen_items(resume_state.get("user_seen_items"))
                model.clear_seen_cache()
                warmup_periods = int(resume_state.get("warmup_periods", warmup_periods))
                resume_accumulated_periods = int(resume_state.get("accumulated_periods", 0))
                accumulated_dfs = periods[:resume_accumulated_periods]
                start_period = int(resume_state.get("next_period", 0))
                resume_current_period = resume_state.get("current_period")
                if resume_current_period is not None:
                    resume_current_period = int(resume_current_period)
                    start_period = resume_current_period
                resume_next_epoch = int(resume_state.get("next_epoch", 0))
                resume_es_best = copy.deepcopy(resume_state.get("es_best"))
                resume_es_best_state = _move_state_to_cpu(resume_state.get("es_best_state"))
                resume_es_best_opt_state = _move_state_to_cpu(resume_state.get("es_best_opt_state"))
                resume_es_no_improve = int(resume_state.get("es_no_improve", 0))
                _console_print(
                    f">> Resume: status={status} | start_period={start_period} | "
                    f"resume_current_period={resume_current_period} | next_epoch={resume_next_epoch} | "
                    f"accumulated_periods={resume_accumulated_periods}"
                )

    for t in range(start_period, len(periods)):
        p_df = periods[t]
        eval_ds = StreamDataset(p_df, llm_scores)
        eval_loader = DataLoader(eval_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

        n_total = len(eval_ds)
        _console_print(f"\n>>> Period {t} (当前: {n_total}, 累积: {sum(len(d) for d in accumulated_dfs) + n_total}) <<<")

        cold_res = {key: 0.0 for key in metrics_keys}
        hot_res = {key: 0.0 for key in metrics_keys}
        n_cold_t, n_hot_t = 0, 0
        resume_this_period = (resume_current_period is not None and t == resume_current_period)

        if resume_this_period:
            _console_print(
                f"  [RESUME] Continue period {t} from epoch {resume_next_epoch + 1}/{cfg.n_epochs} "
                f"(accumulated_periods={resume_accumulated_periods})"
            )
        elif t >= warmup_periods:
            _console_print("  [EVAL-START] Build eval item bank and run sampled/full ranking...")
            all_item_vecs_eval = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
            met_cold, n_cold_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="cold",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            met_hot, n_hot_t = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                n_neg=cfg.eval_n_neg, eval_type="hot",
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_cold, fn_c = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="cold", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
            )
            fmet_hot, fn_h = evaluate_usim(
                model, eval_loader, device, llm_scores, k_list,
                eval_type="hot", full_ranking=True,
                user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_eval
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
            _console_print(f"  采样 Cold={c_s:.4f} Hot={h_s:.4f} | 全库 Cold={c_f:.4f} Hot={h_f:.4f}")
            del all_item_vecs_eval, met_cold, met_hot, fmet_cold, fmet_hot
            _maybe_clear_cuda_cache()
        else:
            _console_print("  [WARMUP] Training only...")

        if not resume_this_period:
            history["Period"].append(t)
            history["Count_cold"].append(n_cold_t)
            history["Count_hot"].append(n_hot_t)
            for key in metrics_keys:
                history["cold_" + key].append(cold_res.get(key, 0.0))
                history["hot_" + key].append(hot_res.get(key, 0.0))

            _add_user_seen_from_df(user_seen_items, p_df)
            model.clear_seen_cache()
            accumulated_dfs.append(p_df)
        train_period_window = max(0, int(getattr(cfg, "ablation_train_window", 0)))
        if train_period_window > 0 and len(accumulated_dfs) > train_period_window:
            train_dfs = accumulated_dfs[-train_period_window:]
            _console_print(
                f"  [WINDOW] Use latest {train_period_window}/{len(accumulated_dfs)} periods for training "
                f"({sum(len(d) for d in train_dfs)} samples)"
            )
        else:
            train_dfs = accumulated_dfs
        combined_df = pd.concat(train_dfs, ignore_index=True)
        train_ds = StreamDataset(combined_df, llm_scores)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        do_early_stop = (t >= warmup_periods) and cfg.use_epoch_early_stop and cfg.n_epochs > 1
        es_best = copy.deepcopy(resume_es_best) if resume_this_period else None
        es_best_state = copy.deepcopy(resume_es_best_state) if resume_this_period else None
        es_best_opt_state = copy.deepcopy(resume_es_best_opt_state) if resume_this_period else None
        es_no_improve = int(resume_es_no_improve) if resume_this_period else 0
        epoch_start_idx = resume_next_epoch if resume_this_period else 0

        if ckpt_enabled and not resume_this_period:
            period_start_state = _build_feedback_ckpt_state(
                model,
                optimizer,
                history,
                accum_cold,
                accum_hot,
                count_cold,
                count_hot,
                full_cold,
                full_hot,
                fc_cold,
                fc_hot,
                user_seen_items,
                accumulated_periods=t + 1,
                warmup_periods=warmup_periods,
                total_periods=len(periods),
                status="in_period",
                next_period=t,
                current_period=t,
                next_epoch=0,
            )
            _save_feedback_checkpoint(ckpt_dir, period_start_state)

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
            _console_print(
                f"  [TRAIN-START] Epoch {epoch + 1}/{cfg.n_epochs} | "
                f"Period {t + 1}/{len(periods)} | samples={len(combined_df)} | batches={num_batches}"
            )
            last_progress_log = epoch_start

            for batch_idx, (batch, pop, llm) in enumerate(train_loader):
                if (
                    cached_user_bank is not None and
                    cfg.user_bank_refresh_steps > 0 and
                    batch_idx > 0 and
                    (batch_idx % cfg.user_bank_refresh_steps == 0)
                ):
                    cached_user_bank = model._build_user_bank_raw()

                batch = {k: v.to(device) for k, v in batch.items()}
                loss, cand_info = model(
                    batch,
                    pop.to(device),
                    llm.to(device),
                    user_bank_raw=cached_user_bank,
                    user_seen_items=user_seen_items,
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
                if _should_log_train_progress(batch_idx, num_batches, cfg, last_progress_log, now_ts):
                    done = batch_idx + 1
                    elapsed = now_ts - epoch_start
                    avg_batch_sec = elapsed / max(1, done)
                    eta = avg_batch_sec * max(0, num_batches - done)
                    pct = 100.0 * done / max(1, num_batches)
                    msg = (
                        f"[ETA] P{t + 1}/{len(periods)} E{epoch + 1}/{cfg.n_epochs} | "
                        f"{done}/{num_batches} ({pct:.0f}%) | eta={_format_eta(eta)}"
                    )
                    print(msg)
                    last_progress_log = now_ts

            epoch_sec = time.time() - epoch_start
            avg_loss = total_loss / max(1, steps)
            if cand_batches > 0:
                avg_dup = cand_dup_sum / cand_batches
                avg_cov = cand_cov_sum / cand_batches
                avg_gain = cand_gain_sum / cand_batches
                avg_pen = cand_pen_sum / cand_batches
                avg_mix = cand_mix_sum / cand_batches
                avg_course_sample_fit = course_sample_fit_sum / cand_batches
                avg_course_prereq = course_prereq_sum / cand_batches
                avg_course_concept = course_concept_sum / cand_batches
                avg_course_diff = course_diff_sum / cand_batches
                avg_course_redundant = course_redundant_sum / cand_batches
                _console_print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s | "
                    f"CandDup: {avg_dup:.4f} | TopMCov: {avg_cov:.4f} | "
                    f"StepGain: {avg_gain:.4f} | CollapsePen: {avg_pen:.4f} | "
                    f"TargetMix: {avg_mix:.4f} | "
                    f"SampleFit: {avg_course_sample_fit:.4f} | "
                    f"Course[p={avg_course_prereq:.4f}, c={avg_course_concept:.4f}, "
                    f"d={avg_course_diff:.4f}, r={avg_course_redundant:.4f}]"
                )
            else:
                _console_print(
                    f"  [TRAIN] Epoch {epoch + 1}/{cfg.n_epochs} | 累积: {len(combined_df)} | "
                    f"Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s"
                )

            if ckpt_enabled:
                epoch_state = _build_feedback_ckpt_state(
                    model,
                    optimizer,
                    history,
                    accum_cold,
                    accum_hot,
                    count_cold,
                    count_hot,
                    full_cold,
                    full_hot,
                    fc_cold,
                    fc_hot,
                    user_seen_items,
                    accumulated_periods=t + 1,
                    warmup_periods=warmup_periods,
                    total_periods=len(periods),
                    status="in_period",
                    next_period=t,
                    current_period=t,
                    next_epoch=epoch + 1,
                    es_best=es_best,
                    es_best_state=es_best_state,
                    es_best_opt_state=es_best_opt_state,
                    es_no_improve=es_no_improve,
                )
                _save_feedback_checkpoint(ckpt_dir, epoch_state)

            if do_early_stop:
                _console_print("  [EARLYSTOP-EVAL] Run full-ranking cold/hot validation...")
                all_item_vecs_es = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
                es_cold, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="cold", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )
                es_hot, _ = evaluate_usim(
                    model, eval_loader, device, llm_scores, k_list,
                    eval_type="hot", full_ranking=True,
                    user_seen_items=user_seen_items, all_item_vecs=all_item_vecs_es
                )

                key_n = f"N@{cfg.early_stop_k}"
                key_r = f"R@{cfg.early_stop_k}"
                cur_n = es_cold.get(key_n, 0.0) if es_cold else 0.0
                cur_cr = es_cold.get(key_r, 0.0) if es_cold else 0.0
                cur_hr = es_hot.get(key_r, 0.0) if es_hot else 0.0

                if es_best is None:
                    is_better = True
                else:
                    hot_floor = es_best["hot_r"] * (1.0 - cfg.early_stop_hot_r10_drop_tol)
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
                    es_best_state = _move_state_to_cpu(model.state_dict())
                    es_best_opt_state = _move_state_to_cpu(optimizer.state_dict())
                    es_no_improve = 0
                    es_tag = "update"
                else:
                    es_no_improve += 1
                    es_tag = f"wait({es_no_improve}/{cfg.early_stop_patience})"

                _console_print(
                    f"  [EARLYSTOP] Epoch {epoch + 1}: Full Cold {key_n}={cur_n:.4f}, "
                    f"Full Cold {key_r}={cur_cr:.4f}, Full Hot {key_r}={cur_hr:.4f} | {es_tag}"
                )

                del all_item_vecs_es, es_cold, es_hot
                _maybe_clear_cuda_cache()

                if ckpt_enabled:
                    epoch_eval_state = _build_feedback_ckpt_state(
                        model,
                        optimizer,
                        history,
                        accum_cold,
                        accum_hot,
                        count_cold,
                        count_hot,
                        full_cold,
                        full_hot,
                        fc_cold,
                        fc_hot,
                        user_seen_items,
                        accumulated_periods=t + 1,
                        warmup_periods=warmup_periods,
                        total_periods=len(periods),
                        status="in_period",
                        next_period=t,
                        current_period=t,
                        next_epoch=epoch + 1,
                        es_best=es_best,
                        es_best_state=es_best_state,
                        es_best_opt_state=es_best_opt_state,
                        es_no_improve=es_no_improve,
                    )
                    _save_feedback_checkpoint(ckpt_dir, epoch_eval_state)

                if es_no_improve >= cfg.early_stop_patience:
                    _console_print(f"  [EARLYSTOP] Triggered at epoch {epoch + 1}.")
                    break

        if do_early_stop and es_best_state is not None:
            model.load_state_dict(es_best_state)
            if es_best_opt_state is not None:
                optimizer.load_state_dict(es_best_opt_state)
                _optimizer_state_to_device(optimizer, device)
            _console_print(
                f"  [EARLYSTOP] Restore best epoch={es_best['epoch']} "
                f"(Full Cold N@{cfg.early_stop_k}={es_best['cold_n']:.4f}, "
                f"R@{cfg.early_stop_k}={es_best['cold_r']:.4f}, "
                f"Full Hot R@{cfg.early_stop_k}={es_best['hot_r']:.4f})"
            )
            _maybe_clear_cuda_cache()

        if ckpt_enabled:
            end_state = _build_feedback_ckpt_state(
                model,
                optimizer,
                history,
                accum_cold,
                accum_hot,
                count_cold,
                count_hot,
                full_cold,
                full_hot,
                fc_cold,
                fc_hot,
                user_seen_items,
                accumulated_periods=t + 1,
                warmup_periods=warmup_periods,
                total_periods=len(periods),
                status="between_periods",
                next_period=t + 1,
            )
            _save_feedback_checkpoint(ckpt_dir, end_state)

        if resume_this_period:
            resume_current_period = None
            resume_next_epoch = 0
            resume_es_best = None
            resume_es_best_state = None
            resume_es_best_opt_state = None
            resume_es_no_improve = 0

    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{cfg.eval_n_neg}) vs 全库排名 (Feedback-Aware RL-USIM)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)

    for key in metrics_keys:
        sc = accum_cold[key] / count_cold if count_cold > 0 else 0.0
        sh = accum_hot[key] / count_hot if count_hot > 0 else 0.0
        fc = full_cold[key] / fc_cold if fc_cold > 0 else 0.0
        fh = full_hot[key] / fc_hot if fc_hot > 0 else 0.0
        print(f"{key:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")

    print("-" * 90)
    print(f"采样 Samples: Cold={count_cold}, Hot={count_hot}")
    print(f"全库 Samples: Cold={fc_cold}, Hot={fc_hot}")
    print("=" * 90)

    final_sampled_cold = {key: (accum_cold[key] / count_cold if count_cold > 0 else 0.0) for key in metrics_keys}
    final_sampled_hot = {key: (accum_hot[key] / count_hot if count_hot > 0 else 0.0) for key in metrics_keys}
    final_full_cold = {key: (full_cold[key] / fc_cold if fc_cold > 0 else 0.0) for key in metrics_keys}
    final_full_hot = {key: (full_hot[key] / fc_hot if fc_hot > 0 else 0.0) for key in metrics_keys}
    detail_path, fullrank_path = _save_final_report_exports(
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
    )

    metrics_path = _feedback_output_path("mooc_metrics_usim_feedback.csv")
    plot_path = _feedback_output_path("mooc_result_usim_feedback.png")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["Period"], history["cold_R@10"], marker="o", label="Cold R@10")
    plt.plot(history["Period"], history["hot_R@10"], marker="s", label="Hot R@10")
    plt.axvline(x=warmup_periods - 0.5, color="r", linestyle="--", label="Warmup End")
    plt.title("Feedback-Aware RL-USIM: Cumulative Training")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(plot_path)
    print(f">> Saved {plot_path}, {metrics_path}, {detail_path}, and {fullrank_path}")

    if ckpt_enabled:
        finished_state = _build_feedback_ckpt_state(
            model,
            optimizer,
            history,
            accum_cold,
            accum_hot,
            count_cold,
            count_hot,
            full_cold,
            full_hot,
            fc_cold,
            fc_hot,
            user_seen_items,
            accumulated_periods=len(periods),
            warmup_periods=warmup_periods,
            total_periods=len(periods),
            status="finished",
            next_period=len(periods),
        )
        _save_feedback_checkpoint(ckpt_dir, finished_state, snapshot_name="finished.pt")


if __name__ == "__main__":
    setup_seed(2025)
    main()
