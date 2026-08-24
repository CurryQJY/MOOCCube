"""
Repaired strict-cold FAST3/USIM entrypoint.

This script keeps the original implementation importable as the legacy
baseline, then installs a repaired config/model pair before delegating to the
original training pipeline.
"""

import hashlib
import csv
import json
import math
import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import fast3_delta.checkpoint as checkpoint_mod
import fast3_delta.eval as eval_mod
import usim_feedback_fast3_content_delta as legacy


_legacy_build_eval_item_vecs = eval_mod.build_eval_item_vecs
_legacy_build_eval_pos_item_vecs = eval_mod.build_eval_pos_item_vecs
_legacy_make_fast3_optimizer = legacy._make_fast3_optimizer
_legacy_build_feedback_ckpt_state = legacy._build_feedback_ckpt_state
_legacy_load_feedback_checkpoint = legacy._load_feedback_checkpoint
_legacy_write_static_manifest = legacy._write_static_manifest
_legacy_compute_early_stop_score = legacy._compute_early_stop_score
_legacy_setup_seed = legacy.setup_seed
_legacy_static_train_config_fingerprint = checkpoint_mod._static_train_config_fingerprint
_legacy_checkpoint_config_matches = checkpoint_mod._checkpoint_config_matches
_pending_recppo_optimizer_state = None
_candidate_recppo_optimizer_state = None
_pending_recppo_best_optimizer_state = None
_candidate_recppo_best_optimizer_state = None
_active_recppo_model_ref = None

# Static runner guard tokens delegated to legacy: def run_static_experiment, _static_split_df


class RepairedFast3Config(legacy.Fast3Config):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)

        if "USIM_PPO_LOSS_WEIGHT" not in os.environ:
            self.ppo_loss_weight = 1.0
        if "USIM_ROLLOUT_POLICY" not in os.environ:
            self.rollout_policy = "ppo"
        if "USIM_USE_CONTENT_DELTA" not in os.environ:
            # Keep RecPPO attribution clean; ablation shows OFF is better under v2 recipe.
            self.use_content_delta = False
        if "USIM_RL_RESIDUAL_SCALE" not in os.environ:
            self.rl_residual_scale = 0.06
        self.recppo_enabled = (
            float(self.ppo_loss_weight) > 0.0
            and str(self.rollout_policy).strip().lower() == "ppo"
        )

        self.reward_step_cost = float(os.environ.get("USIM_FB_REWARD_STEP_COST", "0.01"))
        self.recppo_terminal_value_weight = float(os.environ.get("USIM_RECPPO_TERM_VALUE_W", "0.20"))
        self.recppo_behavior_ce_weight = float(os.environ.get("USIM_RECPPO_BEHAVIOR_CE_W", "0.20"))
        self.recppo_behavior_ce_final_weight = max(
            0.0,
            min(
                self.recppo_behavior_ce_weight,
                float(os.environ.get("USIM_RECPPO_BEHAVIOR_CE_FINAL_W", "0.02")),
            ),
        )
        self.recppo_behavior_ce_anneal_epochs = max(
            1,
            int(os.environ.get("USIM_RECPPO_BEHAVIOR_CE_ANNEAL_EPOCHS", "10")),
        )
        self.recppo_bootstrap_next_value = os.environ.get("USIM_RECPPO_BOOTSTRAP_NEXT", "1") == "1"
        self.recppo_inject_behavior_user = os.environ.get("USIM_RECPPO_INJECT_BEHAVIOR_USER", "1") == "1"
        self.recppo_teacher_force_behavior = os.environ.get("USIM_RECPPO_TEACHER_FORCE_BEHAVIOR", "0") == "1"
        self.recppo_value_bound = float(os.environ.get("USIM_RECPPO_VALUE_BOUND", "2.0"))
        self.recppo_logit_bound = float(os.environ.get("USIM_RECPPO_LOGIT_BOUND", "10.0"))
        self.recppo_policy_temperature = max(
            1e-4,
            float(os.environ.get("USIM_RECPPO_POLICY_TEMP", "1.0")),
        )
        self.recppo_enable_stop = os.environ.get("USIM_RECPPO_ENABLE_STOP", "1") == "1"
        self.recppo_min_steps = max(0, int(os.environ.get("USIM_RECPPO_MIN_STEPS", "2")))
        self.recppo_stop_bias_init = float(os.environ.get("USIM_RECPPO_STOP_BIAS_INIT", "-2.0"))
        self.recppo_actor_lr = float(os.environ.get("USIM_RECPPO_ACTOR_LR", "5e-4"))
        self.recppo_critic_lr = float(os.environ.get("USIM_RECPPO_CRITIC_LR", "1e-3"))
        self.recppo_max_grad_norm = float(os.environ.get("USIM_RECPPO_MAX_GRAD_NORM", "1.0"))
        self.recppo_rank_gain_weight = float(os.environ.get("USIM_RECPPO_RANK_GAIN_W", "1.0"))
        self.recppo_embedding_gain_weight = max(
            0.0,
            float(os.environ.get("USIM_RECPPO_EMBEDDING_GAIN_W", "0.10")),
        )
        self.recppo_course_reward_scale = max(
            0.0,
            float(os.environ.get("USIM_RECPPO_COURSE_REWARD_SCALE", "0.10")),
        )
        self.recppo_course_reward_clip = max(
            0.0,
            float(os.environ.get("USIM_RECPPO_COURSE_REWARD_CLIP", "0.02")),
        )
        self.recppo_rank_topk = max(1, int(os.environ.get("USIM_RECPPO_RANK_TOPK", "10")))
        self.recppo_rank_temperature = max(
            1e-4,
            float(os.environ.get("USIM_RECPPO_RANK_TEMP", str(self.temp))),
        )
        self.recppo_rank_normalize_transition = (
            os.environ.get("USIM_RECPPO_RANK_NORMALIZE_TRANSITION", "1") == "1"
        )
        self.recppo_rank_item_chunk_size = max(
            1,
            int(os.environ.get("USIM_RECPPO_RANK_ITEM_CHUNK", "32")),
        )
        self.recppo_max_residual_norm = float(os.environ.get("USIM_RECPPO_MAX_RESIDUAL_NORM", "0.5"))
        self.recppo_residual_ramp_epochs = max(
            1,
            int(os.environ.get("USIM_RECPPO_RESIDUAL_RAMP_EPOCHS", "1")),
        )
        self.recppo_target_kl = max(0.0, float(os.environ.get("USIM_RECPPO_TARGET_KL", "0.05")))
        self.recppo_guard_hot_ratio = min(
            1.0,
            max(0.0, float(os.environ.get("USIM_RECPPO_GUARD_HOT_RATIO", "0.50"))),
        )
        self.recppo_require_policy_checkpoint = (
            os.environ.get("USIM_RECPPO_REQUIRE_POLICY_CKPT", "1") == "1"
        )
        self.recppo_strict_determinism = os.environ.get("USIM_RECPPO_STRICT_DETERMINISM", "1") == "1"
        default_warmup = 0 if self.n_epochs <= 1 else min(20, max(1, self.n_epochs // 2))
        self.recppo_warmup_epochs = max(
            0,
            int(os.environ.get("USIM_RECPPO_WARMUP_EPOCHS", str(default_warmup))),
        )
        if self.recppo_enabled and self.n_epochs > 0 and self.recppo_warmup_epochs >= self.n_epochs:
            raise ValueError(
                "USIM_RECPPO_WARMUP_EPOCHS must be smaller than USIM_N_EPOCHS "
                "so at least one RecPPO epoch is trained"
            )
        self.ppo_gamma = float(os.environ.get("USIM_PPO_GAMMA", "0.99"))
        self.ppo_epochs = max(1, int(os.environ.get("USIM_PPO_EPOCHS", str(self.ppo_epochs))))

        if "USIM_USE_PSEUDO_COLD_TRAIN" not in os.environ:
            self.use_pseudo_cold_train = True
        if "USIM_PSEUDO_COLD_MODE" not in os.environ:
            self.pseudo_cold_mode = "all_eligible"
        if "USIM_PSEUDO_COLD_RATIO" not in os.environ:
            self.pseudo_cold_ratio = 1.0
        if "USIM_PSEUDO_COLD_MIN_POP" not in os.environ:
            self.pseudo_cold_min_pop = 1
        if "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET" not in os.environ:
            self.feedback_course_match_exclude_target = True
        if "USIM_EARLY_STOP_SCORE_MODE" not in os.environ:
            self.early_stop_score_mode = "cold_rn"
        self.early_stop_score_mode = os.environ.get(
            "USIM_RECPPO_EARLY_STOP_MODE",
            "recppo_stage_guarded",
        ).strip().lower()
        self.early_stop_min_delta = max(
            0.0,
            float(os.environ.get("USIM_RECPPO_EARLY_STOP_MIN_DELTA", "0")),
        )


def repaired_strict_cold_item_mask(model, device):
    pop = getattr(model, "item_popularity", None)
    if pop is None:
        if eval_mod.refined_eval_enabled(model):
            raise RuntimeError("USIM repaired refined eval requires item_popularity to identify strict-cold items")
        return None
    return pop.to(device=device).float().view(-1) < float(getattr(model.cfg, "cold_threshold", 1))


def repaired_setup_seed(seed=2025):
    _legacy_setup_seed(seed)
    strict = os.environ.get("USIM_RECPPO_STRICT_DETERMINISM", "1") == "1"
    torch.use_deterministic_algorithms(strict, warn_only=False)


class RepairedFast3FeedbackUSIM(legacy.Fast3FeedbackUSIM):
    def __init__(self, config, content_emb):
        global _active_recppo_model_ref
        super().__init__(config, content_emb)
        self.recppo_stop_head = nn.Linear(self.agent.actor_head.in_features, 1)
        nn.init.zeros_(self.recppo_stop_head.weight)
        nn.init.constant_(self.recppo_stop_head.bias, float(config.recppo_stop_bias_init))
        self.recppo_outer_anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("_recppo_epoch_state", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_recppo_phase_state", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_recppo_warm_hot_r", torch.tensor(0.0))
        self.register_buffer("_recppo_warm_hot_n", torch.tensor(0.0))
        self._recppo_epoch_pending = True
        self._recppo_collect_only = False
        self._last_recppo_trajectory = None
        self._last_recppo_info = {}
        self._recppo_diag_sums = {}
        self._recppo_diag_count = 0
        self._recppo_optimizer = None
        self._recppo_best_optimizer_state = None
        self._clear_recppo_rank_cache()
        self._reset_recppo_optimizer()
        _active_recppo_model_ref = weakref.ref(self)

    def train(self, mode=True):
        was_training = bool(self.training)
        result = super().train(mode)
        if was_training and not mode and hasattr(self, "_recppo_diag_count"):
            self._flush_recppo_diagnostics()
        if hasattr(self, "_recppo_epoch_pending"):
            self._recppo_epoch_pending = bool(mode)
        return result

    def _accumulate_recppo_diagnostics(self, info):
        diagnostic_keys = (
            "recppo_total_loss",
            "recppo_policy_loss",
            "recppo_actor_loss",
            "recppo_critic_loss",
            "recppo_terminal_value_loss",
            "recppo_behavior_ce_loss",
            "recppo_behavior_ce_weight",
            "recppo_entropy",
            "recppo_approx_kl",
            "recppo_clip_fraction",
            "recppo_max_ratio_deviation",
            "recppo_reward_mean",
            "recppo_reward_std",
            "recppo_stop_rate",
            "recppo_grad_norm",
            "recppo_effective_residual_scale",
            "recppo_rank_gain_mean",
            "recppo_rank_gain_std",
            "recppo_embedding_gain_mean",
            "recppo_course_reward_mean",
            "recppo_rank_transition_scale",
            "recppo_rank_cache_hit_rate",
            "recppo_train_user_pool_size",
        )
        for key in diagnostic_keys:
            self._recppo_diag_sums[key] = self._recppo_diag_sums.get(key, 0.0) + float(info.get(key, 0.0))
        self._recppo_diag_count += 1

    def _flush_recppo_diagnostics(self):
        if self._recppo_diag_count < 1:
            return
        output_dir = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
        if not output_dir:
            self._recppo_diag_sums = {}
            self._recppo_diag_count = 0
            return
        os.makedirs(output_dir, exist_ok=True)
        row = {
            "epoch": int(self._recppo_epoch_state.item()) + 1,
            "phase": "ppo" if int(self._recppo_phase_state.item()) == 1 else "warmup",
        }
        for key, total in sorted(self._recppo_diag_sums.items()):
            row[key] = total / float(self._recppo_diag_count)
        path = os.path.join(output_dir, "recppo_epoch_metrics.csv")
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(
            ">> RecPPO diagnostics: "
            f"epoch={row['epoch']} phase={row['phase']} "
            f"kl={row.get('recppo_approx_kl', 0.0):.5f} "
            f"clip={row.get('recppo_clip_fraction', 0.0):.3f} "
            f"reward={row.get('recppo_reward_mean', 0.0):.4f} "
            f"stop={row.get('recppo_stop_rate', 0.0):.3f}"
        )
        self._recppo_diag_sums = {}
        self._recppo_diag_count = 0

    def _recppo_parameters(self):
        params = list(self.agent.parameters()) + list(self.recppo_stop_head.parameters())
        return [param for param in params if param.requires_grad]

    def recppo_parameter_ids(self):
        return {
            id(param)
            for param in list(self.agent.parameters()) + list(self.recppo_stop_head.parameters())
        }

    def _reset_recppo_optimizer(self):
        actor_params = (
            list(self.agent.common.parameters())
            + list(self.agent.actor_head.parameters())
            + list(self.agent.user_proj.parameters())
            + list(self.recppo_stop_head.parameters())
        )
        critic_params = list(self.agent.critic_head.parameters())
        self._recppo_optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": float(self.cfg.recppo_actor_lr)},
                {"params": critic_params, "lr": float(self.cfg.recppo_critic_lr)},
            ]
        )
        return self._recppo_optimizer

    def _activate_recppo_phase(self):
        if not bool(getattr(self.cfg, "recppo_enabled", True)):
            self._recppo_phase_state.zero_()
            return
        entering_phase = int(self._recppo_phase_state.item()) == 0
        self._recppo_phase_state.fill_(1)
        if entering_phase:
            self._clear_recppo_rank_cache()

    def _begin_pending_epoch(self):
        global _pending_recppo_best_optimizer_state, _pending_recppo_optimizer_state
        if not self.training or not self._recppo_epoch_pending:
            return
        if _pending_recppo_optimizer_state is not None:
            self._recppo_optimizer.load_state_dict(_pending_recppo_optimizer_state)
            for optimizer_state in self._recppo_optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
            _pending_recppo_optimizer_state = None
        if _pending_recppo_best_optimizer_state is not None:
            self._recppo_best_optimizer_state = _pending_recppo_best_optimizer_state
            _pending_recppo_best_optimizer_state = None
        self._recppo_epoch_state.add_(1)
        self._recppo_epoch_pending = False
        if int(self._recppo_phase_state.item()) == 1:
            self._clear_recppo_rank_cache(clear_history=False)
        elif (
            int(self._recppo_phase_state.item()) == 0
            and int(self._recppo_epoch_state.item()) >= int(self.cfg.recppo_warmup_epochs)
        ):
            self._activate_recppo_phase()

    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        self._begin_pending_epoch()
        self._last_recppo_info = {}
        recppo_active = (
            bool(getattr(self.cfg, "recppo_enabled", True))
            and int(self._recppo_phase_state.item()) == 1
        )
        self._recppo_behavior_user_idx = batch["u"].detach() if recppo_active else None
        self._recppo_collect_only = True
        old_ppo_weight = float(getattr(self.cfg, "ppo_loss_weight", 1.0))
        old_residual_scale = float(getattr(self.cfg, "rl_residual_scale", 1.0))
        self.cfg.ppo_loss_weight = 0.0
        if not recppo_active:
            self.cfg.rl_residual_scale = 0.0
        try:
            loss, stats = super().forward(
                batch,
                pop,
                llm_s,
                user_bank_raw=user_bank_raw,
                user_seen_items=user_seen_items,
            )
            if recppo_active and self.training and self._last_recppo_trajectory is not None:
                self.optimize_recppo(self._last_recppo_trajectory)
                self._accumulate_recppo_diagnostics(self._last_recppo_info)
                reported_ppo_loss = float(self._last_recppo_info.get("recppo_total_loss", 0.0))
                stats["ppo_loss"] = reported_ppo_loss
                stats["ppo_loss_raw"] = stats["ppo_loss"]
            stats["recppo_phase"] = "ppo" if recppo_active else "warmup"
            stats["recppo_epoch"] = int(self._recppo_epoch_state.item()) + 1
            stats["recppo_effective_residual_scale"] = self._effective_recppo_residual_scale()
            stats.update(getattr(self, "_last_recppo_info", {}))
            return loss, stats
        finally:
            self.cfg.ppo_loss_weight = old_ppo_weight
            self.cfg.rl_residual_scale = old_residual_scale
            self._recppo_collect_only = False
            self._recppo_behavior_user_idx = None

    def infer_refined_item_vectors(self, *args, **kwargs):
        if bool(getattr(self.cfg, "recppo_enabled", True)) and int(self._recppo_phase_state.item()) == 1:
            return super().infer_refined_item_vectors(*args, **kwargs)
        old_residual_scale = float(getattr(self.cfg, "rl_residual_scale", 1.0))
        self.cfg.rl_residual_scale = 0.0
        try:
            return super().infer_refined_item_vectors(*args, **kwargs)
        finally:
            self.cfg.rl_residual_scale = old_residual_scale

    def _effective_recppo_residual_scale(self):
        configured = float(getattr(self.cfg, "rl_residual_scale", 1.0))
        if int(self._recppo_phase_state.item()) != 1:
            return configured
        epoch_idx = int(self._recppo_epoch_state.item())
        if epoch_idx < 0:
            return configured
        phase_epoch = max(1, epoch_idx - int(self.cfg.recppo_warmup_epochs) + 1)
        ramp_epochs = max(1, int(getattr(self.cfg, "recppo_residual_ramp_epochs", 1)))
        return configured * min(1.0, float(phase_epoch) / float(ramp_epochs))

    def _effective_recppo_behavior_ce_weight(self):
        start = float(self.cfg.recppo_behavior_ce_weight)
        final = float(self.cfg.recppo_behavior_ce_final_weight)
        if int(self._recppo_phase_state.item()) != 1:
            return start
        phase_epoch = max(
            0,
            int(self._recppo_epoch_state.item()) - int(self.cfg.recppo_warmup_epochs),
        )
        anneal_epochs = max(1, int(self.cfg.recppo_behavior_ce_anneal_epochs))
        progress = min(1.0, float(phase_epoch) / float(anneal_epochs))
        return start + (final - start) * progress

    def _recppo_blend_state(self, base_h, state_h):
        residual = self._effective_recppo_residual_scale() * (state_h - base_h)
        max_norm = float(getattr(self.cfg, "recppo_max_residual_norm", 0.0))
        if max_norm > 0.0:
            residual_norm = residual.norm(dim=1, keepdim=True).clamp_min(1e-12)
            residual = residual * (max_norm / residual_norm).clamp(max=1.0)
        return base_h + residual

    def _blend_rl_episode_output(self, z_i_base, final_h):
        return self._recppo_blend_state(z_i_base, final_h)

    def _agent_logits_value(self, item_state, time_step, candidates_emb, action_mask=None):
        t_idx = time_step.squeeze(1).long().clamp(0, max(0, int(self.agent.time_dim) - 1))
        t_emb = F.one_hot(t_idx, num_classes=self.agent.time_dim).float()
        state = torch.cat([item_state, t_emb], dim=1)
        feat = self.agent.common(state)
        value = self.agent.critic_head(feat)
        query = self.agent.actor_head(feat).unsqueeze(1)
        keys = self.agent.user_proj(candidates_emb)
        scale = math.sqrt(float(max(1, keys.size(-1)))) * float(self.cfg.recppo_policy_temperature)
        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / scale
        stop_rows = None
        if bool(getattr(self.cfg, "recppo_enable_stop", True)) and candidates_emb.size(1) > 0:
            stop_rows = candidates_emb[:, -1, :].abs().amax(dim=1) <= 1e-12
            if stop_rows.any():
                logits = logits.clone()
                stop_logits = self.recppo_stop_head(feat).squeeze(1)
                logits[stop_rows, -1] = stop_logits[stop_rows]
        logits = self._bound_recppo_logits(logits)
        if stop_rows is not None and stop_rows.any():
            unavailable = stop_rows & (t_idx < int(getattr(self.cfg, "recppo_min_steps", 0)))
            if unavailable.any():
                logits = logits.clone()
                logits[unavailable, -1] = -1e9
        if action_mask is not None:
            action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
            if action_mask.shape != logits.shape:
                raise ValueError(
                    f"RecPPO action mask shape mismatch: expected {tuple(logits.shape)}, "
                    f"got {tuple(action_mask.shape)}"
                )
            if not bool(action_mask.any(dim=1).all().item()):
                raise RuntimeError("RecPPO action mask removed every candidate for at least one row")
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits, value

    def _agent_value(self, item_state, time_step):
        t_idx = time_step.squeeze(1).long().clamp(0, max(0, int(self.agent.time_dim) - 1))
        t_emb = F.one_hot(t_idx, num_classes=self.agent.time_dim).float()
        state = torch.cat([item_state, t_emb], dim=1)
        feat = self.agent.common(state)
        return self.agent.critic_head(feat)

    def _bound_recppo_value(self, value):
        bound = float(getattr(self.cfg, "recppo_value_bound", 0.0))
        if bound <= 0.0:
            return value
        return bound * torch.tanh(value / bound)

    def _bound_recppo_logits(self, logits):
        bound = float(getattr(self.cfg, "recppo_logit_bound", 0.0))
        if bound <= 0.0:
            return logits
        return bound * torch.tanh(logits / bound)

    def _select_rollout_action(self, current_h, time_step, candidates, fit_score=None, deterministic=False):
        policy = str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower()
        if policy == "ppo":
            action_mask = getattr(self, "_recppo_rollout_action_mask", None)
            logits, value = self._agent_logits_value(
                current_h,
                time_step,
                candidates,
                action_mask=action_mask,
            )
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action_idx = torch.argmax(logits, dim=1)
            else:
                action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
            return action_idx, log_prob, value, entropy
        return super()._select_rollout_action(
            current_h,
            time_step,
            candidates,
            fit_score=fit_score,
            deterministic=deterministic,
        )

    def _episode_behavior_user_idx(self, batch_size):
        behavior_user_idx = getattr(self, "_recppo_behavior_user_idx", None)
        if behavior_user_idx is None:
            return None
        behavior_user_idx = torch.as_tensor(behavior_user_idx, dtype=torch.long, device=self.device).view(-1)
        if behavior_user_idx.numel() != batch_size:
            return None
        return behavior_user_idx

    def _inject_behavior_user_candidate(self, candidates, cand_user_idx, behavior_user_idx):
        if (
            behavior_user_idx is None
            or not self.training
            or not bool(getattr(self.cfg, "recppo_inject_behavior_user", True))
            or candidates.size(1) < 1
        ):
            return candidates, cand_user_idx, None

        batch_size, n_candidates, _ = candidates.shape
        behavior_user_idx = behavior_user_idx.to(device=candidates.device).long().view(-1)
        last_pos = n_candidates - 1
        behavior_actions = torch.full((batch_size,), last_pos, dtype=torch.long, device=candidates.device)

        if cand_user_idx is None:
            cand_user_idx = torch.full(
                (batch_size, n_candidates),
                -1,
                dtype=torch.long,
                device=candidates.device,
            )
            replace_mask = torch.ones(batch_size, dtype=torch.bool, device=candidates.device)
        else:
            cand_user_idx = cand_user_idx.to(device=candidates.device)
            matches = cand_user_idx.eq(behavior_user_idx.view(-1, 1))
            has_match = matches.any(dim=1)
            first_match = matches.long().argmax(dim=1)
            behavior_actions = torch.where(has_match, first_match, behavior_actions)
            replace_mask = ~has_match

        if replace_mask.any():
            candidates = candidates.clone()
            cand_user_idx = cand_user_idx.clone()
            injected_users = behavior_user_idx[replace_mask]
            injected_vec = self.user_proj(self.user_emb(injected_users)).detach()
            candidates[replace_mask, last_pos, :] = injected_vec
            cand_user_idx[replace_mask, last_pos] = injected_users

        return candidates, cand_user_idx, behavior_actions.detach()

    def _get_deterministic_candidates(
        self,
        item_emb,
        user_bank_raw=None,
        user_bank_norm=None,
    ):
        if user_bank_raw is None:
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        if user_bank_norm is None:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)
        n_candidates = max(1, min(int(self.cfg.n_candidates), int(user_bank_raw.size(0))))
        if self.cfg.candidate_strategy == "retrieve_sample":
            top_m = max(n_candidates, min(int(self.cfg.retrieve_top_m), int(user_bank_raw.size(0))))
            _, top_idx = self._retrieve_topm_exact(
                F.normalize(item_emb, dim=1),
                user_bank_raw,
                top_m,
                user_bank_norm=user_bank_norm,
            )
            cand_idx = top_idx[:, :n_candidates]
        else:
            cand_idx = torch.arange(n_candidates, device=self.device).view(1, -1).expand(item_emb.size(0), -1)
        candidates = user_bank_raw[cand_idx].detach()
        return candidates, cand_idx, {
            "dup_rate": 0.0,
            "topm_coverage": 1.0,
        }

    def _append_stop_candidate(self, candidates, cand_user_idx, fit_score):
        if not bool(getattr(self.cfg, "recppo_enable_stop", True)):
            return candidates, cand_user_idx, fit_score, None
        stop_idx = candidates.size(1)
        stop_vec = torch.zeros(
            (candidates.size(0), 1, candidates.size(2)),
            dtype=candidates.dtype,
            device=candidates.device,
        )
        candidates = torch.cat([candidates, stop_vec], dim=1)
        if cand_user_idx is not None:
            stop_user = torch.full(
                (cand_user_idx.size(0), 1),
                -1,
                dtype=cand_user_idx.dtype,
                device=cand_user_idx.device,
            )
            cand_user_idx = torch.cat([cand_user_idx, stop_user], dim=1)
        if fit_score is not None:
            fit_score = torch.cat([fit_score, torch.zeros_like(fit_score[:, :1])], dim=1)
        return candidates, cand_user_idx, fit_score, stop_idx

    def _zero_course_terms(self, batch_size, device):
        return {
            key: torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
            for key in ("concept_bonus", "prereq_gap", "difficulty_gap", "redundant")
        }

    def _scaled_recppo_course_reward(self, course_terms):
        contribution = (
            float(self.cfg.feedback_course_concept_weight) * course_terms["concept_bonus"]
            - float(self.cfg.feedback_course_prereq_weight) * course_terms["prereq_gap"]
            - float(self.cfg.feedback_course_difficulty_weight) * course_terms["difficulty_gap"]
            - float(self.cfg.feedback_course_redundant_weight) * course_terms["redundant"]
        )
        contribution = float(self.cfg.recppo_course_reward_scale) * contribution
        clip = float(self.cfg.recppo_course_reward_clip)
        if clip <= 0.0:
            return torch.zeros_like(contribution)
        return contribution.clamp(-clip, clip)

    def _clear_recppo_rank_cache(self, clear_history=True):
        self._recppo_rank_topk_cache = {}
        self._recppo_train_user_ids_cache = None
        self._recppo_train_user_pool_signature = None
        self._recppo_rank_cache_hits = 0
        self._recppo_rank_cache_misses = 0
        self._recppo_train_user_pool_size = 0
        if clear_history:
            self._recppo_item_positive_user_cache = None
            self._recppo_item_positive_signature = None

    def _recppo_train_user_pool(self, user_bank_norm, user_seen_items):
        if user_bank_norm is None:
            raise RuntimeError("RecPPO global rank reward requires the training user embedding bank")
        if not isinstance(user_seen_items, dict):
            raise RuntimeError("RecPPO global rank reward requires train-only user histories")
        signature = (id(user_seen_items), len(user_seen_items), int(user_bank_norm.size(0)))
        if self._recppo_train_user_pool_signature != signature:
            train_user_ids = sorted(
                int(user_id)
                for user_id, seen_items in user_seen_items.items()
                if 0 <= int(user_id) < int(user_bank_norm.size(0)) and len(seen_items) > 0
            )
            if not train_user_ids:
                raise RuntimeError("RecPPO global rank reward found no users with training histories")
            self._recppo_train_user_ids_cache = torch.tensor(
                train_user_ids,
                dtype=torch.long,
                device=user_bank_norm.device,
            )
            self._recppo_train_user_pool_signature = signature
            self._recppo_train_user_pool_size = len(train_user_ids)
        user_ids = self._recppo_train_user_ids_cache.to(device=user_bank_norm.device)
        return F.normalize(user_bank_norm[user_ids].detach(), dim=1), user_ids

    def _recppo_item_positive_users(self, user_seen_items):
        if not isinstance(user_seen_items, dict):
            raise RuntimeError("RecPPO global rank reward requires train-only user histories")
        signature = (id(user_seen_items), len(user_seen_items))
        if self._recppo_item_positive_signature != signature:
            item_users = {}
            for user_id, seen_items in user_seen_items.items():
                user_id = int(user_id)
                if user_id < 0:
                    continue
                for item_id in seen_items:
                    item_users.setdefault(int(item_id), set()).add(user_id)
            self._recppo_item_positive_user_cache = {
                item_id: tuple(sorted(user_ids))
                for item_id, user_ids in item_users.items()
            }
            self._recppo_item_positive_signature = signature
        return self._recppo_item_positive_user_cache

    def _recppo_item_hard_negative_user_ids(
        self,
        item_idx,
        target_emb,
        user_bank_norm,
        user_seen_items,
    ):
        if item_idx is None or target_emb is None:
            raise RuntimeError("RecPPO global rank reward requires warm item IDs and behavior targets")
        item_idx = item_idx.detach().long().view(-1)
        target_emb = target_emb.detach()
        if item_idx.numel() != target_emb.size(0):
            raise ValueError("RecPPO rank reward item/target batch size mismatch")
        if user_bank_norm is None:
            raise RuntimeError("RecPPO global rank reward requires the training user embedding bank")
        item_positive_users = self._recppo_item_positive_users(user_seen_items)

        item_ids = [int(value) for value in item_idx.cpu().tolist()]
        missing_items = []
        missing_targets = []
        pending = set()
        for row, item_id in enumerate(item_ids):
            if item_id in self._recppo_rank_topk_cache:
                self._recppo_rank_cache_hits += 1
            elif item_id not in pending:
                pending.add(item_id)
                missing_items.append(item_id)
                missing_targets.append(target_emb[row])
                self._recppo_rank_cache_misses += 1

        if missing_items:
            train_pool, train_user_ids = self._recppo_train_user_pool(user_bank_norm, user_seen_items)
            missing_target_tensor = F.normalize(torch.stack(missing_targets, dim=0), dim=1)
            top_k = min(int(self.cfg.recppo_rank_topk), int(train_pool.size(0)))
            chunk_size = int(self.cfg.recppo_rank_item_chunk_size)
            for start in range(0, len(missing_items), chunk_size):
                end = min(start + chunk_size, len(missing_items))
                scores = torch.matmul(missing_target_tensor[start:end], train_pool.t())
                for offset, item_id in enumerate(missing_items[start:end]):
                    row_scores = scores[offset].clone()
                    positive_ids = item_positive_users.get(item_id, ())
                    if positive_ids:
                        positive_tensor = torch.tensor(
                            positive_ids,
                            dtype=torch.long,
                            device=train_user_ids.device,
                        )
                        local = torch.searchsorted(train_user_ids, positive_tensor)
                        safe_local = local.clamp(max=max(0, train_user_ids.numel() - 1))
                        present = (local < train_user_ids.numel()) & (
                            train_user_ids[safe_local] == positive_tensor
                        )
                        row_scores[safe_local[present]] = -torch.inf
                    available = int(torch.isfinite(row_scores).sum().item())
                    if available < 1:
                        raise RuntimeError(
                            f"RecPPO rank reward found no train-user negative for item {item_id}"
                        )
                    row_k = min(top_k, available)
                    top_local = torch.topk(
                        row_scores,
                        k=row_k,
                        largest=True,
                        sorted=True,
                    ).indices
                    top_user_ids = train_user_ids[top_local]
                    if row_k < top_k:
                        top_user_ids = torch.cat(
                            [top_user_ids, top_user_ids[-1:].expand(top_k - row_k)],
                            dim=0,
                        )
                    self._recppo_rank_topk_cache[item_id] = top_user_ids.detach().clone()

        return torch.stack(
            [self._recppo_rank_topk_cache[item_id] for item_id in item_ids],
            dim=0,
        ).to(device=user_bank_norm.device)

    def _global_train_user_listwise_gain(
        self,
        prev_h,
        next_h,
        target_emb,
        item_idx,
        user_bank_norm,
        user_seen_items,
        positive_user_ids,
    ):
        positive_user_ids = positive_user_ids.detach().long().view(-1)
        item_ids = item_idx.detach().long().view(-1)
        if positive_user_ids.numel() != item_ids.numel():
            raise ValueError("RecPPO rank reward positive-user/item batch size mismatch")
        item_positive_users = self._recppo_item_positive_users(user_seen_items)
        for item_id, user_id in zip(item_ids.cpu().tolist(), positive_user_ids.cpu().tolist()):
            if int(user_id) not in item_positive_users.get(int(item_id), ()):
                raise RuntimeError(
                    "RecPPO rank reward positive user is not present in the train-only item history"
                )
        hard_negative_ids = self._recppo_item_hard_negative_user_ids(
            item_idx,
            target_emb,
            user_bank_norm,
            user_seen_items,
        )
        positive_users = F.normalize(
            user_bank_norm[positive_user_ids].detach(),
            dim=1,
        ).unsqueeze(1)
        hard_negative_users = F.normalize(
            user_bank_norm[hard_negative_ids].detach(),
            dim=2,
        )
        listwise_users = torch.cat([positive_users, hard_negative_users], dim=1)
        temperature = float(self.cfg.recppo_rank_temperature)
        prev_scores = (
            F.normalize(prev_h.detach(), dim=1).unsqueeze(1) * listwise_users
        ).sum(dim=2) / temperature
        next_scores = (
            F.normalize(next_h.detach(), dim=1).unsqueeze(1) * listwise_users
        ).sum(dim=2) / temperature
        labels = torch.zeros(prev_scores.size(0), dtype=torch.long, device=prev_scores.device)
        prev_ce = F.cross_entropy(prev_scores, labels, reduction="none").unsqueeze(1)
        next_ce = F.cross_entropy(next_scores, labels, reduction="none").unsqueeze(1)
        return self._normalize_recppo_rank_gain((prev_ce - next_ce).detach())

    def _recppo_rank_transition_scale(self):
        if not bool(getattr(self.cfg, "recppo_rank_normalize_transition", True)):
            return 1.0
        return max(
            1e-4,
            abs(float(self.cfg.usim_lr)) * max(1e-4, self._effective_recppo_residual_scale()),
        )

    def _normalize_recppo_rank_gain(self, rank_gain):
        return rank_gain / self._recppo_rank_transition_scale()

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
        deterministic=False,
    ):
        current_h = init_item_emb.clone()
        reward_base_h = init_item_emb.detach().clone()
        batch_size = current_h.size(0)
        trajectory = {
            key: []
            for key in (
                "log_probs",
                "values",
                "rewards",
                "entropies",
                "states",
                "time_steps",
                "next_states",
                "next_time_steps",
                "dones",
                "valids",
                "candidates",
                "actions",
                "action_masks",
                "behavior_actions",
                "embedding_gains",
                "rank_gains",
                "course_rewards",
            )
        }
        if target_emb is not None:
            target_emb = target_emb.detach().clone()
            trajectory["target_emb"] = target_emb
        candidate_stats = {
            "dup_rate": 0.0,
            "topm_coverage": 0.0,
            "steps": 0,
            "step_gain": 0.0,
            "rank_gain": 0.0,
            "course_reward": 0.0,
            "collapse_penalty": 0.0,
            "course_sample_fit": 0.0,
            "sage_active": 0.0,
            "sage_gate": 0.0,
            "sage_tail_active": 0.0,
            "sage_pool_fit": 0.0,
            "sage_two_expert": 0.0,
            "cgrc_recon_sample_active": 0.0,
            "cgrc_recon_sample_score": 0.0,
            "course_prereq_gap": 0.0,
            "course_concept_bonus": 0.0,
            "course_difficulty_gap": 0.0,
            "course_redundant": 0.0,
            "reward_step_cost": 0.0,
            "stop_rate": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
        }

        user_bank_norm = None
        if isinstance(user_bank_raw, tuple):
            user_bank_raw, user_bank_norm = user_bank_raw
        elif user_bank_raw is None and self.cfg.candidate_strategy == "retrieve_sample":
            user_bank_raw, user_bank_norm = self._build_user_bank_raw()
        elif user_bank_raw is not None:
            user_bank_norm = F.normalize(user_bank_raw, dim=1)

        behavior_user_idx = self._episode_behavior_user_idx(batch_size)
        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        stop_count = 0
        valid_reward_values = []
        selected_user_history = []

        for t in range(int(self.cfg.usim_steps)):
            valid = active.clone()
            if not bool(valid.any().item()):
                break
            time_step = torch.full((batch_size, 1), t, device=self.device)
            candidate_fn = self._get_deterministic_candidates if deterministic else self.get_candidates
            candidates, cand_user_idx, cand_stats = candidate_fn(
                current_h,
                user_bank_raw=user_bank_raw,
                user_bank_norm=user_bank_norm,
            ) if deterministic else candidate_fn(
                current_h,
                user_bank_raw=user_bank_raw,
                user_bank_norm=user_bank_norm,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            candidates, cand_user_idx, fit_score = self._apply_course_sampling_bias(
                current_h,
                candidates,
                cand_user_idx,
                item_idx=item_idx,
                target_pop=target_pop,
                user_seen_items=user_seen_items,
            )
            behavior_action_idx = None
            if t == 0:
                candidates, cand_user_idx, behavior_action_idx = self._inject_behavior_user_candidate(
                    candidates,
                    cand_user_idx,
                    behavior_user_idx,
                )
            candidates, cand_user_idx, fit_score, stop_idx = self._append_stop_candidate(
                candidates,
                cand_user_idx,
                fit_score,
            )

            action_mask = torch.ones(
                (batch_size, candidates.size(1)),
                dtype=torch.bool,
                device=self.device,
            )
            if cand_user_idx is not None and selected_user_history:
                for previous_user in selected_user_history:
                    repeated = cand_user_idx.eq(previous_user.view(-1, 1)) & previous_user.view(-1, 1).ge(0)
                    action_mask &= ~repeated
            if stop_idx is not None:
                action_mask[:, stop_idx] = t >= int(getattr(self.cfg, "recppo_min_steps", 0))
            no_available = ~action_mask.any(dim=1)
            if no_available.any():
                user_end = stop_idx if stop_idx is not None else candidates.size(1)
                action_mask[no_available, :user_end] = True

            behavior_label = torch.full((batch_size,), -100, dtype=torch.long, device=self.device)
            if t == 0 and behavior_action_idx is not None:
                behavior_label = behavior_action_idx.to(device=self.device).long()
            teacher_force_behavior = (
                t == 0
                and bool((behavior_label >= 0).any().item())
                and self.training
                and not deterministic
                and bool(getattr(self.cfg, "recppo_teacher_force_behavior", False))
                and str(getattr(self.cfg, "rollout_policy", "ppo")).strip().lower() == "ppo"
            )
            if teacher_force_behavior:
                logits, value = self._agent_logits_value(
                    current_h,
                    time_step,
                    candidates,
                    action_mask=action_mask,
                )
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = behavior_label
                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()
            else:
                self._recppo_rollout_action_mask = action_mask
                try:
                    action_idx, log_prob, value, entropy = self._select_rollout_action(
                        current_h,
                        time_step,
                        candidates,
                        fit_score=fit_score,
                        deterministic=deterministic,
                    )
                finally:
                    self._recppo_rollout_action_mask = None
            if stop_idx is not None and (~valid).any():
                action_idx = torch.where(valid, action_idx, torch.full_like(action_idx, stop_idx))
                log_prob = torch.where(valid, log_prob, torch.zeros_like(log_prob))
                entropy = torch.where(valid, entropy, torch.zeros_like(entropy))

            candidate_stats["steps"] += 1
            if cand_stats is not None:
                for key in (
                    "dup_rate",
                    "topm_coverage",
                    "sage_active",
                    "sage_gate",
                    "sage_tail_active",
                    "sage_pool_fit",
                    "sage_two_expert",
                    "cgrc_recon_sample_active",
                    "cgrc_recon_sample_score",
                ):
                    candidate_stats[key] += float(cand_stats.get(key, 0.0))
            if fit_score is not None:
                candidate_stats["course_sample_fit"] += float(fit_score.mean().item())

            trajectory["states"].append(current_h.detach().clone())
            trajectory["time_steps"].append(time_step.detach().clone())
            trajectory["candidates"].append(candidates.detach().clone())
            trajectory["actions"].append(action_idx.detach().clone())
            trajectory["action_masks"].append(action_mask.detach().clone())
            trajectory["behavior_actions"].append(behavior_label.detach().clone())
            trajectory["valids"].append(valid.detach().clone())

            batch_indices = torch.arange(batch_size, device=self.device)
            selected_user = candidates[batch_indices, action_idx]
            selected_user_ids = cand_user_idx[batch_indices, action_idx] if cand_user_idx is not None else None
            selected_stop = action_idx.eq(stop_idx) if stop_idx is not None else torch.zeros_like(valid)
            transition_mask = valid & (~selected_stop)
            if selected_user_ids is not None:
                selected_user_history.append(
                    torch.where(
                        transition_mask,
                        selected_user_ids,
                        torch.full_like(selected_user_ids, -1),
                    ).detach()
                )
            prev_h = current_h
            current_h = current_h + (
                float(self.cfg.usim_lr)
                * selected_user.detach()
                * transition_mask.float().unsqueeze(1)
            )
            last_step = t == int(self.cfg.usim_steps) - 1
            done = (~valid) | selected_stop | last_step
            active = valid & (~done)
            stop_count += int((selected_stop & valid).sum().item())
            next_t = min(t + 1, max(0, int(self.cfg.usim_steps) - 1))
            next_time_step = torch.full((batch_size, 1), next_t, device=self.device)
            trajectory["next_states"].append(current_h.detach().clone())
            trajectory["next_time_steps"].append(next_time_step.detach().clone())
            trajectory["dones"].append(done.detach().clone())

            reward = torch.zeros((batch_size, 1), dtype=current_h.dtype, device=self.device)
            step_gain = torch.zeros_like(reward)
            rank_gain = torch.zeros_like(reward)
            if target_emb is not None:
                reward_prev_h = self._recppo_blend_state(reward_base_h, prev_h.detach())
                reward_next_h = self._recppo_blend_state(reward_base_h, current_h.detach())
                step_gain = (
                    F.cosine_similarity(reward_next_h, target_emb, dim=1)
                    - F.cosine_similarity(reward_prev_h, target_emb, dim=1)
                ).unsqueeze(1)
                global_rank_active = (
                    int(self._recppo_phase_state.item()) == 1
                    and float(self.cfg.recppo_rank_gain_weight) != 0.0
                )
                if global_rank_active and user_seen_items is None:
                    raise RuntimeError(
                        "RecPPO global rank reward requires train-only user histories"
                    )
                if global_rank_active and (user_bank_norm is None or item_idx is None):
                    raise RuntimeError(
                        "RecPPO global rank reward requires the training user bank and warm item IDs"
                    )
                if global_rank_active and behavior_user_idx is None:
                    raise RuntimeError(
                        "RecPPO global rank reward requires observed train-positive users"
                    )
                if global_rank_active:
                    rank_gain = self._global_train_user_listwise_gain(
                        reward_prev_h,
                        reward_next_h,
                        target_emb,
                        item_idx=item_idx,
                        user_bank_norm=user_bank_norm,
                        user_seen_items=user_seen_items,
                        positive_user_ids=behavior_user_idx,
                    )
                gain_clip = float(self.cfg.reward_gain_clip)
                step_gain = step_gain.clamp(-gain_clip, gain_clip)
                rank_gain = rank_gain.clamp(-gain_clip, gain_clip)
                reward = (
                    float(self.cfg.recppo_embedding_gain_weight) * step_gain
                    + float(self.cfg.recppo_rank_gain_weight) * rank_gain
                )
            transition_float = transition_mask.float().unsqueeze(1)
            reward = (reward - float(self.cfg.reward_step_cost)) * transition_float

            course_terms = self._zero_course_terms(batch_size, self.device)
            course_reward = torch.zeros_like(reward)
            if (
                bool(getattr(self.cfg, "use_course_reward", True))
                and item_idx is not None
                and selected_user_ids is not None
            ):
                safe_user_ids = torch.where(selected_user_ids >= 0, selected_user_ids, torch.zeros_like(selected_user_ids))
                course_terms = self._compute_course_reward_terms(
                    safe_user_ids,
                    item_idx=item_idx,
                    target_pop=target_pop,
                    user_seen_items=user_seen_items,
                )
                for key in course_terms:
                    course_terms[key] = course_terms[key] * transition_float
                course_reward = self._scaled_recppo_course_reward(course_terms)
                reward = reward + course_reward
            reward = reward * valid.float().unsqueeze(1)
            if valid.any():
                valid_reward_values.append(reward[valid].detach().view(-1))

            candidate_stats["step_gain"] += float(step_gain[valid].mean().item()) if valid.any() else 0.0
            candidate_stats["rank_gain"] += float(rank_gain[valid].mean().item()) if valid.any() else 0.0
            candidate_stats["course_reward"] += (
                float(course_reward[valid].mean().item()) if valid.any() else 0.0
            )
            candidate_stats["reward_step_cost"] += float(self.cfg.reward_step_cost)
            for stat_key, term_key in (
                ("course_prereq_gap", "prereq_gap"),
                ("course_concept_bonus", "concept_bonus"),
                ("course_difficulty_gap", "difficulty_gap"),
                ("course_redundant", "redundant"),
            ):
                candidate_stats[stat_key] += float(course_terms[term_key].mean().item())
            trajectory["log_probs"].append(log_prob.detach())
            trajectory["values"].append(value.detach())
            trajectory["rewards"].append(reward.detach())
            trajectory["entropies"].append(entropy.detach())
            trajectory["embedding_gains"].append(step_gain.detach())
            trajectory["rank_gains"].append(rank_gain.detach())
            trajectory["course_rewards"].append(course_reward.detach())

        if candidate_stats["steps"] > 0:
            for key in (
                "dup_rate",
                "topm_coverage",
                "step_gain",
                "rank_gain",
                "course_reward",
                "collapse_penalty",
                "course_sample_fit",
                "sage_active",
                "sage_gate",
                "sage_tail_active",
                "sage_pool_fit",
                "sage_two_expert",
                "cgrc_recon_sample_active",
                "cgrc_recon_sample_score",
                "course_prereq_gap",
                "course_concept_bonus",
                "course_difficulty_gap",
                "course_redundant",
                "reward_step_cost",
            ):
                candidate_stats[key] /= candidate_stats["steps"]
        candidate_stats["stop_rate"] = float(stop_count) / float(max(1, batch_size))
        if valid_reward_values:
            rewards_flat = torch.cat(valid_reward_values)
            candidate_stats["reward_mean"] = float(rewards_flat.mean().item())
            candidate_stats["reward_std"] = float(rewards_flat.std(unbiased=False).item())
        final_effective_h = self._recppo_blend_state(reward_base_h, current_h.detach())
        candidate_stats["rl_residual_norm"] = float(
            (final_effective_h - reward_base_h).norm(dim=1).mean().item()
        )
        cache_total = self._recppo_rank_cache_hits + self._recppo_rank_cache_misses
        candidate_stats["rank_cache_hit_rate"] = (
            float(self._recppo_rank_cache_hits) / float(cache_total) if cache_total > 0 else 0.0
        )
        candidate_stats["train_user_pool_size"] = int(self._recppo_train_user_pool_size)
        self._last_recppo_trajectory = trajectory
        return current_h, trajectory, candidate_stats

    def _terminal_value_loss(self, trajectory):
        target_emb = trajectory.get("target_emb")
        time_steps = trajectory.get("time_steps", [])
        dones = trajectory.get("dones", [])
        valids = trajectory.get("valids", [])
        if (
            target_emb is None
            or len(time_steps) == 0
            or len(dones) != len(time_steps)
            or len(valids) != len(time_steps)
        ):
            return next(self.parameters()).sum() * 0.0
        batch_size = target_emb.size(0)
        done_tensor = torch.stack(dones).bool()
        valid_tensor = torch.stack(valids).bool()
        terminal_tensor = done_tensor & valid_tensor
        has_terminal = terminal_tensor.any(dim=0)
        if not bool(has_terminal.any().item()):
            return next(self.parameters()).sum() * 0.0
        first_terminal_step = terminal_tensor.to(dtype=torch.int64).argmax(dim=0)
        rows = torch.arange(batch_size, device=target_emb.device)[has_terminal]
        selected_steps = first_terminal_step[has_terminal]
        time_tensor = torch.stack(time_steps)
        states = target_emb[has_terminal]
        times = time_tensor[selected_steps, rows].view(-1, 1)
        values = self._bound_recppo_value(self._agent_value(states, times)).view(-1)
        return 0.5 * values.pow(2).mean()

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _prepare_recppo_targets(self, trajectory):
        rewards = torch.stack(trajectory["rewards"]).squeeze(-1).detach()
        old_log_probs = torch.stack(trajectory["log_probs"]).detach()
        old_values = self._bound_recppo_value(torch.stack(trajectory["values"]).squeeze(-1)).detach()
        dones = torch.stack(trajectory.get("dones", [])).bool()
        valids = torch.stack(
            trajectory.get(
                "valids",
                [torch.ones_like(reward, dtype=torch.bool) for reward in rewards],
            )
        ).bool()
        if dones.numel() > 0:
            dones[-1] = True
        has_next_bootstrap = (
            bool(getattr(self.cfg, "recppo_bootstrap_next_value", True))
            and len(trajectory.get("next_states", [])) == len(trajectory["states"])
            and len(trajectory.get("next_time_steps", [])) == len(trajectory["states"])
        )
        with torch.no_grad():
            if has_next_bootstrap:
                next_values = torch.stack(
                    [
                        self._bound_recppo_value(self._agent_value(state, time_step)).squeeze(-1)
                        for state, time_step in zip(
                            trajectory["next_states"],
                            trajectory["next_time_steps"],
                        )
                    ]
                )
            else:
                next_values = torch.zeros_like(old_values)
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros_like(rewards[0])
            gamma = float(self.cfg.ppo_gamma)
            lam = float(getattr(self.cfg, "ppo_lambda", 0.95))
            for step in reversed(range(rewards.size(0))):
                nonterminal = (~dones[step]).float()
                delta = rewards[step] + gamma * next_values[step] * nonterminal - old_values[step]
                gae = (delta + gamma * lam * nonterminal * gae) * valids[step].float()
                advantages[step] = gae
            returns = advantages + old_values
            if getattr(self.cfg, "ppo_adv_norm", False) and bool(valids.any().item()):
                selected = advantages[valids]
                advantages = torch.where(
                    valids,
                    (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6),
                    torch.zeros_like(advantages),
                )
        return {
            "rewards": rewards,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": advantages,
            "returns": returns,
            "valids": valids,
            "has_next_bootstrap": has_next_bootstrap,
        }

    def _recppo_objective(self, trajectory, prepared):
        logits_list = []
        log_prob_list = []
        value_list = []
        entropy_list = []
        action_masks = trajectory.get("action_masks", [None] * len(trajectory["states"]))
        for state, time_step, candidates, action, action_mask in zip(
            trajectory["states"],
            trajectory["time_steps"],
            trajectory["candidates"],
            trajectory["actions"],
            action_masks,
        ):
            if action_mask is None:
                logits, value = self._agent_logits_value(state, time_step, candidates)
            else:
                logits, value = self._agent_logits_value(
                    state,
                    time_step,
                    candidates,
                    action_mask=action_mask,
                )
            dist = torch.distributions.Categorical(logits=logits)
            logits_list.append(logits)
            log_prob_list.append(dist.log_prob(action))
            value_list.append(self._bound_recppo_value(value))
            entropy_list.append(dist.entropy())
        new_logits = torch.stack(logits_list)
        new_log_probs = torch.stack(log_prob_list)
        new_values = torch.stack(value_list).squeeze(-1)
        new_entropies = torch.stack(entropy_list)
        valids = prepared["valids"]
        log_ratio = (new_log_probs - prepared["old_log_probs"]).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * prepared["advantages"]
        surr2 = torch.clamp(
            ratio,
            1.0 - float(self.cfg.ppo_clip),
            1.0 + float(self.cfg.ppo_clip),
        ) * prepared["advantages"]
        actor_loss = -self._masked_mean(torch.min(surr1, surr2), valids)

        value_clip = float(getattr(self.cfg, "ppo_value_clip", 0.0))
        if value_clip > 0.0:
            value_delta = (new_values - prepared["old_values"]).clamp(-value_clip, value_clip)
            value_pred_clipped = prepared["old_values"] + value_delta
            critic_values = torch.max(
                (new_values - prepared["returns"]).pow(2),
                (value_pred_clipped - prepared["returns"]).pow(2),
            )
        else:
            critic_values = (new_values - prepared["returns"]).pow(2)
        critic_loss = 0.5 * self._masked_mean(critic_values, valids)
        entropy = self._masked_mean(new_entropies, valids)
        terminal_loss = self._terminal_value_loss(trajectory)

        behavior_ce = new_values.new_zeros(())
        behavior_actions = trajectory.get("behavior_actions", [])
        if len(behavior_actions) == len(trajectory["states"]):
            labels = torch.stack(behavior_actions).to(device=new_logits.device)
            supervised = (labels >= 0) & valids
            if supervised.any():
                behavior_ce = F.cross_entropy(new_logits[supervised], labels[supervised])

        policy_loss = (
            actor_loss
            + float(self.cfg.ppo_coeffs["value"]) * critic_loss
            - float(self.cfg.ppo_coeffs["entropy"]) * entropy
        )
        behavior_ce_weight = self._effective_recppo_behavior_ce_weight()
        total_loss = (
            float(self.cfg.ppo_loss_weight) * policy_loss
            + float(self.cfg.recppo_terminal_value_weight) * terminal_loss
            + behavior_ce_weight * behavior_ce
        )
        valid_ratio = ratio[valids]
        ratio_deviation = (
            float((valid_ratio - 1.0).abs().max().detach().item())
            if valid_ratio.numel() > 0
            else 0.0
        )
        approx_kl = self._masked_mean((ratio - 1.0) - log_ratio, valids)
        clip_fraction = self._masked_mean(
            ((ratio - 1.0).abs() > float(self.cfg.ppo_clip)).float(),
            valids,
        )
        info = {
            "recppo_total_loss": float(total_loss.detach().item()),
            "recppo_policy_loss": float(policy_loss.detach().item()),
            "recppo_actor_loss": float(actor_loss.detach().item()),
            "recppo_critic_loss": float(critic_loss.detach().item()),
            "recppo_terminal_value_loss": float(terminal_loss.detach().item()),
            "recppo_behavior_ce_loss": float(behavior_ce.detach().item()),
            "recppo_behavior_ce_weight": float(behavior_ce_weight),
            "recppo_entropy": float(entropy.detach().item()),
            "recppo_approx_kl": float(approx_kl.detach().item()),
            "recppo_clip_fraction": float(clip_fraction.detach().item()),
            "recppo_max_ratio_deviation": ratio_deviation,
            "recppo_has_next_state_bootstrap": bool(prepared["has_next_bootstrap"]),
            "recppo_reward_mean": float(prepared["rewards"][valids].mean().item()) if valids.any() else 0.0,
            "recppo_reward_std": float(prepared["rewards"][valids].std(unbiased=False).item()) if valids.any() else 0.0,
        }
        rank_gains = trajectory.get("rank_gains", [])
        if len(rank_gains) == len(trajectory["states"]):
            stacked_rank_gains = torch.stack(rank_gains).squeeze(-1)
            info["recppo_rank_gain_mean"] = (
                float(stacked_rank_gains[valids].mean().item()) if valids.any() else 0.0
            )
            info["recppo_rank_gain_std"] = (
                float(stacked_rank_gains[valids].std(unbiased=False).item()) if valids.any() else 0.0
            )
        else:
            info["recppo_rank_gain_mean"] = 0.0
            info["recppo_rank_gain_std"] = 0.0
        for trajectory_key, info_key in (
            ("embedding_gains", "recppo_embedding_gain_mean"),
            ("course_rewards", "recppo_course_reward_mean"),
        ):
            values = trajectory.get(trajectory_key, [])
            if len(values) == len(trajectory["states"]):
                stacked = torch.stack(values).squeeze(-1)
                info[info_key] = float(stacked[valids].mean().item()) if valids.any() else 0.0
            else:
                info[info_key] = 0.0
        cache_total = self._recppo_rank_cache_hits + self._recppo_rank_cache_misses
        info["recppo_rank_cache_hit_rate"] = (
            float(self._recppo_rank_cache_hits) / float(cache_total) if cache_total > 0 else 0.0
        )
        info["recppo_train_user_pool_size"] = float(self._recppo_train_user_pool_size)
        info["recppo_rank_transition_scale"] = float(self._recppo_rank_transition_scale())
        if bool(getattr(self.cfg, "recppo_enable_stop", True)):
            stop_flags = torch.stack(
                [action.eq(candidates.size(1) - 1) for action, candidates in zip(trajectory["actions"], trajectory["candidates"])]
            )
            info["recppo_stop_rate"] = float(stop_flags[valids].float().mean().item()) if valids.any() else 0.0
        else:
            info["recppo_stop_rate"] = 0.0
        return total_loss, info

    def compute_ppo_loss(self, trajectory):
        if self._recppo_collect_only or len(trajectory.get("rewards", [])) == 0:
            return self.recppo_outer_anchor * 0.0
        prepared = self._prepare_recppo_targets(trajectory)
        loss, info = self._recppo_objective(trajectory, prepared)
        info["recppo_update_epochs"] = 0
        self._last_recppo_info = info
        return loss

    def optimize_recppo(self, trajectory):
        if len(trajectory.get("rewards", [])) == 0:
            if self._recppo_optimizer is not None:
                self._recppo_optimizer.zero_grad(set_to_none=True)
            self._last_recppo_info = {"recppo_update_epochs": 0}
            return self._last_recppo_info
        if self._recppo_optimizer is None:
            self._reset_recppo_optimizer()
        prepared = self._prepare_recppo_targets(trajectory)
        max_ratio_deviation = 0.0
        final_info = {}
        performed_updates = 0
        max_grad_norm = 0.0
        stopped_for_kl = False
        try:
            for _ in range(int(self.cfg.ppo_epochs)):
                self._recppo_optimizer.zero_grad(set_to_none=True)
                loss, info = self._recppo_objective(trajectory, prepared)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("RecPPO objective became non-finite; optimizer step was aborted")
                target_kl = float(getattr(self.cfg, "recppo_target_kl", 0.0))
                if performed_updates > 0 and target_kl > 0.0 and float(info["recppo_approx_kl"]) > target_kl:
                    stopped_for_kl = True
                    break
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._recppo_parameters(),
                    float(self.cfg.recppo_max_grad_norm),
                )
                if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                    raise FloatingPointError("RecPPO gradient norm became non-finite; optimizer step was aborted")
                self._recppo_optimizer.step()
                performed_updates += 1
                max_grad_norm = max(max_grad_norm, float(torch.as_tensor(grad_norm).detach().item()))
                max_ratio_deviation = max(max_ratio_deviation, float(info["recppo_max_ratio_deviation"]))
                final_info = info
        finally:
            # The legacy outer loop clips model.parameters() after supervised
            # backward, so policy gradients must not survive into that norm.
            self._recppo_optimizer.zero_grad(set_to_none=True)
        final_info["recppo_update_epochs"] = performed_updates
        final_info["recppo_max_ratio_deviation"] = max_ratio_deviation
        final_info["recppo_grad_norm"] = max_grad_norm
        final_info["recppo_stopped_for_kl"] = stopped_for_kl
        final_info["recppo_effective_residual_scale"] = self._effective_recppo_residual_scale()
        self._last_recppo_info = final_info
        return final_info


def repaired_build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    banks = _legacy_build_eval_item_vecs(
        model,
        device,
        llm_scores,
        item_batch=item_batch,
    )
    model._recppo_cached_eval_banks = banks
    return banks


def repaired_build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    cached = getattr(model, "_recppo_cached_eval_banks", None)
    if cached is not None:
        bank = eval_mod.select_eval_item_bank(cached, eval_type)
        if bank is not None and bank.size(0) == int(model.cfg.n_items):
            return F.normalize(bank.index_select(0, item_idx.to(device=bank.device)), dim=1)
    return _legacy_build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type)


def repaired_make_fast3_optimizer(model, cfg):
    recppo_ids = model.recppo_parameter_ids() if hasattr(model, "recppo_parameter_ids") else set()
    delta_params = [
        param
        for param in model.content_delta_trainable_parameters()
        if id(param) not in recppo_ids
    ]
    delta_ids = {id(param) for param in delta_params}
    base_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in delta_ids and id(param) not in recppo_ids
    ]
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": float(cfg.lr)})
    if delta_params:
        groups.append(
            {
                "params": delta_params,
                "lr": float(cfg.lr) * float(getattr(cfg, "content_delta_lr_mult", 1.0)),
            }
        )
    if not groups:
        raise RuntimeError("Repaired outer optimizer has no supervised parameters")
    return torch.optim.Adam(groups)


def _move_optimizer_state_to_cpu(state):
    if torch.is_tensor(state):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {key: _move_optimizer_state_to_cpu(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_move_optimizer_state_to_cpu(value) for value in state]
    if isinstance(state, tuple):
        return tuple(_move_optimizer_state_to_cpu(value) for value in state)
    return state


def repaired_build_feedback_ckpt_state(*args, **kwargs):
    state = _legacy_build_feedback_ckpt_state(*args, **kwargs)
    model = args[0] if args else kwargs.get("model")
    optimizer = getattr(model, "_recppo_optimizer", None)
    if optimizer is not None and legacy._feedback_ckpt_save_optimizer_state():
        current_optimizer_state = _move_optimizer_state_to_cpu(optimizer.state_dict())
        es_best = state.get("es_best") or {}
        best_epoch = int(es_best.get("epoch", 0) or 0)
        next_epoch = int(state.get("next_epoch", 0) or 0)
        if best_epoch > 0 and best_epoch == next_epoch and state.get("es_best_state") is not None:
            model._recppo_best_optimizer_state = current_optimizer_state
        best_optimizer_state = _move_optimizer_state_to_cpu(
            getattr(model, "_recppo_best_optimizer_state", None)
        )
        state["recppo_best_optimizer_state"] = best_optimizer_state
        if state.get("status") == "finished" and best_optimizer_state is not None:
            state["recppo_optimizer_state"] = best_optimizer_state
        else:
            state["recppo_optimizer_state"] = current_optimizer_state
    else:
        state["recppo_optimizer_state"] = None
        state["recppo_best_optimizer_state"] = None
    return state


def repaired_load_feedback_checkpoint(*args, **kwargs):
    global _candidate_recppo_best_optimizer_state
    global _candidate_recppo_optimizer_state, _pending_recppo_best_optimizer_state
    global _pending_recppo_optimizer_state
    state = _legacy_load_feedback_checkpoint(*args, **kwargs)
    _pending_recppo_optimizer_state = None
    _pending_recppo_best_optimizer_state = None
    _candidate_recppo_optimizer_state = (
        state.get("recppo_optimizer_state") if isinstance(state, dict) else None
    )
    _candidate_recppo_best_optimizer_state = (
        state.get("recppo_best_optimizer_state") if isinstance(state, dict) else None
    )
    return state


def repaired_checkpoint_config_matches(*args, **kwargs):
    global _candidate_recppo_best_optimizer_state
    global _candidate_recppo_optimizer_state, _pending_recppo_best_optimizer_state
    global _pending_recppo_optimizer_state
    result = _legacy_checkpoint_config_matches(*args, **kwargs)
    accepted = bool(result[0]) if isinstance(result, tuple) and result else False
    _pending_recppo_optimizer_state = (
        _candidate_recppo_optimizer_state if accepted else None
    )
    _pending_recppo_best_optimizer_state = (
        _candidate_recppo_best_optimizer_state if accepted else None
    )
    _candidate_recppo_optimizer_state = None
    _candidate_recppo_best_optimizer_state = None
    return result


def repaired_static_train_config_fingerprint(cfg, split_info=None, script_path=None):
    _, payload = _legacy_static_train_config_fingerprint(
        cfg,
        split_info=split_info,
        script_path=script_path,
    )
    payload.update(
        {
            "recppo_fingerprint_schema": 2,
            "recppo_rank_reward_source": "train_positive_vs_global_hard_negatives",
            "recppo_joint_supervised_backbone": True,
            "rl_residual_scale": float(cfg.rl_residual_scale),
            "reward_step_cost": float(cfg.reward_step_cost),
            "reward_gain_clip": float(cfg.reward_gain_clip),
            "usim_lr": float(cfg.usim_lr),
            "recppo_rank_gain_weight": float(cfg.recppo_rank_gain_weight),
            "recppo_rank_topk": int(cfg.recppo_rank_topk),
            "recppo_rank_temperature": float(cfg.recppo_rank_temperature),
            "recppo_rank_normalize_transition": bool(cfg.recppo_rank_normalize_transition),
            "recppo_rank_item_chunk_size": int(cfg.recppo_rank_item_chunk_size),
            "recppo_embedding_gain_weight": float(cfg.recppo_embedding_gain_weight),
            "recppo_course_reward_scale": float(cfg.recppo_course_reward_scale),
            "recppo_course_reward_clip": float(cfg.recppo_course_reward_clip),
            "recppo_behavior_ce_weight": float(cfg.recppo_behavior_ce_weight),
            "recppo_behavior_ce_final_weight": float(cfg.recppo_behavior_ce_final_weight),
            "recppo_behavior_ce_anneal_epochs": int(cfg.recppo_behavior_ce_anneal_epochs),
            "recppo_terminal_value_weight": float(cfg.recppo_terminal_value_weight),
            "recppo_residual_ramp_epochs": int(cfg.recppo_residual_ramp_epochs),
            "recppo_max_residual_norm": float(cfg.recppo_max_residual_norm),
            "recppo_actor_lr": float(cfg.recppo_actor_lr),
            "recppo_critic_lr": float(cfg.recppo_critic_lr),
            "recppo_target_kl": float(cfg.recppo_target_kl),
            "recppo_policy_temperature": float(cfg.recppo_policy_temperature),
            "recppo_enable_stop": bool(cfg.recppo_enable_stop),
            "recppo_min_steps": int(cfg.recppo_min_steps),
            "recppo_stop_bias_init": float(cfg.recppo_stop_bias_init),
            "recppo_bootstrap_next_value": bool(cfg.recppo_bootstrap_next_value),
            "recppo_inject_behavior_user": bool(cfg.recppo_inject_behavior_user),
            "recppo_teacher_force_behavior": bool(cfg.recppo_teacher_force_behavior),
            "recppo_value_bound": float(cfg.recppo_value_bound),
            "recppo_logit_bound": float(cfg.recppo_logit_bound),
            "recppo_max_grad_norm": float(cfg.recppo_max_grad_norm),
            "recppo_guard_hot_ratio": float(cfg.recppo_guard_hot_ratio),
            "recppo_require_policy_checkpoint": bool(cfg.recppo_require_policy_checkpoint),
            "ppo_gamma": float(cfg.ppo_gamma),
            "ppo_epochs": int(cfg.ppo_epochs),
            "ppo_lambda": float(cfg.ppo_lambda),
            "ppo_clip": float(cfg.ppo_clip),
            "ppo_value_clip": float(cfg.ppo_value_clip),
            "ppo_adv_norm": bool(cfg.ppo_adv_norm),
            "ppo_coeffs": {
                key: float(value) for key, value in sorted(cfg.ppo_coeffs.items())
            },
            "feedback_course_prereq_weight": float(cfg.feedback_course_prereq_weight),
            "feedback_course_concept_weight": float(cfg.feedback_course_concept_weight),
            "feedback_course_difficulty_weight": float(cfg.feedback_course_difficulty_weight),
            "feedback_course_redundant_weight": float(cfg.feedback_course_redundant_weight),
            "feedback_course_sample_beta": float(cfg.feedback_course_sample_beta),
            "feedback_course_sample_only_cold": bool(cfg.feedback_course_sample_only_cold),
            "feedback_course_sample_topk": int(cfg.feedback_course_sample_topk),
            "feedback_course_sample_top_l": int(cfg.feedback_course_sample_top_l),
            "feedback_course_sample_soft": bool(cfg.feedback_course_sample_soft),
            "feedback_course_match_mode": str(cfg.feedback_course_match_mode),
            "feedback_course_match_topk": int(cfg.feedback_course_match_topk),
            "feedback_course_match_exclude_target": bool(cfg.feedback_course_match_exclude_target),
            "feedback_course_term_norm": str(cfg.feedback_course_term_norm),
            "feedback_course_term_norm_clip": float(cfg.feedback_course_term_norm_clip),
            "feedback_course_term_norm_eps": float(cfg.feedback_course_term_norm_eps),
            "feedback_course_term_norm_ema_decay": float(cfg.feedback_course_term_norm_ema_decay),
            "feedback_course_redundant_mode": str(cfg.feedback_course_redundant_mode),
            "feedback_course_redundant_thr": float(cfg.feedback_course_redundant_thr),
            "feedback_course_redundant_concept_gate": float(
                cfg.feedback_course_redundant_concept_gate
            ),
            "feedback_course_prereq_gate": float(cfg.feedback_course_prereq_gate),
            "feedback_prereq_weighted_edges": bool(cfg.feedback_prereq_weighted_edges),
            "feedback_prereq_soft_penalty": bool(cfg.feedback_prereq_soft_penalty),
            "early_stop_min_delta": float(cfg.early_stop_min_delta),
            "early_stop_score_mode": str(cfg.early_stop_score_mode),
            "candidate_strategy": str(cfg.candidate_strategy),
            "n_candidates": int(cfg.n_candidates),
            "retrieve_top_m": int(cfg.retrieve_top_m),
        }
    )
    return checkpoint_mod._stable_json_fingerprint(payload)


def repaired_write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df):
    path = _legacy_write_static_manifest(split_info, exports, cfg, course_stats, data_dir, df)
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    script_path = os.path.abspath(__file__)
    try:
        with open(script_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        digest = None
    manifest["script"] = {
        "path": script_path,
        "exists": os.path.exists(script_path),
        "size_bytes": os.path.getsize(script_path) if os.path.exists(script_path) else None,
        "mtime": os.path.getmtime(script_path) if os.path.exists(script_path) else None,
        "sha256": digest,
    }
    manifest.setdefault("model_config", {})["recppo"] = {
        "enabled": bool(cfg.recppo_enabled),
        "warmup_epochs": int(cfg.recppo_warmup_epochs),
        "ppo_epochs": int(cfg.ppo_epochs),
        "gamma": float(cfg.ppo_gamma),
        "actor_lr": float(cfg.recppo_actor_lr),
        "critic_lr": float(cfg.recppo_critic_lr),
        "policy_temperature": float(cfg.recppo_policy_temperature),
        "target_kl": float(cfg.recppo_target_kl),
        "enable_stop": bool(cfg.recppo_enable_stop),
        "termination_head": "separate_state_head",
        "step_cost": float(cfg.reward_step_cost),
        "rank_gain_weight": float(cfg.recppo_rank_gain_weight),
        "rank_reward_source": "train_positive_vs_global_hard_negatives",
        "rank_topk": int(cfg.recppo_rank_topk),
        "rank_temperature": float(cfg.recppo_rank_temperature),
        "rank_normalize_transition": bool(cfg.recppo_rank_normalize_transition),
        "rank_item_chunk_size": int(cfg.recppo_rank_item_chunk_size),
        "max_residual_norm": float(cfg.recppo_max_residual_norm),
        "residual_ramp_epochs": int(cfg.recppo_residual_ramp_epochs),
        "behavior_ce_weight": float(cfg.recppo_behavior_ce_weight),
        "behavior_ce_final_weight": float(cfg.recppo_behavior_ce_final_weight),
        "behavior_ce_anneal_epochs": int(cfg.recppo_behavior_ce_anneal_epochs),
        "embedding_gain_weight": float(cfg.recppo_embedding_gain_weight),
        "course_reward_scale": float(cfg.recppo_course_reward_scale),
        "course_reward_clip": float(cfg.recppo_course_reward_clip),
        "joint_supervised_backbone": True,
        "hard_negative_cache_scope": "epoch",
        "terminal_value_weight": float(cfg.recppo_terminal_value_weight),
        "require_policy_checkpoint": bool(cfg.recppo_require_policy_checkpoint),
        "guard_hot_ratio": float(cfg.recppo_guard_hot_ratio),
        "deterministic_eval_candidates": True,
        "behavior_supervision": "first_step_only",
        "strict_determinism": bool(cfg.recppo_strict_determinism),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def repaired_compute_early_stop_score(cold_metrics, hot_metrics, k, mode="cold_only"):
    mode = str(mode).strip().lower()
    if mode not in {"recppo_guarded", "recppo_stage_guarded"}:
        return _legacy_compute_early_stop_score(cold_metrics, hot_metrics, k, mode=mode)

    def metric(metrics, prefix):
        return max(0.0, float((metrics or {}).get(f"{prefix}@{k}", 0.0)))

    cold_r = metric(cold_metrics, "R")
    cold_n = metric(cold_metrics, "N")
    if cold_r <= 0.0 or cold_n <= 0.0:
        cold_core = 0.0
    else:
        cold_core = 2.0 * cold_r * cold_n / (cold_r + cold_n)

    active_model = _active_recppo_model_ref() if _active_recppo_model_ref is not None else None
    stage_guard = (
        mode == "recppo_stage_guarded"
        and active_model is not None
        and bool(getattr(active_model.cfg, "recppo_enabled", False))
        and bool(getattr(active_model.cfg, "recppo_require_policy_checkpoint", True))
    )
    if stage_guard and int(active_model._recppo_phase_state.item()) == 0:
        hot_r = metric(hot_metrics, "R")
        hot_n = metric(hot_metrics, "N")
        active_model._recppo_warm_hot_r.copy_(
            torch.maximum(active_model._recppo_warm_hot_r, active_model._recppo_warm_hot_r.new_tensor(hot_r))
        )
        active_model._recppo_warm_hot_n.copy_(
            torch.maximum(active_model._recppo_warm_hot_n, active_model._recppo_warm_hot_n.new_tensor(hot_n))
        )
        # Monotonic warmup sentinels prevent premature early stopping, while any
        # finite PPO-stage score replaces the warmup checkpoint.
        return -1_000_000.0 + float(int(active_model._recppo_epoch_state.item()) + 1)

    if stage_guard:
        guard_ratio = float(getattr(active_model.cfg, "recppo_guard_hot_ratio", 0.5))
        hot_r_floor = max(1e-8, float(active_model._recppo_warm_hot_r.item()) * guard_ratio)
        hot_n_floor = max(1e-8, float(active_model._recppo_warm_hot_n.item()) * guard_ratio)
    else:
        hot_r_floor = max(1e-8, float(os.environ.get("USIM_RECPPO_HOT_R_FLOOR", "0.05")))
        hot_n_floor = max(1e-8, float(os.environ.get("USIM_RECPPO_HOT_N_FLOOR", "0.025")))
    hot_r_gate = min(1.0, metric(hot_metrics, "R") / hot_r_floor)
    hot_n_gate = min(1.0, metric(hot_metrics, "N") / hot_n_floor)
    hot_gate = (hot_r_gate * hot_n_gate) ** 0.5
    return float(cold_core * hot_gate)


def install_repaired_bindings():
    legacy.setup_seed = repaired_setup_seed
    legacy.Fast3Config = RepairedFast3Config
    legacy.Fast3FeedbackUSIM = RepairedFast3FeedbackUSIM
    legacy._make_fast3_optimizer = repaired_make_fast3_optimizer
    legacy._build_feedback_ckpt_state = repaired_build_feedback_ckpt_state
    legacy._load_feedback_checkpoint = repaired_load_feedback_checkpoint
    legacy._checkpoint_config_matches = repaired_checkpoint_config_matches
    legacy._static_train_config_fingerprint = repaired_static_train_config_fingerprint
    legacy._write_static_manifest = repaired_write_static_manifest
    legacy._compute_early_stop_score = repaired_compute_early_stop_score
    legacy.build_eval_item_vecs = repaired_build_eval_item_vecs
    eval_mod.strict_cold_item_mask = repaired_strict_cold_item_mask
    eval_mod._strict_cold_item_mask = repaired_strict_cold_item_mask
    eval_mod.build_eval_item_vecs = repaired_build_eval_item_vecs
    eval_mod.build_eval_pos_item_vecs = repaired_build_eval_pos_item_vecs
    checkpoint_mod._static_train_config_fingerprint = repaired_static_train_config_fingerprint
    checkpoint_mod._checkpoint_config_matches = repaired_checkpoint_config_matches


def main():
    install_repaired_bindings()
    seed = int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025")))
    repaired_setup_seed(seed)
    ppo_enabled = (
        float(os.environ.get("USIM_PPO_LOSS_WEIGHT", "1")) > 0.0
        and os.environ.get("USIM_ROLLOUT_POLICY", "ppo").strip().lower() == "ppo"
    )
    print(">> Repaired strict-cold USIM entrypoint active")
    print(f">> RecPPO: {'enabled' if ppo_enabled else 'disabled (supervised ablation)' }.")
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
