import os


class BaseConfig:
    def __init__(self, n_users, n_items, content_dim=768):
        self.n_users = n_users
        self.n_items = n_items
        self.validation_only = os.environ.get("USIM_VALIDATION_ONLY", "0") == "1"
        self.emb_dim = 128
        self.hidden_dim = 256
        self.content_dim = content_dim
        # Optional research extensions are opt-in; formal main-table runs must
        # not silently activate them when an environment variable is absent.
        self.use_content_delta = os.environ.get("USIM_USE_CONTENT_DELTA", "0") == "1"
        self.content_delta_paper_style = os.environ.get("USIM_CONTENT_DELTA_PAPER_STYLE", "0") == "1"
        self.content_delta_replace_item = os.environ.get(
            "USIM_CONTENT_DELTA_REPLACE_ITEM",
            "1" if self.content_delta_paper_style else "0",
        ) == "1"
        self.content_delta_max_norm = float(
            os.environ.get("USIM_CONTENT_DELTA_MAX_NORM", os.environ.get("USIM_CONTENT_DELTA_MAX", "0.5"))
        )
        self.content_delta_cold_only = os.environ.get("USIM_CONTENT_DELTA_COLD_ONLY", "0") == "1"
        self.content_delta_normalize_base = os.environ.get("USIM_CONTENT_DELTA_NORMALIZE_BASE", "1") == "1"
        self.content_delta_normalize_output = os.environ.get("USIM_CONTENT_DELTA_NORMALIZE_OUTPUT", "1") == "1"
        self.content_delta_mode = os.environ.get("USIM_CONTENT_DELTA_MODE", "embedding").strip().lower()
        if self.content_delta_mode in {"mlp", "content", "content_projector"}:
            self.content_delta_mode = "projector"
        if self.content_delta_mode not in {"embedding", "projector", "hybrid"}:
            raise ValueError(
                "USIM_CONTENT_DELTA_MODE must be one of: embedding, projector, hybrid"
            )
        self.content_delta_projector_hidden = int(
            os.environ.get("USIM_CONTENT_DELTA_PROJECTOR_HIDDEN", str(self.hidden_dim))
        )
        self.content_delta_train_on_id_dropout = (
            os.environ.get("USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT", "1") == "1"
        )
        self.content_delta_only_after_epoch = int(
            os.environ.get("USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH", "0")
        )
        self.content_delta_scale = float(os.environ.get("USIM_CONTENT_DELTA_SCALE", "0.25"))
        self.content_delta_aux_mode = os.environ.get("USIM_CONTENT_DELTA_AUX_MODE", "base").strip().lower()
        self.content_delta_l2_weight = float(os.environ.get("USIM_CONTENT_DELTA_L2_W", "0.02"))
        self.content_delta_cap_weight = float(os.environ.get("USIM_CONTENT_DELTA_CAP_W", "0.02"))
        self.content_delta_cap_margin = float(os.environ.get("USIM_CONTENT_DELTA_CAP_MARGIN", "0.70"))
        self.content_delta_lr_mult = float(os.environ.get("USIM_CONTENT_DELTA_LR_MULT", "0.10"))
        self.content_delta_eval_bank_mode = os.environ.get(
            "USIM_CONTENT_DELTA_EVAL_BANK_MODE",
            "auto",
        ).strip().lower()
        self.use_sg_urinit = os.environ.get("USIM_USE_SG_URINIT", "0") == "1"
        self.sg_urinit_cluster_k = max(1, int(os.environ.get("USIM_SG_URINIT_CLUSTER_K", "32")))
        self.sg_urinit_local_weight = max(0.0, float(os.environ.get("USIM_SG_URINIT_LOCAL_W", "0.70")))
        self.sg_urinit_global_weight = max(0.0, float(os.environ.get("USIM_SG_URINIT_GLOBAL_W", "0.30")))
        self.sg_urinit_target_norm = max(0.0, float(os.environ.get("USIM_SG_URINIT_TARGET_NORM", "0.0")))
        self.sg_urinit_max_iter = max(1, int(os.environ.get("USIM_SG_URINIT_MAX_ITER", "20")))
        self.sg_urinit_seed = int(os.environ.get("USIM_SG_URINIT_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
        self.sg_urinit_stats = {"enabled": False, "initialized_users": 0}
        self.cold_threshold = int(os.environ.get("USIM_COLD_THRESHOLD", "5"))
        self.lr = 0.0005
        self.temp = 0.07
        self.margin = 0.15
        self.dropout_prob = 0.35
        self.aux_weight = float(os.environ.get("USIM_AUX_WEIGHT", "0.3"))
        # ROLLBACK FLAG (USIM_AUX_HOT_ONLY): when "1", restrict the id<->content
        # auxiliary InfoNCE to hot rows only. Cold rows have under-trained
        # `id_e_true` and inject noise into content_e gradient. Default "0"
        # preserves legacy behavior; set "1" together with the cold-start patch
        # rollout. See _compute_aux_loss for the actual branching.
        self.aux_hot_only = os.environ.get("USIM_AUX_HOT_ONLY", "0") == "1"
        self.train_force_cold = os.environ.get("USIM_TRAIN_FORCE_COLD", "1") == "1"
        self.use_pseudo_cold_train = os.environ.get("USIM_USE_PSEUDO_COLD_TRAIN", "0") == "1"
        self.pseudo_cold_ratio = float(os.environ.get("USIM_PSEUDO_COLD_RATIO", "0.30"))
        self.pseudo_cold_ratio = min(1.0, max(0.0, self.pseudo_cold_ratio))
        self.pseudo_cold_min_pop = int(os.environ.get("USIM_PSEUDO_COLD_MIN_POP", "5"))
        self.pseudo_cold_mode = os.environ.get("USIM_PSEUDO_COLD_MODE", "batch_random").strip().lower()
        if self.pseudo_cold_mode not in {"batch_random", "batch_tail", "item_tail", "all_eligible", "none", "off"}:
            raise ValueError(
                "USIM_PSEUDO_COLD_MODE must be one of: batch_random, batch_tail, item_tail, all_eligible, none, off"
            )
        self.disable_llm_score = os.environ.get("USIM_DISABLE_LLM_SCORE", "0") == "1"
        self.llm_safe_mode = os.environ.get("USIM_LLM_SAFE_MODE", "0") == "1"
        self.llm_weight = float(
            os.environ.get("USIM_LLM_WEIGHT", "0.20" if self.llm_safe_mode else "1.0")
        )
        self.llm_cold_only = os.environ.get(
            "USIM_LLM_COLD_ONLY",
            "1" if self.llm_safe_mode else "0",
        ) == "1"
        self.llm_hot_only = os.environ.get("USIM_LLM_HOT_ONLY", "0") == "1"
        self.llm_bank_mode = os.environ.get(
            "USIM_LLM_BANK_MODE",
            "none" if self.llm_safe_mode else "item",
        ).strip().lower()
        self.ppo_clip = 0.2
        self.ppo_gamma = 0.90
        self.ppo_epochs = 5
        self.ppo_coeffs = {"value": 0.5, "entropy": 0.01}
        self.ppo_loss_weight = float(os.environ.get("USIM_PPO_LOSS_WEIGHT", "1.0"))
        self.rl_residual_scale = float(os.environ.get("USIM_RL_RESIDUAL_SCALE", "1.0"))
        self.rl_residual_scale = min(1.0, max(0.0, self.rl_residual_scale))
        self.rollout_policy = os.environ.get("USIM_ROLLOUT_POLICY", "ppo").strip().lower()
        if self.rollout_policy in {"learned", "agent", "ppo_policy"}:
            self.rollout_policy = "ppo"
        if self.rollout_policy in {"greedy", "similarity", "greedy_sim"}:
            self.rollout_policy = "greedy_similarity"
        if self.rollout_policy in {"course", "course_fit_greedy", "course-fit"}:
            self.rollout_policy = "course_fit"
        if self.rollout_policy not in {"ppo", "random", "greedy_similarity", "course_fit"}:
            raise ValueError(
                "USIM_ROLLOUT_POLICY must be one of: ppo, random, greedy_similarity, course_fit"
            )
        self.usim_steps = int(os.environ.get("USIM_STEPS", "5"))
        self.n_candidates = int(os.environ.get("USIM_N_CANDIDATES", "20"))
        self.usim_lr = 0.3
        self.candidate_strategy = "retrieve_sample"
        self.retrieve_top_m = int(os.environ.get("USIM_RETRIEVE_TOP_M", "256"))
        self.candidate_temp = 0.20
        self.candidate_epsilon = 0.10
        self.retrieval_user_chunk = 16384
        self.retrieval_query_chunk = 256
        self.user_bank_refresh_steps = 200
        self.n_epochs = int(os.environ.get("USIM_N_EPOCHS", "3"))
        self.batch_size = int(os.environ.get("USIM_BATCH_SIZE", "2048"))
        self.accum_steps = 1
        self.eval_n_neg = int(os.environ.get("USIM_EVAL_N_NEG", "200"))
        # Sampled (1+N_neg) eval is no longer the headline metric; final tables
        # report full ranking (item-macro). Default flipped to "0" to save eval
        # time. Set USIM_RUN_SAMPLED_EVAL=1 to restore legacy 1+200 sampled eval.
        self.run_sampled_eval = os.environ.get("USIM_RUN_SAMPLED_EVAL", "0") == "1"
        self.use_mixed_hard_neg = True
        self.train_num_negs = 32
        self.hard_neg_ratio = 0.25
        self.use_structured_hard_neg = False
        self.mask_known_pos_neg = os.environ.get("USIM_MASK_KNOWN_POS_NEG", "0") == "1"
        self.mask_same_item_neg = os.environ.get("USIM_MASK_SAME_ITEM_NEG", "1") == "1"
        self.use_paac = os.environ.get("USIM_USE_PAAC", "0") == "1"
        self.paac_align_weight = float(os.environ.get("USIM_PAAC_ALIGN_W", "0.0"))
        self.paac_align_max_pairs = int(os.environ.get("USIM_PAAC_ALIGN_MAX_PAIRS", "512"))
        self.paac_align_detach_hot = os.environ.get("USIM_PAAC_ALIGN_DETACH_HOT", "1") == "1"
        self.paac_contrast_weight = float(os.environ.get("USIM_PAAC_CONTRAST_W", "0.02"))
        self.paac_contrast_beta = float(os.environ.get("USIM_PAAC_CONTRAST_BETA", "0.20"))
        self.paac_contrast_gamma = float(os.environ.get("USIM_PAAC_CONTRAST_GAMMA", "0.20"))
        self.paac_batch_pop_ratio = float(os.environ.get("USIM_PAAC_BATCH_POP_RATIO", "0.50"))
        self.paac_group_mode = os.environ.get("USIM_PAAC_GROUP_MODE", "batch_quantile").strip().lower()
        self.use_course_rerank = False
        self.rerank_alpha = 0.00
        self.rerank_lambda = 0.01
        self.rerank_min_seen = 8
        self.rerank_top_l = 50
        self.rerank_penalty_cap = 0.10
        self.rerank_only_cold = True
        self.prereq_min_support = 30
        self.prereq_max_per_item = 5
        self.prereq_min_items = 1
        self.prereq_max_forward = 20
        self.concept_overlap_mode = os.environ.get("USIM_CONCEPT_OVERLAP_MODE", "plain").strip().lower()
        self.prereq_graph_source = os.environ.get("USIM_PREREQ_GRAPH_SOURCE", "behavior").strip().lower()
        self.prereq_concept_score_thr = float(os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", "0.10"))
        self.prereq_concept_min_hits = int(os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", "1"))
        self.prereq_concept_file = os.environ.get("USIM_PREREQ_CONCEPT_FILE", "prerequisite-dependency.json")
        self.use_prereq_aux_loss = True
        self.prereq_aux_weight = 0.03
        self.prereq_aux_margin = 0.05
        self.prereq_aux_violation_thr = 0.60
        self.prereq_aux_min_seen = 5
        self.prereq_aux_only_cold = True
        self.use_epoch_early_stop = os.environ.get("USIM_USE_EPOCH_EARLY_STOP", "1") == "1"
        self.early_stop_k = 10
        self.early_stop_patience = int(os.environ.get("USIM_EARLY_STOP_PATIENCE", "1"))
        self.early_stop_min_delta = float(os.environ.get("USIM_EARLY_STOP_MIN_DELTA", "1e-4"))
        self.early_stop_average_mode = os.environ.get("USIM_EARLY_STOP_AVG_MODE", "interaction").strip().lower()
        if self.early_stop_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("USIM_EARLY_STOP_AVG_MODE must be 'interaction' or 'item_macro'")
        # ROLLBACK FLAG (USIM_EARLY_STOP_SCORE_MODE): how validation metrics are
        # combined into the early-stop score. "cold_only" (default) is the
        # legacy behavior. "geometric" / "harmonic" / "sum" let hot pull the
        # selector back when cold gains stop translating into hot improvement.
        # "cold_rn" / "balanced_rn" jointly select by Recall and NDCG.
        # See _compute_early_stop_score for the formulas.
        self.early_stop_score_mode = os.environ.get(
            "USIM_EARLY_STOP_SCORE_MODE", "cold_only"
        ).strip().lower()
        if self.early_stop_score_mode not in {
            "cold_only",
            "geometric",
            "harmonic",
            "sum",
            "cold_rn",
            "balanced_rn",
        }:
            raise ValueError(
                "USIM_EARLY_STOP_SCORE_MODE must be one of: "
                "cold_only, geometric, harmonic, sum, cold_rn, balanced_rn"
            )
        self.early_stop_hot_r10_drop_tol = 0.03
        self.legacy_train_protocol = os.environ.get("USIM_LEGACY_TRAIN_PROTOCOL", "0") == "1"
        self.use_usim_refined_eval = os.environ.get("USIM_USE_REFINED_EVAL", "1") == "1"


class FeedbackConfig(BaseConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.feedback_load_course_artifacts = os.environ.get("USIM_FB_LOAD_COURSE_ARTIFACTS", "1") == "1"
        self.reward_terminal_weight = float(os.environ.get("USIM_FB_REWARD_TERM_W", "10.0"))
        self.reward_gain_weight = float(os.environ.get("USIM_FB_REWARD_GAIN_W", "5.0"))
        self.reward_gain_clip = float(os.environ.get("USIM_FB_REWARD_GAIN_CLIP", "0.05"))
        self.reward_dup_penalty_weight = float(os.environ.get("USIM_FB_REWARD_DUP_W", "0.50"))
        self.reward_cov_bonus_weight = float(os.environ.get("USIM_FB_REWARD_COV_W", "0.00"))
        self.use_course_reward = os.environ.get("USIM_USE_COURSE_REWARD", "1") == "1"
        self.feedback_course_only_cold = os.environ.get("USIM_FB_COURSE_ONLY_COLD", "1") == "1"
        self.feedback_course_warm_seen = int(os.environ.get("USIM_FB_COURSE_WARM_SEEN", "5"))
        self.feedback_course_concept_min = float(os.environ.get("USIM_FB_COURSE_CONCEPT_MIN", "0.12"))
        self.feedback_course_match_mode = os.environ.get("USIM_FB_COURSE_MATCH_MODE", "mean").strip().lower()
        if self.feedback_course_match_mode in {"topk_mean", "top_k", "top-k"}:
            self.feedback_course_match_mode = "topk"
        if self.feedback_course_match_mode not in {"mean", "topk", "max"}:
            raise ValueError("USIM_FB_COURSE_MATCH_MODE must be one of: mean, topk, max")
        self.feedback_course_match_topk = max(1, int(os.environ.get("USIM_FB_COURSE_MATCH_TOPK", "5")))
        default_match_exclude = "1" if self.feedback_course_match_mode in {"topk", "max"} else "0"
        self.feedback_course_match_exclude_target = os.environ.get(
            "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET",
            default_match_exclude,
        ) == "1"
        self.feedback_course_redundant_mode = os.environ.get(
            "USIM_FB_COURSE_REDUNDANT_MODE",
            "concept",
        ).strip().lower()
        self.feedback_course_redundant_thr = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_THR", "0.70"))
        self.feedback_course_struct_video_min = float(
            os.environ.get("USIM_FB_COURSE_STRUCT_VIDEO_MIN", "0.60")
        )
        self.feedback_course_struct_chunk = max(
            1,
            int(os.environ.get("USIM_FB_COURSE_STRUCT_CHUNK", "8192")),
        )
        self.feedback_course_prereq_gate = float(os.environ.get("USIM_FB_COURSE_PREREQ_GATE", "0.20"))
        self.feedback_course_prereq_weight = float(os.environ.get("USIM_FB_COURSE_PREREQ_W", "0.08"))
        self.feedback_prereq_weighted_edges = os.environ.get("USIM_FB_PREREQ_WEIGHTED_EDGES", "0") == "1"
        self.feedback_prereq_soft_penalty = os.environ.get("USIM_FB_PREREQ_SOFT_PENALTY", "0") == "1"
        self.feedback_course_concept_weight = float(os.environ.get("USIM_FB_COURSE_CONCEPT_W", "0.04"))
        self.feedback_course_difficulty_weight = float(os.environ.get("USIM_FB_COURSE_DIFF_W", "0.03"))
        self.feedback_course_redundant_weight = float(os.environ.get("USIM_FB_COURSE_REDUNDANT_W", "0.02"))
        self.feedback_course_redundant_concept_gate = float(
            os.environ.get("USIM_FB_COURSE_REDUNDANT_CONCEPT_GATE", "1.0")
        )
        self.feedback_course_term_norm = os.environ.get("USIM_FB_COURSE_TERM_NORM", "none").strip().lower()
        if self.feedback_course_term_norm in {"off", "0", "false"}:
            self.feedback_course_term_norm = "none"
        if self.feedback_course_term_norm not in {"none", "batch", "ema"}:
            raise ValueError("USIM_FB_COURSE_TERM_NORM must be one of: none, batch, ema")
        self.feedback_course_term_norm_clip = float(os.environ.get("USIM_FB_COURSE_TERM_NORM_CLIP", "2.0"))
        self.feedback_course_term_norm_eps = float(os.environ.get("USIM_FB_COURSE_TERM_NORM_EPS", "1e-6"))
        self.feedback_course_term_norm_ema_decay = float(
            os.environ.get("USIM_FB_COURSE_TERM_NORM_EMA_DECAY", "0.95")
        )
        self.feedback_course_term_norm_ema_decay = min(
            0.999,
            max(0.0, self.feedback_course_term_norm_ema_decay),
        )
        self.feedback_course_sample_beta = float(os.environ.get("USIM_FB_COURSE_SAMPLE_BETA", "0.20"))
        self.feedback_course_sample_only_cold = os.environ.get("USIM_FB_COURSE_SAMPLE_ONLY_COLD", "1") == "1"
        self.feedback_course_sample_topk = int(os.environ.get("USIM_FB_COURSE_SAMPLE_TOPK", "32"))
        self.feedback_course_sample_top_l = int(
            os.environ.get("USIM_FB_COURSE_SAMPLE_TOPL", str(self.feedback_course_sample_topk))
        )
        self.use_sage_lite = os.environ.get("USIM_USE_SAGE_LITE", "0") == "1"
        self.sage_gate_min = float(os.environ.get("USIM_SAGE_GATE_MIN", "0.10"))
        self.sage_gate_max = float(os.environ.get("USIM_SAGE_GATE_MAX", "0.60"))
        self.sage_gate_mode = os.environ.get("USIM_SAGE_GATE_MODE", "heuristic").strip().lower()
        if self.sage_gate_mode in {"pop", "popularity", "rule"}:
            self.sage_gate_mode = "heuristic"
        if self.sage_gate_mode in {"bucket", "mlp"}:
            self.sage_gate_mode = "bucket_mlp"
        if self.sage_gate_mode not in {"heuristic", "bucket_mlp"}:
            raise ValueError("USIM_SAGE_GATE_MODE must be one of: heuristic, bucket_mlp")
        self.sage_gate_bucket_count = max(2, int(os.environ.get("USIM_SAGE_GATE_BUCKETS", "20")))
        self.sage_gate_hidden_dim = max(1, int(os.environ.get("USIM_SAGE_GATE_HIDDEN", "32")))
        self.sage_gate_bucket_strategy = os.environ.get("USIM_SAGE_GATE_BUCKET_STRATEGY", "paper").strip().lower()
        if self.sage_gate_bucket_strategy in {"raw", "equal_width", "raw_equal_width", "original", "sagerec"}:
            self.sage_gate_bucket_strategy = "paper"
        if self.sage_gate_bucket_strategy in {"log_equal_width", "log1p"}:
            self.sage_gate_bucket_strategy = "log"
        if self.sage_gate_bucket_strategy not in {"paper", "log"}:
            raise ValueError("USIM_SAGE_GATE_BUCKET_STRATEGY must be one of: paper, log")
        self.sage_pool_topk = int(os.environ.get("USIM_SAGE_POOL_TOPK", "64"))
        self.sage_course_temp = float(os.environ.get("USIM_SAGE_COURSE_TEMP", "0.20"))
        self.sage_only_cold_or_tail = os.environ.get("USIM_SAGE_ONLY_COLD_OR_TAIL", "0") == "1"
        self.sage_tail_pop_ratio = float(os.environ.get("USIM_SAGE_TAIL_POP_RATIO", "0.10"))
        self.sage_tail_pop_ratio = min(1.0, max(0.0, self.sage_tail_pop_ratio))
        self.sage_use_two_expert = os.environ.get("USIM_SAGE_USE_TWO_EXPERT", "0") == "1"
        self.sage_two_expert_score_fusion = os.environ.get("USIM_SAGE_TWO_EXPERT_SCORE_FUSION", "0") == "1"
        self.use_sage_aux_loss = os.environ.get("USIM_USE_SAGE_AUX_LOSS", "0") == "1"
        self.sage_aux_weight = float(os.environ.get("USIM_SAGE_AUX_WEIGHT", "0.02"))
        self.sage_aux_pool_topk = int(os.environ.get("USIM_SAGE_AUX_POOL_TOPK", str(self.sage_pool_topk)))
        self.sage_aux_course_temp = float(os.environ.get("USIM_SAGE_AUX_COURSE_TEMP", str(self.sage_course_temp)))
        self.sage_aux_retrieval_temp = float(os.environ.get("USIM_SAGE_AUX_RETRIEVAL_TEMP", "1.0"))
        self.sage_aux_only_strict_cold = os.environ.get("USIM_SAGE_AUX_ONLY_STRICT_COLD", "1") == "1"
        self.sage_aux_detach_user = os.environ.get("USIM_SAGE_AUX_DETACH_USER", "1") == "1"
        self.use_cgrc_recon = os.environ.get("USIM_USE_CGRC_RECON", "0") == "1"
        self.cgrc_recon_aux_weight = float(os.environ.get("USIM_CGRC_RECON_AUX_W", "0.0"))
        self.cgrc_recon_sample_weight = float(os.environ.get("USIM_CGRC_RECON_SAMPLE_W", "0.0"))
        self.cgrc_recon_sample_weight = min(1.0, max(0.0, self.cgrc_recon_sample_weight))
        self.cgrc_recon_pseudo_ratio = float(os.environ.get("USIM_CGRC_RECON_PSEUDO_RATIO", "0.30"))
        self.cgrc_recon_pseudo_ratio = min(1.0, max(0.0, self.cgrc_recon_pseudo_ratio))
        self.cgrc_recon_topk = max(2, int(os.environ.get("USIM_CGRC_RECON_TOPK", "64")))
        self.cgrc_recon_temperature = max(1e-6, float(os.environ.get("USIM_CGRC_RECON_TEMP", "0.50")))
        self.cgrc_recon_only_cold_or_tail = os.environ.get("USIM_CGRC_RECON_ONLY_COLD_OR_TAIL", "1") == "1"
        self.cgrc_recon_tail_pop_ratio = float(os.environ.get("USIM_CGRC_RECON_TAIL_POP_RATIO", "0.10"))
        self.cgrc_recon_tail_pop_ratio = min(1.0, max(0.0, self.cgrc_recon_tail_pop_ratio))
        self.cgrc_recon_detach_user = os.environ.get("USIM_CGRC_RECON_DETACH_USER", "0") == "1"
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
        self.prereq_graph_source = os.environ.get("USIM_PREREQ_GRAPH_SOURCE", self.prereq_graph_source).strip().lower()
        self.prereq_concept_score_thr = float(
            os.environ.get("USIM_PREREQ_CONCEPT_SCORE_THR", str(self.prereq_concept_score_thr))
        )
        self.prereq_concept_min_hits = int(
            os.environ.get("USIM_PREREQ_CONCEPT_MIN_HITS", str(self.prereq_concept_min_hits))
        )
        self.prereq_concept_file = os.environ.get("USIM_PREREQ_CONCEPT_FILE", self.prereq_concept_file)
        self.prereq_hybrid_alpha = float(os.environ.get("USIM_PREREQ_HYBRID_ALPHA", "0.70"))
        self.prereq_hybrid_strong_concept_thr = float(
            os.environ.get("USIM_PREREQ_HYBRID_STRONG_CONCEPT_THR", "0.35")
        )


class Fast3Config(FeedbackConfig):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)

        self.ppo_epochs = int(os.environ.get("USIM_PPO_EPOCHS", "2"))
        self.stream_train_window = int(os.environ.get("USIM_TRAIN_WINDOW", "24"))

        self.ppo_lambda = float(os.environ.get("USIM_PPO_LAMBDA", "0.95"))
        self.ppo_value_clip = float(os.environ.get("USIM_PPO_VALUE_CLIP", "0.20"))
        self.ppo_adv_norm = os.environ.get("USIM_PPO_ADV_NORM", "1") == "1"

        self.fast3_target_alpha_cold = float(os.environ.get("USIM_FAST3_TGT_ALPHA_COLD", "0.35"))
        self.fast3_target_alpha_hot = float(os.environ.get("USIM_FAST3_TGT_ALPHA_HOT", "0.60"))
        self.fast3_target_alpha_step = float(os.environ.get("USIM_FAST3_TGT_ALPHA_STEP", "0.20"))
        self.fast3_target_alpha_entropy = float(os.environ.get("USIM_FAST3_TGT_ALPHA_ENT", "0.20"))
        self.fast3_target_alpha_min = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MIN", "0.15"))
        self.fast3_target_alpha_max = float(os.environ.get("USIM_FAST3_TGT_ALPHA_MAX", "0.85"))

        self.feedback_course_sample_soft = os.environ.get("USIM_FB_COURSE_SAMPLE_SOFT", "1") == "1"
        self.feedback_course_sample_top_l = int(
            os.environ.get(
                "USIM_FB_COURSE_SAMPLE_TOPL",
                str(getattr(self, "feedback_course_sample_topk", 32)),
            )
        )


