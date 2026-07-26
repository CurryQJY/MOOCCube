param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [switch]$MinimalLiraMode,
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$OutputRoot = "outputs\content_delta_pop5\static_item_cold",
    [string]$CheckpointRoot = "checkpoints\content_delta_pop5\static_item_cold",
    [ValidateSet("threshold", "strict_item_cold", "strict_item_cold_balanced")]
    [string]$Protocol = "strict_item_cold",
    [Alias("ColdThreshold")]
    [int[]]$ColdThresholds = @(1),
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 15,
    [int]$Patience = 3,
    [ValidateSet("interaction", "item_macro")]
    [string]$EarlyStopAverageMode = "item_macro",
    [ValidateSet("embedding", "projector", "hybrid")]
    [string]$ContentDeltaMode = "embedding",
    [double]$ContentDeltaMaxNorm = 0.05,
    [double]$ContentDeltaScale = 0.25,
    [double]$ContentDeltaLrMult = 0.10,
    [double]$ContentDeltaL2W = 0.02,
    [double]$ContentDeltaCapW = 0.02,
    [double]$ContentDeltaCapMargin = 0.70,
    [int]$ContentDeltaProjectorHidden = 256,
    [ValidateSet("base", "raw", "no_delta", "delta", "content")]
    [string]$ContentDeltaAuxMode = "base",
    [ValidateSet("auto", "item", "item_pop", "cold", "all", "hot", "none", "off")]
    [string]$ContentDeltaEvalBankMode = "auto",
    [bool]$UseContentDelta = $true,
    [bool]$ContentDeltaPaperStyle = $false,
    [bool]$ContentDeltaReplaceItem = $false,
    [bool]$ContentDeltaColdOnly = $true,
    [bool]$ContentDeltaTrainOnIdDropout = $true,
    [int]$ContentDeltaOnlyAfterEpoch = 0,
    [double]$AuxWeight = 0.3,
    [bool]$UsePseudoColdTrain = $true,
    [ValidateSet("batch_random", "batch_tail", "item_tail", "all_eligible", "fixed_item_stratified", "none", "off")]
    [string]$PseudoColdMode = "all_eligible",
    [double]$PseudoColdRatio = 1.00,
    [int]$PseudoColdMinPop = 1,
    [bool]$UsePaac = $false,
    [double]$PaacAlignW = 0.0,
    [double]$PaacContrastW = 0.0,
    [bool]$UseCourseFeedback = $true,
    [bool]$UseCourseReward = $true,
    [bool]$UsePrereqAux = $true,
    [bool]$PrereqAuxOnlyCold = $true,
    [ValidateSet("behavior", "concept", "hybrid")]
    [string]$PrereqGraphSource = "concept",
    [double]$CoursePrereqW = 0.08,
    [double]$CoursePrereqGate = 0.20,
    [double]$CourseConceptW = 0.04,
    [ValidateSet("mean", "topk", "max")]
    [string]$CourseMatchMode = "mean",
    [int]$CourseMatchTopK = 5,
    [double]$CourseDiffW = 0.03,
    [double]$CourseRedundantW = 0.02,
    [double]$CourseRedundantConceptGate = 1.0,
    [ValidateSet("concept", "video_family")]
    [string]$CourseRedundantMode = "concept",
    [ValidateSet("none", "batch", "ema")]
    [string]$CourseTermNorm = "none",
    [double]$CourseTermNormClip = 2.0,
    [double]$CourseTermNormEps = 1e-6,
    [double]$CourseTermNormEmaDecay = 0.95,
    [int]$CourseStructChunk = 8192,
    [bool]$UseSageLite = $false,
    [double]$SageGateMin = 0.10,
    [double]$SageGateMax = 0.60,
    [ValidateSet("heuristic", "bucket_mlp")]
    [string]$SageGateMode = "heuristic",
    [int]$SageGateBuckets = 20,
    [int]$SageGateHidden = 32,
    [ValidateSet("paper", "log")]
    [string]$SageGateBucketStrategy = "paper",
    [int]$SagePoolTopK = 64,
    [double]$SageCourseTemp = 0.20,
    [bool]$SageOnlyColdOrTail = $false,
    [double]$SageTailPopRatio = 0.10,
    [bool]$SageUseTwoExpert = $false,
    [bool]$SageTwoExpertScoreFusion = $false,
    [bool]$UseSageAuxLoss = $false,
    [double]$SageAuxWeight = 0.02,
    [int]$SageAuxPoolTopK = 48,
    [double]$SageAuxCourseTemp = 0.20,
    [double]$SageAuxRetrievalTemp = 1.0,
    [bool]$SageAuxOnlyStrictCold = $true,
    [bool]$SageAuxDetachUser = $true,
    [bool]$UseCgrcRecon = $false,
    [double]$CgrcReconAuxW = 0.0,
    [double]$CgrcReconSampleW = 0.0,
    [double]$CgrcReconPseudoRatio = 0.30,
    [int]$CgrcReconTopK = 64,
    [double]$CgrcReconTemp = 0.50,
    [bool]$CgrcReconOnlyColdOrTail = $true,
    [double]$CgrcReconTailPopRatio = 0.10,
    [bool]$CgrcReconDetachUser = $false,
    [bool]$CourseFeedbackOnlyCold = $true,
    [bool]$UseCourseSample = $true,
    [bool]$CourseSampleOnlyCold = $true,
    [double]$CourseSampleBeta = 0.20,
    [bool]$UseCourseRerank = $false,
    [bool]$CourseRerankOnlyCold = $true,
    [double]$CourseRerankAlpha = 0.00,
    [double]$CourseRerankLambda = 0.01,
    [int]$CourseRerankTopL = 50,
    [bool]$UseStructuredHardNeg = $false,
    [bool]$MaskKnownPosNeg = $true,
    [bool]$MaskSameItemNeg = $true,
    [bool]$UseSgUrinit = $false,
    [int]$SgUrinitClusterK = 32,
    [double]$SgUrinitLocalW = 0.70,
    [double]$SgUrinitGlobalW = 0.30,
    [double]$SgUrinitTargetNorm = 0.0,
    [int]$SgUrinitMaxIter = 20,
    [bool]$TrainForceCold = $true,
    [int]$UsimSteps = 5,
    [int]$PpoEpochs = 2,
    [bool]$UseUsimRefinedEval = $true,
    [double]$PpoLossWeight = 1.0,
    [string]$InitCheckpointDir = "",
    [double]$RlResidualScale = 1.0,
    [ValidateSet("ppo", "random", "greedy_similarity", "course_fit")]
    [string]$RolloutPolicy = "ppo",
    [ValidateSet("legacy_id", "initial_state")]
    [string]$SimulatorTargetMode = "legacy_id",
    [bool]$DeterministicEvalCandidates = $false,
    [bool]$EvalReuseItemBank = $false,
    [int]$DeterministicEvalSeed = 0,
    [bool]$CkgRlV1 = $false,
    [int]$V1PseudoColdPlanCount = 0,
    [int]$V1PseudoColdPlanSeed = 0,
    [ValidateSet("popularity_stratified")]
    [string]$V1PseudoColdPlanStrategy = "popularity_stratified",
    [int]$V1ReferenceBatchSize = 0,
    [bool]$V1TargetHistoryExclusion = $false,
    [ValidateSet("all_course_terms", "concept_only")]
    [string]$V1TargetHistoryExclusionScope = "all_course_terms",
    [double]$V1SelectorHotTolerance = 0.003,
    [double]$V1SelectorOverallTolerance = 0.003,
    [ValidateSet("cold_ndcg_then_recall_with_retention", "cold_ndcg_running_retention")]
    [string]$V1SelectorMode = "cold_ndcg_then_recall_with_retention",
    [double]$RewardDupW = 0.50,
    [string]$CourseMatchExcludeTarget = "",
    # Cold-start patch (2026-05-19) flags. Legacy defaults preserved when the
    # caller doesn't pass them; see docs/COLD_START_PATCH_2026_05_19.md.
    [bool]$AuxHotOnly = $false,
    [ValidateSet("cold_only", "geometric", "harmonic", "sum", "cold_rn", "balanced_rn")]
    [string]$EarlyStopScoreMode = "cold_only",
    [bool]$RunSampledEval = $false,
    [bool]$SaveCkpt = $true,
    # Resume by default; Python-side config fingerprint forces fresh when train knobs change.
    [bool]$AutoResume = $true,
    [bool]$ForceFresh = $false,
    [bool]$SaveOptState = $true,
    [bool]$AllowLegacyCheckpoint = $false,
    [switch]$DryRun,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$trackedEnv = @(
    "USIM_DATA_DIR",
    "USIM_RELATION_DIR",
    "USIM_STATIC",
    "USIM_STATIC_SPLIT_MODE",
    "USIM_STATIC_SEED",
    "USIM_SEED",
    "USIM_COLD_THRESHOLD",
    "USIM_STATIC_TRAIN_RATIO",
    "USIM_STATIC_VAL_RATIO",
    "USIM_STATIC_COLD_ITEM_RATIO",
    "USIM_STATIC_VAL_COLD_ITEM_RATIO",
    "USIM_STATIC_COLD_ITEM_MIN_INTER",
    "USIM_STATIC_COLD_ITEM_FOLDS",
    "USIM_STATIC_ARTIFACT_SOURCE",
    "USIM_STATIC_EXPORT_SPLIT",
    "USIM_STATIC_TEST_HISTORY",
    "USIM_FB_FORCE_FRESH",
    "USIM_FB_AUTO_RESUME",
    "USIM_FB_SAVE_CKPT",
    "USIM_FB_SAVE_OPT_STATE",
    "USIM_FB_OUTPUT_DIR",
    "USIM_FB_OUTPUT_TAG",
    "USIM_FB_CKPT_DIR",
    "USIM_FB_INIT_CKPT_DIR",
    "USIM_FB_ALLOW_LEGACY_CKPT",
    "USIM_RUNNER_PATH",
    "USIM_N_EPOCHS",
    "USIM_EARLY_STOP_PATIENCE",
    "USIM_EARLY_STOP_MIN_DELTA",
    "USIM_EARLY_STOP_AVG_MODE",
    "USIM_EVAL_N_NEG",
    "USIM_RUN_SAMPLED_EVAL",
    "USIM_TRAIN_FORCE_COLD",
    "USIM_STEPS",
    "USIM_PPO_EPOCHS",
    "USIM_PPO_LAMBDA",
    "USIM_PPO_VALUE_CLIP",
    "USIM_PPO_ADV_NORM",
    "USIM_PPO_LOSS_WEIGHT",
    "USIM_RL_RESIDUAL_SCALE",
    "USIM_ROLLOUT_POLICY",
    "USIM_SIMULATOR_TARGET_MODE",
    "USIM_DETERMINISTIC_EVAL_CANDIDATES",
    "USIM_EVAL_REUSE_ITEM_BANK",
    "USIM_DETERMINISTIC_EVAL_SEED",
    "USIM_CKG_RL_V1",
    "USIM_V1_PSEUDO_COLD_PLAN_COUNT",
    "USIM_V1_PSEUDO_COLD_PLAN_SEED",
    "USIM_V1_PSEUDO_COLD_PLAN_STRATEGY",
    "USIM_V1_REFERENCE_BATCH_SIZE",
    "USIM_V1_TARGET_HISTORY_EXCLUSION",
    "USIM_V1_TARGET_HISTORY_EXCLUSION_SCOPE",
    "USIM_V1_SELECTOR_HOT_TOL",
    "USIM_V1_SELECTOR_OVERALL_TOL",
    "USIM_V1_SELECTOR_MODE",
    "USIM_FB_REWARD_DUP_W",
    "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET",
    "USIM_FAST3_TGT_ALPHA_COLD",
    "USIM_FAST3_TGT_ALPHA_HOT",
    "USIM_FAST3_TGT_ALPHA_STEP",
    "USIM_FAST3_TGT_ALPHA_ENT",
    "USIM_FAST3_TGT_ALPHA_MIN",
    "USIM_FAST3_TGT_ALPHA_MAX",
    "USIM_DISABLE_LLM_SCORE",
    "USIM_LLM_SAFE_MODE",
    "USIM_LLM_WEIGHT",
    "USIM_LLM_COLD_ONLY",
    "USIM_LLM_HOT_ONLY",
    "USIM_LLM_BANK_MODE",
    "USIM_USE_CONTENT_DELTA",
    "USIM_CONTENT_DELTA_MAX_NORM",
    "USIM_CONTENT_DELTA_MODE",
    "USIM_CONTENT_DELTA_PAPER_STYLE",
    "USIM_CONTENT_DELTA_REPLACE_ITEM",
    "USIM_CONTENT_DELTA_COLD_ONLY",
    "USIM_CONTENT_DELTA_NORMALIZE_BASE",
    "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT",
    "USIM_CONTENT_DELTA_SCALE",
    "USIM_CONTENT_DELTA_LR_MULT",
    "USIM_CONTENT_DELTA_L2_W",
    "USIM_CONTENT_DELTA_CAP_W",
    "USIM_CONTENT_DELTA_CAP_MARGIN",
    "USIM_CONTENT_DELTA_PROJECTOR_HIDDEN",
    "USIM_CONTENT_DELTA_AUX_MODE",
    "USIM_CONTENT_DELTA_EVAL_BANK_MODE",
    "USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT",
    "USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH",
    "USIM_AUX_WEIGHT",
    "USIM_AUX_HOT_ONLY",
    "USIM_EARLY_STOP_SCORE_MODE",
    "USIM_USE_PSEUDO_COLD_TRAIN",
    "USIM_USE_REFINED_EVAL",
    "USIM_PSEUDO_COLD_MODE",
    "USIM_PSEUDO_COLD_RATIO",
    "USIM_PSEUDO_COLD_MIN_POP",
    "USIM_USE_PAAC",
    "USIM_PAAC_ALIGN_W",
    "USIM_PAAC_CONTRAST_W",
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "USIM_PREREQ_GRAPH_SOURCE",
    "USIM_USE_COURSE_REWARD",
    "USIM_USE_PREREQ_AUX_LOSS",
    "USIM_PREREQ_AUX_ONLY_COLD",
    "USIM_FB_COURSE_ONLY_COLD",
    "USIM_FB_COURSE_PREREQ_W",
    "USIM_FB_COURSE_PREREQ_GATE",
    "USIM_FB_COURSE_CONCEPT_W",
    "USIM_FB_COURSE_MATCH_MODE",
    "USIM_FB_COURSE_MATCH_TOPK",
    "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET",
    "USIM_FB_COURSE_DIFF_W",
    "USIM_FB_COURSE_REDUNDANT_MODE",
    "USIM_FB_COURSE_REDUNDANT_W",
    "USIM_FB_COURSE_REDUNDANT_CONCEPT_GATE",
    "USIM_FB_COURSE_TERM_NORM",
    "USIM_FB_COURSE_TERM_NORM_CLIP",
    "USIM_FB_COURSE_TERM_NORM_EPS",
    "USIM_FB_COURSE_TERM_NORM_EMA_DECAY",
    "USIM_FB_COURSE_STRUCT_CHUNK",
    "USIM_FB_COURSE_SAMPLE_SOFT",
    "USIM_FB_COURSE_SAMPLE_BETA",
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD",
    "USIM_FB_COURSE_SAMPLE_TOPK",
    "USIM_FB_COURSE_SAMPLE_TOPL",
    "USIM_USE_SAGE_LITE",
    "USIM_SAGE_GATE_MIN",
    "USIM_SAGE_GATE_MAX",
    "USIM_SAGE_GATE_MODE",
    "USIM_SAGE_GATE_BUCKETS",
    "USIM_SAGE_GATE_HIDDEN",
    "USIM_SAGE_GATE_BUCKET_STRATEGY",
    "USIM_SAGE_POOL_TOPK",
    "USIM_SAGE_COURSE_TEMP",
    "USIM_SAGE_ONLY_COLD_OR_TAIL",
    "USIM_SAGE_TAIL_POP_RATIO",
    "USIM_SAGE_USE_TWO_EXPERT",
    "USIM_SAGE_TWO_EXPERT_SCORE_FUSION",
    "USIM_USE_SAGE_AUX_LOSS",
    "USIM_SAGE_AUX_WEIGHT",
    "USIM_SAGE_AUX_POOL_TOPK",
    "USIM_SAGE_AUX_COURSE_TEMP",
    "USIM_SAGE_AUX_RETRIEVAL_TEMP",
    "USIM_SAGE_AUX_ONLY_STRICT_COLD",
    "USIM_SAGE_AUX_DETACH_USER",
    "USIM_USE_CGRC_RECON",
    "USIM_CGRC_RECON_AUX_W",
    "USIM_CGRC_RECON_SAMPLE_W",
    "USIM_CGRC_RECON_PSEUDO_RATIO",
    "USIM_CGRC_RECON_TOPK",
    "USIM_CGRC_RECON_TEMP",
    "USIM_CGRC_RECON_ONLY_COLD_OR_TAIL",
    "USIM_CGRC_RECON_TAIL_POP_RATIO",
    "USIM_CGRC_RECON_DETACH_USER",
    "USIM_USE_COURSE_RERANK",
    "USIM_COURSE_RERANK_ONLY_COLD",
    "USIM_COURSE_RERANK_ALPHA",
    "USIM_COURSE_RERANK_LAMBDA",
    "USIM_COURSE_RERANK_TOPL",
    "USIM_USE_STRUCTURED_HARD_NEG",
    "USIM_MASK_KNOWN_POS_NEG",
    "USIM_MASK_SAME_ITEM_NEG",
    "USIM_USE_SG_URINIT",
    "USIM_SG_URINIT_CLUSTER_K",
    "USIM_SG_URINIT_LOCAL_W",
    "USIM_SG_URINIT_GLOBAL_W",
    "USIM_SG_URINIT_TARGET_NORM",
    "USIM_SG_URINIT_MAX_ITER",
    "USIM_SG_URINIT_SEED"
)

$originalEnv = @{}
foreach ($name in $trackedEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$checkpointRootExplicit = $PSBoundParameters.ContainsKey("CheckpointRoot")
$saveCkptExplicit = $PSBoundParameters.ContainsKey("SaveCkpt")
$checkpointSaveDefaulted = $false

$scriptText = Get-Content -Raw -Encoding UTF8 -LiteralPath $ScriptPath
$hasInlineStaticImplementation = (
    $scriptText -match "def run_static_experiment" -and
    $scriptText -match "_static_split_df"
)
$hasExplicitStaticDelegate = $scriptText -match "USIM_STATIC_DELEGATE_ENTRYPOINT\s*=\s*True"
if (-not $hasInlineStaticImplementation -and -not $hasExplicitStaticDelegate) {
    throw "Static runner guard failed: '$ScriptPath' does not contain the static experiment implementation."
}

if ($MinimalLiraMode) {
    $UseContentDelta = $false
    $AuxWeight = 0.0
    $UsePaac = $false
    $UseCourseFeedback = $false
    $UseCourseReward = $false
    $UsePrereqAux = $false
    $UseSageLite = $false
    $UseSageAuxLoss = $false
    $UseCgrcRecon = $false
    $UseCourseSample = $false
    $PpoLossWeight = 0.0
}

if ($Protocol -eq "strict_item_cold_balanced") {
    if (-not $PSBoundParameters.ContainsKey("OutputRoot")) {
        $OutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
    }
    if (-not $PSBoundParameters.ContainsKey("CheckpointRoot")) {
        $CheckpointRoot = "checkpoints\content_delta_pop5\static_item_cold_balanced"
    }
}

if ($checkpointRootExplicit -and -not $saveCkptExplicit) {
    $SaveCkpt = $true
    $checkpointSaveDefaulted = $true
}

$splitMode = switch ($Protocol) {
    "strict_item_cold" { "strict_item_cold" }
    "strict_item_cold_balanced" { "strict_item_cold_balanced" }
    default { "user_threshold_exact" }
}

$base = @{
    "USIM_DATA_DIR" = $DataDir
    "USIM_RELATION_DIR" = $RelationDir
    "USIM_STATIC" = "1"
    "USIM_STATIC_SPLIT_MODE" = $splitMode
    "USIM_STATIC_TRAIN_RATIO" = "0.8"
    "USIM_STATIC_VAL_RATIO" = "0.1"
    "USIM_STATIC_COLD_ITEM_RATIO" = "0.10"
    "USIM_STATIC_VAL_COLD_ITEM_RATIO" = "0.05"
    "USIM_STATIC_COLD_ITEM_MIN_INTER" = "5"
    "USIM_STATIC_COLD_ITEM_FOLDS" = "20"
    "USIM_STATIC_ARTIFACT_SOURCE" = "all_metadata"
    "USIM_STATIC_EXPORT_SPLIT" = "1"
    "USIM_STATIC_TEST_HISTORY" = "train_only"

    "USIM_FB_FORCE_FRESH" = if ($ForceFresh) { "1" } else { "0" }
    "USIM_FB_AUTO_RESUME" = if ($AutoResume) { "1" } else { "0" }
    "USIM_FB_SAVE_CKPT" = if ($SaveCkpt) { "1" } else { "0" }
    "USIM_FB_SAVE_OPT_STATE" = if ($SaveOptState) { "1" } else { "0" }
    "USIM_FB_ALLOW_LEGACY_CKPT" = if ($AllowLegacyCheckpoint) { "1" } else { "0" }
    "USIM_RUNNER_PATH" = $MyInvocation.MyCommand.Path

    "USIM_N_EPOCHS" = [string]$Epochs
    "USIM_EARLY_STOP_PATIENCE" = [string]$Patience
    "USIM_EARLY_STOP_MIN_DELTA" = "1e-4"
    "USIM_EARLY_STOP_AVG_MODE" = $EarlyStopAverageMode
    "USIM_EVAL_N_NEG" = "200"
    "USIM_RUN_SAMPLED_EVAL" = if ($RunSampledEval) { "1" } else { "0" }
    "USIM_TRAIN_FORCE_COLD" = if ($TrainForceCold) { "1" } else { "0" }
    "USIM_STEPS" = [string]$UsimSteps
    "USIM_PPO_EPOCHS" = [string]$PpoEpochs
    "USIM_PPO_LAMBDA" = "0.95"
    "USIM_PPO_VALUE_CLIP" = "0.20"
    "USIM_PPO_ADV_NORM" = "1"
    "USIM_PPO_LOSS_WEIGHT" = [string]$PpoLossWeight
    "USIM_FB_INIT_CKPT_DIR" = $InitCheckpointDir
    "USIM_RL_RESIDUAL_SCALE" = [string]$RlResidualScale
    "USIM_ROLLOUT_POLICY" = $RolloutPolicy
    "USIM_SIMULATOR_TARGET_MODE" = $SimulatorTargetMode
    "USIM_DETERMINISTIC_EVAL_CANDIDATES" = if ($DeterministicEvalCandidates) { "1" } else { "0" }
    "USIM_EVAL_REUSE_ITEM_BANK" = if ($EvalReuseItemBank) { "1" } else { "0" }
    "USIM_CKG_RL_V1" = if ($CkgRlV1) { "1" } else { "0" }
    "USIM_V1_PSEUDO_COLD_PLAN_COUNT" = [string]$V1PseudoColdPlanCount
    "USIM_V1_PSEUDO_COLD_PLAN_STRATEGY" = $V1PseudoColdPlanStrategy
    "USIM_V1_REFERENCE_BATCH_SIZE" = [string]$V1ReferenceBatchSize
    "USIM_V1_TARGET_HISTORY_EXCLUSION" = if ($V1TargetHistoryExclusion) { "1" } else { "0" }
    "USIM_V1_TARGET_HISTORY_EXCLUSION_SCOPE" = $V1TargetHistoryExclusionScope
    "USIM_V1_SELECTOR_HOT_TOL" = [string]$V1SelectorHotTolerance
    "USIM_V1_SELECTOR_OVERALL_TOL" = [string]$V1SelectorOverallTolerance
    "USIM_V1_SELECTOR_MODE" = $V1SelectorMode

    "USIM_FAST3_TGT_ALPHA_COLD" = "0.35"
    "USIM_FAST3_TGT_ALPHA_HOT" = "0.60"
    "USIM_FAST3_TGT_ALPHA_STEP" = "0.20"
    "USIM_FAST3_TGT_ALPHA_ENT" = "0.20"
    "USIM_FAST3_TGT_ALPHA_MIN" = "0.15"
    "USIM_FAST3_TGT_ALPHA_MAX" = "0.85"

    "USIM_DISABLE_LLM_SCORE" = "1"
    "USIM_LLM_SAFE_MODE" = "0"
    "USIM_LLM_WEIGHT" = "1.0"
    "USIM_LLM_COLD_ONLY" = "0"
    "USIM_LLM_HOT_ONLY" = "0"
    "USIM_LLM_BANK_MODE" = "none"

    "USIM_USE_CONTENT_DELTA" = if ($UseContentDelta) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_MODE" = $ContentDeltaMode
    "USIM_CONTENT_DELTA_MAX_NORM" = [string]$ContentDeltaMaxNorm
    "USIM_CONTENT_DELTA_PAPER_STYLE" = if ($ContentDeltaPaperStyle) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_REPLACE_ITEM" = if ($ContentDeltaReplaceItem) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_COLD_ONLY" = if ($ContentDeltaColdOnly) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_NORMALIZE_BASE" = "1"
    "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT" = "1"
    "USIM_CONTENT_DELTA_SCALE" = [string]$ContentDeltaScale
    "USIM_CONTENT_DELTA_LR_MULT" = [string]$ContentDeltaLrMult
    "USIM_CONTENT_DELTA_L2_W" = [string]$ContentDeltaL2W
    "USIM_CONTENT_DELTA_CAP_W" = [string]$ContentDeltaCapW
    "USIM_CONTENT_DELTA_CAP_MARGIN" = [string]$ContentDeltaCapMargin
    "USIM_CONTENT_DELTA_PROJECTOR_HIDDEN" = [string]$ContentDeltaProjectorHidden
    "USIM_CONTENT_DELTA_AUX_MODE" = $ContentDeltaAuxMode
    "USIM_CONTENT_DELTA_EVAL_BANK_MODE" = $ContentDeltaEvalBankMode
    "USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT" = if ($ContentDeltaTrainOnIdDropout) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH" = [string]$ContentDeltaOnlyAfterEpoch
    "USIM_AUX_WEIGHT" = [string]$AuxWeight
    "USIM_AUX_HOT_ONLY" = if ($AuxHotOnly) { "1" } else { "0" }
    "USIM_EARLY_STOP_SCORE_MODE" = $EarlyStopScoreMode
    "USIM_USE_PSEUDO_COLD_TRAIN" = if ($UsePseudoColdTrain) { "1" } else { "0" }
    "USIM_USE_REFINED_EVAL" = if ($UseUsimRefinedEval) { "1" } else { "0" }
    "USIM_PSEUDO_COLD_MODE" = $PseudoColdMode
    "USIM_PSEUDO_COLD_RATIO" = [string]$PseudoColdRatio
    "USIM_PSEUDO_COLD_MIN_POP" = [string]$PseudoColdMinPop

    "USIM_USE_PAAC" = if ($UsePaac) { "1" } else { "0" }
    "USIM_PAAC_ALIGN_W" = [string]$PaacAlignW
    "USIM_PAAC_CONTRAST_W" = [string]$PaacContrastW

    "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
    "USIM_PREREQ_GRAPH_SOURCE" = $PrereqGraphSource
    "USIM_USE_COURSE_REWARD" = if ($UseCourseReward) { "1" } else { "0" }
    "USIM_FB_REWARD_DUP_W" = [string]$RewardDupW
    "USIM_USE_PREREQ_AUX_LOSS" = if ($UsePrereqAux) { "1" } else { "0" }
    "USIM_PREREQ_AUX_ONLY_COLD" = if ($PrereqAuxOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_ONLY_COLD" = if ($CourseFeedbackOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_PREREQ_W" = if ($UseCourseFeedback) { [string]$CoursePrereqW } else { "0" }
    "USIM_FB_COURSE_PREREQ_GATE" = if ($UseCourseFeedback) { [string]$CoursePrereqGate } else { "1.0" }
    "USIM_FB_COURSE_CONCEPT_W" = if ($UseCourseFeedback) { [string]$CourseConceptW } else { "0" }
    "USIM_FB_COURSE_MATCH_MODE" = $CourseMatchMode
    "USIM_FB_COURSE_MATCH_TOPK" = [string]$CourseMatchTopK
    "USIM_FB_COURSE_DIFF_W" = if ($UseCourseFeedback) { [string]$CourseDiffW } else { "0" }
    "USIM_FB_COURSE_REDUNDANT_MODE" = $CourseRedundantMode
    "USIM_FB_COURSE_REDUNDANT_W" = if ($UseCourseFeedback) { [string]$CourseRedundantW } else { "0" }
    "USIM_FB_COURSE_REDUNDANT_CONCEPT_GATE" = if ($UseCourseFeedback) { [string]$CourseRedundantConceptGate } else { "0" }
    "USIM_FB_COURSE_TERM_NORM" = $CourseTermNorm
    "USIM_FB_COURSE_TERM_NORM_CLIP" = [string]$CourseTermNormClip
    "USIM_FB_COURSE_TERM_NORM_EPS" = [string]$CourseTermNormEps
    "USIM_FB_COURSE_TERM_NORM_EMA_DECAY" = [string]$CourseTermNormEmaDecay
    "USIM_FB_COURSE_STRUCT_CHUNK" = [string]$CourseStructChunk

    "USIM_FB_COURSE_SAMPLE_SOFT" = if ($UseCourseSample) { "1" } else { "0" }
    "USIM_FB_COURSE_SAMPLE_BETA" = if ($UseCourseSample) { [string]$CourseSampleBeta } else { "0" }
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD" = if ($CourseSampleOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
    "USIM_FB_COURSE_SAMPLE_TOPL" = "32"
    "USIM_USE_SAGE_LITE" = if ($UseSageLite) { "1" } else { "0" }
    "USIM_SAGE_GATE_MIN" = [string]$SageGateMin
    "USIM_SAGE_GATE_MAX" = [string]$SageGateMax
    "USIM_SAGE_GATE_MODE" = $SageGateMode
    "USIM_SAGE_GATE_BUCKETS" = [string]$SageGateBuckets
    "USIM_SAGE_GATE_HIDDEN" = [string]$SageGateHidden
    "USIM_SAGE_GATE_BUCKET_STRATEGY" = $SageGateBucketStrategy
    "USIM_SAGE_POOL_TOPK" = [string]$SagePoolTopK
    "USIM_SAGE_COURSE_TEMP" = [string]$SageCourseTemp
    "USIM_SAGE_ONLY_COLD_OR_TAIL" = if ($SageOnlyColdOrTail) { "1" } else { "0" }
    "USIM_SAGE_TAIL_POP_RATIO" = [string]$SageTailPopRatio
    "USIM_SAGE_USE_TWO_EXPERT" = if ($SageUseTwoExpert) { "1" } else { "0" }
    "USIM_SAGE_TWO_EXPERT_SCORE_FUSION" = if ($SageTwoExpertScoreFusion) { "1" } else { "0" }
    "USIM_USE_SAGE_AUX_LOSS" = if ($UseSageAuxLoss) { "1" } else { "0" }
    "USIM_SAGE_AUX_WEIGHT" = [string]$SageAuxWeight
    "USIM_SAGE_AUX_POOL_TOPK" = [string]$SageAuxPoolTopK
    "USIM_SAGE_AUX_COURSE_TEMP" = [string]$SageAuxCourseTemp
    "USIM_SAGE_AUX_RETRIEVAL_TEMP" = [string]$SageAuxRetrievalTemp
    "USIM_SAGE_AUX_ONLY_STRICT_COLD" = if ($SageAuxOnlyStrictCold) { "1" } else { "0" }
    "USIM_SAGE_AUX_DETACH_USER" = if ($SageAuxDetachUser) { "1" } else { "0" }
    "USIM_USE_CGRC_RECON" = if ($UseCgrcRecon) { "1" } else { "0" }
    "USIM_CGRC_RECON_AUX_W" = [string]$CgrcReconAuxW
    "USIM_CGRC_RECON_SAMPLE_W" = [string]$CgrcReconSampleW
    "USIM_CGRC_RECON_PSEUDO_RATIO" = [string]$CgrcReconPseudoRatio
    "USIM_CGRC_RECON_TOPK" = [string]$CgrcReconTopK
    "USIM_CGRC_RECON_TEMP" = [string]$CgrcReconTemp
    "USIM_CGRC_RECON_ONLY_COLD_OR_TAIL" = if ($CgrcReconOnlyColdOrTail) { "1" } else { "0" }
    "USIM_CGRC_RECON_TAIL_POP_RATIO" = [string]$CgrcReconTailPopRatio
    "USIM_CGRC_RECON_DETACH_USER" = if ($CgrcReconDetachUser) { "1" } else { "0" }

    "USIM_USE_COURSE_RERANK" = if ($UseCourseRerank) { "1" } else { "0" }
    "USIM_COURSE_RERANK_ONLY_COLD" = if ($CourseRerankOnlyCold) { "1" } else { "0" }
    "USIM_COURSE_RERANK_ALPHA" = [string]$CourseRerankAlpha
    "USIM_COURSE_RERANK_LAMBDA" = [string]$CourseRerankLambda
    "USIM_COURSE_RERANK_TOPL" = [string]$CourseRerankTopL
    "USIM_USE_STRUCTURED_HARD_NEG" = if ($UseStructuredHardNeg) { "1" } else { "0" }
    "USIM_MASK_KNOWN_POS_NEG" = if ($MaskKnownPosNeg) { "1" } else { "0" }
    "USIM_MASK_SAME_ITEM_NEG" = if ($MaskSameItemNeg) { "1" } else { "0" }
    "USIM_USE_SG_URINIT" = if ($UseSgUrinit) { "1" } else { "0" }
    "USIM_SG_URINIT_CLUSTER_K" = [string]$SgUrinitClusterK
    "USIM_SG_URINIT_LOCAL_W" = [string]$SgUrinitLocalW
    "USIM_SG_URINIT_GLOBAL_W" = [string]$SgUrinitGlobalW
    "USIM_SG_URINIT_TARGET_NORM" = [string]$SgUrinitTargetNorm
    "USIM_SG_URINIT_MAX_ITER" = [string]$SgUrinitMaxIter
}

if ($DryRun) {
    Write-Host "FAST3 static runner dry run"
    Write-Host ("Protocol={0} Seeds={1} Epochs={2} UsimSteps={3} V1={4} PseudoMode={5} SelectorMode={6}" -f $Protocol, ($Seeds -join ","), $Epochs, $UsimSteps, $CkgRlV1, $PseudoColdMode, $V1SelectorMode)
    Write-Host ("OutputRoot={0} CheckpointRoot={1}" -f $OutputRoot, $CheckpointRoot)
    return
}

try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

    if ($checkpointSaveDefaulted) {
        Write-Host "CheckpointRoot was provided without -SaveCkpt; enabling checkpoint saving by default." -ForegroundColor Yellow
    }

    foreach ($coldThreshold in $ColdThresholds) {
        foreach ($seed in $Seeds) {
            $tag = "${Protocol}_thr${coldThreshold}_seed_$seed"
            $out = Join-Path $OutputRoot $tag
            $ckpt = Join-Path $CheckpointRoot $tag
            $log = Join-Path $out "run.log"
            New-Item -ItemType Directory -Force -Path $out | Out-Null
            New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

            foreach ($key in $base.Keys) {
                Set-Item "Env:$key" ([string]$base[$key])
            }
            if (-not [string]::IsNullOrWhiteSpace($CourseMatchExcludeTarget)) {
                $env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = $CourseMatchExcludeTarget
            }
            $env:USIM_COLD_THRESHOLD = [string]$coldThreshold
            $env:USIM_STATIC_SEED = [string]$seed
            $env:USIM_SEED = [string]$seed
            $env:USIM_SG_URINIT_SEED = [string]$seed
            $env:USIM_DETERMINISTIC_EVAL_SEED = if ($DeterministicEvalSeed -gt 0) {
                [string]$DeterministicEvalSeed
            } else {
                [string]$seed
            }
            $env:USIM_V1_PSEUDO_COLD_PLAN_SEED = if ($V1PseudoColdPlanSeed -gt 0) {
                [string]$V1PseudoColdPlanSeed
            } else {
                [string]$seed
            }
            $env:USIM_FB_OUTPUT_TAG = $tag
            $env:USIM_FB_OUTPUT_DIR = $out
            $env:USIM_FB_CKPT_DIR = $ckpt

            Write-Host ""
            Write-Host "===== Running FAST3 ContentDelta static threshold=$coldThreshold seed=$seed epochs=$Epochs =====" -ForegroundColor Cyan
            Write-Host "Output: $out"
            Write-Host ("Tune: es_avg={0} es_score={29} aux_hot_only={30} run_sampled_eval={31} delta={1} delta_mode={2} paper={3} replace_item={4} cold_only={5} train_on_id_dropout={6} delta_only_after_epoch={7} delta_max={8} scale={9} lr_mult={10} l2={11} cap={12} cap_margin={33} proj_hidden={34} aux_mode={35} eval_bank={36} aux_w={13} pseudo={14} pseudo_mode={15} pseudo_ratio={16} pseudo_min_pop={17} paac={18} paac_align={19} paac_contrast={20} course_feedback={21} course_reward={32} course_only_cold={22} prereq_aux={23} prereq_only_cold={24} course_sample={25} sample_only_cold={26} rerank={27} structured_hard_neg={28} mask_known_pos={37} mask_same_item={38}" -f $EarlyStopAverageMode, $UseContentDelta, $ContentDeltaMode, $ContentDeltaPaperStyle, $ContentDeltaReplaceItem, $ContentDeltaColdOnly, $ContentDeltaTrainOnIdDropout, $ContentDeltaOnlyAfterEpoch, $ContentDeltaMaxNorm, $ContentDeltaScale, $ContentDeltaLrMult, $ContentDeltaL2W, $ContentDeltaCapW, $AuxWeight, $UsePseudoColdTrain, $PseudoColdMode, $PseudoColdRatio, $PseudoColdMinPop, $UsePaac, $PaacAlignW, $PaacContrastW, $UseCourseFeedback, $CourseFeedbackOnlyCold, $UsePrereqAux, $PrereqAuxOnlyCold, $UseCourseSample, $CourseSampleOnlyCold, $UseCourseRerank, $UseStructuredHardNeg, $EarlyStopScoreMode, $AuxHotOnly, $RunSampledEval, $UseCourseReward, $ContentDeltaCapMargin, $ContentDeltaProjectorHidden, $ContentDeltaAuxMode, $ContentDeltaEvalBankMode, $MaskKnownPosNeg, $MaskSameItemNeg)
            $matchExcludeTarget = if ([string]::IsNullOrWhiteSpace($env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET)) { "auto" } else { $env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET }
            Write-Host ("Core controls: train_force_cold={0} usim_steps={1}" -f $TrainForceCold, $UsimSteps)
            Write-Host ("PPO controls: loss_weight={0}" -f $PpoLossWeight)
            Write-Host ("RL rescue: init_checkpoint={0} residual_scale={1}" -f $InitCheckpointDir, $RlResidualScale)
            Write-Host ("Rollout policy: {0}" -f $RolloutPolicy)
            Write-Host ("Semantic repair: target_mode={0} deterministic_candidates={1} reuse_item_bank={2} eval_seed={3}" -f $SimulatorTargetMode, $DeterministicEvalCandidates, $EvalReuseItemBank, $env:USIM_DETERMINISTIC_EVAL_SEED)
            Write-Host ("Course match: mode={0} topk={1} exclude_target={2}" -f $CourseMatchMode, $CourseMatchTopK, $matchExcludeTarget)
            Write-Host ("Course redundancy: mode={0} struct_chunk={1}" -f $CourseRedundantMode, $CourseStructChunk)
            Write-Host ("Course term norm: mode={0} clip={1} eps={2} ema_decay={3}" -f $CourseTermNorm, $CourseTermNormClip, $CourseTermNormEps, $CourseTermNormEmaDecay)
            Write-Host ("SAGE-lite: enabled={0} gate_mode={1} gate=[{2},{3}] buckets={4} bucket_strategy={5} gate_hidden={6} pool_topk={7} course_temp={8} only_cold_or_tail={9} tail_pop_ratio={10} two_expert={11} score_fusion={12}" -f $UseSageLite, $SageGateMode, $SageGateMin, $SageGateMax, $SageGateBuckets, $SageGateBucketStrategy, $SageGateHidden, $SagePoolTopK, $SageCourseTemp, $SageOnlyColdOrTail, $SageTailPopRatio, $SageUseTwoExpert, $SageTwoExpertScoreFusion)
            Write-Host ("SAGE-aux: enabled={0} weight={1} pool_topk={2} course_temp={3} retrieval_temp={4} strict_cold_only={5} detach_user={6}" -f $UseSageAuxLoss, $SageAuxWeight, $SageAuxPoolTopK, $SageAuxCourseTemp, $SageAuxRetrievalTemp, $SageAuxOnlyStrictCold, $SageAuxDetachUser)
            Write-Host ("CGRC-recon: enabled={0} aux_w={1} sample_w={2} pseudo_ratio={3} topk={4} temp={5} only_cold_or_tail={6} tail_pop_ratio={7} detach_user={8}" -f $UseCgrcRecon, $CgrcReconAuxW, $CgrcReconSampleW, $CgrcReconPseudoRatio, $CgrcReconTopK, $CgrcReconTemp, $CgrcReconOnlyColdOrTail, $CgrcReconTailPopRatio, $CgrcReconDetachUser)
            Write-Host ("SG-URInit: enabled={0} cluster_k={1} local_w={2} global_w={3} target_norm={4} max_iter={5} seed={6}" -f $UseSgUrinit, $SgUrinitClusterK, $SgUrinitLocalW, $SgUrinitGlobalW, $SgUrinitTargetNorm, $SgUrinitMaxIter, $seed)
            Write-Host ("Checkpoint: save={0} resume={1} force_fresh={2} save_opt={3} dir={4}" -f $SaveCkpt, $AutoResume, $ForceFresh, $SaveOptState, $ckpt)

            $commandLine = ('"{0}" -u -X faulthandler "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
            $teeArgs = @{ FilePath = $log }
            if ($AutoResume -and -not $ForceFresh) {
                $teeArgs["Append"] = $true
            }
            & cmd.exe /d /c $commandLine | Tee-Object @teeArgs

            if ($LASTEXITCODE -ne 0) {
                throw "Static experiment failed: threshold=$coldThreshold seed=$seed"
            }
        }
    }

    if (-not $SkipAggregate) {
        Write-Host ""
        Write-Host "===== Aggregating FAST3 static results =====" -ForegroundColor Cyan
        & $PythonRunner "aggregate_fast3_static_results.py" --root $OutputRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Static result aggregation failed"
        }
    }
}
finally {
    foreach ($name in $trackedEnv) {
        if ($null -eq $originalEnv[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $originalEnv[$name]
        }
    }
}
