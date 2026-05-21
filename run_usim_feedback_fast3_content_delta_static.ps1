param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
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
    [string]$EarlyStopAverageMode = "interaction",
    [ValidateSet("embedding", "projector", "hybrid")]
    [string]$ContentDeltaMode = "embedding",
    [double]$ContentDeltaMaxNorm = 0.05,
    [double]$ContentDeltaScale = 0.25,
    [double]$ContentDeltaLrMult = 0.10,
    [double]$ContentDeltaL2W = 0.02,
    [double]$ContentDeltaCapW = 0.02,
    [bool]$UseContentDelta = $true,
    [bool]$ContentDeltaPaperStyle = $false,
    [bool]$ContentDeltaReplaceItem = $false,
    [bool]$ContentDeltaColdOnly = $true,
    [bool]$ContentDeltaTrainOnIdDropout = $true,
    [int]$ContentDeltaOnlyAfterEpoch = 0,
    [double]$AuxWeight = 0.3,
    [bool]$UsePseudoColdTrain = $false,
    [ValidateSet("batch_random", "batch_tail", "all_eligible", "none", "off")]
    [string]$PseudoColdMode = "batch_random",
    [double]$PseudoColdRatio = 0.30,
    [int]$PseudoColdMinPop = 5,
    [bool]$UsePaac = $false,
    [double]$PaacAlignW = 0.0,
    [double]$PaacContrastW = 0.0,
    [bool]$UseCourseFeedback = $true,
    [bool]$UseCourseReward = $true,
    [bool]$UsePrereqAux = $true,
    [bool]$PrereqAuxOnlyCold = $true,
    [double]$CoursePrereqW = 0.08,
    [double]$CourseConceptW = 0.04,
    [double]$CourseDiffW = 0.03,
    [double]$CourseRedundantW = 0.02,
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
    # Cold-start patch (2026-05-19) flags. Legacy defaults preserved when the
    # caller doesn't pass them; see docs/COLD_START_PATCH_2026_05_19.md.
    [bool]$AuxHotOnly = $false,
    [ValidateSet("cold_only", "geometric", "harmonic", "sum")]
    [string]$EarlyStopScoreMode = "cold_only",
    [bool]$RunSampledEval = $false,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$trackedEnv = @(
    "USIM_DATA_DIR",
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
    "USIM_N_EPOCHS",
    "USIM_EARLY_STOP_PATIENCE",
    "USIM_EARLY_STOP_MIN_DELTA",
    "USIM_EARLY_STOP_AVG_MODE",
    "USIM_EVAL_N_NEG",
    "USIM_RUN_SAMPLED_EVAL",
    "USIM_TRAIN_FORCE_COLD",
    "USIM_PPO_EPOCHS",
    "USIM_PPO_LAMBDA",
    "USIM_PPO_VALUE_CLIP",
    "USIM_PPO_ADV_NORM",
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
    "USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT",
    "USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH",
    "USIM_AUX_WEIGHT",
    "USIM_AUX_HOT_ONLY",
    "USIM_EARLY_STOP_SCORE_MODE",
    "USIM_USE_PSEUDO_COLD_TRAIN",
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
    "USIM_FB_COURSE_CONCEPT_W",
    "USIM_FB_COURSE_DIFF_W",
    "USIM_FB_COURSE_REDUNDANT_W",
    "USIM_FB_COURSE_SAMPLE_SOFT",
    "USIM_FB_COURSE_SAMPLE_BETA",
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD",
    "USIM_FB_COURSE_SAMPLE_TOPK",
    "USIM_FB_COURSE_SAMPLE_TOPL",
    "USIM_USE_COURSE_RERANK",
    "USIM_COURSE_RERANK_ONLY_COLD",
    "USIM_COURSE_RERANK_ALPHA",
    "USIM_COURSE_RERANK_LAMBDA",
    "USIM_COURSE_RERANK_TOPL",
    "USIM_USE_STRUCTURED_HARD_NEG"
)

$originalEnv = @{}
foreach ($name in $trackedEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$scriptText = Get-Content -Raw -Encoding UTF8 -LiteralPath $ScriptPath
if ($scriptText -notmatch "def run_static_experiment" -or $scriptText -notmatch "def _static_split_df") {
    throw "Static runner guard failed: '$ScriptPath' does not contain the static experiment implementation."
}

if ($Protocol -eq "strict_item_cold_balanced") {
    if (-not $PSBoundParameters.ContainsKey("OutputRoot")) {
        $OutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
    }
    if (-not $PSBoundParameters.ContainsKey("CheckpointRoot")) {
        $CheckpointRoot = "checkpoints\content_delta_pop5\static_item_cold_balanced"
    }
}

$splitMode = switch ($Protocol) {
    "strict_item_cold" { "strict_item_cold" }
    "strict_item_cold_balanced" { "strict_item_cold_balanced" }
    default { "user_threshold_exact" }
}

$base = @{
    "USIM_DATA_DIR" = "processed_data_hin_clean_pop5"
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

    "USIM_FB_FORCE_FRESH" = "1"
    "USIM_FB_AUTO_RESUME" = "0"
    "USIM_FB_SAVE_CKPT" = "0"
    "USIM_FB_SAVE_OPT_STATE" = "0"

    "USIM_N_EPOCHS" = [string]$Epochs
    "USIM_EARLY_STOP_PATIENCE" = [string]$Patience
    "USIM_EARLY_STOP_MIN_DELTA" = "1e-4"
    "USIM_EARLY_STOP_AVG_MODE" = $EarlyStopAverageMode
    "USIM_EVAL_N_NEG" = "200"
    "USIM_RUN_SAMPLED_EVAL" = if ($RunSampledEval) { "1" } else { "0" }
    "USIM_TRAIN_FORCE_COLD" = "1"
    "USIM_PPO_EPOCHS" = "2"
    "USIM_PPO_LAMBDA" = "0.95"
    "USIM_PPO_VALUE_CLIP" = "0.20"
    "USIM_PPO_ADV_NORM" = "1"

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
    "USIM_CONTENT_DELTA_TRAIN_ON_ID_DROPOUT" = if ($ContentDeltaTrainOnIdDropout) { "1" } else { "0" }
    "USIM_CONTENT_DELTA_ONLY_AFTER_EPOCH" = [string]$ContentDeltaOnlyAfterEpoch
    "USIM_AUX_WEIGHT" = [string]$AuxWeight
    "USIM_AUX_HOT_ONLY" = if ($AuxHotOnly) { "1" } else { "0" }
    "USIM_EARLY_STOP_SCORE_MODE" = $EarlyStopScoreMode
    "USIM_USE_PSEUDO_COLD_TRAIN" = if ($UsePseudoColdTrain) { "1" } else { "0" }
    "USIM_PSEUDO_COLD_MODE" = $PseudoColdMode
    "USIM_PSEUDO_COLD_RATIO" = [string]$PseudoColdRatio
    "USIM_PSEUDO_COLD_MIN_POP" = [string]$PseudoColdMinPop

    "USIM_USE_PAAC" = if ($UsePaac) { "1" } else { "0" }
    "USIM_PAAC_ALIGN_W" = [string]$PaacAlignW
    "USIM_PAAC_CONTRAST_W" = [string]$PaacContrastW

    "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
    "USIM_PREREQ_GRAPH_SOURCE" = "concept"
    "USIM_USE_COURSE_REWARD" = if ($UseCourseReward) { "1" } else { "0" }
    "USIM_USE_PREREQ_AUX_LOSS" = if ($UsePrereqAux) { "1" } else { "0" }
    "USIM_PREREQ_AUX_ONLY_COLD" = if ($PrereqAuxOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_ONLY_COLD" = if ($CourseFeedbackOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_PREREQ_W" = if ($UseCourseFeedback) { [string]$CoursePrereqW } else { "0" }
    "USIM_FB_COURSE_CONCEPT_W" = if ($UseCourseFeedback) { [string]$CourseConceptW } else { "0" }
    "USIM_FB_COURSE_DIFF_W" = if ($UseCourseFeedback) { [string]$CourseDiffW } else { "0" }
    "USIM_FB_COURSE_REDUNDANT_W" = if ($UseCourseFeedback) { [string]$CourseRedundantW } else { "0" }

    "USIM_FB_COURSE_SAMPLE_SOFT" = if ($UseCourseSample) { "1" } else { "0" }
    "USIM_FB_COURSE_SAMPLE_BETA" = if ($UseCourseSample) { [string]$CourseSampleBeta } else { "0" }
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD" = if ($CourseSampleOnlyCold) { "1" } else { "0" }
    "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
    "USIM_FB_COURSE_SAMPLE_TOPL" = "32"

    "USIM_USE_COURSE_RERANK" = if ($UseCourseRerank) { "1" } else { "0" }
    "USIM_COURSE_RERANK_ONLY_COLD" = if ($CourseRerankOnlyCold) { "1" } else { "0" }
    "USIM_COURSE_RERANK_ALPHA" = [string]$CourseRerankAlpha
    "USIM_COURSE_RERANK_LAMBDA" = [string]$CourseRerankLambda
    "USIM_COURSE_RERANK_TOPL" = [string]$CourseRerankTopL
    "USIM_USE_STRUCTURED_HARD_NEG" = if ($UseStructuredHardNeg) { "1" } else { "0" }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

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
            $env:USIM_COLD_THRESHOLD = [string]$coldThreshold
            $env:USIM_STATIC_SEED = [string]$seed
            $env:USIM_SEED = [string]$seed
            $env:USIM_FB_OUTPUT_TAG = $tag
            $env:USIM_FB_OUTPUT_DIR = $out
            $env:USIM_FB_CKPT_DIR = $ckpt

            Write-Host ""
            Write-Host "===== Running FAST3 ContentDelta static threshold=$coldThreshold seed=$seed epochs=$Epochs =====" -ForegroundColor Cyan
            Write-Host "Output: $out"
            Write-Host ("Tune: es_avg={0} es_score={29} aux_hot_only={30} run_sampled_eval={31} delta={1} delta_mode={2} paper={3} replace_item={4} cold_only={5} train_on_id_dropout={6} delta_only_after_epoch={7} delta_max={8} scale={9} lr_mult={10} l2={11} cap={12} aux_w={13} pseudo={14} pseudo_mode={15} pseudo_ratio={16} pseudo_min_pop={17} paac={18} paac_align={19} paac_contrast={20} course_feedback={21} course_reward={32} course_only_cold={22} prereq_aux={23} prereq_only_cold={24} course_sample={25} sample_only_cold={26} rerank={27} structured_hard_neg={28}" -f $EarlyStopAverageMode, $UseContentDelta, $ContentDeltaMode, $ContentDeltaPaperStyle, $ContentDeltaReplaceItem, $ContentDeltaColdOnly, $ContentDeltaTrainOnIdDropout, $ContentDeltaOnlyAfterEpoch, $ContentDeltaMaxNorm, $ContentDeltaScale, $ContentDeltaLrMult, $ContentDeltaL2W, $ContentDeltaCapW, $AuxWeight, $UsePseudoColdTrain, $PseudoColdMode, $PseudoColdRatio, $PseudoColdMinPop, $UsePaac, $PaacAlignW, $PaacContrastW, $UseCourseFeedback, $CourseFeedbackOnlyCold, $UsePrereqAux, $PrereqAuxOnlyCold, $UseCourseSample, $CourseSampleOnlyCold, $UseCourseRerank, $UseStructuredHardNeg, $EarlyStopScoreMode, $AuxHotOnly, $RunSampledEval, $UseCourseReward)

            $commandLine = ('"{0}" -u -X faulthandler "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
            & cmd.exe /d /c $commandLine | Tee-Object -FilePath $log

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
