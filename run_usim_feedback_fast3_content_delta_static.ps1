param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$OutputRoot = "outputs\content_delta_pop5\static_item_cold",
    [string]$CheckpointRoot = "checkpoints\content_delta_pop5\static_item_cold",
    [ValidateSet("threshold", "strict_item_cold")]
    [string]$Protocol = "strict_item_cold",
    [Alias("ColdThreshold")]
    [int[]]$ColdThresholds = @(1),
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 15,
    [int]$Patience = 3,
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
    "USIM_EVAL_N_NEG",
    "USIM_RUN_SAMPLED_EVAL",
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
    "USIM_CONTENT_DELTA_COLD_ONLY",
    "USIM_CONTENT_DELTA_NORMALIZE_BASE",
    "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT",
    "USIM_USE_PAAC",
    "USIM_PAAC_ALIGN_W",
    "USIM_PAAC_CONTRAST_W",
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "USIM_PREREQ_GRAPH_SOURCE",
    "USIM_USE_PREREQ_AUX_LOSS",
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

$splitMode = if ($Protocol -eq "strict_item_cold") { "strict_item_cold" } else { "user_threshold_exact" }

$base = @{
    "USIM_DATA_DIR" = "processed_data_hin_clean_pop5"
    "USIM_STATIC" = "1"
    "USIM_STATIC_SPLIT_MODE" = $splitMode
    "USIM_STATIC_TRAIN_RATIO" = "0.8"
    "USIM_STATIC_VAL_RATIO" = "0.1"
    "USIM_STATIC_COLD_ITEM_RATIO" = "0.10"
    "USIM_STATIC_VAL_COLD_ITEM_RATIO" = "0.05"
    "USIM_STATIC_COLD_ITEM_MIN_INTER" = "5"
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
    "USIM_EVAL_N_NEG" = "200"
    "USIM_RUN_SAMPLED_EVAL" = "1"
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

    "USIM_USE_CONTENT_DELTA" = "1"
    "USIM_CONTENT_DELTA_MAX_NORM" = "0.05"
    "USIM_CONTENT_DELTA_COLD_ONLY" = "1"
    "USIM_CONTENT_DELTA_NORMALIZE_BASE" = "1"
    "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT" = "1"

    "USIM_USE_PAAC" = "0"
    "USIM_PAAC_ALIGN_W" = "0.0"
    "USIM_PAAC_CONTRAST_W" = "0.0"

    "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
    "USIM_PREREQ_GRAPH_SOURCE" = "concept"
    "USIM_USE_PREREQ_AUX_LOSS" = "1"
    "USIM_FB_COURSE_PREREQ_W" = "0.08"
    "USIM_FB_COURSE_CONCEPT_W" = "0.04"
    "USIM_FB_COURSE_DIFF_W" = "0.03"
    "USIM_FB_COURSE_REDUNDANT_W" = "0.02"

    "USIM_FB_COURSE_SAMPLE_SOFT" = "1"
    "USIM_FB_COURSE_SAMPLE_BETA" = "0.20"
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD" = "1"
    "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
    "USIM_FB_COURSE_SAMPLE_TOPL" = "32"

    "USIM_USE_COURSE_RERANK" = "0"
    "USIM_USE_STRUCTURED_HARD_NEG" = "0"
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

            $commandLine = ('"{0}" -u "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
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
