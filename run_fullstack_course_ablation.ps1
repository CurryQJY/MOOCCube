param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$Root = "outputs\content_delta_pop5\fullstack_course_ablation",
    [string]$CkptRoot = "checkpoints\content_delta_pop5\fullstack_course_ablation"
)

$ErrorActionPreference = "Stop"

function Merge-Env {
    param(
        [hashtable]$Base,
        [hashtable]$Override
    )

    $merged = @{}
    foreach ($key in $Base.Keys) {
        $merged[$key] = $Base[$key]
    }
    foreach ($key in $Override.Keys) {
        $merged[$key] = $Override[$key]
    }
    return $merged
}

$base = @{
    "USIM_DATA_DIR" = "processed_data_hin_clean_pop5"

    "USIM_FB_FORCE_FRESH" = "1"
    "USIM_FB_AUTO_RESUME" = "0"
    "USIM_STATIC" = "0"

    "USIM_N_EPOCHS" = "3"
    "USIM_TRAIN_WINDOW" = "24"
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

    "USIM_USE_CONTENT_DELTA" = "0"
    "USIM_CONTENT_DELTA_MAX_NORM" = "0.05"
    "USIM_CONTENT_DELTA_COLD_ONLY" = "1"
    "USIM_CONTENT_DELTA_NORMALIZE_BASE" = "1"
    "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT" = "1"

    "USIM_USE_PAAC" = "0"
    "USIM_PAAC_ALIGN_W" = "0.0"
    "USIM_PAAC_CONTRAST_W" = "0.0"

    "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
    "USIM_FB_COURSE_ONLY_COLD" = "1"
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

$experiments = @(
    @{ name = "full_stack_recheck"; override = @{} },
    @{ name = "wo_prereq_aux"; override = @{ "USIM_USE_PREREQ_AUX_LOSS" = "0" } },
    @{ name = "wo_prereq_reward"; override = @{ "USIM_FB_COURSE_PREREQ_W" = "0.0" } },
    @{ name = "wo_concept_reward"; override = @{ "USIM_FB_COURSE_CONCEPT_W" = "0.0" } },
    @{ name = "wo_difficulty_penalty"; override = @{ "USIM_FB_COURSE_DIFF_W" = "0.0" } },
    @{ name = "wo_redundant_penalty"; override = @{ "USIM_FB_COURSE_REDUNDANT_W" = "0.0" } },
    @{
        name = "wo_course_soft_rerank"
        override = @{
            "USIM_FB_COURSE_SAMPLE_SOFT" = "0"
            "USIM_FB_COURSE_SAMPLE_BETA" = "0.0"
        }
    },
    @{
        name = "wo_course_info"
        override = @{
            "USIM_FB_LOAD_COURSE_ARTIFACTS" = "0"
            "USIM_USE_PREREQ_AUX_LOSS" = "0"
            "USIM_FB_COURSE_PREREQ_W" = "0.0"
            "USIM_FB_COURSE_CONCEPT_W" = "0.0"
            "USIM_FB_COURSE_DIFF_W" = "0.0"
            "USIM_FB_COURSE_REDUNDANT_W" = "0.0"
            "USIM_FB_COURSE_SAMPLE_SOFT" = "0"
            "USIM_FB_COURSE_SAMPLE_BETA" = "0.0"
        }
    }
)

New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path $CkptRoot | Out-Null

foreach ($exp in $experiments) {
    $name = [string]$exp.name
    $out = Join-Path $Root $name
    $ckpt = Join-Path $CkptRoot $name
    $log = Join-Path $out "run.log"

    New-Item -ItemType Directory -Force -Path $out | Out-Null
    New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

    $envs = Merge-Env $base $exp.override
    $envs["USIM_FB_OUTPUT_TAG"] = $name
    $envs["USIM_FB_OUTPUT_DIR"] = $out
    $envs["USIM_FB_CKPT_DIR"] = $ckpt

    foreach ($key in $envs.Keys) {
        Set-Item "Env:$key" ([string]$envs[$key])
    }

    Write-Host ""
    Write-Host "===== Running $name =====" -ForegroundColor Cyan
    Write-Host "Output: $out"

    $commandLine = ('"{0}" -u "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
    & cmd.exe /d /c $commandLine | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $name"
    }
}
