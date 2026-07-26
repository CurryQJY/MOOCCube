param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$OutputDir = "outputs\content_delta_pop5\fullstack_hybrid_prereq_legacy",
    [string]$CkptDir = "checkpoints\content_delta_pop5\fullstack_hybrid_prereq_legacy"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_LEGACY_TRAIN_PROTOCOL = "1"
$env:USIM_FB_FORCE_FRESH = "1"
$env:USIM_FB_AUTO_RESUME = "0"
$env:USIM_STATIC = "0"
$env:USIM_FB_OUTPUT_TAG = "fullstack_hybrid_prereq_legacy"
$env:USIM_FB_OUTPUT_DIR = $OutputDir
$env:USIM_FB_CKPT_DIR = $CkptDir

$env:USIM_N_EPOCHS = "3"
$env:USIM_TRAIN_WINDOW = "24"
$env:USIM_PPO_EPOCHS = "2"
$env:USIM_PPO_LAMBDA = "0.95"
$env:USIM_PPO_VALUE_CLIP = "0.20"
$env:USIM_PPO_ADV_NORM = "1"
$env:USIM_FAST3_TGT_ALPHA_COLD = "0.35"
$env:USIM_FAST3_TGT_ALPHA_HOT = "0.60"
$env:USIM_FAST3_TGT_ALPHA_STEP = "0.20"
$env:USIM_FAST3_TGT_ALPHA_ENT = "0.20"
$env:USIM_FAST3_TGT_ALPHA_MIN = "0.15"
$env:USIM_FAST3_TGT_ALPHA_MAX = "0.85"

$env:USIM_DISABLE_LLM_SCORE = "1"
$env:USIM_LLM_SAFE_MODE = "0"
$env:USIM_LLM_WEIGHT = "1.0"
$env:USIM_LLM_COLD_ONLY = "0"
$env:USIM_LLM_HOT_ONLY = "0"
$env:USIM_LLM_BANK_MODE = "none"

$env:USIM_USE_CONTENT_DELTA = "0"
$env:USIM_CONTENT_DELTA_MAX_NORM = "0.05"
$env:USIM_CONTENT_DELTA_COLD_ONLY = "1"
$env:USIM_CONTENT_DELTA_NORMALIZE_BASE = "1"
$env:USIM_CONTENT_DELTA_NORMALIZE_OUTPUT = "1"
$env:USIM_USE_PAAC = "0"
$env:USIM_PAAC_ALIGN_W = "0.0"
$env:USIM_PAAC_CONTRAST_W = "0.0"

$env:USIM_FB_LOAD_COURSE_ARTIFACTS = "1"
$env:USIM_FB_COURSE_ONLY_COLD = "1"
$env:USIM_USE_PREREQ_AUX_LOSS = "1"
$env:USIM_FB_COURSE_PREREQ_W = "0.08"
$env:USIM_FB_COURSE_CONCEPT_W = "0.04"
$env:USIM_FB_COURSE_DIFF_W = "0.03"
$env:USIM_FB_COURSE_REDUNDANT_W = "0.02"
$env:USIM_FB_COURSE_REDUNDANT_MODE = "concept"
$env:USIM_FB_COURSE_SAMPLE_SOFT = "1"
$env:USIM_FB_COURSE_SAMPLE_BETA = "0.20"
$env:USIM_FB_COURSE_SAMPLE_ONLY_COLD = "1"
$env:USIM_FB_COURSE_SAMPLE_TOPK = "32"
$env:USIM_FB_COURSE_SAMPLE_TOPL = "32"
$env:USIM_USE_COURSE_RERANK = "0"
$env:USIM_USE_STRUCTURED_HARD_NEG = "0"

$env:USIM_PREREQ_GRAPH_SOURCE = "hybrid"
$env:USIM_PREREQ_HYBRID_ALPHA = "0.70"
$env:USIM_PREREQ_HYBRID_STRONG_CONCEPT_THR = "0.35"

$log = Join-Path $OutputDir "run.log"
Write-Host "===== Running fullstack_hybrid_prereq_legacy =====" -ForegroundColor Cyan
$commandLine = ('"{0}" -u "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
& cmd.exe /d /c $commandLine | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: fullstack_hybrid_prereq_legacy"
}
