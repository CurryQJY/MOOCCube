param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputDir = "outputs\pam_official_full_mooccube_seed2025_b2048\mooccube\strict_item_cold_balanced_thr1_seed_2025\main_table_balanced_itemmacro_v1",
    [string]$LogPath = "",
    [int]$Seed = 2025,
    [int]$Epochs = 1,
    [int]$BatchSize = 2048
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $LogPath = Join-Path $OutputDir "pam_train_eval_full.log"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath), $OutputDir | Out-Null

$tfRoot = (Resolve-Path -LiteralPath ".runtime_tmp\aldi_tf1_py37").Path
$env:PATH = (Join-Path $tfRoot "Library\bin") + ";" + (Join-Path $tfRoot "Scripts") + ";" + $tfRoot + ";" + $env:PATH
$env:PYTHONUNBUFFERED = "1"

$env:PAM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:PAM_STATIC_SPLIT_DIR = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_$Seed"
$env:PAM_BASELINE_OUTPUT_DIR = $OutputDir
$env:PAM_ROOT = (Resolve-Path -LiteralPath ".runtime_tmp\PAM").Path
$env:PAM_RELATION_DIR = "MOOCCube\relations"
$env:PAM_SEED = "$Seed"
$env:PAM_STATIC_SEED = "$Seed"
$env:PAM_COLD_THRESHOLD = "1"
$env:PAM_EPOCHS = "$Epochs"
$env:PAM_BATCH_SIZE = "$BatchSize"
$env:PAM_LR = "0.001"
$env:PAM_EMB_DIM = "8"
$env:PAM_HIDDEN_DIM = "16"
$env:PAM_CATE_DIM = "8"
$env:PAM_NEG_PER_POS = "1"
$env:PAM_MAX_TRAIN_POS = "0"
$env:PAM_MAX_EVAL_ROWS = "0"
$env:PAM_EVAL_ITEM_BATCH_SIZE = "1024"
$env:PAM_USE_GPU = "0"

$oldErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & (Join-Path $tfRoot "python.exe") -u "pam_official_static.py" --mode train_eval *> $LogPath
    $exit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $oldErrorActionPreference
}
if ($exit -ne 0) {
    throw "PAM MOOCCube full train/eval failed with exit=$exit. See $LogPath"
}

Write-Host "PAM MOOCCube full train/eval done: $OutputDir"
