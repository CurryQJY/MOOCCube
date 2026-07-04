param(
    [int]$TeacherEpochs = 100,
    [int]$StudentEpochs = 100,
    [int]$Seed = 2025
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$SplitDir = Join-Path $RepoRoot "outputs\coco\single_seed_triage\ours_full\strict_item_cold_balanced_thr1_seed_2025"
$OutDir = Join-Path $SplitDir "main_table_compare"
$QueueDir = Join-Path $RepoRoot "outputs\coco\single_seed_triage\_queue"
$CkptDir = Join-Path $RepoRoot "checkpoints\coco\single_seed_triage\aldi_lightweight\strict_item_cold_balanced_thr1_seed_2025"
$TeacherCkptDir = Join-Path $CkptDir "teacher"
New-Item -ItemType Directory -Force -Path $OutDir, $QueueDir, $CkptDir, $TeacherCkptDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $OutDir "run_aldi.log"
$ResultJson = Join-Path $OutDir "aldi_static_result.json"
if (Test-Path $RunLog) {
    Copy-Item -LiteralPath $RunLog -Destination (Join-Path $OutDir "run_aldi_before_resume_$Stamp.log") -Force
}
if (Test-Path $ResultJson) {
    Copy-Item -LiteralPath $ResultJson -Destination (Join-Path $OutDir "aldi_static_result_before_resume_$Stamp.json") -Force
}

$env:PYTHONUNBUFFERED = "1"
$env:USIM_DATA_DIR = "processed_data_coco"
$env:USIM_RELATION_DIR = "processed_data_coco\relations"
$env:USIM_STATIC_SPLIT_DIR = $SplitDir
$env:USIM_BASELINE_OUTPUT_DIR = $OutDir
$env:USIM_STATIC_SEED = "$Seed"
$env:USIM_SEED = "$Seed"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "200"
$env:USIM_RUN_SAMPLED_EVAL = "0"
$env:USIM_EARLY_STOP_AVG_MODE = "item_macro"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"

$env:BASELINE_DEVICE = "auto"
$env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
$env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
$env:BASELINE_BEST_METRIC = "cold"

$env:ALDI_TEACHER_EPOCHS = "$TeacherEpochs"
$env:ALDI_TEACHER_EVAL_INTERVAL = "10"
$env:ALDI_STATIC_EPOCHS = "$StudentEpochs"
$env:ALDI_EVAL_INTERVAL = "10"
$env:ALDI_EMB_DIM = "64"
$env:ALDI_HIDDEN_DIM = "64"
$env:ALDI_BATCH_SIZE = "1024"
$env:ALDI_EVAL_BATCH_SIZE = "1024"
$env:ALDI_COLD_THRESHOLD = "1"
$env:ALDI_EVAL_N_NEG = "200"
$env:ALDI_STATIC_SEED = "$Seed"
$env:ALDI_SEED = "$Seed"
$env:ALDI_EARLY_STOP_AVG_MODE = "item_macro"
$env:ALDI_CKPT_DIR = $CkptDir
$env:ALDI_TEACHER_CKPT_DIR = $TeacherCkptDir
$env:ALDI_SAVE_CKPT = "1"
$env:ALDI_SAVE_OPT_STATE = "1"
$env:ALDI_AUTO_RESUME = "1"
$env:ALDI_FORCE_FRESH = "0"

Write-Host "[ALDI-RESUME] repo=$RepoRoot"
Write-Host "[ALDI-RESUME] target teacher_epochs=$TeacherEpochs student_epochs=$StudentEpochs seed=$Seed"
Write-Host "[ALDI-RESUME] ckpt=$CkptDir"
Write-Host "[ALDI-RESUME] log=$RunLog"

& .\py.bat -u .\aldi_static_hin.py 2>&1 | Tee-Object -FilePath $RunLog
$TrainExitCode = $LASTEXITCODE
if ($TrainExitCode -ne 0) {
    throw "ALDI resume failed with exit code $TrainExitCode"
}

$AggLog = Join-Path $QueueDir "aggregate_after_aldi_100e_resume.log"
& .\py.bat .\aggregate_main_table_static_results.py `
    --root "outputs\coco\single_seed_triage\ours_full" `
    --split-glob "strict_item_cold_balanced_thr1_seed_*" `
    --result-subdir "main_table_compare" `
    --metric-mode "item_macro" `
    --out-dir "outputs\coco\single_seed_triage\main_table_compare" 2>&1 | Tee-Object -FilePath $AggLog
$AggExitCode = $LASTEXITCODE
if ($AggExitCode -ne 0) {
    throw "Aggregation after ALDI resume failed with exit code $AggExitCode"
}

Write-Host "[ALDI-RESUME] done"
