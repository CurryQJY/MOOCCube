param(
    [int]$Seed = 2025,
    [int]$Epochs = 50,
    [int]$BatchSize = 1024,
    [int]$ReconUserChunk = 1024,
    [int]$ReconTopK = 20
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$SplitName = "strict_item_cold_balanced_thr1_seed_$Seed"
$OutputRoot = Join-Path $RepoRoot "outputs\coco\single_seed_triage"
$SplitDir = Join-Path $OutputRoot "ours_full\$SplitName"
$OutDir = Join-Path $SplitDir "main_table_compare"
$QueueDir = Join-Path $OutputRoot "_queue"
$CkptDir = Join-Path $RepoRoot "checkpoints\coco\single_seed_triage\cgrc_paper\$SplitName"

foreach ($required in @(
    "processed_data_coco\stream_data.pkl",
    "processed_data_coco\relations",
    (Join-Path $SplitDir "static_train.pkl"),
    (Join-Path $SplitDir "static_val.pkl"),
    (Join-Path $SplitDir "static_test.pkl")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required input: $required"
    }
}

New-Item -ItemType Directory -Force -Path $OutDir, $QueueDir, $CkptDir | Out-Null

$RunLog = Join-Path $OutDir "run_cgrc_paper.log"
if (Test-Path -LiteralPath $RunLog) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item -LiteralPath $RunLog -Destination (Join-Path $OutDir "run_cgrc_paper_before_resume_$Stamp.log") -Force
}

$env:PYTHONUNBUFFERED = "1"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

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

$env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
$env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
$env:BASELINE_BEST_METRIC = "cold"

$env:CGRC_PAPER_STATIC_EPOCHS = "$Epochs"
$env:CGRC_PAPER_BATCH_SIZE = "$BatchSize"
$env:CGRC_PAPER_EVAL_N_NEG = "200"
$env:CGRC_PAPER_COLD_THRESHOLD = "1"
$env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
$env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
$env:CGRC_PAPER_MASK_RHO = "0.3"
$env:CGRC_PAPER_RECON_TOPK = "$ReconTopK"
$env:CGRC_PAPER_RECON_USER_CHUNK = "$ReconUserChunk"
$env:CGRC_PAPER_SPARSE_FORMAT = "csr"
$env:CGRC_PAPER_CUDA_MEMORY_FRACTION = "0.75"
$env:CGRC_PAPER_PROGRESS_INTERVAL = "1"
$env:CGRC_PAPER_LAMBDA_E = "1.0"
$env:CGRC_PAPER_TAU = "0.5"
$env:CGRC_PAPER_STATIC_SEED = "$Seed"
$env:CGRC_PAPER_SEED = "$Seed"
$env:CGRC_PAPER_CKPT_DIR = $CkptDir
$env:CGRC_PAPER_SAVE_CKPT = "1"
$env:CGRC_PAPER_SAVE_OPT_STATE = "1"
$env:CGRC_PAPER_AUTO_RESUME = "1"
$env:CGRC_PAPER_FORCE_FRESH = "0"

Write-Host "[COCO-CGRC] seed=$Seed epochs=$Epochs batch=$BatchSize recon_user_chunk=$ReconUserChunk topk=$ReconTopK"
Write-Host "[COCO-CGRC] split=$SplitDir"
Write-Host "[COCO-CGRC] out=$OutDir"
Write-Host "[COCO-CGRC] ckpt=$CkptDir"
Write-Host "[COCO-CGRC] log=$RunLog"

$cmd = '/d /c ""' + (Join-Path $RepoRoot "py.bat") + '" -u ".\cgrc_paper_static_hin.py" > "' + $RunLog + '" 2>&1"'
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -Wait
$TrainExitCode = $proc.ExitCode
if ($TrainExitCode -ne 0) {
    throw "CGRC-paper failed with exit code $TrainExitCode"
}

$AggLog = Join-Path $QueueDir "aggregate_after_cgrc_paper.log"
& .\py.bat .\aggregate_main_table_static_results.py `
    --root "outputs\coco\single_seed_triage\ours_full" `
    --split-glob "strict_item_cold_balanced_thr1_seed_*" `
    --result-subdir "main_table_compare" `
    --metric-mode "item_macro" `
    --out-dir "outputs\coco\single_seed_triage\main_table_compare" 2>&1 | Tee-Object -FilePath $AggLog
$AggExitCode = $LASTEXITCODE
if ($AggExitCode -ne 0) {
    throw "Aggregation after CGRC-paper failed with exit code $AggExitCode"
}

Write-Host "[COCO-CGRC] done"
