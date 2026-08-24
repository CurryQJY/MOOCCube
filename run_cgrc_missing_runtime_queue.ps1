param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -LiteralPath $Repo

$jobs = @(
    @{
        Dataset = "Junyi"; Seed = 2025
        DataDir = "processed_data_junyi"
        OutputRoot = "outputs\junyi\main_table_3seed"
        ResultSubdir = "cgrc_runtime_profile"
        CheckpointRoot = "checkpoints\junyi\cgrc_runtime_profile"
    },
    @{
        Dataset = "Junyi"; Seed = 2026
        DataDir = "processed_data_junyi"
        OutputRoot = "outputs\junyi\main_table_3seed"
        ResultSubdir = "cgrc_runtime_profile"
        CheckpointRoot = "checkpoints\junyi\cgrc_runtime_profile"
    },
    @{
        Dataset = "Junyi"; Seed = 2027
        DataDir = "processed_data_junyi"
        OutputRoot = "outputs\junyi\main_table_3seed"
        ResultSubdir = "cgrc_runtime_profile"
        CheckpointRoot = "checkpoints\junyi\cgrc_runtime_profile"
    },
    @{
        Dataset = "MOOCCube"; Seed = 2025
        DataDir = "processed_data_hin_clean_pop5"
        OutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
        ResultSubdir = "runtime_cgrc_profile"
        CheckpointRoot = "checkpoints\mooccubex\runtime_cgrc_profile"
    }
)

foreach ($job in $jobs) {
    $splitName = "strict_item_cold_balanced_thr1_seed_$($job.Seed)"
    $splitDir = Join-Path $job.OutputRoot $splitName
    if (-not (Test-Path -LiteralPath $splitDir)) {
        throw "Missing split directory: $splitDir"
    }
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split file: $path"
        }
    }
}

foreach ($job in $jobs) {
    $seed = [int]$job.Seed
    $splitName = "strict_item_cold_balanced_thr1_seed_$seed"
    $splitDir = Join-Path $job.OutputRoot $splitName
    $outDir = Join-Path $splitDir $job.ResultSubdir
    $checkpointDir = Join-Path $job.CheckpointRoot $splitName
    $resultPath = Join-Path $outDir "cgrc_paper_static_result.json"
    $logPath = Join-Path $outDir "run.log"

    if (Test-Path -LiteralPath $resultPath) {
        Write-Host "[$(Get-Date -Format o)] SKIP completed $($job.Dataset) CGRC seed=$seed"
        continue
    }

    Write-Host "[$(Get-Date -Format o)] START $($job.Dataset) CGRC seed=$seed"
    Write-Host "  split=$splitDir"
    Write-Host "  output=$outDir"
    Write-Host "  checkpoint=$checkpointDir"
    if ($DryRun) {
        continue
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = $job.DataDir
    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_SEED = "$seed"
    $env:USIM_SEED = "$seed"
    $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_EVAL_N_NEG = "200"

    $env:CGRC_PAPER_STATIC_EPOCHS = "50"
    $env:CGRC_PAPER_BATCH_SIZE = "4096"
    $env:CGRC_PAPER_EVAL_N_NEG = "200"
    $env:CGRC_PAPER_COLD_THRESHOLD = "1"
    $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
    $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
    $env:CGRC_PAPER_DEVICE = "cuda"
    $env:CGRC_PAPER_MASK_RHO = "0.3"
    $env:CGRC_PAPER_RECON_TOPK = "20"
    $env:CGRC_PAPER_LAMBDA_E = "1.0"
    $env:CGRC_PAPER_TAU = "0.5"
    $env:CGRC_PAPER_STATIC_SEED = "$seed"
    $env:CGRC_PAPER_SEED = "$seed"
    $env:CGRC_PAPER_CKPT_DIR = $checkpointDir
    $env:CGRC_PAPER_SAVE_CKPT = "1"
    $env:CGRC_PAPER_AUTO_RESUME = "1"
    $env:CGRC_PAPER_FORCE_FRESH = "0"
    $env:CGRC_PAPER_SAVE_OPT_STATE = "1"

    "[$(Get-Date -Format o)] START $($job.Dataset) CGRC seed=$seed" | Out-File -LiteralPath $logPath -Append -Encoding utf8
    $commandLine = "`".\py.bat`" -u -X faulthandler `"cgrc_paper_static_hin.py`" >> `"$logPath`" 2>&1"
    & cmd.exe /d /c $commandLine
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$($job.Dataset) CGRC seed=$seed failed with exit=$exitCode; see $logPath"
    }
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "$($job.Dataset) CGRC seed=$seed exited without result JSON"
    }
    Write-Host "[$(Get-Date -Format o)] DONE $($job.Dataset) CGRC seed=$seed"
}

if (-not $DryRun) {
    & .\py.bat paper_aaai27\scripts\build_revision_tables.py
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to rebuild revision tables"
    }
    & .\py.bat paper_aaai27\scripts\export_efficiency_table.py
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to export standalone efficiency table"
    }
}

Write-Host "[$(Get-Date -Format o)] CGRC missing-runtime queue complete"
