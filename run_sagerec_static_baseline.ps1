param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 60,
    [int]$EvalInterval = 5,
    [int]$BatchSize = 4096,
    [int]$SampleTopN = 15,
    [int]$MaxHistLen = 100,
    [int]$BucketCount = 20,
    [string]$OutputSubdir = "main_table_balanced_itemmacro_v1",
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$ForceFresh,
    [switch]$NoCheckpoint,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$runEpochs = if ($Smoke) { 1 } else { $Epochs }
$runEvalInterval = if ($Smoke) { 1 } else { $EvalInterval }
$checkpointEnabled = -not $NoCheckpoint

foreach ($seed in $Seeds) {
    $splitName = "strict_item_cold_balanced_thr1_seed_$seed"
    $splitDir = "outputs\content_delta_pop5\static_item_cold_balanced\$splitName"
    if (-not (Test-Path -LiteralPath (Join-Path $splitDir "static_train.pkl"))) {
        throw "Missing static split files under $splitDir"
    }

    if ($Smoke) {
        $outputDir = "outputs\content_delta_pop5\sagerec_baseline\smoke\$splitName"
        $ckptDir = "checkpoints\content_delta_pop5\sagerec_baseline\smoke\$splitName"
    } else {
        $outputDir = Join-Path $splitDir $OutputSubdir
        $ckptDir = "checkpoints\content_delta_pop5\sagerec_baseline\$splitName"
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    if ($checkpointEnabled) {
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    }

    $env:USIM_DATA_DIR = $DataDir
    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outputDir
    $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_SEED = [string]$seed
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_EVAL_N_NEG = "200"
    $env:BASELINE_BEST_METRIC = "cold"
    $env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"

    $env:SAGEREC_STATIC_EPOCHS = [string]$runEpochs
    $env:SAGEREC_EVAL_INTERVAL = [string]$runEvalInterval
    $env:SAGEREC_BATCH_SIZE = [string]$BatchSize
    $env:SAGEREC_STATIC_SEED = [string]$seed
    $env:SAGEREC_SEED = [string]$seed
    $env:SAGEREC_SAMPLE_TOP_N = [string]$SampleTopN
    $env:SAGEREC_MAX_HIST_LEN = [string]$MaxHistLen
    $env:SAGEREC_BUCKET_COUNT = [string]$BucketCount
    $env:SAGEREC_CKPT_DIR = if ($checkpointEnabled) { $ckptDir } else { "" }
    $env:SAGEREC_SAVE_CKPT = if ($checkpointEnabled) { "1" } else { "0" }
    $env:SAGEREC_AUTO_RESUME = if ($Resume) { "1" } else { "0" }
    $env:SAGEREC_FORCE_FRESH = if ($ForceFresh -or -not $Resume) { "1" } else { "0" }
    $env:SAGEREC_SAVE_OPT_STATE = "1"

    Write-Host "== SAGERec baseline =="
    Write-Host "Seed=$seed Split=$splitDir"
    Write-Host "Output=$outputDir"
    Write-Host "Checkpoint=$($env:SAGEREC_CKPT_DIR)"
    Write-Host "Epochs=$runEpochs EvalInterval=$runEvalInterval TopN=$SampleTopN MaxHistLen=$MaxHistLen Buckets=$BucketCount"
    Write-Host "Resume=$Resume ForceFresh=$($env:SAGEREC_FORCE_FRESH) Smoke=$Smoke"

    if ($DryRun) {
        continue
    }

    & $PythonRunner sagerec_static_baseline.py
    if ($LASTEXITCODE -ne 0) {
        throw "SAGERec baseline failed for seed $seed with exit code $LASTEXITCODE"
    }
}
