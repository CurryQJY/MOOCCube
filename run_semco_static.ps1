param(
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$SplitRoot = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$OutputRoot = "outputs\content_delta_pop5\semco_v1",
    [int[]]$Seeds = @(2025),
    [int]$ColdThreshold = 1,
    [int]$Epochs = 3,
    [int]$BatchSize = 2048,
    [int]$EvalBatchSize = 8192,
    [int]$NegativeNumber = 64,
    [int]$EvalInterval = 5,
    [double]$Alpha = 1.5,
    [double]$Tau = 0.10,
    [ValidateSet("mlp", "raw_residual", "raw")]
    [string]$EncoderMode = "raw_residual",
    [double]$RawDeltaScale = 0.05,
    [ValidateSet("interaction", "item_macro")]
    [string]$EarlyStopAverageMode = "item_macro",
    [string]$ResultSubdir = "",
    [switch]$RunSampledEval
)

$ErrorActionPreference = "Stop"

foreach ($seed in $Seeds) {
    $splitName = "strict_item_cold_balanced_thr${ColdThreshold}_seed_$seed"
    $splitDir = Join-Path $SplitRoot $splitName
    if (-not (Test-Path $splitDir)) {
        throw "Missing split directory: $splitDir"
    }

    $tag = if ($ResultSubdir) {
        $ResultSubdir
    } else {
        $modeTag = $EncoderMode.Replace("_", "")
        $alphaTag = ("alpha{0}" -f $Alpha).Replace(".", "p")
        "${modeTag}_${alphaTag}_neg${NegativeNumber}_e${Epochs}"
    }
    $outDir = Join-Path (Join-Path $OutputRoot $splitName) $tag
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $env:USIM_DATA_DIR = $DataDir
    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_COLD_THRESHOLD = [string]$ColdThreshold
    $env:USIM_EVAL_N_NEG = "200"
    $env:USIM_EARLY_STOP_AVG_MODE = $EarlyStopAverageMode

    $env:SEMCO_STATIC_SEED = [string]$seed
    $env:SEMCO_SEED = [string]$seed
    $env:SEMCO_ENCODER_MODE = $EncoderMode
    $env:SEMCO_STATIC_EPOCHS = [string]$Epochs
    $env:SEMCO_BATCH_SIZE = [string]$BatchSize
    $env:SEMCO_EVAL_BATCH_SIZE = [string]$EvalBatchSize
    $env:SEMCO_NEGATIVE_NUMBER = [string]$NegativeNumber
    $env:SEMCO_ENTMAX_ALPHA = [string]$Alpha
    $env:SEMCO_TAU = [string]$Tau
    $env:SEMCO_RAW_DELTA_SCALE = [string]$RawDeltaScale
    $env:SEMCO_EVAL_INTERVAL = [string]$EvalInterval
    $env:SEMCO_RUN_SAMPLED_EVAL = if ($RunSampledEval) { "1" } else { "0" }
    $env:SEMCO_DETACH_QUERY = "1"
    $env:SEMCO_EXCLUDE_TRAIN_TARGET = "1"

    $logPath = Join-Path $outDir "run.log"
    Write-Host "SEMCo seed=$seed mode=$EncoderMode alpha=$Alpha neg=$NegativeNumber epochs=$Epochs"
    Write-Host "  split=$splitDir"
    Write-Host "  out=$outDir"
    & $PythonRunner -u semco_static_hin.py *> $logPath
    if ($LASTEXITCODE -ne 0) {
        Get-Content $logPath -Tail 80
        throw "SEMCo failed for seed=$seed"
    }
    Get-Content $logPath -Tail 30
}
