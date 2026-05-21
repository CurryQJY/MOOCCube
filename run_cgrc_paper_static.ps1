param(
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$Epochs = 50,
    [int]$BatchSize = 4096,
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$OutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$ResultSubdir = "main_table_balanced_itemmacro_cgrc_paper_v1",
    [string]$BestAverageMode = "item_macro",
    [switch]$RunSampledEval,
    [int]$EvalNeg = 200,
    [double]$MaskRho = 0.3,
    [int]$ReconTopK = 20,
    [double]$LambdaE = 1.0,
    [double]$Tau = 0.5
)

$ErrorActionPreference = "Stop"

$env:USIM_DATA_DIR = $DataDir
$env:USIM_COLD_THRESHOLD = "$ColdThreshold"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "$EvalNeg"
$env:PYTHONUNBUFFERED = "1"

$env:CGRC_PAPER_STATIC_EPOCHS = "$Epochs"
$env:CGRC_PAPER_BATCH_SIZE = "$BatchSize"
$env:CGRC_PAPER_EVAL_N_NEG = "$EvalNeg"
$env:CGRC_PAPER_COLD_THRESHOLD = "$ColdThreshold"
$env:CGRC_PAPER_BEST_AVERAGE_MODE = $BestAverageMode
$env:CGRC_PAPER_RUN_SAMPLED_EVAL = if ($RunSampledEval.IsPresent) { "1" } else { "0" }
$env:CGRC_PAPER_MASK_RHO = "$MaskRho"
$env:CGRC_PAPER_RECON_TOPK = "$ReconTopK"
$env:CGRC_PAPER_LAMBDA_E = "$LambdaE"
$env:CGRC_PAPER_TAU = "$Tau"

foreach ($seed in $Seeds) {
    $splitName = "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $seed
    $splitDir = Join-Path $OutputRoot $splitName
    if (-not (Test-Path -LiteralPath $splitDir)) {
        throw "Missing split directory: $splitDir"
    }

    $outDir = Join-Path $splitDir $ResultSubdir
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_SEED = "$seed"
    $env:CGRC_PAPER_STATIC_SEED = "$seed"
    $env:CGRC_PAPER_SEED = "$seed"

    Write-Host ""
    Write-Host "===== Running CGRC-paper seed=$seed threshold=$ColdThreshold epochs=$Epochs ====="
    Write-Host "Split:  $splitDir"
    Write-Host "Output: $outDir"
    .\py.bat cgrc_paper_static_hin.py
}

$splitGlob = "strict_item_cold_balanced_thr{0}_seed_*" -f $ColdThreshold
$summaryDir = Join-Path $OutputRoot $ResultSubdir
.\py.bat -B aggregate_main_table_static_results.py `
    --root $OutputRoot `
    --split-glob $splitGlob `
    --result-subdir $ResultSubdir `
    --metric-mode item_macro `
    --out-dir $summaryDir

Write-Host ""
Write-Host "Summary: $summaryDir\main_table_item_macro_summary.csv"
