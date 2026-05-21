param(
    [string]$Root = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$ResultSubdir = "",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$TeacherEpochs = 80,
    [int]$StudentEpochs = 80,
    [int]$BatchSize = 4096,
    [int]$EvalInterval = 5,
    [int]$EvalNeg = 200
)

$ErrorActionPreference = "Stop"

$EpochTag = "teacher${TeacherEpochs}_student${StudentEpochs}"
if ([string]::IsNullOrWhiteSpace($ResultSubdir)) {
    $ResultSubdir = "main_table_balanced_itemmacro_dropoutnet_official_${EpochTag}_v1"
}

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "$EvalNeg"
$env:USIM_EARLY_STOP_AVG_MODE = "item_macro"
$env:DROPOUT_OFFICIAL_TEACHER_EPOCHS = "$TeacherEpochs"
$env:DROPOUT_OFFICIAL_STATIC_EPOCHS = "$StudentEpochs"
$env:DROPOUT_OFFICIAL_EVAL_INTERVAL = "$EvalInterval"
$env:DROPOUT_OFFICIAL_BATCH_SIZE = "$BatchSize"
$env:DROPOUT_OFFICIAL_EVAL_N_NEG = "$EvalNeg"
$env:DROPOUT_OFFICIAL_EPOCH_TAG = $EpochTag
$env:DROPOUT_OFFICIAL_ITEM_DROPOUT = "0.5"
$env:DROPOUT_OFFICIAL_USER_DROPOUT = "0.0"
$env:PYTHONUNBUFFERED = "1"

foreach ($seed in $Seeds) {
    $split = Join-Path $Root "strict_item_cold_balanced_thr1_seed_$seed"
    if (-not (Test-Path -LiteralPath $split)) {
        throw "Missing split directory: $split"
    }

    $out = Join-Path $split $ResultSubdir
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    $env:USIM_STATIC_SPLIT_DIR = $split
    $env:USIM_BASELINE_OUTPUT_DIR = $out
    $env:USIM_STATIC_SEED = "$seed"
    $env:DROPOUT_OFFICIAL_STATIC_SEED = "$seed"
    $env:DROPOUT_OFFICIAL_SEED = "$seed"

    Write-Host ""
    Write-Host "===== Running DropoutNet official-protocol seed=$seed ====="
    Write-Host "Split:  $split"
    Write-Host "Output: $out"
    .\py.bat -B dropoutnet_official_static_hin.py
}

$summaryOut = Join-Path $Root $ResultSubdir
.\py.bat -B aggregate_main_table_static_results.py `
    --root $Root `
    --split-glob "strict_item_cold_balanced_thr1_seed_*" `
    --result-subdir $ResultSubdir `
    --metric-mode item_macro `
    --out-dir $summaryOut

Write-Host ""
Write-Host "Summary:"
Write-Host (Join-Path $summaryOut "main_table_item_macro_summary.csv")

$officialSummary = Join-Path $summaryOut "main_table_item_macro_summary.csv"
$mergedOut = Join-Path $Root "main_table_item_macro_final_audit_with_dropoutnet_official_${EpochTag}\main_table_item_macro_summary.csv"
.\py.bat -B merge_dropoutnet_official_main_table.py `
    --official $officialSummary `
    --out $mergedOut

Write-Host "Merged main-table summary:"
Write-Host $mergedOut
