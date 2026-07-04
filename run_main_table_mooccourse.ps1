param(
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_mooccourse",
    [string]$RelationDir = "processed_data_mooccourse\relations",
    [string]$OutputRoot = "outputs\mooccourse\static_item_cold_balanced",
    [string]$CheckpointRoot = "checkpoints\mooccourse\static_item_cold_balanced",
    [string]$ResultSubdir = "main_table_mooccourse",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [switch]$Smoke,
    [switch]$SkipOurs,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

if ($Smoke.IsPresent) {
    $OursEpochs = 1
    $OursPatience = 1
    $BprEpochs = 1
    $LightGCNEpochs = 1
    $DropoutEpochs = 1
    $CCFCEpochs = 1
    $AldiTeacherEpochs = 1
    $AldiEpochs = 1
    $CgrcEpochs = 1
    $EvalNeg = [Math]::Min($EvalNeg, 20)
} else {
    $OursEpochs = 15
    $OursPatience = 3
    $BprEpochs = 200
    $LightGCNEpochs = 100
    $DropoutEpochs = 60
    $CCFCEpochs = 80
    $AldiTeacherEpochs = 200
    $AldiEpochs = 100
    $CgrcEpochs = 50
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

if (-not $SkipOurs.IsPresent) {
    & .\run_usim_feedback_fast3_content_delta_static.ps1 `
        -PythonRunner $PythonRunner `
        -DataDir $DataDir `
        -RelationDir $RelationDir `
        -OutputRoot $OutputRoot `
        -CheckpointRoot $CheckpointRoot `
        -Protocol strict_item_cold_balanced `
        -ColdThresholds @($ColdThreshold) `
        -Seeds $Seeds `
        -Epochs $OursEpochs `
        -Patience $OursPatience `
        -PrereqGraphSource behavior `
        -SkipAggregate
}

$env:USIM_DATA_DIR = $DataDir
$env:USIM_RELATION_DIR = $RelationDir
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
$env:USIM_COLD_THRESHOLD = "$ColdThreshold"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "$EvalNeg"
$env:BASELINE_BEST_METRIC = "cold"
$env:PYTHONUNBUFFERED = "1"

foreach ($seed in $Seeds) {
    $splitName = "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $seed
    $splitDir = Join-Path $OutputRoot $splitName
    if (-not (Test-Path -LiteralPath $splitDir)) {
        throw "Missing split directory: $splitDir. Run without -SkipOurs first."
    }

    $outDir = Join-Path $splitDir $ResultSubdir
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_SEED = "$seed"
    $env:USIM_SEED = "$seed"

    $fast3Result = Join-Path $splitDir "fast3_static_result.json"
    if (Test-Path -LiteralPath $fast3Result) {
        Copy-Item -LiteralPath $fast3Result -Destination (Join-Path $outDir "fast3_static_result.json") -Force
    }

    Write-Host ""
    Write-Host "===== MOOCCourse main-table baselines seed=$seed threshold=$ColdThreshold =====" -ForegroundColor Cyan
    Write-Host "Split:  $splitDir"
    Write-Host "Output: $outDir"

    Write-Host "== Popularity =="
    & $PythonRunner popularity_static.py

    Write-Host "== ContentProfile =="
    & $PythonRunner content_profile_static_hin.py

    Write-Host "== BPR =="
    $env:BPR_STATIC_EPOCHS = "$BprEpochs"
    $env:BPR_EVAL_INTERVAL = "5"
    $env:BPR_BATCH_SIZE = "4096"
    & $PythonRunner bpr_static_fair.py

    Write-Host "== LightGCN =="
    $env:LIGHTGCN_STATIC_EPOCHS = "$LightGCNEpochs"
    $env:LIGHTGCN_EVAL_INTERVAL = "5"
    $env:LIGHTGCN_BATCH_SIZE = "4096"
    & $PythonRunner lightgcn_static_hin_fair.py

    Write-Host "== DropoutNet =="
    $env:DROPOUT_STATIC_EPOCHS = "$DropoutEpochs"
    $env:DROPOUT_EVAL_INTERVAL = "5"
    & $PythonRunner drop_static_hin.py

    Write-Host "== CCFCRec =="
    $env:CCFCREC_STATIC_EPOCHS = "$CCFCEpochs"
    $env:CCFCREC_EVAL_INTERVAL = "5"
    $env:CCFCREC_BATCH_SIZE = "4096"
    $env:CCFCREC_EVAL_BATCH_SIZE = "4096"
    $env:CCFCREC_EVAL_ITEM_MODE = "mixed"
    & $PythonRunner ccfc_static_hin.py

    Write-Host "== ALDI =="
    $env:ALDI_TEACHER_EPOCHS = "$AldiTeacherEpochs"
    $env:ALDI_TEACHER_EVAL_INTERVAL = "20"
    $env:ALDI_STATIC_EPOCHS = "$AldiEpochs"
    $env:ALDI_EVAL_INTERVAL = "5"
    $env:ALDI_BATCH_SIZE = "4096"
    & $PythonRunner aldi_static_hin.py

    Write-Host "== CGRC =="
    $env:CGRC_PAPER_STATIC_EPOCHS = "$CgrcEpochs"
    $env:CGRC_PAPER_BATCH_SIZE = "4096"
    $env:CGRC_PAPER_EVAL_N_NEG = "$EvalNeg"
    $env:CGRC_PAPER_COLD_THRESHOLD = "$ColdThreshold"
    $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
    $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
    & $PythonRunner cgrc_paper_static_hin.py
}

if (-not $SkipAggregate.IsPresent) {
    $splitGlob = "strict_item_cold_balanced_thr{0}_seed_*" -f $ColdThreshold
    $summaryDir = Join-Path $OutputRoot $ResultSubdir
    & $PythonRunner -B aggregate_main_table_static_results.py `
        --root $OutputRoot `
        --split-glob $splitGlob `
        --result-subdir $ResultSubdir `
        --metric-mode item_macro `
        --out-dir $summaryDir

    Write-Host ""
    Write-Host "Summary: $summaryDir\main_table_item_macro_summary.csv"
}
