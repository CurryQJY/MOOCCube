param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2026, 2027),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$outputRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
$resultSubdir = "p1_motivation_cgrc_main_table_reproduction"
$checkpointRoot = "checkpoints\content_delta_pop5\p1_motivation_cgrc_main_table_reproduction"
$logRoot = "background_logs\p1_cgrc_main_table_reproduction\resumed_after_recommended_queue"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$tracked = @("USIM_DATA_DIR", "USIM_COLD_THRESHOLD", "USIM_STATIC_TEST_HISTORY", "USIM_EVAL_N_NEG", "USIM_STATIC_SPLIT_DIR", "USIM_BASELINE_OUTPUT_DIR", "USIM_STATIC_SEED", "CGRC_PAPER_STATIC_EPOCHS", "CGRC_PAPER_BATCH_SIZE", "CGRC_PAPER_EVAL_N_NEG", "CGRC_PAPER_COLD_THRESHOLD", "CGRC_PAPER_BEST_AVERAGE_MODE", "CGRC_PAPER_RUN_SAMPLED_EVAL", "CGRC_PAPER_MASK_RHO", "CGRC_PAPER_RECON_TOPK", "CGRC_PAPER_LAMBDA_E", "CGRC_PAPER_TAU", "CGRC_PAPER_STATIC_SEED", "CGRC_PAPER_SEED", "CGRC_PAPER_CKPT_DIR", "CGRC_PAPER_SAVE_CKPT", "CGRC_PAPER_AUTO_RESUME", "CGRC_PAPER_FORCE_FRESH", "CGRC_PAPER_SAVE_OPT_STATE")
try {
    $env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"; $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"; $env:USIM_EVAL_N_NEG = "200"
    $env:CGRC_PAPER_STATIC_EPOCHS = "50"; $env:CGRC_PAPER_BATCH_SIZE = "4096"
    $env:CGRC_PAPER_EVAL_N_NEG = "200"; $env:CGRC_PAPER_COLD_THRESHOLD = "1"
    $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"; $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
    $env:CGRC_PAPER_MASK_RHO = "0.3"; $env:CGRC_PAPER_RECON_TOPK = "20"
    $env:CGRC_PAPER_LAMBDA_E = "1.0"; $env:CGRC_PAPER_TAU = "0.5"
    $env:CGRC_PAPER_SAVE_CKPT = "1"; $env:CGRC_PAPER_AUTO_RESUME = "1"
    $env:CGRC_PAPER_FORCE_FRESH = "0"; $env:CGRC_PAPER_SAVE_OPT_STATE = "1"
    foreach ($seed in @(2026, 2027)) {
        if ($Seeds -notcontains $seed) { continue }
        $splitName = "strict_item_cold_balanced_thr1_seed_$seed"
        $splitDir = Join-Path $outputRoot $splitName
        $outDir = Join-Path $splitDir $resultSubdir
        $result = Join-Path $outDir "cgrc_paper_static_result.json"
        if (Test-Path -LiteralPath $result) { Write-Host "SKIP completed CGRC seed=$seed"; continue }
        if (-not (Test-Path -LiteralPath $splitDir)) { throw "Missing split: $splitDir" }
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $env:USIM_STATIC_SPLIT_DIR = $splitDir; $env:USIM_BASELINE_OUTPUT_DIR = $outDir
        $env:USIM_STATIC_SEED = "$seed"; $env:CGRC_PAPER_STATIC_SEED = "$seed"; $env:CGRC_PAPER_SEED = "$seed"
        $env:CGRC_PAPER_CKPT_DIR = Join-Path $checkpointRoot $splitName
        Write-Host "[$(Get-Date -Format o)] START CGRC seed=$seed checkpoint=$env:CGRC_PAPER_CKPT_DIR"
        if ($DryRun) { continue }
        $seedLog = Join-Path $logRoot "seed_$seed.log"
        $commandLine = '".\py.bat" -u -X faulthandler "cgrc_paper_static_hin.py" 2>&1'
        & cmd.exe /d /c $commandLine | Tee-Object -FilePath $seedLog -Append
        if ($LASTEXITCODE -ne 0) { throw "CGRC seed=$seed failed exit=$LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath $result)) { throw "CGRC seed=$seed exited without result JSON" }
        Write-Host "[$(Get-Date -Format o)] DONE CGRC seed=$seed"
    }
    if (-not $DryRun) {
        $summaryDir = Join-Path $outputRoot $resultSubdir
        & .\py.bat -B aggregate_main_table_static_results.py --root $outputRoot --split-glob "strict_item_cold_balanced_thr1_seed_*" --result-subdir $resultSubdir --metric-mode item_macro --out-dir $summaryDir
        if ($LASTEXITCODE -ne 0) { throw "CGRC aggregation failed" }
    }
}
finally {
    foreach ($name in $tracked) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
}

