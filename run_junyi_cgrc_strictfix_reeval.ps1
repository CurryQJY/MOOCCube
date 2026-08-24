param(
    [int[]]$Seeds = @(2025, 2026, 2027),
    [string]$Repo = "D:\DeskTop\MOOCCube"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $Repo

function Split-Dir([int]$Seed) {
    if ($Seed -eq 2025) {
        return (Join-Path $Repo "outputs\junyi\mask_ablation\mask_tt\strict_item_cold_balanced_thr1_seed_2025")
    }
    return (Join-Path $Repo ("outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_{0}" -f $Seed))
}

function Ckpt-Dir([int]$Seed) {
    if ($Seed -eq 2025) {
        return (Join-Path $Repo "checkpoints\junyi\cgrc_paper_compare\strict_item_cold_balanced_thr1_seed_2025")
    }
    return (Join-Path $Repo ("checkpoints\junyi\main_table_3seed\cgrc_paper_compare\strict_item_cold_balanced_thr1_seed_{0}" -f $Seed))
}

foreach ($seed in $Seeds) {
    $splitDir = Split-Dir $seed
    $ckptDir = Ckpt-Dir $seed
    $latest = Join-Path $ckptDir "latest.pt"
    if (-not (Test-Path -LiteralPath $latest)) {
        throw "Missing CGRC checkpoint for seed=${seed}: $latest"
    }
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split file for seed=${seed}: $path"
        }
    }

    $outDir = Join-Path $splitDir "cgrc_paper_compare_strictfix"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $logPath = Join-Path $outDir "run.log"

    $env:CUDA_VISIBLE_DEVICES = ""
    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = "processed_data_junyi"
    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_SEED = "$seed"
    $env:USIM_SEED = "$seed"
    $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_EVAL_N_NEG = "200"
    $env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
    $env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
    $env:BASELINE_AUTO_RESUME = "1"
    $env:BASELINE_FORCE_FRESH = "0"
    $env:BASELINE_SAVE_CKPT = "0"
    $env:BASELINE_SAVE_OPT_STATE = "0"
    $env:BASELINE_CKPT_DIR = $ckptDir

    $env:CGRC_PAPER_STATIC_EPOCHS = "50"
    $env:CGRC_PAPER_BATCH_SIZE = "4096"
    $env:CGRC_PAPER_EVAL_N_NEG = "200"
    $env:CGRC_PAPER_COLD_THRESHOLD = "1"
    $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
    $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
    $env:CGRC_PAPER_DEVICE = "cpu"
    $env:CGRC_PAPER_MASK_RHO = "0.3"
    $env:CGRC_PAPER_RECON_TOPK = "20"
    $env:CGRC_PAPER_LAMBDA_E = "1.0"
    $env:CGRC_PAPER_TAU = "0.5"
    $env:CGRC_PAPER_STATIC_SEED = "$seed"
    $env:CGRC_PAPER_SEED = "$seed"
    $env:CGRC_PAPER_CKPT_DIR = $ckptDir
    $env:CGRC_PAPER_AUTO_RESUME = "1"
    $env:CGRC_PAPER_FORCE_FRESH = "0"
    $env:CGRC_PAPER_SAVE_CKPT = "0"
    $env:CGRC_PAPER_SAVE_OPT_STATE = "0"

    Write-Host "===== Re-eval CGRC-paper strictfix seed=$seed ====="
    Write-Host "Split:  $splitDir"
    Write-Host "Ckpt:   $ckptDir"
    Write-Host "Output: $outDir"
    .\py.bat -u cgrc_paper_static_hin.py *> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "CGRC strictfix re-eval failed for seed=$seed; see $logPath"
    }
}
