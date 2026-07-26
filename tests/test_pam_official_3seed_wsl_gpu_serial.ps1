$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_pam_official_3datasets_3seed_wsl_gpu_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing PAM official WSL/GPU 3-seed runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pam_official_wsl_gpu_3seed_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script `
        -OutputRoot (Join-Path $tmpRoot "outputs") `
        -DryRun *>&1
    $text = ($out -join "`n")

    foreach ($needle in @(
        "Total tasks: 9",
        "Export runner:",
        "py.bat",
        "WSL GPU Python: /root/venvs/icychesszero_tf2_gpu/bin/python",
        "dataset=mooccube seed=2025",
        "dataset=mooccube seed=2026",
        "dataset=mooccube seed=2027",
        "dataset=junyi seed=2025",
        "dataset=junyi seed=2026",
        "dataset=junyi seed=2027",
        "dataset=coco seed=2025",
        "dataset=coco seed=2026",
        "dataset=coco seed=2027",
        "outputs\junyi\mask_ablation\mask_tt\strict_item_cold_balanced_thr1_seed_2025",
        "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2026",
        "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2027",
        "PAM_USE_GPU=1",
        "batch=2048",
        "max_train_pos=0",
        "max_eval_rows=0"
    )) {
        if ($text -notmatch [regex]::Escape($needle)) {
            throw "Expected dry-run output to contain '$needle'. Output:`n$text"
        }
    }

    $filteredOut = & $script `
        -OutputRoot (Join-Path $tmpRoot "filtered_outputs") `
        -Datasets mooccube `
        -Seeds 2026 `
        -DryRun *>&1
    $filteredText = ($filteredOut -join "`n")
    foreach ($needle in @(
        "Total tasks: 1",
        "dataset=mooccube seed=2026",
        "Seeds: 2026"
    )) {
        if ($filteredText -notmatch [regex]::Escape($needle)) {
            throw "Expected filtered dry-run output to contain '$needle'. Output:`n$filteredText"
        }
    }
    foreach ($needle in @(
        "dataset=mooccube seed=2025",
        "dataset=mooccube seed=2027",
        "dataset=coco",
        "dataset=junyi"
    )) {
        if ($filteredText -match [regex]::Escape($needle)) {
            throw "Filtered dry-run output should not contain '$needle'. Output:`n$filteredText"
        }
    }

    $sweepRoot = Join-Path $tmpRoot "sweep_outputs"
    $split = "strict_item_cold_balanced_thr1_seed_2026"
    $ckptDir = Join-Path (Join-Path (Join-Path (Join-Path (Join-Path $sweepRoot "e5") "mooccube") $split) "main_table_balanced_itemmacro_v1") "checkpoints"
    New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $ckptDir "pam_official_latest.ckpt.index") | Out-Null

    $resumeOut = & $script `
        -OutputRoot (Join-Path $sweepRoot "e10") `
        -Datasets mooccube `
        -Seeds 2026 `
        -Epochs 10 `
        -DryRun *>&1
    $resumeText = ($resumeOut -join "`n")
    foreach ($needle in @(
        "Total tasks: 1",
        "dataset=mooccube seed=2026",
        "epochs=10 batch=2048",
        "init_checkpoint=",
        "start_epoch=5"
    )) {
        if ($resumeText -notmatch [regex]::Escape($needle)) {
            throw "Expected resume dry-run output to contain '$needle'. Output:`n$resumeText"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_pam_official_3seed_wsl_gpu_serial.ps1 passed"
