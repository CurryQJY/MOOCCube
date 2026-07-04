$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_pam_official_epoch_sweep_wsl_gpu.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing PAM official epoch sweep runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pam_epoch_sweep_" + [System.Guid]::NewGuid().ToString("N"))
$outRoot = Join-Path $tmpRoot "outputs"

try {
    $out = & $script `
        -OutputRoot $outRoot `
        -Dataset mooccube `
        -Seed 2026 `
        -EpochList 3,5 `
        -RunId test_epoch_sweep `
        -DryRun *>&1
    $text = ($out -join "`n")

    foreach ($needle in @(
        "Total sweep tasks: 2",
        "Dataset: mooccube",
        "Seed: 2026",
        "Epochs: 3,5",
        "SWEEP TASK epoch=3",
        "SWEEP TASK epoch=5",
        "Total tasks: 1",
        "dataset=mooccube seed=2026",
        "epochs=3 batch=2048",
        "epochs=5 batch=2048",
        "Summary CSV:"
    )) {
        if ($text -notmatch [regex]::Escape($needle)) {
            throw "Expected dry-run output to contain '$needle'. Output:`n$text"
        }
    }

    foreach ($needle in @(
        "epoch=1",
        "dataset=coco",
        "dataset=junyi"
    )) {
        if ($text -match [regex]::Escape($needle)) {
            throw "Dry-run output should not contain '$needle'. Output:`n$text"
        }
    }

    $summary = Join-Path $outRoot "pam_epoch_sweep_summary.csv"
    if (Test-Path -LiteralPath $summary) {
        throw "Dry-run should not create summary CSV: $summary"
    }

    $externalOutRoot = Join-Path $tmpRoot "external_outputs"
    $externalOut = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $script `
        -OutputRoot $externalOutRoot `
        -Dataset mooccube `
        -Seed 2026 `
        -EpochListCsv "3,5" `
        -RunId test_epoch_sweep_external `
        -DryRun *>&1
    $externalText = ($externalOut -join "`n")
    foreach ($needle in @(
        "Total sweep tasks: 2",
        "Epochs: 3,5",
        "SWEEP TASK epoch=3",
        "SWEEP TASK epoch=5"
    )) {
        if ($externalText -notmatch [regex]::Escape($needle)) {
            throw "Expected external dry-run output to contain '$needle'. Output:`n$externalText"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_pam_official_epoch_sweep_wsl_gpu.ps1 passed"
