$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_junyi_sage_tailratio_grid_seed2025.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing Junyi SAGE tail-ratio runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("junyi_sage_gate_mode_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $outputRoot = Join-Path $tmpRoot "outputs"
    $checkpointRoot = Join-Path $tmpRoot "checkpoints"
    $out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
        -Repo $repo `
        -Seed 2025 `
        -Ratios 0.01 `
        -Epochs 60 `
        -Patience 60 `
        -SageGateMode bucket_mlp `
        -SageGateBuckets 13 `
        -SageGateHidden 17 `
        -SageGateBucketStrategy paper `
        -SageTwoExpertScoreFusion `
        -OutputRoot $outputRoot `
        -CheckpointRoot $checkpointRoot `
        -DryRun

    if ($LASTEXITCODE -ne 0) {
        throw "Junyi SAGE gate-mode dry-run exited with code $LASTEXITCODE"
    }

    $text = ($out -join "`n")
    foreach ($expected in @(
        "gate_mode=bucket_mlp",
        "buckets=13",
        "bucket_strategy=paper",
        "gate_hidden=17",
        "score_fusion=True",
        "DRYRUN ratio=0.01"
    )) {
        if ($text -notlike "*$expected*") {
            throw "Missing expected Junyi dry-run setting: $expected"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_junyi_sage_gate_mode_runner.ps1 passed"
