$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "launch_mooccube_sage_bucketmlp_tail0p002_seed2025.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube SAGE bucket-MLP launcher: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "MOOCCube SAGE bucket-MLP launcher dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube SAGE bucket-MLP tail0p002 seed=2025",
    "gate_mode=bucket_mlp",
    "bucket_strategy=paper",
    "ratio=0.002",
    "mask=true",
    "scope=all",
    "two_expert=false",
    "DRYRUN requested"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected MOOCCube launch dry-run setting: $expected"
    }
}

Write-Host "test_mooccube_sage_bucketmlp_launcher.ps1 passed"
