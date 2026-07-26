$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "launch_mooccube_sage_twoexpert_scorefusion_tail0p002_seed2025.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube SAGE two-expert score-fusion launcher: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "MOOCCube SAGE two-expert score-fusion launcher dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube SAGE two-expert score-fusion tail0p002 seed=2025",
    "S12_twoexpert_scorefusion_tail0p002_e60_seed2025",
    "gate_mode=bucket_mlp",
    "bucket_strategy=paper",
    "ratio=0.002",
    "mask=true",
    "only_cold_or_tail=false",
    "candidate_two_expert=false",
    "score_fusion=true",
    "DRYRUN requested"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected MOOCCube score-fusion launch dry-run setting: $expected"
    }
}

Write-Host "test_mooccube_sage_twoexpert_scorefusion_launcher.ps1 passed"
