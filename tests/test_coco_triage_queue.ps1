$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_coco_single_seed_triage.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing COCO triage queue script: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -Seeds 2025 `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "Dataset=COCO",
    "DataDir=processed_data_coco",
    "RelationDir=processed_data_coco\relations",
    "PrereqGraphSource=behavior",
    "OursEpochs=30",
    "BaselineEpochs=10",
    "Models=Popularity,ContentProfile,BPR,LightGCN,DropoutNet,GAR,CCFCRec,LightGCL",
    "run_xes3g5m_ours_sota_serial.ps1",
    "run_xes3g5m_lightweight_baselines.ps1"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run setting: $expected"
    }
}

Write-Host "test_coco_triage_queue.ps1 passed"
