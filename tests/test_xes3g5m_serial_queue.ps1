$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_xes3g5m_ours_sota_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing XES3G5M serial queue script: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -OutputRoot "outputs\_dryrun_xes3g5m_serial" `
    -CheckpointRoot "checkpoints\_dryrun_xes3g5m_serial" `
    -Seeds 2025 `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "DataDir=processed_data_xes3g5m",
    "RelationDir=processed_data_xes3g5m\relations",
    "Protocol=strict_item_cold_balanced",
    "Seeds=2025",
    "Epochs=60",
    "Patience=60",
    "UseContentDelta=False",
    "PrereqGraphSource=concept",
    "OursFull.CourseFeedbackOnlyCold=False",
    "OursFull.CourseSampleOnlyCold=False",
    "OursFull.PrereqAuxOnlyCold=False",
    "NoCourse.UseCourseFeedback=False",
    "NoCourse.UseCourseReward=False",
    "NoCourse.UseCourseSample=False",
    "NoCourse.UsePrereqAux=False",
    "Baseline=ContentProfile",
    "Baseline=CGRC-paper",
    "USIM_STATIC_SPLIT_DIR="
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run setting: $expected"
    }
}

Write-Host "test_xes3g5m_serial_queue.ps1 passed"
