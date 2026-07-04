$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_junyi_course_scope_false_e60.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing Junyi course-scope false runner: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -OutputRoot "outputs\_dryrun_junyi_course_scope_false" `
    -CheckpointRoot "checkpoints\_dryrun_junyi_course_scope_false" `
    -Seeds 2025 `
    -DryRun `
    -NoAggregate

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "DataDir=processed_data_junyi",
    "RelationDir=processed_data_junyi\relations",
    "Protocol=strict_item_cold_balanced",
    "ColdThresholds=1",
    "Seeds=2025",
    "Epochs=60",
    "Patience=60",
    "EarlyStopAverageMode=item_macro",
    "UseContentDelta=False",
    "UsePseudoColdTrain=False",
    "UsePaac=False",
    "UseCourseFeedback=True",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True",
    "CourseFeedbackOnlyCold=False",
    "CourseSampleOnlyCold=False",
    "PrereqAuxOnlyCold=False",
    "UseSageLite=False",
    "SageUseTwoExpert=False",
    "UseSageAuxLoss=False",
    "UseCourseRerank=False",
    "UseStructuredHardNeg=False",
    "MaskKnownPosNeg=True",
    "MaskSameItemNeg=True",
    "RunSampledEval=False",
    "SaveCkpt=True",
    "SaveOptState=True"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run setting: $expected"
    }
}

Write-Host "test_junyi_course_scope_false_script.ps1 passed"
