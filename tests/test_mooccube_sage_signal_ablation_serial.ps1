$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_sage_signal_ablation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube SAGE signal-ablation serial runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[double]`$CoursePrereqGate",
    "[double]`$CourseRedundantConceptGate",
    "USIM_FB_COURSE_PREREQ_GATE",
    "USIM_FB_COURSE_REDUNDANT_CONCEPT_GATE"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected course-signal control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -WaitPid 12345 `
    -SeedsToRun 2025 `
    -VariantList "wo_concept_match,wo_prereq_signal,wo_redundancy_signal,wo_all_course_signals" `
    -OutputRootBase "outputs\_dryrun_sage_signal_ablation" `
    -CheckpointRootBase "checkpoints\_dryrun_sage_signal_ablation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Signal-ablation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube SAGE course-signal ablation serial dry run",
    "WaitPid=12345",
    "Seeds=2025",
    "Variant=wo_concept_match",
    "CourseConceptW=0",
    "Variant=wo_prereq_signal",
    "CoursePrereqW=0",
    "CoursePrereqGate=1",
    "Variant=wo_redundancy_signal",
    "CourseRedundantW=0",
    "CourseRedundantConceptGate=0",
    "Variant=wo_all_course_signals",
    "UseCourseFeedback=False",
    "UseCourseReward=False",
    "UseCourseSample=False",
    "UsePrereqAux=False",
    "SageGateMode=bucket_mlp",
    "SageGateBucketStrategy=log",
    "SageTailPopRatio=0.002",
    "MaskKnownPosNeg=True",
    "MaskSameItemNeg=True"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected signal-ablation dry-run line: $expected"
    }
}

Write-Host "test_mooccube_sage_signal_ablation_serial.ps1 passed"
