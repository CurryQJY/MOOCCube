$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_signal_ablation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main signal-ablation serial runner: $script"
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
    -VariantList "wo_concept_match,wo_prereq_signal,wo_difficulty_signal,wo_redundancy_signal" `
    -OutputRootBase "outputs\_dryrun_paper_signal_ablation" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_signal_ablation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper signal-ablation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main course-signal ablation serial dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "WaitPid=12345",
    "Seeds=2025",
    "UseSageLite=False",
    "SageTwoExpertScoreFusion=False",
    "UseSageAuxLoss=False",
    "CourseFeedbackOnlyCold=False",
    "CourseSampleOnlyCold=False",
    "PrereqAuxOnlyCold=False",
    "MaskKnownPosNeg=False",
    "MaskSameItemNeg=False",
    "UseCourseFeedback=True",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True",
    "Variant=wo_concept_match",
    "CourseConceptW=0",
    "Variant=wo_prereq_signal",
    "CoursePrereqW=0",
    "CoursePrereqGate=1",
    "Variant=wo_difficulty_signal",
    "CourseDiffW=0",
    "Variant=wo_redundancy_signal",
    "CourseRedundantW=0",
    "CourseRedundantConceptGate=0"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper signal-ablation dry-run line: $expected"
    }
}

Write-Host "test_mooccube_paper_signal_ablation_serial.ps1 passed"
