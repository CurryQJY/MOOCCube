$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_hparam_sensitivity_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main hyperparam sensitivity runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[double]`$CoursePrereqGate",
    "[double]`$CourseConceptW",
    "[double]`$CourseDiffW",
    "[double]`$CourseRedundantW",
    "[double]`$CourseSampleBeta",
    "USIM_FB_COURSE_PREREQ_GATE",
    "USIM_FB_COURSE_SAMPLE_BETA"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected hyperparam control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -WaitPid 12345 `
    -SeedsToRun 2025 `
    -VariantList "sample_beta_0p10,reward_scale_1p5,term_norm_batch,prereq_gate_0p35" `
    -OutputRootBase "outputs\_dryrun_paper_hparam_sensitivity" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_hparam_sensitivity" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper hyperparam sensitivity dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main hyperparam sensitivity serial dry run",
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
    "Variant=sample_beta_0p10",
    "CourseSampleBeta=0.1",
    "Variant=reward_scale_1p5",
    "CoursePrereqW=0.12",
    "CourseConceptW=0.06",
    "CourseDiffW=0.045",
    "CourseRedundantW=0.03",
    "Variant=term_norm_batch",
    "CourseTermNorm=batch",
    "Variant=prereq_gate_0p35",
    "CoursePrereqGate=0.35"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper hyperparam dry-run line: $expected"
    }
}

Write-Host "test_mooccube_paper_hparam_sensitivity_serial.ps1 passed"
