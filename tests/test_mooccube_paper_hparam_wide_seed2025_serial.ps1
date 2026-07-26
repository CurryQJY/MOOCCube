$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_hparam_wide_seed2025_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main seed2025 wide hyperparam runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[ValidateSet(`"none`", `"batch`", `"ema`")]",
    "[double]`$CoursePrereqGate",
    "[double]`$CourseConceptW",
    "[double]`$CourseDiffW",
    "[double]`$CourseRedundantW",
    "[double]`$CourseSampleBeta",
    "USIM_FB_COURSE_TERM_NORM",
    "USIM_FB_COURSE_SAMPLE_BETA"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected wide-grid hyperparam control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -Seed 2025 `
    -WaitPid 12345 `
    -VariantList "sample_beta_0p05,reward_scale_2p00,prereq_gate_0p70,term_norm_ema" `
    -OutputRootBase "outputs\_dryrun_paper_hparam_wide_seed2025" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_hparam_wide_seed2025" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper seed2025 wide hyperparam dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main seed2025 wide hyperparam serial dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "PriorHparamRoot=outputs\content_delta_pop5\course_hparam_sensitivity_e60_3seed",
    "WaitPid=12345",
    "Seed=2025",
    "RunCount=4",
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
    "Variant=sample_beta_0p05",
    "CourseSampleBeta=0.05",
    "Variant=reward_scale_2p00",
    "CoursePrereqW=0.16",
    "CourseConceptW=0.08",
    "CourseDiffW=0.06",
    "CourseRedundantW=0.04",
    "Variant=prereq_gate_0p70",
    "CoursePrereqGate=0.7",
    "Variant=term_norm_ema",
    "CourseTermNorm=ema"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper seed2025 wide hyperparam dry-run line: $expected"
    }
}

Write-Host "test_mooccube_paper_hparam_wide_seed2025_serial.ps1 passed"
