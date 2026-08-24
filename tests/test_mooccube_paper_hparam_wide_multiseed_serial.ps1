$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_hparam_wide_multiseed_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main wide-grid multiseed runner: $script"
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
    -SeedList "2026,2027" `
    -WaitPid 12345 `
    -OutputRootBase "outputs\_dryrun_paper_hparam_wide_seed2025" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_hparam_wide_seed2025" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper wide-grid multiseed dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main wide-grid multiseed dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "PriorHparamRoot=outputs\content_delta_pop5\course_hparam_sensitivity_e60_3seed",
    "OutputRootBase=outputs\_dryrun_paper_hparam_wide_seed2025",
    "CheckpointRootBase=checkpoints\_dryrun_paper_hparam_wide_seed2025",
    "Seeds=2026,2027",
    "WaitPid=12345",
    "RunCount=3",
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
    "Variant=reward_scale_0p75",
    "Family=reward_scale",
    "CoursePrereqW=0.06",
    "CourseConceptW=0.03",
    "CourseDiffW=0.0225",
    "CourseRedundantW=0.015",
    "Variant=reward_scale_1p25",
    "CoursePrereqW=0.1",
    "CourseConceptW=0.05",
    "CourseDiffW=0.0375",
    "CourseRedundantW=0.025",
    "Variant=prereq_gate_0p70",
    "Family=prereq_gate",
    "CoursePrereqGate=0.7"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper wide-grid multiseed dry-run line: $expected"
    }
}

$outCustom = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -SeedList "2027" `
    -VariantList "term_norm_ema,sample_beta_0p50" `
    -OutputRootBase "outputs\_dryrun_paper_hparam_wide_seed2025" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_hparam_wide_seed2025" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper wide-grid multiseed custom dry-run exited with code $LASTEXITCODE"
}

$customText = ($outCustom -join "`n")
foreach ($expected in @(
    "Seeds=2027",
    "RunCount=2",
    "Variant=term_norm_ema",
    "CourseTermNorm=ema",
    "Variant=sample_beta_0p50",
    "CourseSampleBeta=0.5"
)) {
    if ($customText -notlike "*$expected*") {
        throw "Missing expected custom dry-run line: $expected"
    }
}

$remainingWorker = Join-Path $repo "run_mooccube_paper_hparam_wide_remaining_worker.cmd"
$remainingLauncher = Join-Path $repo "launch_mooccube_paper_hparam_wide_remaining_now.cmd"
foreach ($path in @($remainingWorker, $remainingLauncher)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing wide-grid remaining-seed launch helper: $path"
    }
}

$remainingText = Get-Content -Raw -Encoding UTF8 -LiteralPath $remainingWorker
foreach ($expected in @(
    "-SeedList `"2026,2027`"",
    "set `"VARIANT_LIST=sample_beta_0p00,sample_beta_0p05,sample_beta_0p15,sample_beta_0p25,sample_beta_0p40,sample_beta_0p50,reward_scale_0p00,reward_scale_0p25,reward_scale_2p00`"",
    "-VariantList `"%VARIANT_LIST%`"",
    "course_hparam_wide_remaining_worker_latest_paths.txt",
    "course_hparam_wide_remaining_queue.log"
)) {
    if (-not $remainingText.Contains($expected)) {
        throw "Remaining wide-grid worker is missing expected configuration: $expected"
    }
}

foreach ($unexpected in @("term_norm_ema", "prereq_gate_", "-AllWideVariants")) {
    if ($remainingText.Contains($unexpected)) {
        throw "Remaining wide-grid worker should not include: $unexpected"
    }
}

Write-Host "test_mooccube_paper_hparam_wide_multiseed_serial.ps1 passed"
