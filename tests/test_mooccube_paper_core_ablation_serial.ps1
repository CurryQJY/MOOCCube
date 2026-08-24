$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_core_ablation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main core-ablation serial runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[bool]`$TrainForceCold",
    "[int]`$UsimSteps",
    "USIM_TRAIN_FORCE_COLD",
    "USIM_STEPS"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected core-ablation control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -WaitPid 12345 `
    -SeedsToRun 2025 `
    -VariantList "wo_forced_cold_masking,wo_simulator_t0" `
    -OutputRootBase "outputs\_dryrun_paper_core_ablation" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_core_ablation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper core-ablation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main core ablation serial dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "WaitPid=12345",
    "Seeds=2025",
    "UseCourseFeedback=True",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True",
    "TrainForceCold=True",
    "UsimSteps=5",
    "Variant=wo_forced_cold_masking",
    "TrainForceCold=False",
    "Variant=wo_simulator_t0",
    "UsimSteps=0"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper core-ablation dry-run line: $expected"
    }
}

Write-Host "test_mooccube_paper_core_ablation_serial.ps1 passed"
