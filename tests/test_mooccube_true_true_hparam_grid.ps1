$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_true_true_hparam_grid.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube True/True hparam grid runner: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -SeedsToRun 2025 `
    -VariantList "beta_0p00,beta_0p15,beta_0p25,reward_0p00,reward_2p00,horizon_1,horizon_3,horizon_10" `
    -OutputRootBase "outputs\_dryrun_true_true_hparam_grid" `
    -CheckpointRootBase "checkpoints\_dryrun_true_true_hparam_grid" `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "True/True hparam grid dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube True/True validation-only hparam grid dry run",
    "Protocol=strict_item_cold_balanced",
    "Seeds=2025",
    "Epochs=60",
    "Patience=60",
    "EarlyStopAverageMode=item_macro",
    "EarlyStopScoreMode=cold_only",
    "RunSampledEval=False",
    "MaskKnownPosNeg=True",
    "MaskSameItemNeg=True",
    "SaveCkpt=True",
    "AutoResume=True",
    "SaveOptState=True",
    "PlanFileMode=variant_list_scoped",
    "true_true_hparam_grid_plan_beta_0p00_beta_0p15_beta_0p25_reward_0p00_reward_2p00_horizon_1_horizon_3_horizon_10.json",
    "Variant=beta_0p00",
    "CourseSampleBeta=0",
    "Variant=beta_0p15",
    "CourseSampleBeta=0.15",
    "Variant=beta_0p25",
    "CourseSampleBeta=0.25",
    "Variant=reward_0p00",
    "CoursePrereqW=0",
    "CourseConceptW=0",
    "CourseDiffW=0",
    "CourseRedundantW=0",
    "Variant=reward_2p00",
    "CoursePrereqW=0.16",
    "CourseConceptW=0.08",
    "CourseDiffW=0.06",
    "CourseRedundantW=0.04",
    "Variant=horizon_1",
    "UsimSteps=1",
    "Variant=horizon_3",
    "UsimSteps=3",
    "Variant=horizon_10",
    "UsimSteps=10"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run line: $expected"
    }
}

Write-Host "test_mooccube_true_true_hparam_grid.ps1 passed"
