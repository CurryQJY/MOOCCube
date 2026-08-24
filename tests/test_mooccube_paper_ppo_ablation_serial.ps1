$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_ppo_ablation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main PPO-ablation serial runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[double]`$PpoLossWeight",
    "USIM_PPO_LOSS_WEIGHT"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected PPO-ablation control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -WaitPid 12345 `
    -SeedsToRun 2025 `
    -VariantList "wo_ppo_loss,wo_simulator_t0,static_content_masked_scorer" `
    -OutputRootBase "outputs\_dryrun_paper_ppo_ablation" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_ppo_ablation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper PPO-ablation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main PPO ablation serial dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "WaitPid=12345",
    "Seeds=2025",
    "Variant=wo_ppo_loss",
    "Label=w/o PPO Loss",
    "UsimSteps=5",
    "PpoLossWeight=0",
    "Variant=wo_simulator_t0",
    "Label=w/o Simulator (T=0)",
    "UsimSteps=0",
    "PpoLossWeight=1",
    "Variant=static_content_masked_scorer",
    "Label=Static content+mask scorer",
    "UseCourseReward=False",
    "UseCourseSample=False"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper PPO-ablation dry-run line: $expected"
    }
}

$defaultOut = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -SeedsToRun 2025 `
    -OutputRootBase "outputs\_dryrun_paper_ppo_ablation_default" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_ppo_ablation_default" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper PPO-ablation default dry-run exited with code $LASTEXITCODE"
}

$defaultText = ($defaultOut -join "`n")
foreach ($expected in @(
    "IncludeAlreadyRunCoreVariants=False",
    "Variant=wo_ppo_loss",
    "Variant=static_content_masked_scorer"
)) {
    if ($defaultText -notlike "*$expected*") {
        throw "Missing expected default paper PPO-ablation dry-run line: $expected"
    }
}

foreach ($unexpected in @(
    "Variant=wo_simulator_t0",
    "Variant=wo_forced_cold_masking"
)) {
    if ($defaultText -like "*$unexpected*") {
        throw "Default PPO-ablation dry-run should not include already-run core variant: $unexpected"
    }
}

Write-Host "test_mooccube_paper_ppo_ablation_serial.ps1 passed"
