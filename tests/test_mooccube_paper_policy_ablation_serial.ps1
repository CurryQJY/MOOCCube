$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_policy_ablation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing MOOCCube paper-main policy-ablation serial runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[string]`$RolloutPolicy",
    "USIM_ROLLOUT_POLICY"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected rollout-policy control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -WaitPid 12345 `
    -SeedsToRun 2025 `
    -OutputRootBase "outputs\_dryrun_paper_policy_ablation" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_policy_ablation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper policy-ablation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main policy ablation serial dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "WaitPid=12345",
    "Seeds=2025",
    "Variant=random_policy_rollout",
    "Label=Random policy rollout",
    "RolloutPolicy=random",
    "UsimSteps=5",
    "PpoLossWeight=0",
    "Variant=greedy_similarity_policy",
    "Label=Greedy similarity policy",
    "RolloutPolicy=greedy_similarity",
    "Variant=course_fit_policy",
    "Label=Course-fit heuristic policy",
    "RolloutPolicy=course_fit",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected paper policy-ablation dry-run line: $expected"
    }
}

Write-Host "test_mooccube_paper_policy_ablation_serial.ps1 passed"
