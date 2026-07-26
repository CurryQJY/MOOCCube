$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_sage_pseudocold_overnight_serial.ps1"

& $script -Repo $repo -DryRun -Epochs 3 -Patience 3 -Seeds @(2025, 2026) -Case Both | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "run_sage_pseudocold_overnight_serial.ps1 dry-run failed with exit code $LASTEXITCODE"
}

$planPath = Join-Path $repo "outputs\content_delta_pop5\pseudo_cold_sage_v1\pseudo_cold_sage_overnight_serial_plan.json"
if (-not (Test-Path -LiteralPath $planPath)) {
    throw "Missing dry-run plan at $planPath"
}

$plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
if ($plan.runs.Count -ne 4) {
    throw "Expected 4 dry-run entries for 2 cases x 2 seeds, got $($plan.runs.Count)"
}

$p0 = $plan.runs | Where-Object { $_.case_id -eq "P0_pseudo_only" } | Select-Object -First 1
$p1 = $plan.runs | Where-Object { $_.case_id -eq "P1_pseudo_sage" } | Select-Object -First 1
if ($null -eq $p0 -or $null -eq $p1) {
    throw "Missing P0/P1 run entries in overnight plan"
}

if ($p0.runner_params.UsePseudoColdTrain -ne $true -or $p1.runner_params.UsePseudoColdTrain -ne $true) {
    throw "Both overnight cases must enable pseudo-cold training"
}
if ($p0.runner_params.UseSageLite -ne $false -or $p0.runner_params.SageOnlyColdOrTail -ne $false) {
    throw "P0 must keep SAGE disabled"
}
if ($p1.runner_params.UseSageLite -ne $true -or $p1.runner_params.SageOnlyColdOrTail -ne $true) {
    throw "P1 must enable tail-gated SAGE"
}

$sharedFields = @(
    "DataDir",
    "RelationDir",
    "Protocol",
    "Epochs",
    "Patience",
    "UseContentDelta",
    "UsePseudoColdTrain",
    "PseudoColdMode",
    "PseudoColdRatio",
    "PseudoColdMinPop",
    "UseCourseFeedback",
    "UseCourseReward",
    "UseCourseSample",
    "UsePrereqAux",
    "MaskKnownPosNeg",
    "MaskSameItemNeg",
    "RunSampledEval",
    "SaveCkpt",
    "AutoResume",
    "ForceFresh",
    "SaveOptState"
)
foreach ($field in $sharedFields) {
    if ($p0.runner_params.$field -ne $p1.runner_params.$field) {
        throw "P0/P1 differ on shared field $field"
    }
}

Write-Host "test_sage_pseudocold_overnight_serial.ps1 passed"

& $script -Repo $repo -DryRun -Epochs 3 -Patience 3 -Seeds @(2025) -Case P2 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "run_sage_pseudocold_overnight_serial.ps1 P2 dry-run failed with exit code $LASTEXITCODE"
}

$p2PlanPath = Join-Path $repo "outputs\content_delta_pop5\pseudo_cold_sage_v1\pseudo_cold_sage_overnight_serial_p2_plan.json"
if (-not (Test-Path -LiteralPath $p2PlanPath)) {
    throw "Missing P2 dry-run plan at $p2PlanPath"
}
$plan = Get-Content -Raw -LiteralPath $p2PlanPath | ConvertFrom-Json
if ($plan.runs.Count -ne 1) {
    throw "Expected 1 P2 dry-run entry, got $($plan.runs.Count)"
}
$p2 = $plan.runs | Select-Object -First 1
if ($p2.case_id -ne "P2_pseudo_sage_twoexpert") {
    throw "Expected P2 case id, got $($p2.case_id)"
}
if ($p2.runner_params.UseSageLite -ne $true -or $p2.runner_params.SageOnlyColdOrTail -ne $true) {
    throw "P2 must enable tail-gated SAGE"
}
if ($p2.runner_params.SageUseTwoExpert -ne $true) {
    throw "P2 must enable SageUseTwoExpert"
}

Write-Host "test_sage_pseudocold_overnight_serial.ps1 P2 passed"
