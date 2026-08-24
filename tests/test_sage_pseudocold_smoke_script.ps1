$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_sage_pseudocold_smoke.ps1"

& $script -Repo $repo -DryRun -Epochs 3 -Seeds @(2025) | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "run_sage_pseudocold_smoke.ps1 dry-run failed with exit code $LASTEXITCODE"
}

$planPath = Join-Path $repo "outputs\content_delta_pop5\pseudo_cold_sage_v1\pseudo_cold_sage_smoke_plan.json"
if (-not (Test-Path -LiteralPath $planPath)) {
    throw "Missing dry-run plan at $planPath"
}

$plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
if ($plan.cases.Count -ne 2) {
    throw "Expected 2 cases, got $($plan.cases.Count)"
}

$pseudoOnly = $plan.cases | Where-Object { $_.case_id -eq "P0_pseudo_only" } | Select-Object -First 1
$pseudoSage = $plan.cases | Where-Object { $_.case_id -eq "P1_pseudo_sage" } | Select-Object -First 1

if ($null -eq $pseudoOnly -or $null -eq $pseudoSage) {
    throw "Missing P0/P1 cases in plan"
}
if ($pseudoOnly.runner_params.UsePseudoColdTrain -ne $true -or $pseudoSage.runner_params.UsePseudoColdTrain -ne $true) {
    throw "Both cases must enable pseudo-cold training"
}
if ($pseudoOnly.runner_params.UseSageLite -ne $false) {
    throw "P0 must keep SAGE disabled"
}
if ($pseudoSage.runner_params.UseSageLite -ne $true -or $pseudoSage.runner_params.SageOnlyColdOrTail -ne $true) {
    throw "P1 must enable tail-gated SAGE"
}
if ($pseudoOnly.runner_params.MaskKnownPosNeg -ne $true -or $pseudoSage.runner_params.MaskKnownPosNeg -ne $true) {
    throw "Smoke config must keep mask=true aligned with the current main comparison"
}

Write-Host "test_sage_pseudocold_smoke_script.ps1 passed"
