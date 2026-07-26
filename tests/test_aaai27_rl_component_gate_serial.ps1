$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_aaai27_rl_component_gate_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing AAAI27 RL component gate runner: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -VariantList "wo_ppo_learning,wo_simulator_rollout" `
    -RunAllSeeds `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "AAAI27 RL component gate dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "run_all_seeds=True",
    "PLAN variant=wo_ppo_learning",
    "PLAN variant=wo_simulator_rollout",
    "DRY_RUN complete"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected AAAI27 gate dry-run line: $expected"
    }
}

Write-Host "test_aaai27_rl_component_gate_serial.ps1 passed"
