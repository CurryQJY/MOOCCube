param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [string]$RunTag = "",
    [int]$TeacherEpochs = 20,
    [int]$GeneratorEpochs = 30,
    [int]$PolicyEpochs = 15,
    [int]$BatchSize = 1024,
    [int]$EvalBatchSize = 2048,
    [double]$ActionTemperature = 0.005,
    [switch]$DryRun,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

if ($Seed -ne 2025) {
    throw "This V3.6 viability launcher is pinned to seed 2025. Create a separate launcher for another seed."
}
if ($RunTag -and $RunTag -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw "RunTag must start with an alphanumeric character and contain only letters, digits, underscores, or hyphens."
}
if ($TeacherEpochs -lt 1 -or $GeneratorEpochs -lt 1 -or $PolicyEpochs -lt 1 -or $BatchSize -lt 1 -or $EvalBatchSize -lt 1) {
    throw "Epoch and batch-size arguments must be positive."
}
if ($ActionTemperature -le 0.0) {
    throw "ActionTemperature must be positive."
}

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$ScriptPath = "ckg_rl_usim_v36_global_stable_distill.py"
$outputRelative = "outputs\ckg_rl_usim_v36_global_stable_distill\seed$Seed"
$checkpointRelative = "checkpoints\ckg_rl_usim_v36_global_stable_distill\seed$Seed"
if ($RunTag) {
    $outputRelative = "outputs\ckg_rl_usim_v36_global_stable_distill\seed${Seed}_${RunTag}"
    $checkpointRelative = "checkpoints\ckg_rl_usim_v36_global_stable_distill\seed${Seed}_${RunTag}"
}
if ($Smoke) {
    $outputRelative = "outputs\ckg_rl_usim_v36_global_stable_distill\smoke_seed$Seed"
    $checkpointRelative = "checkpoints\ckg_rl_usim_v36_global_stable_distill\smoke_seed$Seed"
    if ($RunTag) {
        $outputRelative = "outputs\ckg_rl_usim_v36_global_stable_distill\smoke_seed${Seed}_${RunTag}"
        $checkpointRelative = "checkpoints\ckg_rl_usim_v36_global_stable_distill\smoke_seed${Seed}_${RunTag}"
    }
}
$outputRoot = Join-Path $repoPath $outputRelative
$checkpointRoot = Join-Path $repoPath $checkpointRelative

if (-not $DryRun -and ((Test-Path -LiteralPath $outputRoot) -or (Test-Path -LiteralPath $checkpointRoot))) {
    throw "Refusing to overwrite an existing V3.6 run root. Choose a new isolated RunTag."
}

$lockedEnvironment = @{
    "USIM_CLEAN_RANDOM_ID_DROPOUT" = "0"
    "USIM_CLEAN_CANDIDATE_MODE" = "legal_state_retrieval"
}
$originalEnvironment = @{}
foreach ($name in $lockedEnvironment.Keys) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

Push-Location -LiteralPath $repoPath
try {
    foreach ($pair in $lockedEnvironment.GetEnumerator()) {
        Set-Item "Env:$($pair.Key)" ([string]$pair.Value)
    }
    $arguments = @(
        $ScriptPath,
        "--seed", [string]$Seed,
        "--output-dir", $outputRelative,
        "--checkpoint-dir", $checkpointRelative,
        "--teacher-epochs", [string]$TeacherEpochs,
        "--generator-epochs", [string]$GeneratorEpochs,
        "--policy-epochs", [string]$PolicyEpochs,
        "--batch-size", [string]$BatchSize,
        "--eval-batch-size", [string]$EvalBatchSize,
        "--rank-temperature", "0.20",
        "--panel-size", "48",
        "--panel-positive-count", "8",
        "--panel-hard-count", "16",
        "--action-temperature", [string]$ActionTemperature,
        "--global-anchor-count", "128",
        "--global-stability-weight", "10.0",
        "--expert-action-fraction", "0.5",
        "--use-course-signal"
    )
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    if ($Smoke) {
        $arguments += "--smoke"
    }
    & .\py.bat @arguments
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $lockedEnvironment.Keys) {
        if ($null -eq $originalEnvironment[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $originalEnvironment[$name]
        }
    }
    Pop-Location
}
