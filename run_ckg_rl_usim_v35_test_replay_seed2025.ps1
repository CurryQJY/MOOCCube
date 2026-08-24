param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$RunTag = "",
    [string]$Device = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($RunTag -and $RunTag -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw "RunTag must start with an alphanumeric character and contain only letters, digits, underscores, or hyphens."
}

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$ScriptPath = "ckg_rl_usim_v35_test_replay.py"
$sourceOutputRelative = "outputs\ckg_rl_usim_v35_action_distill\seed2025"
$sourceCheckpointRelative = "checkpoints\ckg_rl_usim_v35_action_distill\seed2025"
$outputRelative = "outputs\ckg_rl_usim_v35_action_distill\test_replay_seed2025"
if ($RunTag) {
    $outputRelative = "outputs\ckg_rl_usim_v35_action_distill\test_replay_seed2025_$RunTag"
}
$outputRoot = Join-Path $repoPath $outputRelative

if (-not $DryRun -and (Test-Path -LiteralPath $outputRoot)) {
    throw "Refusing to overwrite an existing V3.5 test replay root. Choose a new RunTag."
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
        "--source-output-dir", $sourceOutputRelative,
        "--source-checkpoint-dir", $sourceCheckpointRelative,
        "--output-dir", $outputRelative
    )
    if ($Device) {
        $arguments += "--device"
        $arguments += $Device
    }
    if ($DryRun) {
        $arguments += "--dry-run"
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
