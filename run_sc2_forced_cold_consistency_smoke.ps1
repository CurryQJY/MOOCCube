param(
    [switch]$DryRun,
    [switch]$SkipGpuWait,
    [double]$ConsistencyWeight = 0.10,
    [double]$ConsistencyTemperature = 0.20,
    [int]$MinFreeGpuMiB = 9000,
    [int]$GpuPollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$sc2Environment = @(
    "USIM_SC2_CONSISTENCY_WEIGHT",
    "USIM_SC2_CONSISTENCY_TEMP",
    "USIM_SC2_CONSISTENCY_WARM_ONLY"
)
$originalEnvironment = @{}
foreach ($name in $sc2Environment) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

function Get-FreeGpuMemoryMiB {
    try {
        $line = nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        $parsedMiB = 0
        if (
            -not [string]::IsNullOrWhiteSpace($line) -and
            [int]::TryParse(([string]$line).Trim(), [ref]$parsedMiB)
        ) {
            return $parsedMiB
        }
        Write-Host (">> GPU query returned no parseable value: exit={0}, raw='{1}'" -f $LASTEXITCODE, $line)
    }
    catch {
        Write-Host (">> GPU query diagnostic: exception='{0}'" -f $_.Exception.Message)
        return $null
    }
    return $null
}

$runnerArgs = @{
    PythonRunner = ".\py.bat"
    ScriptPath = ".\usim_feedback_fast3_sc2_consistency.py"
    OutputRoot = "outputs\sc2_forced_cold_consistency_smoke"
    CheckpointRoot = "checkpoints\sc2_forced_cold_consistency_smoke"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 1
    Patience = 1
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $true
    PseudoColdMode = "all_eligible"
    PseudoColdRatio = 1.0
    PseudoColdMinPop = 1
    UseCourseFeedback = $true
    UseCourseReward = $true
    UsePrereqAux = $true
    UseCourseSample = $true
    UseUsimRefinedEval = $true
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    RlResidualScale = 1.0
    SaveCkpt = $true
    AutoResume = $false
    ForceFresh = $true
    SaveOptState = $true
    SkipAggregate = $true
}

try {
    $env:USIM_SC2_CONSISTENCY_WEIGHT = [string]$ConsistencyWeight
    $env:USIM_SC2_CONSISTENCY_TEMP = [string]$ConsistencyTemperature
    $env:USIM_SC2_CONSISTENCY_WARM_ONLY = "1"

    if ($DryRun) {
        Write-Host ">> SC2Rec-style forced-cold consistency smoke (dry run)"
        Write-Host ("Consistency: weight={0}, temp={1}, warm_only=1" -f $ConsistencyWeight, $ConsistencyTemperature)
        $runnerArgs.GetEnumerator() |
            Sort-Object Name |
            ForEach-Object { Write-Host ("{0}={1}" -f $_.Name, $_.Value) }
        exit 0
    }

    if (-not $SkipGpuWait) {
        $pollSeconds = [Math]::Max(1, [Math]::Min(60, $GpuPollSeconds))
        while ($true) {
            $freeMiB = Get-FreeGpuMemoryMiB
            if ($null -eq $freeMiB) {
                Write-Host ">> GPU memory query unavailable; retry after ${pollSeconds}s"
                Start-Sleep -Seconds $pollSeconds
                continue
            }
            if ($freeMiB -ge $MinFreeGpuMiB) {
                Write-Host (">> GPU gate passed: {0} MiB free" -f $freeMiB)
                break
            }
            Write-Host (">> Waiting for GPU: {0} MiB free, need {1} MiB" -f $freeMiB, $MinFreeGpuMiB)
            Start-Sleep -Seconds $pollSeconds
        }
    }

    & .\run_usim_feedback_fast3_content_delta_static.ps1 @runnerArgs
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $sc2Environment) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
    }
}
