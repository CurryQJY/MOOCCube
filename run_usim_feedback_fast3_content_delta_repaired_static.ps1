param(
    [int]$RecPpoWarmupEpochs = -1,
    [string]$WarmupStageCheckpoint = "",
    [double]$RecPpoActorLr = 0.0,
    [double]$RecPpoCriticLr = 0.0,
    [double]$RecPpoBehaviorCeWeight = -1.0,
    [double]$RecPpoTerminalValueWeight = -1.0,
    [double]$RecPpoTargetKl = -1.0,
    [int]$RecPpoResidualRampEpochs = -1,
    [double]$RecPpoMaxResidualNorm = -1.0,
    [double]$RecPpoPolicyTemperature = -1.0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_usim_feedback_fast3_content_delta_static.ps1"

function Convert-RunnerBoolValue([object]$Value) {
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    $text = ([string]$Value).Trim()
    switch -Regex ($text) {
        '^\$?true$' { return $true }
        '^\$?false$' { return $false }
        '^1$' { return $true }
        '^0$' { return $false }
        default { return $Value }
    }
}

function Convert-RunnerRemainingArgs([object[]]$ArgsList) {
    $boolParamNames = @(
        "-UseContentDelta",
        "-ContentDeltaPaperStyle",
        "-ContentDeltaReplaceItem",
        "-ContentDeltaColdOnly",
        "-ContentDeltaTrainOnIdDropout",
        "-UsePseudoColdTrain",
        "-UsePaac",
        "-UseCourseFeedback",
        "-UseCourseReward",
        "-UsePrereqAux",
        "-PrereqAuxOnlyCold",
        "-UseSageLite",
        "-SageOnlyColdOrTail",
        "-SageUseTwoExpert",
        "-SageTwoExpertScoreFusion",
        "-UseSageAuxLoss",
        "-SageAuxOnlyStrictCold",
        "-SageAuxDetachUser",
        "-UseCgrcRecon",
        "-CgrcReconOnlyColdOrTail",
        "-CgrcReconDetachUser",
        "-CourseFeedbackOnlyCold",
        "-UseCourseSample",
        "-CourseSampleOnlyCold",
        "-UseCourseRerank",
        "-CourseRerankOnlyCold",
        "-UseStructuredHardNeg",
        "-MaskKnownPosNeg",
        "-MaskSameItemNeg",
        "-UseSgUrinit",
        "-TrainForceCold",
        "-UseUsimRefinedEval",
        "-AuxHotOnly",
        "-RunSampledEval",
        "-SaveCkpt",
        "-AutoResume",
        "-ForceFresh",
        "-SaveOptState"
    )
    $boolParams = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $boolParamNames) {
        [void]$boolParams.Add($name)
    }
    $switchParams = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    [void]$switchParams.Add("-SkipAggregate")

    $parsed = @{}
    for ($idx = 0; $idx -lt $ArgsList.Count; $idx++) {
        $arg = $ArgsList[$idx]
        $text = [string]$arg
        $colonAt = $text.IndexOf(":")
        if ($colonAt -gt 0) {
            $name = $text.Substring(0, $colonAt)
            if ($boolParams.Contains($name)) {
                $parsed[$name.TrimStart("-")] = Convert-RunnerBoolValue $text.Substring($colonAt + 1)
                continue
            }
            if ($switchParams.Contains($name)) {
                $parsed[$name.TrimStart("-")] = Convert-RunnerBoolValue $text.Substring($colonAt + 1)
                continue
            }
        }
        if (-not $text.StartsWith("-", [System.StringComparison]::Ordinal)) {
            throw "Unexpected positional argument for repaired runner wrapper: $text"
        }
        if ($switchParams.Contains($text)) {
            $parsed[$text.TrimStart("-")] = $true
            continue
        }
        if ($boolParams.Contains($text) -and ($idx + 1) -lt $ArgsList.Count) {
            $parsed[$text.TrimStart("-")] = Convert-RunnerBoolValue $ArgsList[$idx + 1]
            $idx++
            continue
        }
        if (($idx + 1) -ge $ArgsList.Count) {
            throw "Missing value for repaired runner wrapper argument: $text"
        }
        $parsed[$text.TrimStart("-")] = $ArgsList[$idx + 1]
        $idx++
    }
    return $parsed
}

$runnerArgs = Convert-RunnerRemainingArgs $RemainingArgs
if (-not $runnerArgs.ContainsKey("ScriptPath")) {
    $runnerArgs["ScriptPath"] = "usim_feedback_fast3_content_delta_repaired.py"
}
if (-not $runnerArgs.ContainsKey("PpoLossWeight")) {
    $runnerArgs["PpoLossWeight"] = 1.0
}
if (-not $runnerArgs.ContainsKey("RolloutPolicy")) {
    $runnerArgs["RolloutPolicy"] = "ppo"
}
if (-not $runnerArgs.ContainsKey("UseContentDelta")) {
    # Default OFF: single-variable ablation under research-v2 found delta hurts cold metrics.
    $runnerArgs["UseContentDelta"] = $false
}
if (-not $runnerArgs.ContainsKey("Epochs")) {
    $runnerArgs["Epochs"] = 30
}
if (-not $runnerArgs.ContainsKey("Patience")) {
    $runnerArgs["Patience"] = 12
}
if (-not $runnerArgs.ContainsKey("PseudoColdMode")) {
    $runnerArgs["PseudoColdMode"] = "all_eligible"
}
if (-not $runnerArgs.ContainsKey("PseudoColdRatio")) {
    $runnerArgs["PseudoColdRatio"] = 1.0
}
if (-not $runnerArgs.ContainsKey("PseudoColdMinPop")) {
    $runnerArgs["PseudoColdMinPop"] = 1
}
if (-not $runnerArgs.ContainsKey("RlResidualScale")) {
    $runnerArgs["RlResidualScale"] = 0.30
}
if (-not $runnerArgs.ContainsKey("SaveCkpt")) {
    $runnerArgs["SaveCkpt"] = $true
}
if (-not $runnerArgs.ContainsKey("AutoResume")) {
    # Resume unless train config fingerprint changes (checked in Python).
    $runnerArgs["AutoResume"] = $true
}
if (-not $runnerArgs.ContainsKey("ForceFresh")) {
    $runnerArgs["ForceFresh"] = $false
}
if (-not $runnerArgs.ContainsKey("SaveOptState")) {
    $runnerArgs["SaveOptState"] = $true
}

$recppoTrackedEnv = @(
    "USIM_RECPPO_EARLY_STOP_MODE",
    "USIM_RECPPO_WARMUP_EPOCHS",
    "USIM_FB_WARMUP_STAGE_CKPT",
    "USIM_RECPPO_ACTOR_LR",
    "USIM_RECPPO_CRITIC_LR",
    "USIM_RECPPO_BEHAVIOR_CE_W",
    "USIM_RECPPO_TERM_VALUE_W",
    "USIM_RECPPO_TARGET_KL",
    "USIM_RECPPO_RESIDUAL_RAMP_EPOCHS",
    "USIM_RECPPO_MAX_RESIDUAL_NORM",
    "USIM_RECPPO_POLICY_TEMP",
    "USIM_RECPPO_STRICT_DETERMINISM",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG"
)
$recppoOriginalEnv = @{}
foreach ($name in $recppoTrackedEnv) {
    $recppoOriginalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$runnerExitCode = 1

try {
if (-not (Test-Path Env:USIM_RECPPO_EARLY_STOP_MODE)) {
    $env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
}
if ($RecPpoWarmupEpochs -ge 0) {
    $env:USIM_RECPPO_WARMUP_EPOCHS = [string]$RecPpoWarmupEpochs
}
if (-not [string]::IsNullOrWhiteSpace($WarmupStageCheckpoint)) {
    $env:USIM_FB_WARMUP_STAGE_CKPT = [System.IO.Path]::GetFullPath($WarmupStageCheckpoint)
}
if ($RecPpoActorLr -gt 0.0) {
    $env:USIM_RECPPO_ACTOR_LR = [string]$RecPpoActorLr
}
if ($RecPpoCriticLr -gt 0.0) {
    $env:USIM_RECPPO_CRITIC_LR = [string]$RecPpoCriticLr
}
if ($RecPpoBehaviorCeWeight -ge 0.0) {
    $env:USIM_RECPPO_BEHAVIOR_CE_W = [string]$RecPpoBehaviorCeWeight
}
if ($RecPpoTerminalValueWeight -ge 0.0) {
    $env:USIM_RECPPO_TERM_VALUE_W = [string]$RecPpoTerminalValueWeight
}
if ($RecPpoTargetKl -ge 0.0) {
    $env:USIM_RECPPO_TARGET_KL = [string]$RecPpoTargetKl
}
if ($RecPpoResidualRampEpochs -gt 0) {
    $env:USIM_RECPPO_RESIDUAL_RAMP_EPOCHS = [string]$RecPpoResidualRampEpochs
}
if ($RecPpoMaxResidualNorm -ge 0.0) {
    $env:USIM_RECPPO_MAX_RESIDUAL_NORM = [string]$RecPpoMaxResidualNorm
}
if ($RecPpoPolicyTemperature -gt 0.0) {
    $env:USIM_RECPPO_POLICY_TEMP = [string]$RecPpoPolicyTemperature
}
if (-not (Test-Path Env:PYTHONHASHSEED)) {
    $env:PYTHONHASHSEED = "0"
}
if (-not (Test-Path Env:CUBLAS_WORKSPACE_CONFIG)) {
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
}
if (-not (Test-Path Env:USIM_RECPPO_STRICT_DETERMINISM)) {
    $env:USIM_RECPPO_STRICT_DETERMINISM = "1"
}

& $runner @runnerArgs
$runnerExitCode = $LASTEXITCODE
}
finally {
    foreach ($name in $recppoTrackedEnv) {
        if ($null -eq $recppoOriginalEnv[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $recppoOriginalEnv[$name]
        }
    }
}

exit $runnerExitCode
