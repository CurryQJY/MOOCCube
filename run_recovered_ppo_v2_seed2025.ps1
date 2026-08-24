param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 20,
    [int]$Patience = 5,
    [double]$PpoLossWeight = 0.50,
    [double]$RewardTerminalWeight = 2.0,
    [double]$RewardGainWeight = 1.0,
    [double]$RewardGainClip = 0.05,
    [bool]$ForceFresh = $true
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$saved = @{}
foreach ($name in @("USIM_FB_REWARD_TERM_W", "USIM_FB_REWARD_GAIN_W", "USIM_FB_REWARD_GAIN_CLIP")) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:USIM_FB_REWARD_TERM_W = [string]$RewardTerminalWeight
    $env:USIM_FB_REWARD_GAIN_W = [string]$RewardGainWeight
    $env:USIM_FB_REWARD_GAIN_CLIP = [string]$RewardGainClip

    & (Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1") `
        -ScriptPath "usim_feedback_fast3_content_delta_ppo_v2.py" `
        -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($Seed) `
        -Epochs $Epochs -Patience $Patience `
        -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
        -UseContentDelta $false -UsePseudoColdTrain $false `
        -PseudoColdMode "batch_random" -PseudoColdRatio 0.3 -PseudoColdMinPop 5 `
        -UsePaac $false -CoursePrereqW 0.08 -CoursePrereqGate 0.20 `
        -CourseConceptW 0.04 -CourseDiffW 0.03 -CourseRedundantW 0.02 `
        -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
        -CourseTermNorm "none" -CourseSampleBeta 0.20 `
        -TrainForceCold $true -UsimSteps 5 -PpoLossWeight $PpoLossWeight `
        -RolloutPolicy "ppo" -UseCourseFeedback $true -UseCourseReward $true `
        -UseCourseSample $true -UsePrereqAux $true `
        -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false `
        -PrereqAuxOnlyCold $false -MaskKnownPosNeg $true -MaskSameItemNeg $true `
        -RunSampledEval $false `
        -OutputRoot "outputs\recovery_validation\main_table_51ea12fc_ppo_v2" `
        -CheckpointRoot "checkpoints\recovery_validation\main_table_51ea12fc_ppo_v2" `
        -SaveCkpt $true -AutoResume $true -ForceFresh $ForceFresh `
        -SaveOptState $true -SkipAggregate
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}
