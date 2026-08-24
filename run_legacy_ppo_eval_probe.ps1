param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$SampleSeeds = @(9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"

function Invoke-Probe([string]$Mode, [int]$EvalSeed, [string]$Label) {
    $env:USIM_EVAL_ACTION_MODE = $Mode
    $env:USIM_EVAL_PROBE_SEED = [string]$EvalSeed
    Write-Host ("===== EVAL PROBE {0} seed={1} START {2} =====" -f $Mode, $EvalSeed, (Get-Date -Format s))
    & $runner `
        -ScriptPath "usim_feedback_fast3_content_delta_eval_probe.py" `
        -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @(2025) `
        -Epochs 20 -Patience 5 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
        -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
        -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
        -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
        -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
        -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
        -PpoLossWeight 0.50 -RolloutPolicy "ppo" -UseCourseFeedback $true `
        -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
        -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
        -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
        -OutputRoot "outputs\recovery_validation\legacy_ppo_eval_probe\$Label" `
        -CheckpointRoot "checkpoints\recovery_validation\main_table_51ea12fc_ppo_weight_screen\ppo_w0p50" `
        -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate
    if ($LASTEXITCODE -ne 0) { throw "Evaluation probe failed: $Label" }
    Write-Host ("===== EVAL PROBE {0} seed={1} DONE {2} =====" -f $Mode, $EvalSeed, (Get-Date -Format s))
}

try {
    foreach ($seed in $SampleSeeds) {
        Invoke-Probe -Mode "sample" -EvalSeed $seed -Label ("sample_seed_{0}" -f $seed)
    }
    Invoke-Probe -Mode "argmax" -EvalSeed 9201 -Label "argmax"
}
finally {
    Remove-Item Env:USIM_EVAL_ACTION_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_EVAL_PROBE_SEED -ErrorAction SilentlyContinue
}
