param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$checkpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_candidate"
$outputRoot = "outputs\recppo_research_repair\validation_inference_policy_screen"
$policies = @("static", "ppo", "greedy_similarity", "course_fit", "random")
$completed = @()

try {
    $env:USIM_ACTOR_EVAL_TARGET = "validation"
    foreach ($seed in $Seeds) {
        $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
        $finishedCheckpoint = Join-Path $checkpointRoot "$seedTag\finished.pt"
        if (-not (Test-Path -LiteralPath $finishedCheckpoint)) {
            Write-Host "SKIP seed=${seed}: finished.pt is not available"
            continue
        }

        foreach ($policy in $policies) {
            $policyRoot = Join-Path $outputRoot $policy
            $armOutput = Join-Path $policyRoot $seedTag
            $finalCsv = Join-Path $armOutput "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
            $auditJson = Join-Path $armOutput "actor_inference_audit.json"
            if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $auditJson)) {
                Write-Host "SKIP completed seed=$seed policy=$policy"
                continue
            }

            $env:USIM_ACTOR_INFERENCE_MODE = $policy
            $env:USIM_ACTOR_INFERENCE_SEED = "7001"
            Write-Host "===== VALIDATION POLICY seed=$seed policy=$policy START $(Get-Date -Format o) ====="

            & $runner `
                -ScriptPath "main_checkpoint_actor_inference_ab.py" `
                -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
                -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
                -Epochs 60 -Patience 60 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
                -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
                -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
                -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
                -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
                -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
                -PpoLossWeight 1.00 -RolloutPolicy "ppo" -RlResidualScale 1.00 `
                -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
                -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
                -UseUsimRefinedEval $true -UseCourseRerank $false -UseStructuredHardNeg $false `
                -UseSageLite $false -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
                -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
                -OutputRoot $policyRoot -CheckpointRoot $checkpointRoot `
                -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate

            if ($LASTEXITCODE -ne 0) {
                throw "Validation policy failed: seed=$seed policy=$policy exit=$LASTEXITCODE"
            }
            Write-Host "===== VALIDATION POLICY seed=$seed policy=$policy DONE $(Get-Date -Format o) ====="
        }
        $completed += $seed
    }

    if ($completed.Count -gt 0) {
        & .\py.bat validation_inference_policy_report.py --root $outputRoot --seeds $completed
        if ($LASTEXITCODE -ne 0) {
            throw "Validation policy report failed: exit=$LASTEXITCODE"
        }
    }
}
finally {
    Remove-Item Env:USIM_ACTOR_EVAL_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_SEED -ErrorAction SilentlyContinue
}
