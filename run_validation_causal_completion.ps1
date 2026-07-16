param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

if ($WaitForProcessId -gt 0) {
    $existing = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host "Waiting for existing experiment PID=$WaitForProcessId at $(Get-Date -Format o)"
        Wait-Process -Id $WaitForProcessId
    }
}

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputBase = "outputs\recppo_research_repair\validation_causal_completion"
$integrityRows = @()

$arms = @(
    [ordered]@{
        Name = "wo_ppo_course_fit"
        CheckpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_core_ablation\wo_ppo_loss"
        UsimSteps = 5
        PpoLossWeight = 0.0
        InferenceStepsOverride = ""
    },
    [ordered]@{
        Name = "t0_course_fit"
        CheckpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_core_ablation\wo_simulator"
        UsimSteps = 0
        PpoLossWeight = 1.0
        InferenceStepsOverride = "5"
    }
)

try {
    $env:USIM_ACTOR_EVAL_TARGET = "validation"
    $env:USIM_ACTOR_INFERENCE_MODE = "course_fit"
    $env:USIM_ACTOR_INFERENCE_SEED = "7001"
    $env:USIM_COURSE_MATCH_EXCLUDE_TARGET = "false"

    foreach ($arm in $arms) {
        $outputRoot = Join-Path $outputBase $arm.Name
        if ([string]::IsNullOrWhiteSpace($arm.InferenceStepsOverride)) {
            Remove-Item Env:USIM_INFERENCE_STEPS_OVERRIDE -ErrorAction SilentlyContinue
        } else {
            $env:USIM_INFERENCE_STEPS_OVERRIDE = $arm.InferenceStepsOverride
        }

        foreach ($seed in $Seeds) {
            $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
            $checkpoint = Join-Path $arm.CheckpointRoot "$seedTag\finished.pt"
            if (-not (Test-Path -LiteralPath $checkpoint)) {
                throw "Missing frozen checkpoint: $checkpoint"
            }

            $itemBefore = Get-Item -LiteralPath $checkpoint
            $hashBefore = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
            $armOutput = Join-Path $outputRoot $seedTag
            $finalCsv = Join-Path $armOutput "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
            $auditJson = Join-Path $armOutput "actor_inference_audit.json"

            if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $auditJson)) {
                Write-Host "SKIP completed validation arm=$($arm.Name) seed=$seed"
            } else {
                Write-Host "===== VALIDATION CAUSAL arm=$($arm.Name) seed=$seed START $(Get-Date -Format o) ====="
                & $runner `
                    -ScriptPath "main_checkpoint_actor_inference_ab.py" `
                    -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
                    -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
                    -Epochs 60 -Patience 60 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
                    -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
                    -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
                    -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
                    -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
                    -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true `
                    -UsimSteps $arm.UsimSteps -PpoLossWeight $arm.PpoLossWeight `
                    -RolloutPolicy "ppo" -RlResidualScale 1.00 `
                    -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
                    -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
                    -UseUsimRefinedEval $true -UseCourseRerank $false -UseStructuredHardNeg $false `
                    -UseSageLite $false -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
                    -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
                    -OutputRoot $outputRoot -CheckpointRoot $arm.CheckpointRoot `
                    -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate

                if ($LASTEXITCODE -ne 0) {
                    throw "Validation evaluation failed: arm=$($arm.Name) seed=$seed exit=$LASTEXITCODE"
                }
            }

            $itemAfter = Get-Item -LiteralPath $checkpoint
            $hashAfter = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
            $unchanged = (
                $itemAfter.Length -eq $itemBefore.Length -and
                $itemAfter.LastWriteTimeUtc -eq $itemBefore.LastWriteTimeUtc -and
                $hashAfter -eq $hashBefore
            )
            $integrityRows += [ordered]@{
                arm = $arm.Name
                seed = $seed
                checkpoint = $itemAfter.FullName
                before_sha256 = $hashBefore
                after_sha256 = $hashAfter
                unchanged = $unchanged
            }
            if (-not $unchanged) {
                throw "Frozen checkpoint changed: arm=$($arm.Name) seed=$seed"
            }
            Write-Host "===== VALIDATION CAUSAL arm=$($arm.Name) seed=$seed DONE $(Get-Date -Format o) ====="
        }
    }

    New-Item -ItemType Directory -Path $outputBase -Force | Out-Null
    $integrityRows | ConvertTo-Json -Depth 4 | Set-Content `
        -LiteralPath (Join-Path $outputBase "checkpoint_integrity.json") -Encoding utf8
    Write-Host "VALIDATION CAUSAL QUEUE COMPLETE $(Get-Date -Format o)"
}
finally {
    Remove-Item Env:USIM_ACTOR_EVAL_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_SEED -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_COURSE_MATCH_EXCLUDE_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_INFERENCE_STEPS_OVERRIDE -ErrorAction SilentlyContinue
}
