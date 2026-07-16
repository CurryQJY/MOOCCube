param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [string[]]$Targets = @("validation", "test"),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$datasets = @(
    [pscustomobject]@{
        Name = "junyi"; DataDir = "processed_data_junyi"; RelationDir = "processed_data_junyi\relations"
        CheckpointRoot = "checkpoints\junyi\main_table_3seed\ours"
        OutputRoot = "outputs\recppo_research_repair\cross_dataset_actor_inference_ab\junyi"
        Epochs = 60; Patience = 60; CourseOnlyCold = $true; SampleOnlyCold = $true; PrereqOnlyCold = $true
    },
    [pscustomobject]@{
        Name = "coco"; DataDir = "processed_data_coco"; RelationDir = "processed_data_coco\relations"
        CheckpointRoot = "checkpoints\coco\single_seed_triage\ours_full"
        OutputRoot = "outputs\recppo_research_repair\cross_dataset_actor_inference_ab\coco"
        Epochs = 30; Patience = 10; CourseOnlyCold = $false; SampleOnlyCold = $false; PrereqOnlyCold = $false
    }
)

try {
    $env:USIM_INFERENCE_STEPS_OVERRIDE = "5"
    foreach ($dataset in $datasets) {
        foreach ($target in $Targets) {
            $env:USIM_ACTOR_EVAL_TARGET = $target
            foreach ($seed in $Seeds) {
                $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
                $finished = Join-Path $dataset.CheckpointRoot "$seedTag\finished.pt"
                if (-not (Test-Path -LiteralPath $finished)) { throw "Missing checkpoint: $finished" }
                foreach ($mode in @("static", "actor")) {
                    $modeRoot = Join-Path $dataset.OutputRoot "$target\$mode"
                    $armDir = Join-Path $modeRoot $seedTag
                    $finalCsv = Join-Path $armDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
                    $audit = Join-Path $armDir "actor_inference_audit.json"
                    if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $audit)) {
                        Write-Host "SKIP completed dataset=$($dataset.Name) target=$target seed=$seed mode=$mode"
                        continue
                    }
                    Write-Host "[$(Get-Date -Format o)] START dataset=$($dataset.Name) target=$target seed=$seed mode=$mode"
                    if ($DryRun) { continue }
                    $env:USIM_ACTOR_INFERENCE_MODE = $mode
                    $env:USIM_ACTOR_INFERENCE_SEED = "7001"
                    & $runner `
                        -ScriptPath "main_checkpoint_actor_inference_ab.py" `
                        -DataDir $dataset.DataDir -RelationDir $dataset.RelationDir `
                        -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
                        -Epochs $dataset.Epochs -Patience $dataset.Patience `
                        -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
                        -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
                        -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
                        -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
                        -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
                        -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
                        -PpoLossWeight 1.0 -RolloutPolicy "ppo" -RlResidualScale 1.0 `
                        -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
                        -CourseFeedbackOnlyCold $dataset.CourseOnlyCold -CourseSampleOnlyCold $dataset.SampleOnlyCold `
                        -PrereqAuxOnlyCold $dataset.PrereqOnlyCold -UseUsimRefinedEval $true `
                        -UseCourseRerank $false -UseStructuredHardNeg $false -UseSageLite $false `
                        -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
                        -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
                        -OutputRoot $modeRoot -CheckpointRoot $dataset.CheckpointRoot `
                        -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate
                    if ($LASTEXITCODE -ne 0) { throw "Cross-dataset evaluation failed" }
                    Write-Host "[$(Get-Date -Format o)] DONE dataset=$($dataset.Name) target=$target seed=$seed mode=$mode"
                }
            }
        }
    }
}
finally {
    foreach ($name in @("USIM_INFERENCE_STEPS_OVERRIDE", "USIM_ACTOR_EVAL_TARGET", "USIM_ACTOR_INFERENCE_MODE", "USIM_ACTOR_INFERENCE_SEED")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

