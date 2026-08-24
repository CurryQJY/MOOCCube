param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [bool]$ForceFresh = $false
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\recppo_research_repair\coursefit_rollout_train5_seed2025"
$checkpointRoot = "checkpoints\recppo_research_repair\coursefit_rollout_train5_seed2025"
$seedTag = "strict_item_cold_balanced_thr1_seed_$Seed"
$checkpointDir = Join-Path $checkpointRoot $seedTag
$configLock = Join-Path $checkpointDir "coursefit_rollout_config.json"

$config = [ordered]@{
    seed = $Seed
    protocol = "strict_item_cold_balanced"
    epochs = 60
    patience = 60
    early_stop_average_mode = "item_macro"
    early_stop_score_mode = "cold_only"
    usim_steps = 5
    rollout_policy = "course_fit"
    ppo_loss_weight = 0.0
    use_course_reward = $false
    use_course_feedback = $true
    use_course_sample = $true
    use_prereq_aux = $true
    use_content_delta = $false
    use_pseudo_cold_train = $false
    train_force_cold = $true
    course_sample_beta = 0.20
    mask_known_pos_neg = $true
    mask_same_item_neg = $true
    source_script = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
}

New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
if (Test-Path -LiteralPath $configLock) {
    $old = Get-Content -LiteralPath $configLock -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 10 -Compress
    $new = $config | ConvertTo-Json -Depth 10 -Compress
    if ($old -ne $new -and -not $ForceFresh) {
        throw "Existing course-fit config differs; refusing resume. Use -ForceFresh true only for an intentional clean restart."
    }
}
$config | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $configLock -Encoding utf8

Write-Host "===== COURSE-FIT ROLLOUT TRAINING START seed=$Seed $(Get-Date -Format o) ====="
Write-Host "OutputRoot=$outputRoot"
Write-Host "CheckpointRoot=$checkpointRoot"

& $runner `
    -ScriptPath "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py" `
    -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
    -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($Seed) `
    -Epochs 60 -Patience 60 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
    -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
    -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
    -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
    -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
    -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
    -PpoLossWeight 0.0 -RolloutPolicy "course_fit" -RlResidualScale 1.0 `
    -UseCourseFeedback $true -UseCourseReward $false -UseCourseSample $true -UsePrereqAux $true `
    -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
    -UseUsimRefinedEval $true -UseCourseRerank $false -UseStructuredHardNeg $false `
    -UseSageLite $false -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
    -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
    -OutputRoot $outputRoot -CheckpointRoot $checkpointRoot `
    -SaveCkpt $true -AutoResume $true -ForceFresh $ForceFresh -SaveOptState $true -SkipAggregate

if ($LASTEXITCODE -ne 0) {
    throw "Course-fit rollout training failed with exit code $LASTEXITCODE"
}
Write-Host "===== COURSE-FIT ROLLOUT TRAINING DONE seed=$Seed $(Get-Date -Format o) ====="
