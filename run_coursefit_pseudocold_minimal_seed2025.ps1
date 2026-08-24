param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$RunName = "coursefit_pseudocold_minimal_seed2025",
    [switch]$ForceFresh
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\recppo_research_repair\$RunName"
$checkpointRoot = "checkpoints\recppo_research_repair\$RunName"
$seedTag = "strict_item_cold_balanced_thr1_seed_$Seed"
$checkpointDir = Join-Path $checkpointRoot $seedTag
$configLock = Join-Path $checkpointDir "locked_config.json"
$forceFreshValue = [bool]$ForceFresh

$config = [ordered]@{
    seed = $Seed
    source_script = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    source_sha256 = (Get-FileHash -LiteralPath "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    protocol = "strict_item_cold_balanced"
    epochs = $Epochs
    patience = $Patience
    early_stop_metric = "cold_item_macro_n10"
    usim_steps = 5
    rollout_policy = "course_fit"
    ppo_loss_weight = 0.0
    use_course_reward = $false
    use_pseudo_cold_train = $true
    pseudo_cold_mode = "item_tail"
    pseudo_cold_ratio = 0.3
    pseudo_cold_min_pop = 5
    train_force_cold = $true
    course_match_exclude_target = $true
    refinement_scope = "effective_cold_only"
    alignment_gradient_scale = "sum_over_fixed_reference_batch"
}

New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
if (Test-Path -LiteralPath $configLock) {
    $old = Get-Content -LiteralPath $configLock -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 10 -Compress
    $new = $config | ConvertTo-Json -Depth 10 -Compress
    if ($old -ne $new -and -not $forceFreshValue) {
        throw "Locked configuration or source hash changed; refusing resume."
    }
}
$config | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $configLock -Encoding utf8

try {
    $env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = "1"
    Write-Host "===== COURSE-FIT PSEUDO-COLD MINIMAL REPAIR START seed=$Seed $(Get-Date -Format o) ====="
    & $runner `
        -ScriptPath "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py" `
        -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
        -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($Seed) `
        -Epochs $Epochs -Patience $Patience -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
        -UseContentDelta $false -UsePseudoColdTrain $true -PseudoColdMode "item_tail" -AuxHotOnly $true `
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
        -SaveCkpt $true -AutoResume $true -ForceFresh $forceFreshValue -SaveOptState $true -SkipAggregate
    if ($LASTEXITCODE -ne 0) {
        throw "Minimal pseudo-cold repair training failed with exit code $LASTEXITCODE"
    }
    Write-Host "===== COURSE-FIT PSEUDO-COLD MINIMAL REPAIR DONE seed=$Seed $(Get-Date -Format o) ====="
}
finally {
    Remove-Item Env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET -ErrorAction SilentlyContinue
}
