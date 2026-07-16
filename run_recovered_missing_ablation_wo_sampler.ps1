param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\recppo_research_repair\original_main_missing_ablation\wo_knowledge_sampler"
$checkpointRoot = "checkpoints\recppo_research_repair\original_main_missing_ablation\wo_knowledge_sampler"
$common = @{
    ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    DataDir = "processed_data_hin_clean_pop5"; RelationDir = "MOOCCube/relations"
    Protocol = "strict_item_cold_balanced"; ColdThresholds = @(1); Seeds = $Seeds
    Epochs = 60; Patience = 60; EarlyStopAverageMode = "item_macro"; EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false; UsePseudoColdTrain = $false; PseudoColdMode = "batch_random"
    PseudoColdRatio = 0.3; PseudoColdMinPop = 5; UsePaac = $false
    CoursePrereqW = 0.08; CoursePrereqGate = 0.20; CourseConceptW = 0.04; CourseDiffW = 0.03
    CourseRedundantW = 0.02; CourseRedundantConceptGate = 1.0; CourseRedundantMode = "concept"
    CourseTermNorm = "none"; CourseSampleBeta = 0.20; TrainForceCold = $true; UsimSteps = 5
    PpoLossWeight = 1.0; RolloutPolicy = "ppo"; RlResidualScale = 1.0
    UseCourseFeedback = $true; UseCourseReward = $true; UseCourseSample = $false; UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false; CourseSampleOnlyCold = $false; PrereqAuxOnlyCold = $false
    UseUsimRefinedEval = $true; UseCourseRerank = $false; UseStructuredHardNeg = $false
    UseSageLite = $false; UseSageAuxLoss = $false; UseCgrcRecon = $false; UseSgUrinit = $false
    MaskKnownPosNeg = $true; MaskSameItemNeg = $true; RunSampledEval = $false
    OutputRoot = $outputRoot; CheckpointRoot = $checkpointRoot
    SaveCkpt = $true; AutoResume = $true; ForceFresh = $false; SaveOptState = $true; SkipAggregate = $true
}

if ($DryRun) {
    Write-Host "DRYRUN train w/o knowledge sampler seeds=$($Seeds -join ',')"
} else {
    Write-Host "[$(Get-Date -Format o)] START training w/o knowledge sampler"
    & $runner @common
    if ($LASTEXITCODE -ne 0) { throw "w/o knowledge sampler training failed" }
    Write-Host "[$(Get-Date -Format o)] DONE training w/o knowledge sampler"
}

try {
    $env:USIM_ACTOR_INFERENCE_MODE = "actor"
    $env:USIM_ACTOR_INFERENCE_SEED = "7001"
    $env:USIM_INFERENCE_STEPS_OVERRIDE = "5"
    foreach ($target in @("validation", "test")) {
        $env:USIM_ACTOR_EVAL_TARGET = $target
        foreach ($seed in $Seeds) {
            $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
            $finished = Join-Path $checkpointRoot "$seedTag\finished.pt"
            if ((-not $DryRun) -and (-not (Test-Path -LiteralPath $finished))) { throw "Missing checkpoint: $finished" }
            $evalRoot = Join-Path $outputRoot "actor_eval\$target"
            $armDir = Join-Path $evalRoot $seedTag
            $finalCsv = Join-Path $armDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
            $audit = Join-Path $armDir "actor_inference_audit.json"
            if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $audit)) { continue }
            Write-Host "[$(Get-Date -Format o)] START actor replay target=$target seed=$seed"
            if ($DryRun) { continue }
            $params = @{}; foreach ($key in $common.Keys) { $params[$key] = $common[$key] }
            $params.ScriptPath = "main_checkpoint_actor_inference_ab.py"
            $params.Seeds = @($seed); $params.OutputRoot = $evalRoot
            & $runner @params
            if ($LASTEXITCODE -ne 0) { throw "Actor replay failed target=$target seed=$seed" }
            Write-Host "[$(Get-Date -Format o)] DONE actor replay target=$target seed=$seed"
        }
    }
}
finally {
    foreach ($name in @("USIM_ACTOR_INFERENCE_MODE", "USIM_ACTOR_INFERENCE_SEED", "USIM_INFERENCE_STEPS_OVERRIDE", "USIM_ACTOR_EVAL_TARGET")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

