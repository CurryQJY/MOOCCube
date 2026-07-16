param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int[]]$RandomEvalSeeds = @(7001, 7002, 7003, 7004, 7005),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$checkpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_candidate"
$outputRoot = "outputs\recppo_research_repair\test_inference_policy_controls"
$common = @{
    ScriptPath = "main_checkpoint_actor_inference_ab.py"; DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"; Protocol = "strict_item_cold_balanced"; ColdThresholds = @(1)
    Epochs = 60; Patience = 60; EarlyStopAverageMode = "item_macro"; EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false; UsePseudoColdTrain = $false; PseudoColdMode = "batch_random"
    PseudoColdRatio = 0.3; PseudoColdMinPop = 5; UsePaac = $false
    CoursePrereqW = 0.08; CoursePrereqGate = 0.20; CourseConceptW = 0.04; CourseDiffW = 0.03
    CourseRedundantW = 0.02; CourseRedundantConceptGate = 1.0; CourseRedundantMode = "concept"
    CourseTermNorm = "none"; CourseSampleBeta = 0.20; TrainForceCold = $true; UsimSteps = 5
    PpoLossWeight = 1.0; RolloutPolicy = "ppo"; RlResidualScale = 1.0
    UseCourseFeedback = $true; UseCourseReward = $true; UseCourseSample = $true; UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false; CourseSampleOnlyCold = $false; PrereqAuxOnlyCold = $false
    UseUsimRefinedEval = $true; UseCourseRerank = $false; UseStructuredHardNeg = $false
    UseSageLite = $false; UseSageAuxLoss = $false; UseCgrcRecon = $false; UseSgUrinit = $false
    MaskKnownPosNeg = $true; MaskSameItemNeg = $true; RunSampledEval = $false
    CheckpointRoot = $checkpointRoot; SaveCkpt = $true; AutoResume = $true; ForceFresh = $false
    SaveOptState = $true; SkipAggregate = $true
}

function Invoke-Policy([int]$seed, [string]$policy, [int]$evalSeed, [string]$armName) {
    $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
    $finished = Join-Path $checkpointRoot "$seedTag\finished.pt"
    if (-not (Test-Path -LiteralPath $finished)) { throw "Missing checkpoint: $finished" }
    $armRoot = Join-Path $outputRoot $armName
    $armDir = Join-Path $armRoot $seedTag
    $finalCsv = Join-Path $armDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $audit = Join-Path $armDir "actor_inference_audit.json"
    if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $audit)) {
        Write-Host "SKIP completed policy=$policy eval_seed=$evalSeed train_seed=$seed"
        return
    }
    Write-Host "[$(Get-Date -Format o)] START policy=$policy eval_seed=$evalSeed train_seed=$seed"
    if ($DryRun) { return }
    $env:USIM_ACTOR_INFERENCE_MODE = $policy
    $env:USIM_ACTOR_INFERENCE_SEED = "$evalSeed"
    $params = @{}; foreach ($key in $common.Keys) { $params[$key] = $common[$key] }
    $params.Seeds = @($seed); $params.OutputRoot = $armRoot
    & $runner @params
    if ($LASTEXITCODE -ne 0) { throw "Policy evaluation failed" }
    Write-Host "[$(Get-Date -Format o)] DONE policy=$policy eval_seed=$evalSeed train_seed=$seed"
}

try {
    $env:USIM_ACTOR_EVAL_TARGET = "test"
    $env:USIM_INFERENCE_STEPS_OVERRIDE = "5"
    foreach ($seed in $Seeds) {
        foreach ($policy in @("static", "ppo", "greedy_similarity", "course_fit")) {
            Invoke-Policy $seed $policy 7001 $policy
        }
        foreach ($evalSeed in @(7001, 7002, 7003, 7004, 7005)) {
            if ($RandomEvalSeeds -notcontains $evalSeed) { continue }
            Invoke-Policy $seed "random" $evalSeed "random_seed_$evalSeed"
        }
    }
}
finally {
    foreach ($name in @("USIM_ACTOR_EVAL_TARGET", "USIM_ACTOR_INFERENCE_MODE", "USIM_ACTOR_INFERENCE_SEED", "USIM_INFERENCE_STEPS_OVERRIDE")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

