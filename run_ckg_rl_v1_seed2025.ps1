param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 40,
    [int]$Patience = 6,
    [int]$PseudoColdCount = 0,
    [string]$RunName = "seed2025",
    [int]$BatchSize = 2048,
    [int]$ReferenceBatchSize = 2048,
    [int]$UsimSteps = 5,
    [ValidateSet("cold_ndcg_then_recall_with_retention", "cold_ndcg_running_retention")]
    [string]$V1SelectorMode = "cold_ndcg_then_recall_with_retention",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$staticRunner = Join-Path $repoPath "run_usim_feedback_fast3_content_delta_static.ps1"
$runOutputRelative = Join-Path "outputs\ckg_rl_v1" $RunName
$runCheckpointRelative = Join-Path "checkpoints\ckg_rl_v1" $RunName
$outputRunRoot = Join-Path $repoPath $runOutputRelative
$checkpointRunRoot = Join-Path $repoPath $runCheckpointRelative

if ($Seed -ne 2025) {
    throw "This isolated launcher is pinned to seed 2025. Create a separate launcher for another seed."
}
if ([string]::IsNullOrWhiteSpace($RunName) -or $RunName -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]*$") {
    throw "RunName must use only letters, digits, underscores, and hyphens."
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be positive."
}
if ($UsimSteps -lt 0) {
    throw "UsimSteps must be non-negative."
}
if (-not $DryRun -and ((Test-Path -LiteralPath $outputRunRoot) -or (Test-Path -LiteralPath $checkpointRunRoot))) {
    throw "Refusing to overwrite an existing CKG-RL V1 run. Use a new isolated run root instead."
}

# These variables are not parameters of the legacy static runner. Set and
# restore them here so a shared shell cannot alter the isolated V1 protocol.
$lockedEnvironment = @{
    "USIM_LEGACY_TRAIN_PROTOCOL" = "0"
    "USIM_STATIC_SPLIT_DIR" = ""
    "USIM_N_CANDIDATES" = "20"
    "USIM_RETRIEVE_TOP_M" = "256"
    "USIM_CANDIDATE_STRATEGY" = "retrieve_sample"
    "USIM_CANDIDATE_TEMP" = "0.20"
    "USIM_CANDIDATE_EPSILON" = "0.10"
    "USIM_BATCH_SIZE" = [string]$BatchSize
    "USIM_CKG_RL_V1" = "1"
    "USIM_USE_EPOCH_EARLY_STOP" = "1"
    "USIM_V1_REFERENCE_BATCH_SIZE" = [string]$ReferenceBatchSize
    "USIM_V1_TARGET_HISTORY_EXCLUSION" = "1"
    "USIM_V1_TARGET_HISTORY_EXCLUSION_SCOPE" = "all_course_terms"
    "USIM_V1_PSEUDO_COLD_PLAN_STRATEGY" = "popularity_stratified"
    "USIM_V1_SELECTOR_HOT_TOL" = "0.003"
    "USIM_V1_SELECTOR_OVERALL_TOL" = "0.003"
    "USIM_V1_SELECTOR_MODE" = $V1SelectorMode
    "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET" = "1"
    "USIM_FB_REWARD_DUP_W" = "0"
    "USIM_PPO_EPOCHS" = "1"
    "USIM_PPO_CLIP" = "0.20"
    "USIM_PPO_GAMMA" = "0.90"
    "USIM_PPO_VALUE_COEFF" = "0.50"
    "USIM_PPO_ENTROPY_COEFF" = "0.01"
    "USIM_FB_ENTRY_SCRIPT" = ""
    "USIM_FORCE_CPU" = "0"
    "USIM_VALIDATION_ONLY" = "0"
    "USIM_CONCEPT_OVERLAP_MODE" = "plain"
    "USIM_PREREQ_CONCEPT_SCORE_THR" = "0.10"
    "USIM_PREREQ_CONCEPT_MIN_HITS" = "1"
    "USIM_PREREQ_CONCEPT_FILE" = "prerequisite-dependency.json"
    "USIM_FB_COURSE_WARM_SEEN" = "5"
    "USIM_FB_COURSE_CONCEPT_MIN" = "0.12"
    "USIM_FB_COURSE_REDUNDANT_THR" = "0.70"
    "USIM_FB_COURSE_STRUCT_VIDEO_MIN" = "0.60"
    "USIM_FB_PREREQ_WEIGHTED_EDGES" = "0"
    "USIM_FB_PREREQ_SOFT_PENALTY" = "0"
}
$originalEnvironment = @{}
foreach ($name in $lockedEnvironment.Keys) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

Push-Location -LiteralPath $repoPath
try {
    foreach ($pair in $lockedEnvironment.GetEnumerator()) {
        Set-Item "Env:$($pair.Key)" ([string]$pair.Value)
    }

    $params = @{
        PythonRunner = ".\py.bat"
        ScriptPath = "usim_feedback_fast3_content_delta.py"
        DataDir = "processed_data_hin_clean_pop5"
        RelationDir = "MOOCCube/relations"
        OutputRoot = $runOutputRelative
        CheckpointRoot = $runCheckpointRelative
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
        Epochs = $Epochs
        Patience = $Patience
        EarlyStopAverageMode = "item_macro"
        UseContentDelta = $false
        UsePseudoColdTrain = $true
        PseudoColdMode = "fixed_item_stratified"
        PseudoColdRatio = 0.0
        PseudoColdMinPop = 5
        TrainForceCold = $false
        AuxWeight = 0.3
        AuxHotOnly = $true
        UsePaac = $false
        UseCourseFeedback = $true
        UseCourseReward = $false
        UsePrereqAux = $false
        PrereqAuxOnlyCold = $true
        PrereqGraphSource = "concept"
        CoursePrereqW = 0.08
        CoursePrereqGate = 0.20
        CourseConceptW = 0.04
        CourseMatchMode = "mean"
        CourseMatchTopK = 5
        CourseDiffW = 0.03
        CourseRedundantW = 0.02
        CourseRedundantMode = "concept"
        CourseFeedbackOnlyCold = $true
        UseCourseSample = $true
        CourseSampleOnlyCold = $true
        CourseSampleBeta = 0.20
        UseCourseRerank = $false
        UseStructuredHardNeg = $false
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        UseSageLite = $false
        UseSageAuxLoss = $false
        UseCgrcRecon = $false
        UseSgUrinit = $false
        UsimSteps = $UsimSteps
        PpoEpochs = 1
        PpoLossWeight = 0.0
        RlResidualScale = 1.0
        RolloutPolicy = "course_fit"
        SimulatorTargetMode = "initial_state"
        DeterministicEvalCandidates = $true
        DeterministicEvalSeed = $Seed
        EvalReuseItemBank = $true
        CkgRlV1 = $true
        V1PseudoColdPlanCount = $PseudoColdCount
        V1PseudoColdPlanSeed = $Seed
        V1PseudoColdPlanStrategy = "popularity_stratified"
        V1ReferenceBatchSize = $ReferenceBatchSize
        V1TargetHistoryExclusion = $true
        V1TargetHistoryExclusionScope = "all_course_terms"
        V1SelectorHotTolerance = 0.003
        V1SelectorOverallTolerance = 0.003
        V1SelectorMode = $V1SelectorMode
        RewardDupW = 0.0
        CourseMatchExcludeTarget = "1"
        RunSampledEval = $false
        SaveCkpt = $true
        AutoResume = $false
        ForceFresh = $true
        SaveOptState = $true
        SkipAggregate = $true
    }
    if ($DryRun) {
        $params["DryRun"] = $true
    }

    & $staticRunner @params
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $lockedEnvironment.Keys) {
        if ($null -eq $originalEnvironment[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $originalEnvironment[$name]
        }
    }
    Pop-Location
}
