param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 40,
    [int]$Patience = 6,
    [double]$PseudoColdRatio = 0.30,
    [int]$BatchSize = 2048,
    [int]$UsimSteps = 5,
    [int]$Candidates = 20,
    [string]$TeacherCheckpointDir = "checkpoints\recovery_validation\main_table_51ea12fc_candidate\strict_item_cold_balanced_thr1_seed_2025",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($Seed -ne 2025) {
    throw "This V3.1 acceptance launcher is pinned to seed 2025. Create a separate launcher for another seed."
}
if ($Epochs -lt 1 -or $Patience -lt 1 -or $BatchSize -lt 1 -or $UsimSteps -lt 1 -or $Candidates -lt 1) {
    throw "Epochs, Patience, BatchSize, UsimSteps, and Candidates must be positive."
}
if ($PseudoColdRatio -le 0.0 -or $PseudoColdRatio -gt 1.0) {
    throw "PseudoColdRatio must be in (0, 1]."
}

function Get-V31CandidateQuota([int]$Count) {
    $weights = @(0.30, 0.30, 0.30, 0.10)
    $quotas = @()
    foreach ($weight in $weights) {
        $quotas += [int][math]::Floor($Count * $weight)
    }
    for ($index = 0; $index -lt ($Count - ($quotas | Measure-Object -Sum).Sum); $index++) {
        $quotas[$index % $quotas.Count]++
    }
    return "$($quotas[0])/$($quotas[1])/$($quotas[2])/$($quotas[3])"
}

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$staticRunner = Join-Path $repoPath "run_usim_feedback_fast3_content_delta_static.ps1"
$runName = "seed$Seed"
$outputRelative = Join-Path "outputs\ckg_rl_usim_v31" $runName
$checkpointRelative = Join-Path "checkpoints\ckg_rl_usim_v31" $runName
$outputRoot = Join-Path $repoPath $outputRelative
$checkpointRoot = Join-Path $repoPath $checkpointRelative
$teacherCheckpointPath = if ([System.IO.Path]::IsPathRooted($TeacherCheckpointDir)) {
    [System.IO.Path]::GetFullPath($TeacherCheckpointDir)
} else {
    Join-Path $repoPath $TeacherCheckpointDir
}
$quota = "6/6/6/2" # Default quota=6/6/6/2 for 20 candidates.
if ($Candidates -ne 20) {
    $quota = Get-V31CandidateQuota $Candidates
}

if (-not $DryRun -and ((Test-Path -LiteralPath $outputRoot) -or (Test-Path -LiteralPath $checkpointRoot))) {
    throw "Refusing to overwrite an existing V3.1 run. Use a new isolated run root instead."
}
if (-not (Test-Path -LiteralPath (Join-Path $teacherCheckpointPath "finished.pt")) -and
    -not (Test-Path -LiteralPath (Join-Path $teacherCheckpointPath "latest.pt"))) {
    throw "TeacherCheckpointDir must contain finished.pt or latest.pt: $teacherCheckpointPath"
}

# The base static runner supplies data, split, course artifacts, and full ranking.
# The V3 wrapper now uses quota-preserving train candidates and policy-visible,
# observable CKG fit while preserving target-free inference.
$lockedEnvironment = @{
    "USIM_ORIGINAL_V2" = "1"
    "USIM_V3_CORE" = "1"
    "USIM_V3_ENGINE_REVISION" = "v3.1"
    "USIM_V3_STEP_SIZE" = "0.05"
    "USIM_V3_STEP_PENALTY" = "0.01"
    "USIM_V3_CANDIDATES" = [string]$Candidates
    "USIM_V3_RETRIEVAL_CHUNK" = "8192"
    "USIM_V3_REPLAY_CAPACITY" = "4096"
    "USIM_V3_REPLAY_BATCH_SIZE" = "512"
    "USIM_V3_GAMMA" = "0.99"
    "USIM_V3_PPO_CLIP" = "0.20"
    "USIM_V3_VALUE_WEIGHT" = "0.50"
    "USIM_V3_TERMINAL_VALUE_WEIGHT" = "1.0"
    "USIM_V3_ENTROPY_WEIGHT" = "0.01"
    "USIM_BATCH_SIZE" = [string]$BatchSize
    "USIM_ACTOR_INFERENCE_SEED" = "7003"
    "USIM_CKG_RL_V1" = "0"
    "USIM_FB_REWARD_DUP_W" = "0"
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
        ScriptPath = "ckg_rl_usim_v3.py"
        DataDir = "processed_data_hin_clean_pop5"
        RelationDir = "MOOCCube/relations"
        OutputRoot = $outputRelative
        CheckpointRoot = $checkpointRelative
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
        Epochs = $Epochs
        Patience = $Patience
        EarlyStopAverageMode = "item_macro"
        EarlyStopScoreMode = "cold_only"
        UseContentDelta = $false
        UsePseudoColdTrain = $true
        PseudoColdMode = "item_tail"
        PseudoColdRatio = $PseudoColdRatio
        PseudoColdMinPop = 1
        TrainForceCold = $true
        AuxWeight = 0.3
        AuxHotOnly = $true
        UsePaac = $false
        UseCourseFeedback = $true
        UseCourseReward = $true
        UseCourseSample = $true
        CourseSampleOnlyCold = $true
        CourseSampleBeta = 0.20
        CoursePrereqW = 0.08
        CourseConceptW = 0.04
        CourseDiffW = 0.03
        CourseRedundantW = 0.02
        UsePrereqAux = $true
        PrereqAuxOnlyCold = $true
        PrereqGraphSource = "concept"
        UseCourseRerank = $false
        UseSageLite = $false
        UseSageAuxLoss = $false
        UseCgrcRecon = $false
        UseSgUrinit = $false
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        UsimSteps = $UsimSteps
        PpoEpochs = 1
        PpoLossWeight = 1.0
        RlResidualScale = 1.0
        RolloutPolicy = "ppo"
        RewardDupW = 0.0
        InitCheckpointDir = $teacherCheckpointPath
        UseUsimRefinedEval = $true
        CkgRlV1 = $false
        RunSampledEval = $false
        SaveCkpt = $true
        AutoResume = $false
        ForceFresh = $true
        SaveOptState = $true
        SkipAggregate = $true
    }
    if ($DryRun) {
        $contractFormat = "CKG-RL V3.1 contract: teacher={0} pseudo_mode=item_tail pseudo_ratio={1} " +
            "step={2} candidates={3} quota={4} ckg_logit_bias=on ckg_reward=on prereq_aux=on"
        Write-Host ($contractFormat -f $teacherCheckpointPath, $PseudoColdRatio, $lockedEnvironment["USIM_V3_STEP_SIZE"], $Candidates, $quota)
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
