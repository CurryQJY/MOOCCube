param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 40,
    [int]$Patience = 6,
    [double]$PseudoColdRatio = 0.30,
    [int]$BatchSize = 2048,
    [int]$UsimSteps = 5,
    [string]$TeacherCheckpointDir = "checkpoints\recovery_validation\main_table_51ea12fc_candidate\strict_item_cold_balanced_thr1_seed_2025",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($Seed -ne 2025) {
    throw "This isolated launcher is pinned to seed 2025. Create a separate launcher for another seed."
}
if ($Epochs -lt 1 -or $Patience -lt 1 -or $BatchSize -lt 1 -or $UsimSteps -lt 1) {
    throw "Epochs, Patience, BatchSize, and UsimSteps must be positive."
}
if ($PseudoColdRatio -le 0.0 -or $PseudoColdRatio -gt 1.0) {
    throw "PseudoColdRatio must be in (0, 1]."
}

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$staticRunner = Join-Path $repoPath "run_usim_feedback_fast3_content_delta_static.ps1"
$runName = "seed$Seed"
$outputRelative = Join-Path "outputs\usim_original_v2" $runName
$checkpointRelative = Join-Path "checkpoints\usim_original_v2" $runName
$outputRoot = Join-Path $repoPath $outputRelative
$checkpointRoot = Join-Path $repoPath $checkpointRelative
$teacherCheckpointPath = if ([System.IO.Path]::IsPathRooted($TeacherCheckpointDir)) {
    [System.IO.Path]::GetFullPath($TeacherCheckpointDir)
} else {
    Join-Path $repoPath $TeacherCheckpointDir
}

if (-not $DryRun -and ((Test-Path -LiteralPath $outputRoot) -or (Test-Path -LiteralPath $checkpointRoot))) {
    throw "Refusing to overwrite an existing USIM V2 run. Use a new isolated run root instead."
}
if (-not (Test-Path -LiteralPath (Join-Path $teacherCheckpointPath "finished.pt")) -and
    -not (Test-Path -LiteralPath (Join-Path $teacherCheckpointPath "latest.pt"))) {
    throw "TeacherCheckpointDir must contain finished.pt or latest.pt: $teacherCheckpointPath"
}

# These are not parameters of the shared static runner.  Preserve and restore
# them so this launcher remains isolated even when called from a reused shell.
$lockedEnvironment = @{
    "USIM_ORIGINAL_V2" = "1"
    "USIM_ORIGINAL_V2_STEP_PENALTY" = "0.01"
    "USIM_ORIGINAL_V2_STEP_SIZE" = "0.05"
    "USIM_PPO_ADV_NORM" = "1"
    "USIM_ACTOR_INFERENCE_SEED" = "7001"
    "USIM_FB_REWARD_DUP_W" = "0"
    "USIM_CKG_RL_V1" = "0"
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
        ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
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
        UseContentDelta = $false
        UsePseudoColdTrain = $true
        PseudoColdMode = "item_tail"
        PseudoColdRatio = $PseudoColdRatio
        PseudoColdMinPop = 1
        TrainForceCold = $true
        AuxWeight = 0.3
        AuxHotOnly = $true
        UsePaac = $false
        UseCourseFeedback = $false
        UseCourseReward = $false
        UseCourseSample = $false
        UsePrereqAux = $false
        UseCourseRerank = $false
        UseSageLite = $false
        UseSageAuxLoss = $false
        UseCgrcRecon = $false
        UseSgUrinit = $false
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        UsimSteps = $UsimSteps
        PpoEpochs = 2
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
        Write-Host ("USIM V2 contract: teacher={0} pseudo_mode=item_tail pseudo_ratio={1} transition_step={2}" -f $teacherCheckpointPath, $PseudoColdRatio, $lockedEnvironment["USIM_ORIGINAL_V2_STEP_SIZE"])
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
