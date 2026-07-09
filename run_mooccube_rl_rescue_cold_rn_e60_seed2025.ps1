$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$OutputRoot = "outputs\rl_rescue\mooccube\warm_residual_0p10_cold_rn_e60_seed2025"
$CheckpointRoot = "checkpoints\rl_rescue\mooccube\warm_residual_0p10_cold_rn_e60_seed2025"
$LauncherLog = Join-Path $OutputRoot "launcher.log"
$SeedTag = "strict_item_cold_balanced_thr1_seed_2025"
$InitCheckpointDir = "checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer\$SeedTag"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

("[{0}] Start MOOCCube RL rescue cold_rn e60 seed2025" -f (Get-Date -Format o)) |
    Out-File -FilePath $LauncherLog -Encoding UTF8
("[{0}] InitCheckpointDir={1}" -f (Get-Date -Format o), $InitCheckpointDir) |
    Add-Content -Path $LauncherLog

$params = @{
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_rn"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    TrainForceCold = $true
    UsimSteps = 5
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    MaskKnownPosNeg = $false
    MaskSameItemNeg = $false
    InitCheckpointDir = $InitCheckpointDir
    RlResidualScale = 0.10
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $false
    SaveOptState = $true
}

try {
    & .\run_usim_feedback_fast3_content_delta_static.ps1 @params *>&1 |
        Tee-Object -FilePath $LauncherLog -Append
    $code = $LASTEXITCODE
    ("[{0}] MOOCCube RL rescue cold_rn e60 seed2025 finished with exit code {1}" -f (Get-Date -Format o), $code) |
        Add-Content -Path $LauncherLog
    exit $code
} catch {
    ("[{0}] MOOCCube RL rescue cold_rn e60 seed2025 failed: {1}" -f (Get-Date -Format o), $_.Exception.Message) |
        Add-Content -Path $LauncherLog
    exit 1
}
