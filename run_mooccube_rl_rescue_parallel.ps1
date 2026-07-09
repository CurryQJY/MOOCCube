$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$OutputRoot = "outputs\rl_rescue\mooccube\warm_residual_0p10_e20_task"
$CheckpointRoot = "checkpoints\rl_rescue\mooccube\warm_residual_0p10_e20_task"
$LauncherLog = Join-Path $OutputRoot "task_launcher.log"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

("[{0}] Start MOOCCube RL rescue task" -f (Get-Date -Format o)) |
    Out-File -FilePath $LauncherLog -Encoding UTF8

$params = @{
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 20
    Patience = 20
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
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
    InitCheckpointDir = "checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer\strict_item_cold_balanced_thr1_seed_2025"
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
    ("[{0}] MOOCCube RL rescue task finished with exit code {1}" -f (Get-Date -Format o), $code) |
        Add-Content -Path $LauncherLog
    exit $code
} catch {
    ("[{0}] MOOCCube RL rescue task failed: {1}" -f (Get-Date -Format o), $_.Exception.Message) |
        Add-Content -Path $LauncherLog
    exit 1
}
