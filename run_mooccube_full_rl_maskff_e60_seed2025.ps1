$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$OutputRoot = "outputs\significance_per_item_exports\mooccube\ckg_rl_full_maskff_current_seed2025"
$CheckpointRoot = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_full_maskff_current_seed2025"
$LauncherLog = Join-Path $OutputRoot "launcher.log"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

("[{0}] Start current-code Full RL mask=false/false e60 seed2025" -f (Get-Date -Format o)) |
    Out-File -FilePath $LauncherLog -Encoding UTF8

$params = @{
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    CoursePrereqW = 0.08
    CoursePrereqGate = 0.20
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantConceptGate = 1.0
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseSampleBeta = 0.20
    TrainForceCold = $true
    UsimSteps = 5
    PpoLossWeight = 1.0
    RlResidualScale = 1.0
    RolloutPolicy = "ppo"
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    RunSampledEval = $false
    MaskKnownPosNeg = $false
    MaskSameItemNeg = $false
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
    ("[{0}] Finished current-code Full RL mask=false/false e60 seed2025 exit_code={1}" -f (Get-Date -Format o), $code) |
        Add-Content -Path $LauncherLog
    exit $code
} catch {
    ("[{0}] Failed current-code Full RL mask=false/false e60 seed2025: {1}" -f (Get-Date -Format o), $_.Exception.Message) |
        Add-Content -Path $LauncherLog
    exit 1
}
