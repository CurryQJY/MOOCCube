param(
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$OutputRoot = "outputs\significance_per_item_exports\mooccube\ckg_rl_full_sg_urinit_K32_lw0p7_gw0p3",
    [string]$CheckpointRoot = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_full_sg_urinit_K32_lw0p7_gw0p3"
)

$ErrorActionPreference = "Stop"

$started = Get-Date
Write-Host "SG-URInit main-table 3-seed static run started: $started" -ForegroundColor Cyan
Write-Host "Reference config: outputs\significance_per_item_exports\mooccube\ckg_rl_full" -ForegroundColor Cyan
Write-Host "Seeds: $($Seeds -join ',')" -ForegroundColor Cyan
Write-Host "OutputRoot: $OutputRoot" -ForegroundColor Cyan

$runnerParams = @{
    PythonRunner = $PythonRunner
    Protocol = "strict_item_cold_balanced"
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    ColdThresholds = @(1)
    Seeds = $Seeds
    Epochs = $Epochs
    Patience = $Patience
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UseSgUrinit = $true
    SgUrinitClusterK = 32
    SgUrinitLocalW = 0.70
    SgUrinitGlobalW = 0.30
    SgUrinitTargetNorm = 0.0
    SgUrinitMaxIter = 20
    UsePseudoColdTrain = $false
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    UseCourseRerank = $false
    UseSageLite = $false
    UseSageAuxLoss = $false
    UseCgrcRecon = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    TrainForceCold = $true
    UsimSteps = 5
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    RunSampledEval = $false
    SaveCkpt = $true
    SaveOptState = $true
    ForceFresh = $true
}

& $StaticRunner @runnerParams
if ($LASTEXITCODE -ne 0) {
    throw "SG-URInit main-table 3-seed static run failed"
}

$finished = Get-Date
Write-Host ""
Write-Host "SG-URInit main-table 3-seed static run finished: $finished" -ForegroundColor Green
Write-Host ("Elapsed: {0:n1} minutes" -f (($finished - $started).TotalMinutes)) -ForegroundColor Green
