# Single-seed schedule upgrade: warm/content 30 + total 60 (closer to USIM MLP30 / RL budget).
# ContentDelta OFF, residual 0.3, repaired RecPPO entrypoint.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$baseRunner = Join-Path $PSScriptRoot "run_usim_feedback_fast3_content_delta_static.ps1"
$outRoot = "outputs\recppo_research_repair\warm30_e60_seed2025"
$ckptRoot = "checkpoints\recppo_research_repair\warm30_e60_seed2025"
$logDir = Join-Path $outRoot "logs"
New-Item -ItemType Directory -Force -Path $outRoot, $ckptRoot, $logDir | Out-Null
$logOut = Join-Path $logDir "seed2025.out.log"

# RecPPO schedule: 30 supervised warmup, then PPO until epoch 60 / early-stop.
$env:USIM_RECPPO_WARMUP_EPOCHS = "30"
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$argsMap = @{
    PythonRunner                 = ".\py.bat"
    ScriptPath                   = "usim_feedback_fast3_content_delta_repaired.py"
    OutputRoot                   = $outRoot
    CheckpointRoot               = $ckptRoot
    Protocol                     = "strict_item_cold_balanced"
    ColdThresholds               = @(1)
    Seeds                        = @(2025)
    Epochs                       = 60
    Patience                     = 20
    UseContentDelta              = $false
    ContentDeltaMode             = "embedding"
    ContentDeltaColdOnly         = $true
    ContentDeltaTrainOnIdDropout = $true
    UsePseudoColdTrain           = $true
    PseudoColdMode               = "all_eligible"
    PseudoColdRatio              = 1.0
    PseudoColdMinPop             = 1
    PpoLossWeight                = 1.0
    RolloutPolicy                = "ppo"
    RlResidualScale              = 0.30
    UsimSteps                    = 5
    UseCourseFeedback            = $true
    UseCourseReward              = $true
    UsePrereqAux                 = $true
    UseCourseSample              = $true
    UseUsimRefinedEval           = $true
    SaveCkpt                     = $true
    # Resume by default; restart from scratch only when train config fingerprint changes.
    ForceFresh                   = $false
    AutoResume                   = $true
    SaveOptState                 = $true
}

Write-Host "===== warm30_e60 seed2025 =====" -ForegroundColor Cyan
Write-Host "OutputRoot=$outRoot"
Write-Host "WarmupEpochs=$($env:USIM_RECPPO_WARMUP_EPOCHS) Epochs=60 Patience=20 ContentDelta=OFF"
Write-Host "Log=$logOut"

& $baseRunner @argsMap *>&1 | Tee-Object -FilePath $logOut
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "FAILED exit=$code" -ForegroundColor Red
    if (Test-Path $logOut) { Get-Content $logOut -Tail 40 }
    exit $code
}
Write-Host "DONE warm30_e60 seed2025" -ForegroundColor Green
exit 0
