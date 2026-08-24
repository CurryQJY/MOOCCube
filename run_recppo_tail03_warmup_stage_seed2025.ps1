$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$name = "tail03_shared_warmup30_seed2025"
$outRoot = "outputs\recppo_research_repair\$name"
$ckptRoot = "checkpoints\recppo_research_repair\$name"
$logDir = "outputs\recppo_research_repair\background_logs"
$logOut = Join-Path $logDir "$name.out.log"
$logErr = Join-Path $logDir "$name.err.log"
New-Item -ItemType Directory -Force -Path $outRoot, $ckptRoot, $logDir | Out-Null

$env:USIM_RECPPO_WARMUP_EPOCHS = "30"
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$argsMap = @{
    PythonRunner                = ".\py.bat"
    ScriptPath                  = "usim_feedback_fast3_content_delta_repaired.py"
    OutputRoot                  = $outRoot
    CheckpointRoot              = $ckptRoot
    Protocol                    = "strict_item_cold_balanced"
    ColdThresholds              = @(1)
    Seeds                       = @(2025)
    Epochs                      = 31
    Patience                    = 5
    UseContentDelta             = $false
    UsePseudoColdTrain          = $true
    PseudoColdMode              = "batch_tail"
    PseudoColdRatio             = 0.3
    PseudoColdMinPop            = 1
    PpoLossWeight               = 0.1
    RolloutPolicy               = "ppo"
    RlResidualScale             = 0.02
    UsimSteps                   = 5
    UseCourseFeedback           = $true
    UseCourseReward             = $true
    UsePrereqAux                = $true
    UseCourseSample             = $true
    UseUsimRefinedEval          = $true
    SaveCkpt                    = $true
    ForceFresh                  = $false
    AutoResume                  = $true
    SaveOptState                = $true
}

"[$(Get-Date -Format o)] START $name" | Add-Content -LiteralPath $logOut
& .\run_usim_feedback_fast3_content_delta_static.ps1 @argsMap 1>> $logOut 2>> $logErr
$code = $LASTEXITCODE
"[$(Get-Date -Format o)] EXIT $code" | Add-Content -LiteralPath $logOut
exit $code
