$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$name = "weightfix_tail03_shared_warmup30_seed2025"
$outRoot = "outputs\recppo_research_repair\$name"
$ckptRoot = "checkpoints\recppo_research_repair\$name"
$logRoot = "outputs\recppo_research_repair\background_logs"
$outLog = Join-Path $logRoot "$name.out.log"
$errLog = Join-Path $logRoot "$name.err.log"
New-Item -ItemType Directory -Force -Path $outRoot, $ckptRoot, $logRoot | Out-Null
$env:USIM_RECPPO_WARMUP_EPOCHS = "30"
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& .\run_usim_feedback_fast3_content_delta_static.ps1 `
    -PythonRunner ".\py.bat" -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
    -OutputRoot $outRoot -CheckpointRoot $ckptRoot `
    -Protocol strict_item_cold_balanced -ColdThresholds @(1) -Seeds @(2025) `
    -Epochs 31 -Patience 5 -UseContentDelta $false `
    -UsePseudoColdTrain $true -PseudoColdMode batch_tail -PseudoColdRatio 0.3 -PseudoColdMinPop 1 `
    -PpoLossWeight 1.0 -RolloutPolicy ppo -RlResidualScale 0.06 -UsimSteps 5 `
    -UseCourseFeedback $true -UseCourseReward $true -UsePrereqAux $true -UseCourseSample $true `
    -UseUsimRefinedEval $true -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
    1>> $outLog 2>> $errLog
exit $LASTEXITCODE
