$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$waitLog = "outputs\recppo_research_repair\recppo_hparam_3seed_completion\logs\queue.log"
$root = "outputs\recppo_research_repair\recppo_core_ablation_seed2025\wo_course_reward"
$ckpt = "checkpoints\recppo_research_repair\recppo_core_ablation_seed2025\wo_course_reward"
$logRoot = "outputs\recppo_research_repair\recppo_core_ablation_seed2025\logs"
$outLog = Join-Path $logRoot "wo_course_reward.out.log"
$errLog = Join-Path $logRoot "wo_course_reward.err.log"
$queueLog = Join-Path $logRoot "queue.log"
New-Item -ItemType Directory -Force -Path $root,$ckpt,$logRoot | Out-Null
while ($true) {
    if ((Test-Path $waitLog) -and ((Get-Content $waitLog -Tail 3) -match "QUEUE COMPLETE")) { break }
    "[$(Get-Date -Format o)] waiting for hparam queue" | Add-Content $queueLog
    Start-Sleep -Seconds 60
}
$env:USIM_RECPPO_WARMUP_EPOCHS="30"
$env:USIM_RECPPO_EARLY_STOP_MODE="recppo_stage_guarded"
$env:USIM_RECPPO_GUARD_HOT_RATIO="0.90"
$env:USIM_RECPPO_STRICT_DETERMINISM="1"
$env:PYTHONHASHSEED="0"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
"[$(Get-Date -Format o)] START wo_course_reward seed=2025" | Add-Content $queueLog
& .\run_usim_feedback_fast3_content_delta_static.ps1 `
 -PythonRunner ".\py.bat" -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
 -OutputRoot $root -CheckpointRoot $ckpt `
 -Protocol strict_item_cold_balanced -ColdThresholds @(1) -Seeds @(2025) `
 -Epochs 40 -Patience 5 -EarlyStopScoreMode cold_only `
 -UseContentDelta $false -UsePseudoColdTrain $true -PseudoColdMode batch_tail `
 -PseudoColdRatio 0.3 -PseudoColdMinPop 1 -PpoLossWeight 0.5 `
 -RolloutPolicy ppo -RlResidualScale 0.04 -UsimSteps 5 `
 -UseCourseFeedback $true -UseCourseReward $false -UsePrereqAux $true -UseCourseSample $true `
 -UseUsimRefinedEval $true -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
 1>> $outLog 2>> $errLog
"[$(Get-Date -Format o)] EXIT $LASTEXITCODE wo_course_reward seed=2025" | Add-Content $queueLog
exit $LASTEXITCODE
