$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$stage = "checkpoints\recppo_research_repair\weightfix_tail03_shared_warmup30_seed2025\strict_item_cold_balanced_thr1_seed_2025\warmup_stage.pt"
$root = "outputs\recppo_research_repair\recppo_residual_stage2_w050_seed2025"
$ckptRoot = "checkpoints\recppo_research_repair\recppo_residual_stage2_w050_seed2025"
$logRoot = Join-Path $root "logs"
$summary = Join-Path $root "residual_summary.csv"
New-Item -ItemType Directory -Force -Path $root,$ckptRoot,$logRoot | Out-Null
if(-not(Test-Path $stage)){throw "Missing warmup stage: $stage"}
$env:USIM_RECPPO_WARMUP_EPOCHS="30"
$env:USIM_RECPPO_EARLY_STOP_MODE="recppo_stage_guarded"
$env:USIM_RECPPO_EARLY_STOP_MIN_DELTA="0.0005"
$env:USIM_RECPPO_GUARD_HOT_RATIO="0.90"
$env:USIM_RECPPO_STRICT_DETERMINISM="1"
$env:USIM_FB_WARMUP_STAGE_CKPT=(Resolve-Path $stage)
$env:PYTHONHASHSEED="0"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$runs=@(
 [pscustomobject]@{Tag="res004_w050";Residual=0.04},
 [pscustomobject]@{Tag="res008_w050";Residual=0.08},
 [pscustomobject]@{Tag="res010_w050";Residual=0.10}
)
function Update-Summary {
 $rows=@()
 foreach($r in $runs){
  $csv=Join-Path $root "$($r.Tag)\strict_item_cold_balanced_thr1_seed_2025\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
  if(Test-Path $csv){$d=Import-Csv $csv|Select -First 1;$rows += [pscustomobject]@{tag=$r.Tag;residual=$r.Residual;ppo_weight=0.5;cold_r5=$d.full_cold_item_macro_r5;cold_r10=$d.full_cold_item_macro_r10;cold_n5=$d.full_cold_item_macro_n5;cold_n10=$d.full_cold_item_macro_n10;hot_r5=$d.full_hot_item_macro_r5;hot_r10=$d.full_hot_item_macro_r10;hot_n5=$d.full_hot_item_macro_n5;hot_n10=$d.full_hot_item_macro_n10}}
 }
 if($rows.Count){$rows|Export-Csv -NoTypeInformation -Encoding UTF8 $summary}
}
foreach($r in $runs){
 $out=Join-Path $root $r.Tag;$ck=Join-Path $ckptRoot $r.Tag
 $final=Join-Path $out "strict_item_cold_balanced_thr1_seed_2025\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
 if(Test-Path $final){"SKIP $($r.Tag)"|Add-Content (Join-Path $logRoot 'queue.log');continue}
 New-Item -ItemType Directory -Force -Path $out,$ck|Out-Null
 "[$(Get-Date -Format o)] START $($r.Tag)"|Add-Content (Join-Path $logRoot 'queue.log')
 & .\run_usim_feedback_fast3_content_delta_static.ps1 `
  -PythonRunner ".\py.bat" -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
  -OutputRoot $out -CheckpointRoot $ck -Protocol strict_item_cold_balanced `
  -ColdThresholds @(1) -Seeds @(2025) -Epochs 35 -Patience 5 `
  -UseContentDelta $false -UsePseudoColdTrain $true -PseudoColdMode batch_tail `
  -PseudoColdRatio 0.3 -PseudoColdMinPop 1 -PpoLossWeight 0.5 `
  -RolloutPolicy ppo -RlResidualScale $r.Residual -UsimSteps 5 `
  -UseCourseFeedback $true -UseCourseReward $true -UsePrereqAux $true -UseCourseSample $true `
  -UseUsimRefinedEval $true -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
  1>> (Join-Path $logRoot "$($r.Tag).out.log") 2>> (Join-Path $logRoot "$($r.Tag).err.log")
 "[$(Get-Date -Format o)] EXIT $LASTEXITCODE $($r.Tag)"|Add-Content (Join-Path $logRoot 'queue.log')
 Update-Summary
}
Update-Summary
"[$(Get-Date -Format o)] QUEUE COMPLETE"|Add-Content (Join-Path $logRoot 'queue.log')
