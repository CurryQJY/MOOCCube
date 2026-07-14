$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$stageBase = "checkpoints\recppo_research_repair\final_candidate_w050_res004_seeds2026_2027"
$root = "outputs\recppo_research_repair\recppo_hparam_3seed_completion"
$ckptRoot = "checkpoints\recppo_research_repair\recppo_hparam_3seed_completion"
$logRoot = Join-Path $root "logs"
$summary = Join-Path $root "hparam_completion_summary.csv"
New-Item -ItemType Directory -Force -Path $root,$ckptRoot,$logRoot | Out-Null
$configs = @(
 [pscustomobject]@{Tag="w025_r004";Weight=0.25;Residual=0.04;Sweep="weight"},
 [pscustomobject]@{Tag="w100_r004";Weight=1.00;Residual=0.04;Sweep="weight"},
 [pscustomobject]@{Tag="w150_r004";Weight=1.50;Residual=0.04;Sweep="weight"},
 [pscustomobject]@{Tag="w050_r006";Weight=0.50;Residual=0.06;Sweep="residual"},
 [pscustomobject]@{Tag="w050_r008";Weight=0.50;Residual=0.08;Sweep="residual"},
 [pscustomobject]@{Tag="w050_r010";Weight=0.50;Residual=0.10;Sweep="residual"}
)
$env:USIM_RECPPO_WARMUP_EPOCHS="30"
$env:USIM_RECPPO_EARLY_STOP_MODE="recppo_stage_guarded"
$env:USIM_RECPPO_GUARD_HOT_RATIO="0.90"
$env:USIM_RECPPO_STRICT_DETERMINISM="1"
$env:PYTHONHASHSEED="0"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
function Update-Summary {
 $rows=@()
 foreach($seed in @(2026,2027)){
  foreach($c in $configs){
   $csv=Join-Path $root "$($c.Tag)\strict_item_cold_balanced_thr1_seed_$seed\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
   if(Test-Path $csv){$d=Import-Csv $csv|Select -First 1;$rows += [pscustomobject]@{seed=$seed;tag=$c.Tag;sweep=$c.Sweep;ppo_weight=$c.Weight;residual=$c.Residual;cold_r5=$d.full_cold_item_macro_r5;cold_r10=$d.full_cold_item_macro_r10;cold_n5=$d.full_cold_item_macro_n5;cold_n10=$d.full_cold_item_macro_n10;hot_r5=$d.full_hot_item_macro_r5;hot_r10=$d.full_hot_item_macro_r10;hot_n5=$d.full_hot_item_macro_n5;hot_n10=$d.full_hot_item_macro_n10}}
  }
 }
 if($rows.Count){$rows|Export-Csv -NoTypeInformation -Encoding UTF8 $summary}
}
foreach($seed in @(2026,2027)){
 $stage=Join-Path $stageBase "strict_item_cold_balanced_thr1_seed_$seed\warmup_stage.pt"
 if(-not(Test-Path $stage)){throw "Missing warmup stage: $stage"}
 $env:USIM_FB_WARMUP_STAGE_CKPT=(Resolve-Path $stage)
 foreach($c in $configs){
  $out=Join-Path $root $c.Tag;$ck=Join-Path $ckptRoot $c.Tag
  $final=Join-Path $out "strict_item_cold_balanced_thr1_seed_$seed\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
  if(Test-Path $final){"SKIP seed=$seed $($c.Tag)"|Add-Content (Join-Path $logRoot 'queue.log');continue}
  New-Item -ItemType Directory -Force -Path $out,$ck|Out-Null
  "[$(Get-Date -Format o)] START seed=$seed $($c.Tag)"|Add-Content (Join-Path $logRoot 'queue.log')
  & .\run_usim_feedback_fast3_content_delta_static.ps1 `
   -PythonRunner ".\py.bat" -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
   -OutputRoot $out -CheckpointRoot $ck -Protocol strict_item_cold_balanced `
   -ColdThresholds @(1) -Seeds @($seed) -Epochs 35 -Patience 5 `
   -UseContentDelta $false -UsePseudoColdTrain $true -PseudoColdMode batch_tail `
   -PseudoColdRatio 0.3 -PseudoColdMinPop 1 -PpoLossWeight $c.Weight `
   -RolloutPolicy ppo -RlResidualScale $c.Residual -UsimSteps 5 `
   -UseCourseFeedback $true -UseCourseReward $true -UsePrereqAux $true -UseCourseSample $true `
   -UseUsimRefinedEval $true -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
   1>> (Join-Path $logRoot "seed${seed}_$($c.Tag).out.log") 2>> (Join-Path $logRoot "seed${seed}_$($c.Tag).err.log")
  "[$(Get-Date -Format o)] EXIT $LASTEXITCODE seed=$seed $($c.Tag)"|Add-Content (Join-Path $logRoot 'queue.log')
  Update-Summary
 }
}
Update-Summary
"[$(Get-Date -Format o)] QUEUE COMPLETE"|Add-Content (Join-Path $logRoot 'queue.log')
