$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$warmName = "weightfix_tail03_shared_warmup30_seed2025"
$warmDir = "checkpoints\recppo_research_repair\$warmName\strict_item_cold_balanced_thr1_seed_2025"
$stage = Join-Path $warmDir "warmup_stage.pt"
$warmFinished = Join-Path $warmDir "finished.pt"
$root = "outputs\recppo_research_repair\weightfix_stage1_seed2025"
$ckptRoot = "checkpoints\recppo_research_repair\weightfix_stage1_seed2025"
$logRoot = Join-Path $root "logs"
$summary = Join-Path $root "stage1_summary.csv"
New-Item -ItemType Directory -Force -Path $root, $ckptRoot, $logRoot | Out-Null
while (-not ((Test-Path $stage) -and (Test-Path $warmFinished))) {
    "[$(Get-Date -Format o)] waiting for shared warmup" | Add-Content (Join-Path $logRoot "queue.log")
    Start-Sleep -Seconds 60
}
$env:USIM_RECPPO_WARMUP_EPOCHS = "30"
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:USIM_RECPPO_EARLY_STOP_MIN_DELTA = "0.0005"
$env:USIM_RECPPO_GUARD_HOT_RATIO = "0.90"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"
$env:USIM_FB_WARMUP_STAGE_CKPT = (Resolve-Path $stage)
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$runs = @(
    [pscustomobject]@{Tag="baseline_res0"; Residual=0.0; Weight=1.0},
    [pscustomobject]@{Tag="res006_w025"; Residual=0.06; Weight=0.25},
    [pscustomobject]@{Tag="res006_w050"; Residual=0.06; Weight=0.50},
    [pscustomobject]@{Tag="res006_w100"; Residual=0.06; Weight=1.00},
    [pscustomobject]@{Tag="res006_w150"; Residual=0.06; Weight=1.50}
)
function Update-Summary {
    $rows=@()
    foreach($r in $runs){
        $csv=Join-Path $root "$($r.Tag)\strict_item_cold_balanced_thr1_seed_2025\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if(-not(Test-Path $csv)){continue}
        $d=Import-Csv $csv|Select-Object -First 1
        $rows += [pscustomobject]@{tag=$r.Tag;residual=$r.Residual;ppo_weight=$r.Weight;cold_r5=$d.full_cold_item_macro_r5;cold_r10=$d.full_cold_item_macro_r10;cold_n5=$d.full_cold_item_macro_n5;cold_n10=$d.full_cold_item_macro_n10;hot_r5=$d.full_hot_item_macro_r5;hot_r10=$d.full_hot_item_macro_r10;hot_n5=$d.full_hot_item_macro_n5;hot_n10=$d.full_hot_item_macro_n10}
    }
    if($rows.Count){$rows|Export-Csv -NoTypeInformation -Encoding UTF8 $summary}
}
foreach($r in $runs){
    $out=Join-Path $root $r.Tag; $ck=Join-Path $ckptRoot $r.Tag
    $final=Join-Path $out "strict_item_cold_balanced_thr1_seed_2025\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if(Test-Path $final){"SKIP $($r.Tag)"|Add-Content (Join-Path $logRoot "queue.log");continue}
    New-Item -ItemType Directory -Force -Path $out,$ck|Out-Null
    "[$(Get-Date -Format o)] START $($r.Tag)"|Add-Content (Join-Path $logRoot "queue.log")
    & .\run_usim_feedback_fast3_content_delta_static.ps1 `
        -PythonRunner ".\py.bat" -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
        -OutputRoot $out -CheckpointRoot $ck -Protocol strict_item_cold_balanced `
        -ColdThresholds @(1) -Seeds @(2025) -Epochs 35 -Patience 5 `
        -UseContentDelta $false -UsePseudoColdTrain $true -PseudoColdMode batch_tail `
        -PseudoColdRatio 0.3 -PseudoColdMinPop 1 -PpoLossWeight $r.Weight `
        -RolloutPolicy ppo -RlResidualScale $r.Residual -UsimSteps 5 `
        -UseCourseFeedback $true -UseCourseReward $true -UsePrereqAux $true -UseCourseSample $true `
        -UseUsimRefinedEval $true -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
        1>> (Join-Path $logRoot "$($r.Tag).out.log") 2>> (Join-Path $logRoot "$($r.Tag).err.log")
    "[$(Get-Date -Format o)] EXIT $LASTEXITCODE $($r.Tag)"|Add-Content (Join-Path $logRoot "queue.log")
    Update-Summary
}
Update-Summary
"[$(Get-Date -Format o)] QUEUE COMPLETE"|Add-Content (Join-Path $logRoot "queue.log")
