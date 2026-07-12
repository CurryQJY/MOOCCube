$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$warmName = "tail03_shared_warmup30_seed2025"
$stage = Join-Path $PSScriptRoot "checkpoints\recppo_research_repair\$warmName\strict_item_cold_balanced_thr1_seed_2025\warmup_stage.pt"
$gridRoot = "outputs\recppo_research_repair\tail03_nightly_grid_seed2025"
$ckptRoot = "checkpoints\recppo_research_repair\tail03_nightly_grid_seed2025"
$logRoot = Join-Path $gridRoot "logs"
$summaryPath = Join-Path $gridRoot "grid_summary.csv"
New-Item -ItemType Directory -Force -Path $gridRoot, $ckptRoot, $logRoot | Out-Null

while (-not (Test-Path -LiteralPath $stage)) {
    "[$(Get-Date -Format o)] Waiting for $stage" | Add-Content (Join-Path $logRoot "queue.log")
    Start-Sleep -Seconds 60
}

$env:USIM_RECPPO_WARMUP_EPOCHS = "30"
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:USIM_RECPPO_EARLY_STOP_MIN_DELTA = "0.0005"
$env:USIM_RECPPO_GUARD_HOT_RATIO = "0.90"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"
$env:USIM_FB_WARMUP_STAGE_CKPT = $stage
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$runs = @()
foreach ($residual in @(0.01, 0.02, 0.03, 0.04)) {
    foreach ($weight in @(0.10, 0.25, 0.50)) {
        $runs += [pscustomobject]@{ Residual=$residual; Weight=$weight; Tier="core" }
    }
}
foreach ($residual in @(0.05, 0.06)) {
    foreach ($weight in @(0.25, 0.50)) {
        $runs += [pscustomobject]@{ Residual=$residual; Weight=$weight; Tier="strong" }
    }
}

function Tag([double]$value) {
    return $value.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
}

function Write-Summary {
    $rows = @()
    foreach ($run in $runs) {
        $tag = "res$(Tag $run.Residual)_w$(Tag $run.Weight)"
        $dir = Join-Path $gridRoot "$tag\strict_item_cold_balanced_thr1_seed_2025"
        $csv = Join-Path $dir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if (-not (Test-Path $csv)) { continue }
        $d = Import-Csv -LiteralPath $csv | Select-Object -First 1
        $mainPass = (
            [double]$d.full_cold_item_macro_r5 -ge 0.22567311106966423 -and
            [double]$d.full_cold_item_macro_r10 -ge 0.27321624805264144 -and
            [double]$d.full_cold_item_macro_n5 -ge 0.17260326124680725 -and
            [double]$d.full_cold_item_macro_n10 -ge 0.18800730284509654
        )
        $rows += [pscustomobject]@{
            tag=$tag; tier=$run.Tier; residual=$run.Residual; ppo_weight=$run.Weight
            cold_r5=$d.full_cold_item_macro_r5; cold_r10=$d.full_cold_item_macro_r10
            cold_n5=$d.full_cold_item_macro_n5; cold_n10=$d.full_cold_item_macro_n10
            hot_r5=$d.full_hot_item_macro_r5; hot_r10=$d.full_hot_item_macro_r10
            hot_n5=$d.full_hot_item_macro_n5; hot_n10=$d.full_hot_item_macro_n10
            main_seed2025_pass=$mainPass
        }
    }
    if ($rows.Count -gt 0) { $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $summaryPath }
}

foreach ($run in $runs) {
    $tag = "res$(Tag $run.Residual)_w$(Tag $run.Weight)"
    $outRoot = Join-Path $gridRoot $tag
    $runCkpt = Join-Path $ckptRoot $tag
    $final = Join-Path $outRoot "strict_item_cold_balanced_thr1_seed_2025\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $logOut = Join-Path $logRoot "$tag.out.log"
    $logErr = Join-Path $logRoot "$tag.err.log"
    if (Test-Path $final) {
        "[$(Get-Date -Format o)] SKIP completed $tag" | Add-Content (Join-Path $logRoot "queue.log")
        continue
    }
    New-Item -ItemType Directory -Force -Path $outRoot, $runCkpt | Out-Null
    "[$(Get-Date -Format o)] START $tag" | Add-Content (Join-Path $logRoot "queue.log")
    & .\run_usim_feedback_fast3_content_delta_static.ps1 `
        -PythonRunner ".\py.bat" `
        -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
        -OutputRoot $outRoot -CheckpointRoot $runCkpt `
        -Protocol strict_item_cold_balanced -ColdThresholds @(1) -Seeds @(2025) `
        -Epochs 35 -Patience 5 `
        -UseContentDelta $false -UsePseudoColdTrain $true `
        -PseudoColdMode batch_tail -PseudoColdRatio 0.3 -PseudoColdMinPop 1 `
        -PpoLossWeight $run.Weight -RolloutPolicy ppo -RlResidualScale $run.Residual `
        -UsimSteps 5 -UseCourseFeedback $true -UseCourseReward $true `
        -UsePrereqAux $true -UseCourseSample $true -UseUsimRefinedEval $true `
        -SaveCkpt $true -ForceFresh $false -AutoResume $true -SaveOptState $true `
        1>> $logOut 2>> $logErr
    $code = $LASTEXITCODE
    "[$(Get-Date -Format o)] EXIT $code $tag" | Add-Content (Join-Path $logRoot "queue.log")
    Write-Summary
}

Write-Summary
"[$(Get-Date -Format o)] QUEUE COMPLETE" | Add-Content (Join-Path $logRoot "queue.log")
