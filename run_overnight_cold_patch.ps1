<#
.SYNOPSIS
Overnight launcher for the cold-start patch (Stage 2.1 + 2.3) A/B/C comparison.

.DESCRIPTION
Runs three configs back-to-back, each over the same set of seeds:

  - runA_legacy_baseline    : legacy aux loss + cold_only early stop
  - runB_aux_hot_only       : aux_hot_only=True   + cold_only early stop
  - runC_aux_hot_only_geo   : aux_hot_only=True   + geometric early stop

A single failure inside one config does NOT abort the remaining configs (each
config is wrapped in its own try/catch). After all runs finish, the script
prints per-seed and per-config-averaged item-macro metrics to the console
and writes them to ``<OutputRoot>\summary.csv`` for easy follow-up.

.PARAMETER Seeds
Random seeds. Default ``@(2025, 2026, 2027)``.

.PARAMETER Epochs
Max training epochs per run. Default ``30`` (cold N@10 plateaus around 30
in the e30 / e60 baselines on this dataset).

.PARAMETER Patience
Early-stop patience. Default ``5``.

.PARAMETER UsePseudoColdTrain
Match the e60+pseudo profile by default. Set to ``$false`` to disable.

.PARAMETER IncludeBaseline
Whether to run ``runA_legacy_baseline``. Default ``$true``. Set to ``$false``
to skip A and only run the patched configs (B + C).

.PARAMETER OutputRoot
Where to write outputs. Default
``outputs\content_delta_pop5\cold_patch_2026_05_19_<timestamp>``.

.EXAMPLE
.\run_overnight_cold_patch.ps1
# Runs A + B + C over seeds 2025/2026/2027, ~7-9 hours total.

.EXAMPLE
.\run_overnight_cold_patch.ps1 -IncludeBaseline:$false -Seeds @(2025)
# Quick single-seed B vs C only.
#>
param(
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 30,
    [int]$Patience = 5,
    [bool]$UsePseudoColdTrain = $true,
    [double]$PseudoColdRatio = 0.30,
    [double]$ContentDeltaMaxNorm = 0.05,
    [double]$ContentDeltaScale = 0.25,
    [bool]$IncludeBaseline = $true,
    [string]$OutputRoot = "",
    [string]$CheckpointRoot = ""
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrEmpty($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $OutputRoot = "outputs\content_delta_pop5\cold_patch_2026_05_19_$stamp"
}
if ([string]::IsNullOrEmpty($CheckpointRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $CheckpointRoot = "checkpoints\content_delta_pop5\cold_patch_2026_05_19_$stamp"
}

New-Item -Path $OutputRoot -ItemType Directory -Force | Out-Null
New-Item -Path $CheckpointRoot -ItemType Directory -Force | Out-Null

# Tee the wrapper's own console output to a log file too.
$wrapperLog = Join-Path $OutputRoot "overnight_wrapper.log"
Start-Transcript -Path $wrapperLog -Append | Out-Null

$runs = @()
if ($IncludeBaseline) {
    $runs += [PSCustomObject]@{
        Tag = "runA_legacy_baseline"
        AuxHotOnly = $false
        ScoreMode = "cold_only"
        Description = "Legacy aux + cold_only early stop (re-baselined under the same training profile)."
    }
}
$runs += [PSCustomObject]@{
    Tag = "runB_aux_hot_only"
    AuxHotOnly = $true
    ScoreMode = "cold_only"
    Description = "USIM_AUX_HOT_ONLY=1, legacy early stop. Isolates the aux-loss contribution."
}
$runs += [PSCustomObject]@{
    Tag = "runC_aux_hot_only_geo"
    AuxHotOnly = $true
    ScoreMode = "geometric"
    Description = "USIM_AUX_HOT_ONLY=1 + geometric early stop. Full Stage 2.1+2.3 patch."
}

$wallStart = Get-Date
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host " Overnight cold-start patch A/B/C launcher" -ForegroundColor Cyan
Write-Host " Started at : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host " OutputRoot : $OutputRoot" -ForegroundColor Cyan
Write-Host " Seeds      : $($Seeds -join ', ')" -ForegroundColor Cyan
Write-Host " Configs    : $($runs.Tag -join ', ')" -ForegroundColor Cyan
Write-Host " Epochs/Pat : $Epochs / $Patience" -ForegroundColor Cyan
Write-Host " Pseudo     : $UsePseudoColdTrain (ratio $PseudoColdRatio)" -ForegroundColor Cyan
Write-Host " DeltaCap   : max=$ContentDeltaMaxNorm scale=$ContentDeltaScale" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

$results = @()

foreach ($r in $runs) {
    $tag = $r.Tag
    Write-Host ""
    Write-Host "##### START $tag at $(Get-Date -Format 'HH:mm:ss') #####" -ForegroundColor Cyan
    Write-Host "      $($r.Description)"
    $runStart = Get-Date

    try {
        .\run_usim_feedback_fast3_content_delta_static.ps1 `
            -Protocol strict_item_cold_balanced `
            -ColdThresholds @(1) `
            -Seeds $Seeds `
            -Epochs $Epochs `
            -Patience $Patience `
            -EarlyStopAverageMode item_macro `
            -UsePseudoColdTrain $UsePseudoColdTrain `
            -PseudoColdRatio $PseudoColdRatio `
            -ContentDeltaMaxNorm $ContentDeltaMaxNorm `
            -ContentDeltaScale $ContentDeltaScale `
            -AuxHotOnly $r.AuxHotOnly `
            -EarlyStopScoreMode $r.ScoreMode `
            -RunSampledEval $false `
            -OutputRoot (Join-Path $OutputRoot $tag) `
            -CheckpointRoot (Join-Path $CheckpointRoot $tag) `
            -SkipAggregate
        $elapsed = (Get-Date) - $runStart
        Write-Host ("##### OK    $tag in {0:hh\:mm\:ss} #####" -f $elapsed) -ForegroundColor Green
    }
    catch {
        $elapsed = (Get-Date) - $runStart
        Write-Warning ("##### FAIL  $tag after {0:hh\:mm\:ss}: $_" -f $elapsed)
    }
}

# ---------- Aggregate ----------
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host " Aggregation: per-seed and config-averaged item-macro metrics" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

foreach ($r in $runs) {
    $tag = $r.Tag
    $rows = @()
    foreach ($s in $Seeds) {
        $csv = Join-Path $OutputRoot "$tag\strict_item_cold_balanced_thr1_seed_$s\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if (Test-Path $csv) {
            $row = Import-Csv $csv | Select-Object -First 1
            $rows += [PSCustomObject]@{
                Run = $tag
                Seed = $s
                Cold_R5_macro  = [math]::Round([double]$row.full_cold_item_macro_r5, 4)
                Cold_R10_macro = [math]::Round([double]$row.full_cold_item_macro_r10, 4)
                Cold_N10_macro = [math]::Round([double]$row.full_cold_item_macro_n10, 4)
                Hot_R10_macro  = [math]::Round([double]$row.full_hot_item_macro_r10, 4)
                Hot_N10_macro  = [math]::Round([double]$row.full_hot_item_macro_n10, 4)
                Cold_R10_int   = [math]::Round([double]$row.full_cold_r10, 4)
                Hot_R10_int    = [math]::Round([double]$row.full_hot_r10, 4)
            }
        }
        else {
            Write-Warning "Missing: $csv"
        }
    }
    if ($rows.Count -gt 0) {
        Write-Host ""
        Write-Host "[$tag]" -ForegroundColor Yellow
        $rows | Format-Table -AutoSize
        $avg = [PSCustomObject]@{
            Run = $tag
            Seed = "MEAN"
            Cold_R5_macro  = [math]::Round(($rows | Measure-Object Cold_R5_macro -Average).Average, 4)
            Cold_R10_macro = [math]::Round(($rows | Measure-Object Cold_R10_macro -Average).Average, 4)
            Cold_N10_macro = [math]::Round(($rows | Measure-Object Cold_N10_macro -Average).Average, 4)
            Hot_R10_macro  = [math]::Round(($rows | Measure-Object Hot_R10_macro -Average).Average, 4)
            Hot_N10_macro  = [math]::Round(($rows | Measure-Object Hot_N10_macro -Average).Average, 4)
            Cold_R10_int   = [math]::Round(($rows | Measure-Object Cold_R10_int -Average).Average, 4)
            Hot_R10_int    = [math]::Round(($rows | Measure-Object Hot_R10_int -Average).Average, 4)
        }
        Write-Host ("    MEAN: cold_R10_m={0:F4} cold_N10_m={1:F4} hot_R10_m={2:F4} hot_N10_m={3:F4} (n={4})" -f $avg.Cold_R10_macro, $avg.Cold_N10_macro, $avg.Hot_R10_macro, $avg.Hot_N10_macro, $rows.Count) -ForegroundColor Yellow
        $results += $rows
        $results += $avg
    }
}

# ---------- Reference: dropout baseline ----------
Write-Host ""
Write-Host "[Reference] DropoutNet (from main_table_balanced_itemmacro_selector_v1)" -ForegroundColor Magenta
$dropRows = @()
foreach ($s in $Seeds) {
    $drop = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_$s\main_table_balanced_itemmacro_selector_v1\drop_static_result.json"
    if (Test-Path $drop) {
        $j = Get-Content $drop -Raw | ConvertFrom-Json
        $dropRows += [PSCustomObject]@{
            Run = "DropoutNet"
            Seed = $s
            Cold_R5_macro  = [math]::Round([double]$j[0].full_cold_item_macro.'R@5', 4)
            Cold_R10_macro = [math]::Round([double]$j[0].full_cold_item_macro.'R@10', 4)
            Cold_N10_macro = [math]::Round([double]$j[0].full_cold_item_macro.'N@10', 4)
            Hot_R10_macro  = [math]::Round([double]$j[0].full_hot_item_macro.'R@10', 4)
            Hot_N10_macro  = [math]::Round([double]$j[0].full_hot_item_macro.'N@10', 4)
            Cold_R10_int   = [math]::Round([double]$j[0].full_cold.'R@10', 4)
            Hot_R10_int    = [math]::Round([double]$j[0].full_hot.'R@10', 4)
        }
    }
}
if ($dropRows.Count -gt 0) {
    $dropRows | Format-Table -AutoSize
    $dropAvg = [PSCustomObject]@{
        Run = "DropoutNet"
        Seed = "MEAN"
        Cold_R5_macro  = [math]::Round(($dropRows | Measure-Object Cold_R5_macro -Average).Average, 4)
        Cold_R10_macro = [math]::Round(($dropRows | Measure-Object Cold_R10_macro -Average).Average, 4)
        Cold_N10_macro = [math]::Round(($dropRows | Measure-Object Cold_N10_macro -Average).Average, 4)
        Hot_R10_macro  = [math]::Round(($dropRows | Measure-Object Hot_R10_macro -Average).Average, 4)
        Hot_N10_macro  = [math]::Round(($dropRows | Measure-Object Hot_N10_macro -Average).Average, 4)
        Cold_R10_int   = [math]::Round(($dropRows | Measure-Object Cold_R10_int -Average).Average, 4)
        Hot_R10_int    = [math]::Round(($dropRows | Measure-Object Hot_R10_int -Average).Average, 4)
    }
    Write-Host ("    MEAN: cold_R10_m={0:F4} cold_N10_m={1:F4} hot_R10_m={2:F4} hot_N10_m={3:F4}" -f $dropAvg.Cold_R10_macro, $dropAvg.Cold_N10_macro, $dropAvg.Hot_R10_macro, $dropAvg.Hot_N10_macro) -ForegroundColor Magenta
    $results += $dropRows
    $results += $dropAvg
}

# Persist combined summary CSV next to the runs.
$summaryCsv = Join-Path $OutputRoot "summary.csv"
$results | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "Summary CSV : $summaryCsv" -ForegroundColor Cyan

$totalElapsed = (Get-Date) - $wallStart
Write-Host ""
Write-Host ("Wall-clock total: {0:hh\:mm\:ss}" -f $totalElapsed) -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

Stop-Transcript | Out-Null
