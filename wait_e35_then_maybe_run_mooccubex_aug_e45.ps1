param(
    [int]$WaitPid = 28408,
    [double]$BaselineValColdN10 = 0.04423820735250226,
    [double]$MinDelta = 0.0001
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "D:\DeskTop\MOOCCube"

$runDir = "outputs\mooccubex\relations_aug_cmin001_e3\strict_item_cold_balanced_thr1_seed_2025"
$historyCsv = Join-Path $runDir "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"
$summaryCsv = Join-Path $runDir "mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv"
$watchLog = Join-Path $runDir "wait_e35_then_maybe_e45.log"
$transcriptLog = Join-Path $runDir "wait_e35_then_maybe_e45_transcript.log"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Start-Transcript -Path $transcriptLog -Append | Out-Null
try {
    function Write-Log {
        param([string]$Message)
        $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
        $line | Tee-Object -FilePath $watchLog -Append
    }

    function Get-BestValColdN10 {
        if (-not (Test-Path $historyCsv)) {
            throw "History CSV not found: $historyCsv"
        }
        $rows = Import-Csv $historyCsv
        $best = $null
        foreach ($row in $rows) {
            $epoch = [int]$row.Epoch
            $score = [double]$row.'Val_full_cold_N@10'
            if ($null -eq $best -or $score -gt $best.Score) {
                $best = [pscustomobject]@{ Epoch = $epoch; Score = $score }
            }
        }
        if ($null -eq $best) {
            throw "No validation rows found in $historyCsv"
        }
        return $best
    }

    function Get-TestColdMacroN10 {
        if (-not (Test-Path $summaryCsv)) {
            return $null
        }
        $row = Import-Csv $summaryCsv | Where-Object { $_.Eval -eq "full_rank_item_macro" } | Select-Object -First 1
        if ($null -eq $row) {
            return $null
        }
        return [double]$row.'Cold_N@10'
    }

    Write-Log "Watcher started. Waiting for E35 PID=$WaitPid."
    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-Log "PID $WaitPid is still running: $($proc.ProcessName)."
        Wait-Process -Id $WaitPid
        Write-Log "PID $WaitPid finished."
    } else {
        Write-Log "PID $WaitPid is not running. Checking result immediately."
    }

    $best = Get-BestValColdN10
    $testColdN10 = Get-TestColdMacroN10
    Write-Log ("E35 best validation cold item-macro N@10: epoch={0}, score={1:F6}; baseline={2:F6}; test cold item-macro N@10={3}" -f $best.Epoch, $best.Score, $BaselineValColdN10, $testColdN10)

    if ($best.Score -le ($BaselineValColdN10 + $MinDelta)) {
        Write-Log ("No sufficient validation improvement. Stop here. threshold={0:F6}" -f ($BaselineValColdN10 + $MinDelta))
        return
    }

    Write-Log "Validation improved. Launching E45 resume."
    $env:USIM_FB_COURSE_CONCEPT_MIN = "0.01"

    .\run_usim_feedback_fast3_content_delta_static.ps1 `
      -DataDir "processed_data_hin_x" `
      -RelationDir "MOOCCubeX\relations_aug" `
      -Protocol strict_item_cold_balanced `
      -OutputRoot "outputs\mooccubex\relations_aug_cmin001_e3" `
      -CheckpointRoot "checkpoints\mooccubex\relations_aug_cmin001_e3" `
      -ColdThresholds 1 `
      -Seeds 2025 `
      -Epochs 45 `
      -Patience 15 `
      -EarlyStopAverageMode item_macro `
      -RunSampledEval $false `
      -UseContentDelta $false `
      -UsePseudoColdTrain $false `
      -UsePaac $false `
      -CourseFeedbackOnlyCold $false `
      -CourseSampleOnlyCold $false `
      -PrereqAuxOnlyCold $false `
      -SaveCkpt $true `
      -AutoResume $true `
      -ForceFresh $false `
      -SaveOptState $true `
      -SkipAggregate

    Write-Log "E45 resume finished."
} finally {
    Stop-Transcript | Out-Null
}
