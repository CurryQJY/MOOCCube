param(
    [int]$WaitPid = 36124
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "D:\DeskTop\MOOCCube"

$logDir = "outputs\mooccubex\relations_aug_cmin001_e3\strict_item_cold_balanced_thr1_seed_2025"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$watchLog = Join-Path $logDir "wait_then_e35.log"
$transcriptLog = Join-Path $logDir "wait_then_e35_transcript.log"

Start-Transcript -Path $transcriptLog -Append | Out-Null
try {
    function Write-Log {
        param([string]$Message)
        $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
        $line | Tee-Object -FilePath $watchLog -Append
    }

    Write-Log "Watcher started. Waiting for PID=$WaitPid before launching MOOCCubeX augmented E35 resume."

    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-Log "PID $WaitPid is still running: $($proc.ProcessName)."
        Wait-Process -Id $WaitPid
        Write-Log "PID $WaitPid finished."
    } else {
        Write-Log "PID $WaitPid is not running. Launching E35 resume immediately."
    }

    $env:USIM_FB_COURSE_CONCEPT_MIN = "0.01"

    Write-Log "Launching MOOCCubeX augmented E35 resume."

    .\run_usim_feedback_fast3_content_delta_static.ps1 `
      -DataDir "processed_data_hin_x" `
      -RelationDir "MOOCCubeX\relations_aug" `
      -Protocol strict_item_cold_balanced `
      -OutputRoot "outputs\mooccubex\relations_aug_cmin001_e3" `
      -CheckpointRoot "checkpoints\mooccubex\relations_aug_cmin001_e3" `
      -ColdThresholds 1 `
      -Seeds 2025 `
      -Epochs 35 `
      -Patience 10 `
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

    Write-Log "MOOCCubeX augmented E35 resume finished."
} finally {
    Stop-Transcript | Out-Null
}
