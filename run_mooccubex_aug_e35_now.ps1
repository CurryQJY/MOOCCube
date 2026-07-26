$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "D:\DeskTop\MOOCCube"

$logDir = "outputs\mooccubex\relations_aug_cmin001_e3\strict_item_cold_balanced_thr1_seed_2025"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$transcriptLog = Join-Path $logDir "run_e35_now_transcript.log"

Start-Transcript -Path $transcriptLog -Append | Out-Null
try {
    $env:USIM_FB_COURSE_CONCEPT_MIN = "0.01"

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
} finally {
    Stop-Transcript | Out-Null
}
