param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$MinFreeGpuMiB = 9000,
    [int]$PollSeconds = 300,
    [switch]$SkipGpuWait
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\content_delta_pop5\sage_lite_v1\S10_tail0p002_twoexpert_no_pseudo_e60_seed2025"
$checkpointRoot = "checkpoints\content_delta_pop5\sage_lite_v1\S10_tail0p002_twoexpert_no_pseudo_e60_seed2025"
$outputRootAbs = Join-Path $repoPath $outputRoot
$checkpointRootAbs = Join-Path $repoPath $checkpointRoot
$queueLog = Join-Path $outputRootAbs "s10_tail0p002_twoexpert_no_pseudo_queue.log"

New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            return [int](([string]$raw).Trim().Split("`n")[0].Trim())
        }
    } catch {
        return $null
    }
    return $null
}

if ($SkipGpuWait) {
    Write-QueueLog "SkipGpuWait requested; starting immediately."
} else {
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -eq $free) {
            Write-QueueLog "GPU free memory unavailable; wait ${PollSeconds}s."
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($free -ge $MinFreeGpuMiB) {
            Write-QueueLog "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB; starting S10."
            break
        }
        Write-QueueLog "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-QueueLog "START S10_tail0p002_twoexpert_no_pseudo seed=2025: output=$outputRoot checkpoint=$checkpointRoot"

& .\run_usim_feedback_fast3_content_delta_static.ps1 `
  -DataDir "processed_data_hin_clean_pop5" `
  -RelationDir "MOOCCube/relations" `
  -OutputRoot $outputRoot `
  -CheckpointRoot $checkpointRoot `
  -Protocol strict_item_cold_balanced `
  -ColdThresholds 1 `
  -Seeds 2025 `
  -Epochs 60 `
  -Patience 60 `
  -EarlyStopAverageMode item_macro `
  -EarlyStopScoreMode cold_only `
  -UseContentDelta $false `
  -UsePseudoColdTrain $false `
  -UsePaac $false `
  -UseCourseFeedback $true `
  -UseCourseReward $true `
  -UseCourseSample $true `
  -UsePrereqAux $true `
  -PrereqGraphSource concept `
  -CoursePrereqW 0.08 `
  -CourseConceptW 0.04 `
  -CourseDiffW 0.03 `
  -CourseRedundantW 0.02 `
  -CourseRedundantMode concept `
  -CourseTermNorm none `
  -CourseFeedbackOnlyCold $false `
  -CourseSampleOnlyCold $false `
  -PrereqAuxOnlyCold $false `
  -CourseSampleBeta 0.20 `
  -UseSageLite $true `
  -SageOnlyColdOrTail $true `
  -SageTailPopRatio 0.002 `
  -SageUseTwoExpert $true `
  -SageGateMin 0.10 `
  -SageGateMax 0.60 `
  -SagePoolTopK 64 `
  -SageCourseTemp 0.20 `
  -UseSageAuxLoss $false `
  -UseCourseRerank $false `
  -UseStructuredHardNeg $false `
  -MaskKnownPosNeg $true `
  -MaskSameItemNeg $true `
  -RunSampledEval $false `
  -SaveCkpt $true `
  -AutoResume $true `
  -ForceFresh $false `
  -SaveOptState $true

$exitCode = $LASTEXITCODE
Write-QueueLog "END S10_tail0p002_twoexpert_no_pseudo seed=2025: exit_code=$exitCode"
exit $exitCode
