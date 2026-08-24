param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$MinFreeGpuMiB = 9000,
    [int]$PollSeconds = 300,
    [switch]$SkipGpuWait,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\content_delta_pop5\sage_lite_v1\S12_twoexpert_scorefusion_tail0p002_e60_seed2025"
$checkpointRoot = "checkpoints\content_delta_pop5\sage_lite_v1\S12_twoexpert_scorefusion_tail0p002_e60_seed2025"
$splitTag = "strict_item_cold_balanced_thr1_seed_2025"
$outputRootAbs = Join-Path $repoPath $outputRoot
$checkpointRootAbs = Join-Path $repoPath $checkpointRoot
$splitDir = Join-Path $outputRootAbs $splitTag
$finalPath = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
$queueLog = Join-Path $outputRootAbs "mooccube_sage_twoexpert_scorefusion_tail0p002_queue.log"

New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $payload = $line + [Environment]::NewLine
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            [System.IO.File]::AppendAllText($queueLog, $payload, [System.Text.Encoding]::UTF8)
            break
        } catch {
            if ($attempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds (200 * $attempt)
        }
    }
    Write-Host $line
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

Write-QueueLog "CONFIG MOOCCube SAGE two-expert score-fusion tail0p002 seed=2025 | output=$outputRoot | checkpoint=$checkpointRoot | gate_mode=bucket_mlp | buckets=20 | bucket_strategy=paper | gate_hidden=32 | ratio=0.002 | mask=true | only_cold_or_tail=false | candidate_two_expert=false | score_fusion=true | epochs=60"

if ($DryRun) {
    Write-QueueLog "DRYRUN requested; no training started."
    exit 0
}

if (Test-Path -LiteralPath $finalPath) {
    Write-QueueLog "SKIP final exists: $finalPath"
    exit 0
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
            Write-QueueLog "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB; starting."
            break
        }
        Write-QueueLog "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-QueueLog "START MOOCCube SAGE two-expert score-fusion tail0p002 seed=2025"

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
    -SageOnlyColdOrTail $false `
    -SageTailPopRatio 0.002 `
    -SageUseTwoExpert $false `
    -SageTwoExpertScoreFusion $true `
    -SageGateMode bucket_mlp `
    -SageGateBuckets 20 `
    -SageGateHidden 32 `
    -SageGateBucketStrategy paper `
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
    -SaveOptState $true `
    -SkipAggregate

$exitCode = $LASTEXITCODE
Write-QueueLog "END MOOCCube SAGE two-expert score-fusion tail0p002 seed=2025 | exit_code=$exitCode"
exit $exitCode
