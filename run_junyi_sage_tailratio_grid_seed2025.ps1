param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [double[]]$Ratios = @(0.01, 0.02),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [int]$SagePoolTopK = 64,
    [ValidateSet("heuristic", "bucket_mlp")]
    [string]$SageGateMode = "heuristic",
    [int]$SageGateBuckets = 20,
    [int]$SageGateHidden = 32,
    [ValidateSet("paper", "log")]
    [string]$SageGateBucketStrategy = "paper",
    [switch]$SageTwoExpertScoreFusion,
    [string]$OutputRoot = "outputs\junyi\sage_lite_v1\S1_tailratio_grid_seed2025",
    [string]$CheckpointRoot = "checkpoints\junyi\sage_lite_v1\S1_tailratio_grid_seed2025",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $Base $Path)
}

function Format-RatioName([double]$Ratio) {
    return ("r{0}" -f ([string]$Ratio).Replace(".", "p"))
}

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $script:QueueLog -Value $line -Encoding UTF8
    Write-Host $line
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputRootAbs = Resolve-RunPath $Repo $OutputRoot
$checkpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$queueDir = Join-Path $outputRootAbs "_queue"
$script:QueueLog = Join-Path $queueDir "tailratio_grid_queue.log"
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

Write-QueueLog "QUEUE START Junyi SAGE tail-ratio grid | seed=$Seed | ratios=$($Ratios -join ',') | epochs=$Epochs | gate_mode=$SageGateMode | buckets=$SageGateBuckets | bucket_strategy=$SageGateBucketStrategy | gate_hidden=$SageGateHidden | score_fusion=$([bool]$SageTwoExpertScoreFusion)"

foreach ($ratio in $Ratios) {
    $ratioName = Format-RatioName $ratio
    $caseOut = Join-Path $outputRootAbs $ratioName
    $caseCkpt = Join-Path $checkpointRootAbs $ratioName
    $splitDir = Join-Path $caseOut ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
    $finalPath = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    New-Item -ItemType Directory -Force -Path $caseOut | Out-Null
    New-Item -ItemType Directory -Force -Path $caseCkpt | Out-Null

    if (Test-Path -LiteralPath $finalPath) {
        Write-QueueLog "SKIP ratio=$ratio | final exists=$finalPath"
        continue
    }

    Write-QueueLog "START ratio=$ratio | out=$caseOut | ckpt=$caseCkpt"

    if ($DryRun) {
        Write-QueueLog "DRYRUN ratio=$ratio | gate_mode=$SageGateMode | buckets=$SageGateBuckets | bucket_strategy=$SageGateBucketStrategy | gate_hidden=$SageGateHidden | score_fusion=$([bool]$SageTwoExpertScoreFusion) | no training started"
        continue
    }

    & $runner `
        -PythonRunner ".\py.bat" `
        -DataDir "processed_data_junyi" `
        -RelationDir "processed_data_junyi\relations" `
        -OutputRoot $caseOut `
        -CheckpointRoot $caseCkpt `
        -Protocol strict_item_cold_balanced `
        -ColdThresholds 1 `
        -Seeds $Seed `
        -Epochs $Epochs `
        -Patience $Patience `
        -EarlyStopAverageMode item_macro `
        -UseContentDelta:$false `
        -UsePseudoColdTrain:$false `
        -UsePaac:$false `
        -CourseFeedbackOnlyCold:$true `
        -CourseSampleOnlyCold:$true `
        -PrereqAuxOnlyCold:$true `
        -RunSampledEval:$false `
        -UseSageLite:$true `
        -SageOnlyColdOrTail:$true `
        -SageTailPopRatio $ratio `
        -SagePoolTopK $SagePoolTopK `
        -SageGateMode $SageGateMode `
        -SageGateBuckets $SageGateBuckets `
        -SageGateHidden $SageGateHidden `
        -SageGateBucketStrategy $SageGateBucketStrategy `
        -SageTwoExpertScoreFusion:$SageTwoExpertScoreFusion `
        -SageUseTwoExpert:$false `
        -UseSageAuxLoss:$false `
        -MaskKnownPosNeg:$true `
        -MaskSameItemNeg:$true `
        -SaveCkpt:$true `
        -AutoResume:$false `
        -ForceFresh:$true `
        -SaveOptState:$true `
        -SkipAggregate

    $exitCode = 0
    if ($null -ne $LASTEXITCODE) {
        $exitCode = $LASTEXITCODE
    }
    Write-QueueLog "END ratio=$ratio | exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "Junyi SAGE tail-ratio grid failed for ratio=$ratio with exit=$exitCode"
    }
}

Write-QueueLog "QUEUE DONE Junyi SAGE tail-ratio grid"
