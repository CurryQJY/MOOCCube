param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [string]$OutputRoot = "outputs\junyi\sage_lite_v1\S0_tail0p002_e60_seed2025",
    [string]$CheckpointRoot = "checkpoints\junyi\sage_lite_v1\S0_tail0p002_e60_seed2025",
    [double]$SageTailPopRatio = 0.002,
    [int]$SagePoolTopK = 64,
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

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$outputRootAbs = Resolve-RunPath $Repo $OutputRoot
$checkpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$launchDir = Join-Path $outputRootAbs "_launch"
$launchLog = Join-Path $launchDir "launch.log"
New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

function Write-LaunchLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launchLog -Value $line -Encoding UTF8
    Write-Host $line
}

$splitDir = Join-Path $outputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
$finalPath = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"

Write-LaunchLog "START Junyi SAGE-lite S0 | seed=$Seed | out=$outputRootAbs | ckpt=$checkpointRootAbs"
Write-LaunchLog "Config: mask_known=true mask_same=true sage=true only_cold_or_tail=true tail_ratio=$SageTailPopRatio pool_topk=$SagePoolTopK epochs=60 patience=60"

if ($DryRun) {
    Write-LaunchLog "DRYRUN only; no training started."
    exit 0
}

if (Test-Path -LiteralPath $finalPath) {
    Write-LaunchLog "SKIP existing final result: $finalPath"
    exit 0
}

& $runner `
    -PythonRunner ".\py.bat" `
    -DataDir "processed_data_junyi" `
    -RelationDir "processed_data_junyi\relations" `
    -OutputRoot $outputRootAbs `
    -CheckpointRoot $checkpointRootAbs `
    -Protocol strict_item_cold_balanced `
    -ColdThresholds 1 `
    -Seeds $Seed `
    -Epochs 60 `
    -Patience 60 `
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
    -SageTailPopRatio $SageTailPopRatio `
    -SagePoolTopK $SagePoolTopK `
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

Write-LaunchLog "END Junyi SAGE-lite S0 | seed=$Seed | exit=$exitCode"
exit $exitCode
