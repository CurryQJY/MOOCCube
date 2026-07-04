param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_coco",
    [string]$RelationDir = "processed_data_coco\relations",
    [string]$OutputRoot = "outputs\coco\single_seed_triage",
    [string]$CheckpointRoot = "checkpoints\coco\single_seed_triage",
    [int[]]$Seeds = @(2025),
    [int]$OursEpochs = 30,
    [int]$OursPatience = 10,
    [int]$BaselineEpochs = 10,
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [int]$OursBatchSize = 2048,
    [int]$BaselineBatchSize = 1024,
    [string[]]$Models = @("Popularity", "ContentProfile", "BPR", "LightGCN", "DropoutNet", "GAR", "CCFCRec", "LightGCL"),
    [switch]$UseGpuBaselines,
    [switch]$AllowConcurrent,
    [switch]$RerunCompleted,
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

function Join-Models([string[]]$Names) {
    $flat = @()
    foreach ($entry in $Names) {
        foreach ($name in ([string]$entry -split ",")) {
            $trimmed = $name.Trim()
            if ($trimmed) {
                $flat += $trimmed
            }
        }
    }
    return $flat
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
$OursScript = Join-Path $Repo "run_xes3g5m_ours_sota_serial.ps1"
$BaselineScript = Join-Path $Repo "run_xes3g5m_lightweight_baselines.ps1"
$ModelNames = Join-Models $Models

if ($DryRun) {
    Write-Host "Dataset=COCO"
    Write-Host "DataDir=$DataDir"
    Write-Host "RelationDir=$RelationDir"
    Write-Host "OutputRoot=$OutputRoot"
    Write-Host "CheckpointRoot=$CheckpointRoot"
    Write-Host "Seeds=$($Seeds -join ',')"
    Write-Host "PrereqGraphSource=behavior"
    Write-Host "Protocol=strict_item_cold_balanced"
    Write-Host "ColdThreshold=$ColdThreshold"
    Write-Host "EvalNeg=$EvalNeg"
    Write-Host "OursEpochs=$OursEpochs"
    Write-Host "OursPatience=$OursPatience"
    Write-Host "BaselineEpochs=$BaselineEpochs"
    Write-Host "Models=$($ModelNames -join ',')"
    Write-Host "OursRunner=$OursScript"
    Write-Host "BaselineRunner=$BaselineScript"
    Write-Host "Plan=run_xes3g5m_ours_sota_serial.ps1 -SkipNoCourse -SkipContentProfile -SkipCgrc -SkipAggregate -PrereqGraphSource behavior"
    Write-Host "Plan=run_xes3g5m_lightweight_baselines.ps1 -PrereqGraphSource behavior -Models $($ModelNames -join ',')"
    return
}

Set-Location $Repo

foreach ($path in @($OursScript, $BaselineScript)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required runner: $path"
    }
}

foreach ($path in @($DataDir, $RelationDir)) {
    $abs = Resolve-RunPath $Repo $path
    if (-not (Test-Path -LiteralPath $abs)) {
        throw "Missing required COCO processed path: $abs"
    }
}

$common = @{
    Repo = $Repo
    PythonRunner = $PythonRunner
    DataDir = $DataDir
    RelationDir = $RelationDir
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    Seeds = $Seeds
    ColdThreshold = $ColdThreshold
    EvalNeg = $EvalNeg
}

$oursArgs = @{
    Repo = $Repo
    PythonRunner = $PythonRunner
    DataDir = $DataDir
    RelationDir = $RelationDir
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    Seeds = $Seeds
    Epochs = $OursEpochs
    Patience = $OursPatience
    ColdThreshold = $ColdThreshold
    EvalNeg = $EvalNeg
    OursBatchSize = $OursBatchSize
    ContentProfileBatchSize = $BaselineBatchSize
    PrereqGraphSource = "behavior"
    UseContentDelta = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    ForceFresh = (-not $RerunCompleted.IsPresent)
    AutoResume = $RerunCompleted.IsPresent
    SkipNoCourse = $true
    SkipContentProfile = $true
    SkipCgrc = $true
    SkipAggregate = $true
}
if ($AllowConcurrent) {
    $oursArgs["AllowConcurrent"] = $true
}
if ($RerunCompleted) {
    $oursArgs["RerunCompleted"] = $true
}

Write-Host "[COCO] START Ours triage"
& $OursScript @oursArgs
if ($LASTEXITCODE -ne 0) {
    throw "COCO Ours triage failed with exit=$LASTEXITCODE"
}
Write-Host "[COCO] END Ours triage"

$baselineArgs = @{
    Repo = $Repo
    PythonRunner = $PythonRunner
    DataDir = $DataDir
    RelationDir = $RelationDir
    PrereqGraphSource = "behavior"
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    Seeds = $Seeds
    ColdThreshold = $ColdThreshold
    EvalNeg = $EvalNeg
    Models = $ModelNames
    PopBatchSize = $BaselineBatchSize
    ContentProfileBatchSize = $BaselineBatchSize
    BprEpochs = $BaselineEpochs
    BprEvalInterval = $BaselineEpochs
    BprBatchSize = 4096
    LightGCNEpochs = $BaselineEpochs
    LightGCNEvalInterval = $BaselineEpochs
    LightGCNBatchSize = 2048
    DropoutEpochs = $BaselineEpochs
    DropoutEvalInterval = $BaselineEpochs
    DropoutBatchSize = $BaselineBatchSize
    GarEpochs = $BaselineEpochs
    GarEvalInterval = $BaselineEpochs
    GarBatchSize = $BaselineBatchSize
    CCFCEpochs = $BaselineEpochs
    CCFCEvalInterval = $BaselineEpochs
    CCFCBatchSize = $BaselineBatchSize
    LightGCLEpochs = $BaselineEpochs
    LightGCLEvalInterval = $BaselineEpochs
    LightGCLBatchSize = $BaselineBatchSize
}
if ($UseGpuBaselines) {
    $baselineArgs["UseGpu"] = $true
}
if ($AllowConcurrent) {
    $baselineArgs["AllowConcurrent"] = $true
}
if ($RerunCompleted) {
    $baselineArgs["RerunCompleted"] = $true
}

Write-Host "[COCO] START lightweight baselines"
& $BaselineScript @baselineArgs
if ($LASTEXITCODE -ne 0) {
    throw "COCO lightweight baseline queue failed with exit=$LASTEXITCODE"
}
Write-Host "[COCO] END lightweight baselines"

Write-Host ""
Write-Host "COCO triage outputs:"
Write-Host ("  Table: {0}" -f (Join-Path (Resolve-RunPath $Repo $OutputRoot) "main_table_compare\main_table_item_macro_paper_narrow.csv"))
Write-Host ("  Queue: {0}" -f (Join-Path (Resolve-RunPath $Repo $OutputRoot) "_queue"))
