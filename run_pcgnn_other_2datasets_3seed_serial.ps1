param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [ValidateSet("junyi", "coco")]
    [string[]]$Datasets = @("junyi", "coco"),
    [int[]]$Seeds = @(2025, 2026, 2027),
    [string]$OutputRoot = "paper_aaai27\baseline_sources\_pcgnn_strict",
    [int]$Epochs = 20,
    [int]$EarlyStopPatience = 5,
    [int]$TrainBatchSize = 32,
    [int]$EvalBatchSize = 64,
    [int]$KgBatchSize = 256,
    [double]$KgLossWeight = 1.0,
    [double]$LearningRate = 0.0001,
    [string]$RunId = "",
    [switch]$SkipPrepare,
    [switch]$ForcePrepare,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $Base $Path)
}

function Split-Name([int]$Seed) {
    return "strict_item_cold_balanced_thr1_seed_{0}" -f $Seed
}

function Get-DatasetSpec([string]$Dataset) {
    switch ($Dataset) {
        "junyi" {
            return [pscustomobject]@{
                Dataset = "junyi"
                DataDir = "processed_data_junyi"
                Stream = "processed_data_junyi\stream_data.pkl"
                RelationDir = "processed_data_junyi\relations"
                AtomicDataset = "junyi_strict_full"
                Seed2025Root = "outputs\junyi\mask_ablation\mask_tt"
                OtherSeedRoot = "outputs\junyi\main_table_3seed"
            }
        }
        "coco" {
            return [pscustomobject]@{
                Dataset = "coco"
                DataDir = "processed_data_coco"
                Stream = "processed_data_coco\stream_data.pkl"
                RelationDir = "processed_data_coco\relations"
                AtomicDataset = "coco_strict_full"
                Seed2025Root = "outputs\coco\single_seed_triage\ours_full"
                OtherSeedRoot = "outputs\coco\single_seed_triage\ours_full"
            }
        }
        default { throw "Unsupported dataset: $Dataset" }
    }
}

function Get-SplitDir([pscustomobject]$Spec, [int]$Seed) {
    $root = if ($Seed -eq 2025) { $Spec.Seed2025Root } else { $Spec.OtherSeedRoot }
    return (Resolve-RunPath $Repo (Join-Path $root (Split-Name $Seed)))
}

function Get-AtomicConfig([pscustomobject]$Spec) {
    return (Join-Path $PCGNNRootAbs ("recbole_{0}.yaml" -f $Spec.AtomicDataset))
}

function Get-AtomicDir([pscustomobject]$Spec) {
    return (Join-Path (Join-Path $PCGNNRootAbs "dataset") $Spec.AtomicDataset)
}

function Assert-Path([string]$Path, [string]$Kind) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing ${Kind}: $Path" }
}

function Assert-SplitArtifacts([string]$Dataset, [int]$Seed, [string]$SplitDir) {
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl", "static_split_summary.json")) {
        $path = Join-Path $SplitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split artifact dataset=$Dataset seed=${Seed}: $path"
        }
    }
}

function Test-AtomicReady([pscustomobject]$Spec) {
    $datasetDir = Get-AtomicDir $Spec
    $configPath = Get-AtomicConfig $Spec
    if (-not (Test-Path -LiteralPath $configPath)) { return $false }
    foreach ($suffix in @("train.inter", "valid.inter", "test.inter", "item", "kg", "link")) {
        $path = Join-Path $datasetDir ("{0}.{1}" -f $Spec.AtomicDataset, $suffix)
        if (-not (Test-Path -LiteralPath $path)) { return $false }
    }
    return $true
}

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    if (-not $DryRun) { Add-Content -LiteralPath $MasterLog -Encoding UTF8 -Value $line }
}

function Invoke-Python([string[]]$CommandArgs, [string]$LogPath, [string]$Label) {
    Write-QueueLog ("START {0} | log={1}" -f $Label, $LogPath)
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonRunnerAbs @CommandArgs *>&1 | Out-File -LiteralPath $LogPath -Encoding utf8
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    Write-QueueLog ("END {0} | exit={1}" -f $Label, $exitCode)
    if ($exitCode -ne 0) { throw "$Label failed with exit=$exitCode. See $LogPath" }
}

function Prepare-Atomic([pscustomobject]$Spec) {
    if ((-not $ForcePrepare) -and (Test-AtomicReady $Spec)) {
        Write-QueueLog "SKIP prepare dataset=$($Spec.Dataset) | atomic dataset already exists"
        return
    }
    if ($SkipPrepare) { throw "Atomic dataset missing while -SkipPrepare is set: $($Spec.AtomicDataset)" }

    $seed2025Split = Get-SplitDir $Spec 2025
    $prepareOut = Join-Path (Join-Path $OutputRootAbs "_prepared") $Spec.Dataset
    $logPath = Join-Path $LogDir ("prepare_{0}.log" -f $Spec.Dataset)
    New-Item -ItemType Directory -Force -Path $prepareOut | Out-Null
    $args = @(
        "paper_aaai27\scripts\course_baseline_adaptability.py",
        "--split-root", $seed2025Split,
        "--stream", (Resolve-RunPath $Repo $Spec.Stream),
        "--rel-dir", (Resolve-RunPath $Repo $Spec.RelationDir),
        "--out", $prepareOut,
        "--pcgnn-root", $PCGNNRootAbs,
        "--pcgnn-dataset-name", $Spec.AtomicDataset,
        "--targets", "pcgnn",
        "--max-train-pos", "-1",
        "--max-val-pos", "-1",
        "--max-test-pos", "-1"
    )
    Invoke-Python $args $logPath ("prepare dataset={0}" -f $Spec.Dataset)
    if (-not (Test-AtomicReady $Spec)) { throw "Preparation finished without complete atomic data: $($Spec.AtomicDataset)" }
}

function New-Task([pscustomobject]$Spec, [int]$Seed) {
    $splitName = Split-Name $Seed
    $outDir = Join-Path $OutputRootAbs ("{0}_seed{1}_full_formal_kg_warm" -f $Spec.Dataset, $Seed)
    return [pscustomobject]@{
        Dataset = $Spec.Dataset
        Seed = $Seed
        SplitName = $splitName
        SplitDir = Get-SplitDir $Spec $Seed
        AtomicDataset = $Spec.AtomicDataset
        ConfigFile = Get-AtomicConfig $Spec
        OutDir = $outDir
        CheckpointDir = Join-Path $outDir "checkpoints"
        ReportPath = Join-Path $outDir "pcgnn_strict_adapter_report.json"
        LogPath = Join-Path $LogDir ("{0}_seed_{1}.log" -f $Spec.Dataset, $Seed)
    }
}

function Test-ValidReport([pscustomobject]$Task) {
    if (-not (Test-Path -LiteralPath $Task.ReportPath)) { return $false }
    try {
        $report = Get-Content -Raw -LiteralPath $Task.ReportPath | ConvertFrom-Json
        $actualSplit = [System.IO.Path]::GetFullPath([string]$report.split_root)
        $expectedSplit = [System.IO.Path]::GetFullPath($Task.SplitDir)
        return (
            [int]$report.seed -eq $Task.Seed -and
            $actualSplit.Equals($expectedSplit, [System.StringComparison]::OrdinalIgnoreCase) -and
            [string]$report.requested_device -eq "cuda" -and
            [string]$report.device -eq "cuda" -and
            [string]$report.session_graph_backend -eq "torch_batch_scatter" -and
            [double]$report.kg_loss_weight -eq $KgLossWeight -and
            [string]$report.rs_candidate_mode -eq "warm" -and
            $null -ne $report.test.full_cold_item_macro.'N@10'
        )
    } catch {
        return $false
    }
}

function Invoke-PCGNNTask([pscustomobject]$Task) {
    if ((-not $Force) -and (Test-ValidReport $Task)) {
        Write-QueueLog "SKIP dataset=$($Task.Dataset) seed=$($Task.Seed) | valid report exists"
        return
    }
    New-Item -ItemType Directory -Force -Path $Task.OutDir, $Task.CheckpointDir | Out-Null
    $args = @(
        "paper_aaai27\scripts\pcgnn_strict_adapter.py",
        "--split-root", $Task.SplitDir,
        "--dataset-name", $Task.AtomicDataset,
        "--config-file", $Task.ConfigFile,
        "--seed", "$($Task.Seed)",
        "--max-train-examples", "-1",
        "--max-val-examples", "-1",
        "--max-test-examples", "-1",
        "--epochs", "$Epochs",
        "--early-stop-patience", "$EarlyStopPatience",
        "--learning-rate", "$LearningRate",
        "--train-batch-size", "$TrainBatchSize",
        "--eval-batch-size", "$EvalBatchSize",
        "--kg-batch-size", "$KgBatchSize",
        "--kg-loss-weight", "$KgLossWeight",
        "--rs-candidate-mode", "warm",
        "--device", "cuda",
        "--out-dir", $Task.OutDir,
        "--checkpoint-dir", $Task.CheckpointDir
    )
    Invoke-Python $args $Task.LogPath ("PCGNN dataset={0} seed={1}" -f $Task.Dataset, $Task.Seed)
    if (-not (Test-ValidReport $Task)) { throw "PCGNN finished without a valid report: $($Task.ReportPath)" }
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo
$PythonRunnerAbs = Resolve-RunPath $Repo $PythonRunner
$PCGNNRootAbs = Resolve-RunPath $Repo "paper_aaai27\baseline_sources\PCGNN_recbole_drive\RecBole-master"
$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$Timestamp = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$LogDir = Join-Path (Join-Path $OutputRootAbs "_logs") $Timestamp
$MasterLog = Join-Path $LogDir "queue.log"

Assert-Path $PythonRunnerAbs "Python runner"
Assert-Path (Join-Path $Repo "paper_aaai27\scripts\course_baseline_adaptability.py") "atomic exporter"
Assert-Path (Join-Path $Repo "paper_aaai27\scripts\pcgnn_strict_adapter.py") "PCGNN adapter"
Assert-Path $PCGNNRootAbs "PCGNN source root"

$specs = @($Datasets | ForEach-Object { Get-DatasetSpec $_ })
$tasks = @()
foreach ($spec in $specs) {
    Assert-Path (Resolve-RunPath $Repo $spec.Stream) "stream data"
    Assert-Path (Resolve-RunPath $Repo $spec.RelationDir) "relation directory"
    foreach ($seed in $Seeds) {
        $task = New-Task $spec $seed
        Assert-SplitArtifacts $task.Dataset $task.Seed $task.SplitDir
        $tasks += $task
    }
}

Write-Host ("PCGNN serial queue: {0} tasks" -f $tasks.Count)
foreach ($task in $tasks) {
    Write-Host ("TASK dataset={0} seed={1} split={2} atomic={3} out={4}" -f $task.Dataset, $task.Seed, $task.SplitDir, $task.AtomicDataset, $task.OutDir)
}

if ($DryRun) {
    Write-Host "DryRun: inputs validated; no atomic export or training executed."
    return
}

New-Item -ItemType Directory -Force -Path $OutputRootAbs, $LogDir | Out-Null
Write-QueueLog ("QUEUE START datasets={0} seeds={1}" -f ($Datasets -join ","), ($Seeds -join ","))
foreach ($spec in $specs) { Prepare-Atomic $spec }
foreach ($task in $tasks) { Invoke-PCGNNTask $task }
Write-QueueLog "QUEUE COMPLETE"
