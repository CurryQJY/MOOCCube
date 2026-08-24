param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputRoot = "outputs\m2vae_coldrec_single_seed_seed2025",
    [int]$Seed = 2025,
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [int]$MFEpochs = 5,
    [int]$M2VAEEpochs = 10,
    [int]$EmbSize = 64,
    [int]$BatchSize = 4096,
    [double]$MFLearningRate = 0.001,
    [double]$MFReg = 0.0001,
    [double]$M2VAELearningRate = 0.00005,
    [int]$EarlyStop = 10,
    [int]$EvalEvery = 1,
    [string]$TopN = "5,10,20",
    [string]$ColdRecRoot = ".runtime_tmp\ColdRec",
    [string]$ExtraArgs = "",
    [string]$ResultSubdir = "main_table_balanced_itemmacro_v1",
    [string]$RunId = "",
    [int]$GpuId = 0,
    [bool]$UseGpu = $true,
    [switch]$Force,
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

function Split-Name([int]$TaskSeed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $TaskSeed
}

function New-Task(
    [string]$Dataset,
    [int]$TaskSeed,
    [string]$DataDir,
    [string]$SplitRoot
) {
    $splitName = Split-Name $TaskSeed
    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $TaskSeed
        DataDir = $DataDir
        SplitName = $splitName
        SplitDir = Join-Path $SplitRoot $splitName
        ResultSubdir = $ResultSubdir
        ColdRecDataset = ("m2vae_mfpre_{0}_{1}" -f $Dataset, $splitName)
    }
}

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        Add-Content -LiteralPath $MasterLog -Encoding UTF8 -Value $line
    }
    Write-Host $line
}

function Assert-InputPath([string]$Path, [string]$Kind) {
    $abs = Resolve-RunPath $Repo $Path
    if (-not (Test-Path -LiteralPath $abs)) {
        throw "Missing ${Kind}: $abs"
    }
}

function Assert-SplitArtifacts([pscustomobject]$Task) {
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl", "static_split_summary.json")) {
        $path = Join-Path $Task.SplitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split artifact for dataset=$($Task.Dataset) seed=$($Task.Seed): $path"
        }
    }
}

function Result-Dir([pscustomobject]$Task) {
    Join-Path (Join-Path (Join-Path $QueueRootAbs $Task.Dataset) $Task.SplitName) $Task.ResultSubdir
}

function Result-Path([pscustomobject]$Task) {
    Join-Path (Result-Dir $Task) "m2vae_coldrec_static_result.json"
}

function Task-LogDir([pscustomobject]$Task) {
    Join-Path $LogDir ("{0}_seed_{1}" -f $Task.Dataset, $Task.Seed)
}

function Run-PythonStdin([string]$Code, [string]$LogPath, [string]$WorkingDirectory) {
    Push-Location $WorkingDirectory
    try {
        $Code | & $PythonRunnerAbs - *> $LogPath
        $exit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exit -ne 0) {
        throw "Python stdin command failed with exit=$exit. See $LogPath"
    }
}

function Invoke-ExportDataset([pscustomobject]$Task, [string]$LogPath) {
    $code = @"
from pathlib import Path
import json
import pandas as pd
from hin_data_common import load_hin_processed
from m2vae_coldrec_static import export_m2vae_coldrec_dataset

coldrec_root = Path(r"$ColdRecRootAbs")
dataset = r"$($Task.ColdRecDataset)"
data_dir = r"$($Task.DataDir)"
split_dir = Path(r"$($Task.SplitDir)")
meta, df, content = load_hin_processed(data_dir)
train_df = pd.read_pickle(split_dir / "static_train.pkl")
val_df = pd.read_pickle(split_dir / "static_val.pkl")
test_df = pd.read_pickle(split_dir / "static_test.pkl")
summary = json.load(open(split_dir / "static_split_summary.json", encoding="utf-8"))
path = export_m2vae_coldrec_dataset(
    coldrec_root=coldrec_root,
    dataset_name=dataset,
    meta=meta,
    content_emb=content,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    cold_threshold=int(summary.get("cold_threshold", $ColdThreshold)),
    source_data_dir=data_dir,
    split_dir=str(split_dir),
)
print(path)
"@
    Run-PythonStdin $code $LogPath $Repo
}

function Invoke-MFBackbone([pscustomobject]$Task, [string]$LogPath, [string]$OutDir) {
    $resultFile = Join-Path $OutDir "coldrec_native_mf_result.txt"
    Push-Location $ColdRecRootAbs
    try {
        & $PythonRunnerAbs -u "main.py" `
            --dataset $Task.ColdRecDataset `
            --model "MF" `
            --cold_object "item" `
            --epochs $MFEpochs `
            --topN $TopN `
            --bs $BatchSize `
            --emb_size $EmbSize `
            --lr $MFLearningRate `
            --reg $MFReg `
            --runs 1 `
            --seed $Task.Seed `
            --use_gpu $(if ($UseGpu) { "true" } else { "false" }) `
            --save_emb true `
            --gpu_id $GpuId `
            --early_stop $EarlyStop `
            --eval_every $EvalEvery `
            --result_file $resultFile `
            --result_overwrite *> $LogPath
        $exit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exit -ne 0) {
        throw "ColdRec MF failed for dataset=$($Task.Dataset) seed=$($Task.Seed) with exit=$exit. See $LogPath"
    }
}

function Set-M2VAEEnv([pscustomobject]$Task, [string]$OutDir) {
    $env:PYTHONUNBUFFERED = "1"
    $env:M2VAE_DATA_DIR = $Task.DataDir
    $env:M2VAE_STATIC_SPLIT_DIR = $Task.SplitDir
    $env:M2VAE_BASELINE_OUTPUT_DIR = $OutDir
    $env:M2VAE_COLDREC_ROOT = $ColdRecRootAbs
    $env:M2VAE_COLDREC_DATASET = $Task.ColdRecDataset
    $env:M2VAE_SEED = "$($Task.Seed)"
    $env:M2VAE_STATIC_SEED = "$($Task.Seed)"
    $env:M2VAE_COLD_THRESHOLD = "$ColdThreshold"
    $env:M2VAE_EVAL_N_NEG = "$EvalNeg"
    $env:M2VAE_STATIC_TEST_HISTORY = "train_only"
    $env:M2VAE_RUN_SAMPLED_EVAL = "0"
    $env:M2VAE_EPOCHS = "$M2VAEEpochs"
    $env:M2VAE_EMB_SIZE = "$EmbSize"
    $env:M2VAE_BATCH_SIZE = "$BatchSize"
    $env:M2VAE_LR = "$M2VAELearningRate"
    $env:M2VAE_EARLY_STOP = "$EarlyStop"
    $env:M2VAE_EVAL_EVERY = "$EvalEvery"
    $env:M2VAE_TOPN = $TopN
    $env:M2VAE_USE_GPU = if ($UseGpu) { "1" } else { "0" }
    $env:M2VAE_GPU_ID = "$GpuId"
    $env:M2VAE_PRETRAIN = "1"
    $env:M2VAE_PRETRAIN_UPDATE = "0"
    $env:M2VAE_BACKBONE = "MF"
    $env:M2VAE_COLDREC_EXTRA_ARGS = $ExtraArgs
}

function Invoke-M2VAETask([pscustomobject]$Task) {
    $outDir = Result-Dir $Task
    $resultPath = Result-Path $Task
    $taskLogDir = Task-LogDir $Task
    $exportLog = Join-Path $taskLogDir "export_dataset.log"
    $mfLog = Join-Path $taskLogDir "mf_backbone.log"
    $m2vaeLog = Join-Path $taskLogDir "m2vae_pretrain.log"

    if ((-not $Force) -and (Test-Path -LiteralPath $resultPath)) {
        Write-QueueLog "SKIP dataset=$($Task.Dataset) seed=$($Task.Seed) | exists=$resultPath"
        return
    }

    Assert-InputPath $Task.DataDir "data dir"
    Assert-SplitArtifacts $Task
    New-Item -ItemType Directory -Force -Path $outDir, $taskLogDir | Out-Null

    Write-QueueLog "START export dataset=$($Task.Dataset) seed=$($Task.Seed) | coldrec_dataset=$($Task.ColdRecDataset) | log=$exportLog"
    Invoke-ExportDataset $Task $exportLog
    Write-QueueLog "END export dataset=$($Task.Dataset) seed=$($Task.Seed)"

    $mfUser = Join-Path $ColdRecRootAbs ("emb\{0}_cold_item_MF_user_emb.pt" -f $Task.ColdRecDataset)
    $mfItem = Join-Path $ColdRecRootAbs ("emb\{0}_cold_item_MF_item_emb.pt" -f $Task.ColdRecDataset)
    if ($Force -or (-not (Test-Path -LiteralPath $mfUser)) -or (-not (Test-Path -LiteralPath $mfItem))) {
        Write-QueueLog "START MF dataset=$($Task.Dataset) seed=$($Task.Seed) | epochs=$MFEpochs | log=$mfLog"
        Invoke-MFBackbone $Task $mfLog $outDir
        Write-QueueLog "END MF dataset=$($Task.Dataset) seed=$($Task.Seed)"
    }
    else {
        Write-QueueLog "SKIP MF dataset=$($Task.Dataset) seed=$($Task.Seed) | existing backbone=$mfUser"
    }

    Set-M2VAEEnv $Task $outDir
    Write-QueueLog "START M2VAE dataset=$($Task.Dataset) seed=$($Task.Seed) | epochs=$M2VAEEpochs | out=$outDir | log=$m2vaeLog"
    Push-Location $Repo
    try {
        & $PythonRunnerAbs -u "m2vae_coldrec_static.py" *> $m2vaeLog
        $exit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-QueueLog "END M2VAE dataset=$($Task.Dataset) seed=$($Task.Seed) | exit=$exit | log=$m2vaeLog"
    if ($exit -ne 0) {
        throw "ColdRec M2VAE failed for dataset=$($Task.Dataset) seed=$($Task.Seed) with exit=$exit. See $m2vaeLog"
    }
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "ColdRec M2VAE finished without expected result: $resultPath"
    }
}

function Invoke-AggregateDataset([string]$Dataset) {
    $root = Join-Path $QueueRootAbs $Dataset
    $outDir = Join-Path (Join-Path $QueueRootAbs "aggregate") $Dataset
    $aggLog = Join-Path $LogDir ("aggregate_{0}.log" -f $Dataset)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Write-QueueLog "START aggregate dataset=$Dataset | root=$root | out=$outDir | log=$aggLog"
    Push-Location $Repo
    try {
        & $PythonRunnerAbs -B "aggregate_main_table_static_results.py" `
            --root $root `
            --split-glob "strict_item_cold_balanced_thr*_seed_*" `
            --result-subdir $ResultSubdir `
            --metric-mode "item_macro" `
            --out-dir $outDir *> $aggLog
        $exit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-QueueLog "END aggregate dataset=$Dataset | exit=$exit | out=$outDir | log=$aggLog"
    if ($exit -ne 0) {
        throw "Aggregation failed for dataset=$Dataset with exit=$exit. See $aggLog"
    }
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

if ($RunId -and ($RunId -match '[\\/:*?"<>|]')) {
    throw "RunId must not contain path separator or invalid filename characters: $RunId"
}

$QueueRootAbs = Resolve-RunPath $Repo $OutputRoot
$ColdRecRootAbs = Resolve-RunPath $Repo $ColdRecRoot
$Timestamp = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$LogDir = Join-Path (Join-Path $QueueRootAbs "_logs") $Timestamp
$MasterLog = Join-Path $LogDir "queue.log"

$PythonRunnerAbs = Resolve-RunPath $Repo $PythonRunner
$Adapter = Join-Path $Repo "m2vae_coldrec_static.py"
$ColdRecMain = Join-Path $ColdRecRootAbs "main.py"
$ColdRecM2VAE = Join-Path $ColdRecRootAbs "model\M2VAE.py"
$ColdRecMF = Join-Path $ColdRecRootAbs "model\MF.py"
if (-not $DryRun) {
    foreach ($path in @($PythonRunnerAbs, $Adapter, $ColdRecMain, $ColdRecM2VAE, $ColdRecMF)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing required runner/source file: $path"
        }
    }
    New-Item -ItemType Directory -Force -Path $LogDir, $QueueRootAbs | Out-Null
}

$MooccubeRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
$CocoRoot = "outputs\coco\single_seed_triage\ours_full"
$JunyiSeed2025Root = "outputs\junyi\mask_ablation\mask_tt"

$tasks = @(
    New-Task "mooccube" $Seed "processed_data_hin_clean_pop5" $MooccubeRoot
    New-Task "junyi" $Seed "processed_data_junyi" $JunyiSeed2025Root
    New-Task "coco" $Seed "processed_data_coco" $CocoRoot
)

Write-Host ("Total tasks: {0}" -f $tasks.Count)
Write-Host ("Adapter: {0}" -f $Adapter)
Write-Host ("ColdRec source: {0}" -f $ColdRecRootAbs)
Write-Host ("MF epochs={0} M2VAE epochs={1} EmbSize={2} BatchSize={3} UseGpu={4} ExtraArgs={5}" -f $MFEpochs, $M2VAEEpochs, $EmbSize, $BatchSize, [bool]$UseGpu, $ExtraArgs)
foreach ($task in $tasks) {
    $outDir = Result-Dir $task
    Write-Host ("TASK dataset={0} seed={1} data={2} split={3} out={4} coldrec_dataset={5}" -f `
        $task.Dataset, $task.Seed, $task.DataDir, $task.SplitDir, $outDir, $task.ColdRecDataset)
}

if ($DryRun) {
    Write-Host "DryRun: no training commands were executed."
    return
}

Write-QueueLog ("QUEUE START ColdRec M2VAE-MFpre single seed | seed={0} | mf_epochs={1} | m2vae_epochs={2} | emb={3} | batch={4} | use_gpu={5} | force={6}" -f `
    $Seed, $MFEpochs, $M2VAEEpochs, $EmbSize, $BatchSize, [bool]$UseGpu, [bool]$Force)

foreach ($task in $tasks) {
    Invoke-M2VAETask $task
}

foreach ($dataset in @("mooccube", "junyi", "coco")) {
    Invoke-AggregateDataset $dataset
}

Write-QueueLog "QUEUE DONE ColdRec M2VAE-MFpre single seed"
Write-Host ""
Write-Host "Master log: $MasterLog"
Write-Host "Aggregates: $(Join-Path $QueueRootAbs 'aggregate')"
