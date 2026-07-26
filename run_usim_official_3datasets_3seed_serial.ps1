param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputRoot = "outputs\usim_official_3datasets_3seed",
    [string]$CheckpointRoot = "checkpoints\usim_official_3datasets_3seed",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [int]$BackboneEpochs = 20,
    [int]$MapperEpochs = 20,
    [int]$RlEpochs = 50,
    [int]$RlBatchSize = 8,
    [string]$ResultSubdir = "main_table_balanced_itemmacro_v1",
    [string]$RunId = "",
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

function Split-Name([int]$Seed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $Seed
}

function New-Task(
    [string]$Dataset,
    [int]$Seed,
    [string]$DataDir,
    [string]$SplitRoot
) {
    $splitName = Split-Name $Seed
    $splitDir = Join-Path $SplitRoot $splitName
    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $Seed
        DataDir = $DataDir
        SplitName = $splitName
        SplitDir = $splitDir
        ResultSubdir = $ResultSubdir
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

function Assert-SafeChildPath([string]$Parent, [string]$Child) {
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd([char[]]@('\', '/'))
    $childFull = [System.IO.Path]::GetFullPath($Child)
    $prefix = $parentFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $childFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside queue root: parent=$parentFull child=$childFull"
    }
}

function Result-Dir([pscustomobject]$Task) {
    Join-Path $Task.SplitDir $Task.ResultSubdir
}

function Result-Path([pscustomobject]$Task) {
    Join-Path (Result-Dir $Task) "usim_official_static_result.json"
}

function Checkpoint-Dir([pscustomobject]$Task) {
    Join-Path (Join-Path $CheckpointRootAbs $Task.Dataset) $Task.SplitName
}

function Task-LogPath([pscustomobject]$Task) {
    Join-Path $LogDir ("{0}_seed_{1}.log" -f $Task.Dataset, $Task.Seed)
}

function Set-OfficialUsimEnv([pscustomobject]$Task, [string]$OutDir, [string]$CkptDir) {
    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = $Task.DataDir
    $env:USIM_STATIC_SPLIT_DIR = $Task.SplitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $OutDir
    $env:USIM_STATIC_SEED = "$($Task.Seed)"
    $env:USIM_SEED = "$($Task.Seed)"
    $env:USIM_OFFICIAL_SEED = "$($Task.Seed)"
    $env:USIM_COLD_THRESHOLD = "$ColdThreshold"
    $env:USIM_EVAL_N_NEG = "$EvalNeg"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_OFFICIAL_BEST_AVG_MODE = "item_macro"
    $env:USIM_OFFICIAL_RUN_SAMPLED_EVAL = "0"

    $env:USIM_OFFICIAL_BACKBONE_EPOCHS = "$BackboneEpochs"
    $env:USIM_OFFICIAL_MLP_EPOCHS = "$MapperEpochs"
    $env:USIM_OFFICIAL_RL_EPOCHS = "$RlEpochs"
    $env:USIM_OFFICIAL_RL_BATCH_SIZE = "$RlBatchSize"
    $env:USIM_OFFICIAL_EVAL_N_NEG = "$EvalNeg"

    $env:USIM_OFFICIAL_MAX_RL_BATCHES = "0"
    $env:USIM_OFFICIAL_MAX_BACKBONE_INTERACTIONS = "0"
    $env:USIM_OFFICIAL_MAX_MLP_INTERACTIONS = "0"
    $env:USIM_OFFICIAL_MAX_RL_ITEMS = "0"

    $env:USIM_OFFICIAL_CKPT_DIR = $CkptDir
    $env:USIM_OFFICIAL_SAVE_CKPT = "1"
    $env:USIM_OFFICIAL_AUTO_RESUME = "1"
    $env:USIM_OFFICIAL_FORCE_FRESH = if ($Force) { "1" } else { "0" }
    $env:USIM_OFFICIAL_SAVE_OPT_STATE = "1"
}

function Invoke-OfficialUsimTask([pscustomobject]$Task) {
    $outDir = Result-Dir $Task
    $resultPath = Result-Path $Task
    $ckptDir = Checkpoint-Dir $Task
    $logPath = Task-LogPath $Task

    if ((-not $Force) -and (Test-Path -LiteralPath $resultPath)) {
        Write-QueueLog "SKIP dataset=$($Task.Dataset) seed=$($Task.Seed) | exists=$resultPath"
        return
    }

    Assert-InputPath $Task.DataDir "data dir"
    Assert-SplitArtifacts $Task
    New-Item -ItemType Directory -Force -Path $outDir, $ckptDir | Out-Null

    Set-OfficialUsimEnv $Task $outDir $ckptDir

    Write-QueueLog "START dataset=$($Task.Dataset) seed=$($Task.Seed) | data=$($Task.DataDir) | split=$($Task.SplitDir) | out=$outDir | ckpt=$ckptDir | log=$logPath"
    $runner = Resolve-RunPath $Repo $PythonRunner
    $cmd = '/d /c ""' + $runner + '" -u "usim_official_static_hin.py" > "' + $logPath + '" 2>&1"'
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
    Write-QueueLog "END dataset=$($Task.Dataset) seed=$($Task.Seed) | exit=$($process.ExitCode) | log=$logPath"
    if ($process.ExitCode -ne 0) {
        throw "Official USIM failed for dataset=$($Task.Dataset) seed=$($Task.Seed) with exit=$($process.ExitCode). See $logPath"
    }
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "Official USIM finished without expected result: $resultPath"
    }
}

function Invoke-AggregateDataset([string]$Dataset, [object[]]$DatasetTasks) {
    $stageRoot = Join-Path (Join-Path $QueueRootAbs "aggregate_inputs") $Dataset
    $outDir = Join-Path (Join-Path $QueueRootAbs "aggregate") $Dataset
    if (Test-Path -LiteralPath $stageRoot) {
        Assert-SafeChildPath $QueueRootAbs $stageRoot
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $stageRoot, $outDir | Out-Null

    foreach ($task in $DatasetTasks) {
        $src = Result-Path $task
        if (-not (Test-Path -LiteralPath $src)) {
            Write-QueueLog "AGGREGATE SKIP missing result dataset=$($task.Dataset) seed=$($task.Seed) | path=$src"
            return
        }
        $dstDir = Join-Path (Join-Path $stageRoot $task.SplitName) $task.ResultSubdir
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        Copy-Item -LiteralPath $src -Destination (Join-Path $dstDir "usim_official_static_result.json") -Force
    }

    $aggLog = Join-Path $LogDir ("aggregate_{0}.log" -f $Dataset)
    Write-QueueLog "START aggregate dataset=$Dataset | stage=$stageRoot | out=$outDir | log=$aggLog"
    $runner = Resolve-RunPath $Repo $PythonRunner
    $cmd = '/d /c ""' + $runner + '" -B "aggregate_main_table_static_results.py" --root "' + $stageRoot + '" --split-glob "strict_item_cold_balanced_thr*_seed_*" --result-subdir "' + $ResultSubdir + '" --metric-mode "item_macro" --out-dir "' + $outDir + '" > "' + $aggLog + '" 2>&1"'
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
    Write-QueueLog "END aggregate dataset=$Dataset | exit=$($process.ExitCode) | out=$outDir | log=$aggLog"
    if ($process.ExitCode -ne 0) {
        throw "Aggregation failed for dataset=$Dataset with exit=$($process.ExitCode). See $aggLog"
    }
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

if ($RunId -and ($RunId -match '[\\/:*?"<>|]')) {
    throw "RunId must not contain path separator or invalid filename characters: $RunId"
}

$QueueRootAbs = Resolve-RunPath $Repo $OutputRoot
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$Timestamp = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$LogDir = Join-Path (Join-Path $QueueRootAbs "_logs") $Timestamp
$MasterLog = Join-Path $LogDir "queue.log"

$PythonRunnerAbs = Resolve-RunPath $Repo $PythonRunner
$Adapter = Join-Path $Repo "usim_official_static_hin.py"
$OfficialUsim = Join-Path $Repo "USIM-main\cold_model\USIM.py"
if (-not $DryRun) {
    foreach ($path in @($PythonRunnerAbs, $Adapter, $OfficialUsim)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing required runner file: $path"
        }
    }
    New-Item -ItemType Directory -Force -Path $LogDir, $CheckpointRootAbs, $QueueRootAbs | Out-Null
}

$MooccubeRoot = "outputs\content_delta_pop5\static_item_cold_balanced"
$CocoRoot = "outputs\coco\single_seed_triage\ours_full"
$JunyiSeed2025Root = "outputs\junyi\mask_ablation\mask_tt"
$JunyiSeed2026Root = "outputs\junyi\main_table_3seed"

$tasks = @()
foreach ($seed in $Seeds) {
    $tasks += New-Task "mooccube" $seed "processed_data_hin_clean_pop5" $MooccubeRoot
}
foreach ($seed in $Seeds) {
    $junyiRoot = if ($seed -eq 2025) { $JunyiSeed2025Root } else { $JunyiSeed2026Root }
    $tasks += New-Task "junyi" $seed "processed_data_junyi" $junyiRoot
}
foreach ($seed in $Seeds) {
    $tasks += New-Task "coco" $seed "processed_data_coco" $CocoRoot
}

Write-Host ("Total tasks: {0}" -f $tasks.Count)
foreach ($task in $tasks) {
    $outDir = Result-Dir $task
    $ckptDir = Checkpoint-Dir $task
    Write-Host ("TASK dataset={0} seed={1} data={2} split={3} out={4} ckpt={5}" -f `
        $task.Dataset, $task.Seed, $task.DataDir, $task.SplitDir, $outDir, $ckptDir)
}

if ($DryRun) {
    Write-Host "DryRun: no training commands were executed."
    return
}

Write-QueueLog ("QUEUE START official USIM 3 datasets x {0} seeds | seeds={1} | epochs=BPR{2}/MLP{3}/RL{4} | rl_batch={5} | force={6}" -f `
    $Seeds.Count, ($Seeds -join ","), $BackboneEpochs, $MapperEpochs, $RlEpochs, $RlBatchSize, [bool]$Force)

foreach ($task in $tasks) {
    Invoke-OfficialUsimTask $task
}

foreach ($dataset in @("mooccube", "junyi", "coco")) {
    $datasetTasks = @($tasks | Where-Object { $_.Dataset -eq $dataset })
    Invoke-AggregateDataset $dataset $datasetTasks
}

Write-QueueLog "QUEUE DONE official USIM 3 datasets x 3 seeds"
Write-Host ""
Write-Host "Master log: $MasterLog"
Write-Host "Aggregates: $(Join-Path $QueueRootAbs 'aggregate')"
