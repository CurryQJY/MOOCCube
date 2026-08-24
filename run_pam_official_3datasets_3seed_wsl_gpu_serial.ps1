param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\pam_official_full_single_seed_seed2025_complete_b2048",
    [ValidateSet("mooccube", "coco", "junyi")]
    [string[]]$Datasets = @("mooccube", "coco", "junyi"),
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$Epochs = 1,
    [int]$BatchSize = 2048,
    [double]$LearningRate = 0.001,
    [int]$EmbDim = 8,
    [int]$HiddenDim = 16,
    [int]$CateDim = 8,
    [int]$NegPerPos = 1,
    [int]$MaxTrainPos = 0,
    [int]$MaxEvalRows = 0,
    [int]$EvalItemBatchSize = 1024,
    [int]$MaxCatesPerItem = 8,
    [string]$PamRoot = ".runtime_tmp\PAM",
    [string]$ResultSubdir = "main_table_balanced_itemmacro_v1",
    [string]$WslPython = "/root/venvs/icychesszero_tf2_gpu/bin/python",
    [string]$ProjectPythonRunner = ".\py.bat",
    [string]$RunId = "",
    [string]$InitCheckpoint = "",
    [int]$StartEpoch = 0,
    [int]$WaitPollSeconds = 60,
    [switch]$SkipInitialWait,
    [switch]$SkipAggregate,
    [switch]$DisableAutoResumeFromSiblingEpoch,
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

function ConvertTo-WslPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath((Resolve-RunPath $Repo $Path))
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    return ($full -replace '\\', '/')
}

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Split-Name([int]$TaskSeed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $TaskSeed
}

function New-PamTask(
    [string]$Dataset,
    [int]$TaskSeed,
    [string]$DataDir,
    [string]$SplitDir,
    [string]$RelationDir
) {
    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $TaskSeed
        DataDir = $DataDir
        SplitName = Split-Name $TaskSeed
        SplitDir = $SplitDir
        RelationDir = $RelationDir
        ResultSubdir = $ResultSubdir
    }
}

function New-AllTasks {
    $tasks = New-Object System.Collections.Generic.List[object]

    if ($Datasets -contains "mooccube") {
        foreach ($seed in $Seeds) {
            $tasks.Add((New-PamTask `
                -Dataset "mooccube" `
                -TaskSeed $seed `
                -DataDir "processed_data_hin_clean_pop5" `
                -SplitDir (Join-Path "outputs\content_delta_pop5\static_item_cold_balanced" (Split-Name $seed)) `
                -RelationDir "MOOCCube\relations"))
        }
    }

    if ($Datasets -contains "coco") {
        foreach ($seed in $Seeds) {
            $tasks.Add((New-PamTask `
                -Dataset "coco" `
                -TaskSeed $seed `
                -DataDir "processed_data_coco" `
                -SplitDir (Join-Path "outputs\coco\single_seed_triage\ours_full" (Split-Name $seed)) `
                -RelationDir "processed_data_coco\relations"))
        }
    }

    if ($Datasets -contains "junyi") {
        foreach ($seed in $Seeds) {
            $junyiRoot = if ($seed -eq 2025) {
                "outputs\junyi\mask_ablation\mask_tt"
            }
            else {
                "outputs\junyi\main_table_3seed"
            }
            $tasks.Add((New-PamTask `
                -Dataset "junyi" `
                -TaskSeed $seed `
                -DataDir "processed_data_junyi" `
                -SplitDir (Join-Path $junyiRoot (Split-Name $seed)) `
                -RelationDir "processed_data_junyi\relations"))
        }
    }

    return $tasks.ToArray()
}

function Result-Dir([pscustomobject]$Task) {
    Join-Path (Join-Path (Join-Path $QueueRootAbs $Task.Dataset) $Task.SplitName) $Task.ResultSubdir
}

function Result-Path([pscustomobject]$Task) {
    Join-Path (Result-Dir $Task) "pam_official_static_result.json"
}

function Resolve-TaskInitState([pscustomobject]$Task) {
    if ($InitCheckpoint) {
        return [pscustomobject]@{
            Checkpoint = Resolve-RunPath $Repo $InitCheckpoint
            StartEpoch = $StartEpoch
        }
    }

    if ($DisableAutoResumeFromSiblingEpoch) {
        return [pscustomobject]@{
            Checkpoint = ""
            StartEpoch = 0
        }
    }

    $leaf = Split-Path -Leaf $QueueRootAbs
    if ($leaf -notmatch '^e(\d+)$') {
        return [pscustomobject]@{
            Checkpoint = ""
            StartEpoch = 0
        }
    }

    $targetEpoch = [int]$matches[1]
    if ($targetEpoch -ne $Epochs) {
        return [pscustomobject]@{
            Checkpoint = ""
            StartEpoch = 0
        }
    }

    $parent = Split-Path -Parent $QueueRootAbs
    if (-not (Test-Path -LiteralPath $parent)) {
        return [pscustomobject]@{
            Checkpoint = ""
            StartEpoch = 0
        }
    }

    $candidates = @(Get-ChildItem -LiteralPath $parent -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^e(\d+)$' -and ([int]$matches[1]) -lt $Epochs } |
        Sort-Object @{ Expression = { [int]($_.Name.Substring(1)) }; Descending = $true })

    foreach ($candidate in $candidates) {
        $candidateEpoch = [int]$candidate.Name.Substring(1)
        $prefix = Join-Path (Join-Path (Join-Path (Join-Path $candidate.FullName $Task.Dataset) $Task.SplitName) $Task.ResultSubdir) "checkpoints\pam_official_latest.ckpt"
        if (Test-Path -LiteralPath "${prefix}.index") {
            return [pscustomobject]@{
                Checkpoint = $prefix
                StartEpoch = $candidateEpoch
            }
        }
    }

    return [pscustomobject]@{
        Checkpoint = ""
        StartEpoch = 0
    }
}

function Task-LogDir([pscustomobject]$Task) {
    Join-Path $LogDir ("{0}_seed_{1}" -f $Task.Dataset, $Task.Seed)
}

function Set-WindowsPamEnv([pscustomobject]$Task, [string]$OutDir) {
    $initState = Resolve-TaskInitState $Task
    $env:PYTHONUNBUFFERED = "1"
    $env:PAM_DATA_DIR = Resolve-RunPath $Repo $Task.DataDir
    $env:PAM_STATIC_SPLIT_DIR = Resolve-RunPath $Repo $Task.SplitDir
    $env:PAM_BASELINE_OUTPUT_DIR = $OutDir
    $env:PAM_ROOT = $PamRootAbs
    $env:PAM_RELATION_DIR = Resolve-RunPath $Repo $Task.RelationDir
    $env:PAM_SEED = "$($Task.Seed)"
    $env:PAM_STATIC_SEED = "$($Task.Seed)"
    $env:PAM_COLD_THRESHOLD = "$ColdThreshold"
    $env:PAM_EPOCHS = "$Epochs"
    $env:PAM_BATCH_SIZE = "$BatchSize"
    $env:PAM_LR = "$LearningRate"
    $env:PAM_EMB_DIM = "$EmbDim"
    $env:PAM_HIDDEN_DIM = "$HiddenDim"
    $env:PAM_CATE_DIM = "$CateDim"
    $env:PAM_NEG_PER_POS = "$NegPerPos"
    $env:PAM_MAX_TRAIN_POS = "$MaxTrainPos"
    $env:PAM_MAX_EVAL_ROWS = "$MaxEvalRows"
    $env:PAM_EVAL_ITEM_BATCH_SIZE = "$EvalItemBatchSize"
    $env:PAM_MAX_CATES_PER_ITEM = "$MaxCatesPerItem"
    $env:PAM_USE_GPU = "0"
    $env:PAM_INIT_CKPT = $initState.Checkpoint
    $env:PAM_START_EPOCH = "$($initState.StartEpoch)"
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
        if (-not (Test-Path -LiteralPath (Resolve-RunPath $Repo $path))) {
            throw "Missing split artifact for dataset=$($Task.Dataset) seed=$($Task.Seed): $path"
        }
    }
}

function Invoke-WslBash([string]$BashScript, [string]$Label, [string]$LogPath) {
    $normalized = $BashScript -replace "`r`n", "`n"
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($normalized))
    & wsl.exe bash -lc "echo $encoded | base64 -d | bash"
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        throw "$Label failed with exit=$exit. See $LogPath"
    }
}

function New-WslPamScript(
    [pscustomobject]$Task,
    [string]$Mode,
    [string]$LogPath
) {
    $repoWsl = ConvertTo-WslPath $Repo
    $outWsl = ConvertTo-WslPath (Result-Dir $Task)
    $logWsl = ConvertTo-WslPath $LogPath
    $logDirWsl = ConvertTo-WslPath (Split-Path -Parent $LogPath)
    $cudaCacheWsl = ConvertTo-WslPath ".runtime_tmp\cuda_cache"
    $dataWsl = ConvertTo-WslPath $Task.DataDir
    $splitWsl = ConvertTo-WslPath $Task.SplitDir
    $relationWsl = ConvertTo-WslPath $Task.RelationDir
    $pamRootWsl = ConvertTo-WslPath $PamRoot
    $initState = Resolve-TaskInitState $Task
    $initCkptWsl = if ($initState.Checkpoint) { ConvertTo-WslPath $initState.Checkpoint } else { "" }
    $preflight = if ($Mode -eq "train_eval") {
@'
"$PY" - <<'PY'
import tensorflow as tf
print('preflight_tf', tf.__version__)
print('preflight_gpus', tf.config.list_physical_devices('GPU'))
PY
'@
    }
    else {
        ""
    }

    $template = @'
set -euo pipefail
mkdir -p __LOG_DIR__ __OUT_DIR__ __CUDA_CACHE__
exec > __LOG_PATH__ 2>&1
SP=/root/venvs/icychesszero_tf2_gpu/lib/python3.10/site-packages
PY=__WSL_PYTHON__
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cuda_cupti/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH=__CUDA_CACHE__
export CUDA_CACHE_MAXSIZE=2147483648
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
export PAM_DATA_DIR=__DATA_DIR__
export PAM_STATIC_SPLIT_DIR=__SPLIT_DIR__
export PAM_BASELINE_OUTPUT_DIR=__OUT_DIR__
export PAM_ROOT=__PAM_ROOT__
export PAM_RELATION_DIR=__RELATION_DIR__
export PAM_SEED=__SEED__
export PAM_STATIC_SEED=__SEED__
export PAM_COLD_THRESHOLD=__COLD_THRESHOLD__
export PAM_EPOCHS=__EPOCHS__
export PAM_BATCH_SIZE=__BATCH_SIZE__
export PAM_LR=__LEARNING_RATE__
export PAM_EMB_DIM=__EMB_DIM__
export PAM_HIDDEN_DIM=__HIDDEN_DIM__
export PAM_CATE_DIM=__CATE_DIM__
export PAM_NEG_PER_POS=__NEG_PER_POS__
export PAM_MAX_TRAIN_POS=__MAX_TRAIN_POS__
export PAM_MAX_EVAL_ROWS=__MAX_EVAL_ROWS__
export PAM_EVAL_ITEM_BATCH_SIZE=__EVAL_ITEM_BATCH_SIZE__
export PAM_MAX_CATES_PER_ITEM=__MAX_CATES_PER_ITEM__
export PAM_USE_GPU=1
export PAM_INIT_CKPT=__INIT_CKPT__
export PAM_START_EPOCH=__START_EPOCH__
cd __REPO__
echo "[$(date '+%F %T')] START PAM __MODE__ dataset=__DATASET__ seed=__SEED__"
echo "python=$PY"
echo "out=__OUT_DIR_TEXT__"
echo "init_checkpoint=__INIT_CKPT_TEXT__ start_epoch=__START_EPOCH__"
__PREFLIGHT__
exec "$PY" -u pam_official_static.py --mode __MODE__
'@

    $replacements = @{
        "__LOG_DIR__" = Quote-Bash $logDirWsl
        "__OUT_DIR__" = Quote-Bash $outWsl
        "__OUT_DIR_TEXT__" = $outWsl
        "__LOG_PATH__" = Quote-Bash $logWsl
        "__CUDA_CACHE__" = Quote-Bash $cudaCacheWsl
        "__WSL_PYTHON__" = Quote-Bash $WslPython
        "__DATA_DIR__" = Quote-Bash $dataWsl
        "__SPLIT_DIR__" = Quote-Bash $splitWsl
        "__PAM_ROOT__" = Quote-Bash $pamRootWsl
        "__RELATION_DIR__" = Quote-Bash $relationWsl
        "__REPO__" = Quote-Bash $repoWsl
        "__SEED__" = "$($Task.Seed)"
        "__COLD_THRESHOLD__" = "$ColdThreshold"
        "__EPOCHS__" = "$Epochs"
        "__BATCH_SIZE__" = "$BatchSize"
        "__LEARNING_RATE__" = "$LearningRate"
        "__EMB_DIM__" = "$EmbDim"
        "__HIDDEN_DIM__" = "$HiddenDim"
        "__CATE_DIM__" = "$CateDim"
        "__NEG_PER_POS__" = "$NegPerPos"
        "__MAX_TRAIN_POS__" = "$MaxTrainPos"
        "__MAX_EVAL_ROWS__" = "$MaxEvalRows"
        "__EVAL_ITEM_BATCH_SIZE__" = "$EvalItemBatchSize"
        "__MAX_CATES_PER_ITEM__" = "$MaxCatesPerItem"
        "__INIT_CKPT__" = Quote-Bash $initCkptWsl
        "__INIT_CKPT_TEXT__" = $initCkptWsl
        "__START_EPOCH__" = "$($initState.StartEpoch)"
        "__MODE__" = $Mode
        "__DATASET__" = $Task.Dataset
        "__PREFLIGHT__" = $preflight
    }

    $script = $template
    foreach ($key in $replacements.Keys) {
        $script = $script.Replace($key, [string]$replacements[$key])
    }
    return $script
}

function Get-RunningPamProcesses {
    $out = & wsl.exe bash -lc "ps -eo pid,etime,args | grep 'pam_official_static.py --mode train_eval' | grep -v grep || true" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return @()
    }
    return @($out | Where-Object { $_ -and $_.Trim() })
}

function Invoke-WindowsPamExport([pscustomobject]$Task, [string]$OutDir, [string]$LogPath) {
    Set-WindowsPamEnv $Task $OutDir
    Push-Location $Repo
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $ProjectPythonRunnerAbs -u "pam_official_static.py" --mode export *> $LogPath
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }
    if ($exit -ne 0) {
        throw "PAM export failed for dataset=$($Task.Dataset) seed=$($Task.Seed) with exit=$exit. See $LogPath"
    }
}

function Wait-ForPamIdle {
    if ($SkipInitialWait) {
        Write-QueueLog "SKIP initial wait for existing PAM processes"
        return
    }
    while ($true) {
        $running = @(Get-RunningPamProcesses)
        if ($running.Count -eq 0) {
            Write-QueueLog "No existing WSL PAM train/eval process detected"
            return
        }
        Write-QueueLog ("WAIT existing WSL PAM process(es): {0}" -f (($running -join " | ").Trim()))
        Start-Sleep -Seconds $WaitPollSeconds
    }
}

function Invoke-PamTask([pscustomobject]$Task) {
    $outDir = Result-Dir $Task
    $resultPath = Result-Path $Task
    $taskLogDir = Task-LogDir $Task
    $exportLog = Join-Path $taskLogDir "pam_export_wsl_gpu.log"
    $trainLog = Join-Path $taskLogDir "pam_train_eval_wsl_gpu.log"

    if ((-not $Force) -and (Test-Path -LiteralPath $resultPath)) {
        Write-QueueLog "SKIP dataset=$($Task.Dataset) seed=$($Task.Seed) | exists=$resultPath"
        return
    }

    Assert-InputPath $Task.DataDir "data dir"
    Assert-InputPath $Task.RelationDir "relation dir"
    Assert-SplitArtifacts $Task
    New-Item -ItemType Directory -Force -Path $outDir, $taskLogDir | Out-Null

    Write-QueueLog "START export dataset=$($Task.Dataset) seed=$($Task.Seed) | runner=$ProjectPythonRunnerAbs | split=$($Task.SplitDir) | log=$exportLog"
    Invoke-WindowsPamExport -Task $Task -OutDir $outDir -LogPath $exportLog
    Write-QueueLog "END export dataset=$($Task.Dataset) seed=$($Task.Seed)"

    Write-QueueLog "START train/eval dataset=$($Task.Dataset) seed=$($Task.Seed) | epochs=$Epochs batch=$BatchSize | log=$trainLog"
    Invoke-WslBash `
        -BashScript (New-WslPamScript -Task $Task -Mode "train_eval" -LogPath $trainLog) `
        -Label "PAM train/eval dataset=$($Task.Dataset) seed=$($Task.Seed)" `
        -LogPath $trainLog
    Write-QueueLog "END train/eval dataset=$($Task.Dataset) seed=$($Task.Seed)"

    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "PAM finished without expected result for dataset=$($Task.Dataset) seed=$($Task.Seed): $resultPath"
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
        & $ProjectPythonRunnerAbs -B "aggregate_main_table_static_results.py" `
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
$PamRootAbs = Resolve-RunPath $Repo $PamRoot
$Timestamp = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$LogDir = Join-Path (Join-Path $QueueRootAbs "_logs") $Timestamp
$MasterLog = Join-Path $LogDir "queue.log"
$ProjectPythonRunnerAbs = Resolve-RunPath $Repo $ProjectPythonRunner

$Adapter = Join-Path $Repo "pam_official_static.py"
$PamModel = Join-Path $PamRootAbs "PAM-F\model.py"
$PamEngine = Join-Path $PamRootAbs "PAM-F\engine.py"
$tasks = @(New-AllTasks)

Write-Host ("Total tasks: {0}" -f $tasks.Count)
Write-Host ("Adapter: {0}" -f $Adapter)
Write-Host ("Official PAM source: {0}" -f $PamRootAbs)
Write-Host ("Export runner: {0}" -f $ProjectPythonRunnerAbs)
Write-Host ("WSL GPU Python: {0}" -f $WslPython)
Write-Host ("Output root: {0}" -f $QueueRootAbs)
Write-Host ("Datasets: {0}" -f (($Datasets | ForEach-Object { "$_" }) -join ","))
Write-Host ("Seeds: {0}" -f (($Seeds | ForEach-Object { "$_" }) -join ","))
Write-Host ("epochs={0} batch={1} lr={2} emb={3} hidden={4} max_train_pos={5} max_eval_rows={6} PAM_USE_GPU=1" -f `
    $Epochs, $BatchSize, $LearningRate, $EmbDim, $HiddenDim, $MaxTrainPos, $MaxEvalRows)
foreach ($task in $tasks) {
    $outDir = Result-Dir $task
    $initState = Resolve-TaskInitState $task
    Write-Host ("TASK dataset={0} seed={1} data={2} split={3} relation={4} out={5}" -f `
        $task.Dataset, $task.Seed, $task.DataDir, $task.SplitDir, $task.RelationDir, $outDir)
    Write-Host ("TASK_RESUME dataset={0} seed={1} init_checkpoint={2} start_epoch={3}" -f `
        $task.Dataset, $task.Seed, $initState.Checkpoint, $initState.StartEpoch)
}

if ($DryRun) {
    Write-Host "DryRun: no WSL commands were executed."
    return
}

foreach ($path in @($ProjectPythonRunnerAbs, $Adapter, $PamModel, $PamEngine)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required runner/source file: $path"
    }
}
New-Item -ItemType Directory -Force -Path $LogDir, $QueueRootAbs | Out-Null

Write-QueueLog ("QUEUE START PAM official WSL/GPU 3dataset 3seed | seeds={0} | epochs={1} | batch={2} | max_train_pos={3} | max_eval_rows={4} | force={5}" -f `
    (($Seeds | ForEach-Object { "$_" }) -join ","), $Epochs, $BatchSize, $MaxTrainPos, $MaxEvalRows, [bool]$Force)

Wait-ForPamIdle

foreach ($task in $tasks) {
    Invoke-PamTask $task
}

if (-not $SkipAggregate) {
    foreach ($dataset in @("mooccube", "coco", "junyi")) {
        Invoke-AggregateDataset $dataset
    }
}

Write-QueueLog "QUEUE DONE PAM official WSL/GPU 3dataset 3seed"
Write-Host ""
Write-Host "Master log: $MasterLog"
Write-Host "Aggregates: $(Join-Path $QueueRootAbs 'aggregate')"
