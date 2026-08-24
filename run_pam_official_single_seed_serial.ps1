param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".runtime_tmp\aldi_tf1_py37\python.exe",
    [string]$ProjectPythonRunner = ".\py.bat",
    [string]$OutputRoot = "outputs\pam_official_single_seed_seed2025",
    [int]$Seed = 2025,
    [int]$ColdThreshold = 1,
    [int]$Epochs = 1,
    [int]$BatchSize = 512,
    [double]$LearningRate = 0.001,
    [int]$EmbDim = 8,
    [int]$HiddenDim = 16,
    [int]$CateDim = 8,
    [int]$NegPerPos = 1,
    [int]$MaxTrainPos = 4096,
    [int]$MaxEvalRows = 2048,
    [int]$EvalItemBatchSize = 1024,
    [int]$MaxCatesPerItem = 8,
    [string]$PamRoot = ".runtime_tmp\PAM",
    [string]$ResultSubdir = "main_table_balanced_itemmacro_v1",
    [string]$RunId = "",
    [bool]$UseGpu = $false,
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
    [string]$SplitRoot,
    [string]$RelationDir = ""
) {
    $splitName = Split-Name $TaskSeed
    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $TaskSeed
        DataDir = $DataDir
        SplitName = $splitName
        SplitDir = Join-Path $SplitRoot $splitName
        ResultSubdir = $ResultSubdir
        RelationDir = $RelationDir
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
    Join-Path (Result-Dir $Task) "pam_official_static_result.json"
}

function Task-LogDir([pscustomobject]$Task) {
    Join-Path $LogDir ("{0}_seed_{1}" -f $Task.Dataset, $Task.Seed)
}

function Set-Tf1Path() {
    $tfRoot = Split-Path -Parent $PythonRunnerAbs
    $env:PATH = (Join-Path $tfRoot "Library\bin") + ";" + (Join-Path $tfRoot "Scripts") + ";" + $tfRoot + ";" + $env:PATH
}

function Set-PamEnv([pscustomobject]$Task, [string]$OutDir) {
    $env:PYTHONUNBUFFERED = "1"
    $env:PAM_DATA_DIR = $Task.DataDir
    $env:PAM_STATIC_SPLIT_DIR = $Task.SplitDir
    $env:PAM_BASELINE_OUTPUT_DIR = $OutDir
    $env:PAM_ROOT = $PamRootAbs
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
    $env:PAM_USE_GPU = if ($UseGpu) { "1" } else { "0" }
    if (-not [string]::IsNullOrWhiteSpace($Task.RelationDir)) {
        $env:PAM_RELATION_DIR = $Task.RelationDir
    }
    else {
        Remove-Item Env:PAM_RELATION_DIR -ErrorAction SilentlyContinue
    }
}

function Invoke-PamExport([pscustomobject]$Task, [string]$LogPath) {
    Push-Location $Repo
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $ProjectPythonRunnerAbs -u "pam_official_static.py" --mode export > $LogPath 2>&1
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

function Invoke-PamTrainEval([pscustomobject]$Task, [string]$LogPath) {
    Push-Location $Repo
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonRunnerAbs -u "pam_official_static.py" --mode train_eval > $LogPath 2>&1
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }
    if ($exit -ne 0) {
        throw "PAM train/eval failed for dataset=$($Task.Dataset) seed=$($Task.Seed) with exit=$exit. See $LogPath"
    }
}

function Invoke-PamTask([pscustomobject]$Task) {
    $outDir = Result-Dir $Task
    $resultPath = Result-Path $Task
    $taskLogDir = Task-LogDir $Task
    $exportLog = Join-Path $taskLogDir "pam_export.log"
    $taskLog = Join-Path $taskLogDir "pam_train_eval.log"

    if ((-not $Force) -and (Test-Path -LiteralPath $resultPath)) {
        Write-QueueLog "SKIP dataset=$($Task.Dataset) seed=$($Task.Seed) | exists=$resultPath"
        return
    }

    Assert-InputPath $Task.DataDir "data dir"
    Assert-SplitArtifacts $Task
    New-Item -ItemType Directory -Force -Path $outDir, $taskLogDir | Out-Null
    Set-Tf1Path
    Set-PamEnv $Task $outDir

    Write-QueueLog "START PAM export dataset=$($Task.Dataset) seed=$($Task.Seed) | max_train_pos=$MaxTrainPos | max_eval_rows=$MaxEvalRows | out=$outDir | log=$exportLog"
    Invoke-PamExport $Task $exportLog
    Write-QueueLog "END PAM export dataset=$($Task.Dataset) seed=$($Task.Seed)"

    Write-QueueLog "START PAM train/eval dataset=$($Task.Dataset) seed=$($Task.Seed) | epochs=$Epochs | log=$taskLog"
    Invoke-PamTrainEval $Task $taskLog
    Write-QueueLog "END PAM train/eval dataset=$($Task.Dataset) seed=$($Task.Seed) | log=$taskLog"
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "PAM official adapter finished without expected result: $resultPath"
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

$PythonRunnerAbs = Resolve-RunPath $Repo $PythonRunner
$ProjectPythonRunnerAbs = Resolve-RunPath $Repo $ProjectPythonRunner
$Adapter = Join-Path $Repo "pam_official_static.py"
$PamModel = Join-Path $PamRootAbs "PAM-F\model.py"
$PamEngine = Join-Path $PamRootAbs "PAM-F\engine.py"
if (-not $DryRun) {
    foreach ($path in @($PythonRunnerAbs, $ProjectPythonRunnerAbs, $Adapter, $PamModel, $PamEngine)) {
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
    New-Task "mooccube" $Seed "processed_data_hin_clean_pop5" $MooccubeRoot "MOOCCube\relations"
    New-Task "junyi" $Seed "processed_data_junyi" $JunyiSeed2025Root "processed_data_junyi\relations"
    New-Task "coco" $Seed "processed_data_coco" $CocoRoot "processed_data_coco\relations"
)

Write-Host ("Total tasks: {0}" -f $tasks.Count)
Write-Host ("Adapter: {0}" -f $Adapter)
Write-Host ("Official PAM source: {0}" -f $PamRootAbs)
Write-Host ("TF1 Python: {0}" -f $PythonRunnerAbs)
Write-Host ("epochs={0} batch={1} lr={2} emb={3} hidden={4} max_train_pos={5} max_eval_rows={6} use_gpu={7}" -f `
    $Epochs, $BatchSize, $LearningRate, $EmbDim, $HiddenDim, $MaxTrainPos, $MaxEvalRows, [bool]$UseGpu)
foreach ($task in $tasks) {
    $outDir = Result-Dir $task
    Write-Host ("TASK dataset={0} seed={1} data={2} split={3} relation={4} out={5}" -f `
        $task.Dataset, $task.Seed, $task.DataDir, $task.SplitDir, $task.RelationDir, $outDir)
}

if ($DryRun) {
    Write-Host "DryRun: no training commands were executed."
    return
}

Write-QueueLog ("QUEUE START PAM official single seed | seed={0} | epochs={1} | batch={2} | max_train_pos={3} | max_eval_rows={4} | use_gpu={5} | force={6}" -f `
    $Seed, $Epochs, $BatchSize, $MaxTrainPos, $MaxEvalRows, [bool]$UseGpu, [bool]$Force)

foreach ($task in $tasks) {
    Invoke-PamTask $task
}

foreach ($dataset in @("mooccube", "junyi", "coco")) {
    Invoke-AggregateDataset $dataset
}

Write-QueueLog "QUEUE DONE PAM official single seed"
Write-Host ""
Write-Host "Master log: $MasterLog"
Write-Host "Aggregates: $(Join-Path $QueueRootAbs 'aggregate')"
