param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_xes3g5m",
    [string]$RelationDir = "processed_data_xes3g5m\relations",
    [string]$OutputRoot = "outputs\xes3g5m\ours_sota_serial",
    [string]$CheckpointRoot = "checkpoints\xes3g5m\ours_sota_serial",
    [int[]]$Seeds = @(2025),
    [int[]]$WaitPids = @(),
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [string[]]$TrainModels = @("BPR", "LightGCN", "DropoutNet", "GAR", "CCFCRec", "ALDI", "MARec", "LightGCL", "SAGERec", "CourseAware-MLP"),
    [int]$PollSeconds = 120,
    [int]$PopBatchSize = 512,
    [int]$ContentProfileBatchSize = 512,
    [int]$BprEpochs = 10,
    [int]$BprEvalInterval = 5,
    [int]$BprEmbDim = 64,
    [int]$BprBatchSize = 4096,
    [int]$LightGCNEpochs = 5,
    [int]$LightGCNEvalInterval = 5,
    [int]$LightGCNEmbDim = 64,
    [int]$LightGCNBatchSize = 2048,
    [int]$LightGCNLayers = 1,
    [double]$LightGCNContentWeight = 0.35,
    [int]$DropoutEpochs = 5,
    [int]$DropoutEvalInterval = 5,
    [int]$DropoutBatchSize = 512,
    [int]$GarEpochs = 5,
    [int]$GarEvalInterval = 5,
    [int]$GarBatchSize = 512,
    [int]$CCFCEpochs = 5,
    [int]$CCFCEvalInterval = 5,
    [int]$CCFCEmbDim = 64,
    [int]$CCFCHiddenDim = 128,
    [int]$CCFCBatchSize = 1024,
    [int]$ALDITeacherEpochs = 5,
    [int]$ALDIStudentEpochs = 5,
    [int]$ALDIEvalInterval = 5,
    [int]$ALDIEmbDim = 64,
    [int]$ALDIBatchSize = 1024,
    [int]$MARecBatchSize = 512,
    [string]$MARecLambdas = "100,300",
    [string]$MARecAlphas = "0.1,0.5",
    [string]$MARecContentBetas = "0,0.5,1.0",
    [int]$LightGCLEpochs = 5,
    [int]$LightGCLEvalInterval = 5,
    [int]$LightGCLEmbDim = 64,
    [int]$LightGCLBatchSize = 1024,
    [int]$LightGCLLayers = 1,
    [int]$LightGCLSvdRank = 5,
    [int]$SageRecEpochs = 5,
    [int]$SageRecEvalInterval = 5,
    [int]$SageRecEmbDim = 64,
    [int]$SageRecBatchSize = 1024,
    [int]$SageRecSampleTopN = 10,
    [int]$SageRecMaxHistLen = 50,
    [int]$CourseMlpEpochs = 5,
    [int]$CourseMlpEvalInterval = 5,
    [int]$CourseMlpBatchSize = 1024,
    [switch]$SkipFastStage,
    [switch]$SkipTrainStage,
    [switch]$CpuOnly,
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

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$QueueDir = Join-Path $OutputRootAbs "_queue"
$QueueLog = Join-Path $QueueDir "overnight_lightweight_serial.log"
$Runner = Join-Path $Repo "run_xes3g5m_lightweight_baselines.ps1"

New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Missing runner: $Runner"
}

function Write-OvernightLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $QueueLog -Encoding UTF8 -Value $line
    Write-Host $line
}

function Wait-ForPids([int[]]$Pids) {
    foreach ($waitPid in $Pids) {
        while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
            Write-OvernightLog "WAIT pid=$waitPid still running"
            Start-Sleep -Seconds $PollSeconds
        }
        Write-OvernightLog "WAIT done pid=$waitPid"
    }
}

function Invoke-LightweightRunner([string[]]$Models, [bool]$UseGpuStage) {
    $runnerArgs = @{
        Repo = $Repo
        PythonRunner = $PythonRunner
        DataDir = $DataDir
        RelationDir = $RelationDir
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        Seeds = $Seeds
        ColdThreshold = $ColdThreshold
        EvalNeg = $EvalNeg
        Models = $Models
        PopBatchSize = $PopBatchSize
        ContentProfileBatchSize = $ContentProfileBatchSize
        BprEpochs = $BprEpochs
        BprEvalInterval = $BprEvalInterval
        BprEmbDim = $BprEmbDim
        BprBatchSize = $BprBatchSize
        LightGCNEpochs = $LightGCNEpochs
        LightGCNEvalInterval = $LightGCNEvalInterval
        LightGCNEmbDim = $LightGCNEmbDim
        LightGCNBatchSize = $LightGCNBatchSize
        LightGCNLayers = $LightGCNLayers
        LightGCNContentWeight = $LightGCNContentWeight
        DropoutEpochs = $DropoutEpochs
        DropoutEvalInterval = $DropoutEvalInterval
        DropoutBatchSize = $DropoutBatchSize
        GarEpochs = $GarEpochs
        GarEvalInterval = $GarEvalInterval
        GarBatchSize = $GarBatchSize
        CCFCEpochs = $CCFCEpochs
        CCFCEvalInterval = $CCFCEvalInterval
        CCFCEmbDim = $CCFCEmbDim
        CCFCHiddenDim = $CCFCHiddenDim
        CCFCBatchSize = $CCFCBatchSize
        ALDITeacherEpochs = $ALDITeacherEpochs
        ALDIStudentEpochs = $ALDIStudentEpochs
        ALDIEvalInterval = $ALDIEvalInterval
        ALDIEmbDim = $ALDIEmbDim
        ALDIBatchSize = $ALDIBatchSize
        MARecBatchSize = $MARecBatchSize
        MARecLambdas = $MARecLambdas
        MARecAlphas = $MARecAlphas
        MARecContentBetas = $MARecContentBetas
        LightGCLEpochs = $LightGCLEpochs
        LightGCLEvalInterval = $LightGCLEvalInterval
        LightGCLEmbDim = $LightGCLEmbDim
        LightGCLBatchSize = $LightGCLBatchSize
        LightGCLLayers = $LightGCLLayers
        LightGCLSvdRank = $LightGCLSvdRank
        SageRecEpochs = $SageRecEpochs
        SageRecEvalInterval = $SageRecEvalInterval
        SageRecEmbDim = $SageRecEmbDim
        SageRecBatchSize = $SageRecBatchSize
        SageRecSampleTopN = $SageRecSampleTopN
        SageRecMaxHistLen = $SageRecMaxHistLen
        CourseMlpEpochs = $CourseMlpEpochs
        CourseMlpEvalInterval = $CourseMlpEvalInterval
        CourseMlpBatchSize = $CourseMlpBatchSize
        PollSeconds = $PollSeconds
        ContinueOnError = $true
    }
    if ($UseGpuStage -and (-not $CpuOnly)) {
        $runnerArgs["UseGpu"] = $true
    }
    if ($RerunCompleted) {
        $runnerArgs["RerunCompleted"] = $true
    }
    if ($DryRun) {
        $runnerArgs["DryRun"] = $true
    }

    Write-OvernightLog "START stage models=$($Models -join ',') | use_gpu=$($UseGpuStage -and (-not $CpuOnly))"
    & $Runner @runnerArgs
    if (($null -ne $LASTEXITCODE) -and ($LASTEXITCODE -ne 0)) {
        throw "Lightweight runner failed for models=$($Models -join ',') with exit=$LASTEXITCODE"
    }
    Write-OvernightLog "END stage models=$($Models -join ',')"
}

Write-OvernightLog "OVERNIGHT START seeds=$($Seeds -join ',') | wait_pids=$($WaitPids -join ',') | cpu_only=$($CpuOnly.IsPresent)"

if ($WaitPids.Count -gt 0) {
    Wait-ForPids $WaitPids
}

if (-not $SkipFastStage) {
    Invoke-LightweightRunner @("Popularity", "ContentProfile") $false
}

if (-not $SkipTrainStage) {
    Invoke-LightweightRunner $TrainModels $true
}

Write-OvernightLog "OVERNIGHT DONE"
