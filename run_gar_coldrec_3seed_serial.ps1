param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputRoot = "paper_aaai27\baseline_sources\_gar_coldrec_strict\mooccube_source_default_3seed",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$MFEpochs = 500,
    [int]$GAREpochs = 500,
    [int]$EmbSize = 64,
    [int]$BatchSize = 4096,
    [double]$MFLearningRate = 0.001,
    [double]$MFReg = 0.0001,
    [double]$GARLearningRate = 0.001,
    [double]$GARReg = 0.0001,
    [double]$Alpha = 0.05,
    [double]$Beta = 0.1,
    [int]$EarlyStop = 5,
    [int]$EvalEvery = 1,
    [int]$EvalBatchSize = 2048,
    [string]$TopN = "5,10,20",
    [string]$ColdRecRoot = "tmp\candidate_repos\ColdRec",
    [int]$GpuId = 0,
    [bool]$UseGpu = $true,
    [string]$RunId = "",
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Path))
}

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        Add-Content -LiteralPath $MasterLog -Encoding UTF8 -Value $line
    }
    Write-Host $line
}

$expectedSeeds = "2025,2026,2027"
$actualSeeds = $Seeds -join ","
if ($actualSeeds -ne $expectedSeeds) {
    throw "Formal GAR queue requires seeds=$expectedSeeds in that order; got $actualSeeds"
}
if ($MFEpochs -ne 500 -or $GAREpochs -ne 500) {
    throw "Formal GAR queue requires source-default ceilings MF=500 and GAR=500."
}
if (-not $UseGpu) {
    throw "Formal GAR queue requires CUDA; UseGpu must be true."
}
if ($RunId -and ($RunId -match '[\\/:*?"<>|]')) {
    throw "RunId contains invalid filename characters: $RunId"
}

$RepoAbs = Resolve-RunPath (Get-Location).Path $Repo
$PythonRunnerAbs = Resolve-RunPath $RepoAbs $PythonRunner
$OutputRootAbs = Resolve-RunPath $RepoAbs $OutputRoot
$SingleRunner = Join-Path $RepoAbs "run_gar_coldrec_single_seed.ps1"
$Aggregator = Join-Path $RepoAbs "aggregate_gar_coldrec_3seed.py"
$AggregateDir = Join-Path $OutputRootAbs "aggregate"
$Timestamp = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$LogDir = Join-Path (Join-Path $OutputRootAbs "_logs") $Timestamp
$MasterLog = Join-Path $LogDir "queue.log"
$AggregateLog = Join-Path $LogDir "aggregate.log"

Write-Host "ColdRec GAR strict source-default three-seed"
Write-Host "seeds=$actualSeeds"
Write-Host "single_runner=$(Split-Path -Leaf $SingleRunner)"
Write-Host "aggregator=$(Split-Path -Leaf $Aggregator)"
Write-Host "MF epochs=$MFEpochs"
Write-Host "GAR epochs=$GAREpochs"
Write-Host "early_stop=$EarlyStop"
Write-Host "use_gpu=$UseGpu gpu_id=$GpuId"
Write-Host "output_root=$OutputRootAbs"
foreach ($seed in $Seeds) {
    $splitName = "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $seed
    $seedOutput = Join-Path $OutputRootAbs ("seed_{0}" -f $seed)
    Write-Host "TASK seed=$seed split=$splitName output=$seedOutput"
}
Write-Host "aggregate=gar_coldrec_3seed_detail.csv"
Write-Host "aggregate=gar_coldrec_3seed_summary.csv"
Write-Host "aggregate=gar_coldrec_3seed_summary.json"
Write-Host "aggregate=gar_coldrec_3seed_report.md"
Write-Host "aggregate_log=$(Split-Path -Leaf $AggregateLog)"

if ($DryRun) {
    Write-Host "DRY RUN: no training commands were executed."
    return
}

foreach ($path in @($RepoAbs, $PythonRunnerAbs, $SingleRunner, $Aggregator)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required path: $path"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRootAbs, $LogDir | Out-Null

Write-QueueLog ((
    "QUEUE START GAR source-default MOOCCube 3seed | seeds={0} | MF epochs={1} | " +
    "GAR epochs={2} | early_stop={3} | gpu={4} | force={5}"
) -f $actualSeeds, $MFEpochs, $GAREpochs, $EarlyStop, $GpuId, [bool]$Force)

try {
    foreach ($seed in $Seeds) {
        $seedOutput = Join-Path $OutputRootAbs ("seed_{0}" -f $seed)
        $resultPath = Join-Path $seedOutput "gar_coldrec_strict_result.json"
        if ((-not $Force) -and (Test-Path -LiteralPath $resultPath)) {
            Write-QueueLog "SKIP seed=$seed | exists=$resultPath"
            continue
        }

        Write-QueueLog "START seed=$seed | output=$seedOutput"
        & $SingleRunner `
            -Repo $RepoAbs `
            -PythonRunner $PythonRunnerAbs `
            -OutputDir $seedOutput `
            -Seed $seed `
            -ColdThreshold $ColdThreshold `
            -MFEpochs $MFEpochs `
            -GAREpochs $GAREpochs `
            -EmbSize $EmbSize `
            -BatchSize $BatchSize `
            -MFLearningRate $MFLearningRate `
            -MFReg $MFReg `
            -GARLearningRate $GARLearningRate `
            -GARReg $GARReg `
            -Alpha $Alpha `
            -Beta $Beta `
            -EarlyStop $EarlyStop `
            -EvalEvery $EvalEvery `
            -EvalBatchSize $EvalBatchSize `
            -TopN $TopN `
            -ColdRecRoot $ColdRecRoot `
            -GpuId $GpuId `
            -UseGpu $UseGpu `
            -Force:$Force

        if (-not (Test-Path -LiteralPath $resultPath)) {
            throw "Seed $seed completed without expected result: $resultPath"
        }
        Write-QueueLog "END seed=$seed | result=$resultPath"
    }

    Write-QueueLog "START aggregate | output=$AggregateDir"
    & $PythonRunnerAbs -u $Aggregator `
        --root $OutputRootAbs `
        --seeds $actualSeeds `
        --out-dir $AggregateDir *> $AggregateLog
    $aggregateExit = $LASTEXITCODE
    if ($aggregateExit -ne 0) {
        throw "GAR three-seed aggregation failed with exit=$aggregateExit. See $AggregateLog"
    }

    foreach ($name in @(
        "gar_coldrec_3seed_detail.csv",
        "gar_coldrec_3seed_summary.csv",
        "gar_coldrec_3seed_summary.json",
        "gar_coldrec_3seed_report.md"
    )) {
        $path = Join-Path $AggregateDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Aggregation completed without expected artifact: $path"
        }
    }
    Write-QueueLog "QUEUE DONE GAR source-default MOOCCube 3seed | aggregate=$AggregateDir"
}
catch {
    Write-QueueLog "QUEUE FAIL | $($_.Exception.Message)"
    throw
}

Write-Host "GAR ColdRec source-default three-seed complete: $AggregateDir"
