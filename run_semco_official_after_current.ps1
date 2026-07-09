param(
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$SplitRoot = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$OutputRoot = "outputs\content_delta_pop5\semco_official_v1",
    [ValidateSet("mooccube", "junyi", "coco", "all")]
    [string[]]$Datasets = @("mooccube"),
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [ValidateSet("sparsemax", "entmax15", "softmax")]
    [string]$Fn = "sparsemax",
    [double]$SmScale = 12.0,
    [double]$Reg = 0.001,
    [int]$Epochs = 5,
    [int]$BatchSize = 512,
    [int]$EvalBatchSize = 8192,
    [int]$ProgressInterval = 300,
    [ValidateSet("interaction", "item_macro")]
    [string]$EarlyStopAverageMode = "item_macro",
    [string]$WaitPattern = "cgrc_paper_static_hin.py|usim_feedback_fast3_content_delta.py|semco_official_static_hin.py|semco_static_hin.py",
    [int]$PollSeconds = 60,
    [switch]$NoWait,
    [switch]$SkipExisting,
    [switch]$Aggregate
)

$ErrorActionPreference = "Stop"

function Get-MatchingProcessSnapshot {
    param([string]$Pattern)
    $selfPid = $PID
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $selfPid -and
            $_.CommandLine -and
            $_.CommandLine -match $Pattern -and
            $_.CommandLine -notmatch "run_semco_official_after_current.ps1"
        } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

function Wait-ForSnapshot {
    param(
        [array]$Snapshot,
        [int]$SleepSeconds
    )
    if (-not $Snapshot -or $Snapshot.Count -eq 0) {
        Write-Host "No matching running experiments found at startup."
        return
    }

    $ids = @($Snapshot | ForEach-Object { [int]$_.ProcessId })
    Write-Host "Waiting for startup experiment PIDs: $($ids -join ', ')"
    foreach ($proc in $Snapshot) {
        Write-Host ("  PID={0} Name={1} Cmd={2}" -f $proc.ProcessId, $proc.Name, $proc.CommandLine)
    }

    while ($true) {
        $alive = @()
        foreach ($id in $ids) {
            $p = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($p) {
                $alive += $id
            }
        }
        if ($alive.Count -eq 0) {
            Write-Host "Startup experiments finished."
            return
        }
        Write-Host ("Still waiting: {0} | next check in {1}s | {2}" -f ($alive -join ', '), $SleepSeconds, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        Start-Sleep -Seconds $SleepSeconds
    }
}

function Resolve-DatasetConfig {
    param([string]$Dataset)

    switch ($Dataset.ToLowerInvariant()) {
        "mooccube" {
            return [PSCustomObject]@{
                Name = "mooccube"
                DataDir = $DataDir
                SplitRoot = $SplitRoot
                OutputRoot = $OutputRoot
            }
        }
        "junyi" {
            return [PSCustomObject]@{
                Name = "junyi"
                DataDir = "processed_data_junyi"
                SplitRoot = "outputs\junyi\main_table_3seed"
                OutputRoot = "outputs\junyi\semco_official_v1"
            }
        }
        "coco" {
            return [PSCustomObject]@{
                Name = "coco"
                DataDir = "processed_data_coco"
                SplitRoot = "outputs\coco\single_seed_triage\ours_full"
                OutputRoot = "outputs\coco\semco_official_v1"
            }
        }
        default {
            throw "Unknown dataset: $Dataset"
        }
    }
}

function Invoke-SemcoOfficialSeed {
    param(
        [object]$DatasetConfig,
        [int]$Seed,
        [string]$Tag
    )

    $splitName = "strict_item_cold_balanced_thr${ColdThreshold}_seed_$Seed"
    $splitDir = Join-Path $DatasetConfig.SplitRoot $splitName
    if (-not (Test-Path $splitDir)) {
        throw "Missing split directory: $splitDir"
    }

    $outDir = Join-Path (Join-Path $DatasetConfig.OutputRoot $splitName) $Tag
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $logPath = Join-Path $outDir "run.log"
    $resultPath = Join-Path $outDir "semco_official_static_result.json"

    if ($SkipExisting -and (Test-Path $resultPath)) {
        Write-Host "Skip seed=$Seed because result exists: $resultPath"
        return "skipped"
    }

    $env:USIM_DATA_DIR = $DatasetConfig.DataDir
    $env:USIM_STATIC_SPLIT_DIR = $splitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $outDir
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_COLD_THRESHOLD = [string]$ColdThreshold
    $env:USIM_EVAL_N_NEG = "200"
    $env:USIM_EARLY_STOP_AVG_MODE = $EarlyStopAverageMode

    $env:SEMCO_OFFICIAL_STATIC_SEED = [string]$Seed
    $env:SEMCO_OFFICIAL_SEED = [string]$Seed
    $env:SEMCO_OFFICIAL_FN = $Fn
    $env:SEMCO_OFFICIAL_SM_SCALE = [string]$SmScale
    $env:SEMCO_OFFICIAL_REG = [string]$Reg
    $env:SEMCO_OFFICIAL_EPOCHS = [string]$Epochs
    $env:SEMCO_OFFICIAL_BATCH_SIZE = [string]$BatchSize
    $env:SEMCO_OFFICIAL_EVAL_BATCH_SIZE = [string]$EvalBatchSize
    $env:SEMCO_OFFICIAL_PROGRESS_INTERVAL = [string]$ProgressInterval

    Write-Host ""
    Write-Host "SEMCo official-code adapted dataset=$($DatasetConfig.Name) seed=$Seed"
    Write-Host "  split=$splitDir"
    Write-Host "  out=$outDir"
    Write-Host "  data=$($DatasetConfig.DataDir)"
    Write-Host "  fn=$Fn scale=$SmScale reg=$Reg epochs=$Epochs batch=$BatchSize"

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonRunner -u semco_official_static_hin.py 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -ne 0) {
        Write-Host "SEMCo official failed for seed=$Seed, exit=$exitCode"
        if (Test-Path $logPath) {
            Get-Content $logPath -Tail 120
        }
        throw "SEMCo official failed for seed=$Seed"
    }
    if (-not (Test-Path $resultPath)) {
        throw "SEMCo official finished but result is missing: $resultPath"
    }
    return "completed"
}

$tagFn = $Fn.Replace(".", "p")
$scaleTag = ("scale{0}" -f $SmScale).Replace(".", "p")
$tag = "${tagFn}_${scaleTag}_e${Epochs}_b${BatchSize}"
$queueRoot = Join-Path $OutputRoot "_queue_logs"
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
$queueLog = Join-Path $queueRoot ("semco_official_after_current_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Start-Transcript -Path $queueLog -Append | Out-Null
try {
    Write-Host "SEMCo official queue started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Tag=$tag"
    $datasetNames = @()
    foreach ($dataset in $Datasets) {
        if ($dataset -eq "all") {
            $datasetNames += @("mooccube", "junyi", "coco")
        } else {
            $datasetNames += $dataset
        }
    }
    $datasetNames = @($datasetNames | Select-Object -Unique)
    Write-Host "Datasets=$($datasetNames -join ', ')"
    Write-Host "Seeds=$($Seeds -join ', ')"
    Write-Host "Aggregate=$([bool]$Aggregate)"
    if (-not $NoWait) {
        $snapshot = @(Get-MatchingProcessSnapshot -Pattern $WaitPattern)
        Wait-ForSnapshot -Snapshot $snapshot -SleepSeconds $PollSeconds
    } else {
        Write-Host "NoWait set: starting immediately."
    }

    foreach ($datasetName in $datasetNames) {
        $datasetConfig = Resolve-DatasetConfig -Dataset $datasetName
        Write-Host ""
        Write-Host "=== Dataset: $($datasetConfig.Name) ==="
        foreach ($seed in $Seeds) {
            Invoke-SemcoOfficialSeed -DatasetConfig $datasetConfig -Seed $seed -Tag $tag
        }

        if ($Aggregate) {
            $summaryDir = Join-Path $datasetConfig.OutputRoot ("summary_{0}" -f $tag)
            & $PythonRunner aggregate_main_table_static_results.py `
                --root $datasetConfig.OutputRoot `
                --split-glob "strict_item_cold_balanced_thr1_seed_*" `
                --result-subdir $tag `
                --metric-mode item_macro `
                --out-dir $summaryDir
            if ($LASTEXITCODE -ne 0) {
                throw "Aggregation failed for dataset=$($datasetConfig.Name)"
            }
        }
    }
    Write-Host "SEMCo official queue completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} finally {
    Stop-Transcript | Out-Null
}
