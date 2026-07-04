param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [ValidateSet("mooccube", "coco", "junyi")]
    [string]$Dataset = "mooccube",
    [int]$Seed = 2026,
    [int[]]$EpochList = @(3, 5, 10),
    [string]$EpochListCsv = "",
    [string]$OutputRoot = "",
    [int]$ColdThreshold = 1,
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
    [string]$BaseRunner = ".\run_pam_official_3datasets_3seed_wsl_gpu_serial.ps1",
    [string]$RunId = "",
    [string]$BaselineResultPath = "",
    [int]$WaitPollSeconds = 60,
    [switch]$SkipInitialWait,
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

function Epoch-OutputRoot([int]$Epoch) {
    Join-Path $OutputRootAbs ("e{0}" -f $Epoch)
}

function Epoch-ResultPath([int]$Epoch) {
    Join-Path (Join-Path (Join-Path (Epoch-OutputRoot $Epoch) $Dataset) (Split-Name $Seed)) (Join-Path $ResultSubdir "pam_official_static_result.json")
}

function Get-JsonMetric([object]$Object, [string]$Name) {
    if ($null -eq $Object) {
        return $null
    }
    $prop = $Object.PSObject.Properties[$Name]
    if (($null -eq $prop) -or ($null -eq $prop.Value)) {
        return $null
    }
    return [double]$prop.Value
}

function Convert-ResultToSummaryRow([string]$ResultPath, [string]$Source) {
    $resolved = Resolve-RunPath $Repo $ResultPath
    $data = Get-Content -Raw -LiteralPath $resolved | ConvertFrom-Json
    $entries = @($data)
    if ($entries.Count -lt 1) {
        throw "Result file has no rows: $resolved"
    }
    $entry = $entries[0]
    $coldItem = $entry.full_cold_item_macro
    $hotItem = $entry.full_hot_item_macro
    $coldInteraction = $entry.full_cold
    $hotInteraction = $entry.full_hot

    [pscustomobject][ordered]@{
        dataset = [string]$entry.manifest.data_dir
        split_dataset = $Dataset
        seed = [int]$entry.seed
        epoch = [int]$entry.epochs
        source = $Source
        training_seconds = [double]$entry.training_seconds
        last_train_loss = [double]$entry.last_train_loss
        cold_item_count = [int]$entry.count_full_cold_item_macro
        hot_item_count = [int]$entry.count_full_hot_item_macro
        cold_item_R5 = Get-JsonMetric $coldItem "R@5"
        cold_item_R10 = Get-JsonMetric $coldItem "R@10"
        cold_item_R20 = Get-JsonMetric $coldItem "R@20"
        cold_item_N5 = Get-JsonMetric $coldItem "N@5"
        cold_item_N10 = Get-JsonMetric $coldItem "N@10"
        cold_item_N20 = Get-JsonMetric $coldItem "N@20"
        hot_item_R5 = Get-JsonMetric $hotItem "R@5"
        hot_item_R10 = Get-JsonMetric $hotItem "R@10"
        hot_item_R20 = Get-JsonMetric $hotItem "R@20"
        hot_item_N5 = Get-JsonMetric $hotItem "N@5"
        hot_item_N10 = Get-JsonMetric $hotItem "N@10"
        hot_item_N20 = Get-JsonMetric $hotItem "N@20"
        cold_R20 = Get-JsonMetric $coldInteraction "R@20"
        cold_N20 = Get-JsonMetric $coldInteraction "N@20"
        hot_R20 = Get-JsonMetric $hotInteraction "R@20"
        hot_N20 = Get-JsonMetric $hotInteraction "N@20"
        result_path = $resolved
    }
}

function Write-SummaryCsv {
    $rows = New-Object System.Collections.Generic.List[object]

    if ($BaselineResultPath) {
        $baselineAbs = Resolve-RunPath $Repo $BaselineResultPath
        if (Test-Path -LiteralPath $baselineAbs) {
            $rows.Add((Convert-ResultToSummaryRow -ResultPath $baselineAbs -Source "baseline"))
        }
        else {
            Write-Host ("Baseline result not found yet, skipping summary row: {0}" -f $baselineAbs)
        }
    }

    foreach ($epoch in $EpochList) {
        $result = Epoch-ResultPath $epoch
        if (Test-Path -LiteralPath $result) {
            $rows.Add((Convert-ResultToSummaryRow -ResultPath $result -Source "sweep"))
        }
    }

    if ($rows.Count -eq 0) {
        Write-Host "No result rows available for summary yet."
        return
    }

    New-Item -ItemType Directory -Force -Path $OutputRootAbs | Out-Null
    $rows |
        Sort-Object epoch, source |
        Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $SummaryCsv
    Write-Host ("Summary CSV written: {0}" -f $SummaryCsv)
}

function Get-ExistingBaseRunnerProcesses {
    $runnerName = [System.IO.Path]::GetFileName($BaseRunnerAbs)
    @(Get-CimInstance Win32_Process |
        Where-Object {
            ($_.ProcessId -ne $PID) -and
            ($_.CommandLine -match [regex]::Escape($runnerName))
        } |
        Select-Object ProcessId, CommandLine)
}

function Wait-ForExistingBaseRunnerIdle {
    if ($DryRun) {
        return
    }
    if ($SkipInitialWait) {
        Write-Host "SkipInitialWait: not waiting for existing PAM runner processes."
        return
    }

    while ($true) {
        $running = @(Get-ExistingBaseRunnerProcesses)
        if ($running.Count -eq 0) {
            Write-Host "No existing PAM runner process detected."
            return
        }
        $summary = ($running | ForEach-Object { "pid=$($_.ProcessId)" }) -join ", "
        Write-Host ("WAIT existing PAM runner process(es): {0}" -f $summary)
        Start-Sleep -Seconds $WaitPollSeconds
    }
}

function Invoke-EpochRun([int]$Epoch) {
    $epochRoot = Epoch-OutputRoot $Epoch
    $epochRunId = if ($RunId) {
        "{0}_e{1}" -f $RunId, $Epoch
    }
    else {
        "epoch_sweep_{0}_seed{1}_e{2}_{3}" -f $Dataset, $Seed, $Epoch, (Get-Date -Format "yyyyMMdd_HHmmss")
    }

    Write-Host ("SWEEP TASK epoch={0} outputRoot={1} runId={2}" -f $Epoch, $epochRoot, $epochRunId)

    $childArgs = @{
        Repo = $Repo
        OutputRoot = $epochRoot
        Datasets = @($Dataset)
        Seeds = @($Seed)
        ColdThreshold = $ColdThreshold
        Epochs = $Epoch
        BatchSize = $BatchSize
        LearningRate = $LearningRate
        EmbDim = $EmbDim
        HiddenDim = $HiddenDim
        CateDim = $CateDim
        NegPerPos = $NegPerPos
        MaxTrainPos = $MaxTrainPos
        MaxEvalRows = $MaxEvalRows
        EvalItemBatchSize = $EvalItemBatchSize
        MaxCatesPerItem = $MaxCatesPerItem
        PamRoot = $PamRoot
        ResultSubdir = $ResultSubdir
        WslPython = $WslPython
        ProjectPythonRunner = $ProjectPythonRunner
        RunId = $epochRunId
        WaitPollSeconds = $WaitPollSeconds
        SkipAggregate = $true
    }
    if ($SkipInitialWait) {
        $childArgs.SkipInitialWait = $true
    }
    if ($Force) {
        $childArgs.Force = $true
    }
    if ($DryRun) {
        $childArgs.DryRun = $true
    }

    & $BaseRunnerAbs @childArgs

    if (-not $DryRun) {
        $result = Epoch-ResultPath $Epoch
        if (-not (Test-Path -LiteralPath $result)) {
            throw "Expected result not found for epoch=$Epoch`: $result"
        }
        Write-SummaryCsv
    }
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

if ($EpochList.Count -lt 1) {
    throw "EpochList must contain at least one epoch."
}
if ($EpochListCsv) {
    $parsedEpochs = @()
    foreach ($part in ($EpochListCsv -split '[,;\s]+')) {
        $trimmed = $part.Trim()
        if (-not $trimmed) {
            continue
        }
        $value = 0
        if (-not [int]::TryParse($trimmed, [ref]$value)) {
            throw "EpochListCsv contains a non-integer value: $trimmed"
        }
        $parsedEpochs += $value
    }
    if ($parsedEpochs.Count -lt 1) {
        throw "EpochListCsv did not contain any epoch values."
    }
    $EpochList = @($parsedEpochs)
}
foreach ($epoch in $EpochList) {
    if ($epoch -lt 1) {
        throw "Epoch values must be positive integers: $epoch"
    }
}

if (-not $OutputRoot) {
    $OutputRoot = "outputs\pam_official_epoch_sweep_{0}_seed{1}" -f $Dataset, $Seed
}

$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$BaseRunnerAbs = Resolve-RunPath $Repo $BaseRunner
$SummaryCsv = Join-Path $OutputRootAbs "pam_epoch_sweep_summary.csv"

if (-not (Test-Path -LiteralPath $BaseRunnerAbs)) {
    throw "Missing base PAM runner: $BaseRunnerAbs"
}

Write-Host ("Total sweep tasks: {0}" -f $EpochList.Count)
Write-Host ("Dataset: {0}" -f $Dataset)
Write-Host ("Seed: {0}" -f $Seed)
Write-Host ("Epochs: {0}" -f (($EpochList | ForEach-Object { "$_" }) -join ","))
Write-Host ("Output root: {0}" -f $OutputRootAbs)
Write-Host ("Base runner: {0}" -f $BaseRunnerAbs)
if ($BaselineResultPath) {
    Write-Host ("Baseline result: {0}" -f (Resolve-RunPath $Repo $BaselineResultPath))
}
Write-Host ("Summary CSV: {0}" -f $SummaryCsv)

Wait-ForExistingBaseRunnerIdle

foreach ($epoch in $EpochList) {
    Invoke-EpochRun $epoch
}

if (-not $DryRun) {
    Write-SummaryCsv
}
