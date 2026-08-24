param(
    [string[]]$Seeds = @("2026", "2027"),
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\junyi\main_table_3seed",
    [string]$CheckpointRoot = "checkpoints\junyi\main_table_3seed",
    [string]$LowCostQueueLog = "",
    [int]$WaitPid = 0,
    [int]$PollSeconds = 300,
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

$ParsedSeeds = @()
foreach ($rawSeed in $Seeds) {
    foreach ($part in ([string]$rawSeed -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) {
            $ParsedSeeds += [int]$trimmed
        }
    }
}
$Seeds = [int[]]$ParsedSeeds

$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
if (-not $LowCostQueueLog) {
    $LowCostQueueLog = Join-Path $OutputRootAbs "_queue_lowcost\queue.log"
} elseif (-not [System.IO.Path]::IsPathRooted($LowCostQueueLog)) {
    $LowCostQueueLog = Join-Path $Repo $LowCostQueueLog
}

$QueueDir = Join-Path $OutputRootAbs "_queue_heavy"
$QueueLog = Join-Path $QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootAbs | Out-Null

function Write-QueueLogLine([string]$Path, [string]$Line) {
    $payload = $Line + [Environment]::NewLine
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            [System.IO.File]::AppendAllText($Path, $payload, [System.Text.Encoding]::UTF8)
            return
        } catch {
            if ($attempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds (200 * $attempt)
        }
    }
}

function Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-QueueLogLine $QueueLog $line
    Write-Host $line
}

function Split-Dir([int]$Seed) {
    Join-Path $OutputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
}

function Assert-Split([int]$Seed) {
    $splitDir = Split-Dir $Seed
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split artifact for seed=${Seed}: $path"
        }
    }
    return $splitDir
}

function Test-LowCostDone([string]$LogPath) {
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }

    $lines = Get-Content -LiteralPath $LogPath -Encoding UTF8
    $lastStart = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "QUEUE START Junyi main-table low-cost") {
            $lastStart = $i
        }
    }
    if ($lastStart -lt 0) {
        return $false
    }

    for ($i = $lastStart + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "QUEUE DONE Junyi main-table low-cost") {
            return $true
        }
    }
    return $false
}

function Wait-LowCostQueue {
    if ($DryRun) {
        if (-not (Test-LowCostDone $LowCostQueueLog)) {
            throw "DryRun requires completed low-cost queue log: $LowCostQueueLog"
        }
        Log "LOWCOST DONE detected | log=$LowCostQueueLog"
        return
    }

    if ($WaitPid -gt 0) {
        Log "WAIT low-cost PID=$WaitPid"
        while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
            Start-Sleep -Seconds $PollSeconds
        }
        Log "LOWCOST PID exited | pid=$WaitPid"
    }

    Log "WAIT low-cost QUEUE DONE | log=$LowCostQueueLog"
    while (-not (Test-LowCostDone $LowCostQueueLog)) {
        Start-Sleep -Seconds $PollSeconds
    }
    Log "LOWCOST DONE detected | log=$LowCostQueueLog"
}

function Set-CommonEnv([int]$Seed, [string]$OutDir, [string]$CkptDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if ($CkptDir) {
        New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
    }

    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = "processed_data_junyi"
    $env:USIM_RELATION_DIR = "processed_data_junyi\relations"
    $env:USIM_STATIC_SPLIT_DIR = Split-Dir $Seed
    $env:USIM_BASELINE_OUTPUT_DIR = $OutDir
    $env:USIM_STATIC_SEED = "$Seed"
    $env:USIM_SEED = "$Seed"
    $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_EVAL_N_NEG = "200"
    $env:USIM_EARLY_STOP_AVG_MODE = "item_macro"
    $env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
    $env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
    $env:BASELINE_BEST_METRIC = "cold"
    $env:BASELINE_SAVE_CKPT = "1"
    $env:BASELINE_SAVE_OPT_STATE = "1"
    $env:BASELINE_AUTO_RESUME = "0"
    $env:BASELINE_FORCE_FRESH = "1"
    $env:BASELINE_CKPT_DIR = $CkptDir
}

function Run-Model(
    [int]$Seed,
    [string]$Name,
    [string]$ScriptName,
    [string]$OutSubdir,
    [string]$ResultFile,
    [scriptblock]$Configure,
    [string]$CkptSubdir
) {
    $splitDir = if ($DryRun) { Split-Dir $Seed } else { Assert-Split $Seed }
    $outDir = Join-Path $splitDir $OutSubdir
    $ckptDir = Join-Path $CheckpointRootAbs ("{0}\strict_item_cold_balanced_thr1_seed_{1}" -f $CkptSubdir, $Seed)
    $resultPath = Join-Path $outDir $ResultFile

    if ((-not $DryRun) -and (Test-Path -LiteralPath $resultPath)) {
        Log "SKIP $Name seed=$Seed | exists=$resultPath"
        return
    }
    if ($DryRun) {
        Log "DRYRUN START $Name seed=$Seed | out=$outDir | ckpt=$ckptDir"
        return
    }

    Set-CommonEnv $Seed $outDir $ckptDir
    & $Configure

    $logPath = Join-Path $outDir "run.log"
    Log "START $Name seed=$Seed | out=$outDir | ckpt=$ckptDir"
    $cmd = '/d /c "".\py.bat" -u "' + $ScriptName + '" > "' + $logPath + '" 2>&1"'
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
    Log "END $Name seed=$Seed | exit=$($p.ExitCode) | log=$logPath"
    if ($p.ExitCode -ne 0) {
        throw "$Name failed for seed=$Seed with exit=$($p.ExitCode)"
    }
}

function Run-Ours([int]$Seed) {
    $splitDir = if ($DryRun) { Split-Dir $Seed } else { Assert-Split $Seed }
    $resultPath = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $ckptRoot = Join-Path $CheckpointRootAbs "ours"
    $ckptDir = Join-Path $ckptRoot ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)

    if ((-not $DryRun) -and (Test-Path -LiteralPath $resultPath)) {
        Log "SKIP Ours seed=$Seed | exists=$resultPath"
        return
    }
    if ($DryRun) {
        Log "DRYRUN START Ours seed=$Seed | out=$splitDir | ckpt=$ckptDir"
        return
    }

    New-Item -ItemType Directory -Force -Path $ckptRoot | Out-Null
    $logPath = Join-Path $splitDir "run.log"
    $runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"

    Log "START Ours seed=$Seed | out=$splitDir | ckpt=$ckptDir"
    $exitCode = 0
    try {
        & $runner `
            -PythonRunner ".\py.bat" `
            -DataDir "processed_data_junyi" `
            -RelationDir "processed_data_junyi\relations" `
            -OutputRoot $OutputRootAbs `
            -CheckpointRoot $ckptRoot `
            -Protocol strict_item_cold_balanced `
            -ColdThresholds 1 `
            -Seeds $Seed `
            -Epochs 60 `
            -Patience 60 `
            -EarlyStopAverageMode item_macro `
            -UseContentDelta:$false `
            -UsePseudoColdTrain:$false `
            -RunSampledEval:$false `
            -SaveCkpt:$true `
            -AutoResume:$false `
            -ForceFresh:$true `
            -SaveOptState:$true `
            -SkipAggregate
        if ($null -ne $LASTEXITCODE) {
            $exitCode = $LASTEXITCODE
        }
    } catch {
        $exitCode = 1
        $_ | Out-File -FilePath $logPath -Append -Encoding UTF8
    }
    Log "END Ours seed=$Seed | exit=$exitCode | log=$logPath"
    if ($exitCode -ne 0) {
        throw "Ours failed for seed=$Seed with exit=$exitCode"
    }
}

Log "QUEUE START Junyi main-table heavy seeds=$($Seeds -join ',')"
Wait-LowCostQueue

foreach ($seed in $Seeds) {
    Run-Model $seed "ALDI" "aldi_static_hin.py" "aldi_compare" "aldi_static_result.json" {
        $env:ALDI_TEACHER_EPOCHS = "200"
        $env:ALDI_TEACHER_EVAL_INTERVAL = "20"
        $env:ALDI_TEACHER_BATCH_SIZE = "4096"
        $env:ALDI_STATIC_EPOCHS = "100"
        $env:ALDI_EVAL_INTERVAL = "5"
        $env:ALDI_BATCH_SIZE = "4096"
        $env:ALDI_EVAL_BATCH_SIZE = "4096"
        $env:ALDI_EVAL_N_NEG = "200"
        $env:ALDI_COLD_THRESHOLD = "1"
        $env:ALDI_STATIC_SEED = "$seed"
        $env:ALDI_SEED = "$seed"
        $env:ALDI_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:ALDI_TEACHER_CKPT_DIR = Join-Path $env:BASELINE_CKPT_DIR "teacher"
        $env:ALDI_SAVE_CKPT = "1"
        $env:ALDI_SAVE_OPT_STATE = "1"
        $env:ALDI_AUTO_RESUME = "0"
        $env:ALDI_FORCE_FRESH = "1"
    } "aldi_compare"
}

foreach ($seed in $Seeds) {
    Run-Model $seed "CGRC-paper" "cgrc_paper_static_hin.py" "cgrc_paper_compare" "cgrc_paper_static_result.json" {
        $env:CGRC_PAPER_STATIC_EPOCHS = "50"
        $env:CGRC_PAPER_BATCH_SIZE = "4096"
        $env:CGRC_PAPER_EVAL_N_NEG = "200"
        $env:CGRC_PAPER_COLD_THRESHOLD = "1"
        $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
        $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
        $env:CGRC_PAPER_MASK_RHO = "0.3"
        $env:CGRC_PAPER_RECON_TOPK = "20"
        $env:CGRC_PAPER_LAMBDA_E = "1.0"
        $env:CGRC_PAPER_TAU = "0.5"
        $env:CGRC_PAPER_STATIC_SEED = "$seed"
        $env:CGRC_PAPER_SEED = "$seed"
        $env:CGRC_PAPER_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:CGRC_PAPER_SAVE_CKPT = "1"
        $env:CGRC_PAPER_SAVE_OPT_STATE = "1"
        $env:CGRC_PAPER_AUTO_RESUME = "0"
        $env:CGRC_PAPER_FORCE_FRESH = "1"
    } "cgrc_paper_compare"
}

foreach ($seed in $Seeds) {
    Run-Ours $seed
}

if (-not $DryRun) {
    $summaryLog = Join-Path $QueueDir "aggregate_fast3.log"
    Log "START aggregate Ours | root=$OutputRootAbs"
    $cmd = '/d /c "".\py.bat" -B "aggregate_fast3_static_results.py" --root "' + $OutputRootAbs + '" > "' + $summaryLog + '" 2>&1"'
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
    Log "END aggregate Ours | exit=$($p.ExitCode) | log=$summaryLog"
    if ($p.ExitCode -ne 0) {
        throw "Ours aggregation failed with exit=$($p.ExitCode)"
    }
}

Log "QUEUE DONE Junyi main-table heavy seeds=$($Seeds -join ',')"
