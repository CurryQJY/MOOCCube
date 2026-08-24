param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$DataDir = "processed_data_xes3g5m",
    [string]$RelationDir = "processed_data_xes3g5m\relations",
    [string]$OutputRoot = "outputs\xes3g5m\ours_sota_serial",
    [string]$CheckpointRoot = "checkpoints\xes3g5m\ours_sota_serial",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [int]$OursBatchSize = 2048,
    [int]$ContentProfileBatchSize = 2048,
    [int]$CgrcEpochs = 50,
    [int]$CgrcBatchSize = 2048,
    [int]$CgrcReconTopK = 20,
    [int]$CgrcReconUserChunk = 2048,
    [bool]$MaskKnownPosNeg = $true,
    [bool]$MaskSameItemNeg = $true,
    [bool]$ForceFresh = $true,
    [bool]$AutoResume = $false,
    [switch]$SkipOursFull,
    [switch]$SkipNoCourse,
    [switch]$SkipContentProfile,
    [switch]$SkipCgrc,
    [switch]$SkipAggregate,
    [switch]$RerunCompleted,
    [switch]$AllowConcurrent,
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
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$StaticRunnerAbs = Resolve-RunPath $Repo $StaticRunner
$QueueDir = Join-Path $OutputRootAbs "_queue"
$QueueLog = Join-Path $QueueDir "queue.log"

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
    if (-not $DryRun) {
        Write-QueueLogLine $QueueLog $line
    }
    Write-Host $line
}

function Split-Name([int]$Seed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $Seed
}

function Ours-Full-Root {
    Join-Path $OutputRootAbs "ours_full"
}

function Ours-Full-SplitDir([int]$Seed) {
    Join-Path (Ours-Full-Root) (Split-Name $Seed)
}

function NoCourse-Root {
    Join-Path $OutputRootAbs "wo_all_course_signals"
}

function NoCourse-SplitDir([int]$Seed) {
    Join-Path (NoCourse-Root) (Split-Name $Seed)
}

function Assert-Inputs {
    if (-not (Test-Path -LiteralPath $StaticRunnerAbs)) {
        throw "Missing static runner: $StaticRunnerAbs"
    }
    foreach ($path in @($DataDir, $RelationDir)) {
        $abs = Resolve-RunPath $Repo $path
        if (-not (Test-Path -LiteralPath $abs)) {
            throw "Missing required path: $abs"
        }
    }
}

function Assert-NoConcurrentTraining {
    if ($AllowConcurrent -or $DryRun) {
        return
    }
    $pattern = "usim_feedback_fast3_content_delta.py|cgrc_paper_static_hin.py|aldi_static_hin.py|dropout|lightgcn_static|bpr_static"
    $running = @(
        Get-CimInstance Win32_Process |
            Where-Object { $_.CommandLine -match $pattern } |
            Where-Object { $_.ProcessId -ne $PID }
    )
    if ($running.Count -gt 0) {
        $summary = ($running | Select-Object ProcessId, Name, CreationDate, CommandLine | Format-List | Out-String)
        throw "Refusing to start while training/baseline processes are running. Use -AllowConcurrent to override.`n$summary"
    }
}

function Assert-SharedSplit([int]$Seed) {
    $splitDir = Ours-Full-SplitDir $Seed
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing shared split artifact for seed=${Seed}: $path. Run OursFull first or remove -SkipOursFull."
        }
    }
    return $splitDir
}

function Show-DryRunHeader {
    Write-Host "DataDir=$DataDir"
    Write-Host "RelationDir=$RelationDir"
    Write-Host "Protocol=strict_item_cold_balanced"
    Write-Host "Seeds=$($Seeds -join ',')"
    Write-Host "Epochs=$Epochs"
    Write-Host "Patience=$Patience"
    Write-Host "EvalNeg=$EvalNeg"
    Write-Host "UseContentDelta=False"
    Write-Host "PrereqGraphSource=concept"
    Write-Host "MaskKnownPosNeg=$MaskKnownPosNeg"
    Write-Host "MaskSameItemNeg=$MaskSameItemNeg"
    Write-Host "OutputRoot=$OutputRootAbs"
    Write-Host "CheckpointRoot=$CheckpointRootAbs"
}

function Invoke-Fast3Variant(
    [int]$Seed,
    [string]$VariantName,
    [string]$OutRoot,
    [string]$CkptRoot,
    [hashtable]$ExtraParams
) {
    $splitDir = Join-Path $OutRoot (Split-Name $Seed)
    $finalPath = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if ((-not $DryRun) -and (Test-Path -LiteralPath $finalPath) -and (-not $RerunCompleted)) {
        Log "SKIP $VariantName seed=$Seed | exists=$finalPath"
        return $splitDir
    }

    if ($DryRun) {
        Write-Host "Plan=$VariantName"
        Write-Host "$VariantName.OutputRoot=$OutRoot"
        Write-Host "$VariantName.CheckpointRoot=$CkptRoot"
        foreach ($key in ($ExtraParams.Keys | Sort-Object)) {
            Write-Host "$VariantName.$key=$($ExtraParams[$key])"
        }
        return $splitDir
    }

    New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $CkptRoot | Out-Null

    $oldBatchSize = [Environment]::GetEnvironmentVariable("USIM_BATCH_SIZE", "Process")
    try {
        $env:USIM_BATCH_SIZE = [string]$OursBatchSize
        $runnerParams = @{
            PythonRunner = $PythonRunner
            DataDir = $DataDir
            RelationDir = $RelationDir
            OutputRoot = $OutRoot
            CheckpointRoot = $CkptRoot
            Protocol = "strict_item_cold_balanced"
            ColdThresholds = @($ColdThreshold)
            Seeds = @($Seed)
            Epochs = $Epochs
            Patience = $Patience
            EarlyStopAverageMode = "item_macro"
            UseContentDelta = $false
            UsePseudoColdTrain = $false
            UsePaac = $false
            RunSampledEval = $false
            PrereqGraphSource = "concept"
            MaskKnownPosNeg = $MaskKnownPosNeg
            MaskSameItemNeg = $MaskSameItemNeg
            SaveCkpt = $true
            AutoResume = $AutoResume
            ForceFresh = $ForceFresh
            SaveOptState = $true
            SkipAggregate = $true
        }
        foreach ($key in $ExtraParams.Keys) {
            $runnerParams[$key] = $ExtraParams[$key]
        }

        Log "START $VariantName seed=$Seed | out=$splitDir"
        & $StaticRunnerAbs @runnerParams
        if ($LASTEXITCODE -ne 0) {
            throw "$VariantName failed for seed=$Seed with exit=$LASTEXITCODE"
        }
        Log "END $VariantName seed=$Seed | out=$splitDir"
    } finally {
        if ($null -eq $oldBatchSize) {
            Remove-Item Env:USIM_BATCH_SIZE -ErrorAction SilentlyContinue
        } else {
            $env:USIM_BATCH_SIZE = $oldBatchSize
        }
    }
    return $splitDir
}

$baselineEnvNames = @(
    "PYTHONUNBUFFERED",
    "USIM_DATA_DIR",
    "USIM_RELATION_DIR",
    "USIM_STATIC_SPLIT_DIR",
    "USIM_BASELINE_OUTPUT_DIR",
    "USIM_STATIC_SEED",
    "USIM_SEED",
    "USIM_COLD_THRESHOLD",
    "USIM_STATIC_TEST_HISTORY",
    "USIM_EVAL_N_NEG",
    "USIM_RUN_SAMPLED_EVAL",
    "BASELINE_EARLY_STOP_AVG_MODE",
    "BASELINE_EARLY_STOP_AVERAGE_MODE",
    "BASELINE_BEST_METRIC",
    "CONTENT_PROFILE_BATCH_SIZE",
    "CONTENT_PROFILE_COLD_THRESHOLD",
    "CONTENT_PROFILE_EVAL_N_NEG",
    "CONTENT_PROFILE_STATIC_SEED",
    "CONTENT_PROFILE_SEED",
    "CGRC_PAPER_STATIC_EPOCHS",
    "CGRC_PAPER_BATCH_SIZE",
    "CGRC_PAPER_EVAL_N_NEG",
    "CGRC_PAPER_COLD_THRESHOLD",
    "CGRC_PAPER_BEST_AVERAGE_MODE",
    "CGRC_PAPER_RUN_SAMPLED_EVAL",
    "CGRC_PAPER_MASK_RHO",
    "CGRC_PAPER_RECON_TOPK",
    "CGRC_PAPER_RECON_USER_CHUNK",
    "CGRC_PAPER_LAMBDA_E",
    "CGRC_PAPER_TAU",
    "CGRC_PAPER_STATIC_SEED",
    "CGRC_PAPER_SEED",
    "CGRC_PAPER_CKPT_DIR",
    "CGRC_PAPER_SAVE_CKPT",
    "CGRC_PAPER_SAVE_OPT_STATE",
    "CGRC_PAPER_AUTO_RESUME",
    "CGRC_PAPER_FORCE_FRESH"
)

function Save-EnvSnapshot {
    $snapshot = @{}
    foreach ($name in $baselineEnvNames) {
        $snapshot[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    return $snapshot
}

function Restore-EnvSnapshot([hashtable]$Snapshot) {
    foreach ($name in $baselineEnvNames) {
        if ($null -eq $Snapshot[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" ([string]$Snapshot[$name])
        }
    }
}

function Set-CommonBaselineEnv([int]$Seed, [string]$SplitDir, [string]$OutDir) {
    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = $DataDir
    $env:USIM_RELATION_DIR = $RelationDir
    $env:USIM_STATIC_SPLIT_DIR = $SplitDir
    $env:USIM_BASELINE_OUTPUT_DIR = $OutDir
    $env:USIM_STATIC_SEED = [string]$Seed
    $env:USIM_SEED = [string]$Seed
    $env:USIM_COLD_THRESHOLD = [string]$ColdThreshold
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_EVAL_N_NEG = [string]$EvalNeg
    $env:USIM_RUN_SAMPLED_EVAL = "0"
    $env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
    $env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
    $env:BASELINE_BEST_METRIC = "cold"
}

function Invoke-Baseline(
    [int]$Seed,
    [string]$Name,
    [string]$ScriptName,
    [string]$ResultFile,
    [scriptblock]$Configure
) {
    $splitDir = if ($DryRun) { Ours-Full-SplitDir $Seed } else { Assert-SharedSplit $Seed }
    $outDir = Join-Path $splitDir "main_table_compare"
    $resultPath = Join-Path $outDir $ResultFile
    $logPath = Join-Path $outDir ("run_{0}.log" -f ($Name -replace "[^A-Za-z0-9]+", "_").Trim("_").ToLowerInvariant())

    if ((-not $DryRun) -and (Test-Path -LiteralPath $resultPath) -and (-not $RerunCompleted)) {
        Log "SKIP $Name seed=$Seed | exists=$resultPath"
        return
    }
    if ($DryRun) {
        Write-Host "Baseline=$Name"
        Write-Host "$Name.OutputDir=$outDir"
        Write-Host "USIM_STATIC_SPLIT_DIR=$splitDir"
        return
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $snapshot = Save-EnvSnapshot
    try {
        Set-CommonBaselineEnv $Seed $splitDir $outDir
        & $Configure

        Log "START $Name seed=$Seed | out=$outDir"
        $cmd = '/d /c ""' + $PythonRunner + '" -u "' + $ScriptName + '" > "' + $logPath + '" 2>&1"'
        $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
        Log "END $Name seed=$Seed | exit=$($p.ExitCode) | log=$logPath"
        if ($p.ExitCode -ne 0) {
            throw "$Name failed for seed=$Seed with exit=$($p.ExitCode)"
        }
    } finally {
        Restore-EnvSnapshot $snapshot
    }
}

function Invoke-ContentProfile([int]$Seed) {
    Invoke-Baseline $Seed "ContentProfile" "content_profile_static_hin.py" "content_profile_static_result.json" {
        $env:CONTENT_PROFILE_BATCH_SIZE = [string]$ContentProfileBatchSize
        $env:CONTENT_PROFILE_COLD_THRESHOLD = [string]$ColdThreshold
        $env:CONTENT_PROFILE_EVAL_N_NEG = [string]$EvalNeg
        $env:CONTENT_PROFILE_STATIC_SEED = [string]$Seed
        $env:CONTENT_PROFILE_SEED = [string]$Seed
    }
}

function Invoke-CgrcPaper([int]$Seed) {
    Invoke-Baseline $Seed "CGRC-paper" "cgrc_paper_static_hin.py" "cgrc_paper_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("cgrc_paper\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:CGRC_PAPER_STATIC_EPOCHS = [string]$CgrcEpochs
        $env:CGRC_PAPER_BATCH_SIZE = [string]$CgrcBatchSize
        $env:CGRC_PAPER_EVAL_N_NEG = [string]$EvalNeg
        $env:CGRC_PAPER_COLD_THRESHOLD = [string]$ColdThreshold
        $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
        $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
        $env:CGRC_PAPER_MASK_RHO = "0.3"
        $env:CGRC_PAPER_RECON_TOPK = [string]$CgrcReconTopK
        $env:CGRC_PAPER_RECON_USER_CHUNK = [string]$CgrcReconUserChunk
        $env:CGRC_PAPER_LAMBDA_E = "1.0"
        $env:CGRC_PAPER_TAU = "0.5"
        $env:CGRC_PAPER_STATIC_SEED = [string]$Seed
        $env:CGRC_PAPER_SEED = [string]$Seed
        $env:CGRC_PAPER_CKPT_DIR = $ckptDir
        $env:CGRC_PAPER_SAVE_CKPT = "1"
        $env:CGRC_PAPER_SAVE_OPT_STATE = "1"
        $env:CGRC_PAPER_AUTO_RESUME = if ($AutoResume) { "1" } else { "0" }
        $env:CGRC_PAPER_FORCE_FRESH = if ($ForceFresh) { "1" } else { "0" }
    }
}

function Invoke-Aggregates {
    if ($SkipAggregate -or $DryRun) {
        return
    }
    $oursRoot = Ours-Full-Root
    $noCourseRoot = NoCourse-Root

    foreach ($root in @($oursRoot, $noCourseRoot)) {
        if (Test-Path -LiteralPath $root) {
            Log "START aggregate FAST3 | root=$root"
            & $PythonRunner "aggregate_fast3_static_results.py" --root $root
            if ($LASTEXITCODE -ne 0) {
                throw "FAST3 aggregation failed: $root"
            }
            Log "END aggregate FAST3 | root=$root"
        }
    }

    $baselineOut = Join-Path $OutputRootAbs "main_table_compare"
    Log "START aggregate baselines | root=$oursRoot"
    & $PythonRunner "aggregate_main_table_static_results.py" `
        --root $oursRoot `
        --split-glob ("strict_item_cold_balanced_thr{0}_seed_*" -f $ColdThreshold) `
        --result-subdir "main_table_compare" `
        --metric-mode item_macro `
        --out-dir $baselineOut
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline aggregation failed"
    }
    Log "END aggregate baselines | out=$baselineOut"
}

if ($DryRun) {
    Show-DryRunHeader
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
    New-Item -ItemType Directory -Force -Path $OutputRootAbs | Out-Null
    New-Item -ItemType Directory -Force -Path $CheckpointRootAbs | Out-Null
    Assert-Inputs
    Assert-NoConcurrentTraining
}

Log "QUEUE START XES3G5M Ours-vs-SOTA serial | seeds=$($Seeds -join ',')"

foreach ($seed in $Seeds) {
    if (-not $SkipOursFull) {
        Invoke-Fast3Variant `
            -Seed $seed `
            -VariantName "OursFull" `
            -OutRoot (Ours-Full-Root) `
            -CkptRoot (Join-Path $CheckpointRootAbs "ours_full") `
            -ExtraParams @{
                UseCourseFeedback = $true
                UseCourseReward = $true
                UseCourseSample = $true
                UsePrereqAux = $true
                CourseFeedbackOnlyCold = $false
                CourseSampleOnlyCold = $false
                PrereqAuxOnlyCold = $false
            } | Out-Null
    }

    if (-not $SkipNoCourse) {
        Invoke-Fast3Variant `
            -Seed $seed `
            -VariantName "NoCourse" `
            -OutRoot (NoCourse-Root) `
            -CkptRoot (Join-Path $CheckpointRootAbs "wo_all_course_signals") `
            -ExtraParams @{
                UseCourseFeedback = $false
                UseCourseReward = $false
                UseCourseSample = $false
                UsePrereqAux = $false
                CourseFeedbackOnlyCold = $true
                CourseSampleOnlyCold = $true
                PrereqAuxOnlyCold = $true
            } | Out-Null
    }

    if (-not $SkipContentProfile) {
        Invoke-ContentProfile $seed
    }

    if (-not $SkipCgrc) {
        Invoke-CgrcPaper $seed
    }
}

Invoke-Aggregates

Log "QUEUE DONE XES3G5M Ours-vs-SOTA serial | seeds=$($Seeds -join ',')"

Write-Host ""
Write-Host "Logs:"
foreach ($seed in $Seeds) {
    Write-Host ("  OursFull:  {0}" -f (Join-Path (Ours-Full-SplitDir $seed) "run.log"))
    Write-Host ("  NoCourse:   {0}" -f (Join-Path (NoCourse-SplitDir $seed) "run.log"))
    Write-Host ("  Compare:    {0}" -f (Join-Path (Join-Path (Ours-Full-SplitDir $seed) "main_table_compare") "run_cgrc_paper.log"))
}
