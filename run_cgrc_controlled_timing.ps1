param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$TimingSeeds = @(9101, 9102, 9103, 9104, 9105),
    [int]$WarmupEpochs = 10,
    [int]$TimedEpochs = 20,
    [double]$TelemetryIntervalSeconds = 2.0,
    [string[]]$Datasets = @("Junyi", "COCO", "MOOCCube"),
    [string]$OutputRoot = "outputs\cgrc_formal_timing_v1",
    [switch]$TimingOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -LiteralPath $Repo

if (-not $TimingOnly.IsPresent) {
    throw "The formal CGRC timing protocol requires -TimingOnly"
}
if ($WarmupEpochs -ne 10 -or $TimedEpochs -ne 20) {
    throw "The formal CGRC timing protocol is fixed at 10 warm-up and 20 timed epochs"
}
if ([string]::Join(",", $TimingSeeds) -ne "9101,9102,9103,9104,9105") {
    throw "The formal CGRC timing protocol requires timing seeds 9101,9102,9103,9104,9105 in order"
}
if ([string]::Join(",", $Datasets) -ne "Junyi,COCO,MOOCCube") {
    throw "The formal CGRC timing protocol requires datasets Junyi,COCO,MOOCCube in order"
}
if ($TelemetryIntervalSeconds -le 0) {
    throw "TelemetryIntervalSeconds must be positive"
}

$protocolId = "cgrc_formal_timing_v1"
$telemetryScript = Join-Path $Repo "paper_aaai27\scripts\monitor_cgrc_gpu_telemetry.py"
if (-not (Test-Path -LiteralPath $telemetryScript)) {
    throw "Missing telemetry script: $telemetryScript"
}
$summaryScript = Join-Path $Repo "paper_aaai27\scripts\summarize_cgrc_controlled_timing.py"
if (-not (Test-Path -LiteralPath $summaryScript)) {
    throw "Missing summary script: $summaryScript"
}
$telemetryLauncher = Join-Path $Repo "py.bat"
if (-not (Test-Path -LiteralPath $telemetryLauncher)) {
    throw "Missing Python launcher: $telemetryLauncher"
}

$jobs = @(
    @{
        Dataset = "Junyi"; Seed = 2026; SourceEpoch = 50
        DataDir = "processed_data_junyi"
        SplitDir = "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2026"
        SourceCheckpoint = "checkpoints\junyi\main_table_3seed\cgrc_paper_compare\strict_item_cold_balanced_thr1_seed_2026\latest.pt"
        BatchSize = 4096; ReconUserChunk = 4096; CudaMemoryFraction = "0.75"
    },
    @{
        Dataset = "COCO"; Seed = 2026; SourceEpoch = 50
        DataDir = "processed_data_coco"
        SplitDir = "outputs\coco\single_seed_triage\ours_full\strict_item_cold_balanced_thr1_seed_2026"
        SourceCheckpoint = "checkpoints\coco\single_seed_triage\cgrc_paper\strict_item_cold_balanced_thr1_seed_2026\latest.pt"
        BatchSize = 512; ReconUserChunk = 256; CudaMemoryFraction = "0.75"
    },
    @{
        Dataset = "MOOCCube"; Seed = 2026; SourceEpoch = 50
        DataDir = "processed_data_hin_clean_pop5"
        SplitDir = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2026"
        SourceCheckpoint = "checkpoints\content_delta_pop5\p1_motivation_cgrc_main_table_reproduction\strict_item_cold_balanced_thr1_seed_2026\latest.pt"
        BatchSize = 4096; ReconUserChunk = 4096; CudaMemoryFraction = "0.85"
    }
)
$jobs = @($jobs | Where-Object { $Datasets -contains $_.Dataset })
if ($jobs.Count -eq 0) {
    throw "Datasets did not match Junyi, COCO, or MOOCCube"
}

foreach ($job in $jobs) {
    if (-not (Test-Path -LiteralPath $job.SourceCheckpoint)) {
        throw "Missing source checkpoint: $($job.SourceCheckpoint)"
    }
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $job.SplitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing split file: $path"
        }
    }
}

foreach ($job in $jobs) {
    $dataset = [string]$job.Dataset
    $seed = [int]$job.Seed
    $sourceEpoch = [int]$job.SourceEpoch
    $timedStartEpoch = $sourceEpoch + $WarmupEpochs + 1
    $targetEpoch = $sourceEpoch + $WarmupEpochs + $TimedEpochs
    $checkpointDir = Split-Path -Parent $job.SourceCheckpoint

    foreach ($timingSeed in $TimingSeeds) {
        $runName = "timing_seed_$timingSeed"
        $outDir = Join-Path $OutputRoot "$dataset\seed_$seed\$runName"
        $resultPath = Join-Path $outDir "cgrc_timing_profile.json"
        $logPath = Join-Path $outDir "run.log"
        $telemetryPath = Join-Path $outDir "timing_telemetry.csv"
        $telemetryStopPath = Join-Path $outDir "timing_telemetry.stop"
        $telemetryStdoutPath = Join-Path $outDir "timing_telemetry.stdout.log"
        $telemetryErrorPath = Join-Path $outDir "timing_telemetry.stderr.log"

        if (Test-Path -LiteralPath $resultPath) {
            & $telemetryLauncher -u $summaryScript `
                --validate-profile $resultPath `
                --dataset $dataset `
                --model-seed $seed `
                --timing-seed $timingSeed `
                --source-epoch $sourceEpoch `
                --warmup-epochs $WarmupEpochs `
                --timed-epochs $TimedEpochs `
                --require-telemetry
            if ($LASTEXITCODE -ne 0) {
                throw "Existing formal timing result failed integrity validation: $resultPath"
            }
            Write-Host "[$(Get-Date -Format o)] SKIP completed dataset=$dataset timing_seed=$timingSeed"
            continue
        }
        Write-Host "[$(Get-Date -Format o)] START dataset=$dataset timing_seed=$timingSeed target_epoch=$targetEpoch"
        Write-Host "  source_checkpoint=$($job.SourceCheckpoint)"
        Write-Host "  output=$outDir"
        if ($DryRun) {
            continue
        }

        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        if (Test-Path -LiteralPath $logPath) {
            $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
            Move-Item -LiteralPath $logPath -Destination "$logPath.interrupted_$stamp"
        }
        Remove-Item -LiteralPath $telemetryStopPath -Force -ErrorAction SilentlyContinue

        $env:CUDA_VISIBLE_DEVICES = "0"
        $env:PYTHONUNBUFFERED = "1"
        $env:USIM_DATA_DIR = [string]$job.DataDir
        $env:USIM_STATIC_SPLIT_DIR = [string]$job.SplitDir
        $env:USIM_BASELINE_OUTPUT_DIR = $outDir
        $env:USIM_STATIC_SEED = "$seed"
        $env:USIM_SEED = "$seed"
        $env:USIM_COLD_THRESHOLD = "1"
        $env:USIM_STATIC_TEST_HISTORY = "train_only"
        $env:USIM_EVAL_N_NEG = "200"

        $env:CGRC_PAPER_STATIC_EPOCHS = "$targetEpoch"
        $env:CGRC_PAPER_BATCH_SIZE = "$($job.BatchSize)"
        $env:CGRC_PAPER_EVAL_N_NEG = "200"
        $env:CGRC_PAPER_COLD_THRESHOLD = "1"
        $env:CGRC_PAPER_BEST_AVERAGE_MODE = "item_macro"
        $env:CGRC_PAPER_RUN_SAMPLED_EVAL = "0"
        $env:CGRC_PAPER_DEVICE = "cuda"
        $env:CGRC_PAPER_CUDA_MEMORY_FRACTION = [string]$job.CudaMemoryFraction
        $env:CGRC_PAPER_MASK_RHO = "0.3"
        $env:CGRC_PAPER_RECON_TOPK = "20"
        $env:CGRC_PAPER_RECON_USER_CHUNK = "$($job.ReconUserChunk)"
        $env:CGRC_PAPER_LAMBDA_E = "1.0"
        $env:CGRC_PAPER_TAU = "0.5"
        $env:CGRC_PAPER_PROGRESS_INTERVAL = "0"
        $env:CGRC_PAPER_EVAL_SPLIT = "validation"
        $env:CGRC_PAPER_TIMING_ONLY = if ($TimingOnly.IsPresent) { "1" } else { "0" }
        $env:CGRC_PAPER_TIMING_PROTOCOL = $protocolId
        $env:CGRC_PAPER_STATIC_SEED = "$seed"
        $env:CGRC_PAPER_SEED = "$seed"
        $env:CGRC_PAPER_TIMING_SEED = "$timingSeed"
        $env:CGRC_PAPER_TIMING_MEASURE_START_EPOCH = "$timedStartEpoch"
        $env:CGRC_PAPER_CKPT_DIR = $checkpointDir
        $env:CGRC_PAPER_SAVE_CKPT = "0"
        $env:CGRC_PAPER_AUTO_RESUME = "1"
        $env:CGRC_PAPER_FORCE_FRESH = "0"
        $env:CGRC_PAPER_SAVE_OPT_STATE = "0"

        "[$(Get-Date -Format o)] [CGRC-FORMAL-TIMING] protocol=$protocolId dataset=$dataset model_seed=$seed timing_seed=$timingSeed source_epoch=$sourceEpoch warmup_epochs=$WarmupEpochs timed_epochs=$TimedEpochs timed_start_epoch=$timedStartEpoch" |
            Out-File -LiteralPath $logPath -Encoding utf8
        $telemetryArguments = "-u `"$telemetryScript`" --output `"$telemetryPath`" --stop-file `"$telemetryStopPath`" --interval-seconds $TelemetryIntervalSeconds --device 0"
        $telemetryProcess = Start-Process -FilePath $telemetryLauncher -ArgumentList $telemetryArguments -WorkingDirectory $Repo -WindowStyle Hidden -RedirectStandardOutput $telemetryStdoutPath -RedirectStandardError $telemetryErrorPath -PassThru
        $exitCode = $null
        try {
            $commandLine = "`".\py.bat`" -u -X faulthandler `"cgrc_paper_static_hin.py`" >> `"$logPath`" 2>&1"
            & cmd.exe /d /c $commandLine
            $exitCode = $LASTEXITCODE
        }
        finally {
            New-Item -ItemType File -Force -Path $telemetryStopPath | Out-Null
            if ($telemetryProcess) {
                try {
                    Wait-Process -Id $telemetryProcess.Id -Timeout 15 -ErrorAction Stop
                }
                catch {
                    Stop-Process -Id $telemetryProcess.Id -Force -ErrorAction SilentlyContinue
                }
            }
        }
        if ($exitCode -ne 0) {
            throw "Formal timing failed: dataset=$dataset timing_seed=$timingSeed exit=$exitCode; see $logPath"
        }
        if (-not (Test-Path -LiteralPath $resultPath)) {
            throw "Formal timing exited without result JSON: dataset=$dataset timing_seed=$timingSeed"
        }
        $telemetryLines = @(Get-Content -LiteralPath $telemetryPath -ErrorAction Stop)
        if ($telemetryLines.Count -lt 2) {
            throw "Formal timing completed without telemetry samples: $telemetryPath"
        }
        Write-Host "[$(Get-Date -Format o)] DONE dataset=$dataset timing_seed=$timingSeed"
    }
}

if (-not $DryRun) {
    & $telemetryLauncher $summaryScript `
        --root $OutputRoot `
        --source-epoch 50 `
        --warmup-epochs $WarmupEpochs `
        --timed-epochs $TimedEpochs `
        --require-telemetry `
        --formal
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to summarize formal CGRC timing profiles"
    }
}

Write-Host "[$(Get-Date -Format o)] Formal CGRC timing queue complete"
