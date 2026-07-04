param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_xes3g5m",
    [string]$RelationDir = "processed_data_xes3g5m\relations",
    [ValidateSet("behavior", "concept", "hybrid")]
    [string]$PrereqGraphSource = "concept",
    [string]$OutputRoot = "outputs\xes3g5m\ours_sota_serial",
    [string]$CheckpointRoot = "checkpoints\xes3g5m\ours_sota_serial",
    [int[]]$Seeds = @(2025),
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [string[]]$Models = @("Popularity", "ContentProfile"),
    [string]$ResultSubdir = "main_table_compare",
    [int]$PopBatchSize = 512,
    [int]$ContentProfileBatchSize = 512,
    [int]$BprEpochs = 5,
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
    [int]$PollSeconds = 120,
    [switch]$AllowConcurrent,
    [switch]$UseGpu,
    [switch]$ContinueOnError,
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
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$QueueDir = Join-Path $OutputRootAbs "_queue"
$QueueLog = Join-Path $QueueDir "lightweight_baselines.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootAbs | Out-Null

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $QueueLog -Encoding UTF8 -Value $line
    Write-Host $line
}

function Split-Name([int]$Seed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $Seed
}

function Ours-SplitDir([int]$Seed) {
    Join-Path (Join-Path $OutputRootAbs "ours_full") (Split-Name $Seed)
}

function Assert-Inputs {
    foreach ($path in @($DataDir, $RelationDir)) {
        $abs = Resolve-RunPath $Repo $path
        if (-not (Test-Path -LiteralPath $abs)) {
            throw "Missing required path: $abs"
        }
    }
    $scripts = @(
        "popularity_static.py",
        "content_profile_static_hin.py",
        "bpr_static_fair.py",
        "lightgcn_static_hin_fair.py",
        "drop_static_hin.py",
        "gar_static_hin.py",
        "ccfc_static_hin.py",
        "aldi_static_hin.py",
        "marec_static_hin_fair.py",
        "lightgcl_static_hin_fair.py",
        "sagerec_static_baseline.py",
        "course_aware_mlp_static_hin.py"
    )
    foreach ($script in $scripts) {
        $abs = Join-Path $Repo $script
        if (-not (Test-Path -LiteralPath $abs)) {
            throw "Missing baseline script: $abs"
        }
    }
}

function Assert-SharedSplit([int]$Seed) {
    $splitDir = Ours-SplitDir $Seed
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing shared split artifact for seed=${Seed}: $path"
        }
    }
    return $splitDir
}

function Get-CgrcProcesses {
    @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.CommandLine -and
                $_.CommandLine -like "*cgrc_paper_static_hin.py*" -and
                $_.ProcessId -ne $PID
            }
    )
}

function Wait-ForCgrcIfNeeded {
    if ($AllowConcurrent -or $DryRun) {
        return
    }
    while ($true) {
        $running = Get-CgrcProcesses
        if ($running.Count -lt 1) {
            return
        }
        $ids = ($running | Select-Object -ExpandProperty ProcessId) -join ","
        Write-Log "WAIT current CGRC process(es) still running: pid=$ids"
        Start-Sleep -Seconds $PollSeconds
    }
}

$baselineEnvNames = @(
    "PYTHONUNBUFFERED",
    "CUDA_VISIBLE_DEVICES",
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
    "BASELINE_DEVICE",
    "BASELINE_EARLY_STOP_AVG_MODE",
    "BASELINE_EARLY_STOP_AVERAGE_MODE",
    "BASELINE_BEST_METRIC",
    "USIM_EARLY_STOP_AVG_MODE",
    "USIM_PREREQ_GRAPH_SOURCE",
    "POP_DEVICE",
    "POP_BATCH_SIZE",
    "POP_COLD_THRESHOLD",
    "POP_EVAL_N_NEG",
    "POP_STATIC_SEED",
    "POP_SEED",
    "CONTENT_PROFILE_DEVICE",
    "CONTENT_PROFILE_BATCH_SIZE",
    "CONTENT_PROFILE_COLD_THRESHOLD",
    "CONTENT_PROFILE_EVAL_N_NEG",
    "CONTENT_PROFILE_STATIC_SEED",
    "CONTENT_PROFILE_SEED",
    "BPR_STATIC_EPOCHS",
    "BPR_EVAL_INTERVAL",
    "BPR_EMB_DIM",
    "BPR_BATCH_SIZE",
    "BPR_COLD_THRESHOLD",
    "BPR_EVAL_N_NEG",
    "BPR_STATIC_SEED",
    "BPR_SEED",
    "BPR_CKPT_DIR",
    "BPR_SAVE_CKPT",
    "BPR_SAVE_OPT_STATE",
    "BPR_AUTO_RESUME",
    "BPR_FORCE_FRESH",
    "LIGHTGCN_STATIC_EPOCHS",
    "LIGHTGCN_EVAL_INTERVAL",
    "LIGHTGCN_EMB_DIM",
    "LIGHTGCN_BATCH_SIZE",
    "LIGHTGCN_N_LAYERS",
    "LIGHTGCN_CONTENT_WEIGHT",
    "LIGHTGCN_COLD_THRESHOLD",
    "LIGHTGCN_EVAL_N_NEG",
    "LIGHTGCN_STATIC_SEED",
    "LIGHTGCN_SEED",
    "LIGHTGCN_CKPT_DIR",
    "LIGHTGCN_SAVE_CKPT",
    "LIGHTGCN_SAVE_OPT_STATE",
    "LIGHTGCN_AUTO_RESUME",
    "LIGHTGCN_FORCE_FRESH",
    "DROPOUT_STATIC_EPOCHS",
    "DROPOUT_EVAL_INTERVAL",
    "DROPOUT_BATCH_SIZE",
    "DROPOUT_COLD_THRESHOLD",
    "DROPOUT_EVAL_N_NEG",
    "DROPOUT_STATIC_SEED",
    "DROPOUT_SEED",
    "DROPOUT_EARLY_STOP_AVG_MODE",
    "DROPOUT_CKPT_DIR",
    "DROPOUT_SAVE_CKPT",
    "DROPOUT_SAVE_OPT_STATE",
    "DROPOUT_AUTO_RESUME",
    "DROPOUT_FORCE_FRESH",
    "GAR_STATIC_EPOCHS",
    "GAR_EVAL_INTERVAL",
    "GAR_BATCH_SIZE",
    "GAR_COLD_THRESHOLD",
    "GAR_EVAL_N_NEG",
    "GAR_STATIC_SEED",
    "GAR_SEED",
    "GAR_CKPT_DIR",
    "GAR_SAVE_CKPT",
    "GAR_SAVE_OPT_STATE",
    "GAR_AUTO_RESUME",
    "GAR_FORCE_FRESH",
    "CCFCREC_STATIC_EPOCHS",
    "CCFCREC_EVAL_INTERVAL",
    "CCFCREC_EMB_DIM",
    "CCFCREC_HIDDEN_DIM",
    "CCFCREC_BATCH_SIZE",
    "CCFCREC_EVAL_BATCH_SIZE",
    "CCFCREC_COLD_THRESHOLD",
    "CCFCREC_EVAL_N_NEG",
    "CCFCREC_STATIC_SEED",
    "CCFCREC_SEED",
    "CCFCREC_EARLY_STOP_AVG_MODE",
    "CCFCREC_CKPT_DIR",
    "CCFCREC_SAVE_CKPT",
    "CCFCREC_SAVE_OPT_STATE",
    "CCFCREC_AUTO_RESUME",
    "CCFCREC_FORCE_FRESH",
    "ALDI_TEACHER_EPOCHS",
    "ALDI_TEACHER_EVAL_INTERVAL",
    "ALDI_STATIC_EPOCHS",
    "ALDI_EVAL_INTERVAL",
    "ALDI_EMB_DIM",
    "ALDI_HIDDEN_DIM",
    "ALDI_BATCH_SIZE",
    "ALDI_EVAL_BATCH_SIZE",
    "ALDI_COLD_THRESHOLD",
    "ALDI_EVAL_N_NEG",
    "ALDI_STATIC_SEED",
    "ALDI_SEED",
    "ALDI_EARLY_STOP_AVG_MODE",
    "ALDI_CKPT_DIR",
    "ALDI_TEACHER_CKPT_DIR",
    "ALDI_SAVE_CKPT",
    "ALDI_SAVE_OPT_STATE",
    "ALDI_AUTO_RESUME",
    "ALDI_FORCE_FRESH",
    "MAREC_BATCH_SIZE",
    "MAREC_COLD_THRESHOLD",
    "MAREC_EVAL_N_NEG",
    "MAREC_STATIC_SEED",
    "MAREC_SEED",
    "MAREC_LAMBDAS",
    "MAREC_ALPHAS",
    "MAREC_CONTENT_BETAS",
    "MAREC_META_TOPK",
    "LIGHTGCL_STATIC_EPOCHS",
    "LIGHTGCL_EVAL_INTERVAL",
    "LIGHTGCL_EMB_DIM",
    "LIGHTGCL_BATCH_SIZE",
    "LIGHTGCL_EVAL_BATCH_SIZE",
    "LIGHTGCL_N_LAYERS",
    "LIGHTGCL_SVD_RANK",
    "LIGHTGCL_COLD_THRESHOLD",
    "LIGHTGCL_EVAL_N_NEG",
    "LIGHTGCL_STATIC_SEED",
    "LIGHTGCL_SEED",
    "LIGHTGCL_CKPT_DIR",
    "LIGHTGCL_SAVE_CKPT",
    "LIGHTGCL_SAVE_OPT_STATE",
    "LIGHTGCL_AUTO_RESUME",
    "LIGHTGCL_FORCE_FRESH",
    "SAGEREC_STATIC_EPOCHS",
    "SAGEREC_EVAL_INTERVAL",
    "SAGEREC_EMB_DIM",
    "SAGEREC_BATCH_SIZE",
    "SAGEREC_SAMPLE_TOP_N",
    "SAGEREC_MAX_HIST_LEN",
    "SAGEREC_COLD_THRESHOLD",
    "SAGEREC_EVAL_N_NEG",
    "SAGEREC_STATIC_SEED",
    "SAGEREC_SEED",
    "SAGEREC_CKPT_DIR",
    "SAGEREC_SAVE_CKPT",
    "SAGEREC_SAVE_OPT_STATE",
    "SAGEREC_AUTO_RESUME",
    "SAGEREC_FORCE_FRESH",
    "COURSE_MLP_STATIC_EPOCHS",
    "COURSE_MLP_EVAL_INTERVAL",
    "COURSE_MLP_BATCH_SIZE",
    "COURSE_MLP_EVAL_BATCH_SIZE",
    "COURSE_MLP_COLD_THRESHOLD",
    "COURSE_MLP_STATIC_SEED",
    "COURSE_MLP_SEED"
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

function Set-CommonEnv([int]$Seed, [string]$SplitDir, [string]$OutDir) {
    $env:PYTHONUNBUFFERED = "1"
    if ($UseGpu) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_VISIBLE_DEVICES = ""
    }
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
    $env:BASELINE_DEVICE = if ($UseGpu) { "auto" } else { "cpu" }
    $env:BASELINE_EARLY_STOP_AVG_MODE = "item_macro"
    $env:BASELINE_EARLY_STOP_AVERAGE_MODE = "item_macro"
    $env:BASELINE_BEST_METRIC = "cold"
    $env:USIM_EARLY_STOP_AVG_MODE = "item_macro"
    $env:USIM_PREREQ_GRAPH_SOURCE = $PrereqGraphSource
}

function Invoke-Baseline(
    [int]$Seed,
    [string]$Name,
    [string]$ScriptName,
    [string]$ResultFile,
    [scriptblock]$Configure
) {
    $splitDir = Assert-SharedSplit $Seed
    $outDir = Join-Path $splitDir $ResultSubdir
    $resultPath = Join-Path $outDir $ResultFile
    $logStem = ($Name -replace "[^A-Za-z0-9]+", "_").Trim("_").ToLowerInvariant()
    $logPath = Join-Path $outDir ("run_{0}.log" -f $logStem)

    if ((Test-Path -LiteralPath $resultPath) -and (-not $RerunCompleted)) {
        Write-Log "SKIP $Name seed=$Seed | exists=$resultPath"
        return
    }
    if ($DryRun) {
        Write-Host "Baseline=$Name"
        Write-Host "Script=$ScriptName"
        Write-Host "OutputDir=$outDir"
        Write-Host "Result=$resultPath"
        return
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $snapshot = Save-EnvSnapshot
    try {
        Set-CommonEnv $Seed $splitDir $outDir
        & $Configure

        Write-Log "START $Name seed=$Seed | out=$outDir"
        $cmd = '/d /c ""' + $PythonRunner + '" -u "' + $ScriptName + '" > "' + $logPath + '" 2>&1"'
        $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
        Write-Log "END $Name seed=$Seed | exit=$($p.ExitCode) | log=$logPath"
        if ($p.ExitCode -ne 0) {
            throw "$Name failed for seed=$Seed with exit=$($p.ExitCode)"
        }
    } finally {
        Restore-EnvSnapshot $snapshot
    }
}

function Invoke-Popularity([int]$Seed) {
    Invoke-Baseline $Seed "Popularity" "popularity_static.py" "popularity_static_result.json" {
        $env:POP_DEVICE = if ($UseGpu) { "auto" } else { "cpu" }
        $env:POP_BATCH_SIZE = [string]$PopBatchSize
        $env:POP_COLD_THRESHOLD = [string]$ColdThreshold
        $env:POP_EVAL_N_NEG = [string]$EvalNeg
        $env:POP_STATIC_SEED = [string]$Seed
        $env:POP_SEED = [string]$Seed
    }
}

function Invoke-ContentProfile([int]$Seed) {
    Invoke-Baseline $Seed "ContentProfile" "content_profile_static_hin.py" "content_profile_static_result.json" {
        $env:CONTENT_PROFILE_DEVICE = if ($UseGpu) { "auto" } else { "cpu" }
        $env:CONTENT_PROFILE_BATCH_SIZE = [string]$ContentProfileBatchSize
        $env:CONTENT_PROFILE_COLD_THRESHOLD = [string]$ColdThreshold
        $env:CONTENT_PROFILE_EVAL_N_NEG = [string]$EvalNeg
        $env:CONTENT_PROFILE_STATIC_SEED = [string]$Seed
        $env:CONTENT_PROFILE_SEED = [string]$Seed
    }
}

function Invoke-Bpr([int]$Seed) {
    Invoke-Baseline $Seed "BPR" "bpr_static_fair.py" "bpr_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("bpr_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:BPR_STATIC_EPOCHS = [string]$BprEpochs
        $env:BPR_EVAL_INTERVAL = [string]$BprEvalInterval
        $env:BPR_EMB_DIM = [string]$BprEmbDim
        $env:BPR_BATCH_SIZE = [string]$BprBatchSize
        $env:BPR_COLD_THRESHOLD = [string]$ColdThreshold
        $env:BPR_EVAL_N_NEG = [string]$EvalNeg
        $env:BPR_STATIC_SEED = [string]$Seed
        $env:BPR_SEED = [string]$Seed
        $env:BPR_CKPT_DIR = $ckptDir
        $env:BPR_SAVE_CKPT = "1"
        $env:BPR_SAVE_OPT_STATE = "1"
        $env:BPR_AUTO_RESUME = "1"
        $env:BPR_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-LightGCN([int]$Seed) {
    Invoke-Baseline $Seed "LightGCN" "lightgcn_static_hin_fair.py" "lightgcn_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("lightgcn_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:LIGHTGCN_STATIC_EPOCHS = [string]$LightGCNEpochs
        $env:LIGHTGCN_EVAL_INTERVAL = [string]$LightGCNEvalInterval
        $env:LIGHTGCN_EMB_DIM = [string]$LightGCNEmbDim
        $env:LIGHTGCN_BATCH_SIZE = [string]$LightGCNBatchSize
        $env:LIGHTGCN_N_LAYERS = [string]$LightGCNLayers
        $env:LIGHTGCN_CONTENT_WEIGHT = [string]$LightGCNContentWeight
        $env:LIGHTGCN_COLD_THRESHOLD = [string]$ColdThreshold
        $env:LIGHTGCN_EVAL_N_NEG = [string]$EvalNeg
        $env:LIGHTGCN_STATIC_SEED = [string]$Seed
        $env:LIGHTGCN_SEED = [string]$Seed
        $env:LIGHTGCN_CKPT_DIR = $ckptDir
        $env:LIGHTGCN_SAVE_CKPT = "1"
        $env:LIGHTGCN_SAVE_OPT_STATE = "1"
        $env:LIGHTGCN_AUTO_RESUME = "1"
        $env:LIGHTGCN_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-DropoutNet([int]$Seed) {
    Invoke-Baseline $Seed "DropoutNet" "drop_static_hin.py" "drop_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("dropoutnet_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:DROPOUT_STATIC_EPOCHS = [string]$DropoutEpochs
        $env:DROPOUT_EVAL_INTERVAL = [string]$DropoutEvalInterval
        $env:DROPOUT_BATCH_SIZE = [string]$DropoutBatchSize
        $env:DROPOUT_COLD_THRESHOLD = [string]$ColdThreshold
        $env:DROPOUT_EVAL_N_NEG = [string]$EvalNeg
        $env:DROPOUT_STATIC_SEED = [string]$Seed
        $env:DROPOUT_SEED = [string]$Seed
        $env:DROPOUT_EARLY_STOP_AVG_MODE = "item_macro"
        $env:DROPOUT_CKPT_DIR = $ckptDir
        $env:DROPOUT_SAVE_CKPT = "1"
        $env:DROPOUT_SAVE_OPT_STATE = "1"
        $env:DROPOUT_AUTO_RESUME = "1"
        $env:DROPOUT_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-GAR([int]$Seed) {
    Invoke-Baseline $Seed "GAR" "gar_static_hin.py" "gar_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("gar_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:GAR_STATIC_EPOCHS = [string]$GarEpochs
        $env:GAR_EVAL_INTERVAL = [string]$GarEvalInterval
        $env:GAR_BATCH_SIZE = [string]$GarBatchSize
        $env:GAR_COLD_THRESHOLD = [string]$ColdThreshold
        $env:GAR_EVAL_N_NEG = [string]$EvalNeg
        $env:GAR_STATIC_SEED = [string]$Seed
        $env:GAR_SEED = [string]$Seed
        $env:GAR_CKPT_DIR = $ckptDir
        $env:GAR_SAVE_CKPT = "1"
        $env:GAR_SAVE_OPT_STATE = "1"
        $env:GAR_AUTO_RESUME = "1"
        $env:GAR_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-CCFCRec([int]$Seed) {
    Invoke-Baseline $Seed "CCFCRec" "ccfc_static_hin.py" "ccfcrec_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("ccfcrec_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:CCFCREC_STATIC_EPOCHS = [string]$CCFCEpochs
        $env:CCFCREC_EVAL_INTERVAL = [string]$CCFCEvalInterval
        $env:CCFCREC_EMB_DIM = [string]$CCFCEmbDim
        $env:CCFCREC_HIDDEN_DIM = [string]$CCFCHiddenDim
        $env:CCFCREC_BATCH_SIZE = [string]$CCFCBatchSize
        $env:CCFCREC_EVAL_BATCH_SIZE = [string]$CCFCBatchSize
        $env:CCFCREC_COLD_THRESHOLD = [string]$ColdThreshold
        $env:CCFCREC_EVAL_N_NEG = [string]$EvalNeg
        $env:CCFCREC_STATIC_SEED = [string]$Seed
        $env:CCFCREC_SEED = [string]$Seed
        $env:CCFCREC_EARLY_STOP_AVG_MODE = "item_macro"
        $env:CCFCREC_CKPT_DIR = $ckptDir
        $env:CCFCREC_SAVE_CKPT = "1"
        $env:CCFCREC_SAVE_OPT_STATE = "1"
        $env:CCFCREC_AUTO_RESUME = "1"
        $env:CCFCREC_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-ALDI([int]$Seed) {
    Invoke-Baseline $Seed "ALDI" "aldi_static_hin.py" "aldi_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("aldi_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:ALDI_TEACHER_EPOCHS = [string]$ALDITeacherEpochs
        $env:ALDI_TEACHER_EVAL_INTERVAL = [string]$ALDIEvalInterval
        $env:ALDI_STATIC_EPOCHS = [string]$ALDIStudentEpochs
        $env:ALDI_EVAL_INTERVAL = [string]$ALDIEvalInterval
        $env:ALDI_EMB_DIM = [string]$ALDIEmbDim
        $env:ALDI_HIDDEN_DIM = [string]$ALDIEmbDim
        $env:ALDI_BATCH_SIZE = [string]$ALDIBatchSize
        $env:ALDI_EVAL_BATCH_SIZE = [string]$ALDIBatchSize
        $env:ALDI_COLD_THRESHOLD = [string]$ColdThreshold
        $env:ALDI_EVAL_N_NEG = [string]$EvalNeg
        $env:ALDI_STATIC_SEED = [string]$Seed
        $env:ALDI_SEED = [string]$Seed
        $env:ALDI_EARLY_STOP_AVG_MODE = "item_macro"
        $env:ALDI_CKPT_DIR = $ckptDir
        $env:ALDI_TEACHER_CKPT_DIR = (Join-Path $ckptDir "teacher")
        $env:ALDI_SAVE_CKPT = "1"
        $env:ALDI_SAVE_OPT_STATE = "1"
        $env:ALDI_AUTO_RESUME = "1"
        $env:ALDI_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-MARec([int]$Seed) {
    Invoke-Baseline $Seed "MARec" "marec_static_hin_fair.py" "marec_static_result.json" {
        $env:MAREC_BATCH_SIZE = [string]$MARecBatchSize
        $env:MAREC_COLD_THRESHOLD = [string]$ColdThreshold
        $env:MAREC_EVAL_N_NEG = [string]$EvalNeg
        $env:MAREC_STATIC_SEED = [string]$Seed
        $env:MAREC_SEED = [string]$Seed
        $env:MAREC_LAMBDAS = $MARecLambdas
        $env:MAREC_ALPHAS = $MARecAlphas
        $env:MAREC_CONTENT_BETAS = $MARecContentBetas
        $env:MAREC_META_TOPK = "50"
    }
}

function Invoke-LightGCL([int]$Seed) {
    Invoke-Baseline $Seed "LightGCL" "lightgcl_static_hin_fair.py" "lightgcl_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("lightgcl_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:LIGHTGCL_STATIC_EPOCHS = [string]$LightGCLEpochs
        $env:LIGHTGCL_EVAL_INTERVAL = [string]$LightGCLEvalInterval
        $env:LIGHTGCL_EMB_DIM = [string]$LightGCLEmbDim
        $env:LIGHTGCL_BATCH_SIZE = [string]$LightGCLBatchSize
        $env:LIGHTGCL_EVAL_BATCH_SIZE = [string]$LightGCLBatchSize
        $env:LIGHTGCL_N_LAYERS = [string]$LightGCLLayers
        $env:LIGHTGCL_SVD_RANK = [string]$LightGCLSvdRank
        $env:LIGHTGCL_COLD_THRESHOLD = [string]$ColdThreshold
        $env:LIGHTGCL_EVAL_N_NEG = [string]$EvalNeg
        $env:LIGHTGCL_STATIC_SEED = [string]$Seed
        $env:LIGHTGCL_SEED = [string]$Seed
        $env:LIGHTGCL_CKPT_DIR = $ckptDir
        $env:LIGHTGCL_SAVE_CKPT = "1"
        $env:LIGHTGCL_SAVE_OPT_STATE = "1"
        $env:LIGHTGCL_AUTO_RESUME = "1"
        $env:LIGHTGCL_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-SAGERec([int]$Seed) {
    Invoke-Baseline $Seed "SAGERec" "sagerec_static_baseline.py" "sagerec_static_result.json" {
        $ckptDir = Join-Path $CheckpointRootAbs ("sagerec_lightweight\{0}" -f (Split-Name $Seed))
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
        $env:SAGEREC_STATIC_EPOCHS = [string]$SageRecEpochs
        $env:SAGEREC_EVAL_INTERVAL = [string]$SageRecEvalInterval
        $env:SAGEREC_EMB_DIM = [string]$SageRecEmbDim
        $env:SAGEREC_BATCH_SIZE = [string]$SageRecBatchSize
        $env:SAGEREC_SAMPLE_TOP_N = [string]$SageRecSampleTopN
        $env:SAGEREC_MAX_HIST_LEN = [string]$SageRecMaxHistLen
        $env:SAGEREC_COLD_THRESHOLD = [string]$ColdThreshold
        $env:SAGEREC_EVAL_N_NEG = [string]$EvalNeg
        $env:SAGEREC_STATIC_SEED = [string]$Seed
        $env:SAGEREC_SEED = [string]$Seed
        $env:SAGEREC_CKPT_DIR = $ckptDir
        $env:SAGEREC_SAVE_CKPT = "1"
        $env:SAGEREC_SAVE_OPT_STATE = "1"
        $env:SAGEREC_AUTO_RESUME = "1"
        $env:SAGEREC_FORCE_FRESH = if ($RerunCompleted) { "1" } else { "0" }
    }
}

function Invoke-CourseMLP([int]$Seed) {
    Invoke-Baseline $Seed "CourseAware-MLP" "course_aware_mlp_static_hin.py" "course_aware_mlp_static_result.json" {
        $env:COURSE_MLP_STATIC_EPOCHS = [string]$CourseMlpEpochs
        $env:COURSE_MLP_EVAL_INTERVAL = [string]$CourseMlpEvalInterval
        $env:COURSE_MLP_BATCH_SIZE = [string]$CourseMlpBatchSize
        $env:COURSE_MLP_EVAL_BATCH_SIZE = [string]$CourseMlpBatchSize
        $env:COURSE_MLP_COLD_THRESHOLD = [string]$ColdThreshold
        $env:COURSE_MLP_STATIC_SEED = [string]$Seed
        $env:COURSE_MLP_SEED = [string]$Seed
    }
}

Assert-Inputs
$ModelNames = @()
foreach ($entry in $Models) {
    foreach ($name in ([string]$entry -split ",")) {
        $trimmed = $name.Trim()
        if ($trimmed) {
            $ModelNames += $trimmed
        }
    }
}

Write-Log "QUEUE START XES3G5M lightweight baselines | models=$($ModelNames -join ',') | seeds=$($Seeds -join ',') | allow_concurrent=$($AllowConcurrent.IsPresent)"
Wait-ForCgrcIfNeeded

foreach ($seed in $Seeds) {
    foreach ($model in $ModelNames) {
        try {
            switch ($model.Trim().ToLowerInvariant()) {
                "popularity" { Invoke-Popularity $seed }
                "contentprofile" { Invoke-ContentProfile $seed }
                "content_profile" { Invoke-ContentProfile $seed }
                "bpr" { Invoke-Bpr $seed }
                "lightgcn" { Invoke-LightGCN $seed }
                "light_gcn" { Invoke-LightGCN $seed }
                "dropoutnet" { Invoke-DropoutNet $seed }
                "dropout" { Invoke-DropoutNet $seed }
                "drop" { Invoke-DropoutNet $seed }
                "gar" { Invoke-GAR $seed }
                "gafc" { Invoke-GAR $seed }
                "ccfcrec" { Invoke-CCFCRec $seed }
                "ccfc" { Invoke-CCFCRec $seed }
                "aldi" { Invoke-ALDI $seed }
                "marec" { Invoke-MARec $seed }
                "lightgcl" { Invoke-LightGCL $seed }
                "light_gcl" { Invoke-LightGCL $seed }
                "sagerec" { Invoke-SAGERec $seed }
                "sage" { Invoke-SAGERec $seed }
                "courseaware-mlp" { Invoke-CourseMLP $seed }
                "course_mlp" { Invoke-CourseMLP $seed }
                "coursemlp" { Invoke-CourseMLP $seed }
                default { throw "Unknown lightweight baseline: $model" }
            }
        } catch {
            Write-Log "FAILED $model seed=$seed | $($_.Exception.Message)"
            if (-not $ContinueOnError) {
                throw
            }
        }
    }
}

if (-not $DryRun) {
    $baselineOut = Join-Path $OutputRootAbs "main_table_compare"
    Write-Log "START aggregate baselines | root=$(Join-Path $OutputRootAbs 'ours_full')"
    & $PythonRunner "aggregate_main_table_static_results.py" `
        --root (Join-Path $OutputRootAbs "ours_full") `
        --split-glob ("strict_item_cold_balanced_thr{0}_seed_*" -f $ColdThreshold) `
        --result-subdir $ResultSubdir `
        --metric-mode item_macro `
        --out-dir $baselineOut
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline aggregation failed"
    }
    Write-Log "END aggregate baselines | out=$baselineOut"
}

Write-Log "QUEUE DONE XES3G5M lightweight baselines"
