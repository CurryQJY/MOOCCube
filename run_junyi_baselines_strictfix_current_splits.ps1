param(
    [string[]]$Seeds = @("2026", "2027"),
    [string[]]$Models = @(
        "Popularity",
        "ContentProfile",
        "BPR",
        "LightGCN",
        "DropoutNet",
        "CCFCRec",
        "ALDI"
    ),
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\junyi\main_table_3seed",
    [string]$CheckpointRoot = "checkpoints\junyi\main_table_strictfix",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

$ParsedModels = @()
foreach ($rawModel in $Models) {
    foreach ($part in ([string]$rawModel -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) {
            $ParsedModels += $trimmed
        }
    }
}
$Models = [string[]]$ParsedModels

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $Base $Path)
}

$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$QueueDir = Join-Path $OutputRootAbs "_strictfix_queue"
$QueueLog = Join-Path $QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootAbs | Out-Null

function Write-QueueLogLine([string]$Path, [string]$Line) {
    $payload = $Line + [Environment]::NewLine
    [System.IO.File]::AppendAllText($Path, $payload, [System.Text.Encoding]::UTF8)
}

function Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-QueueLogLine $QueueLog $line
    Write-Host $line
}

function Split-Dir([int]$Seed) {
    return (Join-Path $OutputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed))
}

function Assert-Split([int]$Seed) {
    $splitDir = Split-Dir $Seed
    foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl", "static_split_summary.json")) {
        $path = Join-Path $splitDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing current strict split artifact for seed=${Seed}: $path"
        }
    }
    $summary = Get-Content -Raw -LiteralPath (Join-Path $splitDir "static_split_summary.json") | ConvertFrom-Json
    if ($summary.split_mode -ne "strict_item_cold_balanced") {
        throw "Unexpected split_mode for seed=${Seed}: $($summary.split_mode)"
    }
    if ([int]$summary.test_cold_items -ne 71) {
        throw "Unexpected test_cold_items for seed=${Seed}: $($summary.test_cold_items)"
    }
    return $splitDir
}

function Set-CommonEnv([int]$Seed, [string]$OutDir, [string]$CkptDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if ($CkptDir) {
        New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
    }

    $env:PYTHONUNBUFFERED = "1"
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
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

function Assert-StrictResult([int]$Seed, [string]$ResultPath) {
    if (-not (Test-Path -LiteralPath $ResultPath)) {
        throw "Missing result for seed=${Seed}: $ResultPath"
    }
    $splitDir = Split-Dir $Seed
    $summary = Get-Content -Raw -LiteralPath (Join-Path $splitDir "static_split_summary.json") | ConvertFrom-Json
    $raw = Get-Content -Raw -LiteralPath $ResultPath | ConvertFrom-Json
    $row = if ($raw -is [System.Array]) { $raw[0] } else { $raw }
    $expected = [int]$summary.test_cold_items
    $got = [int]$row.count_full_cold_item_macro
    if ($got -ne $expected) {
        throw "Strict count mismatch seed=${Seed}: got count_full_cold_item_macro=$got expected=$expected result=$ResultPath"
    }
    Log "CHECK OK seed=$Seed | count_full_cold_item_macro=$got | result=$ResultPath"
}

function Wants-Model([string]$Name) {
    foreach ($m in $Models) {
        if ($m -eq "All" -or $m -eq $Name) {
            return $true
        }
    }
    return $false
}

function Run-Model(
    [int]$Seed,
    [string]$Name,
    [string]$ScriptName,
    [string]$OutSubdir,
    [string]$ResultFile,
    [scriptblock]$Configure,
    [string]$CkptSubdir = ""
) {
    if (-not (Wants-Model $Name)) {
        return
    }

    $splitDir = Assert-Split $Seed
    $outDir = Join-Path $splitDir $OutSubdir
    $ckptDir = if ($CkptSubdir) {
        Join-Path $CheckpointRootAbs ("{0}\strict_item_cold_balanced_thr1_seed_{1}" -f $CkptSubdir, $Seed)
    } else {
        ""
    }
    $resultPath = Join-Path $outDir $ResultFile

    if ((Test-Path -LiteralPath $resultPath) -and (-not $Force)) {
        Log "SKIP $Name seed=$Seed | exists=$resultPath"
        Assert-StrictResult $Seed $resultPath
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
        throw "$Name failed for seed=$Seed with exit=$($p.ExitCode); see $logPath"
    }
    Assert-StrictResult $Seed $resultPath
}

Log "QUEUE START Junyi baseline strictfix | seeds=$($Seeds -join ',') | models=$($Models -join ',')"

foreach ($seed in $Seeds) {
    Run-Model $seed "Popularity" "popularity_static.py" "popularity_compare_strictfix" "popularity_static_result.json" {
        $env:CUDA_VISIBLE_DEVICES = ""
        $env:POP_DEVICE = "cpu"
        $env:POP_STATIC_SEED = "$seed"
        $env:POP_SEED = "$seed"
        $env:POP_COLD_THRESHOLD = "1"
        $env:POP_EVAL_N_NEG = "200"
    }
}

foreach ($seed in $Seeds) {
    Run-Model $seed "ContentProfile" "content_profile_static_hin.py" "content_profile_compare_strictfix" "content_profile_static_result.json" {
        $env:CUDA_VISIBLE_DEVICES = ""
        $env:CONTENT_PROFILE_DEVICE = "cpu"
        $env:CONTENT_PROFILE_BATCH_SIZE = "512"
        $env:CONTENT_PROFILE_STATIC_SEED = "$seed"
        $env:CONTENT_PROFILE_SEED = "$seed"
        $env:CONTENT_PROFILE_COLD_THRESHOLD = "1"
        $env:CONTENT_PROFILE_EVAL_N_NEG = "200"
    }
}

foreach ($seed in $Seeds) {
    Run-Model $seed "BPR" "bpr_static_fair.py" "bpr_compare_strictfix" "bpr_static_result.json" {
        $env:BPR_STATIC_EPOCHS = "200"
        $env:BPR_EVAL_INTERVAL = "10"
        $env:BPR_BATCH_SIZE = "4096"
        $env:BPR_EVAL_N_NEG = "200"
        $env:BPR_COLD_THRESHOLD = "1"
        $env:BPR_STATIC_SEED = "$seed"
        $env:BPR_SEED = "$seed"
        $env:BPR_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:BPR_SAVE_CKPT = "1"
        $env:BPR_SAVE_OPT_STATE = "1"
        $env:BPR_AUTO_RESUME = "0"
        $env:BPR_FORCE_FRESH = "1"
    } "bpr_compare"
}

foreach ($seed in $Seeds) {
    Run-Model $seed "LightGCN" "lightgcn_static_hin_fair.py" "lightgcn_compare_strictfix" "lightgcn_static_result.json" {
        $env:LIGHTGCN_STATIC_EPOCHS = "60"
        $env:LIGHTGCN_EVAL_INTERVAL = "10"
        $env:LIGHTGCN_BATCH_SIZE = "4096"
        $env:LIGHTGCN_EVAL_N_NEG = "200"
        $env:LIGHTGCN_COLD_THRESHOLD = "1"
        $env:LIGHTGCN_STATIC_SEED = "$seed"
        $env:LIGHTGCN_SEED = "$seed"
        $env:LIGHTGCN_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:LIGHTGCN_SAVE_CKPT = "1"
        $env:LIGHTGCN_SAVE_OPT_STATE = "1"
        $env:LIGHTGCN_AUTO_RESUME = "0"
        $env:LIGHTGCN_FORCE_FRESH = "1"
    } "lightgcn_compare"
}

foreach ($seed in $Seeds) {
    Run-Model $seed "DropoutNet" "dropoutnet_official_static_hin.py" "dropoutnet_compare_strictfix" "dropoutnet_official_static_result.json" {
        $env:DROPOUT_OFFICIAL_TEACHER_EPOCHS = "80"
        $env:DROPOUT_OFFICIAL_STATIC_EPOCHS = "80"
        $env:DROPOUT_OFFICIAL_EVAL_INTERVAL = "5"
        $env:DROPOUT_OFFICIAL_BATCH_SIZE = "4096"
        $env:DROPOUT_OFFICIAL_EVAL_N_NEG = "200"
        $env:DROPOUT_OFFICIAL_ITEM_DROPOUT = "0.5"
        $env:DROPOUT_OFFICIAL_USER_DROPOUT = "0.0"
        $env:DROPOUT_OFFICIAL_COLD_THRESHOLD = "1"
        $env:DROPOUT_OFFICIAL_STATIC_SEED = "$seed"
        $env:DROPOUT_OFFICIAL_SEED = "$seed"
        $env:DROPOUT_OFFICIAL_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:DROPOUT_OFFICIAL_SAVE_CKPT = "1"
        $env:DROPOUT_OFFICIAL_SAVE_OPT_STATE = "1"
        $env:DROPOUT_OFFICIAL_AUTO_RESUME = "0"
        $env:DROPOUT_OFFICIAL_FORCE_FRESH = "1"
    } "dropoutnet_compare"
}

foreach ($seed in $Seeds) {
    Run-Model $seed "CCFCRec" "ccfc_static_hin.py" "ccfcrec_compare_strictfix" "ccfcrec_static_result.json" {
        $env:CCFCREC_STATIC_EPOCHS = "80"
        $env:CCFCREC_EVAL_INTERVAL = "5"
        $env:CCFCREC_BATCH_SIZE = "4096"
        $env:CCFCREC_EVAL_BATCH_SIZE = "4096"
        $env:CCFCREC_EVAL_ITEM_MODE = "mixed"
        $env:CCFCREC_COLD_THRESHOLD = "1"
        $env:CCFCREC_EVAL_N_NEG = "200"
        $env:CCFCREC_STATIC_SEED = "$seed"
        $env:CCFCREC_SEED = "$seed"
        $env:CCFCREC_CKPT_DIR = $env:BASELINE_CKPT_DIR
        $env:CCFCREC_SAVE_CKPT = "1"
        $env:CCFCREC_SAVE_OPT_STATE = "1"
        $env:CCFCREC_AUTO_RESUME = "0"
        $env:CCFCREC_FORCE_FRESH = "1"
    } "ccfcrec_compare"
}

foreach ($seed in $Seeds) {
    Run-Model $seed "ALDI" "aldi_static_hin.py" "aldi_compare_strictfix" "aldi_static_result.json" {
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

Log "QUEUE DONE Junyi baseline strictfix | seeds=$($Seeds -join ',') | models=$($Models -join ',')"
