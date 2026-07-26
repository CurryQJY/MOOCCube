param(
    [int[]]$Seeds = @(2026, 2027),
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\junyi\main_table_3seed",
    [string]$CheckpointRoot = "checkpoints\junyi\main_table_3seed"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $Repo

$OutputRootAbs = Join-Path $Repo $OutputRoot
$CheckpointRootAbs = Join-Path $Repo $CheckpointRoot
$QueueDir = Join-Path $OutputRootAbs "_queue_lowcost"
$QueueLog = Join-Path $QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

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

function Ensure-Split([int]$Seed) {
    $splitDir = Split-Dir $Seed
    $trainPath = Join-Path $splitDir "static_train.pkl"
    $valPath = Join-Path $splitDir "static_val.pkl"
    $testPath = Join-Path $splitDir "static_test.pkl"
    if ((Test-Path $trainPath) -and (Test-Path $valPath) -and (Test-Path $testPath)) {
        Log "SKIP split seed=$Seed | exists=$splitDir"
        return $splitDir
    }

    New-Item -ItemType Directory -Force -Path $splitDir | Out-Null
    Log "START split seed=$Seed | out=$splitDir"

    $env:USIM_DATA_DIR = "processed_data_junyi"
    $env:USIM_STATIC = "1"
    $env:USIM_STATIC_SPLIT_MODE = "strict_item_cold_balanced"
    $env:USIM_STATIC_SEED = "$Seed"
    $env:USIM_SEED = "$Seed"
    $env:USIM_COLD_THRESHOLD = "1"
    $env:USIM_STATIC_TRAIN_RATIO = "0.8"
    $env:USIM_STATIC_VAL_RATIO = "0.1"
    $env:USIM_STATIC_COLD_ITEM_RATIO = "0.10"
    $env:USIM_STATIC_VAL_COLD_ITEM_RATIO = "0.05"
    $env:USIM_STATIC_COLD_ITEM_MIN_INTER = "5"
    $env:USIM_STATIC_COLD_ITEM_FOLDS = "20"
    $env:USIM_STATIC_ARTIFACT_SOURCE = "all_metadata"
    $env:USIM_STATIC_EXPORT_SPLIT = "1"
    $env:USIM_STATIC_TEST_HISTORY = "train_only"
    $env:USIM_FB_OUTPUT_DIR = $splitDir

    $code = @'
import os
from pathlib import Path
from types import SimpleNamespace

from hin_data_common import load_hin_processed
from fast3_delta.static_protocol import static_split_df, write_static_split_artifacts

out = Path(os.environ["USIM_FB_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
_, df, _ = load_hin_processed(os.environ.get("USIM_DATA_DIR", "processed_data_junyi"))
train_df, val_df, test_df, split_info = static_split_df(df)
cfg = SimpleNamespace(cold_threshold=int(os.environ.get("USIM_COLD_THRESHOLD", "1")))

def output_path(name: str) -> str:
    return str(out / name)

write_static_split_artifacts(train_df, val_df, test_df, split_info, cfg, output_path)
print(f"split ready: {out} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
'@
    $splitGenLog = Join-Path $splitDir "split_generation.log"
    $code | .\py.bat - *> $splitGenLog
    if ($LASTEXITCODE -ne 0) {
        throw "Split generation failed for seed=$Seed"
    }
    Log "END split seed=$Seed | out=$splitDir"
    return $splitDir
}

function Set-CommonEnv([int]$Seed, [string]$OutDir, [string]$CkptDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if ($CkptDir) {
        New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
    }

    $env:PYTHONUNBUFFERED = "1"
    $env:USIM_DATA_DIR = "processed_data_junyi"
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
    [string]$CkptSubdir = ""
) {
    $splitDir = Ensure-Split $Seed
    $outDir = Join-Path $splitDir $OutSubdir
    $ckptDir = if ($CkptSubdir) {
        Join-Path $CheckpointRootAbs ("{0}\strict_item_cold_balanced_thr1_seed_{1}" -f $CkptSubdir, $Seed)
    } else {
        ""
    }
    Set-CommonEnv $Seed $outDir $ckptDir
    & $Configure

    $resultPath = Join-Path $outDir $ResultFile
    if (Test-Path $resultPath) {
        Log "SKIP $Name seed=$Seed | exists=$resultPath"
        return
    }

    $logPath = Join-Path $outDir "run.log"
    Log "START $Name seed=$Seed | out=$outDir | ckpt=$ckptDir"
    $cmd = '/d /c "".\py.bat" -u "' + $ScriptName + '" > "' + $logPath + '" 2>&1"'
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait
    Log "END $Name seed=$Seed | exit=$($p.ExitCode) | log=$logPath"
    if ($p.ExitCode -ne 0) {
        throw "$Name failed for seed=$Seed with exit=$($p.ExitCode)"
    }
}

Log "QUEUE START Junyi main-table low-cost seeds=$($Seeds -join ',')"

# Keep the order short/low-memory first. These are all models in the docx main table.
foreach ($seed in $Seeds) {
    Run-Model $seed "Popularity" "popularity_static.py" "popularity_compare" "popularity_static_result.json" {
        $env:POP_STATIC_SEED = "$seed"
        $env:POP_SEED = "$seed"
        $env:POP_COLD_THRESHOLD = "1"
        $env:POP_EVAL_N_NEG = "200"
    }
}

foreach ($seed in $Seeds) {
    Run-Model $seed "ContentProfile" "content_profile_static_hin.py" "content_profile_compare" "content_profile_static_result.json" {
        $env:CONTENT_PROFILE_STATIC_SEED = "$seed"
        $env:CONTENT_PROFILE_SEED = "$seed"
        $env:CONTENT_PROFILE_COLD_THRESHOLD = "1"
        $env:CONTENT_PROFILE_EVAL_N_NEG = "200"
    }
}

foreach ($seed in $Seeds) {
    Run-Model $seed "BPR" "bpr_static_fair.py" "bpr_compare" "bpr_static_result.json" {
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
    Run-Model $seed "LightGCN" "lightgcn_static_hin_fair.py" "lightgcn_compare" "lightgcn_static_result.json" {
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
    Run-Model $seed "DropoutNet-official" "dropoutnet_official_static_hin.py" "dropoutnet_compare" "dropoutnet_official_static_result.json" {
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
    Run-Model $seed "CCFCRec" "ccfc_static_hin.py" "ccfcrec_compare" "ccfcrec_static_result.json" {
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

Log "QUEUE DONE Junyi main-table low-cost seeds=$($Seeds -join ',')"
