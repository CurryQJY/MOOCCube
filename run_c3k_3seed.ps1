param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 40,
    [int]$Patience = 6,
    [int]$BatchSize = 512,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$python = Join-Path $repoPath "py.bat"
$entryScript = Join-Path $repoPath "c3k_static.py"
$outputRoot = Join-Path $repoPath "outputs\c3k\full_3seed"
$checkpointRoot = Join-Path $repoPath "checkpoints\c3k\full_3seed"
$logRoot = Join-Path $repoPath "background_logs\c3k_full_3seed"

if ($Epochs -lt 2) {
    throw "Epochs must be at least 2 so stable-period timing can exclude epoch 1."
}
if ($Patience -lt 1 -or $BatchSize -lt 2) {
    throw "Patience must be positive and BatchSize must be at least 2."
}
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $entryScript)) {
    throw "C3K entry point or Python runner is missing under $repoPath."
}

$trackedEnvironment = @(
    "USIM_DATA_DIR",
    "USIM_RELATION_DIR",
    "USIM_STATIC",
    "USIM_STATIC_SPLIT_MODE",
    "USIM_STATIC_SEED",
    "USIM_SEED",
    "USIM_COLD_THRESHOLD",
    "USIM_STATIC_TRAIN_RATIO",
    "USIM_STATIC_VAL_RATIO",
    "USIM_STATIC_COLD_ITEM_RATIO",
    "USIM_STATIC_VAL_COLD_ITEM_RATIO",
    "USIM_STATIC_COLD_ITEM_MIN_INTER",
    "USIM_STATIC_COLD_ITEM_FOLDS",
    "USIM_DISABLE_LLM_SCORE",
    "USIM_PREREQ_GRAPH_SOURCE",
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "C3K_SEED",
    "C3K_OUTPUT_DIR",
    "C3K_CHECKPOINT_DIR",
    "C3K_EPOCHS",
    "C3K_PATIENCE",
    "C3K_BATCH_SIZE",
    "C3K_LR",
    "C3K_PSEUDO_COLD_RATIO",
    "C3K_PSEUDO_COLD_MIN_POP",
    "C3K_GATE_MAX",
    "C3K_CONSISTENCY_WEIGHT",
    "C3K_GATE_WEIGHT",
    "C3K_TRAIN_NEGATIVES",
    "C3K_WARM_SEEN",
    "C3K_REDUNDANCY_THRESHOLD",
    "C3K_HOT_TOLERANCE",
    "C3K_MIN_DELTA",
    "C3K_ITEM_BLOCK",
    "C3K_QUERY_BLOCK",
    "C3K_TEST_HISTORY",
    "C3K_ARTIFACT_SOURCE",
    "C3K_VALIDATION_ONLY",
    "C3K_VALIDATION_MAX_ROWS",
    "C3K_DRY_RUN"
)
$originalEnvironment = @{}
foreach ($name in $trackedEnvironment) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

if ($DryRun) {
    Write-Host "C3K frozen three-seed dry run"
    Write-Host "Seeds=$($Seeds -join ',') Epochs=$Epochs Patience=$Patience BatchSize=$BatchSize"
    Write-Host "OutputRoot=$outputRoot"
    Write-Host "CheckpointRoot=$checkpointRoot"
    Write-Host "Fixed: pseudo_ratio=0.10 gate_max=0.20 consistency=0.10 gate_reg=0.001 hot_tol=0.003"
    return
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

Push-Location -LiteralPath $repoPath
try {
    foreach ($seed in $Seeds) {
        $outputDir = Join-Path $outputRoot ("seed_{0}" -f $seed)
        $checkpointDir = Join-Path $checkpointRoot ("seed_{0}" -f $seed)
        $logPath = Join-Path $logRoot ("seed_{0}.log" -f $seed)
        if ((Test-Path -LiteralPath $outputDir) -or (Test-Path -LiteralPath $checkpointDir)) {
            throw "Refusing to overwrite C3K seed $seed. Choose a new output root."
        }

        $environment = @{
            "USIM_DATA_DIR" = "processed_data_hin_clean_pop5"
            "USIM_RELATION_DIR" = "MOOCCube/relations"
            "USIM_STATIC" = "1"
            "USIM_STATIC_SPLIT_MODE" = "strict_item_cold_balanced"
            "USIM_STATIC_SEED" = [string]$seed
            "USIM_SEED" = [string]$seed
            "USIM_COLD_THRESHOLD" = "1"
            "USIM_STATIC_TRAIN_RATIO" = "0.8"
            "USIM_STATIC_VAL_RATIO" = "0.1"
            "USIM_STATIC_COLD_ITEM_RATIO" = "0.10"
            "USIM_STATIC_VAL_COLD_ITEM_RATIO" = "0.05"
            "USIM_STATIC_COLD_ITEM_MIN_INTER" = "5"
            "USIM_STATIC_COLD_ITEM_FOLDS" = "20"
            "USIM_DISABLE_LLM_SCORE" = "1"
            "USIM_PREREQ_GRAPH_SOURCE" = "concept"
            "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
            "C3K_SEED" = [string]$seed
            "C3K_OUTPUT_DIR" = $outputDir
            "C3K_CHECKPOINT_DIR" = $checkpointDir
            "C3K_EPOCHS" = [string]$Epochs
            "C3K_PATIENCE" = [string]$Patience
            "C3K_BATCH_SIZE" = [string]$BatchSize
            "C3K_LR" = "0.0005"
            "C3K_PSEUDO_COLD_RATIO" = "0.10"
            "C3K_PSEUDO_COLD_MIN_POP" = "1"
            "C3K_GATE_MAX" = "0.20"
            "C3K_CONSISTENCY_WEIGHT" = "0.10"
            "C3K_GATE_WEIGHT" = "0.001"
            "C3K_TRAIN_NEGATIVES" = "16"
            "C3K_WARM_SEEN" = "5"
            "C3K_REDUNDANCY_THRESHOLD" = "0.70"
            "C3K_HOT_TOLERANCE" = "0.003"
            "C3K_MIN_DELTA" = "0.0001"
            "C3K_ITEM_BLOCK" = "128"
            "C3K_QUERY_BLOCK" = "128"
            "C3K_TEST_HISTORY" = "train_only"
            "C3K_ARTIFACT_SOURCE" = "all_metadata"
            "C3K_VALIDATION_ONLY" = "0"
            "C3K_VALIDATION_MAX_ROWS" = "0"
            "C3K_DRY_RUN" = "0"
        }
        foreach ($pair in $environment.GetEnumerator()) {
            Set-Item "Env:$($pair.Key)" ([string]$pair.Value)
        }

        Write-Host "[C3K] starting seed=$seed output=$outputDir"
        & $python -u $entryScript 2>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "C3K seed $seed failed; inspect $logPath"
        }
    }
}
finally {
    foreach ($name in $trackedEnvironment) {
        if ($null -eq $originalEnvironment[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $originalEnvironment[$name]
        }
    }
    Pop-Location
}
