param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputDir = "paper_aaai27\baseline_sources\_gar_coldrec_strict\mooccube_seed2025_single",
    [int]$Seed = 2025,
    [int]$ColdThreshold = 1,
    [int]$MFEpochs = 5,
    [int]$GAREpochs = 10,
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

function Invoke-Logged([scriptblock]$Command, [string]$LogPath, [string]$FailureMessage) {
    & $Command *> $LogPath
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        throw "$FailureMessage exit=$exit. See $LogPath"
    }
}

$RepoAbs = Resolve-RunPath (Get-Location).Path $Repo
$PythonRunnerAbs = Resolve-RunPath $RepoAbs $PythonRunner
$ColdRecRootAbs = Resolve-RunPath $RepoAbs $ColdRecRoot
$OutputDirAbs = Resolve-RunPath $RepoAbs $OutputDir
$SplitName = "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $Seed
$SplitDir = Join-Path $RepoAbs ("outputs\content_delta_pop5\static_item_cold_balanced\{0}" -f $SplitName)
$DataDir = Join-Path $RepoAbs "processed_data_hin_clean_pop5"
$DatasetName = "gar_mfpre_mooccube_{0}" -f $SplitName
$Adapter = Join-Path $RepoAbs "gar_coldrec_static.py"
$ResultPath = Join-Path $OutputDirAbs "gar_coldrec_strict_result.json"
$MFLog = Join-Path $OutputDirAbs "mf_backbone.log"
$GARLog = Join-Path $OutputDirAbs "gar_training.log"
$ExportLog = Join-Path $OutputDirAbs "export_dataset.log"
$MFUser = Join-Path $ColdRecRootAbs ("emb\{0}_cold_item_MF_user_emb.pt" -f $DatasetName)
$MFItem = Join-Path $ColdRecRootAbs ("emb\{0}_cold_item_MF_item_emb.pt" -f $DatasetName)

Write-Host "ColdRec GAR strict single-seed"
Write-Host "dataset=mooccube seed=$Seed"
Write-Host "split=$SplitName"
Write-Host "data=processed_data_hin_clean_pop5"
Write-Host "adapter=gar_coldrec_static.py"
Write-Host "MF epochs=$MFEpochs"
Write-Host "GAR epochs=$GAREpochs"
Write-Host "history=train_only"
Write-Host "use_gpu=$UseGpu gpu_id=$GpuId"
Write-Host "result=gar_coldrec_strict_result.json"
Write-Host "STAGE 1 MF"
Write-Host "STAGE 2 GAR"

if ($DryRun) {
    Write-Host "DRY RUN output=$OutputDirAbs coldrec_dataset=$DatasetName"
    exit 0
}

foreach ($path in @($RepoAbs, $PythonRunnerAbs, $ColdRecRootAbs, $Adapter, $DataDir, $SplitDir)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required path: $path"
    }
}
foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl", "static_split_summary.json")) {
    $path = Join-Path $SplitDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing strict split artifact: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirAbs | Out-Null

if ((-not $Force) -and (Test-Path -LiteralPath $ResultPath)) {
    Write-Host "Existing GAR result retained: $ResultPath"
    exit 0
}

Write-Host "STAGE 0 EXPORT | log=$ExportLog"
$exportCommand = {
    Push-Location $RepoAbs
    try {
        & $PythonRunnerAbs -u $Adapter `
            --data-dir $DataDir `
            --split-dir $SplitDir `
            --output-dir $OutputDirAbs `
            --coldrec-root $ColdRecRootAbs `
            --dataset-name $DatasetName `
            --cold-threshold $ColdThreshold `
            --seed $Seed `
            --prepare-only
    }
    finally {
        Pop-Location
    }
}
Invoke-Logged $exportCommand $ExportLog "ColdRec GAR dataset export failed."

if ($Force -or (-not (Test-Path -LiteralPath $MFUser)) -or (-not (Test-Path -LiteralPath $MFItem))) {
    Write-Host "STAGE 1 MF | log=$MFLog"
    $mfCommand = {
        Push-Location $ColdRecRootAbs
        try {
            & $PythonRunnerAbs -u "main.py" `
                --dataset $DatasetName `
                --model "MF" `
                --cold_object "item" `
                --epochs $MFEpochs `
                --topN $TopN `
                --bs $BatchSize `
                --emb_size $EmbSize `
                --lr $MFLearningRate `
                --reg $MFReg `
                --runs 1 `
                --seed $Seed `
                --use_gpu $(if ($UseGpu) { "true" } else { "false" }) `
                --save_emb true `
                --gpu_id $GpuId `
                --early_stop $EarlyStop `
                --eval_every $EvalEvery `
                --result_file (Join-Path $OutputDirAbs "coldrec_native_mf_result.txt") `
                --result_overwrite
        }
        finally {
            Pop-Location
        }
    }
    Invoke-Logged $mfCommand $MFLog "ColdRec MF teacher failed."
}
else {
    Write-Host "STAGE 1 MF | reuse=$MFUser"
}

foreach ($path in @($MFUser, $MFItem)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing MF teacher artifact after stage 1: $path"
    }
}

Write-Host "STAGE 2 GAR | log=$GARLog"
$garCommand = {
    Push-Location $RepoAbs
    try {
        & $PythonRunnerAbs -u $Adapter `
            --data-dir $DataDir `
            --split-dir $SplitDir `
            --output-dir $OutputDirAbs `
            --coldrec-root $ColdRecRootAbs `
            --dataset-name $DatasetName `
            --cold-threshold $ColdThreshold `
            --seed $Seed `
            --epochs $GAREpochs `
            --emb-size $EmbSize `
            --batch-size $BatchSize `
            --lr $GARLearningRate `
            --reg $GARReg `
            --topn $TopN `
            --gpu-id $GpuId `
            --early-stop $EarlyStop `
            --eval-every $EvalEvery `
            --eval-batch-size $EvalBatchSize `
            --test-history "train_only" `
            --backbone "MF" `
            --alpha $Alpha `
            --beta $Beta `
            $(if ($UseGpu) { "--use-gpu" } else { "--no-use-gpu" })
    }
    finally {
        Pop-Location
    }
}
Invoke-Logged $garCommand $GARLog "ColdRec GAR strict run failed."

if (-not (Test-Path -LiteralPath $ResultPath)) {
    throw "GAR completed without expected result: $ResultPath"
}

Write-Host "GAR ColdRec single-seed complete: $ResultPath"
