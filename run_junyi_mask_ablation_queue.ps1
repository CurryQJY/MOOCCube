param(
    [int[]]$WaitPids = @(),
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$PollSeconds = 300,
    [switch]$RunFreshTrueTrueAll,
    [switch]$FalseFalseSeed2025Only,
    [switch]$Seed2025OnlyBothMasks
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

$QueueDir = Join-Path $Repo "outputs\junyi\mask_ablation\_queue"
$QueueLog = Join-Path $QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

function Write-QueueLogLine([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Add-Content -Path $QueueLog -Value $line -Encoding UTF8
            return
        } catch {
            Start-Sleep -Milliseconds (200 * $attempt)
        }
    }
    Add-Content -Path $QueueLog -Value $line -Encoding UTF8
}

function Wait-ForPids([int[]]$Ids) {
    $valid = @($Ids | Where-Object { $_ -gt 0 })
    if ($valid.Count -eq 0) {
        return
    }

    Write-QueueLogLine "WAIT pids=$($valid -join ',')"
    while ($true) {
        $alive = @()
        foreach ($id in $valid) {
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
                $alive += $id
            }
        }
        if ($alive.Count -eq 0) {
            Write-QueueLogLine "WAIT DONE pids=$($valid -join ',')"
            return
        }
        Write-QueueLogLine "WAIT still_alive=$($alive -join ',')"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Test-CompletedWithMask(
    [string]$OutputRoot,
    [int]$Seed,
    [bool]$MaskKnownPosNeg,
    [bool]$MaskSameItemNeg
) {
    $split = Join-Path $OutputRoot ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
    $result = Join-Path $split "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $manifest = Join-Path $split "static_protocol_manifest.json"
    if (-not (Test-Path -LiteralPath $result) -or -not (Test-Path -LiteralPath $manifest)) {
        return $false
    }
    try {
        $json = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifest | ConvertFrom-Json
        return (($json.model_config.mask_known_pos_neg -eq $MaskKnownPosNeg) -and
                ($json.model_config.mask_same_item_neg -eq $MaskSameItemNeg))
    } catch {
        return $false
    }
}

function Run-OursMask(
    [string]$Label,
    [bool]$MaskKnownPosNeg,
    [bool]$MaskSameItemNeg,
    [int[]]$Seeds,
    [string]$OutputRoot,
    [string]$CheckpointRoot
) {
    $outputRootAbs = Resolve-RunPath $Repo $OutputRoot
    $checkpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
    $runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"

    New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
    New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

    foreach ($seed in $Seeds) {
        if (Test-CompletedWithMask $outputRootAbs $seed $MaskKnownPosNeg $MaskSameItemNeg) {
            Write-QueueLogLine "SKIP $Label seed=$seed | mask_known=$MaskKnownPosNeg mask_same=$MaskSameItemNeg | exists=$outputRootAbs"
            continue
        }

        Write-QueueLogLine "START $Label seed=$seed | mask_known=$MaskKnownPosNeg mask_same=$MaskSameItemNeg | out=$outputRootAbs"
        $exitCode = 0
        try {
            & $runner `
                -PythonRunner ".\py.bat" `
                -DataDir "processed_data_junyi" `
                -RelationDir "processed_data_junyi\relations" `
                -OutputRoot $outputRootAbs `
                -CheckpointRoot $checkpointRootAbs `
                -Protocol strict_item_cold_balanced `
                -ColdThresholds 1 `
                -Seeds $seed `
                -Epochs 60 `
                -Patience 60 `
                -EarlyStopAverageMode item_macro `
                -UseContentDelta:$false `
                -UsePseudoColdTrain:$false `
                -RunSampledEval:$false `
                -MaskKnownPosNeg:$MaskKnownPosNeg `
                -MaskSameItemNeg:$MaskSameItemNeg `
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
            $errLog = Join-Path $QueueDir ("{0}_seed{1}_error.log" -f $Label, $seed)
            $_ | Out-File -FilePath $errLog -Append -Encoding UTF8
        }

        Write-QueueLogLine "END $Label seed=$seed | exit=$exitCode"
        if ($exitCode -ne 0) {
            throw "$Label failed for seed=$seed with exit=$exitCode"
        }

        if (-not (Test-CompletedWithMask $outputRootAbs $seed $MaskKnownPosNeg $MaskSameItemNeg)) {
            throw "$Label seed=$seed completed but manifest/result did not match requested mask flags"
        }
    }
}

Write-QueueLogLine "QUEUE START Junyi Ours mask ablation"
Wait-ForPids $WaitPids

if ($FalseFalseSeed2025Only) {
    Run-OursMask "mask_ff" $false $false @(2025) `
        "outputs\junyi\mask_ablation\mask_ff" `
        "checkpoints\junyi\mask_ablation\mask_ff"
    Write-QueueLogLine "QUEUE DONE Junyi Ours mask ablation | false_false_seed2025_only"
    return
}

if ($Seed2025OnlyBothMasks) {
    Run-OursMask "mask_tt" $true $true @(2025) `
        "outputs\junyi\mask_ablation\mask_tt" `
        "checkpoints\junyi\mask_ablation\mask_tt"
    Run-OursMask "mask_ff" $false $false @(2025) `
        "outputs\junyi\mask_ablation\mask_ff" `
        "checkpoints\junyi\mask_ablation\mask_ff"
    Write-QueueLogLine "QUEUE DONE Junyi Ours mask ablation | seed2025_only_both_masks"
    return
}

if ($RunFreshTrueTrueAll) {
    Run-OursMask "mask_tt" $true $true @(2025, 2026, 2027) `
        "outputs\junyi\mask_ablation\mask_tt" `
        "checkpoints\junyi\mask_ablation\mask_tt"
} else {
    Run-OursMask "mask_tt" $true $true @(2025) `
        "outputs\junyi\mask_ablation\mask_tt" `
        "checkpoints\junyi\mask_ablation\mask_tt"
}

Run-OursMask "mask_ff" $false $false @(2025, 2026, 2027) `
    "outputs\junyi\mask_ablation\mask_ff" `
    "checkpoints\junyi\mask_ablation\mask_ff"

$analysisLog = Join-Path $QueueDir "analyze.log"
Write-QueueLogLine "START analyze mask ablation"
& .\py.bat -B "analyze_junyi_mask_ablation.py" *> $analysisLog
$analysisExit = $LASTEXITCODE
Write-QueueLogLine "END analyze mask ablation | exit=$analysisExit | log=$analysisLog"
if ($analysisExit -ne 0) {
    throw "Mask ablation analysis failed with exit=$analysisExit"
}

Write-QueueLogLine "QUEUE DONE Junyi Ours mask ablation"
