param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [ValidateSet("Both", "TrueTrue", "FalseFalse")]
    [string]$MaskSet = "Both",
    [int[]]$WaitPids = @(),
    [int]$PollSeconds = 300,
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [switch]$Seed2025Only,
    [switch]$NoAnalyze,
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

function Convert-ToStrictBool($Value) {
    if ($Value -is [bool]) {
        return $Value
    }
    if ($Value -is [int]) {
        return ($Value -ne 0)
    }
    if ($Value -is [string]) {
        return ($Value -match "^(1|true|yes)$")
    }
    return [bool]$Value
}

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    foreach ($path in @($script:MasterLog, $script:LatestLog)) {
        Add-Content -LiteralPath $path -Value $line -Encoding UTF8
    }
    Write-Host $line
}

function Wait-ForPids([int[]]$Ids) {
    $valid = @($Ids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($valid.Count -eq 0) {
        return
    }

    Write-Log "WAIT pids=$($valid -join ',') poll_seconds=$PollSeconds"
    while ($true) {
        $alive = @()
        foreach ($id in $valid) {
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
                $alive += $id
            }
        }
        if ($alive.Count -eq 0) {
            Write-Log "WAIT DONE pids=$($valid -join ',')"
            return
        }
        Write-Log "WAIT still_alive=$($alive -join ',')"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-SplitDir([string]$OutputRoot, [int]$Seed) {
    return (Join-Path $OutputRoot ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed))
}

function Test-CompletedWithMask(
    [string]$OutputRoot,
    [int]$Seed,
    [bool]$MaskKnownPosNeg,
    [bool]$MaskSameItemNeg
) {
    $split = Get-SplitDir $OutputRoot $Seed
    $result = Join-Path $split "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $manifest = Join-Path $split "static_protocol_manifest.json"
    if (-not (Test-Path -LiteralPath $result) -or -not (Test-Path -LiteralPath $manifest)) {
        return $false
    }

    try {
        $json = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifest | ConvertFrom-Json
        $cfg = $json.model_config
        if ($null -eq $cfg) {
            return $false
        }
        $actualKnown = Convert-ToStrictBool $cfg.mask_known_pos_neg
        $actualSame = Convert-ToStrictBool $cfg.mask_same_item_neg
        return (($actualKnown -eq $MaskKnownPosNeg) -and ($actualSame -eq $MaskSameItemNeg))
    } catch {
        return $false
    }
}

function Read-DoubleField($Row, [string]$Name) {
    $value = $Row.$Name
    if ([string]::IsNullOrWhiteSpace([string]$value)) {
        return $null
    }
    return [double]::Parse([string]$value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Read-RunMetric(
    [string]$OutputRoot,
    [int]$Seed,
    [string]$Condition,
    [bool]$MaskKnownPosNeg,
    [bool]$MaskSameItemNeg
) {
    $split = Get-SplitDir $OutputRoot $Seed
    $result = Join-Path $split "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if (-not (Test-CompletedWithMask $OutputRoot $Seed $MaskKnownPosNeg $MaskSameItemNeg)) {
        return $null
    }
    $row = Import-Csv -LiteralPath $result | Select-Object -First 1
    return [pscustomobject]@{
        run_stamp = $script:RunStamp
        condition = $Condition
        seed = $Seed
        mask_known_pos_neg = $MaskKnownPosNeg
        mask_same_item_neg = $MaskSameItemNeg
        source_dir = $split
        full_cold_item_macro_r10 = Read-DoubleField $row "full_cold_item_macro_r10"
        full_cold_item_macro_n10 = Read-DoubleField $row "full_cold_item_macro_n10"
        full_cold_item_macro_n20 = Read-DoubleField $row "full_cold_item_macro_n20"
        full_hot_item_macro_n10 = Read-DoubleField $row "full_hot_item_macro_n10"
        full_cold_n10 = Read-DoubleField $row "full_cold_n10"
        full_hot_n10 = Read-DoubleField $row "full_hot_n10"
    }
}

function Get-Mean([double[]]$Values) {
    if ($Values.Count -eq 0) {
        return $null
    }
    return (($Values | Measure-Object -Average).Average)
}

function Get-Std([double[]]$Values) {
    if ($Values.Count -le 1) {
        return 0.0
    }
    $mean = Get-Mean $Values
    $sumSq = 0.0
    foreach ($value in $Values) {
        $sumSq += [math]::Pow(($value - $mean), 2)
    }
    return [math]::Sqrt($sumSq / ($Values.Count - 1))
}

function Write-Summary([object[]]$Conditions, [int[]]$Seeds) {
    $detailRows = @()
    foreach ($condition in $Conditions) {
        foreach ($seed in $Seeds) {
            $row = Read-RunMetric `
                -OutputRoot $condition.OutputRootAbs `
                -Seed $seed `
                -Condition $condition.Label `
                -MaskKnownPosNeg $condition.MaskKnownPosNeg `
                -MaskSameItemNeg $condition.MaskSameItemNeg
            if ($null -ne $row) {
                $detailRows += $row
            }
        }
    }

    if ($detailRows.Count -eq 0) {
        Write-Log "SUMMARY skipped: no completed matching runs found"
        return
    }

    $detailPath = Join-Path $script:LogDir ("overnight_detail_{0}.csv" -f $script:RunStamp)
    $detailLatest = Join-Path $script:LogDir "overnight_detail_latest.csv"
    $summaryPath = Join-Path $script:LogDir ("overnight_summary_{0}.csv" -f $script:RunStamp)
    $summaryLatest = Join-Path $script:LogDir "overnight_summary_latest.csv"
    $comparePath = Join-Path $script:LogDir ("overnight_compare_{0}.csv" -f $script:RunStamp)
    $compareLatest = Join-Path $script:LogDir "overnight_compare_latest.csv"

    $detailRows | Export-Csv -LiteralPath $detailPath -NoTypeInformation -Encoding UTF8
    $detailRows | Export-Csv -LiteralPath $detailLatest -NoTypeInformation -Encoding UTF8

    $metricNames = @(
        "full_cold_item_macro_r10",
        "full_cold_item_macro_n10",
        "full_cold_item_macro_n20",
        "full_hot_item_macro_n10",
        "full_cold_n10",
        "full_hot_n10"
    )

    $summaryRows = @()
    foreach ($conditionName in @($detailRows.condition | Select-Object -Unique)) {
        $rows = @($detailRows | Where-Object { $_.condition -eq $conditionName })
        $summary = [ordered]@{
            run_stamp = $script:RunStamp
            condition = $conditionName
            seeds = (($rows | ForEach-Object { $_.seed }) -join ";")
            runs = $rows.Count
        }
        foreach ($metric in $metricNames) {
            $values = [double[]]@($rows | ForEach-Object { $_.$metric } | Where-Object { $null -ne $_ })
            $summary["${metric}_mean"] = Get-Mean $values
            $summary["${metric}_std"] = Get-Std $values
        }
        $summaryRows += [pscustomobject]$summary
    }

    $summaryRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
    $summaryRows | Export-Csv -LiteralPath $summaryLatest -NoTypeInformation -Encoding UTF8

    $compareRows = @()
    foreach ($seed in $Seeds) {
        $tt = $detailRows | Where-Object { $_.condition -eq "mask_tt" -and $_.seed -eq $seed } | Select-Object -First 1
        $ff = $detailRows | Where-Object { $_.condition -eq "mask_ff" -and $_.seed -eq $seed } | Select-Object -First 1
        if ($null -eq $tt -or $null -eq $ff) {
            continue
        }
        $deltaN10 = $tt.full_cold_item_macro_n10 - $ff.full_cold_item_macro_n10
        $relPctN10 = if ($ff.full_cold_item_macro_n10 -ne 0) { (($tt.full_cold_item_macro_n10 / $ff.full_cold_item_macro_n10) - 1.0) * 100.0 } else { $null }
        $compareRows += [pscustomobject]@{
            run_stamp = $script:RunStamp
            seed = $seed
            tt_full_cold_item_macro_n10 = $tt.full_cold_item_macro_n10
            ff_full_cold_item_macro_n10 = $ff.full_cold_item_macro_n10
            delta_tt_minus_ff_n10 = $deltaN10
            rel_pct_tt_minus_ff_n10 = $relPctN10
            tt_full_cold_item_macro_r10 = $tt.full_cold_item_macro_r10
            ff_full_cold_item_macro_r10 = $ff.full_cold_item_macro_r10
            delta_tt_minus_ff_r10 = ($tt.full_cold_item_macro_r10 - $ff.full_cold_item_macro_r10)
        }
    }

    if ($compareRows.Count -gt 0) {
        $compareRows | Export-Csv -LiteralPath $comparePath -NoTypeInformation -Encoding UTF8
        $compareRows | Export-Csv -LiteralPath $compareLatest -NoTypeInformation -Encoding UTF8
    }

    foreach ($row in $summaryRows) {
        Write-Log ("SUMMARY {0} runs={1} cold_item_N@10={2:N4}+/-{3:N4} cold_item_R@10={4:N4}+/-{5:N4}" -f `
            $row.condition, `
            $row.runs, `
            $row.full_cold_item_macro_n10_mean, `
            $row.full_cold_item_macro_n10_std, `
            $row.full_cold_item_macro_r10_mean, `
            $row.full_cold_item_macro_r10_std)
    }
    Write-Log "SUMMARY files detail=$detailLatest summary=$summaryLatest compare=$compareLatest"
}

function Run-OursMask(
    [string]$Label,
    [bool]$MaskKnownPosNeg,
    [bool]$MaskSameItemNeg,
    [int[]]$RunSeeds,
    [string]$OutputRoot,
    [string]$CheckpointRoot
) {
    $runner = Join-Path $script:Repo "run_usim_feedback_fast3_content_delta_static.ps1"
    $outputRootAbs = Resolve-RunPath $script:Repo $OutputRoot
    $checkpointRootAbs = Resolve-RunPath $script:Repo $CheckpointRoot

    New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
    New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

    foreach ($seed in $RunSeeds) {
        $split = Get-SplitDir $outputRootAbs $seed
        if (Test-CompletedWithMask $outputRootAbs $seed $MaskKnownPosNeg $MaskSameItemNeg) {
            $metric = Read-RunMetric $outputRootAbs $seed $Label $MaskKnownPosNeg $MaskSameItemNeg
            Write-Log ("SKIP {0} seed={1} mask_known={2} mask_same={3} cold_item_N@10={4:N4} dir={5}" -f `
                $Label, $seed, $MaskKnownPosNeg, $MaskSameItemNeg, $metric.full_cold_item_macro_n10, $split)
            continue
        }

        Write-Log "START $Label seed=$seed mask_known=$MaskKnownPosNeg mask_same=$MaskSameItemNeg out=$split"
        if ($DryRun) {
            Write-Log "DRYRUN $Label seed=$seed"
            continue
        }

        $runnerParams = @{
            PythonRunner = ".\py.bat"
            DataDir = "processed_data_junyi"
            RelationDir = "processed_data_junyi\relations"
            OutputRoot = $outputRootAbs
            CheckpointRoot = $checkpointRootAbs
            Protocol = "strict_item_cold_balanced"
            ColdThresholds = @(1)
            Seeds = @($seed)
            Epochs = $Epochs
            Patience = $Patience
            EarlyStopAverageMode = "item_macro"
            UseContentDelta = $false
            UsePseudoColdTrain = $false
            RunSampledEval = $false
            MaskKnownPosNeg = $MaskKnownPosNeg
            MaskSameItemNeg = $MaskSameItemNeg
            SaveCkpt = $true
            AutoResume = $false
            ForceFresh = $true
            SaveOptState = $true
            SkipAggregate = $true
        }

        & $runner @runnerParams
        $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
        Write-Log "END $Label seed=$seed exit=$exitCode"
        if ($exitCode -ne 0) {
            throw "$Label seed=$seed failed with exit=$exitCode"
        }
        if (-not (Test-CompletedWithMask $outputRootAbs $seed $MaskKnownPosNeg $MaskSameItemNeg)) {
            throw "$Label seed=$seed finished but result/manifest did not match requested mask flags"
        }

        $metric = Read-RunMetric $outputRootAbs $seed $Label $MaskKnownPosNeg $MaskSameItemNeg
        Write-Log ("METRIC {0} seed={1} cold_item_N@10={2:N4} cold_item_R@10={3:N4} hot_item_N@10={4:N4}" -f `
            $Label, $seed, $metric.full_cold_item_macro_n10, $metric.full_cold_item_macro_r10, $metric.full_hot_item_macro_n10)
    }
}

try {
    $script:Repo = (Resolve-Path -LiteralPath $Repo).Path
    Set-Location $script:Repo

    if ($Seed2025Only) {
        $Seeds = @(2025)
    }
    $Seeds = @($Seeds | Sort-Object -Unique)

    $script:RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $script:LogDir = Join-Path $script:Repo "outputs\junyi\mask_ablation\_overnight"
    New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
    $script:MasterLog = Join-Path $script:LogDir ("overnight_{0}.log" -f $script:RunStamp)
    $script:LatestLog = Join-Path $script:LogDir "overnight_latest.log"
    Set-Content -LiteralPath $script:LatestLog -Value "" -Encoding UTF8
    Set-Content -LiteralPath $script:MasterLog -Value "" -Encoding UTF8

    $allConditions = @(
        [pscustomobject]@{
            Label = "mask_tt"
            MaskKnownPosNeg = $true
            MaskSameItemNeg = $true
            OutputRoot = "outputs\junyi\mask_ablation\mask_tt"
            CheckpointRoot = "checkpoints\junyi\mask_ablation\mask_tt"
        },
        [pscustomobject]@{
            Label = "mask_ff"
            MaskKnownPosNeg = $false
            MaskSameItemNeg = $false
            OutputRoot = "outputs\junyi\mask_ablation\mask_ff"
            CheckpointRoot = "checkpoints\junyi\mask_ablation\mask_ff"
        }
    )

    $conditions = switch ($MaskSet) {
        "TrueTrue" { @($allConditions | Where-Object { $_.Label -eq "mask_tt" }) }
        "FalseFalse" { @($allConditions | Where-Object { $_.Label -eq "mask_ff" }) }
        default { $allConditions }
    }

    foreach ($condition in $conditions) {
        $condition | Add-Member -NotePropertyName OutputRootAbs -NotePropertyValue (Resolve-RunPath $script:Repo $condition.OutputRoot) -Force
        $condition | Add-Member -NotePropertyName CheckpointRootAbs -NotePropertyValue (Resolve-RunPath $script:Repo $condition.CheckpointRoot) -Force
    }

    Write-Log "OVERNIGHT START repo=$script:Repo mask_set=$MaskSet seeds=$($Seeds -join ',') epochs=$Epochs patience=$Patience dry_run=$DryRun"
    Write-Log "LOG master=$script:MasterLog latest=$script:LatestLog"
    Wait-ForPids $WaitPids

    foreach ($condition in $conditions) {
        Run-OursMask `
            -Label $condition.Label `
            -MaskKnownPosNeg $condition.MaskKnownPosNeg `
            -MaskSameItemNeg $condition.MaskSameItemNeg `
            -RunSeeds $Seeds `
            -OutputRoot $condition.OutputRoot `
            -CheckpointRoot $condition.CheckpointRoot
    }

    if (-not $NoAnalyze) {
        Write-Summary $conditions $Seeds
    }

    Write-Log "OVERNIGHT DONE"
    exit 0
} catch {
    Write-Log "OVERNIGHT FAILED: $($_.Exception.Message)"
    exit 1
}
