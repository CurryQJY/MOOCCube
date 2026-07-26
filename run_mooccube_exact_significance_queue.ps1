param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int[]]$WaitForPids = @(),
    [int]$GpuFreeMiB = 9000,
    [int]$PollSeconds = 60,
    [double]$MetricTolerance = 5e-5,
    [switch]$DryRun,
    [switch]$FreshLog
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location -LiteralPath $Repo

$figDir = Join-Path $Repo "paper_aaai27\figures"
$log = Join-Path $figDir "mooccube_significance_exact_ckg_cgrc_queue.log"
$childLogDir = Join-Path $figDir "mooccube_significance_exact_child_logs"
New-Item -ItemType Directory -Force -Path $figDir | Out-Null
New-Item -ItemType Directory -Force -Path $childLogDir | Out-Null
if ($FreshLog -and (Test-Path -LiteralPath $log)) {
    Remove-Item -LiteralPath $log -Force
}

function Write-QueueLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    [System.IO.File]::AppendAllText($log, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    Write-Host $line
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            return [int](([string]$raw).Trim().Split("`n")[0].Trim())
        }
    } catch {
        return $null
    }
    return $null
}

function Wait-ForPidList {
    param([int[]]$Pids, [string]$Reason)
    if ($DryRun) {
        Write-QueueLine "DRY_RUN skip PID wait for $Reason."
        return
    }
    while ($true) {
        $alive = @()
        foreach ($processId in $Pids) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                $alive += $process
            }
        }
        if ($alive.Count -lt 1) {
            break
        }
        Write-QueueLine ("Waiting for {0} PIDs: {1}" -f $Reason, (($alive | ForEach-Object { [string]$_.Id }) -join ","))
        Start-Sleep -Seconds $PollSeconds
    }
}

function Wait-ForGpuReady {
    param([string]$Reason)
    if ($DryRun) {
        Write-QueueLine "DRY_RUN skip GPU wait for $Reason."
        return
    }
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($free -ne $null -and $free -ge $GpuFreeMiB) {
            Write-QueueLine "GPU free ${free}MiB >= ${GpuFreeMiB}MiB before $Reason."
            break
        }
        Write-QueueLine "GPU not ready before $Reason; free=${free}MiB; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

function ConvertTo-PsSingleQuotedString {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function ConvertTo-PsIntArrayLiteral {
    param([int[]]$Values)
    return "@(" + (($Values | ForEach-Object { [string]$_ }) -join ",") + ")"
}

function Start-LoggedCommand {
    param(
        [string]$Name,
        [string]$CommandText,
        [string]$StdoutName,
        [string]$StderrName
    )
    $stdout = Join-Path $childLogDir $StdoutName
    $stderr = Join-Path $childLogDir $StderrName
    Write-QueueLine ("START {0}" -f $Name)
    Write-QueueLine ("COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -Command {0}" -f $CommandText)
    if ($DryRun) {
        Write-QueueLine ("DRY_RUN skip {0}" -f $Name)
        return 0
    }

    if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force }
    if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }

    $encodedCommand = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($CommandText))
    $proc = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand) `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    Write-QueueLine ("END {0} exit_code={1} stdout={2} stderr={3}" -f $Name, $proc.ExitCode, $stdout, $stderr)
    return [int]$proc.ExitCode
}

function Get-ResultMetric {
    param([string]$Path, [string]$Metric)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [double]::NaN
    }
    if ($Path.ToLowerInvariant().EndsWith(".json")) {
        $json = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
        $row = if ($json -is [array]) { $json[0] } else { $json }
        return [double]$row.full_cold_item_macro.$Metric
    }
    $frame = @(Import-Csv -LiteralPath $Path)
    if ($frame.Count -lt 1) {
        return [double]::NaN
    }
    $suffix = $Metric.ToLowerInvariant().Replace("@", "")
    $candidates = @("full_cold_item_macro_$suffix", $Metric)
    foreach ($col in $candidates) {
        if ($frame[0].PSObject.Properties.Name -contains $col) {
            return [double]$frame[0].$col
        }
    }
    return [double]::NaN
}

function Get-PerItemMetric {
    param([string]$Path, [string]$Metric)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [double]::NaN
    }
    $rows = @(Import-Csv -LiteralPath $Path)
    if ($rows.Count -lt 1) {
        return [double]::NaN
    }
    return [double](($rows | Measure-Object -Property $Metric -Average).Average)
}

function Compare-And-CopyPerItem {
    param(
        [string]$Name,
        [string]$PrimaryResult,
        [string]$TempResult,
        [string]$TempPerItem,
        [string]$TargetPerItem
    )
    $metrics = @("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
    $parts = @()
    $allMatch = $true
    foreach ($metric in $metrics) {
        $primary = Get-ResultMetric -Path $PrimaryResult -Metric $metric
        $temp = Get-ResultMetric -Path $TempResult -Metric $metric
        $item = Get-PerItemMetric -Path $TempPerItem -Metric $metric
        $dTmp = [Math]::Abs($primary - $temp)
        $dItem = [Math]::Abs($primary - $item)
        if (-not ($dTmp -le $MetricTolerance -and $dItem -le $MetricTolerance)) {
            $allMatch = $false
        }
        $parts += ("{0}:primary={1:F8},temp={2:F8},per_item={3:F8},diffs=({4:E2},{5:E2})" -f $metric, $primary, $temp, $item, $dTmp, $dItem)
    }
    $msg = ("{0} | {1}" -f $Name, ($parts -join "; "))

    if ($allMatch) {
        if ($DryRun) {
            Write-QueueLine ("DRY_RUN METRIC_MATCH would copy {0}" -f $msg)
            return $true
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TargetPerItem) | Out-Null
        Copy-Item -LiteralPath $TempPerItem -Destination $TargetPerItem -Force
        Write-QueueLine ("METRIC_MATCH copied {0} -> {1} | {2}" -f $TempPerItem, $TargetPerItem, $msg)
        return $true
    }

    Write-QueueLine ("METRIC_MISMATCH no copy | {0}" -f $msg)
    return $false
}

function SplitName {
    param([int]$Seed)
    return "strict_item_cold_balanced_thr1_seed_$Seed"
}

Write-QueueLine "QUEUE BOOT MOOCCube exact significance CGRC/CKG-RL"
if ($WaitForPids.Count -gt 0) {
    Wait-ForPidList -Pids $WaitForPids -Reason "active prerequisite queue"
}

$seedLiteral = ConvertTo-PsIntArrayLiteral -Values $Seeds
$cgrcScript = ConvertTo-PsSingleQuotedString -Value (Join-Path $Repo "run_cgrc_paper_static.ps1")

Wait-ForGpuReady -Reason "CGRC exact re-export"
$cgrcCommand = "& $cgrcScript -Seeds $seedLiteral -ColdThreshold 1 -Epochs 50 -BatchSize 4096 -DataDir 'processed_data_hin_clean_pop5' -OutputRoot 'outputs\content_delta_pop5\static_item_cold_balanced' -ResultSubdir 'significance_cgrc_exact_reexport' -BestAverageMode 'item_macro' -EvalNeg 200"
$cgrcExit = Start-LoggedCommand -Name "CGRC exact re-export" -CommandText $cgrcCommand -StdoutName "cgrc_exact_stdout.log" -StderrName "cgrc_exact_stderr.log"
if ($cgrcExit -ne 0) {
    Write-QueueLine "CGRC exact re-export failed with exit_code=$cgrcExit"
} else {
    foreach ($seed in $Seeds) {
        $split = SplitName -Seed $seed
        $primaryDir = Join-Path $Repo "outputs\content_delta_pop5\static_item_cold_balanced\$split\main_table_balanced_itemmacro_cgrc_paper_v1"
        $tempDir = Join-Path $Repo "outputs\content_delta_pop5\static_item_cold_balanced\$split\significance_cgrc_exact_reexport"
        Compare-And-CopyPerItem `
            -Name "CGRC seed=$seed" `
            -PrimaryResult (Join-Path $primaryDir "cgrc_paper_static_result.json") `
            -TempResult (Join-Path $tempDir "cgrc_paper_static_result.json") `
            -TempPerItem (Join-Path $tempDir "per_item_full_cold_cgrc_paper_static.csv") `
            -TargetPerItem (Join-Path $primaryDir "per_item_full_cold_cgrc_paper_static.csv") | Out-Null
    }
}

Wait-ForGpuReady -Reason "CKG-RL exact rerun"
$ckgOutputRoot = "outputs\significance_per_item_exports\mooccube\ckg_rl_full"
$ckgCheckpointRoot = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_full"
$ckgScript = ConvertTo-PsSingleQuotedString -Value (Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1")
$ckgOutputRootLit = ConvertTo-PsSingleQuotedString -Value $ckgOutputRoot
$ckgCheckpointRootLit = ConvertTo-PsSingleQuotedString -Value $ckgCheckpointRoot
$ckgCommand = "& $ckgScript -Protocol 'strict_item_cold_balanced' -ColdThresholds @(1) -Seeds $seedLiteral -Epochs 60 -Patience 60 -EarlyStopAverageMode 'item_macro' -EarlyStopScoreMode 'cold_only' -UseContentDelta 0 -UsePseudoColdTrain 0 -UsePaac 0 -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode 'concept' -CourseTermNorm 'none' -CourseSampleBeta 0.20 -TrainForceCold 1 -UsimSteps 5 -PpoLossWeight 1.0 -RolloutPolicy 'ppo' -UseCourseFeedback 1 -UseCourseReward 1 -UseCourseSample 1 -UsePrereqAux 1 -CourseFeedbackOnlyCold 0 -CourseSampleOnlyCold 0 -PrereqAuxOnlyCold 0 -RunSampledEval 0 -OutputRoot $ckgOutputRootLit -CheckpointRoot $ckgCheckpointRootLit -SaveCkpt 1 -AutoResume 1 -ForceFresh 0 -SaveOptState 1"
$ckgExit = Start-LoggedCommand -Name "CKG-RL exact rerun" -CommandText $ckgCommand -StdoutName "ckg_rl_exact_stdout.log" -StderrName "ckg_rl_exact_stderr.log"
if ($ckgExit -ne 0) {
    Write-QueueLine "CKG-RL exact rerun failed with exit_code=$ckgExit"
} else {
    foreach ($seed in $Seeds) {
        $split = SplitName -Seed $seed
        $primaryDir = Join-Path $Repo "outputs\content_delta_pop5\course_ablation_e60_3seed\full\$split"
        $tempDir = Join-Path $Repo "$ckgOutputRoot\$split"
        Compare-And-CopyPerItem `
            -Name "CKG-RL seed=$seed" `
            -PrimaryResult (Join-Path $primaryDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv") `
            -TempResult (Join-Path $tempDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv") `
            -TempPerItem (Join-Path $tempDir "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv") `
            -TargetPerItem (Join-Path $primaryDir "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv") | Out-Null
    }
}

Write-QueueLine "Final audit after MOOCCube exact significance queue"
if (-not $DryRun) {
    & (Join-Path $Repo "py.bat") (Join-Path $Repo "paper_aaai27\scripts\audit_significance_inputs.py") | ForEach-Object { Write-QueueLine $_ }
}
Write-QueueLine "QUEUE END MOOCCube exact significance CGRC/CKG-RL"
