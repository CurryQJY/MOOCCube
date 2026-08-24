param(
    [string]$OldHist = "outputs\mooccubex\relations_aug_cmin001_e3\strict_item_cold_balanced_thr1_seed_2025\mooc_metrics_usim_feedback_fast3_content_delta_static.csv",
    [string]$NewHist = "outputs\mooccubex\course_match_topk_v2_e15\strict_item_cold_balanced_thr1_seed_2025\mooc_metrics_usim_feedback_fast3_content_delta_static.csv",
    [string]$RunLog = "outputs\mooccubex\course_match_topk_v2_e15\strict_item_cold_balanced_thr1_seed_2025\run.log",
    [string]$LogPath = "outputs\mooccubex\course_match_topk_v2_e15\strict_item_cold_balanced_thr1_seed_2025\topk_v2_e15_monitor.log",
    [int]$TargetPythonPid = 25552,
    [int]$TargetCmdPid = 23748,
    [int]$WarmupEpoch = 8,
    [int]$Window = 5,
    [double]$Margin = 0.0005,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

function Write-MonitorLog {
    param([string]$Message)
    $dir = Split-Path -Parent $LogPath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Get-ColValue {
    param($Row, [string]$Name)
    $value = $Row.$Name
    if ($null -eq $value -or $value -eq "") {
        throw "Missing column $Name in $NewHist"
    }
    return [double]$value
}

function Get-NewValidationRows {
    if (Test-Path $NewHist) {
        $rows = @(Import-Csv $NewHist | ForEach-Object {
            [pscustomobject]@{
                Epoch = [int]$_.Epoch
                Score = [double]$_.'Val_full_cold_N@10'
                Source = "csv"
            }
        })
        if ($rows.Count -gt 0) {
            return $rows
        }
    }

    if (-not (Test-Path $RunLog)) {
        return @()
    }

    $pattern = 'Epoch\s+(\d+):\s+Cold N@10=([0-9.]+)'
    $rows = @()
    foreach ($line in Get-Content $RunLog) {
        if ($line -match $pattern) {
            $rows += [pscustomobject]@{
                Epoch = [int]$matches[1]
                Score = [double]$matches[2]
                Source = "run.log"
            }
        }
    }
    return $rows
}

if (-not (Test-Path $OldHist)) {
    throw "Missing old history: $OldHist"
}

$oldRows = Import-Csv $OldHist | Where-Object { [int]$_.Epoch -le 15 }
if (($oldRows | Measure-Object).Count -lt 1) {
    throw "No old E15 rows found in $OldHist"
}
$oldBest = ($oldRows | Measure-Object -Property 'Val_full_cold_N@10' -Maximum).Maximum
$threshold = [double]$oldBest - $Margin

Write-MonitorLog ("Monitor started. old_best_e15={0} threshold={1} warmup={2} window={3} py_pid={4} cmd_pid={5}" -f $oldBest, $threshold, $WarmupEpoch, $Window, $TargetPythonPid, $TargetCmdPid)

while ($true) {
    $py = Get-Process -Id $TargetPythonPid -ErrorAction SilentlyContinue
    $cmd = Get-Process -Id $TargetCmdPid -ErrorAction SilentlyContinue
    if ($null -eq $py -and $null -eq $cmd) {
        Write-MonitorLog "Target process already exited. Monitor stops."
        break
    }

    try {
        $allRows = @(Get-NewValidationRows)
        if ($allRows.Count -lt 1) {
            Write-MonitorLog "Waiting for validation rows in CSV or run.log..."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $newBest = ($allRows | Measure-Object -Property Score -Maximum).Maximum
        $lastRow = $allRows[-1]
        $lastEpoch = [int]$lastRow.Epoch
        $lastScore = [double]$lastRow.Score
        Write-MonitorLog ("epoch={0} last_val={1} best={2} threshold={3} source={4}" -f $lastEpoch, $lastScore, $newBest, $threshold, $lastRow.Source)

        $eligible = @($allRows | Where-Object { [int]$_.Epoch -ge $WarmupEpoch })
        if ($eligible.Count -ge $Window) {
            $lastWindow = @($eligible | Select-Object -Last $Window)
            $bad = @($lastWindow | Where-Object { [double]$_.Score -lt $threshold })
            if ($bad.Count -eq $Window -and [double]$newBest -lt $threshold) {
                Write-MonitorLog "Stop condition met. Stopping target process."
                if ($null -ne $py) {
                    Stop-Process -Id $TargetPythonPid -Force
                    Write-MonitorLog "Stopped python pid=$TargetPythonPid"
                }
                if ($null -ne $cmd) {
                    Stop-Process -Id $TargetCmdPid -Force
                    Write-MonitorLog "Stopped cmd pid=$TargetCmdPid"
                }
                break
            }
        }
    } catch {
        Write-MonitorLog ("Monitor read error: {0}" -f $_.Exception.Message)
    }

    Start-Sleep -Seconds $PollSeconds
}
