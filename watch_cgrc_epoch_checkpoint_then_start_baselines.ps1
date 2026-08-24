param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\xes3g5m\ours_sota_serial",
    [string]$SplitName = "strict_item_cold_balanced_thr1_seed_2025",
    [string]$CheckpointRoot = "checkpoints\xes3g5m\ours_sota_serial",
    [int]$TargetEpoch = 0,
    [int]$PollSeconds = 30,
    [int]$StableSeconds = 10,
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
$QueueDir = Join-Path $OutputRootAbs "_queue"
$WatcherLog = Join-Path $QueueDir "watch_cgrc_epoch_checkpoint_then_start_baselines.log"
$CgrcLog = Join-Path (Join-Path (Join-Path $OutputRootAbs "ours_full") $SplitName) "main_table_compare\run_cgrc_paper.log"
$CkptDir = Join-Path (Resolve-RunPath $Repo $CheckpointRoot) ("cgrc_paper\{0}" -f $SplitName)
$LatestCkpt = Join-Path $CkptDir "latest.pt"
$OvernightScript = Join-Path $Repo "run_xes3g5m_overnight_lightweight_serial.ps1"

New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

function Write-WatcherLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $WatcherLog -Encoding UTF8 -Value $line
    Write-Host $line
}

function Get-LastProgressEpoch {
    if (-not (Test-Path -LiteralPath $CgrcLog)) {
        return 0
    }
    $lines = Get-Content -LiteralPath $CgrcLog -Tail 500
    [array]::Reverse($lines)
    foreach ($line in $lines) {
        if ($line -match '\[CGRC-TRAIN-PROGRESS\]\s+Epoch\s+(\d+)/(\d+)') {
            return [int]$Matches[1]
        }
        if ($line -match '^Epoch\s+\[(\d+)/(\d+)\]') {
            return [int]$Matches[1]
        }
    }
    return 0
}

function Test-EpochSummary([int]$Epoch) {
    if (-not (Test-Path -LiteralPath $CgrcLog)) {
        return $false
    }
    $pattern = "^Epoch \[$Epoch/"
    return [bool](Select-String -LiteralPath $CgrcLog -Pattern $pattern | Select-Object -First 1)
}

function Test-LatestCheckpointStable([datetime]$InitialWriteTime) {
    if (-not (Test-Path -LiteralPath $LatestCkpt)) {
        return $false
    }
    $first = Get-Item -LiteralPath $LatestCkpt
    if ($first.LastWriteTime -le $InitialWriteTime -or $first.Length -le 0) {
        return $false
    }
    Start-Sleep -Seconds $StableSeconds
    if (-not (Test-Path -LiteralPath $LatestCkpt)) {
        return $false
    }
    $second = Get-Item -LiteralPath $LatestCkpt
    return (
        $second.LastWriteTime -eq $first.LastWriteTime -and
        $second.Length -eq $first.Length -and
        $second.Length -gt 0
    )
}

function Stop-CgrcProcesses {
    $patterns = @(
        "cgrc_paper_static_hin.py",
        "resume_cgrc_batch1024_now.ps1",
        "monitor_cgrc_auto_downgrade.ps1"
    )
    $procs = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $cmd = $_.CommandLine
                if (-not $cmd) { return $false }
                foreach ($pat in $patterns) {
                    if ($cmd -like "*$pat*") { return $true }
                }
                return $false
            } |
            Where-Object { $_.ProcessId -ne $PID } |
            Sort-Object CreationDate -Descending
    )
    if ($procs.Count -lt 1) {
        Write-WatcherLog "No CGRC-related process found to stop."
        return
    }
    foreach ($proc in $procs) {
        Write-WatcherLog "STOP pid=$($proc.ProcessId) name=$($proc.Name)"
        if (-not $DryRun) {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-FastQueuePids {
    @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -like "powershell*" -and
                $_.CommandLine -and
                $_.CommandLine -like "*-File*run_xes3g5m_lightweight_baselines.ps1*"
            } |
            Where-Object { $_.ProcessId -ne $PID } |
            Select-Object -ExpandProperty ProcessId
    )
}

function Start-OvernightQueue {
    if (-not (Test-Path -LiteralPath $OvernightScript)) {
        throw "Missing overnight script: $OvernightScript"
    }
    $fastPids = @(Get-FastQueuePids)
    $out = Join-Path $QueueDir "overnight_lightweight_serial_after_cgrc_stop.out.log"
    $err = Join-Path $QueueDir "overnight_lightweight_serial_after_cgrc_stop.err.log"
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $OvernightScript
    )
    if ($fastPids.Count -gt 0) {
        $args += "-WaitPids"
        foreach ($fastPid in $fastPids) {
            $args += [string]$fastPid
        }
        $args += "-SkipFastStage"
        Write-WatcherLog "Start overnight queue after existing fast queue pid=$($fastPids -join ',')"
    } else {
        Write-WatcherLog "Start overnight queue with fast stage included; no existing fast queue found."
    }
    if ($DryRun) {
        Write-WatcherLog "DRY-RUN would start: powershell.exe $($args -join ' ')"
        return
    }
    $p = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $args `
        -WorkingDirectory $Repo `
        -WindowStyle Hidden `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -PassThru
    Write-WatcherLog "Started overnight queue pid=$($p.Id)"
}

if (-not (Test-Path -LiteralPath $CgrcLog)) {
    throw "Missing CGRC log: $CgrcLog"
}
if (-not (Test-Path -LiteralPath $LatestCkpt)) {
    throw "Missing latest checkpoint: $LatestCkpt"
}

$initialCkpt = Get-Item -LiteralPath $LatestCkpt
$initialWriteTime = $initialCkpt.LastWriteTime
if ($TargetEpoch -le 0) {
    $TargetEpoch = Get-LastProgressEpoch
}
if ($TargetEpoch -le 0) {
    throw "Could not infer target epoch from CGRC log. Pass -TargetEpoch explicitly."
}

Write-WatcherLog "WATCH START target_epoch=$TargetEpoch initial_latest=$initialWriteTime latest_path=$LatestCkpt"
while ($true) {
    $summaryOk = Test-EpochSummary $TargetEpoch
    $latest = Get-Item -LiteralPath $LatestCkpt -ErrorAction SilentlyContinue
    $updated = ($null -ne $latest -and $latest.LastWriteTime -gt $initialWriteTime)
    Write-WatcherLog "CHECK target_epoch=$TargetEpoch summary=$summaryOk checkpoint_updated=$updated latest=$($latest.LastWriteTime) size=$($latest.Length)"

    if ($summaryOk -and (Test-LatestCheckpointStable $initialWriteTime)) {
        Write-WatcherLog "Checkpoint for epoch $TargetEpoch appears saved and stable. Stopping CGRC."
        Stop-CgrcProcesses
        Start-Sleep -Seconds 5
        Start-OvernightQueue
        Write-WatcherLog "WATCH DONE"
        return
    }

    Start-Sleep -Seconds $PollSeconds
}
