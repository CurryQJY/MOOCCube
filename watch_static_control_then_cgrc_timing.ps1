param(
    [string]$Repo = $PSScriptRoot,
    [string]$ControlOutputRoot = "outputs\static_prereq_v2",
    [string]$TimingScript = "run_cgrc_controlled_timing.ps1",
    [string]$TimingOutputRoot = "outputs\cgrc_formal_timing_v1",
    [string]$WatcherLog = "",
    [int]$PollSeconds = 60,
    [int]$MaxChecks = 0,
    [switch]$IgnoreStaticProcesses,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if ($PollSeconds -lt 0) {
    throw "PollSeconds must be non-negative"
}
if ($MaxChecks -lt 0) {
    throw "MaxChecks must be non-negative"
}
if ($PollSeconds -eq 0 -and $MaxChecks -eq 0) {
    throw "PollSeconds cannot be zero for an unbounded watcher"
}

function Resolve-RepoPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Path))
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
$ControlOutputRoot = Resolve-RepoPath $Repo $ControlOutputRoot
$TimingScript = Resolve-RepoPath $Repo $TimingScript
$TimingOutputRoot = Resolve-RepoPath $Repo $TimingOutputRoot
$QueueRoot = Join-Path $TimingOutputRoot "_auto_start"
if ($IgnoreStaticProcesses) {
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    $controlIsTemp = $ControlOutputRoot.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)
    $timingIsTemp = $TimingOutputRoot.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)
    if ($MaxChecks -le 0 -or -not $controlIsTemp -or -not $timingIsTemp) {
        throw "IgnoreStaticProcesses is restricted to bounded temporary test roots"
    }
}
if ([string]::IsNullOrWhiteSpace($WatcherLog)) {
    $WatcherLog = Join-Path $QueueRoot "watch_static_control_then_cgrc_timing.log"
} else {
    $WatcherLog = Resolve-RepoPath $Repo $WatcherLog
}
$LaunchMarker = Join-Path $QueueRoot "timing_started.flag"
$LaunchLock = Join-Path $QueueRoot "timing_launch.lock"

New-Item -ItemType Directory -Force -Path $QueueRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WatcherLog) | Out-Null

function Write-WatcherLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $WatcherLog -Encoding UTF8 -Value $line
    Write-Host $line
}

function Get-ControlRunDirectory([int]$Seed) {
    return (Join-Path $ControlOutputRoot "control_seed$Seed")
}

function Test-ControlRunComplete([int]$Seed) {
    $run = Get-ControlRunDirectory $Seed
    foreach ($name in @("best.pt", "val_history.json", "test_metrics.json", "run_manifest.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $run $name) -PathType Leaf)) {
            return $false
        }
    }
    try {
        $history = @(Get-Content -Raw -LiteralPath (Join-Path $run "val_history.json") | ConvertFrom-Json)
        $metrics = Get-Content -Raw -LiteralPath (Join-Path $run "test_metrics.json") | ConvertFrom-Json
        $manifest = Get-Content -Raw -LiteralPath (Join-Path $run "run_manifest.json") | ConvertFrom-Json
        $manifestProperties = @($manifest.PSObject.Properties.Name)
        if ($history.Count -lt 1 -or $null -eq $metrics.best_epoch -or
            $manifestProperties -notcontains "seed" -or
            $manifestProperties -notcontains "prereq_weight") {
            return $false
        }
        if ([string]$manifest.seed -notmatch '^\d+$' -or
            [int]::Parse([string]$manifest.seed) -ne $Seed) {
            return $false
        }
        if ([string]$manifest.prereq_weight -notmatch '^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$' -or
            [double]::Parse([string]$manifest.prereq_weight,
                [Globalization.CultureInfo]::InvariantCulture) -ne 0.0) {
            return $false
        }
    } catch {
        return $false
    }
    return $true
}

function Test-ControlPanelComplete {
    foreach ($seed in @(2025, 2026, 2027)) {
        if (-not (Test-ControlRunComplete $seed)) {
            return $false
        }
    }
    return $true
}

function Test-StaticCommand([string]$CommandLine) {
    if ($CommandLine -match '(?i)(?:^|\s)-File\s+(?:"[^"]*run_static_prereq_v2\.ps1"|[^\s]+run_static_prereq_v2\.ps1)(?:\s|$)') {
        return $true
    }
    return $CommandLine -match '(?i)(?:^|\s)(?:"[^"]*static_prereq_v2\.py"|[^\s]*static_prereq_v2\.py)(?:\s|$)'
}

function Get-BlockingStaticProcesses {
    $knownLaunchers = @("python.exe", "pythonw.exe", "powershell.exe", "pwsh.exe", "cmd.exe")
    try {
        return @(
            Get-CimInstance Win32_Process |
                Where-Object {
                    $process = $_
                    if ($process.ProcessId -eq $PID) {
                        return $false
                    }
                    if (-not $process.CommandLine) {
                        return $knownLaunchers -contains ([string]$process.Name).ToLowerInvariant()
                    }
                    return Test-StaticCommand ([string]$process.CommandLine)
                } |
                Select-Object ProcessId, Name, CommandLine
        )
    } catch {
        throw "Could not inspect running processes safely: $($_.Exception.Message)"
    }
}

function Get-ExistingTimingProcesses {
    $targetOutput = $TimingOutputRoot.TrimEnd('\')
    function Test-TimingCommand([string]$CommandLine) {
        if ($CommandLine -match '(?i)(?:^|\s)-File\s+(?:"[^"]*run_cgrc_controlled_timing\.ps1"|[^\s]+run_cgrc_controlled_timing\.ps1)(?:\s|$)' -and
            $CommandLine -like "*$targetOutput*") {
            return $true
        }
        return $false
    }
    try {
        return @(
            Get-CimInstance Win32_Process |
                Where-Object {
                    $process = $_
                    if ($process.ProcessId -eq $PID -or -not $process.CommandLine) {
                        return $false
                    }
                    return Test-TimingCommand ([string]$process.CommandLine)
                } |
                Select-Object ProcessId, Name, CommandLine
        )
    } catch {
        throw "Could not inspect timing processes safely: $($_.Exception.Message)"
    }
}

function Start-TimingExperiment {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $TimingScript,
        "-Repo", $Repo,
        "-OutputRoot", $TimingOutputRoot,
        "-TimingOnly"
    )
    $stdout = Join-Path $QueueRoot "timing.stdout.log"
    $stderr = Join-Path $QueueRoot "timing.stderr.log"
    $display = "powershell.exe " + ($arguments -join " ")
    if ($DryRun) {
        Write-WatcherLog "DRY-RUN would start: $display"
        return "dry-run"
    }

    $lockStream = $null
    $ownsLock = $false
    try {
        try {
            $lockStream = [System.IO.File]::Open(
                $LaunchLock,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )
            $ownsLock = $true
        } catch [System.IO.IOException] {
            Write-WatcherLog "another watcher owns launch lock; no duplicate launch"
            return "locked"
        }

        # Re-check all guards after claiming the lock to close the launch race.
        if (Test-Path -LiteralPath $LaunchMarker) {
            Write-WatcherLog "launch marker appeared before start; no duplicate launch"
            return "already"
        }
        $lateTiming = @(Get-ExistingTimingProcesses)
        if ($lateTiming.Count -gt 0) {
            $summary = ($lateTiming | ForEach-Object { "pid=$($_.ProcessId) name=$($_.Name)" }) -join ","
            Write-WatcherLog "timing process appeared before start; no duplicate launch: $summary"
            return "already"
        }
        $lateStatic = if ($IgnoreStaticProcesses) { @() } else { @(Get-BlockingStaticProcesses) }
        if ($lateStatic.Count -gt 0) {
            Write-WatcherLog "static process appeared before start; will keep waiting"
            return "blocked"
        }

        $process = Start-Process -FilePath "powershell.exe" `
            -ArgumentList $arguments `
            -WorkingDirectory $Repo `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
        Set-Content -LiteralPath $LaunchMarker -Encoding UTF8 -Value (
            "pid=$($process.Id) started=$(Get-Date -Format o) command=$display"
        )
        Write-WatcherLog "STARTED timing pid=$($process.Id) stdout=$stdout stderr=$stderr"
        return "started"
    } finally {
        if ($ownsLock -and $null -ne $lockStream) {
            $lockStream.Dispose()
        }
        if ($ownsLock) {
            Remove-Item -LiteralPath $LaunchLock -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $TimingScript)) {
    throw "Timing script not found: $TimingScript"
}
if (Test-Path -LiteralPath $LaunchMarker) {
    Write-WatcherLog "timing already launched; no action: $LaunchMarker"
    exit 0
}
$existingTiming = @(Get-ExistingTimingProcesses)
if ($existingTiming.Count -gt 0) {
    $summary = ($existingTiming | ForEach-Object { "pid=$($_.ProcessId) name=$($_.Name)" }) -join ","
    Write-WatcherLog "timing process already exists; no duplicate launch: $summary"
    exit 0
}

Write-WatcherLog "WATCH START control_root=$ControlOutputRoot poll_seconds=$PollSeconds max_checks=$MaxChecks"
$checks = 0
while ($true) {
    $checks++
    $complete = Test-ControlPanelComplete
    $blocking = if ($IgnoreStaticProcesses) { @() } else { @(Get-BlockingStaticProcesses) }
    Write-WatcherLog "CHECK number=$checks complete=$complete blocking_static=$($blocking.Count)"

    if ($complete -and $blocking.Count -eq 0) {
        $launchStatus = Start-TimingExperiment
        if ($launchStatus -in @("started", "dry-run", "already")) {
            Write-WatcherLog "WATCH DONE status=$launchStatus"
            exit 0
        }
    }

    if ($MaxChecks -gt 0 -and $checks -ge $MaxChecks) {
        Write-WatcherLog "MAX_CHECKS reached without launch"
        exit 2
    }
    if ($PollSeconds -gt 0) {
        Start-Sleep -Seconds $PollSeconds
    }
}
