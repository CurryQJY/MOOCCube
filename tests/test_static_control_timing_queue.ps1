$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$watcher = Join-Path $repo "watch_static_control_then_cgrc_timing.ps1"
if (-not (Test-Path -LiteralPath $watcher)) {
    throw "Watcher is missing (expected RED before implementation): $watcher"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("static_control_timing_queue_" + [guid]::NewGuid().ToString("N"))
$controlRoot = Join-Path $tmpRoot "control"
$timingOutputRoot = Join-Path $tmpRoot "timing"
$queueLog = Join-Path $tmpRoot "watcher.log"
$timingScript = Join-Path $tmpRoot "run_cgrc_controlled_timing.ps1"
$launchMarker = Join-Path $tmpRoot "timing_launched.marker"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    Set-Content -LiteralPath $timingScript -Encoding UTF8 -Value @"
param([string]`$Repo, [string]`$OutputRoot, [switch]`$TimingOnly)
Set-Content -LiteralPath '$launchMarker' -Value "TimingOnly=`$TimingOnly" -Encoding UTF8
"@

    # Incomplete panel must not launch and must report a bounded wait.
    New-Item -ItemType Directory -Force -Path (Join-Path $controlRoot "control_seed2025") | Out-Null
    $out = & $watcher `
        -Repo $repo `
        -ControlOutputRoot $controlRoot `
        -TimingScript $timingScript `
        -TimingOutputRoot $timingOutputRoot `
        -WatcherLog $queueLog `
        -PollSeconds 0 `
        -MaxChecks 1 `
        -IgnoreStaticProcesses `
        -DryRun *>&1
    $code = $LASTEXITCODE
    if ($code -ne 2) {
        throw "Incomplete panel should exit 2 after MaxChecks; got $code. Output: $($out -join "`n")"
    }
    if (Test-Path -LiteralPath $launchMarker) {
        throw "Incomplete panel unexpectedly launched timing"
    }

    # Complete-looking directories and a manifest missing prereq_weight must not launch.
    foreach ($seed in 2025, 2026, 2027) {
        $run = Join-Path $controlRoot "control_seed$seed"
        New-Item -ItemType Directory -Force -Path $run | Out-Null
        New-Item -ItemType Directory -Force -Path (Join-Path $run "best.pt") | Out-Null
        Set-Content -LiteralPath (Join-Path $run "val_history.json") -Value '[{"epoch":60}]' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $run "test_metrics.json") -Value '{"best_epoch":60}' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $run "run_manifest.json") -Value (ConvertTo-Json @{
            seed = [int]$seed
            script_sha256 = "fixture"
        }) -Encoding UTF8
    }
    $out = & $watcher `
        -Repo $repo `
        -ControlOutputRoot $controlRoot `
        -TimingScript $timingScript `
        -TimingOutputRoot $timingOutputRoot `
        -WatcherLog $queueLog `
        -PollSeconds 0 `
        -MaxChecks 1 `
        -IgnoreStaticProcesses `
        -DryRun *>&1
    if ($LASTEXITCODE -ne 2) {
        throw "Invalid panel should not launch; got exit $LASTEXITCODE. Output: $($out -join "`n")"
    }
    foreach ($seed in 2025, 2026, 2027) {
        Remove-Item -LiteralPath (Join-Path $controlRoot "control_seed$seed\best.pt") -Recurse -Force
    }

    # A complete, valid panel should pass dry-run without launching.
    foreach ($seed in 2025, 2026, 2027) {
        $run = Join-Path $controlRoot "control_seed$seed"
        New-Item -ItemType Directory -Force -Path $run | Out-Null
        Set-Content -LiteralPath (Join-Path $run "best.pt") -Value "checkpoint" -Encoding ASCII
        Set-Content -LiteralPath (Join-Path $run "val_history.json") -Value '[{"epoch":60}]' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $run "test_metrics.json") -Value '{"best_epoch":60,"cold_N@10":0.1}' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $run "run_manifest.json") -Value (ConvertTo-Json @{
            seed = [int]$seed
            prereq_weight = 0.0
            script_sha256 = "fixture"
        }) -Encoding UTF8
    }
    $decoy = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-Command", "Start-Sleep -Seconds 10; # run_cgrc_controlled_timing.ps1") `
        -WindowStyle Hidden -PassThru
    try {
        $out = & $watcher `
            -Repo $repo `
            -ControlOutputRoot $controlRoot `
            -TimingScript $timingScript `
            -TimingOutputRoot $timingOutputRoot `
            -WatcherLog $queueLog `
            -PollSeconds 0 `
            -MaxChecks 1 `
            -IgnoreStaticProcesses `
            -DryRun *>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Complete panel dry-run failed: $($out -join "`n")"
        }
        if (($out -join "`n") -notmatch "DRY-RUN would start") {
            throw "Complete panel dry-run did not report a launch: $($out -join "`n")"
        }
        if (Test-Path -LiteralPath $launchMarker) {
            throw "Dry-run unexpectedly launched timing"
        }
    }
    finally {
        Stop-Process -Id $decoy.Id -Force -ErrorAction SilentlyContinue
    }

    # A contender must not delete a launch lock it did not acquire.
    $queueRoot = Join-Path $timingOutputRoot "_auto_start"
    New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
    $foreignLock = Join-Path $queueRoot "timing_launch.lock"
    Set-Content -LiteralPath $foreignLock -Value "owned-by-another-watcher" -Encoding UTF8
    $out = & $watcher `
        -Repo $repo `
        -ControlOutputRoot $controlRoot `
        -TimingScript $timingScript `
        -TimingOutputRoot $timingOutputRoot `
        -WatcherLog $queueLog `
        -PollSeconds 0 `
        -MaxChecks 1 `
        -IgnoreStaticProcesses *>&1
    if ($LASTEXITCODE -ne 2) {
        throw "Foreign launch lock should keep watcher waiting; got $LASTEXITCODE. Output: $($out -join "`n")"
    }
    if (-not (Test-Path -LiteralPath $foreignLock)) {
        throw "Watcher deleted a launch lock it did not own"
    }
    if (Test-Path -LiteralPath $launchMarker) {
        throw "Watcher launched timing despite a foreign lock"
    }
    Remove-Item -LiteralPath $foreignLock -Force

    # A real launch is idempotent: the marker prevents a second start.
    $out = & $watcher `
        -Repo $repo `
        -ControlOutputRoot $controlRoot `
        -TimingScript $timingScript `
        -TimingOutputRoot $timingOutputRoot `
        -WatcherLog $queueLog `
        -PollSeconds 0 `
        -MaxChecks 1 `
        -IgnoreStaticProcesses *>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Complete panel launch failed: $($out -join "`n")"
    }
    $deadline = (Get-Date).AddSeconds(10)
    while (-not (Test-Path -LiteralPath $launchMarker) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $launchMarker)) {
        throw "Timing process did not create launch marker"
    }
    $markerContent = Get-Content -Raw -LiteralPath $launchMarker
    if ($markerContent -notmatch "TimingOnly=True") {
        throw "Timing process was not launched with -TimingOnly: $markerContent"
    }

    $second = & $watcher `
        -Repo $repo `
        -ControlOutputRoot $controlRoot `
        -TimingScript $timingScript `
        -TimingOutputRoot $timingOutputRoot `
        -WatcherLog $queueLog `
        -PollSeconds 0 `
        -MaxChecks 1 `
        -IgnoreStaticProcesses *>&1
    if ($LASTEXITCODE -ne 0 -or ($second -join "`n") -notmatch "already launched") {
        throw "Second invocation was not idempotently skipped: $($second -join "`n")"
    }
}
finally {
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "test_static_control_timing_queue.ps1 passed"
