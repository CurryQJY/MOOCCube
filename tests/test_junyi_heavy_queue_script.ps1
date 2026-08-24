$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_junyi_main_table_heavy_after_lowcost.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing watcher script: $script"
}

$tmp = Join-Path $repo ".runtime_tmp\test_junyi_heavy_queue"
Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$outputRoot = Join-Path $tmp "outputs"
$checkpointRoot = Join-Path $tmp "checkpoints"
$lowQueueDir = Join-Path $outputRoot "_queue_lowcost"
New-Item -ItemType Directory -Force -Path $lowQueueDir | Out-Null
Set-Content -Path (Join-Path $lowQueueDir "queue.log") -Encoding UTF8 -Value @(
    "[2026-05-31 16:55:45] QUEUE START Junyi main-table low-cost seeds=2026,2027",
    "[2026-05-31 18:00:00] QUEUE DONE Junyi main-table low-cost seeds=2026,2027"
)

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -OutputRoot $outputRoot `
    -CheckpointRoot $checkpointRoot `
    -Seeds 2026,2027 `
    -PollSeconds 1 `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "DRYRUN START ALDI seed=2026",
    "DRYRUN START ALDI seed=2027",
    "DRYRUN START CGRC-paper seed=2026",
    "DRYRUN START CGRC-paper seed=2027",
    "DRYRUN START Ours seed=2026",
    "DRYRUN START Ours seed=2027",
    "QUEUE DONE Junyi main-table heavy seeds=2026,2027"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run line: $expected"
    }
}

Write-Host "test_junyi_heavy_queue_script.ps1 passed"
