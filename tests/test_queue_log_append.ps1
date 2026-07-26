$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scripts = @(
    "run_junyi_main_table_lowcost_2026_2027.ps1",
    "run_junyi_main_table_heavy_after_lowcost.ps1"
)

foreach ($name in $scripts) {
    $path = Join-Path $repo $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing script: $path"
    }
    $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $path
    if ($text -notmatch "function Write-QueueLogLine") {
        throw "$name should use the shared retrying queue-log writer"
    }
    if ($text -match "Add-Content -Path [`$]QueueLog") {
        throw "$name still writes queue.log with Add-Content"
    }
}

Write-Host "test_queue_log_append.ps1 passed"
