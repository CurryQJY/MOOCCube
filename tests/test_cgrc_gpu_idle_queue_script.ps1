$ErrorActionPreference = "Stop"

$repo = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$scriptPath = Join-Path $repo "run_cgrc_per_item_when_gpu_free.ps1"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Missing script: $scriptPath"
}

$scriptText = Get-Content -LiteralPath $scriptPath -Raw
foreach ($needle in @(
    "MinFreeMemoryMiB",
    "MaxGpuUtilPercent",
    "ConsecutiveOk",
    "MaxChecks",
    "DryRun",
    "per_item_full_cold_cgrc_paper_static.csv",
    "run_cgrc_paper_static.ps1"
)) {
    if ($scriptText -notmatch [regex]::Escape($needle)) {
        throw "Script does not contain expected token: $needle"
    }
}

$tmpRoot = Join-Path $env:TEMP ("cgrc_gpu_idle_queue_test_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
$statusPath = Join-Path $tmpRoot "status.txt"
$logDir = Join-Path $tmpRoot "logs"

try {
    & $scriptPath `
        -Repo $repo `
        -OutputRoot "outputs\content_delta_pop5\static_item_cold_balanced" `
        -ResultSubdir "dry_run_should_not_exist" `
        -MinFreeMemoryMiB 999999 `
        -MaxGpuUtilPercent 0 `
        -ConsecutiveOk 1 `
        -PollSeconds 1 `
        -MaxChecks 1 `
        -LogDir $logDir `
        -StatusPath $statusPath `
        -DryRun
    if ($LASTEXITCODE -ne 0) {
        throw "Dry-run command exited with code $LASTEXITCODE"
    }

    $status = Get-Content -LiteralPath $statusPath -Raw
    if ($status -notmatch "MaxChecks reached without launch") {
        throw "Dry-run status did not record non-launch termination. Status was: $status"
    }
    if ($status -match "Running CGRC") {
        throw "Dry-run unexpectedly attempted to launch CGRC. Status was: $status"
    }
}
finally {
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "CGRC GPU idle queue script test passed."
