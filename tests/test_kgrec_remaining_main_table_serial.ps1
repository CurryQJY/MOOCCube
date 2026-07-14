$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_kgrec_remaining_main_table_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing KGRec remaining main-table queue script: $script"
}

$scriptText = Get-Content -Raw -LiteralPath $script
if ($scriptText -notlike '*function Invoke-NativeLogged*' -or $scriptText -notlike '*$ErrorActionPreference = "Continue"*') {
    throw "Queue must isolate native stderr from ErrorActionPreference=Stop"
}

$output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "KGRec queue dry-run exited with code $LASTEXITCODE"
}

$text = $output -join "`n"
$expected = @(
    "QUEUE PLAN KGRec remaining main-table serial",
    "TEST KGRec unit suite",
    "pytest_basetemp=paper_aaai27\baseline_sources\_kgrec_strict\_remaining_main_table_queue\_pytest_tmp",
    "VALIDATE dataset=MOOCCube seed=2025",
    "VALIDATE dataset=MOOCCube seed=2026",
    "VALIDATE dataset=MOOCCube seed=2027",
    "VALIDATE dataset=COCO seed=2025",
    "EXPORT dataset=COCO seed=2026",
    "RUN dataset=COCO seed=2026 lr=1e-5 epochs=20 patience=5 batch=4096 epoch0_diagnostic_only=true",
    "EXPORT dataset=COCO seed=2027",
    "RUN dataset=COCO seed=2027 lr=1e-5 epochs=20 patience=5 batch=4096 epoch0_diagnostic_only=true",
    "VALIDATE dataset=Junyi seed=2025",
    "VALIDATE dataset=Junyi seed=2026",
    "EXPORT dataset=Junyi seed=2027",
    "RUN dataset=Junyi seed=2027 lr=1e-6 epochs=10 patience=8 batch=4096 epoch0_diagnostic_only=true",
    "SUMMARY 3 datasets x 3 seeds",
    "main_table_summary.json",
    "queue_status.json"
)

foreach ($token in $expected) {
    if ($text -notlike "*$token*") {
        throw "Missing expected dry-run token: $token"
    }
}

$ordered = @(
    "EXPORT dataset=COCO seed=2026",
    "RUN dataset=COCO seed=2026",
    "EXPORT dataset=COCO seed=2027",
    "RUN dataset=COCO seed=2027",
    "EXPORT dataset=Junyi seed=2027",
    "RUN dataset=Junyi seed=2027",
    "SUMMARY 3 datasets x 3 seeds"
)
$lastIndex = -1
foreach ($token in $ordered) {
    $index = $text.IndexOf($token, [System.StringComparison]::Ordinal)
    if ($index -le $lastIndex) {
        throw "Dry-run job order is wrong at token: $token"
    }
    $lastIndex = $index
}

$preflight = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -ValidateExistingOnly
if ($LASTEXITCODE -ne 0) {
    throw "KGRec existing-result preflight exited with code $LASTEXITCODE"
}
$preflightText = $preflight -join "`n"
$match = [regex]::Match($preflightText, 'PREFLIGHT PASS existing_results=(\d+) pending_runs=(\d+)')
if (-not $match.Success) {
    throw "Existing-result preflight did not report completed and pending job counts"
}
$existingCount = [int]$match.Groups[1].Value
$pendingCount = [int]$match.Groups[2].Value
if (($existingCount + $pendingCount) -ne 9 -or $existingCount -lt 6) {
    throw "Existing-result preflight counts are invalid: existing=$existingCount pending=$pendingCount"
}

Write-Host "test_kgrec_remaining_main_table_serial.ps1 passed"
