$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Monitor = Join-Path $Repo "outputs\xes3g5m\ours_sota_serial\_queue\monitor_cgrc_auto_downgrade.ps1"

if (-not (Test-Path -LiteralPath $Monitor)) {
    throw "Missing monitor script: $Monitor"
}

$output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Monitor -SelfTest 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Monitor self-test failed: $output"
}

$text = ($output | Out-String)
foreach ($needle in @(
    "SELFTEST OK",
    "tier0=batch1024_chunk1024",
    "tier1=batch1024_chunk512",
    "tier2=batch768_chunk512",
    "tier3=batch512_chunk256",
    "oom_regex=ok"
)) {
    if ($text -notlike "*$needle*") {
        throw "Missing self-test marker '$needle' in: $text"
    }
}

Write-Host "OK"
