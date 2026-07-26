# Wait for any in-flight seed2025_w2 training to finish, then run fixed robust
# launcher which skips seeds that already have test_metrics.json.
$ErrorActionPreference = "Continue"
$Root = "D:\DeskTop\MOOCCube"
$Log = Join-Path $Root "background_logs\prereq_w2_continue_after_fix_$(Get-Date -Format yyyyMMdd_HHmmss).log"
New-Item -ItemType Directory -Path (Split-Path $Log) -Force | Out-Null
function L($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $Log -Value $line -Encoding UTF8
    Write-Output $line
}

function Has-Metrics([int]$seed) {
    Test-Path (Join-Path $Root "outputs\graph_gated_scorer_clean\seed${seed}_w2\test_metrics.json")
}

function Any-GraphTrain() {
    $ps = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "graph_gated_scorer_clean" }
    return [bool]$ps
}

Set-Location $Root
L "continue script start; log=$Log"

# Wait up to ~45 min for current training to finish writing metrics
$deadline = (Get-Date).AddMinutes(45)
while ((Get-Date) -lt $deadline) {
    $m2025 = Has-Metrics 2025
    $busy = Any-GraphTrain
    L "wait: metrics2025=$m2025 training_busy=$busy"
    if ($m2025 -and -not $busy) { break }
    if (-not $busy -and -not $m2025) {
        L "training gone but no metrics yet; wait 30s for flush"
        Start-Sleep -Seconds 30
        if (Has-Metrics 2025) { break }
        L "still no metrics after flush wait; will let robust re-run only missing seeds"
        break
    }
    Start-Sleep -Seconds 30
}

foreach ($s in 2025, 2026, 2027) {
    L "pre-robust metrics seed$s=$(Has-Metrics $s)"
}

L "launch fixed run_w2_3seed_robust.ps1 (skips seeds with metrics)"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "run_w2_3seed_robust.ps1")
$code = $LASTEXITCODE
L "robust finished exit=$code"
exit $code
