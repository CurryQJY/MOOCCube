# Robust serial 3-seed launcher for graph_gated_scorer_clean prereq-weight=2.0
# - independent of agent shell (run via Start-Process)
# - PYTHONUNBUFFERED
# - per-seed retries on non-zero / missing metrics
# - records exit codes + nvlddmkm events around each attempt
$ErrorActionPreference = "Continue"
$Root = "D:\DeskTop\MOOCCube"
$Py = "D:\Anaconda3\envs\zw\python.exe"
$Script = Join-Path $Root "graph_gated_scorer_clean.py"
$MaxAttempts = 3
$Seeds = @(2025, 2026, 2027)
$Weight = "2.0"

Set-Location $Root
$env:PYTHONUNBUFFERED = "1"
# Prefer fewer silent CUDA failures when debugging; slight slowdown is acceptable for stability runs
$env:CUDA_LAUNCH_BLOCKING = "0"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogRoot = Join-Path $Root "background_logs\prereq_w2_3seed_robust_$ts"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$DriverLog = Join-Path $LogRoot "driver.log"

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $DriverLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Get-NvEvents([datetime]$since, [datetime]$until) {
    try {
        $ev = Get-WinEvent -FilterHashtable @{
            LogName      = "System"
            ProviderName = "nvlddmkm"
            StartTime    = $since
            EndTime      = $until
        } -ErrorAction SilentlyContinue
        if (-not $ev) { return @() }
        return @($ev | ForEach-Object {
            $data = ($_.Properties | ForEach-Object { $_.Value }) -join " | "
            "Time={0} Id={1} Data={2}" -f $_.TimeCreated, $_.Id, $data
        })
    } catch {
        return @("nvlddmkm query failed: $_")
    }
}

function Test-Metrics([int]$seed) {
    $p = Join-Path $Root "outputs\graph_gated_scorer_clean\seed${seed}_w2\test_metrics.json"
    return (Test-Path $p)
}

function Run-Seed([int]$seed, [int]$attempt) {
    $outDir = Join-Path $Root "outputs\graph_gated_scorer_clean\seed${seed}_w2"
    $split = "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_$seed"
    $stdout = Join-Path $LogRoot ("seed{0}_try{1}.log" -f $seed, $attempt)
    $stderr = Join-Path $LogRoot ("seed{0}_try{1}.err" -f $seed, $attempt)
    $meta = Join-Path $LogRoot ("seed{0}_try{1}.meta.txt" -f $seed, $attempt)

    # Only wipe previous outputs if this seed has no finished metrics.
    # Never delete a completed run (prevents "success then re-run" loops).
    if ((Test-Path $outDir) -and -not (Test-Metrics $seed)) {
        Remove-Item -Recurse -Force $outDir -ErrorAction SilentlyContinue
    }

    $t0 = Get-Date
    Log "START seed=$seed attempt=$attempt out=$outDir"

    $p = Start-Process -FilePath $Py -ArgumentList @(
        "-u", $Script,
        "--data-dir", "processed_data_hin_clean_pop5",
        "--split-dir", $split,
        "--output-dir", $outDir,
        "--seed", "$seed",
        "--epochs", "60",
        "--batch-size", "2048",
        "--n-layers", "2",
        "--prereq-weight", $Weight
    ) -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

    Log "PID=$($p.Id) seed=$seed attempt=$attempt"
    @(
        "pid=$($p.Id)"
        "start=$($t0.ToString('o'))"
        "cmdline=$($p.Id) $Py -u $Script --prereq-weight $Weight --seed $seed"
        "stdout=$stdout"
        "stderr=$stderr"
    ) | Set-Content -Path $meta -Encoding UTF8

    # poll until exit; heartbeat every 60s
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 60
        $tail = ""
        if (Test-Path $stdout) {
            $tail = (Get-Content $stdout -Tail 1 -ErrorAction SilentlyContinue)
        }
        $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null)
        Log "HEARTBEAT seed=$seed pid=$($p.Id) alive gpu=[$gpu] last=[$tail]"
        try { $p.Refresh() } catch { break }
    }

    # ExitCode can be $null right after HasExited on some hosts; treat metrics as ground truth.
    $code = $null
    try { $code = $p.ExitCode } catch { $code = $null }
    $t1 = Get-Date
    $nv = Get-NvEvents -since $t0.AddSeconds(-5) -until $t1.AddSeconds(5)
    $hasMetrics = Test-Metrics $seed
    $errBytes = 0
    if (Test-Path $stderr) { $errBytes = (Get-Item $stderr).Length }

    Add-Content -Path $meta -Value "end=$($t1.ToString('o'))" -Encoding UTF8
    Add-Content -Path $meta -Value "exit_code=$code" -Encoding UTF8
    Add-Content -Path $meta -Value "has_metrics=$hasMetrics" -Encoding UTF8
    Add-Content -Path $meta -Value "stderr_bytes=$errBytes" -Encoding UTF8
    if ($nv.Count -gt 0) {
        Add-Content -Path $meta -Value "nvlddmkm_events:" -Encoding UTF8
        $nv | ForEach-Object { Add-Content -Path $meta -Value "  $_" -Encoding UTF8 }
    } else {
        Add-Content -Path $meta -Value "nvlddmkm_events: (none in window)" -Encoding UTF8
    }

    Log "END seed=$seed attempt=$attempt exit=$code metrics=$hasMetrics stderr_bytes=$errBytes elapsed_sec=$([int]($t1-$t0).TotalSeconds)"
    if ($nv.Count -gt 0) {
        Log "NVLDDMKM during seed=$seed attempt=$attempt :"
        foreach ($line in $nv) { Log "  $line" }
    }

    return @{ ExitCode = $code; HasMetrics = $hasMetrics; StderrBytes = $errBytes }
}

Log "=== robust prereq_w2 3seed launcher ==="
Log "LogRoot=$LogRoot"
Log "Python=$Py"
Log "Weight=$Weight MaxAttempts=$MaxAttempts"

# quick preflight: CLI accepts flag
$pre = & $Py -u $Script --help 2>&1 | Out-String
if ($pre -notmatch "prereq-weight") {
    Log "FATAL: --prereq-weight missing from --help"
    "FAIL_NO_CLI $(Get-Date -Format o)" | Set-Content (Join-Path $LogRoot "DONE.flag") -Encoding UTF8
    exit 2
}
Log "preflight: --prereq-weight present in --help"

$allOk = $true
foreach ($seed in $Seeds) {
    if (Test-Metrics $seed) {
        Log "seed=$seed already has metrics; skip"
        continue
    }
    $ok = $false
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        # Re-check before every attempt: another process may have just finished this seed.
        if (Test-Metrics $seed) {
            Log "seed=$seed metrics appeared before attempt=$a; skip"
            $ok = $true
            break
        }
        $r = Run-Seed -seed $seed -attempt $a
        # Ground truth = metrics on disk. Do not require ExitCode -eq 0
        # (ExitCode is often $null after Start-Process wait on Windows).
        if ($r.HasMetrics -or (Test-Metrics $seed)) {
            $ok = $true
            Log "SUCCESS seed=$seed on attempt=$a (metrics present; exit=$($r.ExitCode))"
            break
        }
        Log "RETRY seed=$seed after failed attempt=$a exit=$($r.ExitCode) metrics=$($r.HasMetrics)"
        Start-Sleep -Seconds 15
    }
    if (-not $ok) {
        Log "FAILED seed=$seed after $MaxAttempts attempts"
        $allOk = $false
    }
}

if ($allOk) {
    "ALL_DONE $(Get-Date -Format o)" | Set-Content (Join-Path $LogRoot "DONE.flag") -Encoding UTF8
    Log "ALL_DONE"
    exit 0
} else {
    "PARTIAL_OR_FAIL $(Get-Date -Format o)" | Set-Content (Join-Path $LogRoot "DONE.flag") -Encoding UTF8
    Log "PARTIAL_OR_FAIL"
    exit 1
}
