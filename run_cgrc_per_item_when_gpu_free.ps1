param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$Epochs = 50,
    [int]$BatchSize = 4096,
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$OutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$ResultSubdir = "rq1_per_course_cgrc_export",
    [string]$BestAverageMode = "item_macro",
    [int]$EvalNeg = 200,
    [double]$MaskRho = 0.3,
    [int]$ReconTopK = 20,
    [double]$LambdaE = 1.0,
    [double]$Tau = 0.5,
    [int]$GpuIndex = 0,
    [int]$MinFreeMemoryMiB = 6000,
    [int]$MaxGpuUtilPercent = 20,
    [int]$ConsecutiveOk = 2,
    [int]$PollSeconds = 300,
    [int]$MaxChecks = 0,
    [string]$LogDir = "",
    [string]$StatusPath = "",
    [switch]$DryRun,
    [switch]$RunSampledEval
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if ($ConsecutiveOk -lt 1) {
    throw "ConsecutiveOk must be >= 1"
}
if ($PollSeconds -lt 1) {
    throw "PollSeconds must be >= 1"
}
if ($MaxChecks -lt 0) {
    throw "MaxChecks must be >= 0"
}

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $repoPath (Join-Path $OutputRoot (Join-Path $ResultSubdir "_logs"))
}
elseif (-not [System.IO.Path]::IsPathRooted($LogDir)) {
    $LogDir = Join-Path $repoPath $LogDir
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if ([string]::IsNullOrWhiteSpace($StatusPath)) {
    $StatusPath = Join-Path $LogDir "cgrc_per_item_gpu_watch_status.txt"
}
elseif (-not [System.IO.Path]::IsPathRooted($StatusPath)) {
    $StatusPath = Join-Path $repoPath $StatusPath
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $LogDir "cgrc_per_item_gpu_watch_$stamp.log"

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $StatusPath -Encoding UTF8 -Value $line
}

function Get-ExpectedPerItemPaths {
    $paths = @()
    foreach ($seed in $Seeds) {
        $splitName = "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $seed
        $paths += Join-Path (Join-Path (Join-Path $repoPath $OutputRoot) $splitName) `
            (Join-Path $ResultSubdir "per_item_full_cold_cgrc_paper_static.csv")
    }
    return $paths
}

function Test-PerItemComplete {
    $missing = @()
    foreach ($path in Get-ExpectedPerItemPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            $missing += $path
            continue
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -le 0) {
            $missing += $path
        }
    }
    return @{
        Complete = ($missing.Count -eq 0)
        Missing = $missing
    }
}

function Get-GpuSnapshot {
    $queryArgs = @(
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    )
    $raw = & nvidia-smi @queryArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi query failed: $raw"
    }

    foreach ($line in $raw) {
        $text = [string]$line
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }
        $parts = @($text -split "," | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 3) {
            continue
        }
        if ([int]$parts[0] -eq $GpuIndex) {
            return @{
                Index = [int]$parts[0]
                FreeMiB = [int]$parts[1]
                UtilPercent = [int]$parts[2]
                Raw = $text
            }
        }
    }
    throw "GPU index $GpuIndex was not found in nvidia-smi output: $raw"
}

function Get-CgrcCommandPreview {
    return ".\run_cgrc_paper_static.ps1 -Seeds @($($Seeds -join ', ')) -ColdThreshold $ColdThreshold -Epochs $Epochs -BatchSize $BatchSize -DataDir '$DataDir' -OutputRoot '$OutputRoot' -ResultSubdir '$ResultSubdir' -BestAverageMode '$BestAverageMode' -EvalNeg $EvalNeg"
}

Start-Transcript -LiteralPath $logPath -Append | Out-Null
try {
    Write-Status "CGRC per-item GPU watcher started."
    Write-Status "Log: $logPath"
    Write-Status "DryRun=$($DryRun.IsPresent) | GPU=$GpuIndex | min_free=${MinFreeMemoryMiB}MiB | max_util=${MaxGpuUtilPercent}% | consecutive_ok=$ConsecutiveOk | poll=${PollSeconds}s | max_checks=$MaxChecks"
    Write-Status "Command preview: $(Get-CgrcCommandPreview)"

    $complete = Test-PerItemComplete
    if ($complete.Complete) {
        Write-Status "All expected CGRC per-item files already exist; nothing to run."
        return
    }
    Write-Status "Missing CGRC per-item files: $($complete.Missing.Count)"

    $okStreak = 0
    $checks = 0
    while ($true) {
        $checks += 1
        $snapshot = Get-GpuSnapshot
        $ready = ($snapshot.FreeMiB -ge $MinFreeMemoryMiB -and $snapshot.UtilPercent -le $MaxGpuUtilPercent)
        if ($ready) {
            $okStreak += 1
        }
        else {
            $okStreak = 0
        }

        Write-Status (
            "Check {0}: GPU {1} free={2}MiB util={3}% ready={4} streak={5}/{6}" -f `
                $checks, $snapshot.Index, $snapshot.FreeMiB, $snapshot.UtilPercent, $ready, $okStreak, $ConsecutiveOk
        )

        if ($okStreak -ge $ConsecutiveOk) {
            if ($DryRun.IsPresent) {
                Write-Status "DRY-RUN: launch conditions satisfied; would run CGRC now."
                return
            }

            Write-Status "Launch conditions satisfied. Running CGRC per-item export."
            $cgrcArgs = @{
                Seeds = $Seeds
                ColdThreshold = $ColdThreshold
                Epochs = $Epochs
                BatchSize = $BatchSize
                DataDir = $DataDir
                OutputRoot = $OutputRoot
                ResultSubdir = $ResultSubdir
                BestAverageMode = $BestAverageMode
                EvalNeg = $EvalNeg
                MaskRho = $MaskRho
                ReconTopK = $ReconTopK
                LambdaE = $LambdaE
                Tau = $Tau
            }
            if ($RunSampledEval.IsPresent) {
                $cgrcArgs["RunSampledEval"] = $true
            }
            .\run_cgrc_paper_static.ps1 @cgrcArgs

            $after = Test-PerItemComplete
            if (-not $after.Complete) {
                throw "CGRC finished but per-item exports are still missing: $($after.Missing -join '; ')"
            }
            Write-Status "CGRC per-item export completed."
            foreach ($path in Get-ExpectedPerItemPaths) {
                Write-Status "Output: $path"
            }
            return
        }

        if ($MaxChecks -gt 0 -and $checks -ge $MaxChecks) {
            Write-Status "MaxChecks reached without launch."
            return
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
catch {
    Write-Status ("FAILED: " + $_.Exception.Message)
    throw
}
finally {
    Stop-Transcript | Out-Null
}
