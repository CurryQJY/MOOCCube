param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$WaitPids = @(21764, 34616),
    [string]$OursRootBase = "outputs\content_delta_pop5\rq1_per_course_export\ours",
    [string]$OursCheckpointBase = "checkpoints\content_delta_pop5\rq1_per_course_export\ours",
    [string]$CgrcOutputRoot = "outputs\content_delta_pop5\static_item_cold_balanced",
    [string]$CgrcResultSubdir = "rq1_per_course_cgrc_export",
    [string]$StatsOutDir = "outputs\content_delta_pop5\rq1_per_course_significance",
    [int]$Epochs = 60,
    [int]$CgrcEpochs = 50,
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$logDir = Join-Path $repoPath "outputs\content_delta_pop5\rq1_per_course_export\_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "rq1_per_course_stats_$stamp.log"
$statusPath = Join-Path $logDir "rq1_per_course_stats_status.txt"

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $statusPath -Encoding UTF8 -Value $line
}

Start-Transcript -LiteralPath $logPath -Append | Out-Null
try {
    Write-Status "RQ1 per-course significance job started."
    Write-Status "Log: $logPath"

    foreach ($pidToWait in $WaitPids) {
        while (Get-Process -Id $pidToWait -ErrorAction SilentlyContinue) {
            Write-Status "Waiting for existing GPU job PID=$pidToWait."
            Start-Sleep -Seconds $PollSeconds
        }
        Write-Status "PID=$pidToWait is no longer running."
    }

    Write-Status "Running CKG-RL full 3-seed export with per-item metrics."
    .\run_course_ablation_e60_3seed_overnight.ps1 `
        -VariantList "full" `
        -SeedList "2025,2026,2027" `
        -Epochs $Epochs `
        -Patience $Epochs `
        -TargetRootBase $OursRootBase `
        -CheckpointRootBase $OursCheckpointBase `
        -NoCarrySeed2025

    Write-Status "Running CGRC 3-seed export with per-item metrics."
    .\run_cgrc_paper_static.ps1 `
        -Seeds @(2025, 2026, 2027) `
        -ColdThreshold 1 `
        -Epochs $CgrcEpochs `
        -OutputRoot $CgrcOutputRoot `
        -ResultSubdir $CgrcResultSubdir `
        -BestAverageMode "item_macro"

    Write-Status "Computing paired bootstrap/randomization statistics."
    $oursRootArg = (($OursRootBase -replace "\\", "/") + "/full")
    $baselineRootArg = ($CgrcOutputRoot -replace "\\", "/")
    $statsOutArg = ($StatsOutDir -replace "\\", "/")
    .\py.bat analyze_per_course_significance.py `
        --ours-root $oursRootArg `
        --baseline-root $baselineRootArg `
        --baseline-pattern "**/$CgrcResultSubdir/per_item_full_cold_cgrc_paper_static.csv" `
        --out-dir $statsOutArg

    $summaryPath = Join-Path $repoPath (Join-Path $StatsOutDir "per_course_ours_vs_cgrc_summary.csv")
    $detailPath = Join-Path $repoPath (Join-Path $StatsOutDir "per_course_ours_vs_cgrc_detail.csv")
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "Missing summary output: $summaryPath"
    }
    if (-not (Test-Path -LiteralPath $detailPath)) {
        throw "Missing detail output: $detailPath"
    }

    Write-Status "RQ1 per-course significance job completed."
    Write-Status "Summary: $summaryPath"
    Write-Status "Detail: $detailPath"
}
catch {
    Write-Status ("FAILED: " + $_.Exception.Message)
    throw
}
finally {
    Stop-Transcript | Out-Null
}
