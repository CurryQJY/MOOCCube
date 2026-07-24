param(
    [ValidateSet("control", "prereq", "all")]
    [string]$Mode = "all",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 60,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\Anaconda3\envs\zw\python.exe"
$entry = Join-Path $repo "static_prereq_v2.py"
$outputRoot = Join-Path $repo "outputs\static_prereq_v2"
$logRoot = Join-Path $outputRoot "logs"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python interpreter not found: $python"
}
if (-not (Test-Path -LiteralPath $entry)) {
    throw "Scorer entry point not found: $entry"
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$env:PYTHONUNBUFFERED = "1"

$variants = @()
if ($Mode -eq "control" -or $Mode -eq "all") {
    $variants += [pscustomobject]@{ Name = "control"; Weight = 0.0 }
}
if ($Mode -eq "prereq" -or $Mode -eq "all") {
    $variants += [pscustomobject]@{ Name = "prereq"; Weight = 1.0 }
}

foreach ($variant in $variants) {
    foreach ($seed in $Seeds) {
        $split = Join-Path $repo (
            "outputs\content_delta_pop5\static_item_cold_balanced\" +
            "strict_item_cold_balanced_thr1_seed_$seed"
        )
        $tag = if ($variant.Name -eq "control") {
            "control_seed$seed"
        } else {
            "seed$seed"
        }
        $out = Join-Path $outputRoot $tag
        $metrics = Join-Path $out "test_metrics.json"
        $manifest = Join-Path $out "run_manifest.json"
        $log = Join-Path $logRoot "$tag.log"

        if ((Test-Path -LiteralPath $metrics) -and
            (Test-Path -LiteralPath $manifest)) {
            Write-Output "SKIP tag=$tag metrics+manifest already present"
            continue
        }
        if (Test-Path -LiteralPath $out) {
            throw "Partial output exists; refusing to overwrite: $out"
        }
        if (Test-Path -LiteralPath $log) {
            throw "Log exists without a complete output; refusing to overwrite: $log"
        }
        if (-not (Test-Path -LiteralPath $split)) {
            throw "Split directory not found: $split"
        }

        $args = @(
            "-u", $entry,
            "--split-dir", $split,
            "--output-dir", $out,
            "--seed", "$seed",
            "--epochs", "$Epochs",
            "--prereq-weight", "$($variant.Weight)",
            "--aux-weight", "0.3"
        )
        Write-Output (
            "START tag=$tag seed=$seed prereq_weight=$($variant.Weight) " +
            "epochs=$Epochs output=$out"
        )
        if ($DryRun) {
            Write-Output "DRY command=$python $($args -join ' ')"
            continue
        }

        & $python @args 2>&1 | Tee-Object -FilePath $log
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Training failed tag=$tag exit=$exitCode log=$log"
        }
        if (-not (Test-Path -LiteralPath $metrics)) {
            throw "Training exited 0 but metrics are missing: $metrics"
        }
        if (-not (Test-Path -LiteralPath $manifest)) {
            throw "Training exited 0 but manifest is missing: $manifest"
        }
        Write-Output "SUCCESS tag=$tag metrics=$metrics"
    }
}

if (-not $DryRun) {
    $done = Join-Path $outputRoot "DONE.flag"
    Set-Content -LiteralPath $done -Value (
        "completed mode=$Mode seeds=$($Seeds -join ',') epochs=$Epochs " +
        "time=$(Get-Date -Format o)"
    ) -Encoding utf8
    Write-Output "ALL_DONE flag=$done"
}
