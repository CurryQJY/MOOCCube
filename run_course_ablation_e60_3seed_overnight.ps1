param(
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$VariantList = "",
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$Seed2025Root = "outputs\content_delta_pop5\course_ablation_e60_seed2025",
    [string]$TargetRootBase = "outputs\content_delta_pop5\course_ablation_e60_3seed_corrected",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\course_ablation_e60_3seed_corrected",
    [switch]$CarryLegacySeed2025,
    [switch]$NoCarrySeed2025
)

$ErrorActionPreference = "Stop"

$Protocol = "strict_item_cold_balanced"
$TagPrefix = "strict_item_cold_balanced_thr1_seed"

if ($SeedList.Trim().Length -gt 0) {
    $SeedsToRun = @(
        $SeedList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { [int]$_.Trim() }
    )
}

$Variants = @(
    @{
        Name = "full"
        ExtraParams = @{
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }
    },
    @{
        Name = "wo_course_reward"
        ExtraParams = @{
            UseCourseReward = $false
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }
    },
    @{
        Name = "wo_course_candidate"
        ExtraParams = @{
            UseCourseSample = $false
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }
    },
    @{
        Name = "wo_prereq_aux"
        ExtraParams = @{
            UsePrereqAux = $false
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }
    },
    @{
        Name = "wo_all_course_signals"
        ExtraParams = @{
            UseCourseFeedback = $false
            UseCourseReward = $false
            UseCourseSample = $false
            UsePrereqAux = $false
            CourseFeedbackOnlyCold = $true
            CourseSampleOnlyCold = $true
            PrereqAuxOnlyCold = $true
        }
    }
)

if ($VariantList.Trim().Length -gt 0) {
    $wantedVariants = @(
        $VariantList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Trim() }
    )
    $Variants = @($Variants | Where-Object { $wantedVariants -contains $_.Name })
    if ($Variants.Count -lt 1) {
        throw "No variants matched VariantList='$VariantList'"
    }
}

function Copy-Seed2025Result {
    param(
        [string]$VariantName,
        [string]$TargetRoot
    )

    if ($NoCarrySeed2025 -or -not $CarryLegacySeed2025) {
        return
    }
    if ($VariantName -eq "wo_course_reward") {
        Write-Host "[carry] Skip legacy seed2025 for corrected $VariantName; course reward semantics changed." -ForegroundColor Yellow
        return
    }

    $tag = "${TagPrefix}_2025"
    $src = Join-Path (Join-Path $Seed2025Root $VariantName) $tag
    $dst = Join-Path $TargetRoot $tag
    $required = Join-Path $src "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if (-not (Test-Path $required)) {
        Write-Host "[carry] Skip seed2025 for $VariantName; source final_fullrank missing: $required" -ForegroundColor Yellow
        return
    }

    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $files = @(
        "final_fullrank_usim_feedback_fast3_content_delta_static.csv",
        "final_report_usim_feedback_fast3_content_delta_static.csv",
        "mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv",
        "static_protocol_manifest.json",
        "run.log"
    )
    foreach ($file in $files) {
        $srcFile = Join-Path $src $file
        if (Test-Path $srcFile) {
            Copy-Item -LiteralPath $srcFile -Destination (Join-Path $dst $file) -Force
        }
    }
    Set-Content -LiteralPath (Join-Path $dst "seed2025_carryover_note.txt") -Encoding UTF8 -Value @(
        "This directory contains selected final artifacts copied from:"
        $src
        "The full seed2025 run remains in the source directory above."
    )
    Write-Host "[carry] Seed2025 final artifacts ready for $VariantName -> $dst" -ForegroundColor DarkGray
}

function Get-MissingSeeds {
    param(
        [string]$Root,
        [int[]]$Seeds
    )

    $missing = @()
    foreach ($seed in $Seeds) {
        $tag = "${TagPrefix}_$seed"
        $finalPath = Join-Path (Join-Path $Root $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if (-not (Test-Path $finalPath)) {
            $missing += $seed
        }
    }
    return $missing
}

function Run-Variant {
    param(
        [hashtable]$Variant
    )

    $name = [string]$Variant.Name
    $outRoot = Join-Path $TargetRootBase $name
    $ckptRoot = Join-Path $CheckpointRootBase $name

    New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $ckptRoot | Out-Null
    Copy-Seed2025Result -VariantName $name -TargetRoot $outRoot

    $missingSeeds = @(Get-MissingSeeds -Root $outRoot -Seeds $SeedsToRun)
    if ($missingSeeds.Count -lt 1) {
        Write-Host "[skip] $name already has requested seeds: $($SeedsToRun -join ',')" -ForegroundColor Green
        & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed for $name"
        }
        return
    }

    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host "Running $name | missing seeds: $($missingSeeds -join ',') | epochs=$Epochs" -ForegroundColor Cyan
    Write-Host "OutputRoot: $outRoot" -ForegroundColor Cyan
    Write-Host "====================================================================" -ForegroundColor Cyan

    foreach ($seed in $missingSeeds) {
        Write-Host "[run] $name seed=$seed" -ForegroundColor Cyan
        $runnerParams = @{
            PythonRunner = $PythonRunner
            Protocol = $Protocol
            OutputRoot = $outRoot
            CheckpointRoot = $ckptRoot
            ColdThresholds = @(1)
            Seeds = @($seed)
            Epochs = $Epochs
            Patience = $Patience
            EarlyStopAverageMode = "item_macro"
            UseContentDelta = $false
            UsePseudoColdTrain = $false
            RunSampledEval = $false
            SkipAggregate = $true
        }
        foreach ($key in $Variant.ExtraParams.Keys) {
            $runnerParams[$key] = $Variant.ExtraParams[$key]
        }

        & $StaticRunner @runnerParams
        if ($LASTEXITCODE -ne 0) {
            throw "Variant failed: $name seed=$seed"
        }
    }

    & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed for $name"
    }
}

$started = Get-Date
Write-Host "Course ablation overnight run started: $started" -ForegroundColor Cyan
Write-Host "Variants: $($Variants.Name -join ', ')" -ForegroundColor Cyan
Write-Host "SeedsToRun: $($SeedsToRun -join ',')" -ForegroundColor Cyan

foreach ($variant in $Variants) {
    Run-Variant -Variant $variant
}

$finished = Get-Date
Write-Host ""
Write-Host "All requested course ablations finished: $finished" -ForegroundColor Green
Write-Host ("Elapsed: {0:n1} minutes" -f (($finished - $started).TotalMinutes)) -ForegroundColor Green
