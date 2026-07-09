param(
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$VariantList = "",
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$TargetRootBase = "outputs\significance_per_item_exports\mooccube\ckg_rl_true_true_ablation",
    [string]$CheckpointRootBase = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_true_true_ablation",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

function Write-QueueLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $TargetRootBase | Out-Null
        [System.IO.File]::AppendAllText(
            (Join-Path $TargetRootBase "true_true_component_ablation_queue.log"),
            $line + [Environment]::NewLine,
            [System.Text.Encoding]::UTF8
        )
    }
    Write-Host $line
}

function Get-FinalPath {
    param([string]$Root, [int]$Seed)
    $tag = "${TagPrefix}_$Seed"
    return Join-Path (Join-Path $Root $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Get-MissingSeeds {
    param([string]$Root, [int[]]$Seeds)
    $missing = @()
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Get-FinalPath -Root $Root -Seed $seed))) {
            $missing += $seed
        }
    }
    return $missing
}

function Write-PlanFiles {
    if ($DryRun) {
        return
    }
    New-Item -ItemType Directory -Force -Path $TargetRootBase | Out-Null
    $rows = foreach ($variant in $Variants) {
        [pscustomobject]@{
            variant = $variant.Name
            seeds = ($SeedsToRun -join ",")
            mask_known_pos_neg = $true
            mask_same_item_neg = $true
            output_root = (Join-Path $TargetRootBase $variant.Name)
            checkpoint_root = (Join-Path $CheckpointRootBase $variant.Name)
        }
    }
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $TargetRootBase "true_true_component_ablation_plan.json") -Encoding UTF8
    $rows | Export-Csv -LiteralPath (Join-Path $TargetRootBase "true_true_component_ablation_plan.csv") -NoTypeInformation -Encoding UTF8
}

Write-PlanFiles
Write-QueueLine "MOOCCube True/True component ablation queue | variants=$($Variants.Name -join ',') | seeds=$($SeedsToRun -join ',')"

foreach ($variant in $Variants) {
    $name = [string]$variant.Name
    $outRoot = Join-Path $TargetRootBase $name
    $ckptRoot = Join-Path $CheckpointRootBase $name
    $missingSeeds = @(Get-MissingSeeds -Root $outRoot -Seeds $SeedsToRun)

    if ($missingSeeds.Count -lt 1) {
        Write-QueueLine "SKIP $name; all requested finals already exist."
        if (-not $DryRun) {
            & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Aggregation failed for $name"
            }
        }
        continue
    }

    Write-QueueLine "START $name | missing_seeds=$($missingSeeds -join ',') | output=$outRoot"
    foreach ($seed in $missingSeeds) {
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
            EarlyStopScoreMode = "cold_only"
            UseContentDelta = $false
            UsePseudoColdTrain = $false
            UsePaac = $false
            UseCourseFeedback = $true
            UseCourseReward = $true
            UseCourseSample = $true
            UsePrereqAux = $true
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
            CoursePrereqW = 0.08
            CoursePrereqGate = 0.20
            CourseConceptW = 0.04
            CourseDiffW = 0.03
            CourseRedundantW = 0.02
            CourseRedundantConceptGate = 1.0
            CourseRedundantMode = "concept"
            CourseTermNorm = "none"
            CourseSampleBeta = 0.20
            TrainForceCold = $true
            UsimSteps = 5
            PpoLossWeight = 1.0
            MaskKnownPosNeg = $true
            MaskSameItemNeg = $true
            RunSampledEval = $false
            SaveCkpt = $true
            AutoResume = $true
            ForceFresh = $false
            SaveOptState = $true
            SkipAggregate = $true
        }
        foreach ($key in $variant.ExtraParams.Keys) {
            $runnerParams[$key] = $variant.ExtraParams[$key]
        }

        Write-QueueLine "RUN $name seed=$seed"
        if (-not $DryRun) {
            & $StaticRunner @runnerParams
            if ($LASTEXITCODE -ne 0) {
                throw "Variant failed: $name seed=$seed"
            }
        }
    }

    if (-not $DryRun) {
        & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed for $name"
        }
    }
    Write-QueueLine "END $name"
}

Write-QueueLine "DONE MOOCCube True/True component ablation queue"
