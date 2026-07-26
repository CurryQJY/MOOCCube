param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$VariantList = "",
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$OutputRootBase = "outputs\significance_per_item_exports\mooccube\ckg_rl_true_true_hparam_grid",
    [string]$CheckpointRootBase = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_true_true_hparam_grid",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$Protocol = "strict_item_cold_balanced"
$TagPrefix = "strict_item_cold_balanced_thr1_seed"

if ($SeedList.Trim().Length -gt 0) {
    $SeedsToRun = @(
        $SeedList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { [int]$_.Trim() }
    )
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $repoPath $PathValue)
}

$OutputRootBaseAbs = Resolve-RepoPath $OutputRootBase
$CheckpointRootBaseAbs = Resolve-RepoPath $CheckpointRootBase
$QueueLog = Join-Path $OutputRootBaseAbs "true_true_hparam_grid_queue.log"
$PlanJson = $null
$PlanCsv = $null
$PlanFileMode = "full_grid"

function Write-QueueLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
        [System.IO.File]::AppendAllText($QueueLog, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    Write-Host $line
}

function Write-Setting {
    param([string]$Name, [object]$Value)
    Write-Host ("{0}={1}" -f $Name, $Value)
}

function New-Variant {
    param(
        [string]$Name,
        [string]$Sweep,
        [double]$X,
        [string]$Label,
        [string]$Rationale,
        [hashtable]$Params
    )
    return [ordered]@{
        Name = $Name
        Sweep = $Sweep
        X = $X
        Label = $Label
        Rationale = $Rationale
        Params = $Params
    }
}

$Variants = @(
    (New-Variant -Name "beta_0p00" -Sweep "beta" -X 0.00 -Label "0.00" -Rationale "Remove knowledge-guided candidate sampling to anchor the beta sweep." -Params @{ CourseSampleBeta = 0.00 }),
    (New-Variant -Name "beta_0p10" -Sweep "beta" -X 0.10 -Label "0.10" -Rationale "Lower knowledge-guided candidate sampling weight." -Params @{ CourseSampleBeta = 0.10 }),
    (New-Variant -Name "beta_0p15" -Sweep "beta" -X 0.15 -Label "0.15" -Rationale "Test the best nearby value suggested by the previous validation trend." -Params @{ CourseSampleBeta = 0.15 }),
    (New-Variant -Name "beta_0p25" -Sweep "beta" -X 0.25 -Label "0.25" -Rationale "Probe the immediate neighborhood above the default beta." -Params @{ CourseSampleBeta = 0.25 }),
    (New-Variant -Name "beta_0p30" -Sweep "beta" -X 0.30 -Label "0.30" -Rationale "Raise knowledge-guided candidate sampling weight." -Params @{ CourseSampleBeta = 0.30 }),
    (New-Variant -Name "beta_0p50" -Sweep "beta" -X 0.50 -Label "0.50" -Rationale "Stress-test a stronger knowledge-guided sampling weight." -Params @{ CourseSampleBeta = 0.50 }),
    (New-Variant -Name "reward_0p00" -Sweep "reward" -X 0.00 -Label "0.00" -Rationale "Remove educational reward shaping to anchor the reward-scale sweep." -Params @{
        CoursePrereqW = 0.00
        CourseConceptW = 0.00
        CourseDiffW = 0.00
        CourseRedundantW = 0.00
    }),
    (New-Variant -Name "reward_0p50" -Sweep "reward" -X 0.50 -Label "0.50" -Rationale "Halve all educational reward terms." -Params @{
        CoursePrereqW = 0.04
        CourseConceptW = 0.02
        CourseDiffW = 0.015
        CourseRedundantW = 0.01
    }),
    (New-Variant -Name "reward_1p50" -Sweep "reward" -X 1.50 -Label "1.50" -Rationale "Increase all educational reward terms by 50 percent." -Params @{
        CoursePrereqW = 0.12
        CourseConceptW = 0.06
        CourseDiffW = 0.045
        CourseRedundantW = 0.03
    }),
    (New-Variant -Name "reward_2p00" -Sweep "reward" -X 2.00 -Label "2.00" -Rationale "Double all educational reward terms." -Params @{
        CoursePrereqW = 0.16
        CourseConceptW = 0.08
        CourseDiffW = 0.06
        CourseRedundantW = 0.04
    }),
    (New-Variant -Name "horizon_1" -Sweep "horizon" -X 1.0 -Label "1" -Rationale "Single-step learner simulation to anchor the horizon sweep." -Params @{ UsimSteps = 1 }),
    (New-Variant -Name "horizon_3" -Sweep "horizon" -X 3.0 -Label "3" -Rationale "Shorter learner-simulation horizon." -Params @{ UsimSteps = 3 }),
    (New-Variant -Name "horizon_7" -Sweep "horizon" -X 7.0 -Label "7" -Rationale "Longer learner-simulation horizon." -Params @{ UsimSteps = 7 }),
    (New-Variant -Name "horizon_10" -Sweep "horizon" -X 10.0 -Label "10" -Rationale "Stress-test a substantially longer learner-simulation horizon." -Params @{ UsimSteps = 10 })
)

if ($VariantList.Trim().Length -gt 0) {
    $wanted = @(
        $VariantList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Trim() }
    )
    $Variants = @($Variants | Where-Object { $wanted -contains $_.Name })
    if ($Variants.Count -lt 1) {
        throw "No variants matched VariantList='$VariantList'"
    }
    $planSuffix = (($Variants.Name -join "_") -replace "[^A-Za-z0-9_]+", "_")
    $PlanJson = Join-Path $OutputRootBaseAbs "true_true_hparam_grid_plan_$planSuffix.json"
    $PlanCsv = Join-Path $OutputRootBaseAbs "true_true_hparam_grid_plan_$planSuffix.csv"
    $PlanFileMode = "variant_list_scoped"
} else {
    $PlanJson = Join-Path $OutputRootBaseAbs "true_true_hparam_grid_plan.json"
    $PlanCsv = Join-Path $OutputRootBaseAbs "true_true_hparam_grid_plan.csv"
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
    New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
    $rows = foreach ($variant in $Variants) {
        [pscustomobject]@{
            variant = $variant.Name
            sweep = $variant.Sweep
            x = $variant.X
            label = $variant.Label
            seeds = ($SeedsToRun -join ",")
            mask_known_pos_neg = $true
            mask_same_item_neg = $true
            selection_metric = "validation cold item-macro NDCG@10"
            output_root = (Join-Path $OutputRootBaseAbs $variant.Name)
            checkpoint_root = (Join-Path $CheckpointRootBaseAbs $variant.Name)
            rationale = $variant.Rationale
        }
    }
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PlanJson -Encoding UTF8
    $rows | Export-Csv -LiteralPath $PlanCsv -NoTypeInformation -Encoding UTF8
}

function New-BaseRunnerParams {
    param([string]$OutRoot, [string]$CkptRoot, [int]$Seed)
    return @{
        PythonRunner = $PythonRunner
        Protocol = $Protocol
        OutputRoot = $OutRoot
        CheckpointRoot = $CkptRoot
        ColdThresholds = @(1)
        Seeds = @($Seed)
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
}

Write-PlanFiles

if ($DryRun) {
    Write-Host "MOOCCube True/True validation-only hparam grid dry run"
    Write-Setting "Repo" $repoPath
    Write-Setting "Protocol" $Protocol
    Write-Setting "OutputRootBase" $OutputRootBase
    Write-Setting "CheckpointRootBase" $CheckpointRootBase
    Write-Setting "Seeds" ($SeedsToRun -join ",")
    Write-Setting "Epochs" $Epochs
    Write-Setting "Patience" $Patience
    Write-Setting "EarlyStopAverageMode" "item_macro"
    Write-Setting "EarlyStopScoreMode" "cold_only"
    Write-Setting "RunSampledEval" $false
    Write-Setting "MaskKnownPosNeg" $true
    Write-Setting "MaskSameItemNeg" $true
    Write-Setting "SaveCkpt" $true
    Write-Setting "AutoResume" $true
    Write-Setting "SaveOptState" $true
    Write-Setting "PlanFileMode" $PlanFileMode
    Write-Setting "PlanJson" $PlanJson
    Write-Setting "PlanCsv" $PlanCsv
}

Write-QueueLine "MOOCCube True/True validation-only hparam grid | variants=$($Variants.Name -join ',') | seeds=$($SeedsToRun -join ',')"

foreach ($variant in $Variants) {
    $name = [string]$variant.Name
    $outRoot = Join-Path $OutputRootBaseAbs $name
    $ckptRoot = Join-Path $CheckpointRootBaseAbs $name
    $missingSeeds = @(Get-MissingSeeds -Root $outRoot -Seeds $SeedsToRun)

    if ($DryRun) {
        Write-Setting "Variant" $name
        foreach ($key in $variant.Params.Keys) {
            Write-Setting $key $variant.Params[$key]
        }
        continue
    }

    if ($missingSeeds.Count -lt 1) {
        Write-QueueLine "SKIP $name; all requested finals already exist."
        & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed for $name"
        }
        continue
    }

    Write-QueueLine "START $name | sweep=$($variant.Sweep) | x=$($variant.X) | missing_seeds=$($missingSeeds -join ',')"
    foreach ($seed in $missingSeeds) {
        $runnerParams = New-BaseRunnerParams -OutRoot $outRoot -CkptRoot $ckptRoot -Seed $seed
        foreach ($key in $variant.Params.Keys) {
            $runnerParams[$key] = $variant.Params[$key]
        }
        Write-QueueLine "RUN $name seed=$seed"
        & $StaticRunner @runnerParams
        if ($LASTEXITCODE -ne 0) {
            throw "Variant failed: $name seed=$seed"
        }
    }

    & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed for $name"
    }
    Write-QueueLine "END $name"
}

Write-QueueLine "DONE MOOCCube True/True validation-only hparam grid"
