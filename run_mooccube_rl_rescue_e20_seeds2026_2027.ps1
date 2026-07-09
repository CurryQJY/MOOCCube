$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$OutputRoot = "outputs\rl_rescue\mooccube\warm_residual_0p10_e20_foreground"
$CheckpointRoot = "checkpoints\rl_rescue\mooccube\warm_residual_0p10_e20_foreground"
$LauncherLog = Join-Path $OutputRoot "launcher_seeds2026_2027.log"
$Seeds = @(2026, 2027)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

("[{0}] Start MOOCCube RL rescue e20 cold_only seeds={1}" -f (Get-Date -Format o), ($Seeds -join ",")) |
    Out-File -FilePath $LauncherLog -Encoding UTF8

foreach ($Seed in $Seeds) {
    $SeedTag = "strict_item_cold_balanced_thr1_seed_$Seed"
    $InitCheckpointDir = "checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer\$SeedTag"
    if (-not (Test-Path -LiteralPath (Join-Path $InitCheckpointDir "finished.pt"))) {
        throw "Missing content_masked_sup initialization checkpoint: $InitCheckpointDir\finished.pt"
    }

    ("[{0}] Start seed={1} InitCheckpointDir={2}" -f (Get-Date -Format o), $Seed, $InitCheckpointDir) |
        Add-Content -Path $LauncherLog

    $params = @{
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
        Epochs = 20
        Patience = 20
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
        TrainForceCold = $true
        UsimSteps = 5
        PpoLossWeight = 1.0
        RolloutPolicy = "ppo"
        MaskKnownPosNeg = $false
        MaskSameItemNeg = $false
        InitCheckpointDir = $InitCheckpointDir
        RlResidualScale = 0.10
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        SaveCkpt = $true
        AutoResume = $true
        ForceFresh = $false
        SaveOptState = $true
        SkipAggregate = $true
    }

    & .\run_usim_feedback_fast3_content_delta_static.ps1 @params *>&1 |
        Tee-Object -FilePath $LauncherLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "MOOCCube RL rescue e20 seed=$Seed failed with exit code $LASTEXITCODE"
    }

    ("[{0}] Finished seed={1}" -f (Get-Date -Format o), $Seed) |
        Add-Content -Path $LauncherLog
}

("[{0}] Aggregating e20 rescue root {1}" -f (Get-Date -Format o), $OutputRoot) |
    Add-Content -Path $LauncherLog
& .\py.bat "aggregate_fast3_static_results.py" --root $OutputRoot *>&1 |
    Tee-Object -FilePath $LauncherLog -Append
if ($LASTEXITCODE -ne 0) {
    throw "Aggregation failed with exit code $LASTEXITCODE"
}

("[{0}] Finished MOOCCube RL rescue e20 cold_only seeds={1}" -f (Get-Date -Format o), ($Seeds -join ",")) |
    Add-Content -Path $LauncherLog
