$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$OutputRoot = "outputs\rl_rescue\mooccube\warm_residual_1p00_e20"
$CheckpointRoot = "checkpoints\rl_rescue\mooccube\warm_residual_1p00_e20"
$LauncherLog = Join-Path $OutputRoot "launcher.log"
$Seeds = @(2025, 2026, 2027)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

function Write-LaunchLog {
    param([string]$Message)
    ("[{0}] {1}" -f (Get-Date -Format o), $Message) | Add-Content -Path $LauncherLog
}

function Wait-ForTrainingSlot {
    while ($true) {
        $active = Get-CimInstance Win32_Process |
            Where-Object {
                $_.CommandLine -match "usim_feedback_fast3_content_delta.py" -and
                $_.CommandLine -notmatch "Get-CimInstance"
            }
        if (-not $active) {
            return
        }
        $pids = ($active | ForEach-Object { $_.ProcessId }) -join ","
        Write-LaunchLog "Waiting for existing training process(es): $pids"
        Start-Sleep -Seconds 60
    }
}

("[{0}] Start MOOCCube RL rescue alpha=1.0 e20 cold_only seeds={1}" -f (Get-Date -Format o), ($Seeds -join ",")) |
    Out-File -FilePath $LauncherLog -Encoding UTF8

Wait-ForTrainingSlot

foreach ($Seed in $Seeds) {
    $SeedTag = "strict_item_cold_balanced_thr1_seed_$Seed"
    $InitCheckpointDir = "checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer\$SeedTag"
    if (-not (Test-Path -LiteralPath (Join-Path $InitCheckpointDir "finished.pt"))) {
        throw "Missing content_masked_sup initialization checkpoint: $InitCheckpointDir\finished.pt"
    }

    Write-LaunchLog "Start seed=$Seed InitCheckpointDir=$InitCheckpointDir"

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
        RlResidualScale = 1.0
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
        throw "MOOCCube RL rescue alpha=1.0 e20 seed=$Seed failed with exit code $LASTEXITCODE"
    }

    Write-LaunchLog "Finished seed=$Seed"
}

Write-LaunchLog "Aggregating alpha=1.0 e20 rescue root $OutputRoot"
& .\py.bat "aggregate_fast3_static_results.py" --root $OutputRoot *>&1 |
    Tee-Object -FilePath $LauncherLog -Append
if ($LASTEXITCODE -ne 0) {
    throw "Aggregation failed with exit code $LASTEXITCODE"
}

Write-LaunchLog "Finished MOOCCube RL rescue alpha=1.0 e20 cold_only seeds=$($Seeds -join ',')"
