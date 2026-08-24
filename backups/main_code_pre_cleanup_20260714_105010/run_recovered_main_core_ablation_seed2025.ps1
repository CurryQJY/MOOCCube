param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string[]]$Variants = @("wo_ppo_loss", "wo_simulator", "wo_course_reward"),
    [string]$SeedList = "2025",
    [int]$WaitPid = 0,
    [bool]$ForceFresh = $false
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

if ($WaitPid -gt 0) {
    Write-Host ("===== WAITING FOR PID {0} {1} =====" -f $WaitPid, (Get-Date -Format s))
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
    Write-Host ("===== WAIT COMPLETE PID {0} {1} =====" -f $WaitPid, (Get-Date -Format s))
}

$Seeds = @(
    $SeedList -split "[,\s]+" |
        Where-Object { $_.Trim().Length -gt 0 } |
        ForEach-Object { [int]$_.Trim() }
)
if ($Seeds.Count -lt 1) {
    throw "SeedList must contain at least one seed"
}

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$common = @{
    ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = $Seeds
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    PseudoColdMode = "batch_random"
    PseudoColdRatio = 0.3
    PseudoColdMinPop = 5
    UsePaac = $false
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
    RolloutPolicy = "ppo"
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    RunSampledEval = $false
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $ForceFresh
    SaveOptState = $true
    SkipAggregate = $true
}

$definitions = @{
    wo_ppo_loss = @{ UsimSteps = 5; PpoLossWeight = 0.0 }
    wo_simulator = @{ UsimSteps = 0; PpoLossWeight = 1.0 }
    wo_course_reward = @{ UsimSteps = 5; PpoLossWeight = 1.0; UseCourseReward = $false }
}

foreach ($name in $Variants) {
    if (-not $definitions.ContainsKey($name)) {
        throw "Unknown ablation variant: $name"
    }
    Write-Host ("===== ABLATION START {0} {1} =====" -f $name, (Get-Date -Format s))
    $params = @{}
    foreach ($key in $common.Keys) { $params[$key] = $common[$key] }
    foreach ($key in $definitions[$name].Keys) { $params[$key] = $definitions[$name][$key] }
    $params.OutputRoot = "outputs\recovery_validation\main_table_51ea12fc_core_ablation\$name"
    $params.CheckpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_core_ablation\$name"
    & $runner @params
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation failed: $name, exit=$LASTEXITCODE"
    }
    Write-Host ("===== ABLATION DONE {0} {1} =====" -f $name, (Get-Date -Format s))
}
