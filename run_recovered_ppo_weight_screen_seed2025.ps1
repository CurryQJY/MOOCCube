param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$WaitPid = 0,
    [double[]]$Weights = @(0.05, 0.10, 0.25, 0.50, 1.00),
    [int]$Epochs = 20,
    [int]$Patience = 5
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

if ($WaitPid -gt 0) {
    Write-Host ("===== WAITING FOR CORE ABLATIONS PID {0} =====" -f $WaitPid)
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
}

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$common = @{
    ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = $Epochs
    Patience = $Patience
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
    UsimSteps = 5
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
    ForceFresh = $true
    SaveOptState = $true
    SkipAggregate = $true
}

foreach ($weight in $Weights) {
    $label = $weight.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
    $params = @{}
    foreach ($key in $common.Keys) { $params[$key] = $common[$key] }
    $params.PpoLossWeight = $weight
    $params.OutputRoot = "outputs\recovery_validation\main_table_51ea12fc_ppo_weight_screen\ppo_w$label"
    $params.CheckpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_ppo_weight_screen\ppo_w$label"
    Write-Host ("===== PPO WEIGHT {0} START {1} =====" -f $weight, (Get-Date -Format s))
    & $runner @params
    if ($LASTEXITCODE -ne 0) { throw "PPO weight run failed: $weight" }
    Write-Host ("===== PPO WEIGHT {0} DONE {1} =====" -f $weight, (Get-Date -Format s))
}
