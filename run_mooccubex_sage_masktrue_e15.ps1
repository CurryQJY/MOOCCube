param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputRoot = "outputs\mooccubex\sage_lite_v1\S0_masktrue_tail0p002_e15",
    [string]$CheckpointRoot = "checkpoints\mooccubex\sage_lite_v1\S0_masktrue_tail0p002_e15",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 15,
    [int]$Patience = 5,
    [double]$SageTailPopRatio = 0.002,
    [switch]$SkipAggregate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$runnerParams = @{
    PythonRunner = $PythonRunner
    ScriptPath = "usim_feedback_fast3_content_delta.py"
    DataDir = "processed_data_hin_x"
    RelationDir = "MOOCCubeX\relations"
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = $Seeds
    Epochs = $Epochs
    Patience = $Patience
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UsePrereqAux = $true
    PrereqGraphSource = "concept"
    CoursePrereqW = 0.08
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    UseCourseSample = $true
    CourseSampleBeta = 0.20
    UseSageLite = $true
    SageGateMin = 0.10
    SageGateMax = 0.60
    SagePoolTopK = 64
    SageCourseTemp = 0.20
    SageOnlyColdOrTail = $true
    SageTailPopRatio = $SageTailPopRatio
    UseSageAuxLoss = $false
    SageAuxWeight = 0.02
    SageAuxPoolTopK = 48
    SageAuxCourseTemp = 0.20
    SageAuxRetrievalTemp = 1.0
    SageAuxOnlyStrictCold = $true
    SageAuxDetachUser = $true
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    RunSampledEval = $false
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $false
    SaveOptState = $true
}

if ($SkipAggregate) {
    $runnerParams["SkipAggregate"] = $true
}

if ($DryRun) {
    Write-Host "MOOCCubeX mask=true + SAGE dry run"
    foreach ($key in ($runnerParams.Keys | Sort-Object)) {
        $value = $runnerParams[$key]
        if ($value -is [array]) {
            $value = ($value -join ",")
        }
        Write-Host ("{0}={1}" -f $key, $value)
    }
    return
}

& ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams
if ($LASTEXITCODE -ne 0) {
    throw "MOOCCubeX mask=true + SAGE run failed with exit code $LASTEXITCODE"
}
