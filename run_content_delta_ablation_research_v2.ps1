# Single-variable ContentDelta ablation under research-v2 recipe.
# Only UseContentDelta differs between arms.
param(
    [ValidateSet("both", "on", "off")]
    [string]$Arm = "off",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 30,
    [int]$Patience = 12
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Call the base static runner with repaired ScriptPath so [bool] params bind correctly.
$baseRunner = Join-Path $PSScriptRoot "run_usim_feedback_fast3_content_delta_static.ps1"
$rootOut = "outputs\recppo_research_repair\content_delta_ablation_v2"
$rootCkpt = "checkpoints\recppo_research_repair\content_delta_ablation_v2"
$logDir = Join-Path $rootOut "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Match repaired wrapper defaults / research_v2 env.
$env:USIM_RECPPO_EARLY_STOP_MODE = "recppo_stage_guarded"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:USIM_RECPPO_STRICT_DETERMINISM = "1"

function Invoke-DeltaArm {
    param(
        [bool]$UseContentDelta,
        [string]$Tag
    )

    $outRoot = Join-Path $rootOut $Tag
    $ckptRoot = Join-Path $rootCkpt $Tag
    New-Item -ItemType Directory -Force -Path $outRoot, $ckptRoot | Out-Null

    $logOut = Join-Path $logDir ("{0}.out.log" -f $Tag)
    Write-Host "===== ContentDelta ablation arm=$Tag UseContentDelta=$UseContentDelta =====" -ForegroundColor Cyan
    Write-Host "OutputRoot=$outRoot"
    Write-Host "CheckpointRoot=$ckptRoot"
    Write-Host "Log=$logOut"

    $argsMap = @{
        PythonRunner              = ".\py.bat"
        ScriptPath                = "usim_feedback_fast3_content_delta_repaired.py"
        OutputRoot                = $outRoot
        CheckpointRoot            = $ckptRoot
        Protocol                  = "strict_item_cold_balanced"
        ColdThresholds            = @(1)
        Seeds                     = $Seeds
        Epochs                    = $Epochs
        Patience                  = $Patience
        UseContentDelta           = $UseContentDelta
        ContentDeltaMode          = "embedding"
        ContentDeltaColdOnly      = $true
        ContentDeltaTrainOnIdDropout = $true
        ContentDeltaScale         = 0.25
        ContentDeltaMaxNorm       = 0.05
        ContentDeltaLrMult        = 0.10
        ContentDeltaL2W           = 0.02
        ContentDeltaCapW          = 0.02
        UsePseudoColdTrain        = $true
        PseudoColdMode            = "all_eligible"
        PseudoColdRatio           = 1.0
        PseudoColdMinPop          = 1
        PpoLossWeight             = 1.0
        RolloutPolicy             = "ppo"
        RlResidualScale           = 0.30
        UsimSteps                 = 5
        UseCourseFeedback         = $true
        UseCourseReward           = $true
        UsePrereqAux              = $true
        UseCourseSample           = $true
        UseUsimRefinedEval        = $true
        SaveCkpt                  = $true
        ForceFresh                = $true
        AutoResume                = $false
        SaveOptState              = $true
    }

    & $baseRunner @argsMap *>&1 | Tee-Object -FilePath $logOut
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "Arm $Tag failed with exit code $code" -ForegroundColor Red
        if (Test-Path $logOut) { Get-Content $logOut -Tail 50 }
        throw "ContentDelta ablation arm failed: $Tag (exit=$code)"
    }
    Write-Host "Arm $Tag finished OK" -ForegroundColor Green
}

$started = Get-Date
switch ($Arm) {
    "on"  { Invoke-DeltaArm -UseContentDelta $true  -Tag "delta_on" }
    "off" { Invoke-DeltaArm -UseContentDelta $false -Tag "delta_off" }
    "both" {
        Invoke-DeltaArm -UseContentDelta $true  -Tag "delta_on"
        Invoke-DeltaArm -UseContentDelta $false -Tag "delta_off"
    }
}

Write-Host ("Ablation arm(s) done in {0:n1} min" -f ((Get-Date) - $started).TotalMinutes)
exit 0
