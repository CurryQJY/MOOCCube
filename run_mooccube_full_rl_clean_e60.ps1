param(
    [int[]]$Seeds = @(2025, 2026, 2027),
    [string]$OutputRoot = "outputs\significance_per_item_exports\mooccube\ckg_rl_full_clean_maskff_e60",
    [string]$CheckpointRoot = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_full_clean_maskff_e60"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$LauncherLog = Join-Path $OutputRoot "launcher.log"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

("[{0}] Start clean Full RL e60 seeds={1}" -f (Get-Date -Format o), ($Seeds -join ",")) |
    Out-File -FilePath $LauncherLog -Encoding UTF8

$params = @{
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = $Seeds
    Epochs = 60
    Patience = 60

    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"

    # Main-paper Full RL components.
    UseContentDelta = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false

    # Keep the historical main-table mask setting explicit.
    MaskKnownPosNeg = $false
    MaskSameItemNeg = $false

    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $false
}

try {
    & .\run_usim_feedback_fast3_content_delta_static.ps1 @params *>&1 |
        Tee-Object -FilePath $LauncherLog -Append
    $code = $LASTEXITCODE
    ("[{0}] Finished clean Full RL e60 exit_code={1}" -f (Get-Date -Format o), $code) |
        Add-Content -Path $LauncherLog
    exit $code
} catch {
    ("[{0}] Failed clean Full RL e60: {1}" -f (Get-Date -Format o), $_.Exception.Message) |
        Add-Content -Path $LauncherLog
    exit 1
}
