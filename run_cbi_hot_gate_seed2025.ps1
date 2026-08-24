param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\cbi_hot_gate_single_seed2025"
$checkpointRoot = "checkpoints\cbi_hot_gate_single_seed2025"
$logRoot = "background_logs\cbi_hot_gate_single_seed2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"

$runnerParams = [ordered]@{
    PythonRunner = ".\py.bat"
    ScriptPath = "run_cbi_hot_gate_seed2025.py"
    DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"
    OutputRoot = "outputs\cbi_hot_gate_single_seed2025"
    CheckpointRoot = "checkpoints\cbi_hot_gate_single_seed2025"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 35
    Patience = 8
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $true
    ContentDeltaPaperStyle = $true
    ContentDeltaReplaceItem = $false
    ContentDeltaColdOnly = $false
    ContentDeltaTrainOnIdDropout = $false
    ContentDeltaMode = "embedding"
    ContentDeltaMaxNorm = 0.5
    ContentDeltaScale = 1.0
    ContentDeltaLrMult = 1.0
    ContentDeltaL2W = 0.0
    ContentDeltaCapW = 0.0
    ContentDeltaAuxMode = "base"
    AuxWeight = 0.3
    AuxHotOnly = $false
    UsePseudoColdTrain = $false
    PseudoColdMode = "batch_random"
    PseudoColdRatio = 0.30
    PseudoColdMinPop = 5
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
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
    CourseSampleBeta = 0.20
    UseSageLite = $false
    SageTwoExpertScoreFusion = $false
    UseSageAuxLoss = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    TrainForceCold = $true
    UsimSteps = 5
    UseUsimRefinedEval = $true
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    RunSampledEval = $false
    SaveCkpt = $true
    AutoResume = $false
    ForceFresh = $true
    SaveOptState = $true
}

$lockedConfig = [ordered]@{
    experiment = "cbi_hot_gate_single_seed2025"
    method = "frozen_cbi_delta_plus_initial_cbi_soft_anchor_plus_hot_only_gate"
    hot_only_gate = $true
    cold_bypasses_gate = $true
    normalize_hot_fused_before_simulation = $true
    target_anchor = "initial_cbi"
    inference = "all_item_deterministic_usim_shared_bank"
    selection = "cold_item_macro_n10_screen_with_posthoc_hot_guardrail"
    runner_parameters = $runnerParams
}

if ($DryRun) {
    $lockedConfig | ConvertTo-Json -Depth 20
    exit 0
}

function Get-HashMap([string[]]$Paths) {
    $result = [ordered]@{}
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing reproducibility file: $path" }
        $result[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}

function Write-Json([string]$Path, $Value) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

$sourceFiles = @(
    "run_cbi_hot_gate_seed2025.ps1",
    "run_cbi_hot_gate_seed2025.py",
    "cbi_hot_gate.py",
    "cbi_anchor_sim.py",
    "cbi_trust_sim.py",
    "evaluate_cbi_all_refined_seed2025.py",
    "run_usim_feedback_fast3_content_delta_static.ps1"
)
$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "fast3_delta\eval.py",
    "fast3_delta\config.py",
    "run_fast3_main_table_config.ps1",
    "paper_aaai27\main.tex"
)

New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null
if (Test-Path -LiteralPath $manifestPath) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($existing.status -eq "completed") { throw "Hot-only gate experiment is already completed." }
    throw "Existing incomplete manifest found; inspect before restarting: $manifestPath"
}

$protectedBefore = Get-HashMap $protectedFiles
$manifest = [ordered]@{
    schema_version = 1
    experiment = "cbi_hot_gate_single_seed2025"
    status = "running"
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    completed_at_utc = $null
    elapsed_seconds = $null
    exit_code = $null
    error = $null
    repo = $repoPath
    git_commit = (git rev-parse HEAD).Trim()
    git_dirty = @((git status --porcelain)).Count -gt 0
    locked_config = $lockedConfig
    source_sha256 = Get-HashMap $sourceFiles
    protected_files_before = $protectedBefore
    protected_files_after = $null
    paths = [ordered]@{
        output_root = [System.IO.Path]::GetFullPath((Join-Path $repoPath $outputRoot))
        checkpoint_root = [System.IO.Path]::GetFullPath((Join-Path $repoPath $checkpointRoot))
        log_path = [System.IO.Path]::GetFullPath((Join-Path $repoPath $logPath))
    }
}
Write-Json $manifestPath $manifest

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$runError = $null
try {
    & ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams *>&1 |
        Tee-Object -FilePath $logPath -Append
    if (-not $?) { throw "Static runner returned unsuccessful status." }
    $manifest.status = "completed"
    $manifest.exit_code = 0
}
catch {
    $runError = $_
    $manifest.status = "failed"
    $manifest.exit_code = 1
    $manifest.error = $_.Exception.Message
}
finally {
    $timer.Stop()
    $protectedAfter = Get-HashMap $protectedFiles
    $manifest.protected_files_after = $protectedAfter
    $changed = @($protectedFiles | Where-Object { $protectedBefore[$_] -ne $protectedAfter[$_] })
    if ($changed.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Protected files changed: $($changed -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    Write-Json $manifestPath $manifest
}

if ($null -ne $runError) { throw $runError }
Write-Host "Hot-only gate seed-2025 experiment completed."
