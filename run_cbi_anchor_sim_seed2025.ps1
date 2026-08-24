param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$pythonRunner = ".\py.bat"
$staticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\cbi_anchor_sim_single_seed2025"
$checkpointRoot = "checkpoints\cbi_anchor_sim_single_seed2025"
$logRoot = "background_logs\cbi_anchor_sim_single_seed2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"

$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "fast3_delta\eval.py",
    "fast3_delta\config.py",
    "run_fast3_main_table_config.ps1"
)

function Resolve-RepoPath([string]$Path) {
    return [System.IO.Path]::GetFullPath((Join-Path $repoPath $Path))
}

function Get-HashMap([string[]]$Paths) {
    $result = [ordered]@{}
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Required reproducibility file is missing: $path"
        }
        $result[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}

function Write-JsonFile([string]$Path, $Payload) {
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $Payload | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

$runnerParams = [ordered]@{
    PythonRunner = ".\py.bat"
    ScriptPath = "run_cbi_anchor_sim_seed2025.py"
    DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"
    OutputRoot = "outputs\cbi_anchor_sim_single_seed2025"
    CheckpointRoot = "checkpoints\cbi_anchor_sim_single_seed2025"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 10
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $true
    ContentDeltaPaperStyle = $true
    ContentDeltaReplaceItem = $true
    ContentDeltaColdOnly = $false
    ContentDeltaTrainOnIdDropout = $false
    ContentDeltaMode = "embedding"
    ContentDeltaMaxNorm = 0.5
    ContentDeltaScale = 1.0
    ContentDeltaLrMult = 1.0
    ContentDeltaL2W = 0.0
    ContentDeltaCapW = 0.0
    ContentDeltaAuxMode = "base"
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
    AuxHotOnly = $false
    RunSampledEval = $false
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $false
    SaveOptState = $true
}

$lockedConfig = [ordered]@{
    experiment = "cbi_anchor_sim_single_seed2025"
    method = "frozen_cbi_delta_plus_initial_cbi_soft_anchor"
    target_anchor = "initial_cbi"
    hard_projection = $false
    inference = "all_item_deterministic_usim_shared_bank"
    runner_parameters = $runnerParams
}

if ($DryRun) {
    Write-Host "DRY_RUN CBI soft-anchor single experiment"
    Write-Host ("Repo={0}" -f $repoPath)
    Write-Host ("OutputRoot={0}" -f $outputRoot)
    Write-Host ("CheckpointRoot={0}" -f $checkpointRoot)
    Write-Host ("Seed={0} Epochs={1} Patience={2}" -f $runnerParams.Seeds[0], $runnerParams.Epochs, $runnerParams.Patience)
    Write-Host ("TargetAnchor={0} HardProjection={1} AllItemRefinedEval={2}" -f $lockedConfig.target_anchor, $lockedConfig.hard_projection, $runnerParams.UseUsimRefinedEval)
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

if (Test-Path -LiteralPath $manifestPath) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $existingConfig = $existing.locked_config | ConvertTo-Json -Depth 30 -Compress
    $newConfig = $lockedConfig | ConvertTo-Json -Depth 30 -Compress
    if ($existingConfig -ne $newConfig) {
        throw "Existing CBI anchor run has a different locked configuration."
    }
    if ($existing.status -eq "completed") {
        throw "CBI anchor experiment is already completed."
    }
    $archiveName = "run_manifest_attempt_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $outputRoot $archiveName)
}

$sourceFiles = @(
    "run_cbi_anchor_sim_seed2025.ps1",
    "run_cbi_anchor_sim_seed2025.py",
    "cbi_anchor_sim.py",
    "cbi_trust_sim.py",
    "evaluate_cbi_all_refined_seed2025.py",
    "run_usim_feedback_fast3_content_delta_static.ps1"
)
$protectedBefore = Get-HashMap $protectedFiles
$sourceHashes = Get-HashMap $sourceFiles
$startedAt = (Get-Date).ToUniversalTime()
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$manifest = [ordered]@{
    schema_version = 1
    experiment = "cbi_anchor_sim_single_seed2025"
    status = "running"
    started_at_utc = $startedAt.ToString("o")
    completed_at_utc = $null
    elapsed_seconds = $null
    exit_code = $null
    error = $null
    repo = $repoPath
    git_commit = (git rev-parse HEAD).Trim()
    git_dirty = @((git status --porcelain)).Count -gt 0
    locked_config = $lockedConfig
    paths = [ordered]@{
        output_root = (Resolve-RepoPath $outputRoot)
        checkpoint_root = (Resolve-RepoPath $checkpointRoot)
        log_path = (Resolve-RepoPath $logPath)
        seed_output = (Resolve-RepoPath (Join-Path $outputRoot "strict_item_cold_balanced_thr1_seed_2025"))
    }
    source_sha256 = $sourceHashes
    protected_files_before = $protectedBefore
    protected_files_after = $null
}
Write-JsonFile $manifestPath $manifest

$runError = $null
try {
    & $staticRunner @runnerParams *>&1 | Tee-Object -FilePath $logPath -Append
    if (-not $?) {
        throw "CBI anchor static runner returned an unsuccessful status."
    }
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
    $stopwatch.Stop()
    $protectedAfter = Get-HashMap $protectedFiles
    $manifest.protected_files_after = $protectedAfter
    $changed = @()
    foreach ($path in $protectedFiles) {
        if ($protectedBefore[$path] -ne $protectedAfter[$path]) {
            $changed += $path
        }
    }
    if ($changed.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Protected shared code changed: $($changed -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    Write-JsonFile $manifestPath $manifest
}

if ($null -ne $runError) {
    throw $runError
}

Write-Host "CBI soft-anchor seed-2025 experiment completed."
Write-Host ("Manifest: {0}" -f (Resolve-RepoPath $manifestPath))
Write-Host ("Training log: {0}" -f (Resolve-RepoPath $logPath))
