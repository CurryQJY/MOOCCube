param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$staticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\cbi_anchor_sim_3seed_serial"
$checkpointRoot = "checkpoints\cbi_anchor_sim_3seed_serial"
$logRoot = "background_logs\cbi_anchor_sim_3seed_serial"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"
$seed2025SourceRoot = "outputs\cbi_anchor_sim_single_seed2025"
$seed2025SourceManifestPath = Join-Path $seed2025SourceRoot "run_manifest.json"
$seed2025Tag = "strict_item_cold_balanced_thr1_seed_2025"

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
    OutputRoot = "outputs\cbi_anchor_sim_3seed_serial"
    CheckpointRoot = "checkpoints\cbi_anchor_sim_3seed_serial"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2026, 2027)
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
    experiment = "cbi_anchor_sim_3seed_serial"
    execution = "serial"
    method = "frozen_cbi_delta_plus_initial_cbi_soft_anchor"
    target_anchor = "initial_cbi"
    hard_projection = $false
    inference = "all_item_deterministic_usim_shared_bank"
    aggregate_seeds = @(2025, 2026, 2027)
    trained_seeds = @(2026, 2027)
    reused_seed = 2025
    reused_seed_root = "outputs\cbi_anchor_sim_single_seed2025"
    runner_parameters = $runnerParams
}

if ($DryRun) {
    Write-Host "DRY_RUN CBI soft-anchor three-seed serial experiment"
    Write-Host ("Repo={0}" -f $repoPath)
    Write-Host ("OutputRoot={0}" -f $outputRoot)
    Write-Host ("CheckpointRoot={0}" -f $checkpointRoot)
    Write-Host ("AggregateSeeds={0} TrainedSeeds={1} ReusedSeed={2}" -f ($lockedConfig.aggregate_seeds -join ","), ($runnerParams.Seeds -join ","), $lockedConfig.reused_seed)
    Write-Host ("Epochs={0} Patience={1} Execution={2}" -f $runnerParams.Epochs, $runnerParams.Patience, $lockedConfig.execution)
    Write-Host ("TargetAnchor={0} HardProjection={1} AllItemRefinedEval={2}" -f $lockedConfig.target_anchor, $lockedConfig.hard_projection, $runnerParams.UseUsimRefinedEval)
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

if (-not (Test-Path -LiteralPath $seed2025SourceManifestPath)) {
    throw "Completed seed-2025 source manifest is missing: $seed2025SourceManifestPath"
}
$seed2025SourceManifest = Get-Content -LiteralPath $seed2025SourceManifestPath -Raw | ConvertFrom-Json
if ($seed2025SourceManifest.status -ne "completed") {
    throw "Seed-2025 source run is not completed: status=$($seed2025SourceManifest.status)"
}
if (
    $seed2025SourceManifest.locked_config.method -ne $lockedConfig.method -or
    $seed2025SourceManifest.locked_config.target_anchor -ne $lockedConfig.target_anchor -or
    [bool]$seed2025SourceManifest.locked_config.hard_projection -ne [bool]$lockedConfig.hard_projection
) {
    throw "Seed-2025 source method does not match the three-seed method."
}
$reusedSourceFiles = @(
    "run_cbi_anchor_sim_seed2025.py",
    "cbi_anchor_sim.py",
    "cbi_trust_sim.py",
    "evaluate_cbi_all_refined_seed2025.py",
    "run_usim_feedback_fast3_content_delta_static.ps1"
)
foreach ($path in $reusedSourceFiles) {
    $expectedHash = $seed2025SourceManifest.source_sha256.PSObject.Properties[$path].Value
    $currentHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $expectedHash -or $expectedHash -ne $currentHash) {
        throw "Seed-2025 source hash mismatch for $path"
    }
}

$seed2025SourceDir = Join-Path $seed2025SourceRoot $seed2025Tag
$seed2025TargetDir = Join-Path $outputRoot $seed2025Tag
if (-not (Test-Path -LiteralPath $seed2025SourceDir)) {
    throw "Completed seed-2025 result directory is missing: $seed2025SourceDir"
}
if (-not (Test-Path -LiteralPath $seed2025TargetDir)) {
    Copy-Item -LiteralPath $seed2025SourceDir -Destination $seed2025TargetDir -Recurse
}
if (-not (Test-Path -LiteralPath (Join-Path $seed2025TargetDir "final_report_usim_feedback_fast3_content_delta_static.csv"))) {
    throw "Reused seed-2025 result is incomplete in $seed2025TargetDir"
}

if (Test-Path -LiteralPath $manifestPath) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $existingConfig = $existing.locked_config | ConvertTo-Json -Depth 30 -Compress
    $newConfig = $lockedConfig | ConvertTo-Json -Depth 30 -Compress
    if ($existingConfig -ne $newConfig) {
        throw "Existing CBI anchor three-seed run has a different locked configuration."
    }
    if ($existing.status -eq "completed") {
        throw "CBI anchor three-seed experiment is already completed."
    }
    $archiveName = "run_manifest_attempt_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $outputRoot $archiveName)
}

$sourceFiles = @(
    "run_cbi_anchor_sim_3seed_serial.ps1",
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

$seedOutputs = @()
foreach ($seed in $lockedConfig.aggregate_seeds) {
    $seedOutputs += (Resolve-RepoPath (Join-Path $outputRoot ("strict_item_cold_balanced_thr1_seed_{0}" -f $seed)))
}

$manifest = [ordered]@{
    schema_version = 1
    experiment = "cbi_anchor_sim_3seed_serial"
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
        seed_outputs = $seedOutputs
        aggregate_detail = (Resolve-RepoPath (Join-Path $outputRoot "fast3_static_runs_detail.csv"))
        aggregate_summary = (Resolve-RepoPath (Join-Path $outputRoot "fast3_static_multiseed_summary.csv"))
    }
    source_sha256 = $sourceHashes
    reused_seed = [ordered]@{
        seed = 2025
        source_manifest = (Resolve-RepoPath $seed2025SourceManifestPath)
        source_manifest_sha256 = (Get-FileHash -LiteralPath $seed2025SourceManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        source_result = (Resolve-RepoPath $seed2025SourceDir)
        copied_result = (Resolve-RepoPath $seed2025TargetDir)
    }
    protected_files_before = $protectedBefore
    protected_files_after = $null
}
Write-JsonFile $manifestPath $manifest

$runError = $null
try {
    & $staticRunner @runnerParams *>&1 | Tee-Object -FilePath $logPath -Append
    if (-not $?) {
        throw "CBI anchor three-seed static runner returned an unsuccessful status."
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

Write-Host "CBI soft-anchor three-seed serial experiment completed."
Write-Host ("Manifest: {0}" -f (Resolve-RepoPath $manifestPath))
Write-Host ("Training log: {0}" -f (Resolve-RepoPath $logPath))
