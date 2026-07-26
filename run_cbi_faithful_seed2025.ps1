param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun,
    [switch]$ForceFresh
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$pythonRunner = ".\py.bat"
$staticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1"
$scriptPath = "usim_feedback_fast3_content_delta.py"
$outputRoot = "outputs\cbi_faithful_single_seed2025"
$checkpointRoot = "checkpoints\cbi_faithful_single_seed2025"
$logRoot = "background_logs\cbi_faithful_single_seed2025"
$seedTag = "strict_item_cold_balanced_thr1_seed_2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"

$protectedRoots = @(
    "outputs\content_delta_pop5\course_maincfg_runs\maincfg",
    "outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "checkpoints\content_delta_pop5\course_maincfg_runs\maincfg",
    "checkpoints\content_delta_pop5\course_ablation_e60_3seed\full"
)
$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "run_fast3_main_table_config.ps1",
    "paper_aaai27\main.tex",
    "paper_aaai27\main_table.tex"
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
    $Payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding UTF8
}

foreach ($candidate in @($outputRoot, $checkpointRoot)) {
    $resolved = Resolve-RepoPath $candidate
    foreach ($protected in $protectedRoots) {
        $protectedResolved = Resolve-RepoPath $protected
        $nestedPrefix = $protectedResolved + [System.IO.Path]::DirectorySeparatorChar
        if (
            $resolved.Equals($protectedResolved, [System.StringComparison]::OrdinalIgnoreCase) -or
            $resolved.StartsWith($nestedPrefix, [System.StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "CBI experiment path overlaps protected main-table path: $resolved"
        }
    }
}

$runnerParams = [ordered]@{
    PythonRunner = $pythonRunner
    ScriptPath = $scriptPath
    DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"
    OutputRoot = $outputRoot
    CheckpointRoot = $checkpointRoot
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 60
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
    AutoResume = $false
    ForceFresh = $true
    SaveOptState = $true
}

$lockedConfig = [ordered]@{
    experiment = "cbi_faithful_single_seed2025"
    method = "frozen_standardized_pca_plus_bounded_delta"
    delta_min_cosine_bound = [Math]::Sqrt(1.0 - [Math]::Pow(0.5, 2))
    runner_parameters = $runnerParams
}

if ($DryRun) {
    Write-Host "DRY_RUN CBI faithful single experiment"
    Write-Host ("Repo={0}" -f $repoPath)
    Write-Host ("OutputRoot={0}" -f $outputRoot)
    Write-Host ("CheckpointRoot={0}" -f $checkpointRoot)
    Write-Host ("LogRoot={0}" -f $logRoot)
    Write-Host ("Seed={0} Epochs={1} Patience={2}" -f $runnerParams.Seeds[0], $runnerParams.Epochs, $runnerParams.Patience)
    Write-Host ("CBI paper_style={0} replace_item={1} cold_only={2} delta_max={3} scale={4}" -f $runnerParams.ContentDeltaPaperStyle, $runnerParams.ContentDeltaReplaceItem, $runnerParams.ContentDeltaColdOnly, $runnerParams.ContentDeltaMaxNorm, $runnerParams.ContentDeltaScale)
    exit 0
}

if (Test-Path -LiteralPath $manifestPath) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $existingConfig = $existing.locked_config | ConvertTo-Json -Depth 20 -Compress
    $newConfig = $lockedConfig | ConvertTo-Json -Depth 20 -Compress
    if ($existingConfig -ne $newConfig) {
        throw "Existing CBI run has a different locked configuration; refusing to reuse its directories."
    }
    if (-not $ForceFresh) {
        throw "CBI run manifest already exists with status '$($existing.status)'; use -ForceFresh only for an explicit rerun."
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

$sourceFiles = @(
    "run_cbi_faithful_seed2025.ps1",
    "run_usim_feedback_fast3_content_delta_static.ps1",
    "usim_feedback_fast3_content_delta.py",
    "fast3_delta\config.py",
    "fast3_delta\eval.py",
    "fast3_delta\provenance.py"
)
$protected_files_before = Get-HashMap $protectedFiles
$sourceHashes = Get-HashMap $sourceFiles
$runtimeJson = & $pythonRunner -c "import json,platform,sys,torch; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'platform':platform.platform()}))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to collect Python runtime metadata."
}
$pythonRuntime = $runtimeJson | ConvertFrom-Json
$gitCommit = (git rev-parse HEAD).Trim()
$gitDirty = @((git status --porcelain)).Count -gt 0
$startedAt = (Get-Date).ToUniversalTime()
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$manifest = [ordered]@{
    schema_version = 1
    experiment = "cbi_faithful_single_seed2025"
    status = "running"
    started_at_utc = $startedAt.ToString("o")
    completed_at_utc = $null
    elapsed_seconds = $null
    exit_code = $null
    error = $null
    repo = $repoPath
    git_commit = $gitCommit
    git_dirty = $gitDirty
    normalized_command = "powershell -NoProfile -ExecutionPolicy Bypass -File run_cbi_faithful_seed2025.ps1"
    locked_config = $lockedConfig
    paths = [ordered]@{
        output_root = (Resolve-RepoPath $outputRoot)
        checkpoint_root = (Resolve-RepoPath $checkpointRoot)
        log_path = (Resolve-RepoPath $logPath)
        seed_output = (Resolve-RepoPath (Join-Path $outputRoot $seedTag))
    }
    source_sha256 = $sourceHashes
    protected_files_before = $protected_files_before
    protected_files_after = $null
    runtime = [ordered]@{
        python = $pythonRuntime
        powershell = $PSVersionTable.PSVersion.ToString()
        windows = [System.Environment]::OSVersion.VersionString
    }
}
Write-JsonFile $manifestPath $manifest

$runError = $null
try {
    & $staticRunner @runnerParams *>&1 | Tee-Object -FilePath $logPath -Append
    if (-not $?) {
        throw "CBI static runner returned an unsuccessful PowerShell status."
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
    $protected_files_after = Get-HashMap $protectedFiles
    $manifest.protected_files_after = $protected_files_after
    $changedProtected = @()
    foreach ($path in $protectedFiles) {
        if ($protected_files_before[$path] -ne $protected_files_after[$path]) {
            $changedProtected += $path
        }
    }
    if ($changedProtected.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Protected main-table files changed: $($changedProtected -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    Write-JsonFile $manifestPath $manifest
}

if ($null -ne $runError) {
    throw $runError
}

Write-Host "CBI faithful seed-2025 experiment completed."
Write-Host ("Manifest: {0}" -f (Resolve-RepoPath $manifestPath))
Write-Host ("Training log: {0}" -f (Resolve-RepoPath $logPath))
