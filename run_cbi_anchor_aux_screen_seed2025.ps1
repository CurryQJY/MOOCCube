param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$staticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1"
$outputRoot = "outputs\cbi_anchor_aux_screen_seed2025"
$checkpointRoot = "checkpoints\cbi_anchor_aux_screen_seed2025"
$logRoot = "background_logs\cbi_anchor_aux_screen_seed2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "screen.log"
$arms = @(
    [ordered]@{ Name = "aux_0p0"; AuxWeight = 0.0 },
    [ordered]@{ Name = "aux_0p1"; AuxWeight = 0.1 },
    [ordered]@{ Name = "aux_0p3"; AuxWeight = 0.3 }
)

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

function New-RunnerParams($Arm) {
    $armOutput = Join-Path $outputRoot $Arm.Name
    $armCheckpoint = Join-Path $checkpointRoot $Arm.Name
    return [ordered]@{
        PythonRunner = ".\py.bat"
        ScriptPath = "run_cbi_anchor_sim_seed2025.py"
        DataDir = "processed_data_hin_clean_pop5"
        RelationDir = "MOOCCube/relations"
        OutputRoot = $armOutput
        CheckpointRoot = $armCheckpoint
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @(2025)
        Epochs = 30
        Patience = 6
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
        AuxWeight = [double]$Arm.AuxWeight
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
}

$lockedConfig = [ordered]@{
    experiment = "cbi_anchor_aux_screen_seed2025"
    execution = "serial"
    seed = 2025
    epochs = 30
    patience = 6
    selection = "validation_cold_n10_then_cold_r10"
    arms = $arms
}

if ($DryRun) {
    Write-Host "DRY_RUN CBI anchor auxiliary-ID screen"
    Write-Host ("Repo={0}" -f $repoPath)
    Write-Host ("Arms={0}" -f (($arms | ForEach-Object { $_.AuxWeight }) -join ","))
    Write-Host "Seed=2025 Epochs=30 Patience=6 Execution=serial"
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

if (Test-Path -LiteralPath $manifestPath) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($existing.status -eq "completed") {
        throw "CBI anchor auxiliary screen is already completed."
    }
}

$sourceFiles = @(
    "run_cbi_anchor_aux_screen_seed2025.ps1",
    "run_cbi_anchor_sim_seed2025.py",
    "cbi_anchor_sim.py",
    "cbi_trust_sim.py",
    "run_usim_feedback_fast3_content_delta_static.ps1"
)
$protectedBefore = Get-HashMap $protectedFiles
$sourceHashes = Get-HashMap $sourceFiles
$startedAt = (Get-Date).ToUniversalTime()
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$manifest = [ordered]@{
    schema_version = 1
    experiment = "cbi_anchor_aux_screen_seed2025"
    status = "running"
    current_arm = $null
    started_at_utc = $startedAt.ToString("o")
    completed_at_utc = $null
    elapsed_seconds = $null
    exit_code = $null
    error = $null
    git_commit = (git rev-parse HEAD).Trim()
    locked_config = $lockedConfig
    source_sha256 = $sourceHashes
    protected_files_before = $protectedBefore
    protected_files_after = $null
}
Write-JsonFile $manifestPath $manifest

$runError = $null
try {
    foreach ($arm in $arms) {
        $manifest.current_arm = $arm.Name
        Write-JsonFile $manifestPath $manifest
        $params = New-RunnerParams $arm
        $armFinal = Join-Path $params.OutputRoot "strict_item_cold_balanced_thr1_seed_2025\final_report_usim_feedback_fast3_content_delta_static.csv"
        if (Test-Path -LiteralPath $armFinal) {
            Write-Host ("Skipping completed arm {0}" -f $arm.Name)
            continue
        }
        "===== AUX SCREEN arm=$($arm.Name) AuxWeight=$($arm.AuxWeight) =====" | Tee-Object -FilePath $logPath -Append
        & $staticRunner @params *>&1 | Tee-Object -FilePath $logPath -Append
        if (-not $?) {
            throw "Auxiliary screen arm failed: $($arm.Name)"
        }
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
    foreach ($path in $protectedFiles) {
        if ($protectedBefore[$path] -ne $protectedAfter[$path]) {
            $manifest.status = "failed"
            $manifest.exit_code = 1
            $manifest.error = "Protected shared code changed: $path"
            $runError = [InvalidOperationException]::new($manifest.error)
            break
        }
    }
    $manifest.current_arm = $null
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    Write-JsonFile $manifestPath $manifest
}

if ($null -ne $runError) {
    throw $runError
}

Write-Host "CBI anchor auxiliary-ID screen completed."
