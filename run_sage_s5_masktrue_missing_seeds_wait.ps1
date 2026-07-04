param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$OutputRoot = "outputs\content_delta_pop5\sage_lite_v1\S5_tailratio_0p002_e60_resume_from_s4e8",
    [string]$CheckpointRoot = "checkpoints\content_delta_pop5\sage_lite_v1\S5_tailratio_0p002_e60_resume_from_s4e8",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$MinFreeGpuMiB = 5000,
    [int]$PollSeconds = 300,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$outputRootAbs = Join-Path $repoPath $OutputRoot
$checkpointRootAbs = Join-Path $repoPath $CheckpointRoot
$queueLog = Join-Path $outputRootAbs "queue_masktrue_sage_3seed.log"
New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            return [int](([string]$raw).Trim().Split("`n")[0].Trim())
        }
    } catch {
        return $null
    }
    return $null
}

function Test-CompletedSageMaskTrue {
    param([int]$Seed)
    $dir = Join-Path $outputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
    $summary = Join-Path $dir "mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv"
    $manifest = Join-Path $dir "static_protocol_manifest.json"
    if (-not (Test-Path -LiteralPath $summary) -or -not (Test-Path -LiteralPath $manifest)) {
        return $false
    }
    try {
        $json = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
        return (
            $json.model_config.n_epochs -eq 60 -and
            $json.model_config.mask_known_pos_neg -eq $true -and
            $json.model_config.mask_same_item_neg -eq $true -and
            $json.model_config.use_sage_lite -eq $true -and
            $json.model_config.sage_only_cold_or_tail -eq $true -and
            [math]::Abs([double]$json.model_config.sage_tail_pop_ratio - 0.002) -lt 1e-12 -and
            $json.model_config.use_sage_aux_loss -eq $false
        )
    } catch {
        return $false
    }
}

function Wait-GpuReady {
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -eq $free) {
            Write-QueueLog "GPU free memory unavailable; wait ${PollSeconds}s."
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($free -ge $MinFreeGpuMiB) {
            Write-QueueLog "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB; continue."
            return
        }
        Write-QueueLog "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

$missingSeeds = @()
foreach ($seed in $Seeds) {
    if (Test-CompletedSageMaskTrue -Seed $seed) {
        Write-QueueLog "SKIP seed=$seed; matching mask=true + SAGE summary already exists."
    } else {
        $missingSeeds += $seed
        Write-QueueLog "QUEUE seed=$seed; missing or config mismatch."
    }
}

if ($missingSeeds.Count -eq 0) {
    Write-QueueLog "Nothing to run. All requested seeds are complete."
    exit 0
}

if ($DryRun) {
    Write-QueueLog ("DRYRUN missing seeds: {0}" -f ($missingSeeds -join ","))
    exit 0
}

Wait-GpuReady

$runnerParams = @{
    PythonRunner = ".\py.bat"
    ScriptPath = "usim_feedback_fast3_content_delta.py"
    DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"
    OutputRoot = $OutputRoot
    CheckpointRoot = $CheckpointRoot
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = $missingSeeds
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
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
    UseSageLite = $true
    SageGateMin = 0.10
    SageGateMax = 0.60
    SagePoolTopK = 64
    SageCourseTemp = 0.20
    SageOnlyColdOrTail = $true
    SageTailPopRatio = 0.002
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

Write-QueueLog ("START missing seeds: {0}" -f ($missingSeeds -join ","))
& ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams
$code = $LASTEXITCODE
Write-QueueLog "Static runner exit code=$code"
if ($code -ne 0) {
    exit $code
}

foreach ($seed in $missingSeeds) {
    if (-not (Test-CompletedSageMaskTrue -Seed $seed)) {
        Write-QueueLog "ERROR seed=$seed did not produce matching summary."
        exit 2
    }
}

Write-QueueLog "DONE mask=true + SAGE requested seeds."
