param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int[]]$WaitForPids = @(),
    [string]$WaitForPidCsv = "",
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [int]$PollSeconds = 120,
    [string]$VariantName = "combo_beta0p15_reward2p00_horizon3",
    [string]$OutputRootBase = "outputs\significance_per_item_exports\mooccube\ckg_rl_true_true_hparam_combo",
    [string]$CheckpointRootBase = "checkpoints\significance_per_item_exports\mooccube\ckg_rl_true_true_hparam_combo",
    [string]$PythonRunner = ".\py.bat",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

if ($WaitForPidCsv.Trim().Length -gt 0) {
    $WaitForPids = @(
        $WaitForPidCsv -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { [int]$_.Trim() }
    )
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $repoPath $PathValue)
}

$outputRootBaseAbs = Resolve-RepoPath $OutputRootBase
$checkpointRootBaseAbs = Resolve-RepoPath $CheckpointRootBase
$outRoot = Join-Path $outputRootBaseAbs $VariantName
$ckptRoot = Join-Path $checkpointRootBaseAbs $VariantName
$queueLog = Join-Path $outputRootBaseAbs "true_true_hparam_combo_queue.log"
$staticRunner = Join-Path $repoPath "run_usim_feedback_fast3_content_delta_static.ps1"

function Write-QueueLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $outputRootBaseAbs | Out-Null
        [System.IO.File]::AppendAllText($queueLog, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    Write-Host $line
}

function Wait-ForPidList {
    param([int[]]$Pids)
    if ($DryRun -or $Pids.Count -lt 1) {
        return
    }
    while ($true) {
        $alive = @()
        foreach ($pidValue in $Pids) {
            $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($process) {
                $alive += $process
            }
        }
        if ($alive.Count -lt 1) {
            break
        }
        Write-QueueLine ("Waiting for true-true hparam grid PIDs: {0}" -f (($alive | ForEach-Object { $_.Id }) -join ","))
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-FinalPath {
    param([int]$Seed)
    return Join-Path (Join-Path $outRoot "strict_item_cold_balanced_thr1_seed_$Seed") "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Get-MissingSeeds {
    $missing = @()
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Get-FinalPath -Seed $seed))) {
            $missing += $seed
        }
    }
    return $missing
}

Write-QueueLine "BOOT MOOCCube true-true hparam combo queue | variant=$VariantName | seeds=$($Seeds -join ',')"
Write-QueueLine "Combo: beta=0.15, reward_scale=2.00, horizon T=3; selection remains validation cold item-macro NDCG@10."
Wait-ForPidList -Pids $WaitForPids

$missingSeeds = @(Get-MissingSeeds)
if ($missingSeeds.Count -lt 1) {
    Write-QueueLine "SKIP $VariantName; all requested final files already exist."
} else {
    Write-QueueLine "START $VariantName missing_seeds=$($missingSeeds -join ',')"
    if (-not $DryRun) {
        $runnerParams = @{
            PythonRunner = $PythonRunner
            Protocol = "strict_item_cold_balanced"
            ColdThresholds = @(1)
            Seeds = $missingSeeds
            Epochs = $Epochs
            Patience = $Patience
            EarlyStopAverageMode = "item_macro"
            EarlyStopScoreMode = "cold_only"
            UseContentDelta = $false
            UsePseudoColdTrain = $false
            UsePaac = $false
            UseCourseFeedback = $true
            UseCourseReward = $true
            UseCourseSample = $true
            UsePrereqAux = $true
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
            CoursePrereqW = 0.16
            CoursePrereqGate = 0.20
            CourseConceptW = 0.08
            CourseDiffW = 0.06
            CourseRedundantW = 0.04
            CourseRedundantConceptGate = 1.0
            CourseRedundantMode = "concept"
            CourseTermNorm = "none"
            CourseSampleBeta = 0.15
            TrainForceCold = $true
            UsimSteps = 3
            PpoLossWeight = 1.0
            RolloutPolicy = "ppo"
            MaskKnownPosNeg = $true
            MaskSameItemNeg = $true
            RunSampledEval = $false
            OutputRoot = $outRoot
            CheckpointRoot = $ckptRoot
            SaveCkpt = $true
            AutoResume = $true
            ForceFresh = $false
            SaveOptState = $true
            SkipAggregate = $true
        }
        & $staticRunner @runnerParams
        if ($LASTEXITCODE -ne 0) {
            throw "$VariantName failed with exit code $LASTEXITCODE"
        }
    }
    Write-QueueLine "END $VariantName"
}

if (-not $DryRun) {
    Write-QueueLine "Aggregate $VariantName"
    & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed for $VariantName"
    }
}

Write-QueueLine "DONE MOOCCube true-true hparam combo queue"
