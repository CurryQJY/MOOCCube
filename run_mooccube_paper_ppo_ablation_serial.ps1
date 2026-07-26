param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$BaselineRoot = "outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    [string]$OutputRootBase = "outputs\content_delta_pop5\course_ppo_ablation_e60_3seed",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$VariantList = "",
    [int[]]$WaitPid = @(),
    [switch]$NoAutoWait,
    [int]$PollSeconds = 300,
    [int]$MinFreeGpuMiB = 9000,
    [switch]$SkipGpuWait,
    [switch]$AllowMissingBaseline,
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [bool]$SaveCkpt = $true,
    [bool]$AutoResume = $true,
    [bool]$ForceFresh = $false,
    [bool]$SaveOptState = $true,
    [switch]$IncludeAlreadyRunCoreVariants,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

if ($SeedList.Trim().Length -gt 0) {
    $SeedsToRun = @(
        $SeedList -split "[,\s]+" |
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

$OutputRootBaseAbs = Resolve-RepoPath $OutputRootBase
$CheckpointRootBaseAbs = Resolve-RepoPath $CheckpointRootBase
$BaselineRootAbs = Resolve-RepoPath $BaselineRoot
$QueueLog = Join-Path $OutputRootBaseAbs "course_ppo_ablation_queue.log"
$PlanJson = Join-Path $OutputRootBaseAbs "course_ppo_ablation_plan.json"
$PlanCsv = Join-Path $OutputRootBaseAbs "course_ppo_ablation_plan.csv"

function Write-QueueLogLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
        [System.IO.File]::AppendAllText($QueueLog, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    Write-Host $line
}

function Format-ListValue {
    param([object[]]$Value)
    return ($Value | ForEach-Object { [string]$_ }) -join ","
}

function Write-Setting {
    param(
        [string]$Name,
        [object]$Value
    )
    Write-Host ("{0}={1}" -f $Name, $Value)
}

function Merge-Params {
    param(
        [hashtable]$Base,
        [hashtable]$Override
    )
    $out = @{}
    foreach ($key in $Base.Keys) {
        $out[$key] = $Base[$key]
    }
    foreach ($key in $Override.Keys) {
        $out[$key] = $Override[$key]
    }
    return $out
}

function New-Variant {
    param(
        [string]$Name,
        [string]$Label,
        [string]$Rationale,
        [hashtable]$Params
    )
    return [ordered]@{
        Name = $Name
        Label = $Label
        Rationale = $Rationale
        Params = $Params
    }
}

$alreadyRunCoreVariantNames = @("wo_simulator_t0", "wo_forced_cold_masking")

$Variants = @(
    (New-Variant `
        -Name "wo_ppo_loss" `
        -Label "w/o PPO Loss" `
        -Rationale "Keep the simulator rollout and course-aware signals but set PPO loss weight to zero, isolating the policy-gradient update." `
        -Params @{ UsimSteps = 5; PpoLossWeight = 0.0 }),
    (New-Variant `
        -Name "wo_simulator_t0" `
        -Label "w/o Simulator (T=0)" `
        -Rationale "Keep the full objective but set simulator steps to zero, so ranking uses the initial course state without iterative RL rollout." `
        -Params @{ UsimSteps = 0; PpoLossWeight = 1.0 }),
    (New-Variant `
        -Name "static_content_masked_scorer" `
        -Label "Static content+mask scorer" `
        -Rationale "Keep cold-course ID masking but remove simulator, PPO loss, and course-knowledge reward/sampling/auxiliary signals." `
        -Params @{
            UsimSteps = 0
            PpoLossWeight = 0.0
            UseCourseFeedback = $false
            UseCourseReward = $false
            UseCourseSample = $false
            UsePrereqAux = $false
        }),
    (New-Variant `
        -Name "wo_forced_cold_masking" `
        -Label "w/o Forced-cold Masking" `
        -Rationale "Keep the full PPO/simulator objective but do not force cold-course ID evidence to be masked during training." `
        -Params @{ TrainForceCold = $false; PpoLossWeight = 1.0 })
)

if ($VariantList.Trim().Length -gt 0) {
    $wanted = @(
        $VariantList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Trim() }
    )
    $Variants = @($Variants | Where-Object { $wanted -contains $_.Name })
    if ($Variants.Count -lt 1) {
        throw "No variants matched VariantList='$VariantList'"
    }
} elseif (-not $IncludeAlreadyRunCoreVariants) {
    $Variants = @($Variants | Where-Object { $alreadyRunCoreVariantNames -notcontains $_.Name })
}

$baseRunnerParams = @{
    PythonRunner = $PythonRunner
    ScriptPath = $ScriptPath
    DataDir = $DataDir
    RelationDir = $RelationDir
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
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
    PrereqGraphSource = "concept"
    CoursePrereqW = 0.08
    CoursePrereqGate = 0.20
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantConceptGate = 1.0
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    CourseSampleBeta = 0.20
    TrainForceCold = $true
    UsimSteps = 5
    PpoLossWeight = 1.0
    UseSageLite = $false
    SageUseTwoExpert = $false
    SageTwoExpertScoreFusion = $false
    UseSageAuxLoss = $false
    UseCgrcRecon = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $false
    MaskSameItemNeg = $false
    RunSampledEval = $false
    SaveCkpt = $SaveCkpt
    AutoResume = $AutoResume
    ForceFresh = $ForceFresh
    SaveOptState = $SaveOptState
}

function Get-FinalPath {
    param(
        [string]$Root,
        [int]$Seed
    )
    $tag = "strict_item_cold_balanced_thr1_seed_{0}" -f $Seed
    return Join-Path (Join-Path $Root $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Get-MissingSeeds {
    param(
        [string]$Root,
        [int[]]$Seeds
    )
    $missing = @()
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Get-FinalPath -Root $Root -Seed $seed))) {
            $missing += $seed
        }
    }
    return $missing
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

function Wait-ForGpu {
    param([string]$Reason)
    if ($SkipGpuWait) {
        Write-QueueLogLine "SkipGpuWait requested before $Reason."
        return
    }
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -eq $free) {
            Write-QueueLogLine "GPU free memory unavailable before $Reason; wait ${PollSeconds}s."
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($free -ge $MinFreeGpuMiB) {
            Write-QueueLogLine "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB before $Reason."
            return
        }
        Write-QueueLogLine "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB before $Reason; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-ActiveFast3Pids {
    $patterns = @(
        "usim_feedback_fast3_content_delta.py",
        "run_usim_feedback_fast3_content_delta_static.ps1"
    )
    $processNames = @("python.exe", "pythonw.exe", "cmd.exe", "powershell.exe", "pwsh.exe")
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $_.ProcessId -ne $PID -and
        ($processNames -contains $_.Name) -and
        $cmd -and
        ($cmd -like "*$repoPath*" -or $cmd -like "*MOOCCube*") -and
        @(($patterns | Where-Object { $cmd -like "*$_*" })).Count -gt 0
    }
    return @($procs | Select-Object -ExpandProperty ProcessId -Unique)
}

function Wait-ForPids {
    param([int[]]$Pids)
    $unique = @($Pids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($unique.Count -lt 1) {
        Write-QueueLogLine "No active experiment PID to wait for; starting paper-main PPO ablation queue."
        return
    }
    Write-QueueLogLine ("Waiting for experiment PIDs before paper-main PPO ablation queue: {0}" -f (Format-ListValue $unique))
    while ($true) {
        $alive = @()
        foreach ($pidValue in $unique) {
            if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
                $alive += $pidValue
            }
        }
        if ($alive.Count -lt 1) {
            Write-QueueLogLine "Waited experiment PIDs finished."
            return
        }
        Write-QueueLogLine ("Still running: {0}; wait {1}s." -f (Format-ListValue $alive), $PollSeconds)
        Start-Sleep -Seconds $PollSeconds
    }
}

function Assert-BaselineReady {
    $summary = Join-Path $BaselineRootAbs "fast3_static_multiseed_summary.csv"
    if (Test-Path -LiteralPath $summary) {
        Write-QueueLogLine "Baseline summary found: $summary"
        return
    }
    if ($AllowMissingBaseline) {
        Write-QueueLogLine "WARNING baseline summary missing but AllowMissingBaseline was set: $summary"
        return
    }
    throw "Missing paper-main baseline summary: $summary"
}

function Write-PlanFiles {
    $planRows = @()
    foreach ($variant in $Variants) {
        $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
        $planRows += [pscustomobject]@{
            baseline_root = $BaselineRoot
            variant = $variant.Name
            label = $variant.Label
            rationale = $variant.Rationale
            seeds = Format-ListValue $SeedsToRun
            train_force_cold = $params.TrainForceCold
            usim_steps = $params.UsimSteps
            ppo_loss_weight = $params.PpoLossWeight
            use_course_feedback = $params.UseCourseFeedback
            use_course_reward = $params.UseCourseReward
            use_course_sample = $params.UseCourseSample
            use_prereq_aux = $params.UsePrereqAux
            mask_known_pos_neg = $params.MaskKnownPosNeg
            mask_same_item_neg = $params.MaskSameItemNeg
        }
    }
    if ($DryRun) {
        return
    }
    New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
    $planRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PlanJson -Encoding UTF8
    $planRows | Export-Csv -LiteralPath $PlanCsv -NoTypeInformation -Encoding UTF8
}

function Show-DryRunVariant {
    param(
        [hashtable]$Params,
        [object]$Variant,
        [string]$OutRoot,
        [string]$CkptRoot
    )
    Write-Host ("Variant={0}" -f $Variant.Name)
    Write-Setting "Label" $Variant.Label
    Write-Setting "OutputRoot" $OutRoot
    Write-Setting "CheckpointRoot" $CkptRoot
    foreach ($key in @(
        "TrainForceCold",
        "UsimSteps",
        "PpoLossWeight",
        "UseCourseFeedback",
        "UseCourseReward",
        "UseCourseSample",
        "UsePrereqAux",
        "CourseFeedbackOnlyCold",
        "CourseSampleOnlyCold",
        "PrereqAuxOnlyCold",
        "MaskKnownPosNeg",
        "MaskSameItemNeg"
    )) {
        Write-Setting $key $Params[$key]
    }
}

Write-PlanFiles

if ($DryRun) {
    Write-Host "MOOCCube paper-main PPO ablation serial dry run"
    Write-Setting "Repo" $repoPath
    Write-Setting "StaticRunner" $StaticRunner
    Write-Setting "DataDir" $DataDir
    Write-Setting "RelationDir" $RelationDir
    Write-Setting "BaselineRoot" $BaselineRoot
    Write-Setting "OutputRootBase" $OutputRootBase
    Write-Setting "CheckpointRootBase" $CheckpointRootBase
    Write-Setting "Seeds" (Format-ListValue $SeedsToRun)
    Write-Setting "WaitPid" (Format-ListValue $WaitPid)
    Write-Setting "AutoWait" (-not [bool]$NoAutoWait)
    Write-Setting "IncludeAlreadyRunCoreVariants" ([bool]$IncludeAlreadyRunCoreVariants)
    Write-Setting "Epochs" $Epochs
    Write-Setting "Patience" $Patience
    foreach ($variant in $Variants) {
        $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
        $outRoot = Join-Path $OutputRootBase $variant.Name
        $ckptRoot = Join-Path $CheckpointRootBase $variant.Name
        Show-DryRunVariant -Params $params -Variant $variant -OutRoot $outRoot -CkptRoot $ckptRoot
    }
    return
}

New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootBaseAbs | Out-Null

Write-QueueLogLine "CONFIG MOOCCube paper-main PPO ablation queue | seeds=$(Format-ListValue $SeedsToRun) | variants=$(Format-ListValue ($Variants.Name)) | baseline=$BaselineRoot | output=$OutputRootBase"
Write-QueueLogLine "PLAN files: $PlanJson ; $PlanCsv"
Assert-BaselineReady

$waitTargets = @($WaitPid)
if (-not $NoAutoWait) {
    $autoPids = @(Get-ActiveFast3Pids)
    if ($autoPids.Count -gt 0) {
        Write-QueueLogLine ("Auto-detected active FAST3 PIDs: {0}" -f (Format-ListValue $autoPids))
        $waitTargets += $autoPids
    }
}
Wait-ForPids -Pids $waitTargets

foreach ($variant in $Variants) {
    $outRoot = Join-Path $OutputRootBase $variant.Name
    $ckptRoot = Join-Path $CheckpointRootBase $variant.Name
    $missingSeeds = @(Get-MissingSeeds -Root (Resolve-RepoPath $outRoot) -Seeds $SeedsToRun)
    if ($missingSeeds.Count -lt 1) {
        Write-QueueLogLine "SKIP $($variant.Name); all requested finals already exist. Aggregating."
        & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed for $($variant.Name)"
        }
        continue
    }

    Wait-ForGpu -Reason $variant.Name
    $runnerParams = Merge-Params -Base $baseRunnerParams -Override $variant.Params
    $runnerParams["OutputRoot"] = $outRoot
    $runnerParams["CheckpointRoot"] = $ckptRoot
    $runnerParams["Seeds"] = $missingSeeds

    Write-QueueLogLine ("START {0} | label={1} | missing_seeds={2} | ppo_loss_weight={3} | usim_steps={4}" -f $variant.Name, $variant.Label, (Format-ListValue $missingSeeds), $runnerParams.PpoLossWeight, $runnerParams.UsimSteps)
    & $StaticRunner @runnerParams
    $exitCode = $LASTEXITCODE
    Write-QueueLogLine ("END {0} | exit_code={1}" -f $variant.Name, $exitCode)
    if ($exitCode -ne 0) {
        throw "Variant failed: $($variant.Name) exit_code=$exitCode"
    }
}

Write-QueueLogLine "All requested MOOCCube paper-main PPO ablations finished."
