param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$BaselineRoot = "outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    [string]$PriorHparamRoot = "outputs\content_delta_pop5\course_hparam_sensitivity_e60_3seed",
    [string]$OutputRootBase = "outputs\content_delta_pop5\course_hparam_wide_seed2025",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\course_hparam_wide_seed2025",
    [int]$Seed = 2025,
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
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

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
$PriorHparamRootAbs = Resolve-RepoPath $PriorHparamRoot
$QueueLog = Join-Path $OutputRootBaseAbs "course_hparam_wide_seed2025_queue.log"
$PlanJson = Join-Path $OutputRootBaseAbs "course_hparam_wide_seed2025_plan.json"
$PlanCsv = Join-Path $OutputRootBaseAbs "course_hparam_wide_seed2025_plan.csv"

function Write-QueueLogLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
        $payload = $line + [Environment]::NewLine
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            try {
                [System.IO.File]::AppendAllText($QueueLog, $payload, [System.Text.Encoding]::UTF8)
                break
            } catch {
                if ($attempt -eq 5) {
                    throw
                }
                Start-Sleep -Milliseconds (200 * $attempt)
            }
        }
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
        [string]$Family,
        [string]$Label,
        [string]$Rationale,
        [hashtable]$Params
    )
    return [ordered]@{
        Name = $Name
        Family = $Family
        Label = $Label
        Rationale = $Rationale
        Params = $Params
    }
}

$Variants = @(
    (New-Variant -Name "sample_beta_0p00" -Family "sample_beta" -Label "Course sample beta 0.00" -Rationale "Disable the course-aware sampling strength while keeping the component path active." -Params @{ CourseSampleBeta = 0.00 }),
    (New-Variant -Name "sample_beta_0p05" -Family "sample_beta" -Label "Course sample beta 0.05" -Rationale "Probe whether a smaller sampling strength than 0.10 is better for cold-start ranking." -Params @{ CourseSampleBeta = 0.05 }),
    (New-Variant -Name "sample_beta_0p15" -Family "sample_beta" -Label "Course sample beta 0.15" -Rationale "Probe the interval between the current best 0.10 and paper-main 0.20." -Params @{ CourseSampleBeta = 0.15 }),
    (New-Variant -Name "sample_beta_0p25" -Family "sample_beta" -Label "Course sample beta 0.25" -Rationale "Probe the interval between paper-main 0.20 and the prior 0.30 point." -Params @{ CourseSampleBeta = 0.25 }),
    (New-Variant -Name "sample_beta_0p40" -Family "sample_beta" -Label "Course sample beta 0.40" -Rationale "Probe stronger course-aware sampling beyond the prior 0.30 point." -Params @{ CourseSampleBeta = 0.40 }),
    (New-Variant -Name "sample_beta_0p50" -Family "sample_beta" -Label "Course sample beta 0.50" -Rationale "Probe a high course-aware sampling strength for coarse single-seed screening." -Params @{ CourseSampleBeta = 0.50 }),

    (New-Variant -Name "reward_scale_0p00" -Family "reward_scale" -Label "Course reward weights x0.00" -Rationale "Set all reward-term weights to zero while leaving sampling and auxiliary loss fixed." -Params @{ CoursePrereqW = 0.00; CourseConceptW = 0.00; CourseDiffW = 0.00; CourseRedundantW = 0.00 }),
    (New-Variant -Name "reward_scale_0p25" -Family "reward_scale" -Label "Course reward weights x0.25" -Rationale "Quarter all course reward-term weights for a wider reward-scale curve." -Params @{ CoursePrereqW = 0.02; CourseConceptW = 0.01; CourseDiffW = 0.0075; CourseRedundantW = 0.005 }),
    (New-Variant -Name "reward_scale_0p75" -Family "reward_scale" -Label "Course reward weights x0.75" -Rationale "Probe a moderate reduction between x0.5 and the default x1.0." -Params @{ CoursePrereqW = 0.06; CourseConceptW = 0.03; CourseDiffW = 0.0225; CourseRedundantW = 0.015 }),
    (New-Variant -Name "reward_scale_1p25" -Family "reward_scale" -Label "Course reward weights x1.25" -Rationale "Probe a moderate increase between the default x1.0 and x1.5." -Params @{ CoursePrereqW = 0.10; CourseConceptW = 0.05; CourseDiffW = 0.0375; CourseRedundantW = 0.025 }),
    (New-Variant -Name "reward_scale_2p00" -Family "reward_scale" -Label "Course reward weights x2.00" -Rationale "Double all course reward-term weights to test high reward-shaping strength." -Params @{ CoursePrereqW = 0.16; CourseConceptW = 0.08; CourseDiffW = 0.06; CourseRedundantW = 0.04 }),

    (New-Variant -Name "prereq_gate_0p00" -Family "prereq_gate" -Label "Prerequisite gate 0.00" -Rationale "Make prerequisite reward shaping permissive for a lower-bound gate check." -Params @{ CoursePrereqGate = 0.00 }),
    (New-Variant -Name "prereq_gate_0p10" -Family "prereq_gate" -Label "Prerequisite gate 0.10" -Rationale "Probe a tighter gate than the paper-main 0.20 value." -Params @{ CoursePrereqGate = 0.10 }),
    (New-Variant -Name "prereq_gate_0p50" -Family "prereq_gate" -Label "Prerequisite gate 0.50" -Rationale "Probe a relaxed gate beyond the prior 0.35 point." -Params @{ CoursePrereqGate = 0.50 }),
    (New-Variant -Name "prereq_gate_0p70" -Family "prereq_gate" -Label "Prerequisite gate 0.70" -Rationale "Probe a high gate for coarse single-seed screening." -Params @{ CoursePrereqGate = 0.70 }),

    (New-Variant -Name "term_norm_ema" -Family "term_norm" -Label "EMA-normalized course terms" -Rationale "Switch course-term normalization from none to EMA after the batch-normalization point was checked." -Params @{ CourseTermNorm = "ema" })
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
        [int]$SeedValue
    )
    $tag = "strict_item_cold_balanced_thr1_seed_{0}" -f $SeedValue
    return Join-Path (Join-Path $Root $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
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
    try {
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
    } catch {
        Write-QueueLogLine "WARNING active FAST3 PID detection failed: $($_.Exception.Message)"
        return @()
    }
}

function Wait-ForPids {
    param([int[]]$Pids)
    $unique = @($Pids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($unique.Count -lt 1) {
        Write-QueueLogLine "No active experiment PID to wait for; starting seed2025 wide hyperparam queue."
        return
    }
    Write-QueueLogLine ("Waiting for experiment PIDs before seed2025 wide hyperparam queue: {0}" -f (Format-ListValue $unique))
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
            prior_hparam_root = $PriorHparamRoot
            seed = $Seed
            variant = $variant.Name
            family = $variant.Family
            label = $variant.Label
            rationale = $variant.Rationale
            course_sample_beta = $params.CourseSampleBeta
            course_term_norm = $params.CourseTermNorm
            course_prereq_w = $params.CoursePrereqW
            course_prereq_gate = $params.CoursePrereqGate
            course_concept_w = $params.CourseConceptW
            course_diff_w = $params.CourseDiffW
            course_redundant_w = $params.CourseRedundantW
            use_course_feedback = $params.UseCourseFeedback
            use_course_reward = $params.UseCourseReward
            use_course_sample = $params.UseCourseSample
            use_prereq_aux = $params.UsePrereqAux
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
    Write-Setting "Family" $Variant.Family
    Write-Setting "Label" $Variant.Label
    Write-Setting "OutputRoot" $OutRoot
    Write-Setting "CheckpointRoot" $CkptRoot
    foreach ($key in @(
        "UseCourseFeedback",
        "UseCourseReward",
        "UseCourseSample",
        "UsePrereqAux",
        "CourseFeedbackOnlyCold",
        "CourseSampleOnlyCold",
        "PrereqAuxOnlyCold",
        "CoursePrereqW",
        "CoursePrereqGate",
        "CourseConceptW",
        "CourseDiffW",
        "CourseRedundantW",
        "CourseRedundantConceptGate",
        "CourseSampleBeta",
        "CourseTermNorm",
        "UseSageLite",
        "SageTwoExpertScoreFusion",
        "UseSageAuxLoss",
        "MaskKnownPosNeg",
        "MaskSameItemNeg"
    )) {
        Write-Setting $key $Params[$key]
    }
}

Write-PlanFiles

if ($DryRun) {
    Write-Host "MOOCCube paper-main seed2025 wide hyperparam serial dry run"
    Write-Setting "Repo" $repoPath
    Write-Setting "StaticRunner" $StaticRunner
    Write-Setting "DataDir" $DataDir
    Write-Setting "RelationDir" $RelationDir
    Write-Setting "BaselineRoot" $BaselineRoot
    Write-Setting "PriorHparamRoot" $PriorHparamRoot
    Write-Setting "OutputRootBase" $OutputRootBase
    Write-Setting "CheckpointRootBase" $CheckpointRootBase
    Write-Setting "Seed" $Seed
    Write-Setting "WaitPid" (Format-ListValue $WaitPid)
    Write-Setting "AutoWait" (-not [bool]$NoAutoWait)
    Write-Setting "RunCount" $Variants.Count
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

Write-QueueLogLine "CONFIG MOOCCube paper-main seed2025 wide hyperparam queue | seed=$Seed | variants=$(Format-ListValue ($Variants.Name)) | baseline=$BaselineRoot | prior=$PriorHparamRoot | output=$OutputRootBase"
Write-QueueLogLine "PLAN files: $PlanJson ; $PlanCsv"
Assert-BaselineReady
if (Test-Path -LiteralPath $PriorHparamRootAbs) {
    Write-QueueLogLine "Prior hparam root found for comparison references: $PriorHparamRootAbs"
} else {
    Write-QueueLogLine "WARNING prior hparam root missing; running new wide-grid points only: $PriorHparamRootAbs"
}

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
    $finalPath = Get-FinalPath -Root (Resolve-RepoPath $outRoot) -SeedValue $Seed
    if (Test-Path -LiteralPath $finalPath) {
        Write-QueueLogLine "SKIP $($variant.Name); final already exists for seed $Seed. Aggregating."
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
    $runnerParams["Seeds"] = @($Seed)

    Write-QueueLogLine ("START {0} | family={1} | label={2} | seed={3}" -f $variant.Name, $variant.Family, $variant.Label, $Seed)
    & $StaticRunner @runnerParams
    $exitCode = $LASTEXITCODE
    Write-QueueLogLine ("END {0} | exit_code={1}" -f $variant.Name, $exitCode)
    if ($exitCode -ne 0) {
        throw "Variant failed: $($variant.Name) exit_code=$exitCode"
    }
}

Write-QueueLogLine "All requested MOOCCube paper-main seed2025 wide hyperparam runs finished."
