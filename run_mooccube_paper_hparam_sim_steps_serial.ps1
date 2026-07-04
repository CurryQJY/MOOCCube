param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$BaselineRoot = "outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    [string]$OutputRootBase = "outputs\content_delta_pop5\course_hparam_sim_steps_e60_3seed",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\course_hparam_sim_steps_e60_3seed",
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
$SeedsToRun = @($SeedsToRun | Sort-Object -Unique)
if ($SeedsToRun.Count -lt 1) {
    throw "No seeds requested."
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
$QueueLog = Join-Path $OutputRootBaseAbs "course_hparam_sim_steps_queue.log"
$PlanJson = Join-Path $OutputRootBaseAbs "course_hparam_sim_steps_plan.json"
$PlanCsv = Join-Path $OutputRootBaseAbs "course_hparam_sim_steps_plan.csv"

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
    param([object]$Value)
    $flat = @()
    foreach ($item in @($Value)) {
        if ($null -ne $item -and $item -is [System.Array] -and -not ($item -is [string])) {
            foreach ($inner in $item) {
                $flat += $inner
            }
        } else {
            $flat += $item
        }
    }
    return (($flat | ForEach-Object { [string]$_ }) -join ",")
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

$Variants = @(
    (New-Variant -Name "sim_steps_1" -Label "Simulation steps T=1" -Rationale "Probe a short simulator horizon below the paper-main T=5 setting." -Params @{ UsimSteps = 1 }),
    (New-Variant -Name "sim_steps_3" -Label "Simulation steps T=3" -Rationale "Probe a moderate simulator horizon below the paper-main T=5 setting." -Params @{ UsimSteps = 3 }),
    (New-Variant -Name "sim_steps_7" -Label "Simulation steps T=7" -Rationale "Probe a moderately longer simulator horizon above the paper-main T=5 setting." -Params @{ UsimSteps = 7 }),
    (New-Variant -Name "sim_steps_10" -Label "Simulation steps T=10" -Rationale "Probe a long simulator horizon to test over-simulation sensitivity." -Params @{ UsimSteps = 10 })
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
    TrainForceCold = $true
    UsimSteps = 5
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

function Get-MissingSeeds {
    param(
        [string]$Root,
        [int[]]$Seeds
    )
    $missing = @()
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Get-FinalPath -Root $Root -SeedValue $seed))) {
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
        Write-QueueLogLine "No active experiment PID to wait for; starting simulation-step hyperparam queue."
        return
    }
    Write-QueueLogLine ("Waiting for experiment PIDs before simulation-step hyperparam queue: {0}" -f (Format-ListValue $unique))
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
            seeds = Format-ListValue $SeedsToRun
            variant = $variant.Name
            label = $variant.Label
            rationale = $variant.Rationale
            usim_steps = $params.UsimSteps
            train_force_cold = $params.TrainForceCold
            course_sample_beta = $params.CourseSampleBeta
            course_prereq_w = $params.CoursePrereqW
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
        [string]$CkptRoot,
        [int[]]$MissingSeeds
    )
    Write-Host ("Variant={0}" -f $Variant.Name)
    Write-Setting "Label" $Variant.Label
    Write-Setting "OutputRoot" $OutRoot
    Write-Setting "CheckpointRoot" $CkptRoot
    Write-Setting "MissingSeeds" (Format-ListValue $MissingSeeds)
    foreach ($key in @(
        "TrainForceCold",
        "UsimSteps",
        "UseCourseFeedback",
        "UseCourseReward",
        "UseCourseSample",
        "UsePrereqAux",
        "CourseFeedbackOnlyCold",
        "CourseSampleOnlyCold",
        "PrereqAuxOnlyCold",
        "CourseSampleBeta",
        "CoursePrereqW",
        "CourseConceptW",
        "CourseDiffW",
        "CourseRedundantW",
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
    Write-Host "MOOCCube paper-main simulation-step hyperparam dry run"
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
    Write-Setting "RunCount" $Variants.Count
    Write-Setting "Epochs" $Epochs
    Write-Setting "Patience" $Patience
    foreach ($variant in $Variants) {
        $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
        $outRoot = Join-Path $OutputRootBase $variant.Name
        $ckptRoot = Join-Path $CheckpointRootBase $variant.Name
        $missingSeeds = @(Get-MissingSeeds -Root (Resolve-RepoPath $outRoot) -Seeds $SeedsToRun)
        Show-DryRunVariant -Params $params -Variant $variant -OutRoot $outRoot -CkptRoot $ckptRoot -MissingSeeds $missingSeeds
    }
    return
}

New-Item -ItemType Directory -Force -Path $OutputRootBaseAbs | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootBaseAbs | Out-Null

Write-QueueLogLine "CONFIG MOOCCube paper-main simulation-step hyperparam queue | seeds=$(Format-ListValue $SeedsToRun) | variants=$(Format-ListValue ($Variants.Name)) | baseline=$BaselineRoot | output=$OutputRootBase"
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
    $outRootAbs = Resolve-RepoPath $outRoot
    $missingSeeds = @(Get-MissingSeeds -Root $outRootAbs -Seeds $SeedsToRun)
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

    Write-QueueLogLine ("START {0} | label={1} | usim_steps={2} | missing_seeds={3}" -f $variant.Name, $variant.Label, $runnerParams.UsimSteps, (Format-ListValue $missingSeeds))
    & $StaticRunner @runnerParams
    $exitCode = $LASTEXITCODE
    Write-QueueLogLine ("END {0} | exit_code={1}" -f $variant.Name, $exitCode)
    if ($exitCode -ne 0) {
        throw "Variant failed: $($variant.Name) exit_code=$exitCode"
    }
}

Write-QueueLogLine "All requested MOOCCube paper-main simulation-step hyperparam runs finished."
