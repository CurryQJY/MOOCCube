param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$OutputRootBase = "outputs\content_delta_pop5\sage_lite_v1\S13_course_signal_ablation_e60_3seed",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\sage_lite_v1\S13_course_signal_ablation_e60_3seed",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$VariantList = "",
    [int[]]$WaitPid = @(),
    [switch]$NoAutoWait,
    [int]$PollSeconds = 300,
    [int]$MinFreeGpuMiB = 9000,
    [switch]$SkipGpuWait,
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

$OutputRootBaseAbs = Join-Path $repoPath $OutputRootBase
$CheckpointRootBaseAbs = Join-Path $repoPath $CheckpointRootBase
$QueueLog = Join-Path $OutputRootBaseAbs "course_signal_ablation_queue.log"
$PlanJson = Join-Path $OutputRootBaseAbs "course_signal_ablation_plan.json"
$PlanCsv = Join-Path $OutputRootBaseAbs "course_signal_ablation_plan.csv"

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
    (New-Variant `
        -Name "wo_concept_match" `
        -Label "w/o Concept Match" `
        -Rationale "Drop the positive course-concept match reward and sampling term." `
        -Params @{ CourseConceptW = 0.0 }),
    (New-Variant `
        -Name "wo_prereq_signal" `
        -Label "w/o Prerequisite Signal" `
        -Rationale "Drop prerequisite penalty and neutralize the prerequisite safety gate while keeping the auxiliary row separate." `
        -Params @{ CoursePrereqW = 0.0; CoursePrereqGate = 1.0 }),
    (New-Variant `
        -Name "wo_difficulty_signal" `
        -Label "w/o Difficulty Readiness" `
        -Rationale "Drop the user-readiness versus item-difficulty penalty." `
        -Params @{ CourseDiffW = 0.0 }),
    (New-Variant `
        -Name "wo_redundancy_signal" `
        -Label "w/o Redundancy Control" `
        -Rationale "Drop explicit redundancy penalty and stop redundancy from suppressing concept bonus." `
        -Params @{ CourseRedundantW = 0.0; CourseRedundantConceptGate = 0.0 }),
    (New-Variant `
        -Name "wo_course_candidate" `
        -Label "w/o Course-aware Candidate Selection" `
        -Rationale "Keep course reward terms but remove course-fit candidate ordering." `
        -Params @{ UseCourseSample = $false; CourseSampleBeta = 0.0 }),
    (New-Variant `
        -Name "wo_course_reward" `
        -Label "w/o Course-aware Reward" `
        -Rationale "Keep course-aware candidate selection but remove course terms from PPO reward." `
        -Params @{ UseCourseReward = $false }),
    (New-Variant `
        -Name "wo_prereq_aux" `
        -Label "w/o Prerequisite Auxiliary Loss" `
        -Rationale "Keep reward and sampling signals but remove the training-time prerequisite auxiliary loss." `
        -Params @{ UsePrereqAux = $false }),
    (New-Variant `
        -Name "wo_all_course_signals" `
        -Label "w/o All Course Signals" `
        -Rationale "Remove course-aware reward, sampling, prerequisite auxiliary loss, and all course-term weights." `
        -Params @{
            UseCourseFeedback = $false
            UseCourseReward = $false
            UseCourseSample = $false
            UsePrereqAux = $false
            CoursePrereqW = 0.0
            CoursePrereqGate = 1.0
            CourseConceptW = 0.0
            CourseDiffW = 0.0
            CourseRedundantW = 0.0
            CourseRedundantConceptGate = 0.0
            CourseSampleBeta = 0.0
        })
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
    UseSageLite = $true
    SageOnlyColdOrTail = $false
    SageTailPopRatio = 0.002
    SageUseTwoExpert = $false
    SageTwoExpertScoreFusion = $true
    SageGateMode = "bucket_mlp"
    SageGateBuckets = 20
    SageGateHidden = 32
    SageGateBucketStrategy = "log"
    SageGateMin = 0.10
    SageGateMax = 0.60
    SagePoolTopK = 64
    SageCourseTemp = 0.20
    UseSageAuxLoss = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
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
        "launch_mooccube_sage_twoexpert_scorefusion_logbucket_tail0p002_seed2025.ps1",
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
        Write-QueueLogLine "No active experiment PID to wait for; starting ablation queue."
        return
    }
    Write-QueueLogLine ("Waiting for experiment PIDs before ablation queue: {0}" -f (Format-ListValue $unique))
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

function Write-PlanFiles {
    $planRows = @()
    foreach ($variant in $Variants) {
        $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
        $planRows += [pscustomobject]@{
            variant = $variant.Name
            label = $variant.Label
            rationale = $variant.Rationale
            seeds = Format-ListValue $SeedsToRun
            course_prereq_w = $params.CoursePrereqW
            course_prereq_gate = $params.CoursePrereqGate
            course_concept_w = $params.CourseConceptW
            course_diff_w = $params.CourseDiffW
            course_redundant_w = $params.CourseRedundantW
            course_redundant_concept_gate = $params.CourseRedundantConceptGate
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
    Write-Setting "Label" $Variant.Label
    Write-Setting "OutputRoot" $OutRoot
    Write-Setting "CheckpointRoot" $CkptRoot
    foreach ($key in @(
        "UseCourseFeedback",
        "UseCourseReward",
        "UseCourseSample",
        "UsePrereqAux",
        "CoursePrereqW",
        "CoursePrereqGate",
        "CourseConceptW",
        "CourseDiffW",
        "CourseRedundantW",
        "CourseRedundantConceptGate",
        "CourseSampleBeta",
        "UseSageLite",
        "SageGateMode",
        "SageGateBucketStrategy",
        "SageTailPopRatio",
        "MaskKnownPosNeg",
        "MaskSameItemNeg"
    )) {
        Write-Setting $key $Params[$key]
    }
}

Write-PlanFiles

if ($DryRun) {
    Write-Host "MOOCCube SAGE course-signal ablation serial dry run"
    Write-Setting "Repo" $repoPath
    Write-Setting "StaticRunner" $StaticRunner
    Write-Setting "DataDir" $DataDir
    Write-Setting "RelationDir" $RelationDir
    Write-Setting "OutputRootBase" $OutputRootBase
    Write-Setting "CheckpointRootBase" $CheckpointRootBase
    Write-Setting "Seeds" (Format-ListValue $SeedsToRun)
    Write-Setting "WaitPid" (Format-ListValue $WaitPid)
    Write-Setting "AutoWait" (-not [bool]$NoAutoWait)
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
Write-QueueLogLine "CONFIG MOOCCube SAGE course-signal ablation queue | seeds=$(Format-ListValue $SeedsToRun) | variants=$(Format-ListValue ($Variants.Name)) | output=$OutputRootBase"
Write-QueueLogLine "PLAN files: $PlanJson ; $PlanCsv"

$waitTargets = @($WaitPid)
if (-not $NoAutoWait) {
    $autoPids = @(Get-ActiveFast3Pids)
    if ($autoPids.Count -gt 0) {
        Write-QueueLogLine ("Auto-detected active FAST3/SAGE PIDs: {0}" -f (Format-ListValue $autoPids))
        $waitTargets += $autoPids
    }
}
Wait-ForPids -Pids $waitTargets

foreach ($variant in $Variants) {
    $outRoot = Join-Path $OutputRootBase $variant.Name
    $ckptRoot = Join-Path $CheckpointRootBase $variant.Name
    $missingSeeds = @(Get-MissingSeeds -Root (Join-Path $repoPath $outRoot) -Seeds $SeedsToRun)
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

    Write-QueueLogLine ("START {0} | label={1} | missing_seeds={2}" -f $variant.Name, $variant.Label, (Format-ListValue $missingSeeds))
    & $StaticRunner @runnerParams
    $exitCode = $LASTEXITCODE
    Write-QueueLogLine ("END {0} | exit_code={1}" -f $variant.Name, $exitCode)
    if ($exitCode -ne 0) {
        throw "Variant failed: $($variant.Name) exit_code=$exitCode"
    }
}

Write-QueueLogLine "All requested MOOCCube SAGE course-signal ablations finished."
