param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 15,
    [int]$Patience = 15,
    [ValidateSet("Both", "P0", "P1")]
    [string]$Case = "Both",
    [string]$OutputRootBase = "outputs\content_delta_pop5\pseudo_cold_sage_v1",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\pseudo_cold_sage_v1",
    [ValidateSet("batch_random", "batch_tail", "all_eligible")]
    [string]$PseudoColdMode = "batch_tail",
    [double]$PseudoColdRatio = 0.20,
    [int]$PseudoColdMinPop = 5,
    [double]$SageTailPopRatio = 0.002,
    [int]$SagePoolTopK = 64,
    [int]$MinFreeGpuMiB = 5000,
    [int]$PollSeconds = 300,
    [switch]$SkipGpuWait,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$outputRootBaseAbs = Join-Path $repoPath $OutputRootBase
$checkpointRootBaseAbs = Join-Path $repoPath $CheckpointRootBase
New-Item -ItemType Directory -Force -Path $outputRootBaseAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootBaseAbs | Out-Null

$queueLog = Join-Path $outputRootBaseAbs "pseudo_cold_sage_smoke_queue.log"
$planPath = Join-Path $outputRootBaseAbs "pseudo_cold_sage_smoke_plan.json"

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

function Wait-GpuReady {
    if ($SkipGpuWait) {
        Write-QueueLog "SkipGpuWait requested; starting without GPU memory gate."
        return
    }
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

function ConvertTo-CaseSuffix {
    $seedText = ($Seeds | ForEach-Object { [string]$_ }) -join "_"
    if ($Seeds.Count -eq 1) {
        return "s$seedText"
    }
    return "seeds_$seedText"
}

function New-RunnerParams {
    param(
        [string]$CaseId,
        [bool]$UseSageLite
    )
    $suffix = ConvertTo-CaseSuffix
    $caseDir = "{0}_{1}" -f $CaseId, $suffix
    return [ordered]@{
        PythonRunner = ".\py.bat"
        ScriptPath = "usim_feedback_fast3_content_delta.py"
        DataDir = "processed_data_hin_clean_pop5"
        RelationDir = "MOOCCube/relations"
        OutputRoot = (Join-Path $OutputRootBase $caseDir)
        CheckpointRoot = (Join-Path $CheckpointRootBase $caseDir)
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = $Seeds
        Epochs = $Epochs
        Patience = $Patience
        EarlyStopAverageMode = "item_macro"
        UseContentDelta = $false
        UsePseudoColdTrain = $true
        PseudoColdMode = $PseudoColdMode
        PseudoColdRatio = $PseudoColdRatio
        PseudoColdMinPop = $PseudoColdMinPop
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
        UseSageLite = $UseSageLite
        SageGateMin = 0.10
        SageGateMax = 0.60
        SagePoolTopK = $SagePoolTopK
        SageCourseTemp = 0.20
        SageOnlyColdOrTail = $UseSageLite
        SageTailPopRatio = $SageTailPopRatio
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
}

$allCases = @()
if ($Case -in @("Both", "P0")) {
    $allCases += [pscustomobject]@{
        case_id = "P0_pseudo_only"
        purpose = "Pseudo-cold train episodes without SAGE; controls for item-ID dropout/proxy-cold effect."
        runner_params = [pscustomobject](New-RunnerParams -CaseId "P0_pseudo_only" -UseSageLite $false)
    }
}
if ($Case -in @("Both", "P1")) {
    $allCases += [pscustomobject]@{
        case_id = "P1_pseudo_sage"
        purpose = "Pseudo-cold train episodes plus tail-gated SAGE-lite course-aware sampling."
        runner_params = [pscustomobject](New-RunnerParams -CaseId "P1_pseudo_sage" -UseSageLite $true)
    }
}

$plan = [pscustomobject]@{
    created_at = (Get-Date).ToString("s")
    experiment = "pseudo_cold_sage_v1_smoke"
    hypothesis = "Strict cold test items are unseen during training, so tail training items are converted into pseudo-cold episodes; SAGE-style course-aware sampling is then tested only as an additional variable over the same pseudo-cold setting."
    reference = "SAGERec: Sampling and Gating for Enhanced Long-Tail Item Recommendations, WSDM 2026; implemented here as SAGE-lite adaptation."
    baseline_refs = @(
        "outputs\content_delta_pop5\fn_mask_ab\aligned_oldcfg_mask_e60_3seed",
        "outputs\content_delta_pop5\sage_lite_v1\S5_tailratio_0p002_e60_resume_from_s4e8"
    )
    cases = $allCases
}
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding UTF8
Write-QueueLog "Wrote experiment plan: $planPath"

if ($DryRun) {
    Write-QueueLog "DryRun requested; no training command launched."
    Write-Host "DRYRUN plan=$planPath"
    exit 0
}

Wait-GpuReady

foreach ($casePlan in $allCases) {
    $runnerParams = @{}
    foreach ($prop in $casePlan.runner_params.PSObject.Properties) {
        $runnerParams[$prop.Name] = $prop.Value
    }
    Write-QueueLog ("START {0}: output={1} checkpoint={2}" -f `
        $casePlan.case_id, $runnerParams.OutputRoot, $runnerParams.CheckpointRoot)
    & ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams
    $code = $LASTEXITCODE
    Write-QueueLog ("END {0}: exit_code={1}" -f $casePlan.case_id, $code)
    if ($code -ne 0) {
        exit $code
    }
}

Write-QueueLog "DONE pseudo-cold SAGE smoke."
