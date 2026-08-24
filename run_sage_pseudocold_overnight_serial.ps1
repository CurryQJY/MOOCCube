param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [ValidateSet("Both", "P0", "P1", "P2")]
    [string]$Case = "Both",
    [string]$OutputRootBase = "outputs\content_delta_pop5\pseudo_cold_sage_v1",
    [string]$CheckpointRootBase = "checkpoints\content_delta_pop5\pseudo_cold_sage_v1",
    [ValidateSet("batch_random", "batch_tail", "all_eligible")]
    [string]$PseudoColdMode = "batch_tail",
    [double]$PseudoColdRatio = 0.20,
    [int]$PseudoColdMinPop = 5,
    [double]$SageTailPopRatio = 0.002,
    [int]$SagePoolTopK = 64,
    [int]$MinFreeGpuMiB = 9000,
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

$caseSlug = $Case.ToLowerInvariant()
if ($Case -eq "Both") {
    $queueLog = Join-Path $outputRootBaseAbs "pseudo_cold_sage_overnight_serial_queue.log"
    $planPath = Join-Path $outputRootBaseAbs "pseudo_cold_sage_overnight_serial_plan.json"
} else {
    $queueLog = Join-Path $outputRootBaseAbs ("pseudo_cold_sage_overnight_serial_{0}_queue.log" -f $caseSlug)
    $planPath = Join-Path $outputRootBaseAbs ("pseudo_cold_sage_overnight_serial_{0}_plan.json" -f $caseSlug)
}

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

function ConvertTo-RunSuffix {
    if ($Seeds.Count -eq 3 -and ($Seeds -contains 2025) -and ($Seeds -contains 2026) -and ($Seeds -contains 2027)) {
        return "e${Epochs}_3seed"
    }
    $seedText = ($Seeds | ForEach-Object { [string]$_ }) -join "_"
    return "e${Epochs}_seeds_${seedText}"
}

function Get-CaseRoot {
    param([string]$CaseId)
    $suffix = ConvertTo-RunSuffix
    return "{0}_{1}" -f $CaseId, $suffix
}

function New-RunnerParams {
    param(
        [string]$CaseId,
        [bool]$UseSageLite,
        [bool]$SageUseTwoExpert,
        [int]$Seed
    )
    $caseRoot = Get-CaseRoot -CaseId $CaseId
    return [ordered]@{
        PythonRunner = ".\py.bat"
        ScriptPath = "usim_feedback_fast3_content_delta.py"
        DataDir = "processed_data_hin_clean_pop5"
        RelationDir = "MOOCCube/relations"
        OutputRoot = (Join-Path $OutputRootBase $caseRoot)
        CheckpointRoot = (Join-Path $CheckpointRootBase $caseRoot)
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
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
        SageUseTwoExpert = $SageUseTwoExpert
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

function Test-CompletedRun {
    param(
        [string]$OutputRoot,
        [int]$Seed,
        [bool]$UseSageLite,
        [bool]$SageUseTwoExpert
    )
    $runDir = Join-Path (Join-Path $repoPath $OutputRoot) ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
    $summary = Join-Path $runDir "mooc_metrics_usim_feedback_fast3_content_delta_static_summary.csv"
    $final = Join-Path $runDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    $manifest = Join-Path $runDir "static_protocol_manifest.json"
    if (
        -not (Test-Path -LiteralPath $summary) -or
        -not (Test-Path -LiteralPath $final) -or
        -not (Test-Path -LiteralPath $manifest)
    ) {
        return $false
    }
    try {
        $json = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
        $cfg = $json.model_config
        return (
            $cfg.n_epochs -eq $Epochs -and
            $cfg.use_content_delta -eq $false -and
            $cfg.use_pseudo_cold_train -eq $true -and
            $cfg.pseudo_cold_mode -eq $PseudoColdMode -and
            [math]::Abs([double]$cfg.pseudo_cold_ratio - $PseudoColdRatio) -lt 1e-12 -and
            $cfg.pseudo_cold_min_pop -eq $PseudoColdMinPop -and
            $cfg.use_sage_lite -eq $UseSageLite -and
            $cfg.sage_only_cold_or_tail -eq $UseSageLite -and
            $cfg.sage_use_two_expert -eq $SageUseTwoExpert -and
            [math]::Abs([double]$cfg.sage_tail_pop_ratio - $SageTailPopRatio) -lt 1e-12 -and
            $cfg.use_sage_aux_loss -eq $false -and
            $cfg.mask_known_pos_neg -eq $true -and
            $cfg.mask_same_item_neg -eq $true
        )
    } catch {
        return $false
    }
}

$cases = @()
if ($Case -in @("Both", "P0")) {
    $cases += [pscustomobject]@{
        case_id = "P0_pseudo_only"
        use_sage_lite = $false
        sage_use_two_expert = $false
        purpose = "Pseudo-cold train episodes without SAGE; controls for proxy-cold item-ID dropout."
        root = Get-CaseRoot -CaseId "P0_pseudo_only"
    }
}
if ($Case -in @("Both", "P1")) {
    $cases += [pscustomobject]@{
        case_id = "P1_pseudo_sage"
        use_sage_lite = $true
        sage_use_two_expert = $false
        purpose = "Pseudo-cold train episodes plus tail-gated SAGE-lite course-aware sampling."
        root = Get-CaseRoot -CaseId "P1_pseudo_sage"
    }
}
if ($Case -in @("P2")) {
    $cases += [pscustomobject]@{
        case_id = "P2_pseudo_sage_twoexpert"
        use_sage_lite = $true
        sage_use_two_expert = $true
        purpose = "Pseudo-cold train episodes plus SAGE-lite two-expert candidate fusion."
        root = Get-CaseRoot -CaseId "P2_pseudo_sage_twoexpert"
    }
}

$runs = @()
foreach ($casePlan in $cases) {
    foreach ($seed in $Seeds) {
        $params = New-RunnerParams `
            -CaseId $casePlan.case_id `
            -UseSageLite ([bool]$casePlan.use_sage_lite) `
            -SageUseTwoExpert ([bool]$casePlan.sage_use_two_expert) `
            -Seed $seed
        $runs += [pscustomobject]@{
            case_id = $casePlan.case_id
            purpose = $casePlan.purpose
            seed = $seed
            output_root = $params.OutputRoot
            checkpoint_root = $params.CheckpointRoot
            runner_params = [pscustomobject]$params
        }
    }
}

$plan = [pscustomobject]@{
    created_at = (Get-Date).ToString("s")
    experiment = "pseudo_cold_sage_v1_overnight_serial"
    hypothesis = "Use tail training items as pseudo-cold episodes under strict item cold-start; then test whether SAGE-style course-aware sampling adds value over the same pseudo-cold control."
    reference = "SAGERec: Sampling and Gating for Enhanced Long-Tail Item Recommendations, WSDM 2026; implemented as a SAGE-lite adaptation."
    comparison_policy = ("Serial {0}-seed/e{1} runs. P0/P1/P2 share all settings except SAGE and two-expert switches." -f $Seeds.Count, $Epochs)
    baseline_refs = @(
        "outputs\content_delta_pop5\fn_mask_ab\aligned_oldcfg_mask_e60_3seed",
        "outputs\content_delta_pop5\sage_lite_v1\S5_tailratio_0p002_e60_resume_from_s4e8"
    )
    cases = $cases
    runs = $runs
}
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding UTF8
Write-QueueLog "Wrote overnight serial plan: $planPath"

if ($DryRun) {
    Write-QueueLog "DryRun requested; no training command launched."
    Write-Host "DRYRUN plan=$planPath"
    exit 0
}

foreach ($run in $runs) {
    $useSage = [bool]$run.runner_params.UseSageLite
    $useTwoExpert = [bool]$run.runner_params.SageUseTwoExpert
    if (Test-CompletedRun -OutputRoot $run.output_root -Seed $run.seed -UseSageLite $useSage -SageUseTwoExpert $useTwoExpert) {
        Write-QueueLog ("SKIP {0} seed={1}; matching completed run exists." -f $run.case_id, $run.seed)
        continue
    }

    Wait-GpuReady

    $runnerParams = @{}
    foreach ($prop in $run.runner_params.PSObject.Properties) {
        $runnerParams[$prop.Name] = $prop.Value
    }
    Write-QueueLog ("START {0} seed={1}: output={2} checkpoint={3}" -f `
        $run.case_id, $run.seed, $runnerParams.OutputRoot, $runnerParams.CheckpointRoot)
    & ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams
    $code = $LASTEXITCODE
    Write-QueueLog ("END {0} seed={1}: exit_code={2}" -f $run.case_id, $run.seed, $code)
    if ($code -ne 0) {
        exit $code
    }
    if (-not (Test-CompletedRun -OutputRoot $run.output_root -Seed $run.seed -UseSageLite $useSage -SageUseTwoExpert $useTwoExpert)) {
        Write-QueueLog ("ERROR {0} seed={1}; output missing or config mismatch after run." -f $run.case_id, $run.seed)
        exit 2
    }
}

foreach ($casePlan in $cases) {
    $rootRel = Join-Path $OutputRootBase $casePlan.root
    $rootAbs = Join-Path $repoPath $rootRel
    if (Test-Path -LiteralPath $rootAbs) {
        Write-QueueLog ("AGGREGATE {0}: root={1}" -f $casePlan.case_id, $rootRel)
        & ".\py.bat" "aggregate_fast3_static_results.py" --root $rootRel
        $code = $LASTEXITCODE
        Write-QueueLog ("AGGREGATE {0}: exit_code={1}" -f $casePlan.case_id, $code)
        if ($code -ne 0) {
            exit $code
        }
    }
}

Write-QueueLog "DONE pseudo-cold SAGE overnight serial."
