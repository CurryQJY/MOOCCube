param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$OutputRoot = "outputs\content_delta_pop5\course_maincfg_runs\maincfg",
    [string]$CheckpointRoot = "checkpoints\content_delta_pop5\course_maincfg_runs\maincfg",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int[]]$ColdThresholds = @(1),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [ValidateSet("none", "batch", "ema")]
    [string]$CourseTermNorm = "none",
    [double]$CourseTermNormClip = 2.0,
    [double]$CourseTermNormEps = 1e-6,
    [double]$CourseTermNormEmaDecay = 0.95,
    [double]$CoursePrereqW = 0.08,
    [double]$CourseConceptW = 0.04,
    [double]$CourseDiffW = 0.03,
    [double]$CourseRedundantW = 0.02,
    [double]$CourseSampleBeta = 0.20,
    [ValidateSet("legacy_id", "initial_state")]
    [string]$SimulatorTargetMode = "legacy_id",
    [bool]$DeterministicEvalCandidates = $false,
    [bool]$EvalReuseItemBank = $false,
    [int]$DeterministicEvalSeed = 0,
    [switch]$UseSageLite,
    [double]$SageGateMin = 0.10,
    [double]$SageGateMax = 0.60,
    [ValidateSet("heuristic", "bucket_mlp")]
    [string]$SageGateMode = "heuristic",
    [int]$SageGateBuckets = 20,
    [int]$SageGateHidden = 32,
    [ValidateSet("paper", "log")]
    [string]$SageGateBucketStrategy = "paper",
    [int]$SagePoolTopK = 64,
    [double]$SageCourseTemp = 0.20,
    [switch]$SageOnlyColdOrTail,
    [double]$SageTailPopRatio = 0.10,
    [switch]$SageTwoExpertScoreFusion,
    [switch]$UseSageAuxLoss,
    [double]$SageAuxWeight = 0.02,
    [int]$SageAuxPoolTopK = 48,
    [double]$SageAuxCourseTemp = 0.20,
    [double]$SageAuxRetrievalTemp = 1.0,
    [object]$SageAuxOnlyStrictCold = $true,
    [object]$SageAuxDetachUser = $true,
    [switch]$SaveCkpt,
    [switch]$SaveOptState,
    [switch]$SkipAggregate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $repoPath

$normEnv = @(
    "USIM_FB_COURSE_TERM_NORM",
    "USIM_FB_COURSE_TERM_NORM_CLIP",
    "USIM_FB_COURSE_TERM_NORM_EPS",
    "USIM_FB_COURSE_TERM_NORM_EMA_DECAY",
    "USIM_USE_SAGE_LITE",
    "USIM_SAGE_GATE_MIN",
    "USIM_SAGE_GATE_MAX",
    "USIM_SAGE_GATE_MODE",
    "USIM_SAGE_GATE_BUCKETS",
    "USIM_SAGE_GATE_HIDDEN",
    "USIM_SAGE_GATE_BUCKET_STRATEGY",
    "USIM_SAGE_POOL_TOPK",
    "USIM_SAGE_COURSE_TEMP",
    "USIM_SAGE_ONLY_COLD_OR_TAIL",
    "USIM_SAGE_TAIL_POP_RATIO",
    "USIM_SAGE_TWO_EXPERT_SCORE_FUSION",
    "USIM_USE_SAGE_AUX_LOSS",
    "USIM_SAGE_AUX_WEIGHT",
    "USIM_SAGE_AUX_POOL_TOPK",
    "USIM_SAGE_AUX_COURSE_TEMP",
    "USIM_SAGE_AUX_RETRIEVAL_TEMP",
    "USIM_SAGE_AUX_ONLY_STRICT_COLD",
    "USIM_SAGE_AUX_DETACH_USER",
    "USIM_SIMULATOR_TARGET_MODE",
    "USIM_DETERMINISTIC_EVAL_CANDIDATES",
    "USIM_EVAL_REUSE_ITEM_BANK",
    "USIM_DETERMINISTIC_EVAL_SEED"
)
$originalEnv = @{}
foreach ($name in $normEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
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

function Convert-ToBoolSetting {
    param(
        [object]$Value,
        [string]$Name
    )
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    if ($text -in @("1", "true", "yes", "on")) {
        return $true
    }
    if ($text -in @("0", "false", "no", "off")) {
        return $false
    }
    throw "$Name must be a boolean-like value: true/false/1/0"
}

try {
    $sageAuxOnlyStrictColdBool = Convert-ToBoolSetting $SageAuxOnlyStrictCold "SageAuxOnlyStrictCold"
    $sageAuxDetachUserBool = Convert-ToBoolSetting $SageAuxDetachUser "SageAuxDetachUser"

    $env:USIM_FB_COURSE_TERM_NORM = $CourseTermNorm
    $env:USIM_FB_COURSE_TERM_NORM_CLIP = [string]$CourseTermNormClip
    $env:USIM_FB_COURSE_TERM_NORM_EPS = [string]$CourseTermNormEps
    $env:USIM_FB_COURSE_TERM_NORM_EMA_DECAY = [string]$CourseTermNormEmaDecay
    $env:USIM_USE_SAGE_LITE = if ($UseSageLite) { "1" } else { "0" }
    $env:USIM_SAGE_GATE_MIN = [string]$SageGateMin
    $env:USIM_SAGE_GATE_MAX = [string]$SageGateMax
    $env:USIM_SAGE_GATE_MODE = $SageGateMode
    $env:USIM_SAGE_GATE_BUCKETS = [string]$SageGateBuckets
    $env:USIM_SAGE_GATE_HIDDEN = [string]$SageGateHidden
    $env:USIM_SAGE_GATE_BUCKET_STRATEGY = $SageGateBucketStrategy
    $env:USIM_SAGE_POOL_TOPK = [string]$SagePoolTopK
    $env:USIM_SAGE_COURSE_TEMP = [string]$SageCourseTemp
    $env:USIM_SAGE_ONLY_COLD_OR_TAIL = if ($SageOnlyColdOrTail) { "1" } else { "0" }
    $env:USIM_SAGE_TAIL_POP_RATIO = [string]$SageTailPopRatio
    $env:USIM_SAGE_TWO_EXPERT_SCORE_FUSION = if ($SageTwoExpertScoreFusion) { "1" } else { "0" }
    $env:USIM_USE_SAGE_AUX_LOSS = if ($UseSageAuxLoss) { "1" } else { "0" }
    $env:USIM_SAGE_AUX_WEIGHT = [string]$SageAuxWeight
    $env:USIM_SAGE_AUX_POOL_TOPK = [string]$SageAuxPoolTopK
    $env:USIM_SAGE_AUX_COURSE_TEMP = [string]$SageAuxCourseTemp
    $env:USIM_SAGE_AUX_RETRIEVAL_TEMP = [string]$SageAuxRetrievalTemp
    $env:USIM_SAGE_AUX_ONLY_STRICT_COLD = if ($sageAuxOnlyStrictColdBool) { "1" } else { "0" }
    $env:USIM_SAGE_AUX_DETACH_USER = if ($sageAuxDetachUserBool) { "1" } else { "0" }
    $env:USIM_SIMULATOR_TARGET_MODE = $SimulatorTargetMode
    $env:USIM_DETERMINISTIC_EVAL_CANDIDATES = if ($DeterministicEvalCandidates) { "1" } else { "0" }
    $env:USIM_EVAL_REUSE_ITEM_BANK = if ($EvalReuseItemBank) { "1" } else { "0" }
    $env:USIM_DETERMINISTIC_EVAL_SEED = [string]$DeterministicEvalSeed

    $runnerParams = @{
        PythonRunner = $PythonRunner
        ScriptPath = $ScriptPath
        DataDir = $DataDir
        RelationDir = $RelationDir
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = $ColdThresholds
        Seeds = $Seeds
        Epochs = $Epochs
        Patience = $Patience
        EarlyStopAverageMode = "item_macro"
        UseContentDelta = $false
        UsePseudoColdTrain = $false
        UsePaac = $false
        UseCourseFeedback = $true
        UseCourseReward = $true
        UseCourseSample = $true
        UsePrereqAux = $true
        PrereqGraphSource = "concept"
        CoursePrereqW = $CoursePrereqW
        CourseConceptW = $CourseConceptW
        CourseDiffW = $CourseDiffW
        CourseRedundantW = $CourseRedundantW
        CourseRedundantMode = "concept"
        CourseTermNorm = $CourseTermNorm
        CourseTermNormClip = $CourseTermNormClip
        CourseTermNormEps = $CourseTermNormEps
        CourseTermNormEmaDecay = $CourseTermNormEmaDecay
        CourseFeedbackOnlyCold = $false
        CourseSampleOnlyCold = $false
        PrereqAuxOnlyCold = $false
        CourseSampleBeta = $CourseSampleBeta
        UseSageLite = [bool]$UseSageLite
        SageGateMin = $SageGateMin
        SageGateMax = $SageGateMax
        SageGateMode = $SageGateMode
        SageGateBuckets = $SageGateBuckets
        SageGateHidden = $SageGateHidden
        SageGateBucketStrategy = $SageGateBucketStrategy
        SagePoolTopK = $SagePoolTopK
        SageCourseTemp = $SageCourseTemp
        SageOnlyColdOrTail = [bool]$SageOnlyColdOrTail
        SageTailPopRatio = $SageTailPopRatio
        SageTwoExpertScoreFusion = [bool]$SageTwoExpertScoreFusion
        UseSageAuxLoss = [bool]$UseSageAuxLoss
        SageAuxWeight = $SageAuxWeight
        SageAuxPoolTopK = $SageAuxPoolTopK
        SageAuxCourseTemp = $SageAuxCourseTemp
        SageAuxRetrievalTemp = $SageAuxRetrievalTemp
        SageAuxOnlyStrictCold = $sageAuxOnlyStrictColdBool
        SageAuxDetachUser = $sageAuxDetachUserBool
        UseCourseRerank = $false
        UseStructuredHardNeg = $false
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        RunSampledEval = $false
        SimulatorTargetMode = $SimulatorTargetMode
        DeterministicEvalCandidates = $DeterministicEvalCandidates
        EvalReuseItemBank = $EvalReuseItemBank
        DeterministicEvalSeed = $DeterministicEvalSeed
        SaveCkpt = [bool]$SaveCkpt
        AutoResume = $false
        ForceFresh = $true
        SaveOptState = [bool]$SaveOptState
    }
    if ($SkipAggregate) {
        $runnerParams["SkipAggregate"] = $true
    }

    if ($DryRun) {
        Write-Host "FAST3 main-table config dry run"
        Write-Setting "Repo" $repoPath
        Write-Setting "StaticRunner" $StaticRunner
        Write-Setting "DataDir" $DataDir
        Write-Setting "RelationDir" $RelationDir
        Write-Setting "OutputRoot" $OutputRoot
        Write-Setting "CheckpointRoot" $CheckpointRoot
        Write-Setting "Protocol" "strict_item_cold_balanced"
        Write-Setting "ColdThresholds" (Format-ListValue $ColdThresholds)
        Write-Setting "Seeds" (Format-ListValue $Seeds)
        Write-Setting "Epochs" $Epochs
        Write-Setting "Patience" $Patience
        Write-Setting "EarlyStopAverageMode" "item_macro"
        Write-Setting "RunSampledEval" $false
        Write-Setting "UseContentDelta" $false
        Write-Setting "UsePseudoColdTrain" $false
        Write-Setting "UsePaac" $false
        Write-Setting "UseCourseFeedback" $true
        Write-Setting "UseCourseReward" $true
        Write-Setting "UseCourseSample" $true
        Write-Setting "UsePrereqAux" $true
        Write-Setting "CourseFeedbackOnlyCold" $false
        Write-Setting "CourseSampleOnlyCold" $false
        Write-Setting "PrereqAuxOnlyCold" $false
        Write-Setting "UseCourseRerank" $false
        Write-Setting "UseStructuredHardNeg" $false
        Write-Setting "MaskKnownPosNeg" $true
        Write-Setting "MaskSameItemNeg" $true
        Write-Setting "SaveCkpt" ([bool]$SaveCkpt)
        Write-Setting "SaveOptState" ([bool]$SaveOptState)
        Write-Setting "CourseTermNorm" $CourseTermNorm
        Write-Setting "CourseTermNormClip" $CourseTermNormClip
        Write-Setting "CourseTermNormEps" $CourseTermNormEps
        Write-Setting "CourseTermNormEmaDecay" $CourseTermNormEmaDecay
        Write-Setting "CoursePrereqW" $CoursePrereqW
        Write-Setting "CourseConceptW" $CourseConceptW
        Write-Setting "CourseDiffW" $CourseDiffW
        Write-Setting "CourseRedundantW" $CourseRedundantW
        Write-Setting "CourseSampleBeta" $CourseSampleBeta
        Write-Setting "SimulatorTargetMode" $SimulatorTargetMode
        Write-Setting "DeterministicEvalCandidates" $DeterministicEvalCandidates
        Write-Setting "EvalReuseItemBank" $EvalReuseItemBank
        Write-Setting "DeterministicEvalSeed" $DeterministicEvalSeed
        Write-Setting "UseSageLite" ([bool]$UseSageLite)
        Write-Setting "SageGateMin" $SageGateMin
        Write-Setting "SageGateMax" $SageGateMax
        Write-Setting "SageGateMode" $SageGateMode
        Write-Setting "SageGateBuckets" $SageGateBuckets
        Write-Setting "SageGateHidden" $SageGateHidden
        Write-Setting "SageGateBucketStrategy" $SageGateBucketStrategy
        Write-Setting "SagePoolTopK" $SagePoolTopK
        Write-Setting "SageCourseTemp" $SageCourseTemp
        Write-Setting "SageOnlyColdOrTail" ([bool]$SageOnlyColdOrTail)
        Write-Setting "SageTailPopRatio" $SageTailPopRatio
        Write-Setting "SageTwoExpertScoreFusion" ([bool]$SageTwoExpertScoreFusion)
        Write-Setting "UseSageAuxLoss" ([bool]$UseSageAuxLoss)
        Write-Setting "SageAuxWeight" $SageAuxWeight
        Write-Setting "SageAuxPoolTopK" $SageAuxPoolTopK
        Write-Setting "SageAuxCourseTemp" $SageAuxCourseTemp
        Write-Setting "SageAuxRetrievalTemp" $SageAuxRetrievalTemp
        Write-Setting "SageAuxOnlyStrictCold" $sageAuxOnlyStrictColdBool
        Write-Setting "SageAuxDetachUser" $sageAuxDetachUserBool
        return
    }

    & $StaticRunner @runnerParams
    if ($LASTEXITCODE -ne 0) {
        throw "FAST3 main-table config run failed with exit code $LASTEXITCODE"
    }
}
finally {
    foreach ($name in $normEnv) {
        if ($null -eq $originalEnv[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $originalEnv[$name]
        }
    }
}
