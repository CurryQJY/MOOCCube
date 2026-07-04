param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [int[]]$Seeds = @(2025),
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [string]$OutputRoot = "outputs\junyi\course_scope_ablation_v1\fff_e60_masktt",
    [string]$CheckpointRoot = "checkpoints\junyi\course_scope_ablation_v1\fff_e60_masktt",
    [switch]$ForceFreshRun,
    [switch]$NoAggregate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $Base $Path)
}

function Convert-ToStrictBool($Value) {
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    if ($Value -is [int]) {
        return ($Value -ne 0)
    }
    if ($Value -is [string]) {
        return ($Value -match "^(1|true|yes|on)$")
    }
    return [bool]$Value
}

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $script:QueueLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-SplitDir([string]$Root, [int]$Seed) {
    return (Join-Path $Root ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed))
}

function Test-CompletedWithExpectedConfig([string]$Root, [int]$Seed) {
    $splitDir = Get-SplitDir $Root $Seed
    $manifest = Join-Path $splitDir "static_protocol_manifest.json"
    $result = Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath $result)) {
        return $false
    }

    try {
        $json = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifest | ConvertFrom-Json
        $cfg = $json.model_config
        if ($null -eq $cfg) {
            return $false
        }
        return (
            (Convert-ToStrictBool $cfg.mask_known_pos_neg) -eq $true -and
            (Convert-ToStrictBool $cfg.mask_same_item_neg) -eq $true -and
            (Convert-ToStrictBool $cfg.feedback_course_only_cold) -eq $false -and
            (Convert-ToStrictBool $cfg.feedback_course_sample_only_cold) -eq $false -and
            (Convert-ToStrictBool $cfg.prereq_aux_only_cold) -eq $false -and
            (Convert-ToStrictBool $cfg.use_sage_lite) -eq $false -and
            (Convert-ToStrictBool $cfg.use_content_delta) -eq $false -and
            (Convert-ToStrictBool $cfg.use_pseudo_cold_train) -eq $false -and
            (Convert-ToStrictBool $cfg.use_course_reward) -eq $true -and
            (Convert-ToStrictBool $cfg.use_prereq_aux_loss) -eq $true
        )
    } catch {
        return $false
    }
}

function Write-Setting([string]$Name, [object]$Value) {
    Write-Host ("{0}={1}" -f $Name, $Value)
}

function Invoke-Seed([int]$Seed) {
    $splitDir = Get-SplitDir $script:OutputRootAbs $Seed
    if ((-not $ForceFreshRun) -and (Test-CompletedWithExpectedConfig $script:OutputRootAbs $Seed)) {
        Write-Log "SKIP seed=$Seed existing matching result | out=$splitDir"
        return
    }

    Write-Log "START seed=$Seed | course_scope=False/False/False | mask=True/True | out=$splitDir"
    if ($DryRun) {
        Write-Log "DRYRUN seed=$Seed"
        return
    }

    $runnerParams = @{
        PythonRunner = $PythonRunner
        DataDir = "processed_data_junyi"
        RelationDir = "processed_data_junyi\relations"
        OutputRoot = $script:OutputRootAbs
        CheckpointRoot = $script:CheckpointRootAbs
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
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
        CoursePrereqW = 0.08
        CourseConceptW = 0.04
        CourseDiffW = 0.03
        CourseRedundantW = 0.02
        CourseRedundantMode = "concept"
        CourseSampleBeta = 0.20
        CourseFeedbackOnlyCold = $false
        CourseSampleOnlyCold = $false
        PrereqAuxOnlyCold = $false
        UseSageLite = $false
        SageOnlyColdOrTail = $false
        SageUseTwoExpert = $false
        UseSageAuxLoss = $false
        UseCgrcRecon = $false
        UseCourseRerank = $false
        UseStructuredHardNeg = $false
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        RunSampledEval = $false
        SaveCkpt = $true
        AutoResume = $false
        ForceFresh = $true
        SaveOptState = $true
        SkipAggregate = $true
    }

    & $StaticRunner @runnerParams
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    Write-Log "END seed=$Seed | exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "Junyi course-scope false run failed for seed=$Seed with exit=$exitCode"
    }
    if (-not (Test-CompletedWithExpectedConfig $script:OutputRootAbs $Seed)) {
        throw "Seed=$Seed finished, but result/manifest does not match expected course-scope false + mask true config"
    }
}

$script:Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $script:Repo

$script:OutputRootAbs = Resolve-RunPath $script:Repo $OutputRoot
$script:CheckpointRootAbs = Resolve-RunPath $script:Repo $CheckpointRoot
$script:QueueDir = Join-Path $script:OutputRootAbs "_queue"
$script:QueueLog = Join-Path $script:QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $script:QueueDir | Out-Null
New-Item -ItemType Directory -Force -Path $script:CheckpointRootAbs | Out-Null

$Seeds = [int[]]@($Seeds | Sort-Object -Unique)

if ($DryRun) {
    Write-Host "Junyi course-scope false dry run"
    Write-Setting "Repo" $script:Repo
    Write-Setting "StaticRunner" $StaticRunner
    Write-Setting "PythonRunner" $PythonRunner
    Write-Setting "DataDir" "processed_data_junyi"
    Write-Setting "RelationDir" "processed_data_junyi\relations"
    Write-Setting "OutputRoot" $script:OutputRootAbs
    Write-Setting "CheckpointRoot" $script:CheckpointRootAbs
    Write-Setting "Protocol" "strict_item_cold_balanced"
    Write-Setting "ColdThresholds" "1"
    Write-Setting "Seeds" ($Seeds -join ",")
    Write-Setting "Epochs" $Epochs
    Write-Setting "Patience" $Patience
    Write-Setting "EarlyStopAverageMode" "item_macro"
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
    Write-Setting "UseSageLite" $false
    Write-Setting "SageUseTwoExpert" $false
    Write-Setting "UseSageAuxLoss" $false
    Write-Setting "UseCourseRerank" $false
    Write-Setting "UseStructuredHardNeg" $false
    Write-Setting "MaskKnownPosNeg" $true
    Write-Setting "MaskSameItemNeg" $true
    Write-Setting "RunSampledEval" $false
    Write-Setting "SaveCkpt" $true
    Write-Setting "SaveOptState" $true
    Write-Setting "ForceFreshRun" ([bool]$ForceFreshRun)
    Write-Setting "NoAggregate" ([bool]$NoAggregate)
}

Write-Log "QUEUE START Junyi course-scope false | seeds=$($Seeds -join ',') epochs=$Epochs patience=$Patience dry_run=$DryRun"
foreach ($seed in $Seeds) {
    Invoke-Seed $seed
}

if ((-not $DryRun) -and (-not $NoAggregate)) {
    $aggregateLog = Join-Path $script:QueueDir "aggregate.log"
    Write-Log "START aggregate | root=$script:OutputRootAbs"
    & $PythonRunner "aggregate_fast3_static_results.py" --root $script:OutputRootAbs *> $aggregateLog
    $aggregateExit = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    Write-Log "END aggregate | exit=$aggregateExit | log=$aggregateLog"
    if ($aggregateExit -ne 0) {
        throw "Aggregation failed with exit=$aggregateExit"
    }
}

Write-Log "QUEUE DONE Junyi course-scope false"
