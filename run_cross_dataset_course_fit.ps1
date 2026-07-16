param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("junyi", "coco")]
    [string]$Dataset,
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [ValidateSet("validation", "test")]
    [string]$Target = "validation"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"

$settings = @{
    junyi = [ordered]@{
        DataDir = "processed_data_junyi"
        RelationDir = "processed_data_junyi\relations"
        CheckpointRoot = "checkpoints\junyi\main_table_3seed\ours"
        Epochs = 60
        Patience = 60
        CourseOnlyCold = $true
        SampleOnlyCold = $true
        PrereqOnlyCold = $true
    }
    coco = [ordered]@{
        DataDir = "processed_data_coco"
        RelationDir = "processed_data_coco\relations"
        CheckpointRoot = "checkpoints\coco\single_seed_triage\ours_full"
        Epochs = 30
        Patience = 10
        CourseOnlyCold = $false
        SampleOnlyCold = $false
        PrereqOnlyCold = $false
    }
}

$cfg = $settings[$Dataset]
$outputRoot = "outputs\recppo_research_repair\cross_dataset_course_fit\$Dataset\$Target\course_fit"
$integrityRows = @()

try {
    $env:USIM_INFERENCE_STEPS_OVERRIDE = "5"
    $env:USIM_ACTOR_EVAL_TARGET = $Target
    $env:USIM_ACTOR_INFERENCE_MODE = "course_fit"
    $env:USIM_ACTOR_INFERENCE_SEED = "7001"
    $env:USIM_COURSE_MATCH_EXCLUDE_TARGET = "false"

    foreach ($seed in $Seeds) {
        $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
        $checkpoint = Join-Path $cfg.CheckpointRoot "$seedTag\finished.pt"
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            throw "Missing checkpoint: $checkpoint"
        }

        $before = Get-Item -LiteralPath $checkpoint
        $beforeHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $armDir = Join-Path $outputRoot $seedTag
        $finalCsv = Join-Path $armDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        $auditJson = Join-Path $armDir "actor_inference_audit.json"

        if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $auditJson)) {
            Write-Host "SKIP completed dataset=$Dataset target=$Target seed=$seed course_fit"
        } else {
            Write-Host "[$(Get-Date -Format o)] START dataset=$Dataset target=$Target seed=$seed mode=course_fit"
            & $runner `
                -ScriptPath "main_checkpoint_actor_inference_ab.py" `
                -DataDir $cfg.DataDir -RelationDir $cfg.RelationDir `
                -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
                -Epochs $cfg.Epochs -Patience $cfg.Patience `
                -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
                -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
                -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
                -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
                -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
                -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
                -PpoLossWeight 1.0 -RolloutPolicy "ppo" -RlResidualScale 1.0 `
                -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
                -CourseFeedbackOnlyCold $cfg.CourseOnlyCold -CourseSampleOnlyCold $cfg.SampleOnlyCold `
                -PrereqAuxOnlyCold $cfg.PrereqOnlyCold -UseUsimRefinedEval $true `
                -UseCourseRerank $false -UseStructuredHardNeg $false -UseSageLite $false `
                -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
                -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
                -OutputRoot $outputRoot -CheckpointRoot $cfg.CheckpointRoot `
                -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate

            if ($LASTEXITCODE -ne 0) {
                throw "Course-fit evaluation failed: dataset=$Dataset target=$Target seed=$seed"
            }
            Write-Host "[$(Get-Date -Format o)] DONE dataset=$Dataset target=$Target seed=$seed mode=course_fit"
        }

        $after = Get-Item -LiteralPath $checkpoint
        $afterHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $unchanged = (
            $after.Length -eq $before.Length -and
            $after.LastWriteTimeUtc -eq $before.LastWriteTimeUtc -and
            $afterHash -eq $beforeHash
        )
        $integrityRows += [ordered]@{
            dataset = $Dataset
            target = $Target
            seed = $seed
            checkpoint = $after.FullName
            before_sha256 = $beforeHash
            after_sha256 = $afterHash
            unchanged = $unchanged
        }
        if (-not $unchanged) {
            throw "Frozen checkpoint changed: dataset=$Dataset seed=$seed"
        }
    }

    $integrityPath = "outputs\recppo_research_repair\cross_dataset_course_fit\$Dataset\$Target\checkpoint_integrity.json"
    New-Item -ItemType Directory -Path (Split-Path $integrityPath -Parent) -Force | Out-Null
    $integrityRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $integrityPath -Encoding utf8
    Write-Host "COURSE-FIT COMPLETE dataset=$Dataset target=$Target $(Get-Date -Format o)"
}
finally {
    foreach ($name in @(
        "USIM_INFERENCE_STEPS_OVERRIDE",
        "USIM_ACTOR_EVAL_TARGET",
        "USIM_ACTOR_INFERENCE_MODE",
        "USIM_ACTOR_INFERENCE_SEED",
        "USIM_COURSE_MATCH_EXCLUDE_TARGET"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}
