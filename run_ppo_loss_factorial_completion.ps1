param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$checkpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_core_ablation\wo_ppo_loss"
$factorialRoot = "outputs\recppo_research_repair\ppo_loss_factorial"
$outputRoot = Join-Path $factorialRoot "off_course_fit"
$onStaticRoot = "outputs\recovery_validation\main_table_51ea12fc_candidate"
$onCourseFitRoot = "outputs\recppo_research_repair\test_course_fit_frozen\course_fit"
$offStaticRoot = "outputs\recovery_validation\main_table_51ea12fc_core_ablation\wo_ppo_loss"
$checkpointBefore = @{}
$completed = @()

foreach ($seed in $Seeds) {
    $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
    $checkpoint = Join-Path $checkpointRoot "$seedTag\finished.pt"
    if (Test-Path -LiteralPath $checkpoint) {
        $item = Get-Item -LiteralPath $checkpoint
        $checkpointBefore[$seed] = [ordered]@{
            path = $item.FullName
            length = $item.Length
            last_write_utc = $item.LastWriteTimeUtc.ToString("o")
            sha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}

try {
    $env:USIM_ACTOR_EVAL_TARGET = "test"
    $env:USIM_ACTOR_INFERENCE_MODE = "course_fit"
    $env:USIM_ACTOR_INFERENCE_SEED = "7001"
    $env:USIM_COURSE_MATCH_EXCLUDE_TARGET = "false"

    foreach ($seed in $Seeds) {
        $seedTag = "strict_item_cold_balanced_thr1_seed_$seed"
        $finishedCheckpoint = Join-Path $checkpointRoot "$seedTag\finished.pt"
        if (-not (Test-Path -LiteralPath $finishedCheckpoint)) {
            Write-Host "SKIP seed=${seed}: finished.pt is not available"
            continue
        }

        $armOutput = Join-Path $outputRoot $seedTag
        $finalCsv = Join-Path $armOutput "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        $auditJson = Join-Path $armOutput "actor_inference_audit.json"
        if ((Test-Path -LiteralPath $finalCsv) -and (Test-Path -LiteralPath $auditJson)) {
            Write-Host "SKIP completed PPO-loss factorial arm: seed=$seed"
            $completed += $seed
            continue
        }

        Write-Host "===== PPO-LOSS FACTORIAL seed=$seed train=off inference=course_fit START $(Get-Date -Format o) ====="
        & $runner `
            -ScriptPath "main_checkpoint_actor_inference_ab.py" `
            -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
            -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
            -Epochs 60 -Patience 60 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
            -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
            -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
            -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
            -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
            -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
            -PpoLossWeight 0.00 -RolloutPolicy "ppo" -RlResidualScale 1.00 `
            -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
            -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
            -UseUsimRefinedEval $true -UseCourseRerank $false -UseStructuredHardNeg $false `
            -UseSageLite $false -UseSageAuxLoss $false -UseCgrcRecon $false -UseSgUrinit $false `
            -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
            -OutputRoot $outputRoot -CheckpointRoot $checkpointRoot `
            -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate

        if ($LASTEXITCODE -ne 0) {
            throw "PPO-loss factorial evaluation failed: seed=$seed exit=$LASTEXITCODE"
        }
        $completed += $seed
        Write-Host "===== PPO-LOSS FACTORIAL seed=$seed train=off inference=course_fit DONE $(Get-Date -Format o) ====="
    }

    $checkpointRows = @()
    foreach ($seed in $checkpointBefore.Keys) {
        $before = $checkpointBefore[$seed]
        $item = Get-Item -LiteralPath $before.path
        $afterHash = (Get-FileHash -LiteralPath $before.path -Algorithm SHA256).Hash.ToLowerInvariant()
        $unchanged = (
            $item.Length -eq $before.length -and
            $item.LastWriteTimeUtc.ToString("o") -eq $before.last_write_utc -and
            $afterHash -eq $before.sha256
        )
        $checkpointRows += [ordered]@{
            seed = [int]$seed
            path = $before.path
            before_sha256 = $before.sha256
            after_sha256 = $afterHash
            before_last_write_utc = $before.last_write_utc
            after_last_write_utc = $item.LastWriteTimeUtc.ToString("o")
            unchanged = $unchanged
        }
        if (-not $unchanged) {
            throw "Frozen w/o PPO-loss checkpoint changed: seed=$seed"
        }
    }

    if ($completed.Count -gt 0) {
        New-Item -ItemType Directory -Path $factorialRoot -Force | Out-Null
        $checkpointRows | ConvertTo-Json -Depth 4 | Set-Content `
            -LiteralPath (Join-Path $factorialRoot "checkpoint_integrity.json") -Encoding utf8
        & .\py.bat ppo_loss_factorial_report.py `
            --on-static-root $onStaticRoot `
            --on-course-fit-root $onCourseFitRoot `
            --off-static-root $offStaticRoot `
            --off-course-fit-root $outputRoot `
            --output-root $factorialRoot `
            --seeds $completed
        if ($LASTEXITCODE -ne 0) {
            throw "PPO-loss factorial report failed: exit=$LASTEXITCODE"
        }
    }
}
finally {
    Remove-Item Env:USIM_ACTOR_EVAL_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_SEED -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_COURSE_MATCH_EXCLUDE_TARGET -ErrorAction SilentlyContinue
}
