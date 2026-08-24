param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$queueRoot = "background_logs\lira_v2_completed_test"
$outputBase = "outputs\learner_guided_full_testcompleted"
New-Item -ItemType Directory -Path $queueRoot,$outputBase -Force | Out-Null
$queueLog = Join-Path $queueRoot "queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $queueLog -Append
}

function Format-Number([double]$Value) {
    return $Value.ToString([Globalization.CultureInfo]::InvariantCulture)
}

$jobs = @(
    [ordered]@{ RunName="lira_v2_dynamic_dualloss_seed2025"; Seed=2025; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_dynamic_dualloss_seed2026"; Seed=2026; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_dynamic_dualloss_seed2027"; Seed=2027; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t0_seed2025"; Seed=2025; Steps=0; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t0_seed2026"; Seed=2026; Steps=0; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t0_seed2027"; Seed=2027; Steps=0; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t1_seed2025"; Seed=2025; Steps=1; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t1_seed2026"; Seed=2026; Steps=1; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_t1_seed2027"; Seed=2027; Steps=1; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_no_stop_seed2025"; Seed=2025; Steps=3; MinFit=0.0; MinGain=0.0; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_no_stop_seed2027"; Seed=2027; Steps=3; MinFit=0.0; MinGain=0.0; RefineW=0.5; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_no_refined_loss_seed2025"; Seed=2025; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ RunName="lira_v2_ablation_no_stability_seed2025"; Seed=2025; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.0 }
)

$trainingHashes = @{
    source_sha256 = (Get-FileHash "lira_entry.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    model_sha256 = (Get-FileHash "lira\model.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    refinement_sha256 = (Get-FileHash "lira\refinement.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    adapter_sha256 = (Get-FileHash "lira\protocol_adapter.py" -Algorithm SHA256).Hash.ToLowerInvariant()
}

foreach ($job in $jobs) {
    $checkpointRoot = "checkpoints\learner_guided_full\$($job.RunName)"
    $checkpointDir = Join-Path $checkpointRoot "strict_item_cold_balanced_thr1_seed_$($job.Seed)"
    $checkpoint = Join-Path $checkpointDir "validation_finished.pt"
    $lockPath = Join-Path $checkpointDir "locked_config.json"
    $marker = Join-Path $queueRoot "$($job.RunName).test_finished"
    if (Test-Path -LiteralPath $marker) {
        Write-QueueLog "SKIP $($job.RunName) test completed"
        continue
    }
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Completed-test queue received an unfinished run: $($job.RunName)"
    }
    if (-not (Test-Path -LiteralPath $lockPath)) {
        throw "Missing locked configuration: $lockPath"
    }
    $locked = Get-Content $lockPath -Raw | ConvertFrom-Json
    foreach ($key in $trainingHashes.Keys) {
        if ([string]$locked.$key -ne [string]$trainingHashes[$key]) {
            throw "Training source hash changed for $($job.RunName): $key"
        }
    }
    if ([int]$locked.seed -ne [int]$job.Seed -or
        [int]$locked.usim_steps -ne [int]$job.Steps -or
        [double]$locked.min_fit -ne [double]$job.MinFit -or
        [double]$locked.min_gain -ne [double]$job.MinGain -or
        [double]$locked.refinement_loss_weight -ne [double]$job.RefineW -or
        [double]$locked.stability_loss_weight -ne [double]$job.StableW) {
        throw "Locked config mismatch for completed test replay: $($job.RunName)"
    }
    if ($DryRun) {
        Write-QueueLog "DRYRUN TEST $($job.RunName)"
        continue
    }

    $stdout = Join-Path $queueRoot "$($job.RunName).stdout.log"
    $stderr = Join-Path $queueRoot "$($job.RunName).stderr.log"
    $outputRoot = Join-Path $outputBase $job.RunName
    $env:USIM_DISABLE_LLM_SCORE = "1"
    $env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = "1"
    $env:USIM_VALIDATION_ONLY = "0"
    $env:LIRA_UPDATE_LR = "0.1"
    $env:LIRA_MIN_FIT = Format-Number $job.MinFit
    $env:LIRA_STEP_CAP = "0.05"
    $env:LIRA_TOTAL_CAP = "0.1"
    $env:LIRA_MIN_GAIN = Format-Number $job.MinGain
    $env:LIRA_REFINEMENT_LOSS_WEIGHT = Format-Number $job.RefineW
    $env:LIRA_STABILITY_LOSS_WEIGHT = Format-Number $job.StableW
    try {
        Write-QueueLog "START TEST $($job.RunName)"
        & .\run_usim_feedback_fast3_content_delta_static.ps1 `
            -MinimalLiraMode -ScriptPath "lira_checkpoint_test_eval.py" `
            -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
            -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($job.Seed) `
            -Epochs 35 -Patience 10 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
            -UsePseudoColdTrain $true -PseudoColdMode "item_tail" -PseudoColdRatio 0.3 -PseudoColdMinPop 5 `
            -TrainForceCold $true -AuxHotOnly $true `
            -UsimSteps $job.Steps -RolloutPolicy "course_fit" `
            -UseUsimRefinedEval $true -RunSampledEval $false `
            -OutputRoot $outputRoot -CheckpointRoot $checkpointRoot `
            -SaveCkpt $true -AutoResume $true -SaveOptState $false -SkipAggregate `
            *> $stdout
        if ($LASTEXITCODE -ne 0) { throw "Test replay failed with exit code $LASTEXITCODE" }
        New-Item -ItemType File -Path $marker -Force | Out-Null
        Write-QueueLog "DONE TEST $($job.RunName)"
    }
    catch {
        $_ | Out-String | Set-Content $stderr -Encoding utf8
        throw
    }
    finally {
        Remove-Item Env:USIM_DISABLE_LLM_SCORE,Env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET,Env:USIM_VALIDATION_ONLY -ErrorAction SilentlyContinue
        Remove-Item Env:LIRA_UPDATE_LR,Env:LIRA_MIN_FIT,Env:LIRA_STEP_CAP,Env:LIRA_TOTAL_CAP,Env:LIRA_MIN_GAIN,Env:LIRA_REFINEMENT_LOSS_WEIGHT,Env:LIRA_STABILITY_LOSS_WEIGHT -ErrorAction SilentlyContinue
    }
}

Write-QueueLog "ALL COMPLETED TEST REPLAYS DONE"
