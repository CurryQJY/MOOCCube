param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 15,
    [int]$Patience = 15,
    [int]$UsimSteps = 3,
    [bool]$ValidationOnly = $true,
    [double]$UpdateLr = 0.10,
    [double]$MinFit = 0.05,
    [double]$StepCap = 0.05,
    [double]$TotalCap = 0.10,
    [double]$MinGain = 0.001,
    [double]$RefinementLossWeight = 0.5,
    [double]$StabilityLossWeight = 0.01,
    [string]$RunName = "learner_guided_full_seed2025",
    [switch]$ForceFresh
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$outputRoot = "outputs\learner_guided_full\$RunName"
$checkpointRoot = "checkpoints\learner_guided_full\$RunName"
$seedTag = "strict_item_cold_balanced_thr1_seed_$Seed"
$checkpointDir = Join-Path $checkpointRoot $seedTag
$lockPath = Join-Path $checkpointDir "locked_config.json"
$config = [ordered]@{
    method = "learner_guided_full"
    source = "lira_entry.py"
    source_sha256 = (Get-FileHash "lira_entry.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    model_sha256 = (Get-FileHash "lira\model.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    refinement_sha256 = (Get-FileHash "lira\refinement.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    adapter_sha256 = (Get-FileHash "lira\protocol_adapter.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    seed = $Seed
    epochs = $Epochs
    patience = $Patience
    pseudo_cold_mode = "item_tail"
    pseudo_cold_ratio = 0.3
    usim_steps = $UsimSteps
    update_lr = $UpdateLr
    min_fit = $MinFit
    step_cap = $StepCap
    total_cap = $TotalCap
    min_gain = $MinGain
    refinement_loss_weight = $RefinementLossWeight
    stability_loss_weight = $StabilityLossWeight
    validation_only = $ValidationOnly
}

New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
if (Test-Path $lockPath) {
    $old = Get-Content $lockPath -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 10 -Compress
    $new = $config | ConvertTo-Json -Depth 10 -Compress
    if ($old -ne $new -and -not [bool]$ForceFresh) {
        throw "Locked LIRA configuration or source hash changed; refusing resume."
    }
}
$config | ConvertTo-Json -Depth 10 | Set-Content $lockPath -Encoding utf8

$env:USIM_DISABLE_LLM_SCORE = "1"
$env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = "1"
$env:USIM_VALIDATION_ONLY = if ($ValidationOnly) { "1" } else { "0" }
$env:LIRA_UPDATE_LR = [string]$UpdateLr
$env:LIRA_MIN_FIT = [string]$MinFit
$env:LIRA_STEP_CAP = [string]$StepCap
$env:LIRA_TOTAL_CAP = [string]$TotalCap
$env:LIRA_MIN_GAIN = [string]$MinGain
$env:LIRA_REFINEMENT_LOSS_WEIGHT = [string]$RefinementLossWeight
$env:LIRA_STABILITY_LOSS_WEIGHT = [string]$StabilityLossWeight
try {
    & .\run_usim_feedback_fast3_content_delta_static.ps1 `
        -MinimalLiraMode `
        -ScriptPath "lira_entry.py" `
        -DataDir "processed_data_hin_clean_pop5" -RelationDir "MOOCCube/relations" `
        -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($Seed) `
        -Epochs $Epochs -Patience $Patience -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
        -UsePseudoColdTrain $true -PseudoColdMode "item_tail" -PseudoColdRatio 0.3 -PseudoColdMinPop 5 `
        -TrainForceCold $true -AuxHotOnly $true `
        -UsimSteps $UsimSteps -RolloutPolicy "course_fit" `
        -UseUsimRefinedEval $true -RunSampledEval $false `
        -OutputRoot $outputRoot -CheckpointRoot $checkpointRoot `
        -SaveCkpt $true -AutoResume $true -ForceFresh ([bool]$ForceFresh) -SaveOptState $true -SkipAggregate
    if ($LASTEXITCODE -ne 0) { throw "LIRA run failed with exit code $LASTEXITCODE" }
}
finally {
    Remove-Item Env:USIM_DISABLE_LLM_SCORE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_VALIDATION_ONLY -ErrorAction SilentlyContinue
    Remove-Item Env:LIRA_UPDATE_LR,Env:LIRA_MIN_FIT,Env:LIRA_STEP_CAP,Env:LIRA_TOTAL_CAP,Env:LIRA_MIN_GAIN,Env:LIRA_REFINEMENT_LOSS_WEIGHT,Env:LIRA_STABILITY_LOSS_WEIGHT -ErrorAction SilentlyContinue
}
