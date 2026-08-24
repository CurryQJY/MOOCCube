$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_paper_supervised_ckg_validation_serial.ps1"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing paper supervised-CKG validation runner: $script"
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[int]`$UsimSteps",
    "[double]`$PpoLossWeight",
    "[string]`$RolloutPolicy",
    "USIM_STEPS",
    "USIM_PPO_LOSS_WEIGHT",
    "USIM_ROLLOUT_POLICY"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose expected validation control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -DatasetList "mooccube,junyi,coco" `
    -SeedList "2025" `
    -VariantList "no_ppo_rollout,ckg_sup_t0,content_masked_sup" `
    -OutputRootBase "outputs\_dryrun_paper_supervised_ckg_validation" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_supervised_ckg_validation" `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Paper supervised-CKG validation dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "Paper supervised CKG validation dry run",
    "Datasets=mooccube,junyi,coco",
    "Seeds=2025",
    "Dataset=mooccube",
    "DataDir=processed_data_hin_clean_pop5",
    "RelationDir=MOOCCube/relations",
    "LegacyResultRoot=outputs\content_delta_pop5\course_ppo_ablation_e60_3seed\wo_ppo_loss",
    "RunMode=reuse_legacy_if_available",
    "Dataset=junyi",
    "DataDir=processed_data_junyi",
    "RelationDir=processed_data_junyi\relations",
    "Dataset=coco",
    "DataDir=processed_data_coco",
    "RelationDir=processed_data_coco\relations",
    "Variant=no_ppo_rollout",
    "Label=No-PPO rollout",
    "UsimSteps=5",
    "PpoLossWeight=0",
    "RolloutPolicy=ppo",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True",
    "Variant=ckg_sup_t0",
    "Label=CKG-Sup (T=0)",
    "UseCourseFeedback=True",
    "UseCourseReward=False",
    "UseCourseSample=False",
    "UsePrereqAux=True",
    "Variant=content_masked_sup",
    "Label=Content-masked Sup",
    "LegacyResultRoot=outputs\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer",
    "UseCourseFeedback=False",
    "UseCourseReward=False",
    "UseCourseSample=False",
    "UsePrereqAux=False"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected supervised-CKG validation dry-run line: $expected"
    }
}

Write-Host "test_paper_supervised_ckg_validation_serial.ps1 passed"
