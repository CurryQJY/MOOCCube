$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_fast3_main_table_config.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing main-table FAST3 runner: $script"
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -OutputRoot "outputs\_dryrun_maincfg" `
    -CheckpointRoot "checkpoints\_dryrun_maincfg" `
    -Seeds 2025 `
    -CourseTermNorm batch `
    -CoursePrereqW 0.12 `
    -CourseConceptW 0.04 `
    -CourseDiffW 0.02 `
    -CourseRedundantW 0.01 `
    -CourseSampleBeta 0.20 `
    -UseSageLite `
    -SageGateMin 0.10 `
    -SageGateMax 0.60 `
    -SageGateMode bucket_mlp `
    -SageGateBuckets 13 `
    -SageGateHidden 17 `
    -SageGateBucketStrategy paper `
    -SagePoolTopK 48 `
    -SageCourseTemp 0.25 `
    -SageOnlyColdOrTail `
    -SageTailPopRatio 0.10 `
    -SageTwoExpertScoreFusion `
    -UseSageAuxLoss `
    -SageAuxWeight 0.02 `
    -SageAuxPoolTopK 48 `
    -SageAuxCourseTemp 0.20 `
    -SageAuxRetrievalTemp 1.0 `
    -SageAuxOnlyStrictCold 1 `
    -SageAuxDetachUser 1 `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "DataDir=processed_data_hin_clean_pop5",
    "Protocol=strict_item_cold_balanced",
    "Seeds=2025",
    "Epochs=60",
    "Patience=60",
    "UseContentDelta=False",
    "UsePseudoColdTrain=False",
    "UsePaac=False",
    "CourseFeedbackOnlyCold=False",
    "CourseSampleOnlyCold=False",
    "PrereqAuxOnlyCold=False",
    "MaskKnownPosNeg=True",
    "MaskSameItemNeg=True",
    "CourseTermNorm=batch",
    "CoursePrereqW=0.12",
    "CourseConceptW=0.04",
    "CourseDiffW=0.02",
    "CourseRedundantW=0.01",
    "CourseSampleBeta=0.2",
    "UseSageLite=True",
    "SageGateMin=0.1",
    "SageGateMax=0.6",
    "SageGateMode=bucket_mlp",
    "SageGateBuckets=13",
    "SageGateHidden=17",
    "SageGateBucketStrategy=paper",
    "SagePoolTopK=48",
    "SageCourseTemp=0.25",
    "SageOnlyColdOrTail=True",
    "SageTailPopRatio=0.1",
    "SageTwoExpertScoreFusion=True",
    "UseSageAuxLoss=True",
    "SageAuxWeight=0.02",
    "SageAuxPoolTopK=48",
    "SageAuxCourseTemp=0.2",
    "SageAuxRetrievalTemp=1",
    "SageAuxOnlyStrictCold=True",
    "SageAuxDetachUser=True"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run setting: $expected"
    }
}

$auxOnlyOut = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -OutputRoot "outputs\_dryrun_maincfg_aux_only" `
    -CheckpointRoot "checkpoints\_dryrun_maincfg_aux_only" `
    -Seeds 2025 `
    -UseSageAuxLoss `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Aux-only dry-run exited with code $LASTEXITCODE"
}

$auxOnlyText = ($auxOnlyOut -join "`n")
foreach ($expected in @(
    "UseSageLite=False",
    "UseSageAuxLoss=True",
    "SageAuxOnlyStrictCold=True",
    "SageAuxDetachUser=True"
)) {
    if ($auxOnlyText -notlike "*$expected*") {
        throw "Missing expected aux-only dry-run setting: $expected"
    }
}

Write-Host "test_fast3_main_table_config_runner.ps1 passed"
