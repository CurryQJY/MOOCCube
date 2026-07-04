param(
    [int]$Epochs = 30,
    [int]$Patience = 15
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

# Keep this matched to the MOOCCubeX augv2 e30 main-table protocol.
$env:USIM_FB_COURSE_CONCEPT_MIN = "0.01"

.\run_usim_feedback_fast3_content_delta_static.ps1 `
    -PythonRunner ".\py.bat" `
    -ScriptPath "usim_feedback_fast3_content_delta.py" `
    -DataDir "processed_data_hin_x" `
    -RelationDir "MOOCCubeX\relations_aug_v2" `
    -OutputRoot "outputs\mooccubex\relations_aug_v2_cgrc_recon_auxonly_e30" `
    -CheckpointRoot "checkpoints\mooccubex\relations_aug_v2_cgrc_recon_auxonly_e30" `
    -Protocol "strict_item_cold_balanced" `
    -ColdThresholds @(1) `
    -Seeds @(2025) `
    -Epochs $Epochs `
    -Patience $Patience `
    -EarlyStopAverageMode "item_macro" `
    -EarlyStopScoreMode "cold_only" `
    -UseContentDelta $false `
    -UseSageLite $false `
    -UseSageAuxLoss $false `
    -UseCgrcRecon $true `
    -CgrcReconAuxW 0.02 `
    -CgrcReconSampleW 0.00 `
    -CgrcReconPseudoRatio 0.30 `
    -CgrcReconTopK 64 `
    -CgrcReconTemp 0.50 `
    -CgrcReconOnlyColdOrTail $true `
    -CgrcReconTailPopRatio 0.10 `
    -CgrcReconDetachUser $false `
    -UseCourseFeedback $true `
    -UseCourseReward $true `
    -UsePrereqAux $true `
    -PrereqAuxOnlyCold $false `
    -PrereqGraphSource "concept" `
    -CourseFeedbackOnlyCold $false `
    -CoursePrereqW 0.08 `
    -CourseConceptW 0.04 `
    -CourseMatchMode "mean" `
    -CourseMatchTopK 5 `
    -CourseDiffW 0.03 `
    -CourseRedundantW 0.02 `
    -CourseRedundantMode "concept" `
    -CourseTermNorm "none" `
    -UseCourseSample $true `
    -CourseSampleOnlyCold $false `
    -CourseSampleBeta 0.20 `
    -UseCourseRerank $false `
    -MaskKnownPosNeg $true `
    -MaskSameItemNeg $true `
    -RunSampledEval $false `
    -SaveCkpt $true `
    -AutoResume $true `
    -ForceFresh $false `
    -SaveOptState $true
