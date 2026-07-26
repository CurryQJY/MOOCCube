param(
    [string]$OutputRootBase = "outputs\mooccubex\tuning_beta_0p4_e15",
    [string]$CheckpointRootBase = "checkpoints\mooccubex\tuning_beta_0p4_e15",
    [int]$Seed = 2025
)

$ErrorActionPreference = "Stop"

$beta = 0.40
$oldPrereqAuxWeight = $env:USIM_PREREQ_AUX_WEIGHT
$env:USIM_PREREQ_AUX_WEIGHT = "0.03"

try {
    $tag = "beta_0p4"
    $out = Join-Path $OutputRootBase $tag
    $ckpt = Join-Path $CheckpointRootBase $tag

    New-Item -ItemType Directory -Force -Path $out | Out-Null
    New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

    $meta = [ordered]@{
        id = $tag
        group = "beta"
        param = "CourseSampleBeta"
        value = $beta
        seed = $Seed
        selection_metric = "validation full-ranking cold item-macro NDCG@10"
        tuned_config = [ordered]@{
            course_sample_beta = $beta
            course_reward_scale = 1.0
            course_prereq_w = 0.08
            course_concept_w = 0.04
            course_diff_w = 0.03
            course_redundant_w = 0.02
            prereq_aux_weight = 0.03
        }
    }
    $meta | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $out "tuning_variant.json") -Encoding UTF8

    $params = @{
        PythonRunner = ".\py.bat"
        DataDir = "processed_data_hin_x"
        RelationDir = "MOOCCubeX\relations"
        OutputRoot = $out
        CheckpointRoot = $ckpt
        Protocol = "strict_item_cold_balanced"
        ColdThresholds = @(1)
        Seeds = @($Seed)
        Epochs = 15
        Patience = 5
        EarlyStopAverageMode = "item_macro"
        UseContentDelta = $false
        UsePseudoColdTrain = $false
        UsePaac = $false
        UseCourseFeedback = $true
        UseCourseReward = $true
        UsePrereqAux = $true
        PrereqAuxOnlyCold = $false
        CoursePrereqW = 0.08
        CourseConceptW = 0.04
        CourseDiffW = 0.03
        CourseRedundantW = 0.02
        UseCourseSample = $true
        CourseSampleBeta = $beta
        CourseFeedbackOnlyCold = $false
        CourseSampleOnlyCold = $false
        UseCourseRerank = $false
        AuxWeight = 0.3
        RunSampledEval = $false
        SaveCkpt = $true
        AutoResume = $true
        ForceFresh = $false
        SaveOptState = $true
        SkipAggregate = $true
    }

    Write-Host ("===== Running {0} E15 | only CourseSampleBeta={1} changed | seed={2} =====" -f $tag, $beta, $Seed) -ForegroundColor Cyan
    & .\run_usim_feedback_fast3_content_delta_static.ps1 @params
}
finally {
    if ($null -eq $oldPrereqAuxWeight) {
        Remove-Item Env:USIM_PREREQ_AUX_WEIGHT -ErrorAction SilentlyContinue
    } else {
        $env:USIM_PREREQ_AUX_WEIGHT = $oldPrereqAuxWeight
    }
}
