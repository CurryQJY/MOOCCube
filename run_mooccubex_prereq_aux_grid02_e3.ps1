param(
    [string]$OutputRootBase = "outputs\mooccubex\tuning_prereq_aux_grid02_e3",
    [string]$CheckpointRootBase = "checkpoints\mooccubex\tuning_prereq_aux_grid02_e3",
    [int]$Seed = 2025
)

$ErrorActionPreference = "Stop"

$oldPrereqAuxWeight = $env:USIM_PREREQ_AUX_WEIGHT

try {
    foreach ($auxW in @(0.0, 0.02, 0.04, 0.06, 0.08)) {
        $tag = ("prereq_aux_{0:0.00}" -f [double]$auxW).Replace(".", "p")
        $out = Join-Path $OutputRootBase $tag
        $ckpt = Join-Path $CheckpointRootBase $tag

        New-Item -ItemType Directory -Force -Path $out | Out-Null
        New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

        $meta = [ordered]@{
            id = $tag
            group = "prereq_aux"
            param = "PrereqAuxWeight"
            value = $auxW
            seed = $Seed
            selection_metric = "validation full-ranking cold item-macro NDCG@10"
            tuned_config = [ordered]@{
                course_sample_beta = 0.20
                course_reward_scale = 1.0
                course_prereq_w = 0.08
                course_concept_w = 0.04
                course_diff_w = 0.03
                course_redundant_w = 0.02
                prereq_aux_weight = $auxW
            }
        }
        $meta | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $out "tuning_variant.json") -Encoding UTF8

        $env:USIM_PREREQ_AUX_WEIGHT = [string]$auxW

        $params = @{
            PythonRunner = ".\py.bat"
            DataDir = "processed_data_hin_x"
            RelationDir = "MOOCCubeX\relations"
            OutputRoot = $out
            CheckpointRoot = $ckpt
            Protocol = "strict_item_cold_balanced"
            ColdThresholds = @(1)
            Seeds = @($Seed)
            Epochs = 3
            Patience = 1
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
            CourseSampleBeta = 0.20
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

        Write-Host ""
        Write-Host ("===== Running {0} | PrereqAuxWeight={1} | seed={2} =====" -f $tag, $auxW, $Seed) -ForegroundColor Cyan
        & .\run_usim_feedback_fast3_content_delta_static.ps1 @params
    }
}
finally {
    if ($null -eq $oldPrereqAuxWeight) {
        Remove-Item Env:USIM_PREREQ_AUX_WEIGHT -ErrorAction SilentlyContinue
    } else {
        $env:USIM_PREREQ_AUX_WEIGHT = $oldPrereqAuxWeight
    }
}
