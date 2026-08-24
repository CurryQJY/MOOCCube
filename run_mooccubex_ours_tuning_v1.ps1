param(
    [string]$PythonRunner = ".\py.bat",
    [string]$Runner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$DataDir = "processed_data_hin_x",
    [string]$RelationDir = "MOOCCubeX\relations",
    [string]$OutputRootBase = "outputs\mooccubex\tuning_v1",
    [string]$CheckpointRootBase = "checkpoints\mooccubex\tuning_v1",
    [int]$Seed = 2025,
    [int]$Epochs = 15,
    [int]$Patience = 5,
    [string[]]$Groups = @("default", "beta", "reward", "prereq_aux"),
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

function Test-GroupEnabled {
    param([string]$Name)
    return (($Groups -contains "all") -or ($Groups -contains $Name))
}

function Format-ValueTag {
    param([double]$Value)
    $text = ("{0:g}" -f $Value)
    return $text.Replace("-", "m").Replace(".", "p")
}

function New-Variant {
    param(
        [string]$Id,
        [string]$Group,
        [string]$Param,
        [double]$Value,
        [double]$CourseSampleBeta = 0.20,
        [double]$CourseRewardScale = 1.00,
        [double]$PrereqAuxWeight = 0.03
    )
    return [pscustomobject]@{
        Id = $Id
        Group = $Group
        Param = $Param
        Value = $Value
        CourseSampleBeta = $CourseSampleBeta
        CourseRewardScale = $CourseRewardScale
        PrereqAuxWeight = $PrereqAuxWeight
        CoursePrereqW = 0.08 * $CourseRewardScale
        CourseConceptW = 0.04 * $CourseRewardScale
        CourseDiffW = 0.03 * $CourseRewardScale
        CourseRedundantW = 0.02 * $CourseRewardScale
    }
}

$variants = New-Object System.Collections.Generic.List[object]
if (Test-GroupEnabled "default") {
    $variants.Add((New-Variant -Id "default" -Group "default" -Param "default" -Value 1.0))
}
if (Test-GroupEnabled "beta") {
    foreach ($v in @(0.0, 0.1, 0.5, 1.0)) {
        $variants.Add((New-Variant -Id ("beta_" + (Format-ValueTag $v)) -Group "beta" -Param "CourseSampleBeta" -Value $v -CourseSampleBeta $v))
    }
}
if (Test-GroupEnabled "reward") {
    foreach ($v in @(0.0, 0.5, 2.0)) {
        $variants.Add((New-Variant -Id ("reward_" + (Format-ValueTag $v)) -Group "reward" -Param "CourseRewardScale" -Value $v -CourseRewardScale $v))
    }
}
if (Test-GroupEnabled "prereq_aux") {
    foreach ($v in @(0.0, 0.01, 0.06)) {
        $variants.Add((New-Variant -Id ("prereq_aux_" + (Format-ValueTag $v)) -Group "prereq_aux" -Param "PrereqAuxWeight" -Value $v -PrereqAuxWeight $v))
    }
}

if ($variants.Count -lt 1) {
    throw "No tuning variants selected. Use -Groups default,beta,reward,prereq_aux or -Groups all."
}

New-Item -ItemType Directory -Force -Path $OutputRootBase | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRootBase | Out-Null

$oldPrereqAuxWeight = $env:USIM_PREREQ_AUX_WEIGHT

try {
    foreach ($variant in $variants) {
        $variantOutputRoot = Join-Path $OutputRootBase $variant.Id
        $variantCheckpointRoot = Join-Path $CheckpointRootBase $variant.Id
        New-Item -ItemType Directory -Force -Path $variantOutputRoot | Out-Null
        New-Item -ItemType Directory -Force -Path $variantCheckpointRoot | Out-Null

        $variantMeta = [ordered]@{
            id = $variant.Id
            group = $variant.Group
            param = $variant.Param
            value = $variant.Value
            seed = $Seed
            selection_metric = "validation full-ranking cold item-macro NDCG@10"
            data_dir = $DataDir
            protocol = "strict_item_cold_balanced"
            cold_threshold = 1
            epochs = $Epochs
            patience = $Patience
            fixed_config = [ordered]@{
                use_content_delta = $false
                use_pseudo_cold_train = $false
                use_paac = $false
                use_course_feedback = $true
                use_course_reward = $true
                use_prereq_aux = $true
                prereq_aux_only_cold = $false
                use_course_sample = $true
                course_feedback_only_cold = $false
                course_sample_only_cold = $false
                use_course_rerank = $false
                aux_weight = 0.3
                run_sampled_eval = $false
            }
            tuned_config = [ordered]@{
                course_sample_beta = $variant.CourseSampleBeta
                course_reward_scale = $variant.CourseRewardScale
                course_prereq_w = $variant.CoursePrereqW
                course_concept_w = $variant.CourseConceptW
                course_diff_w = $variant.CourseDiffW
                course_redundant_w = $variant.CourseRedundantW
                prereq_aux_weight = $variant.PrereqAuxWeight
            }
        }
        $variantMeta | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $variantOutputRoot "tuning_variant.json") -Encoding UTF8

        $env:USIM_PREREQ_AUX_WEIGHT = [string]$variant.PrereqAuxWeight

        Write-Host ""
        Write-Host ("===== TUNING {0}: {1}={2} | seed={3} =====" -f $variant.Id, $variant.Param, $variant.Value, $Seed) -ForegroundColor Cyan
        Write-Host ("OutputRoot: {0}" -f $variantOutputRoot)
        Write-Host ("CheckpointRoot: {0}" -f $variantCheckpointRoot)

        & $Runner `
            -PythonRunner $PythonRunner `
            -DataDir $DataDir `
            -RelationDir $RelationDir `
            -OutputRoot $variantOutputRoot `
            -CheckpointRoot $variantCheckpointRoot `
            -Protocol strict_item_cold_balanced `
            -ColdThresholds 1 `
            -Seeds $Seed `
            -Epochs $Epochs `
            -Patience $Patience `
            -EarlyStopAverageMode item_macro `
            -UseContentDelta $false `
            -UsePseudoColdTrain $false `
            -UsePaac $false `
            -UseCourseFeedback $true `
            -UseCourseReward $true `
            -UsePrereqAux $true `
            -PrereqAuxOnlyCold $false `
            -CoursePrereqW $variant.CoursePrereqW `
            -CourseConceptW $variant.CourseConceptW `
            -CourseDiffW $variant.CourseDiffW `
            -CourseRedundantW $variant.CourseRedundantW `
            -UseCourseSample $true `
            -CourseSampleBeta $variant.CourseSampleBeta `
            -CourseFeedbackOnlyCold $false `
            -CourseSampleOnlyCold $false `
            -UseCourseRerank $false `
            -AuxWeight 0.3 `
            -RunSampledEval $false `
            -SaveCkpt $true `
            -AutoResume $true `
            -ForceFresh $false `
            -SaveOptState $true `
            -SkipAggregate
    }
}
finally {
    if ($null -eq $oldPrereqAuxWeight) {
        Remove-Item Env:USIM_PREREQ_AUX_WEIGHT -ErrorAction SilentlyContinue
    } else {
        $env:USIM_PREREQ_AUX_WEIGHT = $oldPrereqAuxWeight
    }
}

if (-not $SkipAggregate) {
    Write-Host ""
    Write-Host "===== Summarizing MOOCCubeX tuning v1 =====" -ForegroundColor Cyan
    & $PythonRunner "summarize_mooccubex_tuning_v1.py" --root $OutputRootBase
    if ($LASTEXITCODE -ne 0) {
        throw "Tuning summary failed"
    }
}
