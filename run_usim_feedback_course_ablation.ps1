param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback.py",
    [string]$OutputRoot = "outputs\usim_feedback_course_ablation",
    [string]$CheckpointRoot = "checkpoints\usim_feedback_course_ablation",
    [string[]]$IncludeExperiments = @(),
    [string[]]$SkipExperiments = @()
)

$ErrorActionPreference = "Stop"

function Set-Or-ClearEnv {
    param(
        [string]$Name,
        [AllowEmptyString()][string]$Value
    )

    if ([string]::IsNullOrEmpty($Value)) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    } else {
        Set-Item "Env:$Name" $Value
    }
}

function New-Experiment {
    param(
        [string]$Name,
        [hashtable]$Env,
        [string]$Notes
    )

    return @{
        name = $Name
        env = $Env
        notes = $Notes
    }
}

$trackedEnv = @(
    "USIM_FB_FORCE_FRESH",
    "USIM_FB_AUTO_RESUME",
    "USIM_FB_OUTPUT_TAG",
    "USIM_FB_OUTPUT_DIR",
    "USIM_FB_CKPT_DIR",
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "USIM_USE_PREREQ_AUX_LOSS",
    "USIM_PREREQ_AUX_WEIGHT",
    "USIM_USE_STRUCTURED_HARD_NEG",
    "USIM_USE_COURSE_RERANK",
    "USIM_COURSE_RERANK_ALPHA",
    "USIM_COURSE_RERANK_LAMBDA",
    "USIM_COURSE_RERANK_MIN_SEEN",
    "USIM_COURSE_RERANK_TOPL",
    "USIM_COURSE_RERANK_PENALTY_CAP",
    "USIM_COURSE_RERANK_ONLY_COLD",
    "USIM_FB_COURSE_ONLY_COLD",
    "USIM_FB_COURSE_PREREQ_W",
    "USIM_FB_COURSE_CONCEPT_W",
    "USIM_FB_COURSE_DIFF_W",
    "USIM_FB_COURSE_REDUNDANT_W",
    "USIM_FB_COURSE_SAMPLE_BETA",
    "USIM_FB_COURSE_SAMPLE_ONLY_COLD",
    "USIM_FB_COURSE_SAMPLE_TOPK",
    "USIM_FB_COURSE_SAMPLE_TOPL",
    "USIM_STATIC"
)

$originalEnv = @{}
foreach ($name in $trackedEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$baseNoCourse = @{
    "USIM_FB_LOAD_COURSE_ARTIFACTS" = "0"
    "USIM_USE_PREREQ_AUX_LOSS" = "0"
    "USIM_USE_STRUCTURED_HARD_NEG" = "0"
    "USIM_USE_COURSE_RERANK" = "0"
    "USIM_FB_COURSE_PREREQ_W" = "0.0"
    "USIM_FB_COURSE_CONCEPT_W" = "0.0"
    "USIM_FB_COURSE_DIFF_W" = "0.0"
    "USIM_FB_COURSE_REDUNDANT_W" = "0.0"
    "USIM_FB_COURSE_SAMPLE_BETA" = "0.0"
}

$experiments = @(
    (New-Experiment "feedback_ref" $baseNoCourse "Reference feedback baseline with course-side signals disabled."),
    (New-Experiment "plus_prereq_aux" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_USE_PREREQ_AUX_LOSS" = "1"
    }) "Add only prerequisite auxiliary supervision."),
    (New-Experiment "plus_prereq_reward" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_PREREQ_W" = "0.08"
    }) "Add only prerequisite-gap reward."),
    (New-Experiment "plus_concept_reward" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_CONCEPT_W" = "0.04"
    }) "Add only concept-match reward."),
    (New-Experiment "plus_difficulty_reward" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_DIFF_W" = "0.03"
    }) "Add only difficulty-gap penalty."),
    (New-Experiment "plus_redundant_penalty" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_REDUNDANT_W" = "0.02"
    }) "Add only redundancy penalty."),
    (New-Experiment "plus_course_reward_all" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_PREREQ_W" = "0.08"
        "USIM_FB_COURSE_CONCEPT_W" = "0.04"
        "USIM_FB_COURSE_DIFF_W" = "0.03"
        "USIM_FB_COURSE_REDUNDANT_W" = "0.02"
    }) "Enable all course reward terms, no course sampling."),
    (New-Experiment "plus_course_sampling" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_FB_COURSE_SAMPLE_BETA" = "0.20"
        "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
        "USIM_FB_COURSE_SAMPLE_TOPL" = "32"
    }) "Enable only course-aware candidate sampling."),
    (New-Experiment "plus_course_rerank" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_USE_COURSE_RERANK" = "1"
        "USIM_COURSE_RERANK_ALPHA" = "0.03"
        "USIM_COURSE_RERANK_LAMBDA" = "0.015"
        "USIM_COURSE_RERANK_MIN_SEEN" = "8"
        "USIM_COURSE_RERANK_TOPL" = "15"
        "USIM_COURSE_RERANK_PENALTY_CAP" = "0.08"
        "USIM_COURSE_RERANK_ONLY_COLD" = "1"
    }) "Enable only inference-time course rerank."),
    (New-Experiment "plus_all_course" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_USE_PREREQ_AUX_LOSS" = "1"
        "USIM_FB_COURSE_PREREQ_W" = "0.08"
        "USIM_FB_COURSE_CONCEPT_W" = "0.04"
        "USIM_FB_COURSE_DIFF_W" = "0.03"
        "USIM_FB_COURSE_REDUNDANT_W" = "0.02"
        "USIM_FB_COURSE_SAMPLE_BETA" = "0.20"
        "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
        "USIM_FB_COURSE_SAMPLE_TOPL" = "32"
    }) "Enable prerequisite aux + all course reward + course sampling."),
    (New-Experiment "plus_all_course_rerank" (@{} + $baseNoCourse + @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_USE_PREREQ_AUX_LOSS" = "1"
        "USIM_FB_COURSE_PREREQ_W" = "0.08"
        "USIM_FB_COURSE_CONCEPT_W" = "0.04"
        "USIM_FB_COURSE_DIFF_W" = "0.03"
        "USIM_FB_COURSE_REDUNDANT_W" = "0.02"
        "USIM_FB_COURSE_SAMPLE_BETA" = "0.20"
        "USIM_FB_COURSE_SAMPLE_TOPK" = "32"
        "USIM_FB_COURSE_SAMPLE_TOPL" = "32"
        "USIM_USE_COURSE_RERANK" = "1"
        "USIM_COURSE_RERANK_ALPHA" = "0.03"
        "USIM_COURSE_RERANK_LAMBDA" = "0.015"
        "USIM_COURSE_RERANK_MIN_SEEN" = "8"
        "USIM_COURSE_RERANK_TOPL" = "15"
        "USIM_COURSE_RERANK_PENALTY_CAP" = "0.08"
        "USIM_COURSE_RERANK_ONLY_COLD" = "1"
    }) "Enable all current course-side signals, including rerank.")
)

if ($IncludeExperiments.Count -gt 0) {
    $includeSet = @{}
    foreach ($name in $IncludeExperiments) {
        $includeSet[$name] = $true
    }
    $experiments = @($experiments | Where-Object { $includeSet.ContainsKey([string]$_.name) })
}

if ($SkipExperiments.Count -gt 0) {
    $skipSet = @{}
    foreach ($name in $SkipExperiments) {
        $skipSet[$name] = $true
    }
    $experiments = @($experiments | Where-Object { -not $skipSet.ContainsKey([string]$_.name) })
}

if ($experiments.Count -lt 1) {
    throw "No experiments selected. Check -IncludeExperiments / -SkipExperiments."
}

try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

    $summaryRows = New-Object System.Collections.Generic.List[object]

    foreach ($exp in $experiments) {
        $tag = [string]$exp.name
        $outputDir = Join-Path $OutputRoot $tag
        $ckptDir = Join-Path $CheckpointRoot $tag
        $logPath = Join-Path $outputDir "run.log"
        $detailPath = Join-Path $outputDir "final_report_usim_feedback.csv"
        $fullrankPath = Join-Path $outputDir "final_fullrank_usim_feedback.csv"

        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

        Set-Or-ClearEnv "USIM_FB_FORCE_FRESH" "1"
        Set-Or-ClearEnv "USIM_FB_AUTO_RESUME" "0"
        Set-Or-ClearEnv "USIM_STATIC" "0"
        Set-Or-ClearEnv "USIM_FB_OUTPUT_TAG" $tag
        Set-Or-ClearEnv "USIM_FB_OUTPUT_DIR" $outputDir
        Set-Or-ClearEnv "USIM_FB_CKPT_DIR" $ckptDir

        foreach ($name in $exp.env.Keys) {
            Set-Or-ClearEnv $name ([string]$exp.env[$name])
        }

        Write-Host ""
        Write-Host ("=" * 80)
        Write-Host ("Running experiment: {0}" -f $tag)
        Write-Host ("Notes: {0}" -f $exp.notes)
        Write-Host ("Output dir: {0}" -f $outputDir)
        Write-Host ("Checkpoint dir: {0}" -f $ckptDir)
        Write-Host ("=" * 80)

        if (Test-Path $detailPath) { Remove-Item $detailPath -Force }
        if (Test-Path $fullrankPath) { Remove-Item $fullrankPath -Force }

        $commandLine = ('"{0}" "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
        & cmd.exe /d /c $commandLine | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            Write-Warning ("Experiment failed: {0} (exit code {1})" -f $tag, $exitCode)
            $summaryRows.Add([pscustomobject]@{
                experiment = $tag
                status = "failed"
                exit_code = $exitCode
                notes = $exp.notes
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
            }) | Out-Null
            continue
        }

        if ((-not (Test-Path $detailPath)) -or (-not (Test-Path $fullrankPath))) {
            Write-Warning ("Experiment finished but report files are incomplete for {0}" -f $tag)
            $summaryRows.Add([pscustomobject]@{
                experiment = $tag
                status = "missing_reports"
                notes = $exp.notes
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
            }) | Out-Null
            continue
        }

        $detailRows = Import-Csv $detailPath
        $fullRow = Import-Csv $fullrankPath | Select-Object -First 1
        $detailByMetric = @{}
        foreach ($row in $detailRows) {
            $detailByMetric[[string]$row.metric] = $row
        }

        $summaryRows.Add([pscustomobject]@{
            experiment = $tag
            status = "ok"
            notes = $exp.notes
            sampled_cold_r5 = $detailByMetric["R@5"].sampled_cold
            sampled_cold_r10 = $detailByMetric["R@10"].sampled_cold
            sampled_cold_r20 = $detailByMetric["R@20"].sampled_cold
            sampled_cold_n5 = $detailByMetric["N@5"].sampled_cold
            sampled_cold_n10 = $detailByMetric["N@10"].sampled_cold
            sampled_cold_n20 = $detailByMetric["N@20"].sampled_cold
            sampled_hot_r5 = $detailByMetric["R@5"].sampled_hot
            sampled_hot_r10 = $detailByMetric["R@10"].sampled_hot
            sampled_hot_r20 = $detailByMetric["R@20"].sampled_hot
            sampled_hot_n5 = $detailByMetric["N@5"].sampled_hot
            sampled_hot_n10 = $detailByMetric["N@10"].sampled_hot
            sampled_hot_n20 = $detailByMetric["N@20"].sampled_hot
            full_cold_r5 = $fullRow.full_cold_r5
            full_cold_r10 = $fullRow.full_cold_r10
            full_cold_r20 = $fullRow.full_cold_r20
            full_cold_n5 = $fullRow.full_cold_n5
            full_cold_n10 = $fullRow.full_cold_n10
            full_cold_n20 = $fullRow.full_cold_n20
            full_hot_r5 = $fullRow.full_hot_r5
            full_hot_r10 = $fullRow.full_hot_r10
            full_hot_r20 = $fullRow.full_hot_r20
            full_hot_n5 = $fullRow.full_hot_n5
            full_hot_n10 = $fullRow.full_hot_n10
            full_hot_n20 = $fullRow.full_hot_n20
            sampled_cold_count = $fullRow.sampled_cold_count
            sampled_hot_count = $fullRow.sampled_hot_count
            full_cold_count = $fullRow.full_cold_count
            full_hot_count = $fullRow.full_hot_count
            output_dir = $outputDir
            checkpoint_dir = $ckptDir
        }) | Out-Null
    }

    $summaryPath = Join-Path $OutputRoot "summary_course_ablation.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    Write-Host ""
    Write-Host ("Saved summary: {0}" -f $summaryPath)
}
finally {
    foreach ($name in $trackedEnv) {
        Set-Or-ClearEnv $name $originalEnv[$name]
    }
}
