param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_standalone.py",
    [string]$OutputRoot = "outputs\usim_feedback_fast3_standalone_new_redundant_weight_sweep",
    [string]$CheckpointRoot = "checkpoints\usim_feedback_fast3_standalone_new_redundant_weight_sweep",
    [switch]$ForceCPU,
    [string[]]$Weights = @("0.005", "0.01", "0.015", "0.02"),
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

function Merge-Hashtables {
    param(
        [hashtable[]]$Tables
    )

    $merged = @{}
    foreach ($table in $Tables) {
        if ($null -eq $table) {
            continue
        }
        foreach ($key in $table.Keys) {
            $merged[$key] = $table[$key]
        }
    }
    return $merged
}

function Format-EnvHashtable {
    param([hashtable]$Env)

    if ($null -eq $Env -or $Env.Count -lt 1) {
        return ""
    }

    return (($Env.Keys | Sort-Object) | ForEach-Object {
        "{0}={1}" -f $_, [string]$Env[$_]
    }) -join "; "
}

$trackedEnv = @(
    "USIM_FB_SAVE_CKPT",
    "USIM_FB_SAVE_OPT_STATE",
    "USIM_FORCE_CPU",
    "USIM_FB_FORCE_FRESH",
    "USIM_FB_AUTO_RESUME",
    "USIM_FB_OUTPUT_TAG",
    "USIM_FB_OUTPUT_DIR",
    "USIM_FB_CKPT_DIR",
    "USIM_N_EPOCHS",
    "USIM_TRAIN_WINDOW",
    "USIM_PPO_EPOCHS",
    "USIM_PPO_LAMBDA",
    "USIM_PPO_VALUE_CLIP",
    "USIM_PPO_ADV_NORM",
    "USIM_FAST3_TGT_ALPHA_COLD",
    "USIM_FAST3_TGT_ALPHA_HOT",
    "USIM_FAST3_TGT_ALPHA_STEP",
    "USIM_FAST3_TGT_ALPHA_ENT",
    "USIM_FAST3_TGT_ALPHA_MIN",
    "USIM_FAST3_TGT_ALPHA_MAX",
    "USIM_FB_COURSE_ONLY_COLD",
    "USIM_FB_COURSE_SAMPLE_SOFT",
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "USIM_FB_COURSE_REDUNDANT_MODE",
    "USIM_FB_COURSE_STRUCT_VIDEO_MIN",
    "USIM_USE_PREREQ_AUX_LOSS",
    "USIM_USE_STRUCTURED_HARD_NEG",
    "USIM_USE_COURSE_RERANK",
    "USIM_FB_COURSE_PREREQ_W",
    "USIM_FB_COURSE_CONCEPT_W",
    "USIM_FB_COURSE_DIFF_W",
    "USIM_FB_COURSE_REDUNDANT_W",
    "USIM_FB_COURSE_SAMPLE_BETA",
    "USIM_FB_COURSE_SAMPLE_TOPK",
    "USIM_FB_COURSE_SAMPLE_TOPL",
    "USIM_STATIC"
)

$originalEnv = @{}
foreach ($name in $trackedEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$safeBackbone = @{
    "USIM_FB_SAVE_CKPT" = "0"
    "USIM_FB_SAVE_OPT_STATE" = "0"
    "USIM_N_EPOCHS" = "3"
    "USIM_TRAIN_WINDOW" = "24"
    "USIM_PPO_EPOCHS" = "2"
    "USIM_PPO_LAMBDA" = "0.95"
    "USIM_PPO_VALUE_CLIP" = "0.15"
    "USIM_PPO_ADV_NORM" = "1"
    "USIM_FAST3_TGT_ALPHA_COLD" = "0.28"
    "USIM_FAST3_TGT_ALPHA_HOT" = "0.58"
    "USIM_FAST3_TGT_ALPHA_STEP" = "0.15"
    "USIM_FAST3_TGT_ALPHA_ENT" = "0.12"
    "USIM_FAST3_TGT_ALPHA_MIN" = "0.12"
    "USIM_FAST3_TGT_ALPHA_MAX" = "0.78"
    "USIM_FB_COURSE_ONLY_COLD" = "1"
    "USIM_FB_COURSE_SAMPLE_SOFT" = "1"
}

if ($ForceCPU) {
    $safeBackbone["USIM_FORCE_CPU"] = "1"
}

$conceptOnlyBase = Merge-Hashtables @(
    $safeBackbone,
    @{
        "USIM_FB_LOAD_COURSE_ARTIFACTS" = "1"
        "USIM_USE_PREREQ_AUX_LOSS" = "0"
        "USIM_USE_STRUCTURED_HARD_NEG" = "0"
        "USIM_USE_COURSE_RERANK" = "0"
        "USIM_FB_COURSE_PREREQ_W" = "0.0"
        "USIM_FB_COURSE_CONCEPT_W" = "0.04"
        "USIM_FB_COURSE_DIFF_W" = "0.0"
        "USIM_FB_COURSE_REDUNDANT_W" = "0.0"
        "USIM_FB_COURSE_SAMPLE_BETA" = "0.0"
    }
)

$summaryEnvKeys = @(
    "USIM_FB_LOAD_COURSE_ARTIFACTS",
    "USIM_USE_PREREQ_AUX_LOSS",
    "USIM_USE_STRUCTURED_HARD_NEG",
    "USIM_USE_COURSE_RERANK",
    "USIM_FB_COURSE_PREREQ_W",
    "USIM_FB_COURSE_CONCEPT_W",
    "USIM_FB_COURSE_DIFF_W",
    "USIM_FB_COURSE_REDUNDANT_W",
    "USIM_FB_COURSE_REDUNDANT_MODE",
    "USIM_FB_COURSE_STRUCT_VIDEO_MIN",
    "USIM_FB_COURSE_SAMPLE_BETA",
    "USIM_FB_COURSE_SAMPLE_TOPK",
    "USIM_FB_COURSE_SAMPLE_TOPL",
    "USIM_N_EPOCHS",
    "USIM_TRAIN_WINDOW",
    "USIM_PPO_EPOCHS",
    "USIM_PPO_LAMBDA",
    "USIM_PPO_VALUE_CLIP",
    "USIM_PPO_ADV_NORM",
    "USIM_FAST3_TGT_ALPHA_COLD",
    "USIM_FAST3_TGT_ALPHA_HOT",
    "USIM_FAST3_TGT_ALPHA_STEP",
    "USIM_FAST3_TGT_ALPHA_ENT",
    "USIM_FAST3_TGT_ALPHA_MIN",
    "USIM_FAST3_TGT_ALPHA_MAX"
)

$experiments = New-Object System.Collections.Generic.List[object]
$experiments.Add((New-Experiment "concept_only_no_redundant" $conceptOnlyBase "Concept reward only baseline with redundant disabled.")) | Out-Null

foreach ($weight in $Weights) {
    $weightLabel = ([string]$weight).Replace(".", "p")
    $experiments.Add((New-Experiment ("concept_only_new_redundant_w_{0}" -f $weightLabel) (Merge-Hashtables @(
        $conceptOnlyBase,
        @{
            "USIM_FB_COURSE_REDUNDANT_W" = [string]$weight
            "USIM_FB_COURSE_REDUNDANT_MODE" = "video_family"
            "USIM_FB_COURSE_STRUCT_VIDEO_MIN" = "0.60"
        }
    )) ("Concept reward + new structural redundant with weight={0}." -f $weight))) | Out-Null
}

$experiments = @($experiments)

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
        $detailPath = Join-Path $outputDir "final_report_usim_feedback_fast3_standalone.csv"
        $fullrankPath = Join-Path $outputDir "final_fullrank_usim_feedback_fast3_standalone.csv"
        $configString = Format-EnvHashtable $exp.env
        $summaryConfig = @{}
        foreach ($name in $summaryEnvKeys) {
            if ($exp.env.ContainsKey($name)) {
                $summaryConfig[$name] = [string]$exp.env[$name]
            } else {
                $summaryConfig[$name] = ""
            }
        }

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

        $commandLine = ('"{0}" -u "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
        & cmd.exe /d /c $commandLine | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            Write-Warning ("Experiment failed: {0} (exit code {1})" -f $tag, $exitCode)
            $row = [ordered]@{
                experiment = $tag
                status = "failed"
                exit_code = $exitCode
                notes = $exp.notes
                config_overrides = $configString
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
            }
            foreach ($name in $summaryEnvKeys) {
                $row[$name] = $summaryConfig[$name]
            }
            $summaryRows.Add([pscustomobject]$row) | Out-Null
            continue
        }

        if ((-not (Test-Path $detailPath)) -or (-not (Test-Path $fullrankPath))) {
            Write-Warning ("Experiment finished but report files are incomplete for {0}" -f $tag)
            $row = [ordered]@{
                experiment = $tag
                status = "missing_reports"
                notes = $exp.notes
                config_overrides = $configString
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
            }
            foreach ($name in $summaryEnvKeys) {
                $row[$name] = $summaryConfig[$name]
            }
            $summaryRows.Add([pscustomobject]$row) | Out-Null
            continue
        }

        $detailRows = Import-Csv $detailPath
        $fullRow = Import-Csv $fullrankPath | Select-Object -First 1
        $detailByMetric = @{}
        foreach ($row in $detailRows) {
            $detailByMetric[[string]$row.metric] = $row
        }

        $row = [ordered]@{
            experiment = $tag
            status = "ok"
            notes = $exp.notes
            config_overrides = $configString
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
        }
        foreach ($name in $summaryEnvKeys) {
            $row[$name] = $summaryConfig[$name]
        }
        $summaryRows.Add([pscustomobject]$row) | Out-Null
    }

    $summaryPath = Join-Path $OutputRoot "summary_new_redundant_weight_sweep.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    Write-Host ""
    Write-Host ("Saved summary: {0}" -f $summaryPath)
}
finally {
    foreach ($name in $trackedEnv) {
        Set-Or-ClearEnv $name $originalEnv[$name]
    }
}
