param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback.py",
    [string]$OutputRoot = "outputs\usim_feedback_ablation",
    [string]$CheckpointRoot = "checkpoints\usim_feedback_ablation",
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

$trackedEnv = @(
    "USIM_FB_FORCE_FRESH",
    "USIM_FB_AUTO_RESUME",
    "USIM_FB_OUTPUT_TAG",
    "USIM_FB_OUTPUT_DIR",
    "USIM_FB_CKPT_DIR",
    "USIM_ABL_FAST3_TARGET_MIX",
    "USIM_ABL_FAST3_SOFT_SAMPLING",
    "USIM_ABL_FAST3_PPO",
    "USIM_ABL_TRAIN_WINDOW",
    "USIM_PPO_EPOCHS",
    "USIM_STATIC"
)

$originalEnv = @{}
foreach ($name in $trackedEnv) {
    $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$experiments = @(
    @{
        name = "current"
        target_mix = "0"
        soft_sampling = "0"
        fast3_ppo = "0"
        train_window = "0"
        ppo_epochs = ""
    },
    @{
        name = "mix"
        target_mix = "1"
        soft_sampling = "0"
        fast3_ppo = "0"
        train_window = "0"
        ppo_epochs = ""
    },
    @{
        name = "soft"
        target_mix = "0"
        soft_sampling = "1"
        fast3_ppo = "0"
        train_window = "0"
        ppo_epochs = ""
    },
    @{
        name = "ppo"
        target_mix = "0"
        soft_sampling = "0"
        fast3_ppo = "1"
        train_window = "0"
        ppo_epochs = "2"
    },
    @{
        name = "mix_soft"
        target_mix = "1"
        soft_sampling = "1"
        fast3_ppo = "0"
        train_window = "0"
        ppo_epochs = ""
    },
    @{
        name = "mix_soft_ppo"
        target_mix = "1"
        soft_sampling = "1"
        fast3_ppo = "1"
        train_window = "0"
        ppo_epochs = "2"
    },
    @{
        name = "mix_soft_ppo_win24"
        target_mix = "1"
        soft_sampling = "1"
        fast3_ppo = "1"
        train_window = "24"
        ppo_epochs = "2"
    }
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
        $fullrankPath = Join-Path $outputDir "final_fullrank_usim_feedback.csv"

        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

        Set-Or-ClearEnv "USIM_FB_FORCE_FRESH" "1"
        Set-Or-ClearEnv "USIM_FB_AUTO_RESUME" "0"
        Set-Or-ClearEnv "USIM_STATIC" "0"
        Set-Or-ClearEnv "USIM_FB_OUTPUT_TAG" $tag
        Set-Or-ClearEnv "USIM_FB_OUTPUT_DIR" $outputDir
        Set-Or-ClearEnv "USIM_FB_CKPT_DIR" $ckptDir
        Set-Or-ClearEnv "USIM_ABL_FAST3_TARGET_MIX" ([string]$exp.target_mix)
        Set-Or-ClearEnv "USIM_ABL_FAST3_SOFT_SAMPLING" ([string]$exp.soft_sampling)
        Set-Or-ClearEnv "USIM_ABL_FAST3_PPO" ([string]$exp.fast3_ppo)
        Set-Or-ClearEnv "USIM_ABL_TRAIN_WINDOW" ([string]$exp.train_window)
        Set-Or-ClearEnv "USIM_PPO_EPOCHS" ([string]$exp.ppo_epochs)

        Write-Host ""
        Write-Host ("=" * 72)
        Write-Host ("Running experiment: {0}" -f $tag)
        Write-Host ("Output dir: {0}" -f $outputDir)
        Write-Host ("Checkpoint dir: {0}" -f $ckptDir)
        Write-Host ("Flags: mix={0} soft={1} ppo={2} window={3}" -f $exp.target_mix, $exp.soft_sampling, $exp.fast3_ppo, $exp.train_window)
        Write-Host ("=" * 72)

        if (Test-Path $fullrankPath) {
            Remove-Item $fullrankPath -Force
        }

        $commandLine = ('"{0}" "{1}" 2>&1' -f $PythonRunner, $ScriptPath)
        & cmd.exe /d /c $commandLine | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            Write-Warning ("Experiment failed: {0} (exit code {1})" -f $tag, $exitCode)
            $summaryRows.Add([pscustomobject]@{
                experiment = $tag
                status = "failed"
                exit_code = $exitCode
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
                fullrank_file = $fullrankPath
            }) | Out-Null
            continue
        }

        if (Test-Path $fullrankPath) {
            $row = Import-Csv $fullrankPath | Select-Object -First 1
            if ($null -ne $row) {
                $summaryRows.Add([pscustomobject]@{
                    experiment = $tag
                    status = "ok"
                    protocol = $row.protocol
                    full_cold_r5 = $row.full_cold_r5
                    full_cold_r10 = $row.full_cold_r10
                    full_cold_r20 = $row.full_cold_r20
                    full_cold_n5 = $row.full_cold_n5
                    full_cold_n10 = $row.full_cold_n10
                    full_cold_n20 = $row.full_cold_n20
                    full_hot_r5 = $row.full_hot_r5
                    full_hot_r10 = $row.full_hot_r10
                    full_hot_r20 = $row.full_hot_r20
                    full_hot_n5 = $row.full_hot_n5
                    full_hot_n10 = $row.full_hot_n10
                    full_hot_n20 = $row.full_hot_n20
                    sampled_cold_count = $row.sampled_cold_count
                    sampled_hot_count = $row.sampled_hot_count
                    full_cold_count = $row.full_cold_count
                    full_hot_count = $row.full_hot_count
                    output_dir = $outputDir
                    checkpoint_dir = $ckptDir
                    fullrank_file = $fullrankPath
                }) | Out-Null
            }
        } else {
            Write-Warning ("Experiment finished but no fullrank file found: {0}" -f $fullrankPath)
            $summaryRows.Add([pscustomobject]@{
                experiment = $tag
                status = "missing_fullrank"
                output_dir = $outputDir
                checkpoint_dir = $ckptDir
                fullrank_file = $fullrankPath
            }) | Out-Null
        }
    }

    $summaryPath = Join-Path $OutputRoot "summary_fullrank.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    Write-Host ""
    Write-Host ("Saved summary: {0}" -f $summaryPath)
}
finally {
    foreach ($name in $trackedEnv) {
        Set-Or-ClearEnv $name $originalEnv[$name]
    }
}
