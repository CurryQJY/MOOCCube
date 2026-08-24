param(
    [switch]$DryRun,
    [switch]$NoWait,
    [switch]$NoPruneLatest,
    [int[]]$WaitForPids = @(24572)
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$QueueName = "recppo_ablation_explore_serial_20260713"
$LogRoot = "outputs\recppo_research_repair\background_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Step {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

function Get-SeedTag {
    param([int]$Seed)
    return "strict_item_cold_balanced_thr1_seed_$Seed"
}

function Get-FinalCsvPath {
    param(
        [string]$OutputRoot,
        [int]$Seed
    )
    $tag = Get-SeedTag -Seed $Seed
    return (Join-Path (Join-Path $OutputRoot $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv")
}

function Get-LatestCheckpointPath {
    param(
        [string]$CheckpointRoot,
        [int]$Seed
    )
    $tag = Get-SeedTag -Seed $Seed
    return (Join-Path (Join-Path $CheckpointRoot $tag) "latest.pt")
}

function Invoke-WithEnv {
    param(
        [hashtable]$EnvVars,
        [scriptblock]$Body
    )
    $old = @{}
    foreach ($key in $EnvVars.Keys) {
        $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        Set-Item "Env:$key" ([string]$EnvVars[$key])
    }
    try {
        & $Body
    }
    finally {
        foreach ($key in $EnvVars.Keys) {
            if ($null -eq $old[$key]) {
                Remove-Item "Env:$key" -ErrorAction SilentlyContinue
            } else {
                Set-Item "Env:$key" $old[$key]
            }
        }
    }
}

function Wait-ForExistingWork {
    if ($NoWait) {
        Write-Step "NoWait set; not waiting for existing processes."
        return
    }
    foreach ($waitPid in $WaitForPids) {
        $proc = Get-Process -Id $waitPid -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            Write-Step "Waiting for existing process PID=$waitPid ($($proc.ProcessName)) before starting GPU queue."
            Wait-Process -Id $waitPid -ErrorAction SilentlyContinue
            Write-Step "Existing process PID=$waitPid finished."
        }
    }
}

function Invoke-Branch {
    param(
        [string]$Name,
        [int[]]$Seeds,
        [int]$Epochs,
        [int]$Patience,
        [double]$PpoLossWeight,
        [double]$ResidualScale,
        [string]$RolloutPolicy,
        [hashtable]$ExtraEnv = @{}
    )

    $outRoot = "outputs\recppo_research_repair\$Name"
    $ckptRoot = "checkpoints\recppo_research_repair\$Name"
    New-Item -ItemType Directory -Force -Path $outRoot, $ckptRoot | Out-Null

    Write-Step "Branch start: $Name | seeds=$($Seeds -join ',') | epochs=$Epochs | ppo_weight=$PpoLossWeight | residual=$ResidualScale | rollout=$RolloutPolicy"

    foreach ($seed in $Seeds) {
        $finalCsv = Get-FinalCsvPath -OutputRoot $outRoot -Seed $seed
        if (Test-Path $finalCsv) {
            Write-Step "Skip completed seed: branch=$Name seed=$seed final=$finalCsv"
            continue
        }

        $envVars = @{
            "USIM_RECPPO_WARMUP_EPOCHS" = "30"
            "USIM_RECPPO_EARLY_STOP_MODE" = "recppo_stage_guarded"
            "USIM_RECPPO_GUARD_HOT_RATIO" = "0.90"
            "USIM_RECPPO_STRICT_DETERMINISM" = "1"
            "PYTHONHASHSEED" = "0"
            "CUBLAS_WORKSPACE_CONFIG" = ":4096:8"
        }
        foreach ($key in $ExtraEnv.Keys) {
            $envVars[$key] = $ExtraEnv[$key]
        }

        if ($DryRun) {
            Write-Step "DryRun: would run branch=$Name seed=$seed output=$outRoot ckpt=$ckptRoot"
            continue
        }

        Write-Step "Run seed: branch=$Name seed=$seed"
        Invoke-WithEnv -EnvVars $envVars -Body {
            & .\run_usim_feedback_fast3_content_delta_static.ps1 `
                -PythonRunner ".\py.bat" `
                -ScriptPath "usim_feedback_fast3_content_delta_repaired.py" `
                -OutputRoot $outRoot `
                -CheckpointRoot $ckptRoot `
                -Protocol strict_item_cold_balanced `
                -ColdThresholds @(1) `
                -Seeds @($seed) `
                -Epochs $Epochs `
                -Patience $Patience `
                -EarlyStopScoreMode cold_only `
                -UseContentDelta $false `
                -UsePseudoColdTrain $true `
                -PseudoColdMode batch_tail `
                -PseudoColdRatio 0.3 `
                -PseudoColdMinPop 1 `
                -PpoLossWeight $PpoLossWeight `
                -RolloutPolicy $RolloutPolicy `
                -RlResidualScale $ResidualScale `
                -UsimSteps 5 `
                -UseCourseFeedback $true `
                -UseCourseReward $true `
                -UsePrereqAux $true `
                -UseCourseSample $true `
                -UseUsimRefinedEval $true `
                -SaveCkpt $true `
                -ForceFresh $false `
                -AutoResume $true `
                -SaveOptState $true `
                -SkipAggregate
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Runner failed: branch=$Name seed=$seed exit=$LASTEXITCODE"
        }
        if (-not (Test-Path $finalCsv)) {
            throw "Missing final CSV after run: branch=$Name seed=$seed path=$finalCsv"
        }

        if (-not $NoPruneLatest) {
            $latest = Get-LatestCheckpointPath -CheckpointRoot $ckptRoot -Seed $seed
            if (Test-Path $latest) {
                Remove-Item -LiteralPath $latest -Force
                Write-Step "Pruned completed latest checkpoint: $latest"
            }
        }
    }

    if (-not $DryRun) {
        Write-Step "Aggregate branch: $Name"
        & .\py.bat "aggregate_fast3_static_results.py" --root $outRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed: branch=$Name exit=$LASTEXITCODE"
        }
    }

    Write-Step "Branch done: $Name"
}

Write-Step "Queue start: $QueueName"
Wait-ForExistingWork

$formalSeeds = @(2025, 2026, 2027)
$exploreSeeds = @(2026, 2027)

Invoke-Branch `
    -Name "ablation_warmup_only_w000_res000_seeds2025_2027" `
    -Seeds $formalSeeds `
    -Epochs 30 `
    -Patience 30 `
    -PpoLossWeight 0.0 `
    -ResidualScale 0.0 `
    -RolloutPolicy "ppo"

Invoke-Branch `
    -Name "ablation_recppo_zero_lr_w050_res004_seeds2025_2027" `
    -Seeds $formalSeeds `
    -Epochs 35 `
    -Patience 5 `
    -PpoLossWeight 0.5 `
    -ResidualScale 0.04 `
    -RolloutPolicy "ppo" `
    -ExtraEnv @{
        "USIM_RECPPO_ACTOR_LR" = "0"
        "USIM_RECPPO_CRITIC_LR" = "0"
    }

Invoke-Branch `
    -Name "explore_w050_res008_seeds2026_2027" `
    -Seeds $exploreSeeds `
    -Epochs 35 `
    -Patience 5 `
    -PpoLossWeight 0.5 `
    -ResidualScale 0.08 `
    -RolloutPolicy "ppo"

Invoke-Branch `
    -Name "explore_w050_res010_seeds2026_2027" `
    -Seeds $exploreSeeds `
    -Epochs 35 `
    -Patience 5 `
    -PpoLossWeight 0.5 `
    -ResidualScale 0.10 `
    -RolloutPolicy "ppo"

Write-Step "Queue finished: $QueueName"
