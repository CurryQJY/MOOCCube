param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = ".\usim_feedback_fast3_content_delta.py",
    [string]$DataDir = "processed_data_hin_clean_pop5",
    [string]$RelationDir = "MOOCCube/relations",
    [string]$OutputRootBase = "outputs\significance_per_item_exports\mooccube\aaai27_rl_component_gate_v1",
    [string]$CheckpointRootBase = "checkpoints\significance_per_item_exports\mooccube\aaai27_rl_component_gate_v1",
    [int]$PilotSeed = 2025,
    [int[]]$RemainingSeeds = @(2026, 2027),
    [string]$VariantList = "",
    [string]$GateMetric = "full_cold_item_macro_n10",
    [double]$GateMinDrop = 0.0001,
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [int]$MaxAttempts = 3,
    [int]$RetryDelaySeconds = 120,
    [int]$MinFreeGpuMiB = 9000,
    [int]$GpuPollSeconds = 300,
    [switch]$SkipGpuWait,
    [switch]$PreflightOnly,
    [switch]$StartupProbe,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoPath $PathValue))
}

$pythonRunnerAbs = Resolve-RepoPath $PythonRunner
$staticRunnerSourceAbs = Resolve-RepoPath $StaticRunner
$scriptSourceAbs = Resolve-RepoPath $ScriptPath
$outputRootAbs = Resolve-RepoPath $OutputRootBase
$checkpointRootAbs = Resolve-RepoPath $CheckpointRootBase
$snapshotRootAbs = Join-Path $outputRootAbs "_source_snapshot"
$snapshotScriptAbs = Join-Path $snapshotRootAbs "usim_feedback_fast3_content_delta.py"
$snapshotRunnerAbs = Join-Path $snapshotRootAbs "run_usim_feedback_fast3_content_delta_static.ps1"
$snapshotAggregateAbs = Join-Path $snapshotRootAbs "aggregate_fast3_static_results.py"
$queueLog = Join-Path $outputRootAbs "queue.log"
$decisionCsv = Join-Path $outputRootAbs "component_gate_decisions.csv"
$existingAuditCsv = Join-Path $outputRootAbs "existing_result_audit.csv"
$stateJson = Join-Path $outputRootAbs "queue_state.json"
$completionJson = Join-Path $outputRootAbs "queue_completion.json"
$pidFile = Join-Path $outputRootAbs "queue.pid"
$sourceHashCsv = Join-Path $snapshotRootAbs "source_hashes.csv"

function Write-QueueLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
        $payload = $line + [Environment]::NewLine
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            try {
                [System.IO.File]::AppendAllText($queueLog, $payload, [System.Text.Encoding]::UTF8)
                break
            } catch {
                if ($attempt -eq 5) { throw }
                Start-Sleep -Milliseconds (200 * $attempt)
            }
        }
    }
    Write-Host $line
}

function Write-State {
    param(
        [string]$Phase,
        [string]$Variant = "",
        [int]$Seed = 0,
        [string]$Message = ""
    )
    if ($DryRun) { return }
    [ordered]@{
        updated_at = (Get-Date).ToString("o")
        phase = $Phase
        variant = $Variant
        seed = $Seed
        message = $Message
        pid = $PID
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $stateJson -Encoding UTF8
}

function Format-ListValue {
    param([object[]]$Values)
    return (($Values | ForEach-Object { [string]$_ }) -join ",")
}

function Convert-ToBool {
    param([object]$Value)
    if ($Value -is [bool]) { return [bool]$Value }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    return $text -in @("1", "true", "yes", "on")
}

function Merge-Params {
    param(
        [hashtable]$Base,
        [hashtable]$Override
    )
    $merged = @{}
    foreach ($key in $Base.Keys) { $merged[$key] = $Base[$key] }
    foreach ($key in $Override.Keys) { $merged[$key] = $Override[$key] }
    return $merged
}

function New-Variant {
    param(
        [string]$Name,
        [string]$Label,
        [string]$Rationale,
        [hashtable]$Params,
        [string[]]$CandidateRoots
    )
    return [ordered]@{
        Name = $Name
        Label = $Label
        Rationale = $Rationale
        Params = $Params
        CandidateRoots = $CandidateRoots
    }
}

$fullDefinition = New-Variant `
    -Name "full_reference" `
    -Label "Full CKG-RL reference" `
    -Rationale "Frozen current method code under the exact AAAI main-table protocol and training configuration." `
    -Params @{
        UsimSteps = 5
        PpoLossWeight = 1.0
        RolloutPolicy = "ppo"
        RlResidualScale = 1.0
    } `
    -CandidateRoots @(
        "outputs\significance_per_item_exports\mooccube\ckg_rl_full",
        "outputs\significance_per_item_exports\mooccube\ckg_rl_full_clean_maskff_e60"
    )

$variants = @(
    (New-Variant `
        -Name "wo_ppo_learning" `
        -Label "w/o PPO learning" `
        -Rationale "Keep the five-step PPO-policy rollout architecture but set the PPO objective weight to zero, isolating learned actor-critic updates." `
        -Params @{
            UsimSteps = 5
            PpoLossWeight = 0.0
            RolloutPolicy = "ppo"
            RlResidualScale = 1.0
        } `
        -CandidateRoots @(
            "outputs\content_delta_pop5\course_ppo_ablation_e60_3seed\wo_ppo_loss",
            "outputs\paper_supervised_ckg_validation\mooccube\no_ppo_rollout"
        )),
    (New-Variant `
        -Name "greedy_similarity_policy" `
        -Label "Greedy-similarity rollout" `
        -Rationale "Replace the learned PPO actor with deterministic nearest-user selection while retaining retrieval, five state updates, and course signals." `
        -Params @{
            UsimSteps = 5
            PpoLossWeight = 0.0
            RolloutPolicy = "greedy_similarity"
            RlResidualScale = 1.0
        } `
        -CandidateRoots @(
            "outputs\content_delta_pop5\course_policy_ablation_e60_3seed\greedy_similarity_policy"
        )),
    (New-Variant `
        -Name "course_fit_policy" `
        -Label "Course-fit heuristic rollout" `
        -Rationale "Replace PPO with the strongest implemented non-learned educational-fit action rule while keeping the same simulator horizon and state update." `
        -Params @{
            UsimSteps = 5
            PpoLossWeight = 0.0
            RolloutPolicy = "course_fit"
            RlResidualScale = 1.0
        } `
        -CandidateRoots @(
            "outputs\content_delta_pop5\course_policy_ablation_e60_3seed\course_fit_policy"
        )),
    (New-Variant `
        -Name "wo_simulator_rollout" `
        -Label "w/o simulator rollout" `
        -Rationale "Set the simulator horizon to zero and remove PPO learning so ranking uses the initial content-anchored, cold-masked course representation." `
        -Params @{
            UsimSteps = 0
            PpoLossWeight = 0.0
            RolloutPolicy = "ppo"
            RlResidualScale = 1.0
        } `
        -CandidateRoots @(
            "outputs\content_delta_pop5\course_core_ablation_e60_3seed\wo_simulator_t0",
            "outputs\paper_supervised_ckg_validation\mooccube\ckg_sup_t0"
        ))
)

if ($VariantList.Trim().Length -gt 0) {
    $wanted = @(
        $VariantList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Trim() }
    )
    $variants = @($variants | Where-Object { $wanted -contains $_.Name })
    if ($variants.Count -lt 1) {
        throw "No variants matched VariantList='$VariantList'"
    }
}

function Initialize-SourceSnapshot {
    if ($DryRun) {
        return [ordered]@{
            ScriptPath = $scriptSourceAbs
            RunnerPath = $staticRunnerSourceAbs
            AggregatePath = (Resolve-RepoPath ".\aggregate_fast3_static_results.py")
            ScriptHash = (Get-FileHash -LiteralPath $scriptSourceAbs -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

    New-Item -ItemType Directory -Force -Path $snapshotRootAbs | Out-Null
    $snapshotPackage = Join-Path $snapshotRootAbs "fast3_delta"
    if (-not (Test-Path -LiteralPath $snapshotScriptAbs)) {
        Copy-Item -LiteralPath $scriptSourceAbs -Destination $snapshotScriptAbs
    }
    if (-not (Test-Path -LiteralPath $snapshotRunnerAbs)) {
        Copy-Item -LiteralPath $staticRunnerSourceAbs -Destination $snapshotRunnerAbs
    }
    if (-not (Test-Path -LiteralPath $snapshotAggregateAbs)) {
        Copy-Item -LiteralPath (Resolve-RepoPath ".\aggregate_fast3_static_results.py") -Destination $snapshotAggregateAbs
    }
    if (-not (Test-Path -LiteralPath $snapshotPackage)) {
        Copy-Item -LiteralPath (Resolve-RepoPath ".\fast3_delta") -Destination $snapshotPackage -Recurse
    }

    $required = @($snapshotScriptAbs, $snapshotRunnerAbs, $snapshotAggregateAbs, (Join-Path $snapshotPackage "config.py"))
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Incomplete frozen source snapshot: missing $path"
        }
    }

    $snapshotPrefix = $snapshotRootAbs.TrimEnd("\") + "\"
    $hashRows = @(
        Get-ChildItem -LiteralPath $snapshotRootAbs -Recurse -File |
            Where-Object { $_.Name -ne "source_hashes.csv" } |
            Sort-Object FullName |
            ForEach-Object {
                [pscustomobject]@{
                    relative_path = $_.FullName.Substring($snapshotPrefix.Length)
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
    $hashRows | Export-Csv -LiteralPath $sourceHashCsv -NoTypeInformation -Encoding UTF8

    return [ordered]@{
        ScriptPath = $snapshotScriptAbs
        RunnerPath = $snapshotRunnerAbs
        AggregatePath = $snapshotAggregateAbs
        ScriptHash = (Get-FileHash -LiteralPath $snapshotScriptAbs -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$snapshot = Initialize-SourceSnapshot
$frozenScriptAbs = [string]$snapshot.ScriptPath
$frozenRunnerAbs = [string]$snapshot.RunnerPath
$aggregateScriptAbs = [string]$snapshot.AggregatePath
$expectedScriptHash = [string]$snapshot.ScriptHash

$baseRunnerParams = @{
    PythonRunner = $pythonRunnerAbs
    ScriptPath = $frozenScriptAbs
    DataDir = $DataDir
    RelationDir = $RelationDir
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Epochs = $Epochs
    Patience = $Patience
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    PrereqGraphSource = "concept"
    CoursePrereqW = 0.08
    CoursePrereqGate = 0.20
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantConceptGate = 1.0
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    CourseSampleBeta = 0.20
    TrainForceCold = $true
    UseUsimRefinedEval = $true
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    RunSampledEval = $false
    UseSageLite = $false
    SageUseTwoExpert = $false
    SageTwoExpertScoreFusion = $false
    UseSageAuxLoss = $false
    UseCgrcRecon = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    SaveCkpt = $true
    AutoResume = $true
    ForceFresh = $false
    SaveOptState = $true
    SkipAggregate = $true
}

function Get-VariantOutputRoot {
    param([object]$Definition)
    return Join-Path $outputRootAbs ([string]$Definition.Name)
}

function Get-VariantCheckpointRoot {
    param([object]$Definition)
    return Join-Path $checkpointRootAbs ([string]$Definition.Name)
}

function Get-SplitRoot {
    param([string]$Root, [int]$Seed)
    return Join-Path $Root ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
}

function Get-FinalPath {
    param([string]$Root, [int]$Seed)
    return Join-Path (Get-SplitRoot -Root $Root -Seed $Seed) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Get-ManifestPath {
    param([string]$Root, [int]$Seed)
    return Join-Path (Get-SplitRoot -Root $Root -Seed $Seed) "static_protocol_manifest.json"
}

function Test-CompatibleResult {
    param(
        [string]$Root,
        [int]$Seed,
        [hashtable]$ExpectedParams,
        [ref]$Reason
    )
    $finalPath = Get-FinalPath -Root $Root -Seed $Seed
    $manifestPath = Get-ManifestPath -Root $Root -Seed $Seed
    if (-not (Test-Path -LiteralPath $finalPath)) {
        $Reason.Value = "missing final result"
        return $false
    }
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        $Reason.Value = "missing manifest"
        return $false
    }
    try {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    } catch {
        $Reason.Value = "manifest parse error: $($_.Exception.Message)"
        return $false
    }

    $checks = @(
        @("script sha256", ([string]$manifest.script.sha256).ToLowerInvariant(), $expectedScriptHash),
        @("split seed", [string]$manifest.split.seed, [string]$Seed),
        @("split mode", [string]$manifest.split.split_mode, "strict_item_cold_balanced"),
        @("epochs", [string]$manifest.model_config.n_epochs, [string]$Epochs),
        @("early-stop average", [string]$manifest.model_config.early_stop_average_mode, "item_macro"),
        @("early-stop score", [string]$manifest.model_config.early_stop_score_mode, "cold_only"),
        @("mask known positives", [string](Convert-ToBool $manifest.model_config.mask_known_pos_neg), [string]$true),
        @("mask same item", [string](Convert-ToBool $manifest.model_config.mask_same_item_neg), [string]$true),
        @("simulator steps", [string]$manifest.env.USIM_STEPS, [string]$ExpectedParams.UsimSteps),
        @("PPO loss weight", [string]$manifest.env.USIM_PPO_LOSS_WEIGHT, [string]$ExpectedParams.PpoLossWeight),
        @("rollout policy", [string]$manifest.env.USIM_ROLLOUT_POLICY, [string]$ExpectedParams.RolloutPolicy),
        @("RL residual scale", [string]$manifest.env.USIM_RL_RESIDUAL_SCALE, [string]$ExpectedParams.RlResidualScale),
        @("forced-cold training", [string](Convert-ToBool $manifest.env.USIM_TRAIN_FORCE_COLD), [string]$true),
        @("refined evaluation", [string](Convert-ToBool $manifest.env.USIM_USE_REFINED_EVAL), [string]$true)
    )
    foreach ($check in $checks) {
        $name = [string]$check[0]
        $actual = [string]$check[1]
        $expected = [string]$check[2]
        if (-not [string]::Equals($actual, $expected, [System.StringComparison]::OrdinalIgnoreCase)) {
            $Reason.Value = "$name mismatch: actual='$actual' expected='$expected'"
            return $false
        }
    }

    $Reason.Value = "compatible"
    return $true
}

function Get-CandidateRoots {
    param([object]$Definition)
    $roots = @((Get-VariantOutputRoot -Definition $Definition))
    foreach ($candidate in @($Definition.CandidateRoots)) {
        $roots += Resolve-RepoPath ([string]$candidate)
    }
    return @($roots | Select-Object -Unique)
}

function Resolve-CompatibleRoot {
    param(
        [object]$Definition,
        [int]$Seed,
        [hashtable]$ExpectedParams
    )
    foreach ($root in (Get-CandidateRoots -Definition $Definition)) {
        $reason = ""
        if (Test-CompatibleResult -Root $root -Seed $Seed -ExpectedParams $ExpectedParams -Reason ([ref]$reason)) {
            return $root
        }
    }
    return $null
}

function Audit-ExistingResults {
    $rows = @()
    foreach ($definition in @($fullDefinition) + @($variants)) {
        $expected = Merge-Params -Base $baseRunnerParams -Override $definition.Params
        foreach ($seed in @($PilotSeed) + @($RemainingSeeds)) {
            foreach ($root in (Get-CandidateRoots -Definition $definition)) {
                $reason = ""
                $compatible = Test-CompatibleResult -Root $root -Seed $seed -ExpectedParams $expected -Reason ([ref]$reason)
                $rows += [pscustomobject]@{
                    checked_at = (Get-Date).ToString("o")
                    variant = $definition.Name
                    seed = $seed
                    root = $root
                    compatible = $compatible
                    reason = $reason
                }
            }
        }
    }
    if (-not $DryRun) {
        $rows | Export-Csv -LiteralPath $existingAuditCsv -NoTypeInformation -Encoding UTF8
    }
    return $rows
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            return [int](([string]$raw).Trim().Split("`n")[0].Trim())
        }
    } catch {
        return $null
    }
    return $null
}

function Wait-ForGpu {
    param([string]$Reason)
    if ($SkipGpuWait -or $DryRun) {
        Write-QueueLine "GPU wait skipped before $Reason."
        return
    }
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -ne $free -and $free -ge $MinFreeGpuMiB) {
            Write-QueueLine "GPU ready before ${Reason}: free=${free}MiB >= ${MinFreeGpuMiB}MiB."
            return
        }
        Write-State -Phase "waiting_gpu" -Message "$Reason; free=${free}MiB"
        Write-QueueLine "GPU not ready before ${Reason}: free=${free}MiB; waiting ${GpuPollSeconds}s."
        Start-Sleep -Seconds $GpuPollSeconds
    }
}

function Assert-FrozenSnapshot {
    if ($DryRun) { return }
    $currentHash = (Get-FileHash -LiteralPath $frozenScriptAbs -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($currentHash -ne $expectedScriptHash) {
        throw "Frozen method snapshot changed during queue execution: $frozenScriptAbs"
    }
}

function Invoke-SeedRun {
    param(
        [object]$Definition,
        [int]$Seed
    )
    $expected = Merge-Params -Base $baseRunnerParams -Override $Definition.Params
    $compatibleRoot = Resolve-CompatibleRoot -Definition $Definition -Seed $Seed -ExpectedParams $expected
    if ($null -ne $compatibleRoot) {
        Write-QueueLine "REUSE variant=$($Definition.Name) seed=$Seed root=$compatibleRoot"
        return [ordered]@{ Success = $true; Root = $compatibleRoot; Reused = $true; Attempts = 0 }
    }

    $targetRoot = Get-VariantOutputRoot -Definition $Definition
    $targetCkpt = Get-VariantCheckpointRoot -Definition $Definition
    if ($DryRun) {
        Write-QueueLine "DRY_RUN variant=$($Definition.Name) seed=$Seed output=$targetRoot"
        return [ordered]@{ Success = $true; Root = $targetRoot; Reused = $false; Attempts = 0 }
    }

    New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $targetCkpt | Out-Null
    $seedLog = Join-Path $targetRoot ("seed_{0}.runner.log" -f $Seed)

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Assert-FrozenSnapshot
        Wait-ForGpu -Reason ("variant={0} seed={1} attempt={2}" -f $Definition.Name, $Seed, $attempt)
        Write-State -Phase "running" -Variant $Definition.Name -Seed $Seed -Message "attempt=$attempt/$MaxAttempts"
        Write-QueueLine "START variant=$($Definition.Name) seed=$Seed attempt=$attempt/$MaxAttempts"

        $runnerParams = Merge-Params -Base $expected -Override @{
            OutputRoot = $targetRoot
            CheckpointRoot = $targetCkpt
            Seeds = @($Seed)
        }

        $exitCode = -1
        try {
            & $frozenRunnerAbs @runnerParams *>&1 | Tee-Object -FilePath $seedLog -Append
            $exitCode = $LASTEXITCODE
        } catch {
            $message = "Runner exception: $($_.Exception.Message)"
            Add-Content -LiteralPath $seedLog -Value $message -Encoding UTF8
            Write-QueueLine "ERROR variant=$($Definition.Name) seed=$Seed attempt=$attempt | $message"
        }

        $reason = ""
        $compatible = Test-CompatibleResult -Root $targetRoot -Seed $Seed -ExpectedParams $expected -Reason ([ref]$reason)
        if ($compatible) {
            Write-QueueLine "SUCCESS variant=$($Definition.Name) seed=$Seed attempt=$attempt exit_code=$exitCode"
            return [ordered]@{ Success = $true; Root = $targetRoot; Reused = $false; Attempts = $attempt }
        }

        Write-QueueLine "RETRY_NEEDED variant=$($Definition.Name) seed=$Seed attempt=$attempt exit_code=$exitCode reason=$reason"
        if ($attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }

    Write-QueueLine "FAILED variant=$($Definition.Name) seed=$Seed after $MaxAttempts attempts; queue will continue."
    return [ordered]@{ Success = $false; Root = $targetRoot; Reused = $false; Attempts = $MaxAttempts }
}

function Get-MetricRow {
    param([string]$Root, [int]$Seed)
    $path = Get-FinalPath -Root $Root -Seed $Seed
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing result file: $path"
    }
    $rows = @(Import-Csv -LiteralPath $path)
    if ($rows.Count -lt 1) {
        throw "Empty result file: $path"
    }
    return $rows[0]
}

function Get-MetricValue {
    param([object]$Row, [string]$Metric)
    if (-not ($Row.PSObject.Properties.Name -contains $Metric)) {
        throw "Metric '$Metric' not present in result row."
    }
    return [double]$Row.$Metric
}

function Invoke-AggregateIfPresent {
    param([object]$Definition)
    if ($DryRun) { return }
    $root = Get-VariantOutputRoot -Definition $Definition
    $finals = @(Get-ChildItem -LiteralPath $root -Recurse -Filter "final_fullrank_usim_feedback_fast3_content_delta_static.csv" -File -ErrorAction SilentlyContinue)
    if ($finals.Count -lt 1) { return }
    Write-QueueLine "AGGREGATE variant=$($Definition.Name) finals=$($finals.Count)"
    & $pythonRunnerAbs $aggregateScriptAbs --root $root *>&1 | Tee-Object -FilePath (Join-Path $root "aggregate.log") -Append
    if ($LASTEXITCODE -ne 0) {
        Write-QueueLine "WARNING aggregate failed for variant=$($Definition.Name) exit_code=$LASTEXITCODE"
    }
}

function Acquire-QueueLock {
    if ($DryRun) { return }
    New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
    if (Test-Path -LiteralPath $pidFile) {
        $oldText = (Get-Content -Raw -LiteralPath $pidFile).Trim()
        $oldPid = 0
        if ([int]::TryParse($oldText, [ref]$oldPid) -and $oldPid -gt 0) {
            if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
                throw "Another queue process is active with PID $oldPid"
            }
        }
        Write-QueueLine "Removing stale PID lock: $oldText"
    }
    Set-Content -LiteralPath $pidFile -Value ([string]$PID) -Encoding ASCII
}

function Enable-SleepPrevention {
    if ($DryRun) { return }
    if (-not ("Aaai27PowerState" -as [type])) {
        Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Aaai27PowerState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    [void][Aaai27PowerState]::SetThreadExecutionState([uint32]2147483649)
}

function Disable-SleepPrevention {
    if ($DryRun) { return }
    if ("Aaai27PowerState" -as [type]) {
        [void][Aaai27PowerState]::SetThreadExecutionState([uint32]2147483648)
    }
}

$decisionRows = @()
$failures = @()

Write-QueueLine "PLAN frozen_script=$frozenScriptAbs sha256=$expectedScriptHash"
Write-QueueLine "PLAN pilot_seed=$PilotSeed remaining_seeds=$(Format-ListValue $RemainingSeeds) gate_metric=$GateMetric min_drop=$GateMinDrop"
foreach ($definition in @($fullDefinition) + @($variants)) {
    Write-QueueLine "PLAN variant=$($definition.Name) | $($definition.Rationale)"
}

$auditRows = Audit-ExistingResults
foreach ($row in @($auditRows | Where-Object { $_.compatible })) {
    Write-QueueLine "EXISTING_COMPATIBLE variant=$($row.variant) seed=$($row.seed) root=$($row.root)"
}

if ($DryRun) {
    Write-QueueLine "DRY_RUN complete; no experiment was launched."
    return
}

if ($PreflightOnly) {
    Write-QueueLine "PREFLIGHT complete; frozen snapshot and existing-result audit are ready."
    return
}

Acquire-QueueLock
Enable-SleepPrevention

try {
    Write-State -Phase "starting" -Message "queue initialized"
    Write-QueueLine "QUEUE_START pid=$PID"

    if ($StartupProbe) {
        Write-QueueLine "STARTUP_PROBE complete; lock and sleep-prevention lifecycle succeeded."
        return
    }

    $fullExpected = Merge-Params -Base $baseRunnerParams -Override $fullDefinition.Params
    $fullPilotResult = Invoke-SeedRun -Definition $fullDefinition -Seed $PilotSeed
    if (-not $fullPilotResult.Success) {
        throw "Full reference pilot seed failed after retries; component gates cannot be evaluated."
    }
    $fullPilotRow = Get-MetricRow -Root $fullPilotResult.Root -Seed $PilotSeed
    $fullGateValue = Get-MetricValue -Row $fullPilotRow -Metric $GateMetric
    Write-QueueLine ("FULL_PILOT metric={0} value={1:F8} root={2}" -f $GateMetric, $fullGateValue, $fullPilotResult.Root)

    $remainingFullEnsured = $false
    foreach ($definition in $variants) {
        $pilotResult = Invoke-SeedRun -Definition $definition -Seed $PilotSeed
        if (-not $pilotResult.Success) {
            $failures += "$($definition.Name):$PilotSeed"
            $decisionRows += [pscustomobject]@{
                variant = $definition.Name
                label = $definition.Label
                pilot_seed = $PilotSeed
                gate_metric = $GateMetric
                full_value = $fullGateValue
                ablation_value = $null
                drop_full_minus_ablation = $null
                effective = $false
                action = "pilot_failed_after_retries"
                lower_early_rank_metrics = $null
                result_root = $pilotResult.Root
                decided_at = (Get-Date).ToString("o")
            }
            $decisionRows | Export-Csv -LiteralPath $decisionCsv -NoTypeInformation -Encoding UTF8
            continue
        }

        $variantRow = Get-MetricRow -Root $pilotResult.Root -Seed $PilotSeed
        $variantGateValue = Get-MetricValue -Row $variantRow -Metric $GateMetric
        $drop = $fullGateValue - $variantGateValue
        $earlyMetrics = @(
            "full_cold_item_macro_r5",
            "full_cold_item_macro_r10",
            "full_cold_item_macro_n5",
            "full_cold_item_macro_n10"
        )
        $lowerCount = 0
        foreach ($metric in $earlyMetrics) {
            if ((Get-MetricValue -Row $variantRow -Metric $metric) -lt (Get-MetricValue -Row $fullPilotRow -Metric $metric)) {
                $lowerCount++
            }
        }
        $effective = $drop -ge $GateMinDrop
        $action = if ($effective) { "run_remaining_seeds" } else { "skip_remaining_seeds" }
        Write-QueueLine ("GATE variant={0} full={1:F8} ablation={2:F8} drop={3:F8} lower_early={4}/4 effective={5} action={6}" -f $definition.Name, $fullGateValue, $variantGateValue, $drop, $lowerCount, $effective, $action)

        $decisionRows += [pscustomobject]@{
            variant = $definition.Name
            label = $definition.Label
            pilot_seed = $PilotSeed
            gate_metric = $GateMetric
            full_value = $fullGateValue
            ablation_value = $variantGateValue
            drop_full_minus_ablation = $drop
            effective = $effective
            action = $action
            lower_early_rank_metrics = $lowerCount
            result_root = $pilotResult.Root
            decided_at = (Get-Date).ToString("o")
        }
        $decisionRows | Export-Csv -LiteralPath $decisionCsv -NoTypeInformation -Encoding UTF8

        if (-not $effective) {
            continue
        }

        if (-not $remainingFullEnsured) {
            foreach ($seed in $RemainingSeeds) {
                $result = Invoke-SeedRun -Definition $fullDefinition -Seed $seed
                if (-not $result.Success) {
                    $failures += "$($fullDefinition.Name):$seed"
                }
            }
            $remainingFullEnsured = $true
            Invoke-AggregateIfPresent -Definition $fullDefinition
        }

        foreach ($seed in $RemainingSeeds) {
            $result = Invoke-SeedRun -Definition $definition -Seed $seed
            if (-not $result.Success) {
                $failures += "$($definition.Name):$seed"
            }
        }
        Invoke-AggregateIfPresent -Definition $definition
    }

    Invoke-AggregateIfPresent -Definition $fullDefinition
    foreach ($definition in $variants) {
        Invoke-AggregateIfPresent -Definition $definition
    }

    $status = if ($failures.Count -eq 0) { "complete" } else { "complete_with_failures" }
    [ordered]@{
        completed_at = (Get-Date).ToString("o")
        status = $status
        pid = $PID
        frozen_script = $frozenScriptAbs
        frozen_script_sha256 = $expectedScriptHash
        pilot_seed = $PilotSeed
        remaining_seeds = $RemainingSeeds
        gate_metric = $GateMetric
        gate_min_drop = $GateMinDrop
        failures = $failures
        decision_csv = $decisionCsv
        existing_result_audit_csv = $existingAuditCsv
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $completionJson -Encoding UTF8

    Write-State -Phase $status -Message ("failures={0}" -f (Format-ListValue $failures))
    Write-QueueLine "QUEUE_END status=$status failures=$(Format-ListValue $failures)"
    if ($failures.Count -gt 0) {
        exit 2
    }
} catch {
    $message = $_.Exception.Message
    [ordered]@{
        completed_at = (Get-Date).ToString("o")
        status = "fatal_error"
        pid = $PID
        message = $message
        failures = $failures
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $completionJson -Encoding UTF8
    Write-State -Phase "fatal_error" -Message $message
    Write-QueueLine "FATAL $message"
    throw
} finally {
    Disable-SleepPrevention
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force
    }
}
