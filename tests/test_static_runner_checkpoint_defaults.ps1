$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing static FAST3 runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("fast3_ckpt_defaults_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $fakeRunner = Join-Path $tmpRoot "fake_py.cmd"
    $fakeScript = Join-Path $tmpRoot "fake_static_runner.py"
    Set-Content -Encoding ASCII -LiteralPath $fakeRunner -Value @(
        "@echo off",
        "echo USIM_FB_SAVE_CKPT=%USIM_FB_SAVE_CKPT%",
        "echo USIM_FB_AUTO_RESUME=%USIM_FB_AUTO_RESUME%",
        "echo USIM_FB_FORCE_FRESH=%USIM_FB_FORCE_FRESH%",
        "echo USIM_FB_SAVE_OPT_STATE=%USIM_FB_SAVE_OPT_STATE%",
        "echo USIM_FB_CKPT_DIR=%USIM_FB_CKPT_DIR%",
        "echo USIM_SAGE_GATE_MODE=%USIM_SAGE_GATE_MODE%",
        "echo USIM_SAGE_GATE_BUCKETS=%USIM_SAGE_GATE_BUCKETS%",
        "echo USIM_SAGE_GATE_HIDDEN=%USIM_SAGE_GATE_HIDDEN%",
        "echo USIM_SAGE_GATE_BUCKET_STRATEGY=%USIM_SAGE_GATE_BUCKET_STRATEGY%",
        "echo USIM_SAGE_TWO_EXPERT_SCORE_FUSION=%USIM_SAGE_TWO_EXPERT_SCORE_FUSION%",
        "echo USIM_TRAIN_FORCE_COLD=%USIM_TRAIN_FORCE_COLD%",
        "echo USIM_STEPS=%USIM_STEPS%",
        "echo USIM_PPO_LOSS_WEIGHT=%USIM_PPO_LOSS_WEIGHT%",
        "echo USIM_ROLLOUT_POLICY=%USIM_ROLLOUT_POLICY%",
        "exit /b 0"
    )
    Set-Content -Encoding ASCII -LiteralPath $fakeScript -Value @(
        "def run_static_experiment():",
        "    pass",
        "_static_split_df = None"
    )

    function Invoke-StaticRunnerDry([string]$Name, [hashtable]$ExtraParams) {
        $outputRoot = Join-Path $tmpRoot "outputs\$Name"
        $params = @{
            PythonRunner = $fakeRunner
            ScriptPath = $fakeScript
            OutputRoot = $outputRoot
            Protocol = "strict_item_cold_balanced"
            ColdThresholds = @(1)
            Seeds = @(2025)
            Epochs = 1
            Patience = 1
            RunSampledEval = $false
            SkipAggregate = $true
        }
        foreach ($key in $ExtraParams.Keys) {
            $params[$key] = $ExtraParams[$key]
        }
        $out = & $script @params *>&1
        return ($out -join "`n")
    }

    $explicitRoot = Invoke-StaticRunnerDry "explicit_root_default_save" @{
        CheckpointRoot = (Join-Path $tmpRoot "checkpoints\explicit_root_default_save")
    }
    if ($explicitRoot -notmatch "CheckpointRoot was provided without -SaveCkpt; enabling checkpoint saving by default") {
        throw "Expected explicit CheckpointRoot run to warn that checkpoint saving was enabled"
    }
    if ($explicitRoot -notmatch "Checkpoint: save=True") {
        throw "Expected explicit CheckpointRoot run to print Checkpoint: save=True"
    }
    if ($explicitRoot -notmatch "USIM_FB_SAVE_CKPT=1") {
        throw "Expected explicit CheckpointRoot run to export USIM_FB_SAVE_CKPT=1"
    }

    $explicitOff = Invoke-StaticRunnerDry "explicit_root_save_off" @{
        CheckpointRoot = (Join-Path $tmpRoot "checkpoints\explicit_root_save_off")
        SaveCkpt = $false
    }
    if ($explicitOff -match "enabling checkpoint saving by default") {
        throw "Explicit SaveCkpt=false should not be overridden"
    }
    if ($explicitOff -notmatch "Checkpoint: save=False") {
        throw "Expected explicit SaveCkpt=false run to print Checkpoint: save=False"
    }
    if ($explicitOff -notmatch "USIM_FB_SAVE_CKPT=0") {
        throw "Expected explicit SaveCkpt=false run to export USIM_FB_SAVE_CKPT=0"
    }

    $implicitRoot = Invoke-StaticRunnerDry "implicit_root_legacy_default" @{}
    if ($implicitRoot -match "enabling checkpoint saving by default") {
        throw "Implicit default CheckpointRoot should keep legacy no-save behavior"
    }
    if ($implicitRoot -notmatch "Checkpoint: save=False") {
        throw "Expected implicit default CheckpointRoot run to print Checkpoint: save=False"
    }
    if ($implicitRoot -notmatch "USIM_FB_SAVE_CKPT=0") {
        throw "Expected implicit default CheckpointRoot run to export USIM_FB_SAVE_CKPT=0"
    }

    $sageGate = Invoke-StaticRunnerDry "sage_gate_bucket_mlp" @{
        UseSageLite = $true
        SageGateMode = "bucket_mlp"
        SageGateBuckets = 13
        SageGateHidden = 17
        SageGateBucketStrategy = "paper"
        SageTwoExpertScoreFusion = $true
    }
    if ($sageGate -notmatch "SAGE-lite: enabled=True gate_mode=bucket_mlp") {
        throw "Expected static runner to print bucket_mlp SAGE gate mode"
    }
    if ($sageGate -notmatch "bucket_strategy=paper") {
        throw "Expected static runner to print paper bucket strategy"
    }
    if ($sageGate -notmatch "USIM_SAGE_GATE_MODE=bucket_mlp") {
        throw "Expected static runner to export USIM_SAGE_GATE_MODE=bucket_mlp"
    }
    if ($sageGate -notmatch "USIM_SAGE_GATE_BUCKETS=13") {
        throw "Expected static runner to export USIM_SAGE_GATE_BUCKETS=13"
    }
    if ($sageGate -notmatch "USIM_SAGE_GATE_HIDDEN=17") {
        throw "Expected static runner to export USIM_SAGE_GATE_HIDDEN=17"
    }
    if ($sageGate -notmatch "USIM_SAGE_GATE_BUCKET_STRATEGY=paper") {
        throw "Expected static runner to export USIM_SAGE_GATE_BUCKET_STRATEGY=paper"
    }
    if ($sageGate -notmatch "USIM_SAGE_TWO_EXPERT_SCORE_FUSION=1") {
        throw "Expected static runner to export USIM_SAGE_TWO_EXPERT_SCORE_FUSION=1"
    }

    $coreControls = Invoke-StaticRunnerDry "core_ablation_controls" @{
        TrainForceCold = $false
        UsimSteps = 0
    }
    if ($coreControls -notmatch "Core controls: train_force_cold=False usim_steps=0") {
        throw "Expected static runner to print core ablation control values"
    }
    if ($coreControls -notmatch "USIM_TRAIN_FORCE_COLD=0") {
        throw "Expected static runner to export USIM_TRAIN_FORCE_COLD=0"
    }
    if ($coreControls -notmatch "USIM_STEPS=0") {
        throw "Expected static runner to export USIM_STEPS=0"
    }

    $ppoControls = Invoke-StaticRunnerDry "ppo_loss_weight_controls" @{
        PpoLossWeight = 0.0
    }
    if ($ppoControls -notmatch "PPO controls: loss_weight=0") {
        throw "Expected static runner to print PPO loss weight"
    }
    if ($ppoControls -notmatch "USIM_PPO_LOSS_WEIGHT=0") {
        throw "Expected static runner to export USIM_PPO_LOSS_WEIGHT=0"
    }

    $policyControls = Invoke-StaticRunnerDry "rollout_policy_controls" @{
        RolloutPolicy = "greedy_similarity"
    }
    if ($policyControls -notmatch "Rollout policy: greedy_similarity") {
        throw "Expected static runner to print rollout policy"
    }
    if ($policyControls -notmatch "USIM_ROLLOUT_POLICY=greedy_similarity") {
        throw "Expected static runner to export USIM_ROLLOUT_POLICY=greedy_similarity"
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_static_runner_checkpoint_defaults.ps1 passed"
