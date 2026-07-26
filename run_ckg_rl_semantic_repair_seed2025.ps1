param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Seed = 2025,
    [int]$Epochs = 60,
    [int]$Patience = 8,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$mainRunner = Join-Path $repoPath "run_fast3_main_table_config.ps1"

$params = @{
    Repo = $repoPath
    Seeds = @($Seed)
    ColdThresholds = @(1)
    Epochs = $Epochs
    Patience = $Patience
    OutputRoot = "outputs\ckg_rl_semantic_repair\seed2025"
    CheckpointRoot = "checkpoints\ckg_rl_semantic_repair\seed2025"
    SimulatorTargetMode = "initial_state"
    DeterministicEvalCandidates = $true
    EvalReuseItemBank = $true
    DeterministicEvalSeed = $Seed
    SaveCkpt = $true
    SaveOptState = $true
}

if ($DryRun) {
    $params["DryRun"] = $true
}

& $mainRunner @params
exit $LASTEXITCODE
