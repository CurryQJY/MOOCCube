param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\ckg_frozen_hot_pseudocold_adapter_seed2025"
$checkpointRoot = "checkpoints\ckg_frozen_hot_pseudocold_adapter_seed2025"
$logRoot = "background_logs\ckg_frozen_hot_pseudocold_adapter_seed2025"
$splitDir = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025"
$hotOutputRoot = "outputs\ckg_hot_graph_preflight_seed2025"
$hotCheckpointRoot = "checkpoints\ckg_hot_graph_preflight_seed2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"
$resultPath = Join-Path $outputRoot "adapter_preflight_result.json"
$epochsPath = Join-Path $outputRoot "validation_epochs.csv"

$runnerParams = [ordered]@{
    PythonRunner = ".\py.bat"
    ScriptPath = "ckg_frozen_hot_pseudocold_adapter.py"
    DataDir = "processed_data_hin_clean_pop5"
    SplitDir = $splitDir
    OutputRoot = $outputRoot
    CheckpointRoot = $checkpointRoot
    HotOutputRoot = $hotOutputRoot
    HotCheckpointRoot = $hotCheckpointRoot
    Seeds = @(2025)
    TestEvaluation = $false
    UseCbi = $false
    UseSimulator = $false
    UsePpo = $false
    UseCourseRewards = $false
}

$lockedProtocol = [ordered]@{
    n_items = 698
    warm_item_count = 596
    train_zero_item_count = 102
    pseudo_cold_item_count = 102
    trust_tau = 0.24929234
    epochs = 15
    emb_dim = 64
    hidden_dim = 64
    layers_full = 2
    batch_size = 4096
    negatives_per_positive = 32
    ranking_temperature = 0.5
    optimizer = "Adam"
    lr = 1e-3
    weight_decay = 0.0
    delta_reg_weight = 0.0
    parity_atol = 1e-5
    retention_tolerance = 0.003
    cold_gain_minimum = 0.003
}

$lockedConfig = [ordered]@{
    experiment = "ckg_frozen_hot_pseudocold_adapter_seed2025"
    method = "frozen_hot_edge_masked_pseudocold_shared_content_adapter"
    selection = "runtime_epoch0_retention_guards_then_cold_ndcg"
    hot_checkpoint = "checkpoints\ckg_hot_graph_preflight_seed2025\epoch_015.pt"
    test_evaluation = $false
    runner_parameters = $runnerParams
    locked_protocol = $lockedProtocol
}

if ($DryRun) {
    $lockedConfig | ConvertTo-Json -Depth 20
    exit 0
}

function Get-HashMap {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths,
        [switch]$AllowMissing
    )
    $result = [ordered]@{}
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            if ($AllowMissing) {
                $result[$path] = "<missing>"
                continue
            }
            throw "Missing reproducibility file: $path"
        }
        try {
            $result[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        }
        catch {
            if (-not $AllowMissing) { throw }
            $result[$path] = "<unreadable>"
        }
    }
    return $result
}

function Write-Json([string]$Path, $Payload) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $Payload | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-NativeLogged([scriptblock]$Command, [string]$LogPath) {
    $previousPreference = $ErrorActionPreference
    try {
        # Native stderr is log output; process exit status is the failure signal.
        $ErrorActionPreference = "Continue"
        & $Command *> $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return [int]$exitCode
}

$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "fast3_delta\eval.py",
    "fast3_delta\config.py",
    "run_fast3_main_table_config.ps1",
    "paper_aaai27\main.tex"
)
$sourceFiles = @(
    "ckg_frozen_hot_pseudocold_adapter.py",
    "run_ckg_frozen_hot_pseudocold_adapter_seed2025.ps1",
    "ckg_hot_graph_preflight.py",
    "cgrc_paper_static_hin.py",
    "hin_data_common.py",
    "hin_eval_common.py",
    "lightgcn_static_hin.py"
)
$splitFiles = @(
    (Join-Path $splitDir "static_train.pkl"),
    (Join-Path $splitDir "static_val.pkl")
)
$dataFiles = @(
    (Join-Path $runnerParams.DataDir "meta.json"),
    (Join-Path $runnerParams.DataDir "content_emb.pt")
)
$hotArtifactFiles = @(
    (Join-Path $hotOutputRoot "run_manifest.json"),
    (Join-Path $hotOutputRoot "preflight_result.json"),
    (Join-Path $hotCheckpointRoot "epoch_015.pt")
)

$formalRoots = @($outputRoot, $checkpointRoot, $logRoot)
$existingRoots = @($formalRoots | Where-Object { Test-Path -LiteralPath $_ })
if ($existingRoots.Count -gt 0) {
    throw "Formal Stage B run requires fresh roots; refusing to reuse: $($existingRoots -join ', ')"
}

$protectedBefore = Get-HashMap -Paths $protectedFiles
$sourceBefore = Get-HashMap -Paths $sourceFiles
$splitBefore = Get-HashMap -Paths $splitFiles
$dataBefore = Get-HashMap -Paths $dataFiles
$hotArtifactsBefore = Get-HashMap -Paths $hotArtifactFiles
New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

$manifest = [ordered]@{
    schema_version = 1
    experiment = "ckg_frozen_hot_pseudocold_adapter_seed2025"
    status = "running"
    gate_status = $null
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    completed_at_utc = $null
    elapsed_seconds = $null
    exit_code = $null
    error = $null
    repo = $repoPath
    git_commit = (git rev-parse HEAD).Trim()
    git_dirty = @((git status --porcelain)).Count -gt 0
    locked_config = $lockedConfig
    source_sha256 = $sourceBefore
    source_sha256_after = $null
    split_sha256 = $splitBefore
    split_sha256_after = $null
    data_sha256 = $dataBefore
    data_sha256_after = $null
    hot_artifacts_sha256 = $hotArtifactsBefore
    hot_artifacts_sha256_after = $null
    protected_files_before = $protectedBefore
    protected_files_after = $null
    validation_epochs = @()
    validation_epochs_import_error = $null
    paths = [ordered]@{
        output_root = (Resolve-Path -LiteralPath $outputRoot).Path
        checkpoint_root = (Resolve-Path -LiteralPath $checkpointRoot).Path
        log_path = (Join-Path $repoPath $logPath)
        result_path = (Join-Path $repoPath $resultPath)
    }
}
Write-Json $manifestPath $manifest

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$runError = $null
try {
    $pythonExit = Invoke-NativeLogged -LogPath $logPath -Command {
        & $runnerParams.PythonRunner -u $runnerParams.ScriptPath `
            --seed 2025 `
            --data-dir $runnerParams.DataDir `
            --split-dir $runnerParams.SplitDir `
            --output-dir $runnerParams.OutputRoot `
            --checkpoint-dir $runnerParams.CheckpointRoot `
            --hot-output-dir $runnerParams.HotOutputRoot `
            --hot-checkpoint-dir $runnerParams.HotCheckpointRoot
    }
    if ($pythonExit -notin @(0, 2)) {
        throw "Stage B entrypoint returned exit code $pythonExit."
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Stage B runner did not write $resultPath"
    }
    if (-not (Test-Path -LiteralPath $epochsPath -PathType Leaf)) {
        throw "Stage B runner did not write $epochsPath"
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($result -isnot [pscustomobject]) { throw "adapter_preflight_result.json must contain a JSON object." }
    $passedProperty = $result.PSObject.Properties["passed_stage_b_screen"]
    if ($null -eq $passedProperty -or $passedProperty.Value -isnot [bool]) {
        throw "adapter_preflight_result.json must contain JSON Boolean passed_stage_b_screen."
    }
    $passed = [bool]$passedProperty.Value
    $expectedGate = if ($passed) { "completed" } else { "completed_gate_failed" }
    $gateProperty = $result.PSObject.Properties["gate_status"]
    if ($null -eq $gateProperty -or $gateProperty.Value -isnot [string] -or $gateProperty.Value -cne $expectedGate) {
        throw "adapter_preflight_result.json gate_status must be '$expectedGate'."
    }
    if (($passed -and $pythonExit -ne 0) -or ((-not $passed) -and $pythonExit -ne 2)) {
        throw "Stage B entrypoint exit code does not match gate status."
    }
    $manifest.gate_status = $expectedGate
    if ($passed) {
        $manifest.status = "completed"
        $manifest.exit_code = 0
    }
    else {
        $manifest.status = "completed_gate_failed"
        $manifest.exit_code = 2
        $manifest.error = "Stage B feasibility screen did not pass its registered Cold gain condition."
    }
}
catch {
    $runError = $_
    $manifest.status = "failed"
    $manifest.exit_code = 1
    $manifest.error = $_.Exception.Message
}
finally {
    $timer.Stop()
    if (Test-Path -LiteralPath $epochsPath -PathType Leaf) {
        try { $manifest.validation_epochs = @(Import-Csv -LiteralPath $epochsPath -ErrorAction Stop) }
        catch {
            $manifest.validation_epochs_import_error = $_.Exception.Message
            $manifest.status = "failed"
            $manifest.exit_code = 1
            $manifest.error = "Could not import validation epochs: $($_.Exception.Message)"
            $runError = [System.InvalidOperationException]::new($manifest.error)
        }
    }
    $protectedAfter = Get-HashMap -Paths $protectedFiles -AllowMissing
    $sourceAfter = Get-HashMap -Paths $sourceFiles -AllowMissing
    $splitAfter = Get-HashMap -Paths $splitFiles -AllowMissing
    $dataAfter = Get-HashMap -Paths $dataFiles -AllowMissing
    $hotArtifactsAfter = Get-HashMap -Paths $hotArtifactFiles -AllowMissing
    $manifest.protected_files_after = $protectedAfter
    $manifest.source_sha256_after = $sourceAfter
    $manifest.split_sha256_after = $splitAfter
    $manifest.data_sha256_after = $dataAfter
    $manifest.hot_artifacts_sha256_after = $hotArtifactsAfter
    $changed = @($protectedFiles | Where-Object { $protectedBefore[$_] -ne $protectedAfter[$_] })
    if ($changed.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Protected files changed: $($changed -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $sourceChanged = @($sourceFiles | Where-Object { $sourceBefore[$_] -ne $sourceAfter[$_] })
    if ($sourceChanged.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Experiment source changed while running: $($sourceChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $splitChanged = @($splitFiles | Where-Object { $splitBefore[$_] -ne $splitAfter[$_] })
    if ($splitChanged.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Static split changed while running: $($splitChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $dataChanged = @($dataFiles | Where-Object { $dataBefore[$_] -ne $dataAfter[$_] })
    if ($dataChanged.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Consumed data changed while running: $($dataChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $hotChanged = @($hotArtifactFiles | Where-Object { $hotArtifactsBefore[$_] -ne $hotArtifactsAfter[$_] })
    if ($hotChanged.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Hot preflight artifacts changed while running: $($hotChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    Write-Json $manifestPath $manifest
}

if ($null -ne $runError) { throw $runError }
if ($manifest.status -eq "completed_gate_failed") {
    Write-Host "Stage B completed but did not pass its registered feasibility gate."
    exit 2
}
Write-Host "Stage B frozen-Hot masked pseudo-cold adapter completed."
Write-Host ("Manifest: {0}" -f (Resolve-Path -LiteralPath $manifestPath).Path)
