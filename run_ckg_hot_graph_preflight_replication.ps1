param(
    [ValidateSet(2026, 2027)]
    [int]$Seed,
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\ckg_hot_graph_preflight_replication_seed$Seed"
$checkpointRoot = "checkpoints\ckg_hot_graph_preflight_replication_seed$Seed"
$logRoot = "background_logs\ckg_hot_graph_preflight_replication_seed$Seed"
$splitDir = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_$Seed"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$resultPath = Join-Path $outputRoot "preflight_result.json"
$epochsPath = Join-Path $outputRoot "validation_epochs.csv"
$logPath = Join-Path $logRoot "training.log"
$contractLogPath = Join-Path $logRoot "checkpoint_contract.log"

$runnerParams = [ordered]@{
    PythonRunner = ".\py.bat"
    ScriptPath = "ckg_hot_graph_preflight_replication.py"
    ContractScriptPath = "ckg_hot_replication_contract.py"
    DataDir = "processed_data_hin_clean_pop5"
    SplitDir = $splitDir
    OutputRoot = $outputRoot
    CheckpointRoot = $checkpointRoot
    Seed = $Seed
    TestEvaluation = $false
    UseCbi = $false
    UseSimulator = $false
    UsePpo = $false
    UseCourseRewards = $false
}

$lockedProtocol = [ordered]@{
    epochs = 15
    batch_size = 4096
    emb_dim = 64
    mlp_hidden = 64
    layers_gprime = 2
    layers_full = 2
    mask_rho = 0.30
    lambda_e = 1.0
    tau = 0.50
    ranking_neg_per_user = 32
    le_max_edges = 4096
    recon_user_chunk = 4096
    lr = 1e-3
    reg_weight = 1e-4
    cold_threshold = 1
    hot_r10_floor = 0.2219
    hot_n10_floor = 0.1442
}

$lockedConfig = [ordered]@{
    experiment = "ckg_hot_graph_preflight_replication"
    method = "fresh_cgrc_style_warm_graph_expert_capacity_preflight"
    selection = "validation_hot_capacity_only_with_dynamic_checkpoint_contract"
    test_evaluation = $false
    runner_parameters = $runnerParams
    python_training_knobs = $lockedProtocol
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
        # Native stderr is retained in the log; the process status is authoritative.
        $ErrorActionPreference = "Continue"
        & $Command *> $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return [int]$exitCode
}

function Require-JsonProperty([object]$Object, [string]$Name, [string]$Context) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Context is missing required property '$Name'."
    }
    return $property.Value
}

function Register-SelectedCheckpointContract([object]$Result, [System.Collections.IDictionary]$Manifest) {
    $selected = Require-JsonProperty -Object $Result -Name "selected_validation_epoch" -Context "preflight_result.json"
    if ($selected -isnot [pscustomobject]) {
        throw "preflight_result.json selected_validation_epoch must be a JSON object."
    }
    $selectedEpoch = [int](Require-JsonProperty -Object $selected -Name "epoch" -Context "selected_validation_epoch")
    if ($selectedEpoch -lt 1) {
        throw "preflight_result.json selected checkpoint epoch must be positive."
    }

    $contractExit = Invoke-NativeLogged -LogPath $contractLogPath -Command {
        & $runnerParams.PythonRunner -u $runnerParams.ContractScriptPath `
            --seed $Seed `
            --result-path $resultPath `
            --checkpoint-dir $checkpointRoot `
            --data-dir $runnerParams.DataDir `
            --split-dir $runnerParams.SplitDir
    }
    if ($contractExit -ne 0) {
        throw "Hot checkpoint contract registration returned exit code $contractExit."
    }

    $registeredResult = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($registeredResult -isnot [pscustomobject]) {
        throw "Registered preflight_result.json must contain a JSON object."
    }
    $contract = Require-JsonProperty -Object $registeredResult -Name "selected_checkpoint_contract" -Context "preflight_result.json"
    if ($contract -isnot [pscustomobject]) {
        throw "selected_checkpoint_contract must be a JSON object."
    }
    if ([int](Require-JsonProperty -Object $contract -Name "seed" -Context "selected_checkpoint_contract") -ne $Seed) {
        throw "selected_checkpoint_contract seed does not match the requested seed."
    }
    if ([int](Require-JsonProperty -Object $contract -Name "epoch" -Context "selected_checkpoint_contract") -ne $selectedEpoch) {
        throw "selected_checkpoint_contract epoch does not match selected_validation_epoch."
    }
    if ([int](Require-JsonProperty -Object $contract -Name "schema_version" -Context "selected_checkpoint_contract") -ne 1) {
        throw "selected_checkpoint_contract schema version is unsupported."
    }
    $relativePath = [string](Require-JsonProperty -Object $contract -Name "relative_path" -Context "selected_checkpoint_contract")
    $expectedName = "epoch_{0:D3}.pt" -f $selectedEpoch
    if ($relativePath -cne $expectedName -or $relativePath -cne [System.IO.Path]::GetFileName($relativePath)) {
        throw "selected_checkpoint_contract does not name the selected checkpoint safely."
    }
    $contractHash = [string](Require-JsonProperty -Object $contract -Name "sha256" -Context "selected_checkpoint_contract")
    if ($contractHash -cnotmatch "^[0-9a-f]{64}$") {
        throw "selected_checkpoint_contract SHA256 must be lowercase hexadecimal."
    }
    $tau = [double](Require-JsonProperty -Object $contract -Name "fixed_trust_tau" -Context "selected_checkpoint_contract")
    if ([Math]::Abs($tau - 0.24929234) -gt 1e-12) {
        throw "selected_checkpoint_contract does not preserve the fixed trust tau."
    }
    $q75 = [double](Require-JsonProperty -Object $contract -Name "warm_q75_audit" -Context "selected_checkpoint_contract")
    if ([double]::IsNaN($q75) -or [double]::IsInfinity($q75) -or $q75 -lt 0.0) {
        throw "selected_checkpoint_contract warm q75 audit is invalid."
    }
    $architecture = Require-JsonProperty -Object $contract -Name "architecture" -Context "selected_checkpoint_contract"
    if ($architecture -isnot [pscustomobject] -or [int]$architecture.emb_dim -ne 64 -or [int]$architecture.mlp_hidden -ne 64 -or [int]$architecture.layers_full -ne 2) {
        throw "selected_checkpoint_contract architecture is incompatible with the locked Hot protocol."
    }
    $checkpointPath = Join-Path $checkpointRoot $relativePath
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "The selected Hot checkpoint is missing: $checkpointPath"
    }
    $actualHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
    if ($actualHash -cne $contractHash) {
        throw "selected_checkpoint_contract SHA256 does not match the current checkpoint."
    }

    $Manifest.selected_checkpoint_contract = $contract
    $Manifest.selected_checkpoint_sha256 = $actualHash
    $Manifest.selected_checkpoint_epoch = $selectedEpoch
    $Manifest.selected_checkpoint_relative_path = $relativePath
    $Manifest.contract_registration_exit_code = $contractExit
}

$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "fast3_delta\eval.py",
    "fast3_delta\config.py",
    "run_fast3_main_table_config.ps1",
    "paper_aaai27\main.tex"
)
$sourceFiles = @(
    "ckg_hot_graph_preflight_replication.py",
    "ckg_hot_replication_contract.py",
    "ckg_hot_graph_preflight.py",
    "cgrc_paper_static_hin.py",
    "hin_data_common.py",
    "hin_eval_common.py",
    "lightgcn_static_hin.py"
)
$launcherFiles = @("run_ckg_hot_graph_preflight_replication.ps1")
$splitFiles = @(
    (Join-Path $splitDir "static_train.pkl"),
    (Join-Path $splitDir "static_val.pkl")
)
$dataFiles = @(
    (Join-Path $runnerParams.DataDir "meta.json"),
    (Join-Path $runnerParams.DataDir "content_emb.pt")
)

$formalRoots = @($outputRoot, $checkpointRoot, $logRoot)
$existingRoots = @($formalRoots | Where-Object { Test-Path -LiteralPath $_ })
if ($existingRoots.Count -gt 0) {
    throw "Strict Hot replication requires fresh roots; refusing to reuse: $($existingRoots -join ', ')"
}

$protectedBefore = Get-HashMap -Paths $protectedFiles
$sourceBefore = Get-HashMap -Paths $sourceFiles
$launcherBefore = Get-HashMap -Paths $launcherFiles
$splitBefore = Get-HashMap -Paths $splitFiles
$dataBefore = Get-HashMap -Paths $dataFiles
New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null

$manifest = [ordered]@{
    schema_version = 1
    experiment = "ckg_hot_graph_preflight_replication"
    seed = $Seed
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
    launcher_sha256 = $launcherBefore
    launcher_sha256_after = $null
    split_sha256 = $splitBefore
    split_sha256_after = $null
    data_sha256 = $dataBefore
    data_sha256_after = $null
    protected_files_before = $protectedBefore
    protected_files_after = $null
    selected_checkpoint_contract = $null
    selected_checkpoint_sha256 = $null
    selected_checkpoint_epoch = $null
    selected_checkpoint_relative_path = $null
    contract_registration_exit_code = $null
    validation_epochs = @()
    validation_epochs_import_error = $null
    paths = [ordered]@{
        output_root = (Resolve-Path -LiteralPath $outputRoot).Path
        checkpoint_root = (Resolve-Path -LiteralPath $checkpointRoot).Path
        log_path = (Join-Path $repoPath $logPath)
        contract_log_path = (Join-Path $repoPath $contractLogPath)
        result_path = (Join-Path $repoPath $resultPath)
    }
}
Write-Json $manifestPath $manifest

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$runError = $null
try {
    $pythonExit = Invoke-NativeLogged -LogPath $logPath -Command {
        & $runnerParams.PythonRunner -u $runnerParams.ScriptPath `
            --seed $Seed `
            --data-dir $runnerParams.DataDir `
            --split-dir $runnerParams.SplitDir `
            --output-dir $runnerParams.OutputRoot `
            --checkpoint-dir $runnerParams.CheckpointRoot
    }
    if ($pythonExit -notin @(0, 2)) {
        throw "Hot replication entrypoint returned exit code $pythonExit."
    }
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Hot replication did not write $resultPath"
    }
    if (-not (Test-Path -LiteralPath $epochsPath -PathType Leaf)) {
        throw "Hot replication did not write $epochsPath"
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($result -isnot [pscustomobject]) {
        throw "preflight_result.json must contain a JSON object."
    }
    $config = Require-JsonProperty -Object $result -Name "config" -Context "preflight_result.json"
    if ($config -isnot [pscustomobject] -or [int]$config.seed -ne $Seed) {
        throw "preflight_result.json seed does not match the requested seed."
    }
    $testEvaluation = Require-JsonProperty -Object $result -Name "test_evaluation" -Context "preflight_result.json"
    if ($testEvaluation -isnot [bool] -or $testEvaluation) {
        throw "Hot replication result must be validation-only."
    }
    $passedHotPreflight = Require-JsonProperty -Object $result -Name "passed_hot_preflight" -Context "preflight_result.json"
    if ($passedHotPreflight -isnot [bool]) {
        throw "preflight_result.json passed_hot_preflight must be a JSON Boolean."
    }
    $expectedGateStatus = if ([bool]$passedHotPreflight) { "completed" } else { "completed_gate_failed" }
    $gateStatus = Require-JsonProperty -Object $result -Name "gate_status" -Context "preflight_result.json"
    if ($gateStatus -isnot [string] -or $gateStatus -cne $expectedGateStatus) {
        throw "preflight_result.json gate_status must be '$expectedGateStatus'."
    }
    if (([bool]$passedHotPreflight -and $pythonExit -ne 0) -or ((-not [bool]$passedHotPreflight) -and $pythonExit -ne 2)) {
        throw "Hot replication exit code does not match its gate status."
    }

    $manifest.gate_status = $expectedGateStatus
    if ([bool]$passedHotPreflight) {
        Register-SelectedCheckpointContract -Result $result -Manifest $manifest
        $manifest.status = "completed"
        $manifest.exit_code = 0
    }
    else {
        $manifest.status = "completed_gate_failed"
        $manifest.exit_code = 2
        $manifest.error = "Hot capacity gate failed; Stage B remains locked."
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
        try {
            $manifest.validation_epochs = @(Import-Csv -LiteralPath $epochsPath -ErrorAction Stop)
        }
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
    $launcherAfter = Get-HashMap -Paths $launcherFiles -AllowMissing
    $splitAfter = Get-HashMap -Paths $splitFiles -AllowMissing
    $dataAfter = Get-HashMap -Paths $dataFiles -AllowMissing
    $manifest.protected_files_after = $protectedAfter
    $manifest.source_sha256_after = $sourceAfter
    $manifest.launcher_sha256_after = $launcherAfter
    $manifest.split_sha256_after = $splitAfter
    $manifest.data_sha256_after = $dataAfter
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
    $launcherChanged = @($launcherFiles | Where-Object { $launcherBefore[$_] -ne $launcherAfter[$_] })
    if ($launcherChanged.Count -gt 0) {
        $manifest.status = "failed"; $manifest.exit_code = 1
        $manifest.error = "Launcher source changed while running: $($launcherChanged -join ', ')"
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
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    Write-Json $manifestPath $manifest
}

if ($null -ne $runError) { throw $runError }
if ($manifest.status -eq "completed_gate_failed") {
    Write-Host "Hot replication completed but did not pass the registered capacity gate."
    exit 2
}
Write-Host "Strict Hot replication completed."
Write-Host ("Manifest: {0}" -f (Resolve-Path -LiteralPath $manifestPath).Path)
