param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\ckg_hot_graph_preflight_seed2025"
$checkpointRoot = "checkpoints\ckg_hot_graph_preflight_seed2025"
$logRoot = "background_logs\ckg_hot_graph_preflight_seed2025"
$splitDir = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"

$runnerParams = [ordered]@{
    PythonRunner = ".\py.bat"
    ScriptPath = "ckg_hot_graph_preflight.py"
    DataDir = "processed_data_hin_clean_pop5"
    SplitDir = $splitDir
    OutputRoot = "outputs\ckg_hot_graph_preflight_seed2025"
    CheckpointRoot = "checkpoints\ckg_hot_graph_preflight_seed2025"
    Seeds = @(2025)
    Epochs = 15
    BatchSize = 4096
    HotR10Floor = 0.2219
    HotN10Floor = 0.1442
    TestEvaluation = $false
    UseCbi = $false
    UseSimulator = $false
    UsePpo = $false
    UseCourseRewards = $false
}

$pythonTrainingKnobs = [ordered]@{
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
}

$lockedConfig = [ordered]@{
    experiment = "ckg_hot_graph_preflight_seed2025"
    method = "fresh_cgrc_style_warm_graph_expert_capacity_preflight"
    selection = "validation_hot_capacity_only"
    validation_reference = [ordered]@{
        source = "CGRC seed-2025 validation reproduction"
        hot_r10_floor = $runnerParams.HotR10Floor
        hot_n10_floor = $runnerParams.HotN10Floor
        reference_hot_r10 = 0.2269028334
        reference_hot_n10 = 0.1492342676
    }
    runner_parameters = $runnerParams
    python_training_knobs = $pythonTrainingKnobs
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
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                $result[$path] = "<unreadable>"
            }
            else {
                $result[$path] = "<missing>"
            }
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
        # Windows PowerShell turns native stderr into NativeCommandError under Stop.
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
    "ckg_hot_graph_preflight.py",
    "run_ckg_hot_graph_preflight_seed2025.ps1",
    "cgrc_paper_static_hin.py",
    "hin_data_common.py",
    "hin_eval_common.py",
    "lightgcn_static_hin.py"
)
$splitFiles = @(
    (Join-Path $splitDir "static_train.pkl"),
    (Join-Path $splitDir "static_val.pkl"),
    (Join-Path $splitDir "static_test.pkl"),
    (Join-Path $splitDir "static_split_assignments.csv")
)
$dataFiles = @(
    (Join-Path $runnerParams.DataDir "meta.json"),
    (Join-Path $runnerParams.DataDir "content_emb.pt"),
    (Join-Path $runnerParams.DataDir "stream_data.pkl")
)

$formalRoots = @($outputRoot, $checkpointRoot, $logRoot)
$existingRoots = @($formalRoots | Where-Object { Test-Path -LiteralPath $_ })
if ($existingRoots.Count -gt 0) {
    throw "Formal CKG Hot Graph preflight requires fresh roots; refusing to reuse: $($existingRoots -join ', ')"
}

$protectedBefore = Get-HashMap -Paths $protectedFiles
$sourceBefore = Get-HashMap -Paths $sourceFiles
$splitBefore = Get-HashMap -Paths $splitFiles
$dataBefore = Get-HashMap -Paths $dataFiles
New-Item -ItemType Directory -Force -Path $outputRoot, $checkpointRoot, $logRoot | Out-Null
$manifest = [ordered]@{
    schema_version = 1
    experiment = "ckg_hot_graph_preflight_seed2025"
    status = "running"
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
    protected_files_before = $protectedBefore
    protected_files_after = $null
    validation_epochs = @()
    validation_epochs_import_error = $null
    gate_status = $null
    paths = [ordered]@{
        output_root = (Resolve-Path -LiteralPath $outputRoot).Path
        checkpoint_root = (Resolve-Path -LiteralPath $checkpointRoot).Path
        log_path = (Join-Path $repoPath $logPath)
        result_path = (Join-Path $repoPath (Join-Path $outputRoot "preflight_result.json"))
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
            --epochs $runnerParams.Epochs `
            --batch-size $runnerParams.BatchSize `
            --hot-r10-floor $runnerParams.HotR10Floor `
            --hot-n10-floor $runnerParams.HotN10Floor `
            --emb-dim $($pythonTrainingKnobs.emb_dim) `
            --mlp-hidden $($pythonTrainingKnobs.mlp_hidden) `
            --layers-gprime $($pythonTrainingKnobs.layers_gprime) `
            --layers-full $($pythonTrainingKnobs.layers_full) `
            --mask-rho $($pythonTrainingKnobs.mask_rho) `
            --lambda-e $($pythonTrainingKnobs.lambda_e) `
            --tau $($pythonTrainingKnobs.tau) `
            --ranking-neg-per-user $($pythonTrainingKnobs.ranking_neg_per_user) `
            --le-max-edges $($pythonTrainingKnobs.le_max_edges) `
            --recon-user-chunk $($pythonTrainingKnobs.recon_user_chunk) `
            --lr $($pythonTrainingKnobs.lr) `
            --reg-weight $($pythonTrainingKnobs.reg_weight) `
            --cold-threshold $($pythonTrainingKnobs.cold_threshold)
    }
    if ($pythonExit -ne 0) { throw "Hot graph preflight entrypoint returned exit code $pythonExit." }
    $resultPath = Join-Path $outputRoot "preflight_result.json"
    $epochsPath = Join-Path $outputRoot "validation_epochs.csv"
    if (-not (Test-Path -LiteralPath $resultPath)) { throw "Preflight did not write $resultPath" }
    if (-not (Test-Path -LiteralPath $epochsPath)) { throw "Preflight did not write $epochsPath" }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($result -isnot [pscustomobject]) {
        throw "preflight_result.json must contain a JSON object."
    }
    $passedProperty = $result.PSObject.Properties["passed_hot_preflight"]
    if ($null -eq $passedProperty) {
        throw "preflight_result.json is missing required JSON Boolean passed_hot_preflight."
    }
    if ($passedProperty.Value -isnot [bool]) {
        throw "preflight_result.json passed_hot_preflight must be a JSON Boolean."
    }
    $passedHotPreflight = [bool]$passedProperty.Value
    $expectedGateStatus = if ($passedHotPreflight) { "completed" } else { "completed_gate_failed" }
    $gateProperty = $result.PSObject.Properties["gate_status"]
    if ($null -eq $gateProperty -or $gateProperty.Value -isnot [string] -or $gateProperty.Value -cne $expectedGateStatus) {
        throw "preflight_result.json gate_status must be '$expectedGateStatus' when passed_hot_preflight is $passedHotPreflight."
    }
    $manifest.gate_status = $expectedGateStatus
    if ($passedHotPreflight) {
        $manifest.status = "completed"
        $manifest.exit_code = 0
    }
    else {
        $manifest.status = "completed_gate_failed"
        $manifest.exit_code = 2
        $manifest.error = "Hot capacity gate failed; no CBI adapter is unlocked."
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
    $epochsPath = Join-Path $outputRoot "validation_epochs.csv"
    if (Test-Path -LiteralPath $epochsPath) {
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
    $splitAfter = Get-HashMap -Paths $splitFiles -AllowMissing
    $dataAfter = Get-HashMap -Paths $dataFiles -AllowMissing
    $manifest.protected_files_after = $protectedAfter
    $manifest.source_sha256_after = $sourceAfter
    $manifest.split_sha256_after = $splitAfter
    $manifest.data_sha256_after = $dataAfter
    $changed = @($protectedFiles | Where-Object { $protectedBefore[$_] -ne $protectedAfter[$_] })
    if ($changed.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Protected files changed: $($changed -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $sourceChanged = @($sourceFiles | Where-Object { $manifest.source_sha256[$_] -ne $sourceAfter[$_] })
    if ($sourceChanged.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Experiment source changed while running: $($sourceChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $splitChanged = @($splitFiles | Where-Object { $manifest.split_sha256[$_] -ne $splitAfter[$_] })
    if ($splitChanged.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Static split changed while running: $($splitChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $dataChanged = @($dataFiles | Where-Object { $manifest.data_sha256[$_] -ne $dataAfter[$_] })
    if ($dataChanged.Count -gt 0) {
        $manifest.status = "failed"
        $manifest.exit_code = 1
        $manifest.error = "Consumed data changed while running: $($dataChanged -join ', ')"
        $runError = [System.InvalidOperationException]::new($manifest.error)
    }
    $manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    $manifest.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    Write-Json $manifestPath $manifest
}

if ($null -ne $runError) { throw $runError }
if ($manifest.status -eq "completed_gate_failed") {
    Write-Host "CKG Hot Graph preflight completed but did not pass the Hot capacity gate."
    exit 2
}
Write-Host "CKG Hot Graph preflight completed."
Write-Host ("Manifest: {0}" -f (Resolve-Path -LiteralPath $manifestPath).Path)
