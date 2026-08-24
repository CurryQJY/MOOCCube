param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [switch]$RerunCompleted,
    [switch]$ValidateExistingOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $Base $Path)
}

function New-KGRecJob(
    [string]$Dataset,
    [int]$Seed,
    [string]$SplitRoot,
    [string]$AtomicDir,
    [string]$OutputDir,
    [double]$LearningRate,
    [int]$Epochs,
    [int]$Patience,
    [bool]$Runnable,
    [string]$LinkPath = "",
    [string]$KgPath = ""
) {
    [pscustomobject]@{
        Dataset = $Dataset
        Seed = $Seed
        SplitRoot = $SplitRoot
        AtomicDir = $AtomicDir
        OutputDir = $OutputDir
        LearningRate = $LearningRate
        Epochs = $Epochs
        Patience = $Patience
        BatchSize = 4096
        Runnable = $Runnable
        LinkPath = $LinkPath
        KgPath = $KgPath
    }
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo
$PythonRunnerAbs = Resolve-RunPath $Repo $PythonRunner
$Runner = Join-Path $Repo "paper_aaai27\scripts\run_kgrec_strict_seed.py"
$ResultRoot = Join-Path $Repo "paper_aaai27\baseline_sources\_kgrec_strict"
$QueueDir = Join-Path $ResultRoot "_remaining_main_table_queue"
$QueueLog = Join-Path $QueueDir "queue.log"
$StatusPath = Join-Path $QueueDir "queue_status.json"
$SummaryPath = Join-Path $QueueDir "main_table_summary.json"
$SummaryCsvPath = Join-Path $QueueDir "main_table_summary.csv"
$PytestBaseTemp = Join-Path $QueueDir "_pytest_tmp"
$RecBoleRoot = Join-Path $Repo "paper_aaai27\baseline_sources\PCGNN_recbole_drive\RecBole-master\dataset"
$CocoLink = Join-Path $RecBoleRoot "coco_strict_full\coco_strict_full.link"
$CocoKg = Join-Path $RecBoleRoot "coco_strict_full\coco_strict_full.kg"
$JunyiLink = Join-Path $RecBoleRoot "junyi_strict_full\junyi_strict_full.link"
$JunyiKg = Join-Path $RecBoleRoot "junyi_strict_full\junyi_strict_full.kg"

$jobs = @(
    New-KGRecJob "MOOCCube" 2025 "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2025_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2025_retuned_lr1e-5" 1e-5 20 5 $false
    New-KGRecJob "MOOCCube" 2026 "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2026" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2026_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2026_retuned_lr1e-5" 1e-5 20 5 $false
    New-KGRecJob "MOOCCube" 2027 "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2027" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2027_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\mooccube_seed2027_retuned_lr1e-5" 1e-5 20 5 $false
    New-KGRecJob "COCO" 2025 "outputs\coco\single_seed_triage\ours_full\strict_item_cold_balanced_thr1_seed_2025" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2025_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2025_single_lr1e-5" 1e-5 20 5 $false $CocoLink $CocoKg
    New-KGRecJob "COCO" 2026 "outputs\coco\single_seed_triage\ours_full\strict_item_cold_balanced_thr1_seed_2026" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2026_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2026_main_lr1e-5" 1e-5 20 5 $true $CocoLink $CocoKg
    New-KGRecJob "COCO" 2027 "outputs\coco\single_seed_triage\ours_full\strict_item_cold_balanced_thr1_seed_2027" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2027_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\coco_seed2027_main_lr1e-5" 1e-5 20 5 $true $CocoLink $CocoKg
    New-KGRecJob "Junyi" 2025 "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2025" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2025_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2025_relationfix_lr1e-6_p8_trainonly" 1e-6 10 8 $false $JunyiLink $JunyiKg
    New-KGRecJob "Junyi" 2026 "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2026" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2026_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2026_relationfix_lr1e-6_p8" 1e-6 10 8 $false $JunyiLink $JunyiKg
    New-KGRecJob "Junyi" 2027 "outputs\junyi\main_table_3seed\strict_item_cold_balanced_thr1_seed_2027" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2027_atomic" "paper_aaai27\baseline_sources\_kgrec_strict\junyi_seed2027_main_lr1e-6_p8" 1e-6 10 8 $true $JunyiLink $JunyiKg
)

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) { Add-Content -LiteralPath $QueueLog -Encoding UTF8 -Value $line }
    Write-Host $line
}

function Set-QueueStatus([string]$State, [string]$Current, [string]$Message) {
    if ($DryRun) { return }
    [ordered]@{
        state = $State
        current = $Current
        message = $Message
        updated_at = (Get-Date).ToString("o")
        queue_log = $QueueLog
        summary_json = $SummaryPath
        summary_csv = $SummaryCsvPath
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-NativeLogged([scriptblock]$Command, [string]$LogPath) {
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5 wraps any native stderr line as NativeCommandError
        # when ErrorActionPreference is Stop, even if the process exits with code 0.
        $ErrorActionPreference = "Continue"
        & $Command *> $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    return [int]$exitCode
}

function Test-FiniteMetric($Value) {
    try { $number = [double]$Value } catch { return $false }
    return (-not [double]::IsNaN($number)) -and (-not [double]::IsInfinity($number))
}

function Get-ValidationResult($Job) {
    $outputDir = Resolve-RunPath $Repo $Job.OutputDir
    $atomicDir = Resolve-RunPath $Repo $Job.AtomicDir
    $reportPath = Join-Path $outputDir "kgrec_strict_adapter_report.json"
    $manifestPath = Join-Path $atomicDir "strict_manifest.json"
    if (-not (Test-Path -LiteralPath $reportPath)) { return [pscustomobject]@{ Valid=$false; Reason="missing report" } }
    if (-not (Test-Path -LiteralPath $manifestPath)) { return [pscustomobject]@{ Valid=$false; Reason="missing manifest" } }
    try {
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    } catch { return [pscustomobject]@{ Valid=$false; Reason="invalid JSON: $($_.Exception.Message)" } }

    $checks = @(
        [pscustomobject]@{ Ok=($report.status -eq "complete"); Reason="status is not complete" }
        [pscustomobject]@{ Ok=([int]$report.seed -eq [int]$Job.Seed); Reason="seed mismatch" }
        [pscustomobject]@{ Ok=([string]$report.device -eq "cuda"); Reason="run did not use CUDA" }
        [pscustomobject]@{ Ok=([int]$report.best_epoch -gt 0); Reason="best_epoch must be greater than zero" }
        [pscustomobject]@{ Ok=([int]$report.config.batch_size -eq [int]$Job.BatchSize); Reason="batch size mismatch" }
        [pscustomobject]@{ Ok=([int]$report.config.epochs -eq [int]$Job.Epochs); Reason="epoch budget mismatch" }
        [pscustomobject]@{ Ok=([int]$report.config.patience -eq [int]$Job.Patience); Reason="patience mismatch" }
        [pscustomobject]@{ Ok=([math]::Abs([double]$report.config.lr - [double]$Job.LearningRate) -lt 1e-12); Reason="learning rate mismatch" }
        [pscustomobject]@{ Ok=([int]$report.config.dim -eq 64); Reason="embedding dimension mismatch" }
        [pscustomobject]@{ Ok=([int]$report.config.context_hops -eq 2); Reason="context hops mismatch" }
        [pscustomobject]@{ Ok=([bool]$report.strict_protocol.item_macro_metrics); Reason="item-macro protocol missing" }
        [pscustomobject]@{ Ok=([bool]$report.strict_protocol.train_history_masking); Reason="train-history masking missing" }
        [pscustomobject]@{ Ok=([int]$report.data.n_test_pairs -eq [int]$manifest.n_test_pairs); Reason="test pair count differs from manifest" }
        [pscustomobject]@{ Ok=([int]$report.data.n_validation_pairs -eq [int]$manifest.n_validation_pairs); Reason="validation pair count differs from manifest" }
        [pscustomobject]@{ Ok=([int]$report.test.counts.all_rows -eq [int]$manifest.n_test_pairs); Reason="evaluated test rows differ from manifest" }
        [pscustomobject]@{ Ok=([int]$report.validation.counts.all_rows -eq [int]$manifest.n_validation_pairs); Reason="evaluated validation rows differ from manifest" }
        [pscustomobject]@{ Ok=([int]$report.test.counts.cold_items -gt 0); Reason="test has no cold items" }
        [pscustomobject]@{ Ok=([int]$report.test.counts.cold_rows -gt 0); Reason="test has no cold rows" }
        [pscustomobject]@{ Ok=([int]$report.validation.counts.cold_items -gt 0); Reason="validation has no cold items" }
        [pscustomobject]@{ Ok=([int]$report.validation.counts.cold_rows -gt 0); Reason="validation has no cold rows" }
    )
    foreach ($check in $checks) { if (-not [bool]$check.Ok) { return [pscustomobject]@{ Valid=$false; Reason=[string]$check.Reason } } }

    $checkpoint = [string]$report.checkpoint_path
    if (-not $checkpoint -or -not (Test-Path -LiteralPath $checkpoint)) { return [pscustomobject]@{ Valid=$false; Reason="missing checkpoint" } }
    if ((Get-Item -LiteralPath $checkpoint).Length -le 0) { return [pscustomobject]@{ Valid=$false; Reason="empty checkpoint" } }
    foreach ($metric in @("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")) {
        if (-not (Test-FiniteMetric $report.test.full_cold_item_macro.$metric)) {
            return [pscustomobject]@{ Valid=$false; Reason="invalid cold test metric $metric" }
        }
    }
    return [pscustomobject]@{ Valid=$true; Reason="valid"; Report=$report; Manifest=$manifest; ReportPath=$reportPath }
}

function Assert-Inputs {
    foreach ($path in @($PythonRunnerAbs, $Runner, $CocoLink, $CocoKg, $JunyiLink, $JunyiKg)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing required KGRec input: $path" }
    }
    foreach ($job in $jobs) {
        $split = Resolve-RunPath $Repo $job.SplitRoot
        foreach ($name in @("static_train.pkl", "static_val.pkl", "static_test.pkl")) {
            if (-not (Test-Path -LiteralPath (Join-Path $split $name))) { throw "Missing split file: $(Join-Path $split $name)" }
        }
    }
}

function Invoke-KGRecTests {
    Write-QueueLog "TEST KGRec unit suite"
    Set-QueueStatus "running" "tests" "Running KGRec unit suite"
    $log = Join-Path $QueueDir "unit_tests.log"
    $exitCode = Invoke-NativeLogged { & $PythonRunnerAbs -m pytest tests/test_kgrec_native_scatter.py tests/test_kgrec_strict_adapter.py tests/test_kgrec_strict_runner.py -q --basetemp $PytestBaseTemp } $log
    if ($exitCode -ne 0) { throw "KGRec unit suite failed with exit=$exitCode; see $log" }
}

function Export-Atomic($Job) {
    $split = Resolve-RunPath $Repo $Job.SplitRoot
    $atomic = Resolve-RunPath $Repo $Job.AtomicDir
    New-Item -ItemType Directory -Force -Path $atomic | Out-Null
    $log = Join-Path $QueueDir ("export_{0}_seed{1}.log" -f $Job.Dataset.ToLowerInvariant(), $Job.Seed)
    $code = "from paper_aaai27.scripts.kgrec_strict_adapter import export_recbole_kgrec_dataset; import sys; export_recbole_kgrec_dataset(split_root=sys.argv[1], link_path=sys.argv[2], kg_path=sys.argv[3], output_dir=sys.argv[4])"
    Write-QueueLog "EXPORT dataset=$($Job.Dataset) seed=$($Job.Seed) atomic=$atomic"
    Set-QueueStatus "running" "export:$($Job.Dataset):$($Job.Seed)" "Exporting strict KGRec atomic dataset"
    $exitCode = Invoke-NativeLogged { & $PythonRunnerAbs -c $code $split $Job.LinkPath $Job.KgPath $atomic } $log
    if ($exitCode -ne 0) { throw "Atomic export failed with exit=$exitCode for $($Job.Dataset) seed=$($Job.Seed); see $log" }
    $manifestPath = Join-Path $atomic "strict_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Exporter did not create $manifestPath" }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if (-not [bool]$manifest.strict_checks.cold_items_absent_from_train) { throw "Strict split check failed for $($Job.Dataset) seed=$($Job.Seed)" }
    if (-not [bool]$manifest.strict_checks.all_cold_items_have_kg_edges) { throw "Cold KG coverage failed for $($Job.Dataset) seed=$($Job.Seed)" }
}

function Clear-IncompleteOutput($Job) {
    $output = Resolve-RunPath $Repo $Job.OutputDir
    if (-not (Test-Path -LiteralPath $output)) { return }
    $rootFull = [System.IO.Path]::GetFullPath($ResultRoot).TrimEnd('\') + '\'
    $outputFull = [System.IO.Path]::GetFullPath($output)
    if (-not $outputFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean KGRec output outside result root: $outputFull"
    }
    Write-QueueLog "CLEAN incomplete output dataset=$($Job.Dataset) seed=$($Job.Seed) path=$outputFull"
    Remove-Item -LiteralPath $outputFull -Recurse -Force
}

function Run-KGRecJob($Job) {
    $atomic = Resolve-RunPath $Repo $Job.AtomicDir
    $output = Resolve-RunPath $Repo $Job.OutputDir
    Clear-IncompleteOutput $Job
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $log = Join-Path $QueueDir ("run_{0}_seed{1}.log" -f $Job.Dataset.ToLowerInvariant(), $Job.Seed)
    $lrText = if ($Job.LearningRate -eq 1e-5) { "1e-5" } else { "1e-6" }
    Write-QueueLog "RUN dataset=$($Job.Dataset) seed=$($Job.Seed) lr=$lrText epochs=$($Job.Epochs) patience=$($Job.Patience) batch=$($Job.BatchSize) epoch0_diagnostic_only=true"
    Set-QueueStatus "running" "train:$($Job.Dataset):$($Job.Seed)" "Training KGRec serially"
    $exitCode = Invoke-NativeLogged {
        & $PythonRunnerAbs $Runner `
            --atomic-dir $atomic `
            --output-dir $output `
            --seed $Job.Seed `
            --epochs $Job.Epochs `
            --batch-size $Job.BatchSize `
            --eval-batch-size 4096 `
            --eval-every 1 `
            --patience $Job.Patience `
            --dim 64 `
            --lr $Job.LearningRate `
            --l2 1e-5 `
            --context-hops 2 `
            --device cuda `
            --epoch0-diagnostic-only
    } $log
    if ($exitCode -ne 0) { throw "KGRec run failed with exit=$exitCode for $($Job.Dataset) seed=$($Job.Seed); see $log" }
}

function Get-SampleStd([double[]]$Values) {
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = ($Values | Measure-Object -Average).Average
    $sum = 0.0
    foreach ($value in $Values) { $sum += [math]::Pow($value - $mean, 2) }
    return [math]::Sqrt($sum / ($Values.Count - 1))
}

function Write-MainTableSummary {
    Write-QueueLog "SUMMARY 3 datasets x 3 seeds -> $SummaryPath"
    Set-QueueStatus "running" "summary" "Validating all nine reports and aggregating metrics"
    $metricNames = @("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
    $datasetRows = @()
    $csvRows = @()
    foreach ($dataset in @("MOOCCube", "COCO", "Junyi")) {
        $seedRows = @()
        foreach ($job in @($jobs | Where-Object Dataset -eq $dataset | Sort-Object Seed)) {
            $validation = Get-ValidationResult $job
            if (-not $validation.Valid) { throw "Final validation failed for $dataset seed=$($job.Seed): $($validation.Reason)" }
            $metrics = [ordered]@{}
            foreach ($metric in $metricNames) { $metrics[$metric] = [double]$validation.Report.test.full_cold_item_macro.$metric }
            $seedRows += [ordered]@{
                seed = [int]$job.Seed
                best_epoch = [int]$validation.Report.best_epoch
                validation_n10 = [double]$validation.Report.best_validation_score
                test_cold_item_macro = $metrics
                report_path = [string]$validation.ReportPath
            }
        }
        $aggregate = [ordered]@{}
        foreach ($metric in $metricNames) {
            [double[]]$values = @($seedRows | ForEach-Object { [double]$_.test_cold_item_macro[$metric] })
            $mean = ($values | Measure-Object -Average).Average
            $std = Get-SampleStd $values
            $aggregate[$metric] = [ordered]@{ mean=[double]$mean; sample_std=[double]$std }
            $csvRows += [pscustomobject]@{
                dataset=$dataset; metric=$metric; mean=[double]$mean; sample_std=[double]$std
                seed_2025=[double]$values[0]; seed_2026=[double]$values[1]; seed_2027=[double]$values[2]
            }
        }
        $datasetRows += [ordered]@{ dataset=$dataset; seeds=$seedRows; aggregate=$aggregate }
    }
    $summary = [ordered]@{
        model = "KGRec (adapted)"
        status = "main_table_ready"
        protocol = "strict_item_cold_full_catalog_item_macro"
        generated_at = (Get-Date).ToString("o")
        datasets = $datasetRows
    }
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
    $csvRows | Export-Csv -LiteralPath $SummaryCsvPath -NoTypeInformation -Encoding UTF8
}

if ($DryRun) {
    Write-Host "QUEUE PLAN KGRec remaining main-table serial"
    Write-Host "TEST KGRec unit suite"
    Write-Host "pytest_basetemp=paper_aaai27\baseline_sources\_kgrec_strict\_remaining_main_table_queue\_pytest_tmp"
    foreach ($job in $jobs) {
        Write-Host "VALIDATE dataset=$($job.Dataset) seed=$($job.Seed) report=$(Join-Path (Resolve-RunPath $Repo $job.OutputDir) 'kgrec_strict_adapter_report.json')"
        if ($job.Runnable) {
            $lrText = if ($job.LearningRate -eq 1e-5) { "1e-5" } else { "1e-6" }
            Write-Host "EXPORT dataset=$($job.Dataset) seed=$($job.Seed) atomic=$(Resolve-RunPath $Repo $job.AtomicDir) (skipped at runtime if report validates)"
            Write-Host "RUN dataset=$($job.Dataset) seed=$($job.Seed) lr=$lrText epochs=$($job.Epochs) patience=$($job.Patience) batch=$($job.BatchSize) epoch0_diagnostic_only=true"
        }
    }
    Write-Host "SUMMARY 3 datasets x 3 seeds -> $SummaryPath"
    Write-Host "STATUS -> $StatusPath"
    exit 0
}

if ($ValidateExistingOnly) {
    Assert-Inputs
    $validCount = 0
    $pendingCount = 0
    foreach ($job in $jobs) {
        $validation = Get-ValidationResult $job
        if ($validation.Valid) {
            Write-Host "PREFLIGHT VALID dataset=$($job.Dataset) seed=$($job.Seed)"
            $validCount += 1
        } elseif ($job.Runnable -and $validation.Reason -eq "missing report") {
            Write-Host "PREFLIGHT PENDING dataset=$($job.Dataset) seed=$($job.Seed)"
            $pendingCount += 1
        } else {
            throw "PREFLIGHT INVALID dataset=$($job.Dataset) seed=$($job.Seed) reason=$($validation.Reason)"
        }
    }
    Write-Host "PREFLIGHT PASS existing_results=$validCount pending_runs=$pendingCount"
    exit 0
}

New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
try {
    Assert-Inputs
    Set-QueueStatus "running" "startup" "Queue initialized"
    Write-QueueLog "QUEUE START KGRec remaining main-table serial"
    Invoke-KGRecTests
    foreach ($job in $jobs) {
        $validation = Get-ValidationResult $job
        Write-QueueLog "VALIDATE dataset=$($job.Dataset) seed=$($job.Seed) valid=$($validation.Valid) reason=$($validation.Reason)"
        if ($validation.Valid -and (-not $RerunCompleted)) {
            Write-QueueLog "SKIP valid result dataset=$($job.Dataset) seed=$($job.Seed)"
            continue
        }
        if (-not $job.Runnable) {
            throw "Required existing result is invalid and is not in the remaining-run set: dataset=$($job.Dataset) seed=$($job.Seed) reason=$($validation.Reason)"
        }
        Export-Atomic $job
        Run-KGRecJob $job
        $after = Get-ValidationResult $job
        Write-QueueLog "POST-VALIDATE dataset=$($job.Dataset) seed=$($job.Seed) valid=$($after.Valid) reason=$($after.Reason)"
        if (-not $after.Valid) { throw "Completed run failed strict validation: dataset=$($job.Dataset) seed=$($job.Seed) reason=$($after.Reason)" }
    }
    Write-MainTableSummary
    Set-QueueStatus "complete" "done" "All nine KGRec reports validated; main-table summary generated"
    Write-QueueLog "QUEUE DONE KGRec main_table_ready summary=$SummaryPath"
} catch {
    Set-QueueStatus "failed" "error" $_.Exception.Message
    Write-QueueLog "QUEUE FAILED $($_.Exception.Message)"
    throw
}

Write-Host "Queue log: $QueueLog"
Write-Host "Status: $StatusPath"
Write-Host "Summary JSON: $SummaryPath"
Write-Host "Summary CSV: $SummaryCsvPath"
