# CBI-Faithful Single-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run one isolated, reproducible MOOCCube seed-2025 experiment using the existing CBI-faithful frozen PCA content base plus a bounded delta with maximum norm 0.5, without modifying any main-table source or result file.

**Architecture:** A new PowerShell launcher supplies the complete formal main-configuration parameter set plus the CBI overrides to the existing static runner. A separate Python summarizer reads the new strict course-macro report, the retained seed-2025 reference, validation history, and delta diagnostics from the run log. Tests inspect the launcher contract and exercise the summarizer with temporary fixtures before the real experiment is launched.

**Tech Stack:** PowerShell 5.1, Python 3.12, PyTorch 2.8, pytest, CSV/JSON, existing `run_usim_feedback_fast3_content_delta_static.ps1` and `usim_feedback_fast3_content_delta.py`.

---

## File Structure

- Create `run_cbi_faithful_seed2025.ps1`: isolated launcher, path guards, locked configuration, source hashes, runtime metadata, protected-file hashes, synchronous training call, completion manifest, and log capture.
- Create `summarize_cbi_faithful_seed2025.py`: parse strict course-macro metrics, select the validation epoch by cold NDCG@10, parse final delta statistics, compare against the retained seed-2025 reference, and write JSON/CSV/Markdown summaries.
- Create `tests/test_cbi_faithful_single_run.py`: launcher contract, protected path, selected configuration, result parsing, delta parsing, and screening-decision tests.
- Create at runtime `outputs/cbi_faithful_single_seed2025/run_manifest.json`: immutable run configuration and mutable execution status.
- Create at runtime `outputs/cbi_faithful_single_seed2025/cbi_comparison.{json,csv,md}`: experiment-versus-reference summary.
- Create at runtime `background_logs/cbi_faithful_single_seed2025/training.log`: complete runner output.

Protected files remain unchanged:

- `usim_feedback_fast3_content_delta.py`
- `run_fast3_main_table_config.ps1`
- `paper_aaai27/main.tex`
- `paper_aaai27/main_table.tex`

### Task 1: Define Launcher and Summary Contracts with Failing Tests

**Files:**
- Create: `tests/test_cbi_faithful_single_run.py`
- Test: `tests/test_cbi_faithful_single_run.py`

- [ ] **Step 1: Write launcher contract tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_cbi_faithful_seed2025.ps1"


def test_launcher_is_isolated_and_uses_cbi_faithful_configuration():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'outputs\\cbi_faithful_single_seed2025' in text
    assert 'checkpoints\\cbi_faithful_single_seed2025' in text
    assert 'background_logs\\cbi_faithful_single_seed2025' in text
    assert 'ContentDeltaPaperStyle = $true' in text
    assert 'ContentDeltaReplaceItem = $true' in text
    assert 'ContentDeltaColdOnly = $false' in text
    assert 'ContentDeltaMaxNorm = 0.5' in text
    assert 'ContentDeltaScale = 1.0' in text
    assert 'ContentDeltaLrMult = 1.0' in text
    assert 'ContentDeltaL2W = 0.0' in text
    assert 'ContentDeltaCapW = 0.0' in text
    assert 'ContentDeltaTrainOnIdDropout = $false' in text
    assert 'Seeds = @(2025)' in text
    assert 'Epochs = 60' in text
    assert 'Patience = 60' in text


def test_launcher_does_not_target_main_table_outputs_or_sources():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'course_maincfg_runs\\maincfg' not in text
    assert 'course_ablation_e60_3seed\\full' not in text
    assert 'Set-Content "paper_aaai27' not in text
    assert 'Set-Content "usim_feedback_fast3_content_delta.py' not in text
    assert 'protected_files_before' in text
    assert 'protected_files_after' in text
```

- [ ] **Step 2: Write summary parser tests**

```python
import csv

from summarize_cbi_faithful_seed2025 import (
    build_comparison,
    parse_delta_stats,
    read_report,
    select_validation_epoch,
)


def _write_report(path, r10, n10):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "full_cold",
                "full_hot",
                "full_cold_item_macro",
                "full_hot_item_macro",
            ],
        )
        writer.writeheader()
        values = {
            "R@5": (r10 - 0.04, 0.18),
            "R@10": (r10, 0.20),
            "R@20": (r10 + 0.04, 0.24),
            "N@5": (n10 - 0.02, 0.08),
            "N@10": (n10, 0.10),
            "N@20": (n10 + 0.02, 0.12),
        }
        for metric, (cold, hot) in values.items():
            writer.writerow({"metric": metric, "full_cold_item_macro": cold, "full_hot_item_macro": hot})


def test_summary_uses_item_macro_metrics_and_screening_rule(tmp_path):
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write_report(candidate, 0.2540, 0.1870)
    _write_report(baseline, 0.2530, 0.1830)
    result = build_comparison(read_report(candidate), read_report(baseline))
    assert result["metrics"]["N@10"]["cold_delta"] == 0.004
    assert result["metrics"]["R@10"]["cold_delta"] == 0.001
    assert result["screening"]["promising"] is True


def test_validation_epoch_is_selected_by_cold_ndcg10(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "Epoch,Val_full_cold_R@10,Val_full_hot_R@10,Val_full_cold_N@10,Val_full_hot_N@10\n"
        "1,0.20,0.10,0.15,0.08\n"
        "2,0.21,0.11,0.18,0.09\n",
        encoding="utf-8",
    )
    assert select_validation_epoch(history)["epoch"] == 2


def test_delta_stats_parser_reads_last_epoch_diagnostics(tmp_path):
    log = tmp_path / "training.log"
    log.write_text(
        "DeltaNorm[mean=0.1000, max=0.5000, eff_mean=0.1000, eff_max=0.5000, clip=12.50%]\n"
        "DeltaNorm[mean=0.2000, max=0.5000, eff_mean=0.2000, eff_max=0.5000, clip=25.00%]\n",
        encoding="utf-8",
    )
    stats = parse_delta_stats(log)
    assert stats == {
        "mean_norm": 0.2,
        "max_norm": 0.5,
        "effective_mean_norm": 0.2,
        "effective_max_norm": 0.5,
        "clipped_ratio": 0.25,
    }
```

- [ ] **Step 3: Run tests and verify the missing files fail**

Run:

```powershell
.\py.bat -m pytest tests\test_cbi_faithful_single_run.py -q
```

Expected: collection fails because `run_cbi_faithful_seed2025.ps1` and `summarize_cbi_faithful_seed2025.py` do not exist.

- [ ] **Step 4: Commit the failing tests**

```powershell
git add tests/test_cbi_faithful_single_run.py
git commit -m "test: define CBI single-run contract"
```

### Task 2: Implement the Isolated Reproducible Launcher

**Files:**
- Create: `run_cbi_faithful_seed2025.ps1`
- Test: `tests/test_cbi_faithful_single_run.py`

- [ ] **Step 1: Implement fixed paths, configuration, and path guards**

The launcher must define the selected configuration directly and reject any path that is equal to or nested inside the retained main-table roots:

```powershell
param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun,
    [switch]$ForceFresh
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$outputRoot = "outputs\cbi_faithful_single_seed2025"
$checkpointRoot = "checkpoints\cbi_faithful_single_seed2025"
$logRoot = "background_logs\cbi_faithful_single_seed2025"
$protectedRoots = @(
    "outputs\content_delta_pop5\course_maincfg_runs\maincfg",
    "outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "checkpoints\content_delta_pop5\course_maincfg_runs\maincfg",
    "checkpoints\content_delta_pop5\course_ablation_e60_3seed\full"
)

function Resolve-RepoPath([string]$Path) {
    return [System.IO.Path]::GetFullPath((Join-Path $Repo $Path))
}

foreach ($candidate in @($outputRoot, $checkpointRoot)) {
    $resolved = Resolve-RepoPath $candidate
    foreach ($protected in $protectedRoots) {
        $protectedResolved = Resolve-RepoPath $protected
        if ($resolved.Equals($protectedResolved, [System.StringComparison]::OrdinalIgnoreCase) -or
            $resolved.StartsWith($protectedResolved + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "CBI experiment path overlaps protected main-table path: $resolved"
        }
    }
}
```

- [ ] **Step 2: Implement locked configuration and protected-file hashes**

Use an ordered configuration containing every material parameter. Hash the four protected files before training and include the hashes in the manifest:

```powershell
$config = [ordered]@{
    method = "cbi_faithful_bounded_delta"
    seed = 2025
    protocol = "strict_item_cold_balanced"
    epochs = 60
    patience = 60
    delta_max_norm = 0.5
    delta_scale = 1.0
    delta_lr_mult = 1.0
    paper_style = $true
    replace_item = $true
    cold_only = $false
}

$protectedFiles = @(
    "usim_feedback_fast3_content_delta.py",
    "run_fast3_main_table_config.ps1",
    "paper_aaai27\main.tex",
    "paper_aaai27\main_table.tex"
)

function Get-HashMap([string[]]$Paths) {
    $result = [ordered]@{}
    foreach ($path in $Paths) {
        $result[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}
```

If `run_manifest.json` already exists with status `completed`, refuse to launch unless `-ForceFresh` is supplied. If a locked configuration differs, refuse regardless of resume state.

- [ ] **Step 3: Implement runtime/source provenance and manifest updates**

Record hashes for the launcher, static runner, model entry point, `fast3_delta/config.py`, `fast3_delta/eval.py`, and `fast3_delta/provenance.py`. Record Git commit and dirty state, Python/PyTorch/CUDA metadata, PowerShell/Windows metadata, explicit runner parameters, start/end timestamps, elapsed seconds, status, exit code, and protected hashes before/after.

The Python runtime command must be:

```powershell
$runtimeJson = & .\py.bat -c "import json,platform,sys,torch; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'platform':platform.platform()}))"
$pythonRuntime = $runtimeJson | ConvertFrom-Json
```

- [ ] **Step 4: Implement the explicit existing-runner invocation**

Use this complete parameter map; do not call or modify the main-table launcher:

```powershell
$runnerParams = @{
    PythonRunner = ".\py.bat"
    ScriptPath = "usim_feedback_fast3_content_delta.py"
    DataDir = "processed_data_hin_clean_pop5"
    RelationDir = "MOOCCube/relations"
    OutputRoot = $outputRoot
    CheckpointRoot = $checkpointRoot
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $true
    ContentDeltaPaperStyle = $true
    ContentDeltaReplaceItem = $true
    ContentDeltaColdOnly = $false
    ContentDeltaTrainOnIdDropout = $false
    ContentDeltaMode = "embedding"
    ContentDeltaMaxNorm = 0.5
    ContentDeltaScale = 1.0
    ContentDeltaLrMult = 1.0
    ContentDeltaL2W = 0.0
    ContentDeltaCapW = 0.0
    ContentDeltaAuxMode = "base"
    UsePseudoColdTrain = $false
    PseudoColdMode = "batch_random"
    PseudoColdRatio = 0.30
    PseudoColdMinPop = 5
    UsePaac = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UseCourseSample = $true
    UsePrereqAux = $true
    PrereqGraphSource = "concept"
    CoursePrereqW = 0.08
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseFeedbackOnlyCold = $false
    CourseSampleOnlyCold = $false
    PrereqAuxOnlyCold = $false
    CourseSampleBeta = 0.20
    UseSageLite = $false
    SageTwoExpertScoreFusion = $false
    UseSageAuxLoss = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $true
    MaskSameItemNeg = $true
    TrainForceCold = $true
    UsimSteps = 5
    UseUsimRefinedEval = $true
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    AuxHotOnly = $false
    RunSampledEval = $false
    SaveCkpt = $true
    AutoResume = $false
    ForceFresh = $true
    SaveOptState = $true
}
```

Capture the command output with `Tee-Object`, update the manifest in `finally`, and throw if any protected-file hash changes.

- [ ] **Step 5: Run launcher tests**

Run:

```powershell
.\py.bat -m pytest tests\test_cbi_faithful_single_run.py -q
```

Expected: launcher contract tests pass; summary imports still fail until Task 3.

- [ ] **Step 6: Commit the launcher**

```powershell
git add run_cbi_faithful_seed2025.ps1 tests/test_cbi_faithful_single_run.py
git commit -m "feat: add isolated CBI single-run launcher"
```

### Task 3: Implement Result Summarization and Screening

**Files:**
- Create: `summarize_cbi_faithful_seed2025.py`
- Modify: `tests/test_cbi_faithful_single_run.py`
- Test: `tests/test_cbi_faithful_single_run.py`

- [ ] **Step 1: Implement strict report parsing**

```python
def read_report(path):
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row["metric"]
            rows[metric] = {
                "cold": float(row["full_cold_item_macro"]),
                "hot": float(row["full_hot_item_macro"]),
            }
    required = {f"{prefix}@{k}" for prefix in ("R", "N") for k in (5, 10, 20)}
    missing = required - rows.keys()
    if missing:
        raise ValueError(f"missing strict item-macro metrics: {sorted(missing)}")
    return rows
```

- [ ] **Step 2: Implement validation and delta parsers**

`select_validation_epoch()` must choose the row with maximum `Val_full_cold_N@10`, matching `EarlyStopScoreMode="cold_only"`. `parse_delta_stats()` must use the last `DeltaNorm[...]` match in the training log and convert the percentage to a ratio in `[0,1]`.

- [ ] **Step 3: Implement comparison and decision rule**

```python
def build_comparison(candidate, baseline):
    metrics = {}
    for metric in sorted(candidate):
        cold_delta = candidate[metric]["cold"] - baseline[metric]["cold"]
        hot_delta = candidate[metric]["hot"] - baseline[metric]["hot"]
        metrics[metric] = {
            "candidate_cold": candidate[metric]["cold"],
            "baseline_cold": baseline[metric]["cold"],
            "cold_delta": round(cold_delta, 12),
            "candidate_hot": candidate[metric]["hot"],
            "baseline_hot": baseline[metric]["hot"],
            "hot_delta": round(hot_delta, 12),
        }
    promising = metrics["N@10"]["cold_delta"] >= 0.003 and metrics["R@10"]["cold_delta"] >= -0.002
    return {"metrics": metrics, "screening": {"promising": promising}}
```

- [ ] **Step 4: Implement CLI output**

Default paths must be:

```python
CANDIDATE_DIR = Path("outputs/cbi_faithful_single_seed2025/strict_item_cold_balanced_thr1_seed_2025")
BASELINE_DIR = Path("outputs/content_delta_pop5/course_ablation_e60_3seed/full/strict_item_cold_balanced_thr1_seed_2025")
LOG_PATH = Path("background_logs/cbi_faithful_single_seed2025/training.log")
OUTPUT_ROOT = Path("outputs/cbi_faithful_single_seed2025")
```

Write `cbi_comparison.json`, `cbi_comparison.csv`, and `cbi_comparison.md`. The Markdown output must state that the result is a one-seed screening result and cannot alter the main table.

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
.\py.bat -m pytest tests\test_cbi_faithful_single_run.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the summarizer**

```powershell
git add summarize_cbi_faithful_seed2025.py tests/test_cbi_faithful_single_run.py
git commit -m "feat: summarize CBI single-run results"
```

### Task 4: Verify Isolation and Reproducibility Before Training

**Files:**
- Verify: `run_cbi_faithful_seed2025.ps1`
- Verify: `summarize_cbi_faithful_seed2025.py`
- Verify: protected files listed above

- [ ] **Step 1: Record protected hashes outside the experiment**

Run:

```powershell
Get-FileHash usim_feedback_fast3_content_delta.py,run_fast3_main_table_config.ps1,paper_aaai27\main.tex,paper_aaai27\main_table.tex -Algorithm SHA256
```

Expected: four hashes are printed and retained for the post-run check.

- [ ] **Step 2: Run the focused and provenance tests**

Run:

```powershell
.\py.bat -m pytest tests\test_cbi_faithful_single_run.py tests\test_experiment_provenance.py tests\test_static_manifest_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Parse and dry-run the launcher**

Run:

```powershell
$errors=$null; [Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'run_cbi_faithful_seed2025.ps1'),[ref]$null,[ref]$errors) | Out-Null; if($errors.Count){$errors | ForEach-Object Message; exit 1}
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_cbi_faithful_seed2025.ps1 -DryRun
```

Expected: no parser errors; dry run prints seed 2025, 60 epochs, delta max norm 0.5, and the isolated paths without starting Python.

- [ ] **Step 4: Confirm no protected diff was introduced**

Run:

```powershell
git diff -- usim_feedback_fast3_content_delta.py run_fast3_main_table_config.ps1 paper_aaai27/main.tex paper_aaai27/main_table.tex
```

Expected: no new diff attributable to this implementation. Existing user changes, if any, remain byte-identical to the hashes recorded in Step 1.

### Task 5: Launch and Monitor the Single Experiment

**Files:**
- Runtime: `run_cbi_faithful_seed2025.ps1`
- Runtime output: `outputs/cbi_faithful_single_seed2025/**`
- Runtime checkpoint: `checkpoints/cbi_faithful_single_seed2025/**`
- Runtime log: `background_logs/cbi_faithful_single_seed2025/training.log`

- [ ] **Step 1: Launch the experiment in a hidden background PowerShell process**

Run:

```powershell
$p = Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Resolve-Path 'run_cbi_faithful_seed2025.ps1')) -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru
"CBI_PID=$($p.Id)"
```

Expected: a process ID is printed and `outputs/cbi_faithful_single_seed2025/run_manifest.json` reports status `running`.

- [ ] **Step 2: Monitor progress without starting another experiment**

Run periodically:

```powershell
$manifest='outputs\cbi_faithful_single_seed2025\run_manifest.json'
$log='background_logs\cbi_faithful_single_seed2025\training.log'
if(Test-Path $manifest){Get-Content $manifest -Raw}
if(Test-Path $log){Get-Content $log -Tail 30}
```

Expected: epoch progress and delta diagnostics advance; only one launcher/training process exists.

- [ ] **Step 3: Wait for terminal state**

Continue monitoring until the manifest status is `completed` or `failed`. On failure, preserve all artifacts and diagnose from the manifest/log; do not silently rerun with another norm.

### Task 6: Summarize, Verify, and Commit the Experiment Harness

**Files:**
- Runtime output: `outputs/cbi_faithful_single_seed2025/cbi_comparison.json`
- Runtime output: `outputs/cbi_faithful_single_seed2025/cbi_comparison.csv`
- Runtime output: `outputs/cbi_faithful_single_seed2025/cbi_comparison.md`

- [ ] **Step 1: Generate the comparison outputs**

Run:

```powershell
.\py.bat summarize_cbi_faithful_seed2025.py
```

Expected: JSON, CSV, and Markdown summaries are created and contain strict course-macro metrics, selected epoch, delta diagnostics, and the screening decision.

- [ ] **Step 2: Re-run verification tests**

Run:

```powershell
.\py.bat -m pytest tests\test_cbi_faithful_single_run.py tests\test_experiment_provenance.py tests\test_static_manifest_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify protected hashes and paths**

Run:

```powershell
Get-FileHash usim_feedback_fast3_content_delta.py,run_fast3_main_table_config.ps1,paper_aaai27\main.tex,paper_aaai27\main_table.tex -Algorithm SHA256
git status --short
```

Expected: protected hashes match their pre-run values; only the isolated harness commits and runtime artifacts are new.

- [ ] **Step 4: Commit the completed implementation harness**

Runtime outputs and checkpoints remain uncommitted. Commit only source, tests, and plan tracking changes:

```powershell
git add run_cbi_faithful_seed2025.ps1 summarize_cbi_faithful_seed2025.py tests/test_cbi_faithful_single_run.py docs/superpowers/plans/2026-07-16-cbi-faithful-single-run.md
git commit -m "feat: add reproducible CBI single-run validation"
```
