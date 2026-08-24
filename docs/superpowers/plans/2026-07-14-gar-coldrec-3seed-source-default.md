# ColdRec GAR Source-Default Three-Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and aggregate a source-default ColdRec GAR baseline on MOOCCube seeds 2025, 2026, and 2027 under the strict full-catalog course-cold protocol.

**Architecture:** A dedicated Python aggregator validates each existing GAR strict result before computing three-seed statistics. A PowerShell queue calls the tested single-seed runner serially with ColdRec's 500-epoch ceilings, then invokes the aggregator only after all requested seeds have valid result files.

**Tech Stack:** Python 3.12, pandas, pytest/unittest, PowerShell, PyTorch/ColdRec, CUDA.

---

### Task 1: Strict Three-Seed Aggregator

**Files:**
- Create: `tests/test_aggregate_gar_coldrec_3seed.py`
- Create: `aggregate_gar_coldrec_3seed.py`

- [ ] **Step 1: Write failing tests for strict validation and statistics**

Create synthetic seed directories containing `gar_coldrec_strict_result.json`
and matching per-course CSVs. The result factory must populate the same fields
as the real adapter:

```python
def make_result(root: Path, seed: int, cold_n10: float) -> Path:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    cold_csv = seed_dir / "per_item_full_cold_gar_coldrec.csv"
    hot_csv = seed_dir / "per_item_full_hot_gar_coldrec.csv"
    pd.DataFrame({"item_id": [1, 2], "N@10": [cold_n10, cold_n10]}).to_csv(cold_csv, index=False)
    pd.DataFrame({"item_id": [3], "N@10": [0.2]}).to_csv(hot_csv, index=False)
    payload = [{
        "seed": seed,
        "official_commit": "18efd24",
        "source_model_unchanged": True,
        "candidate_mode": "full_catalog",
        "checkpoint_metric": "validation_full_cold_item_macro.N@10",
        "train_history_masking": True,
        "train_only_interaction_evidence": True,
        "test_history_policy": "train_only",
        "cuda_used": True,
        "device": "cuda:0",
        "strict_audit": {"heldout_cold_item_count": 3, "train_overlap_count": 0},
        "counts": {"full_cold": 10, "full_hot": 5, "full_cold_item": 2, "full_hot_item": 1},
        "best_epoch": 7,
        "best_val_full_cold_item_macro_n10": cold_n10,
        "strict_validation_history": [{"epoch": 7, "item_count": 1, "N@10": cold_n10}],
        "full_cold": {metric: cold_n10 for metric in METRICS},
        "full_hot": {metric: 0.2 for metric in METRICS},
        "full_cold_item_macro": {metric: cold_n10 for metric in METRICS},
        "full_hot_item_macro": {metric: 0.2 for metric in METRICS},
        "per_item_full_cold_path": str(cold_csv),
        "per_item_full_hot_path": str(hot_csv),
    }]
    path = seed_dir / "gar_coldrec_strict_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
```

Test `aggregate(root, [2025, 2026, 2027], out_dir)` with cold N@10 values
`0.1, 0.2, 0.3`. Assert the mean is `0.2`, sample standard deviation is `0.1`,
all four output files exist, and detail rows preserve seed order. Add rejection
tests for a missing seed, nonzero train overlap, nonfinite metric, and per-course
row-count mismatch.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\py.bat -m pytest tests\test_aggregate_gar_coldrec_3seed.py -q
```

Expected: import failure because `aggregate_gar_coldrec_3seed.py` does not yet
exist.

- [ ] **Step 3: Implement the strict aggregator**

Implement these public boundaries:

```python
METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
SECTIONS = ("full_cold", "full_hot", "full_cold_item_macro", "full_hot_item_macro")

def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(result, dict):
        raise ValueError(f"Invalid GAR result payload: {path}")
    return result

def validate_result(result: dict, path: Path, expected_seed: int) -> None:
    required_equal = {
        "seed": expected_seed,
        "official_commit": "18efd24",
        "source_model_unchanged": True,
        "candidate_mode": "full_catalog",
        "checkpoint_metric": "validation_full_cold_item_macro.N@10",
        "train_history_masking": True,
        "train_only_interaction_evidence": True,
        "test_history_policy": "train_only",
        "cuda_used": True,
    }
    for key, expected in required_equal.items():
        if result.get(key) != expected:
            raise ValueError(f"seed {expected_seed}: expected {key}={expected!r}, got {result.get(key)!r}")
    if result.get("strict_audit", {}).get("train_overlap_count") != 0:
        raise ValueError(f"seed {expected_seed}: held-out cold/train overlap is nonzero")
    history = result.get("strict_validation_history") or []
    if not history or not all(int(row.get("item_count", 0)) > 0 for row in history):
        raise ValueError(f"seed {expected_seed}: empty strict validation cold-course set")
    for section in SECTIONS:
        block = result.get(section)
        if not isinstance(block, dict):
            raise ValueError(f"seed {expected_seed}: missing {section}")
        for metric in METRICS:
            value = float(block[metric])
            if not math.isfinite(value):
                raise ValueError(f"seed {expected_seed}: nonfinite {section}.{metric}")
    counts = result.get("counts", {})
    if int(counts.get("full_cold_item", 0)) <= 0:
        raise ValueError(f"seed {expected_seed}: empty test cold-course set")
    for key, count_key in (("per_item_full_cold_path", "full_cold_item"), ("per_item_full_hot_path", "full_hot_item")):
        csv_path = Path(result[key])
        if len(pd.read_csv(csv_path)) != int(counts[count_key]):
            raise ValueError(f"seed {expected_seed}: per-course row-count mismatch for {csv_path}")
```

`aggregate` locates exactly one result under each `seed_<seed>` directory,
calls `validate_result`, creates one detail row per seed, and calculates
`statistics.mean` plus `statistics.stdev` for every section/metric pair. Write:

- `gar_coldrec_3seed_detail.csv`
- `gar_coldrec_3seed_summary.csv`
- `gar_coldrec_3seed_summary.json`
- `gar_coldrec_3seed_report.md`

Expose CLI arguments `--root`, `--seeds`, and `--out-dir`.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the Task 1 pytest command. Expected: all aggregator tests pass.

- [ ] **Step 5: Commit the aggregator**

```powershell
git add aggregate_gar_coldrec_3seed.py tests/test_aggregate_gar_coldrec_3seed.py
git commit -m "feat: aggregate strict GAR three-seed results"
```

### Task 2: Source-Default Serial Runner

**Files:**
- Create: `tests/test_gar_coldrec_3seed_serial.ps1`
- Create: `run_gar_coldrec_3seed_serial.ps1`
- Reuse without modification: `run_gar_coldrec_single_seed.ps1`

- [ ] **Step 1: Write the failing PowerShell contract test**

The test invokes the new runner with `-DryRun` and a temporary output root,
then asserts output contains seeds 2025, 2026, and 2027 in increasing order,
`MF epochs=500`, `GAR epochs=500`, `early_stop=5`, `use_gpu=True`, the strict
split naming pattern, the single-seed runner path, and all four aggregate
filenames. It also asserts each seed appears exactly once.

```powershell
$out = & $script -OutputRoot (Join-Path $tmpRoot "formal") -DryRun *>&1
$text = $out -join "`n"
foreach ($needle in @(
    "seed=2025", "seed=2026", "seed=2027",
    "MF epochs=500", "GAR epochs=500", "early_stop=5",
    "use_gpu=True", "run_gar_coldrec_single_seed.ps1",
    "gar_coldrec_3seed_summary.json", "gar_coldrec_3seed_report.md"
)) {
    if ($text -notmatch [regex]::Escape($needle)) { throw "Missing $needle" }
}
```

- [ ] **Step 2: Run the contract test and verify RED**

```powershell
powershell -ExecutionPolicy Bypass -File tests\test_gar_coldrec_3seed_serial.ps1
```

Expected: failure because the runner does not exist.

- [ ] **Step 3: Implement the serial queue**

Use parameters:

```powershell
param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$OutputRoot = "paper_aaai27\baseline_sources\_gar_coldrec_strict\mooccube_source_default_3seed",
    [int[]]$Seeds = @(2025, 2026, 2027),
    [int]$MFEpochs = 500,
    [int]$GAREpochs = 500,
    [int]$EarlyStop = 5,
    [int]$GpuId = 0,
    [bool]$UseGpu = $true,
    [switch]$Force,
    [switch]$DryRun
)
```

Resolve absolute paths, verify the seed list is exactly `2025,2026,2027`, and
print every task before execution. For each seed, call the existing runner
synchronously:

```powershell
& $SingleRunner `
    -Repo $RepoAbs `
    -PythonRunner $PythonRunnerAbs `
    -OutputDir (Join-Path $OutputRootAbs ("seed_{0}" -f $seed)) `
    -Seed $seed `
    -MFEpochs $MFEpochs `
    -GAREpochs $GAREpochs `
    -EarlyStop $EarlyStop `
    -GpuId $GpuId `
    -UseGpu $UseGpu `
    -Force:$Force
```

After each call, require the expected JSON. After all three seeds, invoke:

```powershell
& $PythonRunnerAbs -u $Aggregator `
    --root $OutputRootAbs `
    --seeds ($Seeds -join ",") `
    --out-dir (Join-Path $OutputRootAbs "aggregate")
```

Write timestamped START/END/FAIL lines to `_logs/<run-id>/queue.log`. Do not
create files in dry-run mode.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run the Task 2 PowerShell command. Expected:
`GAR ColdRec three-seed runner contract: PASS`.

- [ ] **Step 5: Commit the runner**

```powershell
git add run_gar_coldrec_3seed_serial.ps1 tests/test_gar_coldrec_3seed_serial.ps1
git commit -m "feat: run source-default GAR three-seed queue"
```

### Task 3: Preflight Verification

**Files:**
- Verify: `aggregate_gar_coldrec_3seed.py`
- Verify: `run_gar_coldrec_3seed_serial.ps1`
- Verify: existing GAR adapter tests

- [ ] **Step 1: Run Python syntax checks**

```powershell
.\py.bat -m py_compile gar_coldrec_static.py aggregate_gar_coldrec_3seed.py
```

- [ ] **Step 2: Run focused Python regression tests**

```powershell
.\py.bat -m pytest tests\test_gar_coldrec_static.py tests\test_aggregate_gar_coldrec_3seed.py -q
```

- [ ] **Step 3: Run both PowerShell contracts**

```powershell
powershell -ExecutionPolicy Bypass -File tests\test_gar_coldrec_single_seed_serial.ps1
powershell -ExecutionPolicy Bypass -File tests\test_gar_coldrec_3seed_serial.ps1
```

- [ ] **Step 4: Run queue dry-run and inspect seed order**

```powershell
powershell -ExecutionPolicy Bypass -File run_gar_coldrec_3seed_serial.ps1 -DryRun
```

Expected: exactly three serial tasks, source-default ceilings, CUDA true, and
no output-directory writes.

### Task 4: Formal GPU Run

**Files:**
- Output: `paper_aaai27/baseline_sources/_gar_coldrec_strict/mooccube_source_default_3seed/`

- [ ] **Step 1: Confirm GPU capacity and no duplicate GAR queue**

Run `nvidia-smi` and inspect active PowerShell/Python command lines. Do not stop
unrelated experiments.

- [ ] **Step 2: Launch the serial queue from a hidden PowerShell process**

```powershell
powershell -ExecutionPolicy Bypass -File run_gar_coldrec_3seed_serial.ps1 -Force
```

- [ ] **Step 3: Monitor every seed to completion**

Read the queue log and each seed's `mf_backbone.log` and `gar_training.log`.
Require finite losses, CUDA device usage, strict validation with nonzero cold
course counts, and a retained checkpoint. Do not launch a second queue.

### Task 5: Final Audit and Decision

**Files:**
- Verify: `aggregate/gar_coldrec_3seed_detail.csv`
- Verify: `aggregate/gar_coldrec_3seed_summary.csv`
- Verify: `aggregate/gar_coldrec_3seed_summary.json`
- Verify: `aggregate/gar_coldrec_3seed_report.md`

- [ ] **Step 1: Re-run the aggregator against completed artifacts**

Run the aggregator CLI explicitly and require exit code 0.

- [ ] **Step 2: Inspect seed coverage and statistics**

Require exactly seeds 2025, 2026, and 2027, `runs=3`, finite means and sample
standard deviations, and matching per-course row counts.

- [ ] **Step 3: Run fresh regression verification**

Repeat Task 3 syntax and test commands before making completion claims.

- [ ] **Step 4: Report results without editing the main table**

Report source fidelity, protocol validity, per-seed values, mean plus standard
deviation, epochs selected, and whether GAR is competitive enough for later
main-table insertion. Do not modify `paper_aaai27/main_table.tex`.
