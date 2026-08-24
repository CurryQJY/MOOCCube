# UPGPR Single Seed GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one GPU-backed UPGPR strict course-cold seed as a main-table candidate gate before launching the full multi-seed matrix.

**Architecture:** Reuse the existing UPGPR strict adapter and official UPGPR training code. The run writes an isolated seed-2025 output directory, preserves strict data-leakage gates, records training throughput/progress, and evaluates full-catalog course-macro metrics plus diagnostics.

**Tech Stack:** Python, PyTorch/CUDA, pytest, PowerShell, existing `paper_aaai27/scripts/upgpr_strict_adapter.py`.

---

### Task 1: Preflight Current Adapter

**Files:**
- Read-only: `paper_aaai27/scripts/upgpr_strict_adapter.py`
- Read-only: `tests/test_upgpr_strict_adapter.py`
- Read-only: `tests/test_course_baseline_adaptability.py`

- [ ] **Step 1: Verify tests for the UPGPR adapter and strict exporter**

Run:

```powershell
.\py.bat -m pytest tests/test_upgpr_strict_adapter.py tests/test_course_baseline_adaptability.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Check GPU availability**

Run:

```powershell
nvidia-smi
```

Expected: at least one CUDA GPU is visible and memory has enough free capacity for a single UPGPR run.

### Task 2: Run One Formal-Profile Seed

**Files:**
- Execute: `paper_aaai27/scripts/upgpr_strict_adapter.py`
- Create directory: `paper_aaai27/baseline_sources/_upgpr_strict/mooccube_seed2025_single_gpu_main_candidate`

- [ ] **Step 1: Launch seed-2025 GPU run with bounded policy steps**

Run:

```powershell
.\py.bat paper_aaai27/scripts/upgpr_strict_adapter.py `
  --profile formal-throughput `
  --device cuda `
  --seed 2025 `
  --max-policy-steps 2000 `
  --checkpoint-every-steps 500 `
  --output paper_aaai27/baseline_sources/_upgpr_strict/mooccube_seed2025_single_gpu_main_candidate
```

Expected: the command exits successfully and writes `upgpr_strict_feasibility_report.json`.

- [ ] **Step 2: If the run exceeds the interactive window, poll progress**

Run:

```powershell
Get-Content -Tail 40 paper_aaai27\baseline_sources\_upgpr_strict\mooccube_seed2025_single_gpu_main_candidate\train_policy.log
```

Expected: policy steps advance and checkpoints are saved every 500 steps.

### Task 3: Audit Single-Seed Result

**Files:**
- Read-only: `paper_aaai27/baseline_sources/_upgpr_strict/mooccube_seed2025_single_gpu_main_candidate/upgpr_strict_feasibility_report.json`
- Read-only: `paper_aaai27/baseline_sources/_upgpr_strict/mooccube_seed2025_single_gpu_main_candidate/upgpr_strict_feasibility_report.md`

- [ ] **Step 1: Extract strict gates and metrics**

Run:

```powershell
@'
import json
from pathlib import Path
p = Path("paper_aaai27/baseline_sources/_upgpr_strict/mooccube_seed2025_single_gpu_main_candidate/upgpr_strict_feasibility_report.json")
r = json.loads(p.read_text())
print("verdict:", r["verdict"])
print("strict_gates:", r["strict_gates"])
print("validation:", r["validation"])
print("test:", r["test"])
print("training_config:", r["training_config"])
'@ | .\py.bat -
```

Expected: all boolean strict gates are true; validation/test rows are nonempty; report includes `full_cold_item_macro`, `transe_full_candidate_fallback_item_macro`, and `native_path_proxy_item_macro`.

- [ ] **Step 2: Decide next action**

Pass criteria for launching the full matrix:

```text
verdict == FEASIBLE_FOR_FORMALIZATION
all boolean strict_gates == True
test.full_cold_item_macro.count > 0
test.full_cold_item_macro["N@10"] > 0 or transe fallback gives a defensible nonzero diagnostic that motivates a scorer patch before full matrix
```

If full-catalog UPGPR remains exactly zero while TransE fallback is nonzero, pause before multi-seed and patch the policy/score fusion path rather than spending GPU on three seeds.
