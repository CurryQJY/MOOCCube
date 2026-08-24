# ColdRec GAR Strict Single-Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-faithful ColdRec GAR seed-2025 feasibility experiment under the MOOCCube strict full-catalog course-cold protocol.

**Architecture:** A new adapter reuses the existing ColdRec dataset exporter and external embedding evaluator, instantiates the released GAR class through ColdRec's factory, and binds a strict validation callback without editing ColdRec source. A PowerShell runner first exports the dataset and trains the released MF teacher, then launches GAR on CUDA and verifies its artifacts.

**Tech Stack:** Python 3.12, PyTorch, pandas, ColdRec commit `18efd24`, pytest/unittest, PowerShell, CUDA.

---

### Task 1: Adapter Contract Tests

**Files:**
- Create: `tests/test_gar_coldrec_static.py`

- [ ] **Step 1: Write failing tests for source selection and CLI construction**

Add tests that import `gar_coldrec_static`, construct a minimal config, call
`_build_coldrec_argv`, and assert `--model GAR`, `--backbone MF`, seed 2025,
one run, CUDA flags, and no M2VAE/FS-GNN-only parameters.

- [ ] **Step 2: Write failing tests for MF teacher requirements**

Create a temporary ColdRec `emb` directory and assert
`require_mf_embeddings(...)` raises `FileNotFoundError` until both matching
`*_MF_user_emb.pt` and `*_MF_item_emb.pt` files exist.

- [ ] **Step 3: Write failing tests for strict validation selection**

Use a tiny fake GAR trainer and a deterministic validation function. Assert the
bound callback calls `save()` on an improved cold item-macro N@10, does not save
on regression, updates patience, and records the best epoch/score.

- [ ] **Step 4: Run the tests and verify RED**

Run:

```powershell
.\py.bat -m pytest tests\test_gar_coldrec_static.py -q
```

Expected: collection/import failure because `gar_coldrec_static.py` does not
exist.

### Task 2: Strict GAR Adapter

**Files:**
- Create: `gar_coldrec_static.py`
- Reuse without modification: `fsgnn_coldrec_static.py`
- Reuse without modification: `hin_data_common.py`
- Reuse without modification: `hin_eval_common.py`
- Reuse without modification: `tmp/candidate_repos/ColdRec/model/GAR.py`

- [ ] **Step 1: Implement configuration and command construction**

Add a `Config` dataclass/loader with explicit data, split, output, ColdRec root,
dataset name, seed, epochs, embedding size, batch size, learning rate,
regularization, top-N, device, early stop, validation interval, evaluation batch
size, history policy, backbone, and GAR alpha/beta parameters.

- [ ] **Step 2: Implement strict input validation**

Use `load_static_split` and `export_coldrec_dataset`. Verify every held-out item
with `popularity < cold_threshold` is absent from train item IDs. Verify content
rows equal catalog size and require the matching MF embedding files.

- [ ] **Step 3: Bind strict validation callback**

Bind an instance method to the ColdRec GAR trainer. At each released
`fast_evaluation` call, restore current mapped user/item embeddings to source
order, evaluate validation cold item-macro N@10 with full ranking and train-only
history masking, call the released `save()` on improvement, and implement the
existing patience contract.

- [ ] **Step 4: Execute released GAR training and external test evaluation**

Instantiate through `Config -> model_factory`, bind the callback, call GAR's
released `train()`, restore the retained best embeddings, and evaluate full
cold/hot interaction-macro and item-macro R/N at 5/10/20. Export per-item CSVs,
JSON, Markdown report, source status, protocol gates, and runtime metadata.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run:

```powershell
.\py.bat -m pytest tests\test_gar_coldrec_static.py -q
```

Expected: all tests pass.

### Task 3: Serial Runner Tests and Implementation

**Files:**
- Create: `tests/test_gar_coldrec_single_seed_serial.ps1`
- Create: `run_gar_coldrec_single_seed.ps1`

- [ ] **Step 1: Write the failing PowerShell contract test**

Assert the runner exists and contains the seed-2025 balanced split, output path,
ColdRec dataset name, MF-before-GAR ordering, 5 MF epochs, 10 GAR epochs, CUDA
flag, train-only history, and expected result filename.

- [ ] **Step 2: Run the PowerShell test and verify RED**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tests\test_gar_coldrec_single_seed_serial.ps1
```

Expected: failure because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Follow `run_m2vae_coldrec_single_seed_serial.ps1` for path resolution, dataset
export, MF execution, environment setup, log capture, failure propagation, and
artifact checks, but keep only one MOOCCube seed-2025 task and GAR parameters.

- [ ] **Step 4: Run the PowerShell test and verify GREEN**

Run the same command and expect `GAR ColdRec single-seed runner contract: PASS`.

### Task 4: Adapter Verification and Smoke Run

**Files:**
- Verify: `gar_coldrec_static.py`
- Verify: `run_gar_coldrec_single_seed.ps1`
- Verify: targeted existing ColdRec adapter tests

- [ ] **Step 1: Run Python syntax checks**

```powershell
.\py.bat -m py_compile gar_coldrec_static.py
```

- [ ] **Step 2: Run targeted regression tests**

```powershell
.\py.bat -m pytest tests\test_gar_coldrec_static.py tests\test_m2vae_coldrec_static.py tests\test_fsgnn_coldrec_static.py -q
```

- [ ] **Step 3: Run a one-epoch uncached smoke experiment**

Run the serial script with `-MFEpochs 1 -GAREpochs 1 -Force` and confirm CUDA,
nonempty strict validation/test cold sets, a retained checkpoint, JSON, and
per-item files.

### Task 5: Seed-2025 GPU Feasibility Run

**Files:**
- Output: `paper_aaai27/baseline_sources/_gar_coldrec_strict/mooccube_seed2025_single/`

- [ ] **Step 1: Launch the configured run**

```powershell
powershell -ExecutionPolicy Bypass -File run_gar_coldrec_single_seed.ps1 -Force
```

- [ ] **Step 2: Monitor to completion**

Do not end the task while MF or GAR is running. Read `mf_backbone.log`,
`gar_training.log`, process status, and GPU state until completion or a concrete
failure is recorded.

- [ ] **Step 3: Audit artifacts and protocol gates**

Verify result JSON fields, per-item row counts, strict split path, train-only
history, full-catalog candidate mode, item-macro metrics, finite losses, CUDA
device, ColdRec commit/status, and zero held-out-cold/train overlap.

- [ ] **Step 4: Report the single-seed decision**

State whether GAR is source-faithful, protocol-valid, nondegenerate, and worth
expanding. Do not edit the paper main table or launch additional seeds.

