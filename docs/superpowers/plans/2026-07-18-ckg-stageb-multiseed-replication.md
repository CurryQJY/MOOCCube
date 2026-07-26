# Stage B Multi-Seed Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the already validated Stage B mechanism on seeds 2026 and 2027 without tuning any registered training choice.

**Architecture:** A strict validation-only Hot replication entrypoint manually loads train/validation, creates a fresh expert, and records the actual selected checkpoint contract. A separate adapter replication entrypoint accepts only that per-seed contract, keeps tau fixed at `0.24929234`, records q75 as an audit value, and runs the fixed validation-only adapter protocol. A serial launcher invokes the two phases in seed order and never invokes a test evaluator.

**Tech Stack:** Python, PyTorch, SciPy sparse graphs, pytest, PowerShell.

---

### Task 1: Define Seed-Generalized Contracts

**Files:**
- Create: `tests/test_ckg_stageb_multiseed_replication.py`
- Create: `ckg_hot_graph_preflight_replication.py`
- Create: `ckg_hot_replication_contract.py`
- Create: `ckg_frozen_hot_pseudocold_adapter_replication.py`

- [ ] **Step 1: Write failing tests.**

```python
def test_replication_config_allows_only_2026_and_2027_with_fixed_protocol():
    cfg = ReplicationConfig.for_seed(2026)
    assert cfg.seed == 2026
    assert cfg.pseudo_cold_item_count == 102
    assert cfg.trust_quantile == 0.75
    assert cfg.test_evaluation is False


def test_runtime_q75_is_recorded_not_supplied_as_a_tuned_constant():
    tau = calibrate_trust_tau(content, hot, warm_ids, quantile=0.75)
    assert tau == pytest.approx(expected_q75)
    assert ReplicationConfig.for_seed(2026).trust_tau == pytest.approx(0.24929234)
```

- [ ] **Step 2: Run the new tests and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_stageb_multiseed_replication.py -q --basetemp .pytest_tmp/ckg_stageb_replication_red`

Expected: import failure because the replication entrypoint does not exist.

- [ ] **Step 3: Implement the isolated replication entrypoint.**

The Hot module must manually load only meta/content/static train/static
validation and reject any seed outside 2026/2027. The contract writer records
the actual Hot-selected epoch/path/SHA256/architecture/q75. The adapter module
uses the 2025 helper implementation only for shared content-only mechanics,
requires that completed same-seed contract, uses fixed tau `0.24929234`,
recomputes q75 only as a tamper/audit value, and performs epoch-0 parity,
fixed masked ranking, validation selection, and JSON/CSV output.

- [ ] **Step 4: Run GREEN and compile.**

Run: `./py.bat -m pytest tests/test_ckg_stageb_multiseed_replication.py -q --basetemp .pytest_tmp/ckg_stageb_replication_green`

Run: `./py.bat -m py_compile ckg_frozen_hot_pseudocold_adapter_replication.py`

### Task 2: Add Reproducible Per-Seed Launchers

**Files:**
- Create: `run_ckg_hot_graph_preflight_replication.ps1`
- Create: `run_ckg_frozen_hot_pseudocold_replication.ps1`
- Modify: `tests/test_ckg_stageb_multiseed_replication.py`

- [ ] **Step 1: Write launcher contract tests.**

```python
def test_hot_replication_launcher_requires_seed_2026_or_2027_and_fresh_roots():
    source = Path("run_ckg_hot_graph_preflight_replication.ps1").read_text(encoding="utf-8")
    assert "ValidateSet(2026, 2027)" in source
    assert "selected_checkpoint_sha256" in source
    assert "Invoke-NativeLogged" in source


def test_adapter_replication_launcher_binds_hot_contract_and_is_validation_only():
    source = Path("run_ckg_frozen_hot_pseudocold_replication.ps1").read_text(encoding="utf-8")
    assert "ValidateSet(2026, 2027)" in source
    assert "TestEvaluation = $false" in source
    assert "selected_checkpoint_sha256" in source
```

- [ ] **Step 2: Run the launcher tests and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_stageb_multiseed_replication.py -q --basetemp .pytest_tmp/ckg_stageb_replication_launcher_red`

Expected: launcher files are missing.

- [ ] **Step 3: Implement the launchers.**

Both launchers must create seed-specific fresh roots, audit all consumed files
before and after execution, preserve validation rows on failures, and accept
native stderr when the process exit status is valid. The Hot launcher records
the selected checkpoint SHA256 after a completed Hot result. The adapter
launcher verifies that exact recorded SHA256 before passing the checkpoint to
the Python replication entrypoint.

- [ ] **Step 4: Run GREEN and PowerShell dry-runs.**

Run: `./py.bat -m pytest tests/test_ckg_stageb_multiseed_replication.py -q --basetemp .pytest_tmp/ckg_stageb_replication_launcher_green`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_hot_graph_preflight_replication.ps1 -Seed 2026 -DryRun`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_frozen_hot_pseudocold_replication.ps1 -Seed 2026 -DryRun`

### Task 3: Run And Audit The Replication

**Files:**
- Run: `run_ckg_hot_graph_preflight_replication.ps1`
- Run: `run_ckg_frozen_hot_pseudocold_replication.ps1`

- [ ] **Step 1: Verify both seeds' formal roots are absent and GPU has no compute workload.**

- [ ] **Step 2: Run Hot preflight then Stage B serially for seed 2026, followed by seed 2027.**

Do not launch Stage B for a seed whose Hot manifest/result is not completed and
whose selected checkpoint SHA256 is not recorded.

- [ ] **Step 3: Independently audit each completed result.**

Recompute the selected epoch from `validation_epochs.csv`; verify epoch-0
parity, four guards, all before/after hashes, validation-only flags, q75 tau,
and the Stage B gate.

- [ ] **Step 4: Export the three-seed validation summary.**

Combine seed 2025/2026/2027 selected rows, epoch-0 rows, deltas, pass flags,
and mean/std. Do not inspect or evaluate test data.
