# Experiment Provenance Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve an immutable source snapshot for every formal FAST3 run and reject checkpoint resume only when the normalized training configuration or data split changes.

**Architecture:** Add a focused `fast3_delta.provenance` module for source capture, runtime metadata, split fingerprints, and audit records. Refactor checkpoint compatibility into independent training-config and split fingerprints; source drift is compared separately and reported as a warning. Integrate the module into static manifest creation and checkpoint save/load without changing model defaults or the active recovery run.

**Tech Stack:** Python 3.12, PyTorch checkpoints, JSON/SHA256, pytest, PowerShell static runner.

---

## File Map

- Create `fast3_delta/provenance.py`: deterministic hashing, immutable snapshot creation, source comparison, split fingerprinting, and resume audit records.
- Modify `fast3_delta/checkpoint.py`: versioned training fingerprint, split fingerprint validation, legacy policy, and field-level mismatch reports.
- Modify `usim_feedback_fast3_content_delta.py`: create provenance before training, enrich the static manifest, save both fingerprints, and report source-only warnings.
- Modify `usim_feedback_fast3_content_delta_repaired.py`: preserve RecPPO-specific fingerprint fields while delegating the new base compatibility contract.
- Modify `run_usim_feedback_fast3_content_delta_static.ps1`: expose the explicit legacy-checkpoint switch and pass runner identity to provenance.
- Create `tests/test_experiment_provenance.py`: unit tests for snapshots, hashes, audit records, and corruption detection.
- Create `tests/test_checkpoint_provenance_guard.py`: checkpoint compatibility tests.
- Modify `tests/test_usim_strict_cold_repair.py`: wrapper compatibility regression tests.
- Modify `tests/test_static_runner_checkpoint_defaults.ps1`: runner default and environment propagation tests.

### Task 1: Provenance Snapshot Module

**Files:**
- Create: `fast3_delta/provenance.py`
- Create: `tests/test_experiment_provenance.py`

- [ ] **Step 1: Write failing snapshot and comparison tests**

Add tests that create a temporary entrypoint, `fast3_delta` directory, and runner, then assert that the first call creates an immutable snapshot and a second call does not overwrite it:

```python
def test_create_provenance_snapshot_is_immutable(tmp_path):
    source_root = tmp_path / "repo"
    entrypoint = source_root / "train.py"
    module = source_root / "fast3_delta" / "config.py"
    runner = source_root / "run.ps1"
    module.parent.mkdir(parents=True)
    entrypoint.write_text("print('v1')\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")
    runner.write_text("Write-Host run\n", encoding="utf-8")

    first = create_provenance_snapshot(tmp_path / "output", source_root, entrypoint, runner)
    entrypoint.write_text("print('v2')\n", encoding="utf-8")
    second = create_provenance_snapshot(tmp_path / "output", source_root, entrypoint, runner)

    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert (tmp_path / "output/provenance/source/train.py").read_text() == "print('v1')\n"
```

Also test `compare_source_manifests()` for added, removed, and modified files, and test that a snapshot file modified after creation raises `ProvenanceCorruptionError`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\py.bat -m pytest tests\test_experiment_provenance.py -q
```

Expected: collection fails because `fast3_delta.provenance` does not exist.

- [ ] **Step 3: Implement deterministic source capture**

Implement these public interfaces:

```python
PROVENANCE_SCHEMA_VERSION = 1

class ProvenanceCorruptionError(RuntimeError):
    pass

def sha256_file(path): ...
def build_source_manifest(source_root, entrypoint, runner_path): ...
def create_provenance_snapshot(output_dir, source_root, entrypoint, runner_path): ...
def verify_provenance_snapshot(output_dir): ...
def compare_source_manifests(expected, current): ...
def write_resume_source_audit(output_dir, source_root, entrypoint, runner_path): ...
def build_runtime_metadata(source_root, normalized_command=None): ...
def stable_fingerprint(payload): ...
def build_split_fingerprint(split_info, exports=None): ...
```

Capture only the selected entrypoint, `fast3_delta/**/*.py`, and selected runner. Copy via temporary directory followed by atomic rename. Store paths relative to the project root, sort all path lists, and hash raw bytes. If `provenance/source_manifest.json` already exists, verify it and return it without overwriting files.

- [ ] **Step 4: Run provenance tests**

Run:

```powershell
.\py.bat -m pytest tests\test_experiment_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add fast3_delta/provenance.py tests/test_experiment_provenance.py
git commit -m "feat: capture immutable experiment provenance"
```

### Task 2: Separate Checkpoint Configuration and Split Guards

**Files:**
- Modify: `fast3_delta/checkpoint.py`
- Create: `tests/test_checkpoint_provenance_guard.py`

- [ ] **Step 1: Write failing compatibility tests**

Cover these exact outcomes:

```python
def test_source_change_does_not_reject_resume(base_cfg, split_info):
    state = checkpoint_state(base_cfg, split_info, source_hash="old")
    decision = checkpoint_resume_decision(state, base_cfg, split_info, source_manifest={"train.py": "new"})
    assert decision.ok is True
    assert decision.source_warning

def test_training_config_change_rejects_resume(base_cfg, split_info):
    state = checkpoint_state(base_cfg, split_info)
    base_cfg.ppo_loss_weight = 0.5
    decision = checkpoint_resume_decision(state, base_cfg, split_info)
    assert decision.ok is False
    assert "ppo_loss_weight" in decision.reason

def test_epoch_extension_is_resume_compatible(base_cfg, split_info):
    state = checkpoint_state(base_cfg, split_info)
    base_cfg.n_epochs = 100
    base_cfg.early_stop_patience = 100
    assert checkpoint_resume_decision(state, base_cfg, split_info).ok is True

def test_split_change_rejects_resume(base_cfg, split_info):
    state = checkpoint_state(base_cfg, split_info)
    split_info["test_fold_ids"] = [9]
    decision = checkpoint_resume_decision(state, base_cfg, split_info)
    assert decision.ok is False
    assert "split" in decision.reason
```

Add a legacy checkpoint test requiring `USIM_FB_ALLOW_LEGACY_CKPT=1`; without it, the decision must reject.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\py.bat -m pytest tests\test_checkpoint_provenance_guard.py -q
```

Expected: failures show that script SHA, epoch ceiling, and patience currently invalidate resume and that legacy checkpoints are accepted implicitly.

- [ ] **Step 3: Implement the versioned compatibility contract**

In `fast3_delta/checkpoint.py`:

- set `CHECKPOINT_FINGERPRINT_SCHEMA_VERSION = 2`;
- remove `script_name`, `script_sha256`, `n_epochs`, and `early_stop_patience` from the training payload;
- retain all model, optimizer, sampling, PPO, reward, and stochastic controls;
- store a separate split fingerprint and payload using `build_split_fingerprint()`;
- introduce a structured result:

```python
@dataclass(frozen=True)
class ResumeDecision:
    ok: bool
    reason: str
    train_fingerprint: str
    split_fingerprint: str
    source_warning: str = ""
    legacy_override: bool = False
```

- implement `checkpoint_resume_decision(...)` and keep `_checkpoint_config_matches(...)` as a compatibility adapter for existing call sites;
- reject missing or old schema fingerprints unless `USIM_FB_ALLOW_LEGACY_CKPT=1`;
- never allow that legacy switch to bypass a known configuration or split mismatch.

- [ ] **Step 4: Run checkpoint tests**

Run:

```powershell
.\py.bat -m pytest tests\test_checkpoint_provenance_guard.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run existing checkpoint-related tests**

Run:

```powershell
.\py.bat -m pytest tests\test_usim_strict_cold_repair.py -q
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_static_runner_checkpoint_defaults.ps1
```

Expected: existing tests either pass or fail only where they still assert the old coarse fingerprint behavior; update those assertions in Task 3.

- [ ] **Step 6: Commit Task 2**

```powershell
git add fast3_delta/checkpoint.py tests/test_checkpoint_provenance_guard.py
git commit -m "feat: guard checkpoint resume by config and split"
```

### Task 3: Static Experiment and RecPPO Integration

**Files:**
- Modify: `usim_feedback_fast3_content_delta.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Modify: `tests/test_usim_strict_cold_repair.py`

- [ ] **Step 1: Write failing integration tests**

Add tests asserting that manifest creation includes:

```python
assert manifest["provenance"]["schema_version"] == 1
assert manifest["provenance"]["training_config_sha256"]
assert manifest["provenance"]["split_sha256"]
assert manifest["provenance"]["source_manifest_sha256"]
```

Add a test that source-only drift emits `WARNING: source provenance differs` but returns a successful resume decision. Add a RecPPO regression asserting that reward controls added by `repaired_static_train_config_fingerprint()` remain in the normalized training payload.

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```powershell
.\py.bat -m pytest tests\test_usim_strict_cold_repair.py -q
```

Expected: new provenance assertions fail.

- [ ] **Step 3: Integrate provenance before training**

In `run_static_experiment()`:

```python
provenance = create_provenance_snapshot(
    output_dir=_feedback_output_dir(),
    source_root=os.path.dirname(os.path.abspath(__file__)),
    entrypoint=os.path.abspath(__file__),
    runner_path=os.environ.get("USIM_RUNNER_PATH", ""),
)
```

Perform this after the output directory is known but before loading a checkpoint or beginning training. On resume, verify the original snapshot and write a timestamped source audit. Pass the source comparison into `checkpoint_resume_decision()` and print its warning without rejecting a compatible checkpoint.

- [ ] **Step 4: Enrich checkpoint state and static manifest**

Save these fields in every checkpoint:

```python
state.update({
    "fingerprint_schema_version": CHECKPOINT_FINGERPRINT_SCHEMA_VERSION,
    "train_config_fingerprint": train_fp,
    "train_config_payload": train_payload,
    "split_fingerprint": split_fp,
    "split_payload": split_payload,
    "source_manifest_sha256": provenance["source_manifest_sha256"],
})
```

Extend `_write_static_manifest()` with a `provenance` section while retaining existing keys. Record the resume decision, source warning, and legacy override.

- [ ] **Step 5: Preserve repaired-entrypoint fingerprint extensions**

Change the repaired wrapper so it augments the base normalized training payload before hashing, rather than replacing compatibility logic. Ensure RecPPO reward mode, loss weights, warmup, rollout policy, simulator controls, and residual scale remain config-invalidating fields.

- [ ] **Step 6: Run integration tests**

Run:

```powershell
.\py.bat -m pytest tests\test_usim_strict_cold_repair.py tests\test_checkpoint_provenance_guard.py tests\test_experiment_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add usim_feedback_fast3_content_delta.py usim_feedback_fast3_content_delta_repaired.py tests/test_usim_strict_cold_repair.py
git commit -m "feat: integrate provenance into static experiments"
```

### Task 4: Runner Controls and End-to-End Verification

**Files:**
- Modify: `run_usim_feedback_fast3_content_delta_static.ps1`
- Modify: `tests/test_static_runner_checkpoint_defaults.ps1`

- [ ] **Step 1: Write failing runner tests**

Assert these defaults and environment mappings:

```powershell
Assert-Contains $runnerText '[bool]$AllowLegacyCheckpoint = $false'
Assert-Contains $runnerText 'USIM_FB_ALLOW_LEGACY_CKPT'
Assert-Contains $runnerText 'USIM_RUNNER_PATH'
```

The fake Python runner must observe `USIM_FB_ALLOW_LEGACY_CKPT=0` by default and `1` only when `-AllowLegacyCheckpoint $true` is supplied.

- [ ] **Step 2: Run runner test and verify failure**

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_static_runner_checkpoint_defaults.ps1
```

Expected: failure because the parameters and environment variables do not exist.

- [ ] **Step 3: Add explicit runner controls**

Add:

```powershell
[bool]$AllowLegacyCheckpoint = $false
```

Export:

```powershell
$env:USIM_FB_ALLOW_LEGACY_CKPT = if ($AllowLegacyCheckpoint) { "1" } else { "0" }
$env:USIM_RUNNER_PATH = $MyInvocation.MyCommand.Path
```

Do not expose a switch that bypasses known config or split mismatches.

- [ ] **Step 4: Run the full focused suite**

Run:

```powershell
.\py.bat -m pytest tests\test_experiment_provenance.py tests\test_checkpoint_provenance_guard.py tests\test_usim_strict_cold_repair.py -q
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_static_runner_checkpoint_defaults.ps1
.\py.bat -m py_compile fast3_delta\provenance.py fast3_delta\checkpoint.py usim_feedback_fast3_content_delta.py usim_feedback_fast3_content_delta_repaired.py
```

Expected: all pytest and PowerShell tests pass; compilation exits 0.

- [ ] **Step 5: Run an isolated one-epoch smoke test**

Run with a new output/checkpoint root and no impact on the active recovery run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_usim_feedback_fast3_content_delta_static.ps1 `
  -ScriptPath .\usim_feedback_fast3_content_delta.py `
  -Protocol strict_item_cold_balanced -ColdThresholds 1 -Seeds 9091 `
  -Epochs 1 -Patience 1 -OutputRoot outputs\provenance_smoke `
  -CheckpointRoot checkpoints\provenance_smoke -ForceFresh $true -SkipAggregate
```

Expected: the run creates `provenance/source_manifest.json`, a manifest with separate hashes, and a checkpoint with schema version 2.

- [ ] **Step 6: Verify resume behavior on the smoke checkpoint**

Resume once with only `-Epochs 2 -Patience 2`; expected: accepted. Then perform a dry compatibility check with a changed PPO weight; expected: rejected before training with a field-level `ppo_loss_weight` difference.

- [ ] **Step 7: Commit Task 4**

```powershell
git add run_usim_feedback_fast3_content_delta_static.ps1 tests/test_static_runner_checkpoint_defaults.ps1
git commit -m "feat: expose provenance-safe checkpoint controls"
```

### Task 5: Final Regression and Handoff

**Files:**
- Verify only; update documentation only if test output reveals a necessary clarification.

- [ ] **Step 1: Check the active recovery process is unaffected**

Run:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'usim_feedback_fast3_content_delta_recovered_51ea_candidate' }
```

Expected: the existing recovery process remains active or has completed normally; it was never restarted by this change.

- [ ] **Step 2: Run final focused verification**

Run the commands from Task 4 Step 4 again from a clean shell. Expected: all pass.

- [ ] **Step 3: Inspect only task-related changes**

```powershell
git status --short
git diff --check HEAD~4..HEAD
```

Expected: no whitespace errors; unrelated user changes remain untouched.

- [ ] **Step 4: Report compatibility behavior**

Document in the handoff that configuration and split mismatches reject resume, source-only drift warns and continues, epoch/patience extensions are allowed, and legacy checkpoints require `-AllowLegacyCheckpoint $true`.
