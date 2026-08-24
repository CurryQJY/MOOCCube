# CKG-RL V3.5 Frozen Test Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a one-time strict-cold test replay from the frozen V3.5 seed-2025 checkpoint without training or selection.

**Architecture:** A standalone replay module validates the source manifest and all three checkpoint digests before loading test rows. It reconstructs the V3.5 target-free item bank, evaluates the selected policy once, and writes a diagnostic-only report in a fresh output directory.

**Tech Stack:** Python 3, PyTorch, pandas, pytest, `ckg_rl_usim_v32_clean.py`, `ckg_rl_usim_v33_rank_distill.py`, and `ckg_rl_usim_v35_action_distill.py`.

---

### Task 1: Frozen Source Validation

**Files:**
- Create: `tests/test_ckg_rl_usim_v35_test_replay.py`
- Create: `ckg_rl_usim_v35_test_replay.py`

- [ ] **Step 1: Write failing checkpoint-contract tests.**

```python
def test_replay_rejects_a_source_checkpoint_with_a_manifest_hash_mismatch(tmp_path):
    source = _write_frozen_source(tmp_path)
    source["teacher_path"].write_bytes(b"hash drift")
    with pytest.raises(ValueError, match="sha256"):
        load_frozen_v35_source(source["manifest_path"])
```

- [ ] **Step 2: Run the test before implementation.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_test_replay.py -q --basetemp .pytest_tmp/v35_replay_red`

Expected: collection failure because `ckg_rl_usim_v35_test_replay` does not exist.

- [ ] **Step 3: Implement manifest loading, SHA-256 verification, and strict stage checks.**

```python
for stage in ("teacher", "generator", "policy"):
    payload = _load_checkpoint(checkpoint_root / f"{stage}.pt")
    if payload["stage"] != stage or _sha256(path) != hashes[stage]:
        raise ValueError(f"{stage} checkpoint sha256 or stage does not match source manifest")
```

- [ ] **Step 4: Run the contract test.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_test_replay.py -q --basetemp .pytest_tmp/v35_replay_green1`

Expected: PASS for hash-drift rejection.

### Task 2: Target-Free Frozen Test Evaluation

**Files:**
- Modify: `tests/test_ckg_rl_usim_v35_test_replay.py`
- Modify: `ckg_rl_usim_v35_test_replay.py`

- [ ] **Step 1: Write a failing replay test using a tiny source run.**

```python
def test_replay_loads_frozen_v35_models_without_training_and_writes_test_metrics(tmp_path, monkeypatch):
    source = _run_tiny_v35_source(tmp_path)
    monkeypatch.setattr(clean, "train_clean_teacher", _fail_training)
    monkeypatch.setattr(clean, "train_content_generator", _fail_training)
    result = run_v35_test_replay(_replay_config(source, tmp_path / "replay"))
    assert result["diagnostic_only"] is True
    assert result["test_loaded"] is True
    assert result["policy_mode"] == "action_distill_rollout"
```

- [ ] **Step 2: Run the test before implementing evaluation.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_test_replay.py -q --basetemp .pytest_tmp/v35_replay_red2`

Expected: FAIL because `run_v35_test_replay` is missing.

- [ ] **Step 3: Load checkpoint modules, rebuild the fixed partition, and evaluate once.**

```python
test_df = clean.load_clean_test_inputs(source_config, partitions.h_train)
partitions = clean.attach_clean_test_rows(partitions, test_df)
metrics = clean.evaluate_clean_route(
    teacher, generator, engine, hot_frame=partitions.h_test,
    cold_frame=partitions.c_test, warm_item_ids=warm_item_ids,
    content=content, user_history=user_history, config=source_config,
    policy_epoch=selected_epoch, export_dir=output_dir, export_prefix="test_",
)
metrics["policy_mode"] = v35._action_policy_mode(selected_epoch)
```

- [ ] **Step 4: Run all replay tests.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_test_replay.py -q --basetemp .pytest_tmp/v35_replay_green2`

Expected: PASS with no call to either training function.

### Task 3: CLI, Launcher, and One-Time Replay

**Files:**
- Modify: `tests/test_ckg_rl_usim_v35_test_replay.py`
- Modify: `ckg_rl_usim_v35_test_replay.py`
- Create: `run_ckg_rl_usim_v35_test_replay_seed2025.ps1`

- [ ] **Step 1: Write failing launcher and dry-run tests.**

```python
def test_replay_launcher_uses_a_fresh_output_root():
    launcher = Path("run_ckg_rl_usim_v35_test_replay_seed2025.ps1").read_text()
    assert 'ScriptPath = "ckg_rl_usim_v35_test_replay.py"' in launcher
    assert "test_replay_seed2025" in launcher
    assert "--dry-run" in launcher
```

- [ ] **Step 2: Implement CLI and launcher with overwrite refusal.**

```python
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError("refusing to overwrite an existing V3.5 test replay")
```

- [ ] **Step 3: Run dry-run, tests, then the one seed-2025 test replay.**

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v35_test_replay_seed2025.ps1 -DryRun`

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v35_test_replay_seed2025.ps1`

Expected: the replay writes a separate diagnostic manifest and test metrics; no source checkpoint or source P-only artifact changes.
