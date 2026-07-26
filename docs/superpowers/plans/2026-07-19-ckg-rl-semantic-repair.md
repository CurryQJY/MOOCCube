# CKG-RL Semantic Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a versioned, reproducible CKG-RL correction path that makes refined full-ranking evaluation self-consistent and anchors simulator training and inference to the initial course state.

**Architecture:** Preserve the historical main-table behavior behind explicit legacy defaults. The corrected path is activated only by three recorded controls: deterministic per-item candidate sampling, full-ranking positive-score reuse from the catalog bank, and `initial_state` simulator target mode. It uses fresh output/checkpoint roots and rejects incompatible resumes through the train-config fingerprint.

**Tech Stack:** Python, PyTorch, pytest, Windows PowerShell, existing FAST3 static runner.

---

### Task 1: Write semantic-repair regression tests

**Files:**
- Create: `tests/test_ckg_rl_semantic_repairs.py`

- [ ] **Step 1: Write the deterministic-candidate RED test**

```python
first = model.get_candidates(item_state, user_bank_raw=bank, item_idx=item_ids,
                             deterministic=True, rollout_step=0)
second = model.get_candidates(item_state, user_bank_raw=bank, item_idx=item_ids,
                              deterministic=True, rollout_step=0)
assert torch.equal(first[1], second[1])
```

Also assert the same item gets the same candidates when the batch order is permuted and that normal calls retain their stochastic API.

- [ ] **Step 2: Write the full-ranking bank RED test**

```python
monkeypatch.setattr(eval_mod, "build_eval_pos_item_vecs", fail_if_called)
metrics, count = eval_mod.evaluate_usim(
    model, loader, torch.device("cpu"), None,
    eval_type="cold", full_ranking=True, all_item_vecs={"cold": bank},
)
assert count == 1
```

Require a seen positive target to be restored from the original catalog matrix score, not recomputed by a second simulator call.

- [ ] **Step 3: Write the h0-anchor RED tests**

```python
assert torch.equal(captured_train_target, z_i_base.detach())
assert torch.equal(captured_infer_target, z_i_base.detach())
assert not torch.equal(captured_train_target, id_e_true)
```

Construct the configuration with `USIM_SIMULATOR_TARGET_MODE=initial_state` and spy on `run_usim_episode`.

- [ ] **Step 4: Write the fingerprint RED test**

```python
legacy_fp, _ = _static_train_config_fingerprint(legacy_cfg)
repair_fp, _ = _static_train_config_fingerprint(repaired_cfg)
assert legacy_fp != repair_fp
```

- [ ] **Step 5: Run RED**

Run: `./py.bat -m pytest tests/test_ckg_rl_semantic_repairs.py -q --basetemp .pytest_tmp/ckg_rl_semantic_repair_red`

Expected: failures because the candidate API lacks deterministic controls, full ranking still recomputes positives, initial-state mode is ignored, and the fingerprint is unchanged.

### Task 2: Add versioned deterministic evaluation controls

**Files:**
- Modify: `fast3_delta/config.py`
- Modify: `usim_feedback_fast3_content_delta.py`
- Modify: `fast3_delta/eval.py`
- Modify: `fast3_delta/checkpoint.py`
- Test: `tests/test_ckg_rl_semantic_repairs.py`

- [ ] **Step 1: Add configuration fields with legacy defaults**

```python
self.deterministic_eval_candidates = os.environ.get(
    "USIM_DETERMINISTIC_EVAL_CANDIDATES", "0"
) == "1"
self.eval_reuse_item_bank = os.environ.get("USIM_EVAL_REUSE_ITEM_BANK", "0") == "1"
self.simulator_target_mode = os.environ.get(
    "USIM_SIMULATOR_TARGET_MODE", "legacy_id"
).strip().lower()
```

Reject target modes other than `legacy_id` and `initial_state`.

- [ ] **Step 2: Implement per-item deterministic multinomial sampling**

Extend `get_candidates(..., deterministic=False, rollout_step=0)`. When deterministic, create one local `torch.Generator` per row from a stable integer function of the configured seed, item ID, and rollout step, then call the existing `torch.multinomial` on that row's existing `probs`. Keep the non-deterministic path unchanged.

- [ ] **Step 3: Propagate deterministic evaluation mode**

Call `get_candidates(..., deterministic=deterministic and self.cfg.deterministic_eval_candidates, rollout_step=t)` from `run_usim_episode`.

- [ ] **Step 4: Reuse catalog scores only for corrected full ranking**

In `evaluate_usim`, when `full_ranking` and `model.cfg.eval_reuse_item_bank` are true, calculate `scores = z_u @ item_bank.T`, capture `scores[row_idx, i]` before history masking, and restore that exact value afterwards. Do not call `build_eval_pos_item_vecs` on this path. Preserve the historical positive-vector path otherwise and leave sampled evaluation unchanged.

- [ ] **Step 5: Make resumes reject semantic mismatches**

Add all three semantic controls and the deterministic evaluation seed to `_static_train_config_fingerprint`.

- [ ] **Step 6: Run GREEN**

Run: `./py.bat -m pytest tests/test_ckg_rl_semantic_repairs.py -q --basetemp .pytest_tmp/ckg_rl_semantic_repair_green`

Expected: all semantic-repair tests pass.

### Task 3: Route the simulator target to h0

**Files:**
- Modify: `usim_feedback_fast3_content_delta.py`
- Test: `tests/test_ckg_rl_semantic_repairs.py`

- [ ] **Step 1: Select the target in training**

```python
if self.cfg.simulator_target_mode == "initial_state":
    target_emb = z_i_base.detach().clone()
else:
    target_emb = id_e_true.detach().clone()
```

- [ ] **Step 2: Select the target in refined inference**

```python
target_emb = (
    z_i_base.detach().clone()
    if self.cfg.simulator_target_mode == "initial_state"
    else None
)
```

Pass it to `run_usim_episode` unchanged. Do not remove `id_e_true`, because auxiliary objectives may still use it.

- [ ] **Step 3: Run GREEN**

Run: `./py.bat -m pytest tests/test_ckg_rl_semantic_repairs.py -q --basetemp .pytest_tmp/ckg_rl_semantic_anchor_green`

Expected: anchor routing tests pass while legacy mode continues to route the historical target behavior.

### Task 4: Expose an isolated corrected launcher

**Files:**
- Modify: `run_usim_feedback_fast3_content_delta_static.ps1`
- Create: `run_ckg_rl_semantic_repair_seed2025.ps1`
- Test: `tests/test_ckg_rl_semantic_repairs.py`

- [ ] **Step 1: Add explicit PowerShell parameters**

Add `SimulatorTargetMode`, `DeterministicEvalCandidates`, `EvalReuseItemBank`, and `DeterministicEvalSeed` to the static runner. Put their `USIM_*` values in the tracked environment list and emitted settings manifest.

- [ ] **Step 2: Create the isolated seed-2025 launcher**

Set `SimulatorTargetMode=initial_state`, deterministic candidates and bank reuse to true, `ForceFresh=true`, `AutoResume=false`, `SaveCkpt=true`, `RunSampledEval=false`, and separate output/checkpoint roots. Keep ContentDelta, pseudo-cold, trust, and Hot simulation disabled.

- [ ] **Step 3: Parse and contract test**

Run: `./py.bat -m pytest tests/test_ckg_rl_semantic_repairs.py -q --basetemp .pytest_tmp/ckg_rl_semantic_launcher_green`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_ckg_rl_semantic_repair_seed2025.ps1 -DryRun`

Expected: zero parser errors, a fresh corrected manifest, and no training process.

### Task 5: Verify and prepare the first experiment

**Files:**
- Verify only: files above.

- [ ] **Step 1: Run focused regression tests**

Run: `./py.bat -m pytest tests/test_ckg_rl_semantic_repairs.py tests/test_usim_strict_cold_repair.py -q --basetemp .pytest_tmp/ckg_rl_semantic_full_green`

- [ ] **Step 2: Run static-runner dry-run**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_ckg_rl_semantic_repair_seed2025.ps1 -DryRun`

- [ ] **Step 3: Record the gate for the later GPU run**

Require: fresh roots, no checkpoint resume, stored config fingerprint, deterministic refined vectors, shared full-ranking item bank, and `initial_state` target routing.
