# Warmup Stage Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable warmup-boundary checkpoints for fair RecPPO branch experiments.

**Architecture:** Extend the repaired checkpoint wrapper with a stage fingerprint that excludes approved PPO-only controls. Hook the static training checkpoint builder so the warmup boundary writes a stable `warmup_stage.pt`, and add a stage-load path that restores the warm model and outer optimizer while discarding RecPPO optimizer state.

**Tech Stack:** Python, PyTorch checkpoint serialization, pytest, PowerShell runner environment variables.

---

### Task 1: Stage compatibility contract

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Write tests proving PPO-only controls are ignored and pseudo-cold/split controls remain protected.
- [ ] Run the focused tests and verify they fail because stage fingerprint helpers do not exist.
- [ ] Implement the minimal stage fingerprint and compatibility helpers.
- [ ] Run the focused tests and verify they pass.

### Task 2: Stage checkpoint payload

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Write a failing test requiring outer optimizer and RNG state while excluding RecPPO optimizer state.
- [ ] Implement stage-state construction in the repaired checkpoint wrapper.
- [ ] Verify the focused checkpoint tests pass.

### Task 3: Save and load integration

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Modify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`

- [ ] Write failing integration tests for the warmup snapshot environment and branch checkpoint argument.
- [ ] Save `warmup_stage.pt` at the configured warmup boundary.
- [ ] Load the stage file only when ordinary resume has not succeeded, restoring the outer optimizer and resetting RecPPO optimizer state.
- [ ] Expose the stage checkpoint path through the repaired PowerShell runner.
- [ ] Run focused and full repair tests.

### Task 4: Verification

**Files:**
- Verify: `usim_feedback_fast3_content_delta_repaired.py`
- Verify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`

- [ ] Run `D:\anaconda3\envs\zw\python.exe -m py_compile usim_feedback_fast3_content_delta_repaired.py`.
- [ ] Run the focused pytest checkpoint suite.
- [ ] Confirm the currently running process and its existing files were not restarted or modified.
