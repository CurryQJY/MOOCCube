# RecPPO Research-Grade Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repaired strict-cold entrypoint implement a deterministic, two-stage, genuinely clipped RecPPO method without changing the legacy baseline or ContentDelta.

**Architecture:** Keep the legacy static pipeline as orchestration, but monkeypatch repaired model, optimizer, evaluation, checkpoint, and manifest bindings. The repaired model owns a separate PPO optimizer and stage lifecycle; the legacy optimizer trains only the warm phase.

**Tech Stack:** Python 3.12, PyTorch, pytest, PowerShell runner.

---

### Task 1: Lock PPO And Inference Semantics

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`

- [x] Add a test proving `optimize_recppo` performs distinct PPO steps and changes policy ratios.
- [x] Add a test proving repeated deterministic inference returns identical vectors.
- [x] Add a test proving cached evaluation bank vectors are reused for positive scores.
- [x] Run the focused tests and confirm each fails for the missing behavior.

### Task 2: Implement Two-Stage RecPPO Core

**Files:**
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Test: `tests/test_usim_strict_cold_repair.py`

- [x] Add warmup/RL phase state and freeze all non-RecPPO parameters at the phase transition.
- [x] Exclude agent/stop parameters from the legacy optimizer.
- [x] Add a dedicated grouped Adam optimizer and true per-epoch PPO update loop.
- [x] Keep `compute_ppo_loss` as a side-effect-free objective for tests and diagnostics.
- [x] Run focused tests to green.

### Task 3: Repair MDP And Reward

**Files:**
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Test: `tests/test_usim_strict_cold_repair.py`

- [x] Add a learned stop candidate, active masks, done masks, and valid-transition masks.
- [x] Restrict behavior CE labels to the first valid step.
- [x] Replace raw MSE reward with normalized embedding gain plus candidate-ranking gain.
- [x] Remove batch-global duplicate/coverage terms from reward.
- [x] Add stop/reward tests and run them to green.

### Task 4: Make Evaluation Deterministic And Consistent

**Files:**
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Test: `tests/test_usim_strict_cold_repair.py`

- [x] Use deterministic top candidates when `deterministic=True`.
- [x] Cache repaired evaluation banks on the model.
- [x] Serve positive item vectors from the same cached bank.
- [x] Patch both eval-module and legacy imported bindings.
- [x] Run deterministic evaluation tests to green.

### Task 5: Defaults, Diagnostics, And Provenance

**Files:**
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Modify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`
- Test: `tests/test_usim_strict_cold_repair.py`

- [x] Set repaired pseudo-cold defaults to `batch_tail`, ratio `0.30`, min-pop `5`.
- [x] Set repaired schedule defaults that leave room for warmup and PPO phases.
- [x] Export PPO KL, clip fraction, entropy, reward moments, and stop rate in training stats.
- [x] Patch manifest script provenance and include resolved RecPPO config.
- [x] Preserve and restore dedicated PPO optimizer state in checkpoints.
- [x] Run focused tests to green.

### Task 6: Regression And Smoke Verification

**Files:**
- Verify: `usim_feedback_fast3_content_delta_repaired.py`
- Verify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`

- [ ] Run `pytest tests/test_usim_strict_cold_repair.py -q`.
- [ ] Run existing core/static regression tests.
- [ ] Run Python and PowerShell syntax checks.
- [ ] Run a one-seed, one-epoch repaired smoke experiment and inspect phase/PPO/eval diagnostics.
- [ ] Compare the working tree and confirm no legacy method logic or unrelated user files were modified.
