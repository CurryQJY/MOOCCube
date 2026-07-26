# RecPPO Joint Objective Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align repaired RecPPO training with strict-cold full-ranking evaluation while continuing supervised backbone learning.

**Architecture:** Keep separate policy and recommender optimizers, but execute both in the PPO phase. Replace teacher Top-K self-distillation with train-positive listwise hard-negative reward and bound auxiliary course shaping.

**Tech Stack:** Python 3.12, PyTorch, pytest, PowerShell.

---

### Task 1: Lock Joint-Training Behavior

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`

- [x] Replace the frozen-backbone expectation with a test that the PPO-phase forward returns a differentiable supervised loss.
- [x] Verify the new test fails against the current repaired model.

### Task 2: Lock Train-Positive Listwise Reward

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`

- [x] Add tests for train-only positive inversion and hard-negative exclusion.
- [x] Add a test that moving toward the observed positive user improves listwise reward.
- [x] Add a test that the hard-negative cache is reset at an epoch boundary.
- [x] Verify the tests fail for the current teacher Top-K implementation.

### Task 3: Lock Reward Scaling And CE Annealing

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`

- [x] Add tests for linear behavior-CE decay and its final floor.
- [x] Add a test that course reward contribution is scaled and clipped.
- [x] Verify the tests fail before implementation.

### Task 4: Implement The Repaired Objective

**Files:**
- Modify: `usim_feedback_fast3_content_delta_repaired.py`
- Modify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`

- [x] Keep backbone parameters trainable after phase activation and return the supervised outer loss.
- [x] Build item-positive sets from train-only histories and mine global hard negatives each epoch.
- [x] Compute positive-vs-hard-negative listwise CE gain.
- [x] Add weak embedding shaping, bounded course shaping, and CE annealing.
- [x] Export all new controls and manifest fields.

### Task 5: Verify

**Files:**
- Verify: `tests/test_usim_strict_cold_repair.py`
- Verify: `usim_feedback_fast3_content_delta_repaired.py`
- Verify: `run_usim_feedback_fast3_content_delta_repaired_static.ps1`

- [x] Run the focused RecPPO tests.
- [x] Run existing static/core regression tests.
- [x] Run Python and PowerShell syntax checks.
- [x] Run a short strict-cold smoke experiment and inspect gradients, rewards, and manifest values.
