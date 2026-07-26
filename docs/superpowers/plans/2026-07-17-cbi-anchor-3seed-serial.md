# CBI Soft-Anchor Three-Seed Serial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch one reproducible serial runner for seeds 2025, 2026, and 2027 using the validated CBI soft-anchor method.

**Architecture:** Reuse the existing anchor model and Python delegate entrypoint. Add only a new PowerShell orchestrator with isolated three-seed paths, checkpoint resume, source/protected hashes, and the existing static runner's serial seed loop and aggregate stage.

**Tech Stack:** Windows PowerShell, Python, PyTorch, pytest, existing FAST3 static runner.

---

### Task 1: Three-seed launcher contract

**Files:**
- Create: `run_cbi_anchor_sim_3seed_serial.ps1`
- Modify: `tests/test_cbi_anchor_sim.py`

- [ ] **Step 1: Add a failing launcher test**

Require the anchor Python entrypoint, isolated `cbi_anchor_sim_3seed_serial` output/checkpoint/log roots, `Seeds = @(2025, 2026, 2027)`, 60 epochs, patience 10, delta 0.5, five steps, auto-resume, no hard projection flag, and no manuscript file in the protected runtime list.

- [ ] **Step 2: Run RED verification**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

Expected: failure because `run_cbi_anchor_sim_3seed_serial.ps1` is absent.

- [ ] **Step 3: Implement the serial launcher**

Create the top-level manifest, pass the three seeds in one ordered array to `run_usim_feedback_fast3_content_delta_static.ps1`, enable checkpoint/optimizer resume, archive an interrupted manifest, and let the static runner aggregate after the final seed.

- [ ] **Step 4: Verify launcher and regressions**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py tests\test_cbi_trust_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py -q`

Run: `.\run_cbi_anchor_sim_3seed_serial.ps1 -DryRun`

- [ ] **Step 5: Commit**

Commit only the serial launcher and its test.

### Task 2: Background launch verification

**Files:**
- Runtime artifacts only under the isolated three-seed roots.

- [ ] **Step 1: Confirm GPU availability and no existing three-seed process**

- [ ] **Step 2: Start the launcher in a hidden PowerShell process**

- [ ] **Step 3: Verify the manifest locks all three seeds and status is running**

- [ ] **Step 4: Verify seed 2025 begins from a fresh isolated checkpoint and reaches training progress**

- [ ] **Step 5: Report the PID, log path, serial order, training/inference semantics, and estimated duration**
