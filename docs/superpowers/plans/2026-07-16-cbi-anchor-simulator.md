# CBI Soft-Anchor Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch one isolated seed-2025 CBI experiment that replaces the training ID target with the initial CBI representation and uses all-item refined inference.

**Architecture:** Add a small model subclass that delegates the complete simulator implementation to the existing parent while replacing only `target_emb`. Add a delegate entrypoint and an isolated PowerShell launcher that installs the subclass and existing all-item evaluation adapter in the current process.

**Tech Stack:** Python, PyTorch, pytest, Windows PowerShell, existing FAST3 static runner.

---

### Task 1: Soft-anchor model behavior

**Files:**
- Create: `cbi_anchor_sim.py`
- Create: `tests/test_cbi_anchor_sim.py`

- [ ] **Step 1: Write a failing target-routing test**

Test that two different caller targets are ignored and that the parent simulator receives `init_item_emb.detach()` in both cases.

- [ ] **Step 2: Run the target test and verify RED**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

Expected: import or behavior failure because `CBIAnchorFast3FeedbackUSIM` does not exist.

- [ ] **Step 3: Implement the minimal subclass**

Implement `run_usim_episode` with the parent signature and delegate to `super().run_usim_episode(..., target_emb=init_item_emb.detach(), ...)`. Do not copy the parent rollout and do not add projection.

- [ ] **Step 4: Verify GREEN**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

Expected: all anchor tests pass.

- [ ] **Step 5: Commit**

Commit only `cbi_anchor_sim.py` and `tests/test_cbi_anchor_sim.py`.

### Task 2: Isolated entrypoint and all-item evaluation

**Files:**
- Create: `run_cbi_anchor_sim_seed2025.py`
- Modify: `tests/test_cbi_anchor_sim.py`

- [ ] **Step 1: Write failing entrypoint tests**

Require `USIM_STATIC_DELEGATE_ENTRYPOINT = True`, installation of `CBIAnchorFast3FeedbackUSIM`, installation of the existing all-item cached-bank hooks, and the resume-reason compatibility wrapper.

- [ ] **Step 2: Verify RED**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

- [ ] **Step 3: Implement the delegate entrypoint**

Install the subclass and `install_trust_eval_adapter` only inside the experiment process, then call `protocol.main()`.

- [ ] **Step 4: Verify GREEN and evaluation regressions**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py -q`

- [ ] **Step 5: Commit**

Commit the entrypoint and its tests only.

### Task 3: Reproducible launcher

**Files:**
- Create: `run_cbi_anchor_sim_seed2025.ps1`
- Modify: `tests/test_cbi_anchor_sim.py`

- [ ] **Step 1: Write failing launcher-contract tests**

Lock seed 2025, 60 epochs, patience 10, cold-only item-macro early stopping, delta norm 0.5, five simulator steps, auto-resume, checkpoint optimizer state, isolated paths, and protected shared-code hashes.

- [ ] **Step 2: Verify RED**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

- [ ] **Step 3: Implement the launcher**

Follow the existing isolated launcher pattern, use roots named `cbi_anchor_sim_single_seed2025`, write a manifest, preserve interrupted checkpoints, and exclude manuscript files from runtime protection.

- [ ] **Step 4: Verify GREEN and dry-run**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py -q`

Run: `.\run_cbi_anchor_sim_seed2025.ps1 -DryRun`

- [ ] **Step 5: Commit**

Commit the launcher and launcher tests only.

### Task 4: Launch and first-epoch verification

**Files:**
- Runtime outputs only under the isolated output, checkpoint, and background-log roots.

- [ ] **Step 1: Confirm no existing anchor experiment process and inspect GPU availability**

- [ ] **Step 2: Start the launcher in a hidden PowerShell process**

- [ ] **Step 3: Verify manifest status and resume/fresh-start semantics**

- [ ] **Step 4: Monitor through the first complete epoch**

Require finite loss, a validation report, checkpoint creation, and no shared-code hash changes.

- [ ] **Step 5: Report PID, log path, first validation metrics, and estimated remaining duration**
