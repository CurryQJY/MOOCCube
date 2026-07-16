# CBI Anchor Auxiliary-ID Screen Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Queue a three-arm seed-2025 AuxWeight screen behind the active CBI anchor three-seed baseline.

**Architecture:** Add one serial screen launcher and one lightweight manifest-polling queue launcher. Both reuse the existing anchor Python entrypoint and static runner, keep all outputs isolated, and fail closed when the upstream baseline fails.

**Tech Stack:** Windows PowerShell, pytest, existing FAST3 static runner.

---

### Task 1: Screen and queue contracts

**Files:**
- Create: `run_cbi_anchor_aux_screen_seed2025.ps1`
- Create: `wait_cbi_anchor_3seed_then_aux_screen.ps1`
- Modify: `tests/test_cbi_anchor_sim.py`

- [ ] **Step 1: Write failing tests**

Require AuxWeight arms `0.0`, `0.1`, `0.3`, seed 2025, 30 epochs, patience 6, isolated roots, and serial execution. Require the queue to poll the current three-seed manifest, launch only on `completed`, stop on `failed`, and invoke the screen script.

- [ ] **Step 2: Verify RED**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py -q`

- [ ] **Step 3: Implement both PowerShell scripts**

The screen writes a top-level manifest and invokes the static runner once per arm. The queue writes a queue manifest, polls at a bounded interval, and invokes the screen synchronously after upstream completion.

- [ ] **Step 4: Verify GREEN and dry-runs**

Run: `.\py.bat -m pytest tests\test_cbi_anchor_sim.py tests\test_cbi_trust_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py -q`

Run both scripts with `-DryRun` and parse both with the PowerShell parser.

- [ ] **Step 5: Commit and launch the hidden queue**

Commit only the two scripts and their test, then start the queue in a hidden PowerShell process and verify its queue manifest reports `waiting`.
