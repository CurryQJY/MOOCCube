# Simulator-Training Factorial Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate fixed T=5 course-fit inference on the three frozen T=0-training checkpoints and report the complete simulator-training 2×2.

**Architecture:** Extend the read-only inference wrapper with an optional nonnegative inference-step override that applies only inside the rollout and is restored afterward. Reuse the existing generic four-cell calculations through a simulator-specific report front-end, and run the missing cell with checkpoint hash protection.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell.

---

### Task 1: Inference-only step override

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing tests**

Test parsing of an optional nonnegative integer and assert that a T=0 model observes T=5 during `run_usim_episode`, returns to T=0 afterward, and exports checkpoint/effective step values.

- [ ] **Step 2: Run RED**

Run the focused wrapper tests. Expect failures because the override API and audit fields do not exist.

- [ ] **Step 3: Implement minimal override**

Read `USIM_INFERENCE_STEPS_OVERRIDE`, preserve `cfg.usim_steps`, optionally set the override immediately before rollout, record both values, and restore the original in `finally`.

- [ ] **Step 4: Run GREEN**

Run all wrapper and legacy inference-probe tests.

### Task 2: Simulator factorial reporter

**Files:**
- Create: `tests/test_simulator_factorial_report.py`
- Create: `simulator_factorial_report.py`

- [ ] **Step 1: Write failing audit test**

Require the new off-course-fit audit to declare checkpoint steps `[0]` and effective inference steps `[5]`; reject any other pair.

- [ ] **Step 2: Run RED, implement, and run GREEN**

Reuse `load_factorial` and `summarize_factorial` from `ppo_loss_factorial_report.py`, add the simulator-step audit gate, and export `simulator_factorial_by_seed.csv` and `simulator_factorial_summary.csv`.

### Task 3: Read-only runner

**Files:**
- Create: `run_simulator_factorial_completion.ps1`

- [ ] **Step 1: Configure the missing cell**

Use checkpoint root `main_table_51ea12fc_core_ablation/wo_simulator`, launch with `UsimSteps=0`, set `USIM_INFERENCE_STEPS_OVERRIDE=5`, and write only under `outputs/recppo_research_repair/simulator_factorial/t0_training_course_fit`.

- [ ] **Step 2: Add checkpoint hashing and cleanup**

Verify SHA-256, size, and timestamp before/after; clear every evaluation-only environment variable in `finally`.

- [ ] **Step 3: Parse-check and dry-run seed 9999**

Require zero parser errors and a clean missing-checkpoint skip.

### Task 4: Experiment and final verification

- [ ] **Step 1: Run focused tests and three-seed evaluation**

Require all tests to pass, then execute the serial runner.

- [ ] **Step 2: Verify audit and report**

Require test target, course-fit mode, Actor calls 0, train-only histories, target-seen pairs 0, checkpoint T=0, inference T=5, 102 refined items, and unchanged checkpoint hashes.

- [ ] **Step 3: Report causal effects**

Report the four cell means/stds, simulator-training effects under static/course-fit, course-fit effects under T=5/T=0 training, and their interaction.
