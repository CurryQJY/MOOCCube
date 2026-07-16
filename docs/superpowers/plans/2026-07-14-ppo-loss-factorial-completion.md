# PPO-Loss Factorial Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the missing `PPO loss=0 training + course-fit inference` cell and produce a locked three-seed 2×2 causal comparison against the three existing cells.

**Architecture:** Reuse the three frozen `wo_ppo_loss` checkpoints in read-only mode and evaluate them on test with the validation-selected `course_fit`, `T=5`, and residual 1.0 configuration. A dedicated reporter reads all four fixed result roots and computes per-seed values, mean/std, training effects, inference effects, and the difference-in-differences interaction.

**Tech Stack:** Python 3.12, pandas, pytest, PyTorch 2.8, PowerShell, existing read-only inference wrapper.

---

### Task 1: Factorial reporter

**Files:**
- Create: `tests/test_ppo_loss_factorial_report.py`
- Create: `ppo_loss_factorial_report.py`

- [ ] **Step 1: Write failing tests**

Create four temporary roots containing two seeds and one cold item-macro metric. Assert that the reporter computes:

```python
training_effect_static = on_static - off_static
training_effect_course_fit = on_course_fit - off_course_fit
inference_effect_ppo_on = on_course_fit - on_static
inference_effect_ppo_off = off_course_fit - off_static
interaction = inference_effect_ppo_on - inference_effect_ppo_off
```

Also assert that a missing seed/cell raises a clear error and that an off-course-fit audit with a nonzero Actor call is rejected.

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_ppo_loss_factorial_report.py -q --basetemp .pytest_tmp\ppo_factorial_red`

Expected: FAIL because `ppo_loss_factorial_report` is absent.

- [ ] **Step 3: Implement the minimal reporter**

Read the six cold item-macro metrics from these fixed cells:

```text
PPO=1, static     outputs/recovery_validation/main_table_51ea12fc_candidate
PPO=1, course-fit outputs/recppo_research_repair/test_course_fit_frozen/course_fit
PPO=0, static     outputs/recovery_validation/main_table_51ea12fc_core_ablation/wo_ppo_loss
PPO=0, course-fit outputs/recppo_research_repair/ppo_loss_factorial/off_course_fit
```

Validate the off-course-fit audit as test/course-fit with zero Actor calls, train-only history, zero target-seen pairs, and 102 refined items. Export `ppo_loss_factorial_by_seed.csv` and `ppo_loss_factorial_summary.csv`.

- [ ] **Step 4: Run GREEN**

Run the reporter tests and expect all tests to pass.

### Task 2: Read-only completion runner

**Files:**
- Create: `run_ppo_loss_factorial_completion.ps1`

- [ ] **Step 1: Create the runner**

For seeds 2025/2026/2027, set test targeting, `course_fit`, evaluation RNG 7001, and `USIM_COURSE_MATCH_EXCLUDE_TARGET=false`. Invoke the recovered runner with `PpoLossWeight=0`, `T=5`, residual 1.0, and checkpoint root `main_table_51ea12fc_core_ablation/wo_ppo_loss`. Write only under the new off-course-fit output root.

- [ ] **Step 2: Protect checkpoints**

Hash all three `finished.pt` files before and after evaluation and fail if hash, size, or timestamp changes.

- [ ] **Step 3: Parse-check and dry-run**

Use the PowerShell parser and seed 9999. Expect zero parser errors and a clean skip.

### Task 3: Experiment and verification

**Files:**
- Generate only: `outputs/recppo_research_repair/ppo_loss_factorial/`.

- [ ] **Step 1: Run focused tests**

Run wrapper, audit, factorial reporter, and legacy inference-probe tests.

- [ ] **Step 2: Run three-seed completion**

Run: `& .\run_ppo_loss_factorial_completion.ps1`

- [ ] **Step 3: Verify provenance**

Require all three audits to report test targeting, course-fit mode, Actor calls 0, train-only histories, target-seen pairs 0, and 102 refined items. Require all checkpoint hashes to remain unchanged.

- [ ] **Step 4: Interpret the locked 2×2 table**

Report all six cold item-macro means/stds and factorial effects. Treat this as an ablation of the explicit PPO objective only; simulator rollout/state transition remains present in both training conditions.
