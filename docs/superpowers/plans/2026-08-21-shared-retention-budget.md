# Shared Retention Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce one Backbone-relative retention budget across anchored Ridge and PPO while preserving historical runner defaults.

**Architecture:** Add pure grid validation and policy-row selection helpers, compute both Backbone and Ridge validation baselines, and choose the configured reference at the runner boundary. Pass an explicit refined grid and Backbone reference only for the new route.

**Tech Stack:** Python 3.12, PyTorch, argparse, pytest.

---

### Task 1: Specify Grid And Selection Contracts

**Files:**
- Modify: `tests/test_ridge_course_reward_rl_pilot.py`
- Modify: `ridge_course_reward_rl_pilot.py`

- [ ] **Step 1: Write failing tests**

Test that the historical grid remains the parser default, the refined grid is
accepted, invalid/non-increasing grids are rejected, and a candidate that
passes Ridge-relative retention but fails Backbone-relative retention is
rejected when Backbone metrics are supplied.

- [ ] **Step 2: Verify RED**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q --basetemp=.pytest-tmp/shared_budget_red tests/test_ridge_course_reward_rl_pilot.py -k "delta_grid or policy_row"
```

Expected: failure because the resolver and selector do not exist.

- [ ] **Step 3: Implement pure helpers**

Add `resolve_delta_grid(values)` and
`select_policy_row(epoch_rows, retention_baseline, tolerance)`. Preserve the
existing tie-break order exactly.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and then the complete Ridge test file.

### Task 2: Wire The Shared Reference

**Files:**
- Modify: `tests/test_ridge_course_reward_rl_pilot.py`
- Modify: `ridge_course_reward_rl_pilot.py`

- [ ] **Step 1: Extend parser tests**

Assert defaults are `retention_reference=ridge` and the historical grid, while
explicit Backbone reference and refined grid parse unchanged.

- [ ] **Step 2: Compute both validation baselines**

Evaluate Backbone and anchored Ridge banks on the same validation dataframe.
Store both metrics and the resolved reference in the manifest and bundle.

- [ ] **Step 3: Use the configured reference everywhere**

Pass the resolved grid to PPO, greedy, and random-arm evaluation. Use the same
configured retention metrics for all eligibility checks. Keep `ridge_base` as
the reported incremental PPO baseline.

- [ ] **Step 4: Run the Ridge regression suite**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q --basetemp=.pytest-tmp/shared_budget_ridge tests/test_ridge_course_reward_rl_pilot.py
```

Expected: all tests pass.

### Task 3: Thread Staged Runner Options

**Files:**
- Modify: `tests/test_graph_course_core_finetune_pilot.py`
- Modify: `graph_course_core_finetune_pilot.py`

- [ ] **Step 1: Write a failing forwarding test**

Pass `retention_reference=backbone` and the refined grid through
`make_downstream_args`, and assert exact preservation.

- [ ] **Step 2: Add CLI and Namespace forwarding**

Keep historical defaults. Do not change core epoch selection in this task.

- [ ] **Step 3: Verify both suites and compilation**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q --basetemp=.pytest-tmp/shared_budget_all tests/test_ridge_course_reward_rl_pilot.py tests/test_graph_course_core_finetune_pilot.py
D:/anaconda3/envs/req_py312/python.exe -m py_compile ridge_course_reward_rl_pilot.py graph_course_core_finetune_pilot.py
```

### Task 4: Run The Global-Budget Screen

**Files:**
- Create generated output under: `outputs/xds_mooccube_globalret_a075_sim1/frozen/seed3030/`

- [ ] **Step 1: Run validation-only seed 3030**

Use frozen Graph-KNP, real alpha `0.075`, simulation alpha `1.0`, Backbone
retention, the fixed refined grid, and `--skip-test`.

- [ ] **Step 2: Apply the screen**

Require nonzero delta, positive PPO Cold NDCG@10 over Ridge, and all four final
metrics within `0.003` of Backbone.

- [ ] **Step 3: Freeze or reject**

Only a passing screen advances to the full 3-dataset validation matrix.
