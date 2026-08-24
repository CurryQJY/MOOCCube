# Decoupled Pseudo-Cold Severity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate real-cold Ridge initialization strength from pseudo-cold PPO simulation severity while preserving historical defaults.

**Architecture:** Resolve an optional simulation alpha at the Ridge runner boundary, apply it only to the pseudo-cold simulation bank, and persist both resolved coefficients. Thread the parameter through the staged core runner without changing any other training or selection behavior.

**Tech Stack:** Python 3.12, PyTorch, argparse, pytest.

---

### Task 1: Define The Runner Contract

**Files:**
- Modify: `tests/test_ridge_course_reward_rl_pilot.py`
- Modify: `ridge_course_reward_rl_pilot.py`

- [ ] **Step 1: Write failing parser and resolver tests**

Assert that the parser default leaves `simulation_ridge_alpha` unset, an
explicit `--simulation-ridge-alpha 1.0` is parsed, and a pure resolver returns
the real alpha when unset but the explicit value when supplied.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q tests/test_ridge_course_reward_rl_pilot.py -k simulation_ridge_alpha
```

Expected: collection or assertion failure because the resolver and parser
option do not exist.

- [ ] **Step 3: Implement the minimal resolver and parser option**

Add `resolve_simulation_ridge_alpha(ridge_alpha, simulation_ridge_alpha)` and
`--simulation-ridge-alpha`. Reject values outside `[0, 1]` through the same
coefficient contract used by `blend_ridge_rows`.

- [ ] **Step 4: Use and record the resolved coefficient**

Apply `ridge_alpha` to `ridge_bank`, apply the resolved simulation coefficient
to `sim_bank`, and record both in the manifest. Keep the default behavior equal
to the previous shared coefficient.

- [ ] **Step 5: Run focused and full Ridge tests**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q tests/test_ridge_course_reward_rl_pilot.py
```

Expected: all tests pass.

### Task 2: Thread The Staged Route Option

**Files:**
- Modify: `tests/test_graph_course_core_finetune_pilot.py`
- Modify: `graph_course_core_finetune_pilot.py`

- [ ] **Step 1: Write a failing forwarding test**

Extend the existing downstream namespace test to pass
`simulation_ridge_alpha=1.0` and assert the Ridge runner receives it unchanged.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q tests/test_graph_course_core_finetune_pilot.py -k simulation_ridge_alpha
```

Expected: assertion failure because the staged runner does not forward the new
field.

- [ ] **Step 3: Add the parser field and forwarding assignment**

Add the optional CLI argument and include it in the downstream Namespace. Do
not change any other defaults.

- [ ] **Step 4: Run the combined regression suite**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest -q tests/test_ridge_course_reward_rl_pilot.py tests/test_graph_course_core_finetune_pilot.py
D:/anaconda3/envs/req_py312/python.exe -m py_compile ridge_course_reward_rl_pilot.py graph_course_core_finetune_pilot.py
```

Expected: all tests and compilation pass.

### Task 3: Validate The Mechanism Hypothesis

**Files:**
- Create generated outputs under: `outputs/xds_coco_decoupled_a075_sim1/`

- [ ] **Step 1: Run COCO seed 2025 validation-only**

Use the existing frozen source checkpoint and all existing PPO/reward/selection
arguments, changing only `--simulation-ridge-alpha 1.0`. Keep `--skip-test`.

- [ ] **Step 2: Apply the single-seed screen**

Require selected epoch greater than zero, selected delta greater than zero,
positive Cold NDCG@10 versus anchored Ridge, and all four retention metrics.

- [ ] **Step 3: Freeze or reject**

If the screen passes, freeze `(alpha, alpha_sim)=(0.075, 1.0)` for the 3-dataset
3-seed validation matrix. If it fails, reject this hypothesis without stacking
additional changes.
