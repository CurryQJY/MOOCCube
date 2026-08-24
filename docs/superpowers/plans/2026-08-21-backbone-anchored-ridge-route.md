# Backbone-Anchored Ridge Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate a bounded Backbone-to-Ridge residual into core and PPO training, enforce strict validation retention, and automatically validate the selected route on Junyi.

**Architecture:** Add one pure row-blending helper to the Ridge runner and reuse it for real-cold and pseudo-cold banks. Thread `ridge_alpha` through the core-to-downstream boundary, and make core epoch selection use the same four retention metrics as the downstream route. Historical behavior remains the default at `alpha=1.0`.

**Tech Stack:** Python 3.12, PyTorch, pandas, pytest, existing GraphContentScorer/Ridge/PPO runners.

---

### Task 1: Specify the anchored bank contract

**Files:**
- Modify: `tests/test_ridge_course_reward_rl_pilot.py`
- Modify: `ridge_course_reward_rl_pilot.py`

- [ ] **Step 1: Write failing endpoint and validation tests**

```python
def test_blend_ridge_rows_has_identity_and_ridge_endpoints():
    base = F.normalize(torch.randn(5, 4), dim=1)
    ridge = F.normalize(torch.randn(5, 4), dim=1)
    ids = torch.tensor([1, 3])
    assert torch.equal(blend_ridge_rows(base, ridge, ids, alpha=0.0), base)
    actual = blend_ridge_rows(base, ridge, ids, alpha=1.0)
    assert torch.allclose(actual[ids], ridge[ids])
    assert torch.equal(actual[[0, 2, 4]], base[[0, 2, 4]])

@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_blend_ridge_rows_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="ridge alpha"):
        blend_ridge_rows(torch.eye(2), torch.eye(2), [0], alpha=alpha)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:/anaconda3/envs/req_py312/python.exe -m pytest tests/test_ridge_course_reward_rl_pilot.py -k blend_ridge_rows -q`

Expected: collection fails because `blend_ridge_rows` is not defined.

- [ ] **Step 3: Implement the pure helper**

Add a function that validates matching 2-D shapes and `alpha` in `[0, 1]`,
copies the Backbone bank, blends only selected rows, normalizes moved rows, and
returns the copy.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

### Task 2: Wire the coefficient through Ridge and PPO

**Files:**
- Modify: `tests/test_ridge_course_reward_rl_pilot.py`
- Modify: `ridge_course_reward_rl_pilot.py`

- [ ] **Step 1: Write failing runner contract tests**

Assert that parser/default arguments expose `ridge_alpha=1.0`, manifests record
the configured value, and both final and simulation Ridge banks call the pure
blend helper before validation or rollout.

- [ ] **Step 2: Verify RED**

Run: `D:/anaconda3/envs/req_py312/python.exe -m pytest tests/test_ridge_course_reward_rl_pilot.py -k ridge_alpha -q`

Expected: assertions fail because the argument and manifest field are absent.

- [ ] **Step 3: Implement minimal wiring**

Add `--ridge-alpha`, default `1.0`. Keep full Ridge predictions for diagnostics,
then transform final strict-cold rows and simulation target rows with
`blend_ridge_rows`. Record the coefficient and initializer name in the run
manifest.

- [ ] **Step 4: Verify GREEN**

Run the focused command and the complete Ridge pilot test file.

### Task 3: Align core selection and downstream handoff

**Files:**
- Modify: `tests/test_graph_course_core_finetune_pilot.py`
- Modify: `graph_course_core_finetune_pilot.py`

- [ ] **Step 1: Write failing strict-selection tests**

Create a history where an epoch has better cold NDCG but violates each of hot
Recall, hot NDCG, overall Recall, and overall NDCG separately. Assert all four
epochs are rejected and a fully feasible epoch is selected.

- [ ] **Step 2: Verify RED**

Run: `D:/anaconda3/envs/req_py312/python.exe -m pytest tests/test_graph_course_core_finetune_pilot.py -k "strict or ridge_alpha" -q`

Expected: the current hot-NDCG-only selector chooses an invalid epoch.

- [ ] **Step 3: Implement strict selection and alpha forwarding**

Add the four floors in `select_validation_epoch`, blend the validation Ridge
bank with the current Backbone item bank, add `--ridge-alpha` default `1.0`,
and copy the value into `make_downstream_args`.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run both complete focused test files.

### Task 4: Run Route A validation gate

**Files:**
- Create outputs under: `outputs/xds_junyi_anchored_a075/`

- [ ] **Step 1: Run seed 2026 validation-only**

```powershell
D:/anaconda3/envs/req_py312/python.exe graph_course_core_finetune_pilot.py `
  --seed 2026 --epochs 3 --lr 3e-5 --lambda-course 0.05 `
  --lambda-pseudo 0.10 --ridge-alpha 0.075 `
  --data-dir processed_data_junyi --split-root outputs/junyi/main_table_3seed `
  --source-root outputs/graph_knp_junyi_ms12ft03 `
  --prereq-path outputs/prereq_target_junyi/prereq_index_topk10.pt `
  --prereq-weight 2.0 --course-relation-dir processed_data_junyi/relations `
  --downstream-ppo-arms ridge_ppo_full --downstream-skip-test `
  --out outputs/xds_junyi_anchored_a075/full/seed2026
```

- [ ] **Step 2: Apply the automatic gate**

Read `core_manifest.json`, `core_val_history.json`, and downstream
`pilot_results.json`. Advance Route A only under the four criteria in the
design. Otherwise stop Route A and write the evidence required for Route B.

- [ ] **Step 3: Run remaining validation seeds when eligible**

Repeat Step 1 for seeds 2025 and 2027 without changing any hyperparameter.

- [ ] **Step 4: Lock or reject Route A**

Aggregate per-seed cold gain and retention relative to epoch 0. Lock Route A
only when all seeds pass retention and mean cold gain is positive. Do not read
test otherwise.

### Task 5: Final verification

**Files:**
- Inspect: `ridge_course_reward_rl_pilot.py`
- Inspect: `graph_course_core_finetune_pilot.py`
- Inspect: `outputs/xds_junyi_anchored_a075/`

- [ ] **Step 1: Run unit and syntax verification**

```powershell
D:/anaconda3/envs/req_py312/python.exe -m pytest tests/test_ridge_course_reward_rl_pilot.py tests/test_graph_course_core_finetune_pilot.py -q
D:/anaconda3/envs/req_py312/python.exe -m py_compile ridge_course_reward_rl_pilot.py graph_course_core_finetune_pilot.py
```

- [ ] **Step 2: Verify protocol provenance**

Confirm every validation-only manifest records `ridge_alpha=0.075`,
`test_loaded=false` in core, and `test_skipped=true` downstream.

- [ ] **Step 3: Review the final diff**

Run `git diff -- ridge_course_reward_rl_pilot.py graph_course_core_finetune_pilot.py tests/test_ridge_course_reward_rl_pilot.py tests/test_graph_course_core_finetune_pilot.py docs/superpowers/specs/2026-08-21-backbone-anchored-ridge-route-design.md docs/superpowers/plans/2026-08-21-backbone-anchored-ridge-route.md` and confirm no unrelated changes.
