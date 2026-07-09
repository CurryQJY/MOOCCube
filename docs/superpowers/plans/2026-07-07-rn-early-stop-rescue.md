# R/N Early Stop Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add R/N joint validation selection for RL rescue experiments and launch a clean MOOCCube seed2025 e60 run initialized from `content_masked_sup`.

**Architecture:** Keep the existing early-stop score hook and add two explicit modes: `cold_rn` for cold Recall/NDCG harmonic selection, and `balanced_rn` for cold/hot Recall/NDCG harmonic selection. The runner exposes the modes through the existing `-EarlyStopScoreMode` parameter; the rescue run uses a fresh output/checkpoint directory and `-InitCheckpointDir` rather than resuming old early-stop state.

**Tech Stack:** Python `unittest`, PyTorch checkpoints, PowerShell experiment runner.

---

### Task 1: Add Score Tests

**Files:**
- Modify: `D:/DeskTop/MOOCCube/tests/test_core_ablation_controls.py`

- [ ] **Step 1: Write failing tests**

Add tests importing `_compute_early_stop_score` and asserting:

```python
score = _compute_early_stop_score({"R@10": 0.4, "N@10": 0.2}, None, 10, "cold_rn")
self.assertAlmostEqual(score, 2 * 0.4 * 0.2 / (0.4 + 0.2))
```

and:

```python
score = _compute_early_stop_score(
    {"R@10": 0.4, "N@10": 0.2},
    {"R@10": 0.2, "N@10": 0.1},
    10,
    "balanced_rn",
)
self.assertAlmostEqual(score, 4 / (1 / 0.4 + 1 / 0.2 + 1 / 0.2 + 1 / 0.1))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.\py.bat tests\test_core_ablation_controls.py`

Expected: failure because `cold_rn` and `balanced_rn` still fall back to `cold_only`.

### Task 2: Implement Score Modes

**Files:**
- Modify: `D:/DeskTop/MOOCCube/usim_feedback_fast3_content_delta.py`
- Modify: `D:/DeskTop/MOOCCube/fast3_delta/config.py`
- Modify: `D:/DeskTop/MOOCCube/run_usim_feedback_fast3_content_delta_static.ps1`

- [ ] **Step 1: Add a positive harmonic helper**

Add `_positive_harmonic(values)` near `_metric_or_zero`. Return `0.0` if any value is non-positive; otherwise return `len(values) / sum(1/value)`.

- [ ] **Step 2: Add modes in `_compute_early_stop_score`**

Use Cold `R@k` and `N@k` for `cold_rn`, and Cold/Hot `R@k`/`N@k` for `balanced_rn`.

- [ ] **Step 3: Update mode validation**

Add `cold_rn` and `balanced_rn` to `Fast3Config` and the PowerShell `ValidateSet`.

- [ ] **Step 4: Run tests**

Run: `.\py.bat tests\test_core_ablation_controls.py`

Expected: all tests pass.

### Task 3: Launch Clean Seed2025 e60

**Files:**
- Create: `D:/DeskTop/MOOCCube/run_mooccube_rl_rescue_cold_rn_e60_seed2025.ps1`

- [ ] **Step 1: Create launcher**

Use `-InitCheckpointDir 'checkpoints\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer\strict_item_cold_balanced_thr1_seed_2025'`, `-EarlyStopScoreMode cold_rn`, fresh `outputs\rl_rescue\mooccube\warm_residual_0p10_cold_rn_e60_seed2025`, and fresh matching checkpoint root.

- [ ] **Step 2: Dry-run or start safely**

Check active training processes before starting. If GPU is already occupied by the old exact significance queue, do not start a competing long run unless explicitly requested.

- [ ] **Step 3: Verify log startup**

Confirm log prints the init checkpoint path, residual scale, `score_mode=cold_rn`, and checkpoint directory.
