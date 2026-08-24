# CKG-RL V3.6 Globally Stable Action Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated P-only V3.6 policy route that improves V3.5 action imitation while constraining global score drift.

**Architecture:** The V3.6 module reuses V3.5's teacher, generator, panels, and evaluator but adds an H_G-only deterministic anchor bank to action utilities and an expert/actor state-transition mixture. It selects only with P_val and does not open test data.

**Tech Stack:** Python 3, PyTorch, pandas, pytest, V3.2 clean primitives, V3.3 rank panels, and V3.5 action-distillation helpers.

---

### Task 1: Deterministic Global Anchor Utility

**Files:**
- Create: `tests/test_ckg_rl_usim_v36_global_stable_distill.py`
- Create: `ckg_rl_usim_v36_global_stable_distill.py`

- [ ] **Step 1: Write failing anchor and utility tests.**

```python
def test_global_anchor_bank_is_h_g_only_and_deterministic():
    first = build_global_anchor_bank(h_g_rows, seed=7, anchor_count=2)
    second = build_global_anchor_bank(h_g_rows, seed=7, anchor_count=2)
    assert first.user_ids == second.user_ids
    assert set(first.user_ids).issubset(set(h_g_rows["u_idx"]))

def test_global_stability_weight_changes_candidate_preference_but_not_end_utility():
    teacher, engine, anchors = _small_engine_and_anchors()
    _, _, utilities, _ = stable_counterfactual_action_targets(
        engine, state=torch.tensor([[0.0, 1.0]]),
        target_emb=torch.tensor([[1.0, 0.0]]), user_bank=teacher.user_vectors(),
        item_ids=torch.tensor([0]), anchor_vectors=anchors,
        action_temperature=0.005, global_stability_weight=10.0,
    )
    assert utilities[0, -1].item() == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests before implementation.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v36_global_stable_distill.py -q --basetemp .pytest_tmp/v36_red1`

Expected: collection failure for `ckg_rl_usim_v36_global_stable_distill`.

- [ ] **Step 3: Implement immutable anchor bank and stable action utilities.**

```python
drift = torch.matmul(candidate_states - target_emb.unsqueeze(1), anchor_vectors.t()).pow(2).mean(dim=2)
stability_delta = drift_after - drift_before
utilities = torch.cat((local_gain - weight * stability_delta, end_zero), dim=1)
```

- [ ] **Step 4: Run anchor and utility tests.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v36_global_stable_distill.py -q --basetemp .pytest_tmp/v36_green1`

Expected: PASS.

### Task 2: Expert-State Mixture and P-Val Diagnostics

**Files:**
- Modify: `tests/test_ckg_rl_usim_v36_global_stable_distill.py`
- Modify: `ckg_rl_usim_v36_global_stable_distill.py`

- [ ] **Step 1: Write failing deterministic-mixture and selection tests.**

```python
def test_expert_mask_is_repeatable_and_respects_zero_and_one_fraction():
    ids = torch.tensor([10, 11, 12])
    assert not deterministic_expert_mask(ids, seed=7, epoch=1, step=0, fraction=0.0).any()
    assert deterministic_expert_mask(ids, seed=7, epoch=1, step=0, fraction=1.0).all()

def test_v36_selection_uses_only_nonnegative_p_val_gain():
    assert select_stable_policy_row([{"epoch": 0, "p_val_rank_gain": 0.0}, {"epoch": 1, "p_val_rank_gain": -0.1}])["epoch"] == 0
```

- [ ] **Step 2: Run tests before mixture implementation.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v36_global_stable_distill.py -q --basetemp .pytest_tmp/v36_red2`

Expected: FAIL for missing mixture and selection helpers.

- [ ] **Step 3: Implement train-only mixed trajectory collection.**

```python
expert_actions = utilities.argmax(dim=1)
use_expert = deterministic_expert_mask(item_ids, seed=config.seed, epoch=epoch, step=step, fraction=config.expert_action_fraction)
rollout_actions = torch.where(use_expert, expert_actions, actor_actions)
next_state = state + engine.step_size * selected_user_vectors(rollout_actions)
```

Compute P_val rank gain exactly as V3.5 and separately record P_val anchor
drift, action agreement, expert-transition rate, and actor END rate.

- [ ] **Step 4: Run all V3.6 unit tests.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v36_global_stable_distill.py -q --basetemp .pytest_tmp/v36_green2`

Expected: PASS.

### Task 3: P-Only Pipeline, Provenance, and Seed-2025 Screen

**Files:**
- Modify: `tests/test_ckg_rl_usim_v36_global_stable_distill.py`
- Modify: `ckg_rl_usim_v36_global_stable_distill.py`
- Create: `run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1`

- [ ] **Step 1: Write failing tiny-pipeline and launcher tests.**

```python
def test_v36_pipeline_writes_anchor_manifest_and_never_loads_test(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "load_clean_test_inputs", _fail_test_access)
    result = run_global_stable_pipeline(_tiny_config(tmp_path))
    manifest = json.loads((tmp_path / "output" / "v36_manifest.json").read_text())
    assert manifest["test_loaded"] is False
    assert manifest["outer_c_val_evaluated"] is False
    assert (tmp_path / "output" / "global_anchor_manifest.json").is_file()
```

- [ ] **Step 2: Implement source pipeline and pinned launcher.**

```python
partitions = v35._build_p_only_partitions(train_df, val_df, n_items=int(content.size(0)), config=config)
anchor_bank = build_global_anchor_bank(views.generator_train, seed=config.seed, anchor_count=config.global_anchor_count)
engine, selected, rows = train_global_stable_policy(
    teacher, generator, engine, views, content=content, user_history=user_history,
    anchor_bank=anchor_bank, config=config,
)
```

Write source stage hashes, fixed switches, anchor digest, P_val-only selection,
and no test metric to the V3.6 manifest.

- [ ] **Step 3: Run regression, dry-run, smoke, and one P-only seed-2025 screen.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py tests/test_ckg_rl_usim_v33_rank_distill.py tests/test_ckg_rl_usim_v34_rank_reward_control.py tests/test_ckg_rl_usim_v35_action_distill.py tests/test_ckg_rl_usim_v36_global_stable_distill.py -q --basetemp .pytest_tmp/v36_regression`

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1 -DryRun`

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1 -Smoke -RunTag preflight`

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1`

Expected: a source-hash-matched, P-only selected checkpoint with no C_val/test output. Compare its P_val gain and action agreement against V3.5 before considering any fresh-seed outer evaluation.
