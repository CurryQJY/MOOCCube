# CKG-RL V3.4 Rank-Reward-Only Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. The shared workspace is dirty; create only V3.4 files and preserve all earlier routes and artifacts.

**Goal:** Run a single-seed causal control that holds V3.2 vector generation fixed while evaluating V3.3 rank-gain PPO reward.

**Architecture:** `ckg_rl_usim_v34_rank_reward_control.py` imports V3.2's teacher/generator/evaluator and V3.3's rank-panel policy engine. It trains the V3.2 generator before constructing policy panels, then runs only the V3.3 target-free rank-reward policy. A new launcher writes only V3.4 roots.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell.

---

### Task 1: Lock the vector-generator control contract

**Files:**
- Create: `tests/test_ckg_rl_usim_v34_rank_reward_control.py`
- Create: `ckg_rl_usim_v34_rank_reward_control.py`

- [ ] Write a failing test showing that `RankRewardControlConfig` rejects a nonzero generator rank weight and resolves V3.4-only roots.

```python
def test_control_config_forbids_generator_rank_loss_and_uses_v34_roots():
    config = RankRewardControlConfig.for_seed(2025)
    assert "v34_rank_reward_control" in str(config.output_dir)
    with pytest.raises(ValueError, match="rank weight"):
        validate_rank_reward_control_config(replace(config, generator_rank_weight=0.1))
```

- [ ] Run to RED, implement the config/validation, then run to GREEN.

### Task 2: Lock policy-only panels and target-free engine wiring

**Files:**
- Modify: `tests/test_ckg_rl_usim_v34_rank_reward_control.py`
- Modify: `ckg_rl_usim_v34_rank_reward_control.py`

- [ ] Write a failing test proving `build_policy_rank_panels` accepts exactly `P_train union P_val`, rejects `H_G`, and returns no panel for a strict-cold ID.

```python
def test_control_panels_are_limited_to_policy_pseudo_items():
    panels = build_policy_rank_panels(_teacher(), _train(), p_train={1}, p_val={2}, config=_config())
    assert panels.item_ids == (1, 2)
    with pytest.raises(KeyError, match="panel"):
        panels.panel_for(torch.tensor([9]), device=torch.device("cpu"))
```

- [ ] Run to RED, implement the wrapper around V3.3 panel construction, then run to GREEN.

### Task 3: Implement isolated pipeline and audit files

**Files:**
- Modify: `tests/test_ckg_rl_usim_v34_rank_reward_control.py`
- Modify: `ckg_rl_usim_v34_rank_reward_control.py`

- [ ] Write a failing temporary-data pipeline test that asserts `generator_rank_loss=false`, `P_train_rank_gain_reward_legal_candidates`, delayed test load, and all three checkpoints.
- [ ] Implement the pipeline using `clean.train_content_generator` before `build_policy_rank_panels`, then V3.3 `RankDistilledUSIMEngine` and `train_rank_distilled_policy`.
- [ ] Write `generator_vector_epochs.csv`, rank diagnostics, manifests, and final metrics. Run the test to GREEN.

### Task 4: Add launcher, regression gate, and seed-2025 run

**Files:**
- Create: `run_ckg_rl_usim_v34_rank_reward_control_seed2025.ps1`
- Modify: `tests/test_ckg_rl_usim_v34_rank_reward_control.py`

- [ ] Add a failing launcher text-contract test for V3.4 roots, `--use-course-signal`, `DryRun`, no random ID mask, and no legacy checkpoint variables.
- [ ] Implement the launcher and run focused V3.4/V3.2/V3.3 tests, compile, and DryRun.
- [ ] Run a V3.4 smoke and compare its teacher/generator hashes with V3.2 before launching seed 2025.
- [ ] Launch the isolated full seed-2025 run. Report epoch-0 versus selected-PPO cold/hot/overall metrics, P_val gain, and V3.2 generator-hash parity.
