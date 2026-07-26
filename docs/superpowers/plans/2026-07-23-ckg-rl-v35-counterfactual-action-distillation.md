# CKG-RL V3.5 Counterfactual Action Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated V3.5 seed-2025 viability route that distills legal counterfactual teacher actions on `P_train`, selects only with `P_val`, and never loads test data.

**Architecture:** `ckg_rl_usim_v35_action_distill.py` reuses the frozen V3.2 teacher/generator and V3.3 rank panels, but replaces PPO with a supervised actor loss over all legal actions plus `END`.  The pipeline writes separate provenance artifacts and performs P-only selection.

**Tech Stack:** Python 3, PyTorch, pandas, pytest, existing `ckg_rl_usim_v32_clean.py` and `ckg_rl_usim_v33_rank_distill.py` helpers.

---

## File Structure

- Create: `ckg_rl_usim_v35_action_distill.py` - V3.5 config, counterfactual labels, on-policy action distillation, P-only pipeline, and CLI.
- Create: `run_ckg_rl_usim_v35_action_distill_seed2025.ps1` - isolated non-overwriting launcher with clean-route environment locks.
- Create: `tests/test_ckg_rl_usim_v35_action_distill.py` - math, visibility, artifact, and launcher contracts.
- Create: `docs/superpowers/specs/2026-07-23-ckg-rl-v35-counterfactual-action-distillation-design.md` - approved protocol and gates.
- Create: `docs/superpowers/plans/2026-07-23-ckg-rl-v35-counterfactual-action-distillation.md` - this execution plan.

### Task 1: Counterfactual Action Labels

**Files:**
- Create: `tests/test_ckg_rl_usim_v35_action_distill.py`
- Create: `ckg_rl_usim_v35_action_distill.py`

- [ ] **Step 1: Write failing tests for `END`, candidate axes, and label probability invariants.**

```python
def _small_rank_engine() -> tuple[clean.CleanTeacher, rank.RankDistilledUSIMEngine, rank.RankPanels]:
    teacher = _teacher()
    panels = rank.RankPanels(
        item_ids=(0,), panel_ids=torch.tensor([[0, 1]]),
        positive_counts=(0,), hard_counts=(0,), panel_size=2, seed=7,
    )
    engine = rank.RankDistilledUSIMEngine(
        emb_dim=2, hidden_dim=4, max_steps=1, candidate_count=1,
        step_size=0.5, step_penalty=0.0, max_delta=1.0,
        rank_panels=panels, rank_temperature=0.2,
        course_reward_weight=0.0, delta_weight=0.0,
    )
    return teacher, engine, panels

def test_counterfactual_action_targets_include_end_and_rank_the_positive_action():
    teacher, engine, _ = _small_rank_engine()
    labels, utilities = counterfactual_action_targets(
        engine, state=torch.tensor([[0.0, 1.0]]),
        target_emb=torch.tensor([[1.0, 0.0]]), user_bank=torch.eye(2),
        item_ids=torch.tensor([0]), action_temperature=0.005,
    )
    assert labels.shape == (1, 2)
    assert utilities.shape == (1, 2)
    assert labels.sum(dim=1).item() == pytest.approx(1.0)
    assert utilities.argmax(dim=1).item() == 0

def test_counterfactual_action_targets_prefer_end_when_every_user_action_hurts():
    _, engine, _ = _small_rank_engine()
    labels, utilities = counterfactual_action_targets(
        engine, state=torch.tensor([[1.0, 0.0]]),
        target_emb=torch.tensor([[1.0, 0.0]]), user_bank=torch.eye(2),
        item_ids=torch.tensor([0]), action_temperature=0.005,
    )
    assert utilities.argmax(dim=1).item() == 1
    assert torch.isfinite(labels).all()
```

- [ ] **Step 2: Run the tests and verify they fail because the V3.5 module does not exist.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py -q`

Expected: collection failure for `ckg_rl_usim_v35_action_distill`.

- [ ] **Step 3: Implement `ActionDistillConfig`, `counterfactual_action_targets`, and a finite soft-target loss.**

```python
utilities = torch.cat((candidate_gain, torch.zeros((batch, 1), device=state.device)), dim=1)
target_probs = torch.softmax(utilities / config.action_temperature, dim=1)
loss = -(target_probs * actor_log_probs).sum(dim=1).mean()
```

Candidate gains must be evaluated with flattened `[batch * candidates, dim]`
states and reshaped back to `[batch, candidates]`; do not broadcast a
`[batch, 1, 1]` tensor against `[batch, candidates]`.

- [ ] **Step 4: Run the label tests and verify they pass.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py -q`

Expected: PASS for the two action-label tests.

### Task 2: On-Policy Actor Distillation and P-Val Selection

**Files:**
- Modify: `tests/test_ckg_rl_usim_v35_action_distill.py`
- Modify: `ckg_rl_usim_v35_action_distill.py`

- [ ] **Step 1: Write failing tests for inference oracle rejection and P-only selection.**

```python
def test_action_distill_policy_selection_rejects_negative_p_val_gain():
    selected = select_action_distill_policy_row([
        {"epoch": 0, "p_val_rank_gain": 0.0},
        {"epoch": 1, "p_val_rank_gain": -0.01},
    ])
    assert selected["epoch"] == 0

def test_target_free_v35_inference_accepts_unpanelled_cold_item_and_rejects_target():
    _, engine, _ = _small_rank_engine()
    state = torch.tensor([[1.0, 0.0]])
    result = engine.rollout(
        state, user_bank=torch.eye(2), training=False,
        item_ids=torch.tensor([99]), user_history={},
    )
    assert result.final_state.shape == state.shape
    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(
            state, user_bank=torch.eye(2), training=False,
            target_emb=state, item_ids=torch.tensor([99]), user_history={},
        )
```

- [ ] **Step 2: Run the tests and verify selection/training helpers are missing.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py -q`

Expected: FAIL for missing selection/training helpers.

- [ ] **Step 3: Implement on-policy state collection and `train_action_distilled_policy`.**

```python
for step in range(config.max_steps):
    labels = counterfactual_action_targets(
        engine, state=active_state, target_emb=teacher_items,
        user_bank=user_bank, item_ids=active_ids,
        action_temperature=config.action_temperature,
    )
    cached_steps.append(labels)
    candidate_ids = engine.legal_candidate_ids(active_state, user_bank)
    candidate_vectors = engine._candidate_vectors(user_bank, candidate_ids)
    actor_action = engine.policy.action_value(
        active_state, remaining_steps, candidate_vectors,
        candidate_logit_bias=candidate_bias, deterministic=True,
    )[0]
    end_action = candidate_vectors.size(1)
    safe_action = actor_action.clamp_max(end_action - 1)
    selected = candidate_vectors[torch.arange(active_state.size(0)), safe_action]
    active_state = torch.where(
        actor_action.eq(end_action).view(-1, 1), active_state,
        active_state + engine.step_size * selected,
    )
loss = torch.stack([action_distillation_loss(engine.policy, step) for step in cached_steps]).mean()
```

Use only `P_train` item IDs for label construction.  Evaluate each candidate
checkpoint with `rank._policy_rank_diagnostics` only on `P_val`; choose the
largest non-negative `P_val` gain, with epoch 0 as identity.

- [ ] **Step 4: Run the unit tests and verify they pass.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py -q`

Expected: PASS.

### Task 3: P-Only Pipeline, Provenance, and Launcher

**Files:**
- Modify: `tests/test_ckg_rl_usim_v35_action_distill.py`
- Modify: `ckg_rl_usim_v35_action_distill.py`
- Create: `run_ckg_rl_usim_v35_action_distill_seed2025.ps1`

- [ ] **Step 1: Write failing smoke-pipeline and launcher contracts.**

```python
def test_v35_pipeline_writes_p_only_artifacts_and_never_loads_test(tmp_path):
    config = _config(tmp_path)
    _write_tiny_split(config.data_dir, config.split_dir)
    result = run_action_distill_pipeline(config)
    manifest = json.loads((output_dir / "action_distill_manifest.json").read_text())
    assert manifest["test_loaded"] is False
    assert manifest["generator_rank_loss"] is False
    assert not (output_dir / "final_metrics.json").exists()
    assert (output_dir / "p_val_selected_metrics.json").is_file()

def test_v35_launcher_locks_target_free_controls():
    launcher = Path("run_ckg_rl_usim_v35_action_distill_seed2025.ps1").read_text()
    assert 'USIM_CLEAN_RANDOM_ID_DROPOUT' in launcher
    assert 'USIM_CLEAN_CANDIDATE_MODE' in launcher
    assert '--dry-run' in launcher
```

- [ ] **Step 2: Run the tests and verify they fail because the P-only route and launcher are absent.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py -q`

Expected: FAIL for missing pipeline/launcher.

- [ ] **Step 3: Implement `run_action_distill_pipeline`, manifests, CSV exports, and the pinned seed-2025 launcher.**

```python
meta, content, train_df, val_df = clean.load_clean_train_val_inputs(config)
partitions = _build_p_only_partitions(
    train_df, val_df, n_items=int(content.size(0)), config=config,
)
views = clean.build_stage_views(partitions)
engine, selected, rows = train_action_distilled_policy(
    teacher, generator, engine, views, content=content,
    user_history=user_history, config=config,
)
clean._write_json(output_dir / "p_val_selected_metrics.json", selected)
```

The manifest must show that generator fitting occurred before panel construction,
write teacher/generator/policy hashes, and state `test_loaded=false`.

- [ ] **Step 4: Run the full V3.5 test module and existing V3.2/V3.3/V3.4 regression tests.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v35_action_distill.py tests/test_ckg_rl_usim_v32_clean.py tests/test_ckg_rl_usim_v33_rank_distill.py tests/test_ckg_rl_usim_v34_rank_reward_control.py -q`

Expected: PASS.

### Task 4: Preflight and Seed-2025 Viability Experiment

**Files:**
- Modify: `run_ckg_rl_usim_v35_action_distill_seed2025.ps1`

- [ ] **Step 1: Run dry-run contract validation.**

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v35_action_distill_seed2025.ps1 -DryRun`

Expected: JSON reports `route=ckg_rl_usim_v35_action_distill`, valid `P_train`/`P_val` counts, and no output overwrite.

- [ ] **Step 2: Run one smoke pipeline in an isolated smoke root.**

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v35_action_distill_seed2025.ps1 -Smoke -RunTag preflight`

Expected: exit code 0, P-only artifacts, and `test_loaded=false`.

- [ ] **Step 3: Launch the one seed-2025 P-only viability run in a fresh root.**

Run: `powershell -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v35_action_distill_seed2025.ps1`

Expected: no `static_test.pkl` read, matching teacher/generator hashes, and a selected epoch determined solely by `P_val` rank gain.

- [ ] **Step 4: Audit gates before any outer/cross-seed experiment.**

Run: inspect `action_distill_manifest.json`, `policy_action_epochs.csv`, and `p_val_selected_metrics.json`.

Expected: selected non-identity epoch with strictly positive `P_val` rank gain; otherwise reject V3.5 and do not run a multi-seed campaign.
