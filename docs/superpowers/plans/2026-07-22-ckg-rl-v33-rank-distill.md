# CKG-RL V3.3 Rank-Distilled USIM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. The worktree is deliberately dirty. Create only the V3.3 files listed below and do not edit historical/V3.2 routes.

**Goal:** Build an isolated single-model CKG-RL V3.3 route where content generation and USIM rollout are trained against frozen teacher user-affinity ranking distributions, then run the seed-2025 acceptance experiment.

**Architecture:** `ckg_rl_usim_v33_rank_distill.py` imports V3.2's split, teacher, evaluator, and target-free item-bank utilities without modifying them. It adds deterministic train-only rank panels, a rank-calibrated generator trainer, and a `RankDistilledUSIMEngine` whose PPO reward is incremental panel-KL improvement plus an observable course constraint. `run_ckg_rl_usim_v33_rank_distill_seed2025.ps1` owns fresh output/checkpoint roots.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell.

---

## Files

- Create: `ckg_rl_usim_v33_rank_distill.py` - V3.3 config, panel construction, rank losses, engine subclass, orchestration, CLI.
- Create: `tests/test_ckg_rl_usim_v33_rank_distill.py` - unit and minimal pipeline contracts.
- Create: `run_ckg_rl_usim_v33_rank_distill_seed2025.ps1` - fresh seed-2025 launcher.
- Create: `docs/superpowers/specs/2026-07-22-ckg-rl-v33-rank-distill-design.md` - approved design.
- Create: `docs/superpowers/plans/2026-07-22-ckg-rl-v33-rank-distill.md` - this plan.

## Task 1: Lock deterministic panel and rank math contracts

- [ ] Write failing tests for a panel builder whose result is identical across calls, uses only the supplied `H_train` frame, has no duplicate IDs, and records fixed panel width.

```python
def test_rank_panels_are_deterministic_train_only_and_fixed_width():
    teacher = _teacher_with_known_vectors()
    train = _frame([(0, 0), (1, 0), (2, 1), (3, 1)])
    first = build_rank_panels(teacher, train, item_ids={0, 1}, seed=7, panel_size=4,
                              positive_count=1, hard_count=1)
    changed_outer = _frame([(99, 0), (98, 9)])
    second = build_rank_panels(teacher, train, item_ids={0, 1}, seed=7, panel_size=4,
                               positive_count=1, hard_count=1)
    assert first.item_ids == second.item_ids
    assert first.panel_ids.equal(second.panel_ids)
    assert first.panel_ids.shape == (2, 4)
    assert all(len(set(row.tolist())) == 4 for row in first.panel_ids)
```

- [ ] Run the focused test and verify it fails because `build_rank_panels` is absent.

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v33_rank_distill.py -q --basetemp .pytest_tmp\v33_panels_red
```

- [ ] Implement `RankPanels`, `build_rank_panels`, `panel_distribution`, `rank_kl`, and `incremental_rank_gain`. `rank_kl` must return `KL(q_target || q_state)`, so a state equal to the teacher item has zero KL and a transition toward it has positive gain.

```python
def panel_distribution(user_vectors, state, panel_ids, temperature):
    users = F.normalize(user_vectors.index_select(0, panel_ids.reshape(-1)), dim=1)
    users = users.view(panel_ids.size(0), panel_ids.size(1), -1)
    scores = torch.einsum("bkd,bd->bk", users, F.normalize(state, dim=1)) / temperature
    return F.softmax(scores, dim=1)

def rank_kl(target_q, current_q):
    return (target_q * (target_q.clamp_min(1e-12).log() - current_q.clamp_min(1e-12).log())).sum(1, keepdim=True)
```

- [ ] Re-run the focused tests and require PASS.

## Task 2: Add H_G-only rank-calibrated generator training

- [ ] Add failing tests proving `generator_rank_objective` combines V3.2 vector loss with rank KL and that panel labels for generator fitting are drawn only from the supplied `H_G` item set.

```python
def test_generator_rank_objective_is_zero_at_teacher_state_and_positive_away():
    users = torch.eye(2)
    panels = torch.tensor([[0, 1]])
    target = torch.tensor([[1.0, 0.0]])
    assert generator_rank_objective(target, target, users, panels, 0.2, 1.0).item() == pytest.approx(0.0)
    assert generator_rank_objective(torch.tensor([[0.0, 1.0]]), target, users, panels, 0.2, 1.0).item() > 0.0
```

- [ ] Run the test to RED, then implement `RankDistillRunConfig`, `train_rank_calibrated_generator`, and generator history export. Split only `views.generator_item_ids` using the existing deterministic generator split helper. Select by held-out rank KL, then vector loss; freeze the selected generator before the policy stage.

- [ ] Ensure metadata includes H_G train/validation panel hashes and never contains P/train or outer-test rows.

- [ ] Run the focused tests to GREEN.

## Task 3: Replace PPO reward with panel rank gain

- [ ] Add failing tests for an engine subclass that yields positive reward when panel KL decreases, yields zero reward after `END`, applies a course reward only with its configured weight, and refuses missing training panel IDs.

```python
def test_rank_engine_reward_is_incremental_kl_gain_not_embedding_gain():
    engine = _rank_engine(panel_ids=torch.tensor([[0, 1]]), temperature=0.2)
    users = torch.eye(2)
    target = torch.tensor([[1.0, 0.0]])
    reward, _, rank_gain, _ = engine._training_reward(
        torch.tensor([[0.0, 1.0]]), target, target, [torch.empty(0, dtype=torch.long)],
        users, torch.tensor([True]), torch.tensor([0]), torch.tensor([7]), {}
    )
    assert rank_gain.item() > 0.0
    assert reward.item() == pytest.approx(rank_gain.item() - engine.step_penalty - engine.delta_weight * 2 ** 0.5)
```

- [ ] Run the tests to RED. Implement `RankDistilledUSIMEngine(base.CleanUSIMEngine)` with a fixed item-to-panel table. Override `_training_reward` to calculate target/current distributions, rank gain, course constraint, and state-step penalty; do not call `full_positive_score_gain` or item-vector distance reward.

- [ ] Add a training-only `rollout` adapter that accepts target embeddings and item IDs, injects empty compatibility positive-user rows for the unchanged V3.2 rollout contract, and never exposes these inputs on inference.

- [ ] Add target-free item-bank tests. `base.build_clean_item_bank` must see the V3.3 engine with `training=False`, `target_emb=None`, and `positive_user_ids=None`; warm bank rows must equal the frozen teacher rows exactly.

- [ ] Run the focused test file to GREEN.

## Task 4: Add V3.3 orchestration, audit artifacts, and launcher

- [ ] Add failing tests for `run_rank_distill_pipeline` with temporary data. Assert teacher/generator/policy checkpoints, a panel manifest, rank epoch CSVs, V3.3 manifest, and delayed outer-test read.

- [ ] Implement `train_rank_distilled_policy` using `P_train` panels for PPO and `P_val` panels for rank diagnostics. Reuse only V3.2 legal candidate retrieval, course artifacts, evaluation, replay PPO, and split utilities. Store policy state by epoch; an epoch above zero is eligible only when it retains hot validation and has non-negative `P_val` rank gain, otherwise select the identity generator baseline.

- [ ] Implement `run_rank_distill_pipeline` as a fresh copy of V3.2's high-level pipeline with V3.3 trainers. It must call `load_clean_test_inputs` only after the generator and policy checkpoint are fixed.

- [ ] Create the launcher. It must pin seed 2025, reject existing roots, set `USIM_CLEAN_RANDOM_ID_DROPOUT=0`, set `USIM_CLEAN_CANDIDATE_MODE=legal_state_retrieval`, enable the observable course signal, and write only beneath `outputs/ckg_rl_usim_v33_rank_distill` and `checkpoints/ckg_rl_usim_v33_rank_distill`.

- [ ] Run all V3.3 tests, V3.2 regression, compilation, and launcher dry run.

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v33_rank_distill.py tests\test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp\v33_regression
.\py.bat -m py_compile ckg_rl_usim_v33_rank_distill.py
.\run_ckg_rl_usim_v33_rank_distill_seed2025.ps1 -DryRun
```

## Task 5: Smoke and seed-2025 acceptance experiment

- [ ] Run a one-epoch CPU/GPU smoke under a unique `smoke_seed2025` root. Verify the panel manifest, a train rank-gain statistic, a target-free strict-cold rollout, and no outer-test selection.

- [ ] Launch the full fresh seed-2025 process in a hidden PowerShell process with stdout/stderr redirected under `background_logs/ckg_rl_usim_v33_rank_distill_seed2025_<timestamp>/`.

- [ ] On completion, report cold/hot/overall item-macro R@10/N@10, selected epoch, generator held-out KL, P_val rollout KL gain, active-step/end rate, and course/rank reward means. Compare only same-split V3.2 and repaired-old outputs. Do not claim a three-seed result from this run.
