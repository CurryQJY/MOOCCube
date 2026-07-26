# CKG-RL V3 USIM Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task-by-task. The current worktree is intentionally dirty and contains the V2 teacher repair, so preserve all existing files and do not commit historical artifacts.

**Goal:** Build an isolated CKG-RL V3 route whose simulator implements explicit USIM state/action/termination/replay semantics while retaining the current strict-cold MOOCCube and course-knowledge method, then launch the seed-2025 acceptance experiment.

**Architecture:** `ckg_rl_usim_v3.py` wraps the current candidate implementation instead of copying its 250 KB static training/evaluation code. It replaces the model factory with `CKGRLV3USIM`, a subclass that inherits content initialization, course artifacts, ranking loss, checkpoint loading, and evaluation while overriding the V2 rollout and PPO paths. A learned end-aware actor, bounded replay buffer, full-positive recommendation reward, and target-free inference episode live in the wrapper. The static runner invokes the wrapper through its supported delegate marker.

**Tech Stack:** Python 3.12, PyTorch 2.8, pytest, existing PowerShell static runner, MOOCCube strict item-cold split.

---

### Task 1: Add a failing V3 core-contract test file

**Files:**
- Create: `tests/test_ckg_rl_usim_v3_core.py`
- Read only: `USIM-main/cold_model/USIM.py`, `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`

- [ ] **Step 1: Write a failing test for an active user transition.**

```python
def test_v3_active_user_action_applies_exact_embedding_step(monkeypatch):
    model = _v3_model(monkeypatch, steps=2)
    model.train()
    _install_teacher(model)
    model._v3_train_item_users = {1: torch.tensor([0, 1])}
    _force_actions(model, [0, 2])  # candidate 0, then a_end

    h0 = torch.zeros((1, model.cfg.emb_dim))
    users = _teacher_users(model)
    final, trajectory, stats = model.run_usim_episode(
        h0, target_emb=torch.ones_like(h0), user_bank_raw=users,
        item_idx=torch.tensor([1]), target_pop=torch.tensor([5.0]),
        user_seen_items={},
    )

    assert torch.allclose(final, 0.05 * users[0:1])
    assert trajectory["done"][0].tolist() == [False]
    assert stats["v3_end_rate"] == pytest.approx(1.0)
```

- [ ] **Step 2: Write a failing test for terminal freezing and zero later reward.**

```python
def test_v3_end_action_freezes_state_and_masks_later_rewards(monkeypatch):
    model = _v3_model(monkeypatch, steps=3)
    model.train()
    _install_teacher(model)
    model._v3_train_item_users = {1: torch.tensor([0, 1])}
    _force_actions(model, [model.v3_candidate_count, 0, 0])

    h0 = torch.randn(1, model.cfg.emb_dim)
    final, trajectory, _ = model.run_usim_episode(
        h0, target_emb=torch.zeros_like(h0), user_bank_raw=_teacher_users(model),
        item_idx=torch.tensor([1]), target_pop=torch.tensor([5.0]), user_seen_items={},
    )

    assert torch.equal(final, h0)
    assert all(torch.equal(reward, torch.zeros_like(reward)) for reward in trajectory["rewards"][1:])
```

- [ ] **Step 3: Write a failing test for full-positive recommendation reward.**

```python
def test_v3_recommendation_reward_averages_all_course_users(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    target = torch.tensor([[2.0, 0.0]])
    before = torch.tensor([[0.0, 0.0]])
    after = torch.tensor([[1.0, 0.0]])
    users = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    model._v3_train_item_users = {1: torch.tensor([0, 1])}

    reward = model._v3_recommendation_reward(
        before, after, target, torch.tensor([1]), users,
    )

    assert reward.item() == pytest.approx(2.0)
```

- [ ] **Step 4: Write a failing test for target-free inference.**

```python
def test_v3_inference_never_passes_teacher_target_or_training_users(monkeypatch):
    model = _v3_model(monkeypatch, steps=2)
    captured = {}
    original = model._v3_build_candidates

    def capture(*args, **kwargs):
        captured["training"] = kwargs["training"]
        captured["target_emb"] = kwargs.get("target_emb")
        captured["positive_users"] = kwargs.get("positive_user_ids")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_v3_build_candidates", capture)
    model.infer_refined_item_vectors(torch.tensor([0]), force_cold=True)

    assert captured == {"training": False, "target_emb": None, "positive_users": None}
```

- [ ] **Step 5: Run the new tests and verify they fail because V3 is missing.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_core.py -q --basetemp .pytest_tmp\v3_core_red
```

Expected: collection failure for `ckg_rl_usim_v3` or missing V3 model methods, not an unrelated test failure.

### Task 2: Implement the isolated V3 USIM core

**Files:**
- Create: `ckg_rl_usim_v3.py`
- Modify: `tests/test_ckg_rl_usim_v3_core.py`

- [ ] **Step 1: Add a delegate wrapper and an end-aware actor.**

```python
USIM_STATIC_DELEGATE_ENTRYPOINT = True

class EndAwareRecActorCritic(nn.Module):
    # `action_value` concatenates a normalized remaining-step scalar to state,
    # produces one logit per candidate user and one final `a_end` logit, then
    # returns `(action, log_prob, value, entropy)` from a categorical policy.

class CKGRLV3USIM(base.Fast3FeedbackUSIM):
    def __init__(self, config, content_emb):
        super().__init__(config, content_emb)
        self.agent = EndAwareRecActorCritic(config.emb_dim)
        self.v3_replay = V3ReplayBuffer(capacity=_v3_replay_capacity())
        self.v3_target_agent = copy.deepcopy(self.agent).eval()
```

`a_end` is the final action index, so no fake user ID can enter user-history or course-reward code.

- [ ] **Step 2: Implement train-only user-set indexing and candidate construction.**

```python
def _v3_train_users_for_items(self, item_idx, user_seen_items):
    # Lazily invert the training-only user->seen-items map once per model.
    # Return CPU/GPU LongTensor lists for each requested pseudo-cold item.

def _v3_build_candidates(self, current_h, user_bank, *, training, target_emb,
                         positive_user_ids, item_idx, target_pop, user_seen_items):
    # Training: residual top-k + positive users + state top-k + random users.
    # Inference: state top-k + random users only.
    # Deduplicate, pad to fixed count, then apply existing observable CKG bias.
    return candidate_emb, candidate_ids, fit_score
```

Candidate IDs must always be real user indices. The terminal action is represented only in the actor's final logit.

- [ ] **Step 3: Implement the USIM episode and all-positive reward.**

```python
def run_usim_episode(self, init_item_emb, target_emb=None, user_bank_raw=None,
                     item_idx=None, target_pop=None, user_seen_items=None, **_):
    training = bool(self.training and target_emb is not None)
    # Build `done`; use target data and observed positives only when training.
    # Apply active-user or a_end transition; sum R_emb + R_rec - p + optional CKG.
    # Record detached replay fields and V3 diagnostics.
    return final_h, trajectory, stats
```

`_v3_recommendation_reward` must calculate the per-course mean score error across each course's whole observed training-user set in chunks, not by consuming the current batch label.

- [ ] **Step 4: Implement replayed RecPPO.**

```python
def compute_ppo_loss(self, trajectory):
    self.v3_replay.append(trajectory)
    batch = self.v3_replay.sample(_v3_replay_batch_size(), self.device)
    td_target = reward + gamma * (1.0 - done) * target_value(next_state)
    actor_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
    critic_loss = F.mse_loss(value, td_target) + terminal_value.pow(2).mean()
    self._v3_sync_target_on_next_forward = True
    return actor_loss + value_weight * critic_loss - entropy_weight * entropy.mean()
```

`old_log_prob`, target values, and TD advantages must be detached. Synchronize the target actor/critic at the next forward call, after the outer optimizer has applied the previous update.

- [ ] **Step 5: Implement wrapper entry points.**

```python
_BASE_STATIC_EXPERIMENT = base.run_static_experiment

def run_static_experiment():
    previous = base.Fast3FeedbackUSIM
    base.Fast3FeedbackUSIM = CKGRLV3USIM
    try:
        return _BASE_STATIC_EXPERIMENT()
    finally:
        base.Fast3FeedbackUSIM = previous

def main():
    previous = base.run_static_experiment
    base.run_static_experiment = run_static_experiment
    try:
        base.main()
    finally:
        base.run_static_experiment = previous
```

- [ ] **Step 6: Run the V3 core test file.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_core.py -q --basetemp .pytest_tmp\v3_core_green
```

Expected: all V3 core tests pass.

### Task 3: Add integration and launcher contracts

**Files:**
- Create: `run_ckg_rl_usim_v3_seed2025.ps1`
- Create: `tests/test_ckg_rl_usim_v3_launcher.py`
- Modify: `ckg_rl_usim_v3.py`

- [ ] **Step 1: Write a failing launcher contract test.**

```python
def test_v3_launcher_is_seed2025_isolated_and_enables_ckg_terms():
    text = Path("run_ckg_rl_usim_v3_seed2025.ps1").read_text(encoding="utf-8")
    assert 'ScriptPath = "ckg_rl_usim_v3.py"' in text
    assert 'outputs\\ckg_rl_usim_v3' in text
    assert 'checkpoints\\ckg_rl_usim_v3' in text
    assert '"USIM_ORIGINAL_V2" = "1"' in text
    assert 'UseCourseReward = $true' in text
    assert 'UsePrereqAux = $true' in text
    assert 'UseCourseSample = $true' in text
```

- [ ] **Step 2: Run the launcher test and verify it fails.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_launcher.py -q --basetemp .pytest_tmp\v3_launcher_red
```

Expected: failure because the launcher does not yet exist.

- [ ] **Step 3: Create the seed-2025 launcher.**

The launcher must:

- reject every seed except 2025;
- refuse to overwrite `outputs\\ckg_rl_usim_v3\\seed2025` or the matching checkpoint root;
- load the same `checkpoints\\recovery_validation\\main_table_51ea12fc_candidate\\strict_item_cold_balanced_thr1_seed_2025` teacher checkpoint used by V2;
- run `ckg_rl_usim_v3.py` through `run_usim_feedback_fast3_content_delta_static.ps1`;
- set `USIM_ORIGINAL_V2=1` only to reuse the base teacher/pseudo-cold branch, and set explicit `USIM_V3_CORE=1`, replay/candidate parameters, and a V3 inference seed;
- use strict item-cold balanced, train-only history, deterministic `item_tail` pseudo-cold masking, 40 requested epochs, patience 6, and a new V3-only output/checkpoint root;
- enable course candidate sampling, course reward, and prerequisite auxiliary loss for the intended CKG method; and
- support `-DryRun`, which validates paths and emits the resolved run configuration without launching Python.

- [ ] **Step 4: Add a V3 manifest assertion.**

In `ckg_rl_usim_v3.py`, patch the delegated static manifest payload after its creation or write `v3_engine_manifest.json` next to it with `route="ckg_rl_usim_v3"`, core hyperparameters, teacher path, and an explicit `inference_oracle_access=false` field.

- [ ] **Step 5: Run launcher tests and DryRun.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_launcher.py -q --basetemp .pytest_tmp\v3_launcher_green
.\run_ckg_rl_usim_v3_seed2025.ps1 -DryRun
```

Expected: launcher test passes and DryRun exits successfully without creating an experiment output directory.

### Task 4: Validate the combined V3 route before GPU training

**Files:**
- Modify: `tests/test_ckg_rl_usim_v3_core.py`
- Read only: `tests/test_usim_v2_core_alignment.py`, `tests/test_coursefit_pseudocold_minimal_repair.py`

- [ ] **Step 1: Add a failing integration test for pseudo-cold routing.**

```python
def test_v3_parent_forward_routes_only_pseudocold_rows_through_v3_episode(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.train()
    _install_teacher(model)
    captured = {}
    model.run_usim_episode = _capture_episode(captured)

    model({"u": torch.tensor([0, 1]), "i": torch.tensor([0, 2])},
          torch.tensor([8.0, 8.0]), torch.full((2,), -1.0), user_seen_items={})

    assert captured["item_idx"].tolist() == [2]
    assert captured["target_emb"] is not None
```

- [ ] **Step 2: Run the test and verify it fails for the intended missing contract.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_core.py::test_v3_parent_forward_routes_only_pseudocold_rows_through_v3_episode -q --basetemp .pytest_tmp\v3_route_red
```

Expected: failing assertion until the V3 wrapper correctly enters the base pseudo-cold branch.

- [ ] **Step 3: Implement the minimum routing correction and re-run focused tests.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_core.py tests\test_ckg_rl_usim_v3_launcher.py tests\test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp\v3_route_green
.\py.bat -m py_compile ckg_rl_usim_v3.py
```

Expected: all selected tests pass and the wrapper compiles.

### Task 5: Execute the seed-2025 acceptance experiment

**Files:**
- Create at runtime: `outputs/ckg_rl_usim_v3/seed2025/strict_item_cold_balanced_thr1_seed_2025/`
- Create at runtime: `checkpoints/ckg_rl_usim_v3/seed2025/strict_item_cold_balanced_thr1_seed_2025/`
- Create at runtime: `background_logs/ckg_rl_usim_v3_seed2025_<timestamp>/`

- [ ] **Step 1: Run the complete regression gate.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v3_core.py tests\test_ckg_rl_usim_v3_launcher.py tests\test_usim_v2_core_alignment.py tests\test_coursefit_pseudocold_minimal_repair.py tests\test_learner_guided_cold_refinement.py tests\test_main_checkpoint_actor_inference_ab.py tests\test_legacy_ppo_eval_probe.py -q --basetemp .pytest_tmp\v3_full_regression
```

Expected: all selected tests pass.

- [ ] **Step 2: Launch the isolated experiment in a background PowerShell process.**

Run:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = "background_logs\\ckg_rl_usim_v3_seed2025_$stamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','.\\run_ckg_rl_usim_v3_seed2025.ps1') -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log') -PassThru
```

Expected: the process starts, logs show the V3 route, frozen teacher checkpoint, CKG components enabled, and no output-root collision.

- [ ] **Step 3: Monitor until a terminal checkpoint/result is produced.**

Read the current `stdout.log` at bounded intervals under 60 seconds. Do not restart a live process. If it fails, preserve its run directory and diagnose from the first traceback before changing code.

- [ ] **Step 4: Compare results against same-split old CKG-RL and V2.**

Read:

```text
outputs/content_delta_pop5/course_ablation_e60_3seed/full/strict_item_cold_balanced_thr1_seed_2025/final_fullrank_usim_feedback_fast3_content_delta_static.csv
outputs/usim_original_v2/seed2025/strict_item_cold_balanced_thr1_seed_2025/final_fullrank_usim_feedback_fast3_content_delta_static.csv
outputs/ckg_rl_usim_v3/seed2025/strict_item_cold_balanced_thr1_seed_2025/final_fullrank_usim_feedback_fast3_content_delta_static.csv
```

Report cold, hot, and overall item-macro R@10/N@10 plus V3 termination and reward diagnostics. Do not claim a three-seed improvement from this acceptance run.
