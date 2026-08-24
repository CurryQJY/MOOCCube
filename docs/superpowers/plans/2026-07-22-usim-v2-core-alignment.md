# USIM V2 Core-Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task-by-task. The shared worktree is already dirty, so do not create commits or modify historical result artifacts.

**Goal:** Add an isolated USIM V2 route in which a warm proxy item is trained as pseudo-cold from content-only state toward its held-out behavioural embedding, and the rollout output is the representation consumed by the ranking loss and by cold-item deployment.

**Architecture:** Preserve `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`'s historical defaults so the `.2863` run remains reproducible. A versioned `USIM_ORIGINAL_V2=1` branch activates only for a new launcher. It uses the candidate script's existing deterministic `item_tail` pseudo-cold mask rather than the unsupported `fixed_item_stratified` mode, hides the pseudo-cold ID input, loads a matching full-warm checkpoint, snapshots and freezes its IV user/item space as the teacher, and routes the rollout output into the contrastive ranking loss. The V2 transition is exactly user-driven (`h <- h + lambda * e_u`): the oracle is used in reward and residual candidate retrieval, never injected into the state update. Inference keeps no target oracle and produces one cached cold-item vector. The first experiment deliberately disables all CKG reward/sampling terms; its purpose is to test the content-to-behaviour learning problem before adding CKG terms back.

**Tech Stack:** Python 3, PyTorch, pytest, PowerShell, existing FAST3 static protocol.

---

## Audit amendments before execution

- `fixed_item_stratified` cannot be used by the candidate source: its configuration rejects it without V1, and its `_effective_train_cold_mask` does not implement the mode. V2 therefore uses the already implemented, deterministic `item_tail` mode and a fixed seed.
- The old plan corrected the target but not the transition. The candidate source currently mixes `target_emb` into `_batch_invariant_alignment_grad`; this leaks the oracle into the deployed update and is inconsistent with USIM Eq. (6). V2 must use only the selected user embedding for the state update.
- A launcher's ambient variables are overwritten by parameters inside `run_usim_feedback_fast3_content_delta_static.ps1`. The new launcher must pass `UseCourseSample = $false`, `UseCourseReward = $false`, `RolloutPolicy = "ppo"`, and `PpoLossWeight = 1.0` explicitly, not merely set process variables.
- The pseudo-cold oracle must be stable. V2 requires `USIM_FB_INIT_CKPT_DIR`; after loading the full-warm checkpoint it snapshots and freezes `item_id_emb`, `user_emb`, and `user_proj`. The launcher also sets `AuxHotOnly = $true`, so pseudo-cold rows cannot change the frozen teacher path.
- The plan remains a core repair, not a claim of exact USIM reproduction: explicit termination is deferred. V2 adds a stochastic one-positive estimate of USIM's recommendation reward and residual/positive/random candidate sources, but a learned end action requires a separate actor-head change.

---

### Task 1: Lock down the current failure with RED tests

**Files:**
- Create: `tests/test_usim_v2_core_alignment.py`
- Read only: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`

- [ ] **Step 1: Write a failing pseudo-cold training contract test.**

```python
def test_usim_v2_uses_content_initial_state_and_behavior_oracle_for_pseudocold_rows(monkeypatch):
    model = _v2_model(monkeypatch)
    model.set_pseudo_cold_item_mask(torch.tensor([False, True, False]))
    base = torch.full((2, model.cfg.emb_dim), 0.25, requires_grad=True)
    behaviour = torch.full((2, model.cfg.emb_dim), 2.0, requires_grad=True)
    captured = {}

    model.get_item_vector = lambda *args, **kwargs: (base, behaviour, torch.zeros_like(base))
    def episode(h0, target_emb=None, **kwargs):
        captured["h0"] = h0.detach().clone()
        captured["target"] = target_emb.detach().clone()
        return h0 + 1.0, {"rewards": []}, {"steps": 1}
    model.run_usim_episode = episode
    model.compute_ppo_loss = lambda _: torch.zeros((), requires_grad=True)

    model({"u": torch.tensor([0, 1]), "i": torch.tensor([0, 1])},
          torch.tensor([8.0, 8.0]), torch.full((2,), -1.0))

    assert torch.equal(captured["h0"], base[1:2].detach())
    assert torch.equal(captured["target"], behaviour[1:2].detach())
    assert not torch.equal(captured["h0"], captured["target"])
```

The helper must set `USIM_ORIGINAL_V2=1`, `USIM_USE_PSEUDO_COLD_TRAIN=1`, `USIM_PSEUDO_COLD_MODE=item_tail`, `USIM_TRAIN_FORCE_COLD=1`, and disable unrelated auxiliary losses. Seed the candidate model's existing `_fixed_pseudo_cold_item_mask_cache` with `[False, True, False]`; assert that only item 1 is passed to the episode and hot item 0 retains its base representation.

- [ ] **Step 2: Write a failing ranking-routing test.**

```python
def test_usim_v2_ranking_loss_observes_rollout_output_for_pseudocold_rows(monkeypatch):
    model = _v2_model(monkeypatch)
    model.set_pseudo_cold_item_mask(torch.tensor([False, True, False]))
    base = torch.stack([torch.zeros(model.cfg.emb_dim), torch.ones(model.cfg.emb_dim)])
    behaviour = torch.full_like(base, 2.0)
    captured = {}
    model.get_item_vector = lambda *args, **kwargs: (base, behaviour, torch.zeros_like(base))
    model.run_usim_episode = lambda h0, target_emb=None, **kwargs: (
        h0 + 3.0, {"rewards": []}, {"steps": 1})
    model.compute_ppo_loss = lambda _: torch.zeros((), requires_grad=True)
    original_normalize = F.normalize
    def capture_normalize(value, *args, **kwargs):
        if value.shape == base.shape:
            captured["item_vectors"] = value.detach().clone()
        return original_normalize(value, *args, **kwargs)
    monkeypatch.setattr(F, "normalize", capture_normalize)
    model({"u": torch.tensor([0, 1]), "i": torch.tensor([0, 1])},
          torch.tensor([8.0, 8.0]), torch.full((2,), -1.0))
    assert torch.equal(captured["item_vectors"][0], base[0])
    assert torch.equal(captured["item_vectors"][1], base[1] + 3.0)
```

Use a two-row batch, make `run_usim_episode` return `h0 + 3`, and monkeypatch `torch.nn.functional.cross_entropy` only around the main-loss call. This test must fail if `_apply_refinement_only_to_effective_cold` is bypassed or if `effective_cold` is all false.

- [ ] **Step 3: Write a failing inference-oracle test.**

```python
def test_usim_v2_inference_starts_content_only_and_never_supplies_behavior_target(monkeypatch):
    model = _v2_model(monkeypatch)
    captured = {}
    def episode(h0, target_emb=None, **kwargs):
        captured["h0"] = h0.detach().clone()
        captured["target"] = target_emb
        return h0, {"rewards": []}, {"steps": 0}
    model.run_usim_episode = episode
    model.infer_refined_item_vectors(torch.tensor([1]))
    assert captured["target"] is None
```

Implement the spy as a normal function rather than the compact lambda so it records `target_emb`; assert `force_cold=True` was forwarded to `get_item_vector` and `target_emb is None`.

- [ ] **Step 4: Run the focused test file.**

Run: `./py.bat -m pytest tests/test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp/usim_v2_core_red`

Expected: FAIL because `USIM_ORIGINAL_V2` does not exist and the historical candidate source supplies `z_i_base` as the training target for every row.

### Task 2: Add the V2 implementation branch without changing shared configuration semantics

**Files:**
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`
- Test: `tests/test_usim_v2_core_alignment.py`

- [ ] **Step 1: Add a local V2 environment helper.**

Add beside `_training_episode_target`:

```python
def _original_usim_v2_enabled():
    return os.environ.get("USIM_ORIGINAL_V2", "0") == "1"
```

Do not modify `fast3_delta/config.py`: it is shared and currently carries user changes. The isolated launcher uses a fresh checkpoint root and `ForceFresh = $true`; the candidate manifest already records every `USIM_*` environment variable and its script SHA-256.

- [ ] **Step 2: Add a target/row-routing V2 branch to `forward`.**

```python
if _original_usim_v2_enabled():
    episode_rows = effective_cold.nonzero(as_tuple=False).view(-1)
    final_h = z_i_base
    trajectory, candidate_stats = {"rewards": []}, {"steps": 0}
    if episode_rows.numel() > 0:
        episode_base = z_i_base.index_select(0, episode_rows)
        episode_target = id_e_true.detach().index_select(0, episode_rows)
        episode_final, trajectory, candidate_stats = self.run_usim_episode(
            episode_base, episode_target, user_bank_raw=user_bank_raw,
            item_idx=i.index_select(0, episode_rows),
            target_pop=episode_pop.index_select(0, episode_rows),
            user_seen_items=user_seen_items,
            oracle_user_idx=u.index_select(0, episode_rows),
            oracle_user_emb=z_u_base.detach().index_select(0, episode_rows),
        )
        final_h = z_i_base.index_copy(
            0, episode_rows, self._blend_rl_episode_output(episode_base, episode_final)
        )
else:
    target_emb = _training_episode_target(
        z_i_base,
        getattr(self.cfg, "rollout_policy", "ppo"),
        getattr(self.cfg, "ppo_loss_weight", 1.0),
    )
    episode_final, trajectory, candidate_stats = self.run_usim_episode(
        z_i_base, target_emb, user_bank_raw=user_bank_raw, item_idx=i,
        target_pop=episode_pop, user_seen_items=user_seen_items,
    )
    final_h = _apply_refinement_only_to_effective_cold(
        z_i_base, self._blend_rl_episode_output(z_i_base, episode_final), effective_cold
    )
```

For V2, detach `z_u_base` at pseudo-cold rows before calculating main logits; this prevents the hidden pseudo-cold interaction from training a user vector that later becomes simulator evidence.

- [ ] **Step 3: Run the V2 routing tests.**

Run: `./py.bat -m pytest tests/test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp/usim_v2_routing_green`

Expected: target/row-routing tests pass; the transition test still fails.

### Task 3: Implement V2 user-only transition, training candidates, and reward

**Files:**
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`
- Test: `tests/test_usim_v2_core_alignment.py`

- [ ] **Step 1: Extend `run_usim_episode` with oracle arguments.**

Add optional `oracle_user_idx=None` and `oracle_user_emb=None` arguments. For V2 training (`target_emb is not None`), retrieve residual candidates using `target_emb - current_h`, replace candidate slot 0 with the observed positive user `oracle_user_idx`, and replace slot 1 with a uniformly sampled global user. Keep the existing candidate path untouched for historical or inference calls.

```python
if _original_usim_v2_enabled():
    grad = selected_user
else:
    grad = _batch_invariant_alignment_grad(
        current_h, selected_user, target_emb=target_emb,
        target_alpha=target_alpha,
        reference_batch_size=getattr(self.cfg, "batch_size", current_h.size(0)),
    )
```

For V2 reward, calculate `R_emb = ||h_t-e_i|| - ||h_{t+1}-e_i||`; when `oracle_user_emb` is available, add the one-positive stochastic estimate `R_rec = |h_t·e_u-e_i·e_u| - |h_{t+1}·e_u-e_i·e_u|`; subtract `float(os.environ.get("USIM_ORIGINAL_V2_STEP_PENALTY", "0.01"))`. Do not add CKG reward, collapse penalty, or target-alpha terms on the V2 branch.

- [ ] **Step 2: Ensure the main loss receives the scattered V2 representation.**

```python
final_h = z_i_base.index_copy(0, episode_rows, episode_final)
z_i = F.normalize(final_h, dim=1)
logits = torch.matmul(z_u, z_i.t()) / self.cfg.temp
```

Do not call `_apply_refinement_only_to_effective_cold` again inside the V2 branch; the `index_copy` already enforces the row mask and avoids accidentally discarding `h_T`.

- [ ] **Step 3: Keep inference target-free.**

Do not alter `infer_refined_item_vectors`'s `target_emb=None` call. Add a one-line assertion in the method only if `original_usim_v2` is enabled and an internal caller supplies a non-`None` target; public inference must continue to have no access to `id_e_true`.

- [ ] **Step 4: Run all V2 core tests.**

Run: `./py.bat -m pytest tests/test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp/usim_v2_core_green`

Expected: PASS.

### Task 4: Add a dedicated, reproducible V2 launcher

**Files:**
- Create: `run_usim_original_v2_seed2025.ps1`
- Test: `tests/test_usim_v2_core_alignment.py`

- [ ] **Step 1: Create an isolated launcher with the following locked controls.**

```powershell
$lockedEnvironment = @{
  "USIM_ORIGINAL_V2" = "1"
  "USIM_USE_PSEUDO_COLD_TRAIN" = "1"
  "USIM_PSEUDO_COLD_MODE" = "item_tail"
  "USIM_TRAIN_FORCE_COLD" = "1"
  "USIM_ROLLOUT_POLICY" = "ppo"
  "USIM_PPO_LOSS_WEIGHT" = "1"
  "USIM_USE_COURSE_REWARD" = "0"
  "USIM_USE_COURSE_SAMPLE" = "0"
  "USIM_CKG_RL_V1" = "0"
}
```

Call `run_usim_feedback_fast3_content_delta_static.ps1` with `ScriptPath = "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"`, `PseudoColdMode = "item_tail"`, `UseCourseSample = $false`, `UseCourseReward = $false`, `RolloutPolicy = "ppo"`, `PpoLossWeight = 1.0`, `AuxHotOnly = $true`, `ForceFresh = $true`, `AutoResume = $false`, `OutputRoot = "outputs\\usim_original_v2\\seed2025"`, and `CheckpointRoot = "checkpoints\\usim_original_v2\\seed2025"`. Do not point this launcher at any existing `.2863` output or checkpoint directory.

- [ ] **Step 2: Add a launcher contract test.**

Read the file as text and assert all of: V2 on, pseudo-cold fixed plan, forced cold input, PPO on, nonzero PPO loss, course reward off, fresh isolated roots, candidate source path, and `-DryRun` parsing.

- [ ] **Step 3: Run tests and dry-run.**

Run: `./py.bat -m pytest tests/test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp/usim_v2_launcher_green`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_usim_original_v2_seed2025.ps1 -DryRun`

Expected: tests pass; dry-run creates no training process and prints an isolated V2 configuration.

### Task 5: Instrument and gate the first V2 run

**Files:**
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`
- Test: `tests/test_usim_v2_core_alignment.py`

- [ ] **Step 1: Add V2-only epoch diagnostics.**

Record the mean values below in `candidate_stats` and the static metrics CSV:

```python
candidate_stats["v2_initial_target_l2"] = float(
    (episode_base.detach() - episode_target).norm(dim=1).mean().item()
)
candidate_stats["v2_rollout_delta_l2"] = float(
    (episode_final.detach() - episode_base.detach()).norm(dim=1).mean().item()
)
```

For a batch with no pseudo-cold rows, report both as `0.0`. The test must assert the first metric is positive when mocked `h0` and target differ and the second is positive when the mocked rollout moves.

- [ ] **Step 2: Run focused regression.**

Run: `./py.bat -m pytest tests/test_usim_v2_core_alignment.py tests/test_usim_strict_cold_repair.py -q --basetemp .pytest_tmp/usim_v2_full_green`

Expected: PASS with no mutation of historical manifest files.

- [ ] **Step 3: Run one seed only and inspect the two gates before any 3-seed campaign.**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_usim_original_v2_seed2025.ps1`

Require from the resulting CSV: `EffectiveColdRatio > 0`, `V2InitialTargetL2 > 0`, and `V2RolloutDeltaL2 > 0`. Compare the frozen checkpoint's full-ranking cold, hot, and weighted-overall metrics to the `.2863` historical baseline. Do not update paper tables until this single-seed gate is passed.

---

## Deliberately deferred: exact original candidate set, recommendation reward, and termination

The five tasks repair the demonstrated root cause without pretending that the current simulator already equals USIM. A second, separately approved method revision is needed to reproduce original §3.3–3.4 fully: build a train-only inverse `item -> U_i` index; add residual `TopK(e_i-h_t,e_u)`, sampled true-positive, and random candidate sources; add the recommendation-performance reward; and add an explicit `a_end` policy head with masked PPO trajectories. Those are not safe to bundle into the first correction because they change action-space semantics, critic targets, and the AAAI method claim simultaneously.
