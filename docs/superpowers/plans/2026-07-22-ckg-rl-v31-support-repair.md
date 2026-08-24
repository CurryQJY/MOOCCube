# CKG-RL V3.1 Candidate-Support Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` task-by-task. This repository is deliberately dirty: modify only the files named below and never stage unrelated artifacts.

**Goal:** Repair the V3 train--inference support mismatch and make the CKG signal affect policy decisions, while exposing V3 rollout diagnostics for a single-seed-2025 acceptance run.

**Architecture:** Keep the V3 USIM state transition, frozen teacher, target-free inference, and RecPPO intact. Replace append-and-truncate candidate construction with source quotas (20 candidates: residual/positive/state/random = 6/6/6/2); calculate observable CKG fit on the final candidate IDs; add it directly to user-action logits and replay the exact detached bias. Add V3-only epoch CSV/stdout fields in the static runner, with zeros for non-V3 routes.

**Tech Stack:** Python 3.12, PyTorch 2.8, pytest, existing PowerShell static experiment launcher.

**Out of scope:** Do not alter the strict split, teacher checkpoint, ranking loss, `USIM_V3_STEP_SIZE`, reward coefficients, or V2 route. A displacement trust gate is deferred until the repaired candidate support is measured in the V3.1 acceptance run.

---

## File map

- Modify: `ckg_rl_usim_v3.py` -- candidate quotas, CKG logit bias, replay payload, V3 episode diagnostics, manifest.
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py` -- generic static-runner aggregation and persistence of V3 diagnostics only; V2 behaviour remains numerically unchanged.
- Modify: `tests/test_ckg_rl_usim_v3_core.py` -- behavioural regression tests for the repair.
- Modify: `tests/test_ckg_rl_usim_v3_launcher.py` -- V3.1 launcher and manifest contracts.
- Create: `run_ckg_rl_usim_v31_seed2025.ps1` -- isolated V3.1 seed-2025 launcher and output root.

### Task 1: Lock the observed failures in tests

**Files:**
- Modify: `tests/test_ckg_rl_usim_v3_core.py`

- [ ] **Step 1: Write a failing explicit-quota test.**

```python
def _v3_model(monkeypatch, *, steps=2, n_users=4, candidates=3):
    monkeypatch.setenv("USIM_ORIGINAL_V2", "1")
    monkeypatch.setenv("USIM_USE_PSEUDO_COLD_TRAIN", "1")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MODE", "item_tail")
    monkeypatch.setenv("USIM_PSEUDO_COLD_RATIO", "0.50")
    monkeypatch.setenv("USIM_PSEUDO_COLD_MIN_POP", "1")
    monkeypatch.setenv("USIM_TRAIN_FORCE_COLD", "1")
    monkeypatch.setenv("USIM_V3_STEP_SIZE", "0.05")
    monkeypatch.setenv("USIM_V3_STEP_PENALTY", "0.01")
    monkeypatch.setenv("USIM_V3_REPLAY_CAPACITY", "16")
    monkeypatch.setenv("USIM_V3_REPLAY_BATCH_SIZE", "2")
    cfg = Fast3Config(n_users=n_users, n_items=3, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.usim_steps = int(steps)
    cfg.n_candidates = int(candidates)
    cfg.retrieve_top_m = int(candidates)
    cfg.use_course_reward = False
    model = CKGRLV3USIM(cfg, torch.zeros((3, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    return model

def test_v31_training_candidates_reserve_all_four_sources(monkeypatch):
    model = _v3_model(monkeypatch, n_users=30, candidates=20)
    model.train()
    state_top = torch.tensor([list(range(10, 30))])
    residual_top = torch.tensor([list(range(0, 20))])

    def fake_topk(query, *_args, **_kwargs):
        return state_top if torch.allclose(query, torch.zeros_like(query)) else residual_top

    monkeypatch.setattr(model, "_v3_topk_user_ids", fake_topk)
    _, candidate_ids, _ = model._v3_build_candidates(
        torch.zeros((1, model.cfg.emb_dim)), torch.randn(30, model.cfg.emb_dim),
        training=True, target_emb=torch.ones((1, model.cfg.emb_dim)),
        positive_user_ids=[torch.tensor([20, 21, 22, 23, 24, 25])],
        item_idx=torch.tensor([1]), target_pop=torch.tensor([0.0]), user_seen_items={},
    )

    chosen = set(candidate_ids[0].tolist())
    assert set(range(0, 6)).issubset(chosen)
    assert set(range(20, 26)).issubset(chosen)
    assert set(range(10, 16)).issubset(chosen)
    assert len(chosen) == 20
```

- [ ] **Step 2: Run the test and verify the current V3 fails because residual Top-20 consumes every slot.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_core.py::test_v31_training_candidates_reserve_all_four_sources -q --basetemp .pytest_tmp/v31_quota_red`

Expected: FAIL; the positive and state assertions fail under the append-and-truncate implementation.

- [ ] **Step 3: Write a failing direct-logit test.**

```python
def test_v31_actor_bias_changes_the_user_action_without_reordering():
    actor = EndAwareRecActorCritic(embedding_dim=2, hidden_dim=4)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.end_head.bias.fill_(-3.0)

    action, *_ = actor.action_value(
        torch.zeros((1, 2)), torch.ones((1, 1)), torch.zeros((1, 3, 2)),
        candidate_logit_bias=torch.tensor([[0.0, 1.0, 0.0]]), deterministic=True,
    )

    assert action.tolist() == [1]
```

- [ ] **Step 4: Run the test and verify the current actor rejects the new bias argument.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_core.py::test_v31_actor_bias_changes_the_user_action_without_reordering -q --basetemp .pytest_tmp/v31_logit_red`

Expected: FAIL with an unexpected `candidate_logit_bias` keyword argument.

- [ ] **Step 5: Replace the legacy inference reordering test with a target-free CKG-bias contract.**

```python
def test_v31_inference_passes_observable_ckg_bias_to_actor(monkeypatch):
    model = _v3_model(monkeypatch, steps=1)
    model.eval()
    model.cfg.feedback_course_sample_beta = 0.2
    captured = {}

    monkeypatch.setattr(
        model, "_compute_candidate_course_fit",
        lambda candidate_ids, **kwargs: torch.tensor([[0.0, 2.0, -2.0]]),
    )
    def capture_actor(state, remaining, candidates, *, candidate_logit_bias=None, **kwargs):
        captured["bias"] = candidate_logit_bias.detach().clone()
        return torch.full((1,), candidates.size(1), dtype=torch.long), torch.zeros(1), torch.zeros((1, 1)), torch.zeros(1)
    monkeypatch.setattr(model.agent, "action_value", capture_actor)

    model.infer_refined_item_vectors(torch.tensor([0]), force_cold=True,
                                     user_bank_raw=torch.zeros((4, model.cfg.emb_dim)),
                                     user_seen_items={0: {1}})
    assert torch.allclose(captured["bias"], torch.tensor([[0.0, 0.2, -0.2]]))
```

- [ ] **Step 6: Add a replay contract requiring the detached per-candidate bias and run it red.**

```python
def test_v31_replay_uses_the_same_detached_candidate_bias_for_ppo(monkeypatch):
    model = _v3_model(monkeypatch)
    model.train()
    trajectory = {
        "states": [torch.zeros((2, model.cfg.emb_dim))],
        "next_states": [torch.zeros((2, model.cfg.emb_dim))],
        "candidate_ids": [torch.tensor([[0, 1, 2], [1, 2, 3]])],
        "candidate_logit_bias": [torch.tensor([[0.0, 0.4, -0.4], [0.1, 0.0, -0.1]])],
        "actions": [torch.tensor([1, 0])],
        "rewards": [torch.tensor([[0.2], [0.0]])],
        "done": [torch.tensor([False, True])],
        "old_log_probs": [torch.tensor([-0.4, -0.6])],
        "remaining_steps": [torch.tensor([[2.0], [1.0]])],
        "next_remaining_steps": [torch.tensor([[1.0], [0.0]])],
        "terminal_states": [torch.zeros((2, model.cfg.emb_dim))],
        "terminal_remaining_steps": [torch.tensor([[1.0], [0.0]])],
    }
    observed = []
    def capture(state, remaining, candidates, *, candidate_logit_bias, action=None, **_kwargs):
        observed.append(candidate_logit_bias.detach().clone())
        scalar = next(model.parameters()).reshape(-1)[0]
        resolved_action = torch.zeros(state.size(0), dtype=torch.long) if action is None else action
        return resolved_action, scalar.expand(state.size(0)), scalar.expand(state.size(0), 1), scalar.expand(state.size(0))
    monkeypatch.setattr(model.agent, "action_value", capture)
    model.compute_ppo_loss(trajectory)
    expected_rows = {tuple(row.tolist()) for row in trajectory["candidate_logit_bias"][0]}
    assert any({tuple(row.tolist()) for row in value} == expected_rows for value in observed)
```

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_core.py -q --basetemp .pytest_tmp/v31_core_red`

Expected: the new quota, actor-bias, inference, and replay tests fail; pre-existing tests remain meaningful.

### Task 2: Implement quota-preserving candidate support and policy-visible CKG fit

**Files:**
- Modify: `ckg_rl_usim_v3.py:98-125, 131-177, 296-358, 422-600`
- Test: `tests/test_ckg_rl_usim_v3_core.py`

- [ ] **Step 1: Add `candidate_logit_bias: torch.Tensor | None = None` to `EndAwareRecActorCritic.action_value`.**

```python
if candidate_logit_bias is None:
    candidate_logit_bias = user_logits.new_zeros(user_logits.shape)
else:
    candidate_logit_bias = torch.as_tensor(
        candidate_logit_bias, dtype=user_logits.dtype, device=user_logits.device
    )
    if candidate_logit_bias.shape != user_logits.shape:
        raise ValueError("candidate_logit_bias must be [batch, candidate_count]")
user_logits = user_logits + candidate_logit_bias
```

- [ ] **Step 2: Implement quota helpers with the exact default 20-way allocation.**

```python
def _v3_training_candidate_quotas(self) -> tuple[int, int, int, int]:
    count = self.v3_candidate_count
    weights = (0.30, 0.30, 0.30, 0.10)
    quotas = [int(count * weight) for weight in weights]
    for index in range(count - sum(quotas)):
        quotas[index % len(quotas)] += 1
    return tuple(quotas)  # 20 -> (6, 6, 6, 2)

@staticmethod
def _v3_append_unique(dst, seen, source, quota, n_users):
    for candidate in source:
        value = int(candidate)
        if 0 <= value < n_users and value not in seen:
            dst.append(value); seen.add(value)
            if len(dst) >= quota:
                break
```

Use one source-local counter (not `len(dst)`) and use candidates in this order: residual, observed positives, state retrieval, random. At inference, use the full state-retrieval Top-K only. If deduplication leaves a training row short, fill from state retrieval, then residual retrieval, then bounded random draws; pad only when the user bank itself is smaller than the requested count. Generate the random source once as a CPU `(batch_size, 8 * candidate_count)` tensor and reuse each row's draw/fallback start, never call `torch.randperm(n_users)` or launch a random draw per row. Record the source count in a local `source_counts` dictionary, set `self._v3_last_candidate_stats` immediately before returning, and read that dictionary in the same rollout step; do not infer source membership later from IDs.

- [ ] **Step 3: Compute an observable CKG bias without calling `_apply_course_sampling_bias`.**

```python
def _v3_course_logit_bias(self, candidate_ids, *, item_idx, target_pop, user_seen_items):
    beta = float(getattr(self.cfg, "feedback_course_sample_beta", 0.0))
    if beta <= 0.0 or item_idx is None:
        return candidate_ids.new_zeros(candidate_ids.shape, dtype=torch.float32)
    fit = self._compute_candidate_course_fit(candidate_ids, item_idx=item_idx,
                                             target_pop=target_pop, user_seen_items=user_seen_items)
    fit = torch.nan_to_num(fit, nan=0.0, posinf=0.0, neginf=0.0)
    return beta * fit / fit.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
```

The result must keep candidate order unchanged and must use only `candidate_ids`, course metadata, and observed user history. Do not pass target embeddings or positive users to it at inference.

- [ ] **Step 4: Thread the bias through rollout and replay.**

Pass `candidate_logit_bias` to every online and target-agent `action_value` call. Add it to `V3ReplayBuffer._REQUIRED_KEYS`, append it from the trajectory, and use the stored detached tensor when recomputing PPO log probabilities. The critic is allowed to ignore the value numerically, but receiving the same shaped input keeps the actor/critic call contract stable.

- [ ] **Step 5: Emit episode-level diagnostics from the V3 model.**

Return the existing `v3_end_rate`, `v3_active_steps`, `v3_embedding_reward`, `v3_recommendation_reward`, `v3_course_reward`, and `v3_rollout_delta_l2`, plus `v3_course_logit_bias_abs`, `v3_train_residual_share`, `v3_train_positive_share`, `v3_train_state_share`, and `v3_train_random_share`. Shares are measured over non-padding training candidates only; inference must report zero for the train-only source shares.

- [ ] **Step 6: Run focused tests green.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_core.py -q --basetemp .pytest_tmp/v31_core_green`

Expected: PASS; the actor action changes from the bias, all four sources retain their quota when available, and PPO receives detached rollout bias.

### Task 3: Persist V3 diagnostics and create an isolated V3.1 launcher

**Files:**
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py:3670-4200`
- Modify: `ckg_rl_usim_v3.py:627-638`
- Create: `run_ckg_rl_usim_v31_seed2025.ps1`
- Modify: `tests/test_ckg_rl_usim_v3_launcher.py`

- [ ] **Step 1: Add V3 keys to the static history schema.**

Add these zero-default keys to `static_diag_keys`, the per-epoch accumulators, the `cand_info.get(...)` collection, and `epoch_diag`:

```python
"V3EndRate", "V3ActiveSteps", "V3EmbeddingReward", "V3RecommendationReward",
"V3CourseReward", "V3RolloutDeltaL2", "V3CourseLogitBiasAbs",
"V3TrainResidualShare", "V3TrainPositiveShare", "V3TrainStateShare",
"V3TrainRandomShare",
```

Print them in a separate `V3[...]` stdout block. Do not rename or change the existing `V2[...]` block, so historical V2 CSVs remain schema-compatible.

- [ ] **Step 2: Add V3.1 manifest provenance.**

Include `engine_revision: "v3.1"`, the resolved training quota tuple, and the CKG logit-bias mechanism in `v3_engine_manifest.json`.

- [ ] **Step 3: Create `run_ckg_rl_usim_v31_seed2025.ps1` by copying the V3 acceptance launcher with only these intentional changes.**

```powershell
$outputRelative = Join-Path "outputs\ckg_rl_usim_v31" $runName
$checkpointRelative = Join-Path "checkpoints\ckg_rl_usim_v31" $runName
"USIM_V3_ENGINE_REVISION" = "v3.1"
"USIM_V3_CANDIDATES" = [string]$Candidates
```

Keep seed pinning, frozen teacher checkpoint, strict item-cold protocol, pseudo-cold ratio, CKG reward, prerequisite auxiliary loss, and all other hyperparameters unchanged. The dry-run contract must say `CKG-RL V3.1` and include `quota=6/6/6/2` for the default 20 candidates.

- [ ] **Step 4: Run launcher/manifest tests.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_launcher.py -q --basetemp .pytest_tmp/v31_launcher_green`

Expected: PASS; V3.1 cannot overwrite V3 output/checkpoint roots and declares the diagnostic contract.

### Task 4: Verify before the acceptance experiment

**Files:**
- Test: `tests/test_ckg_rl_usim_v3_core.py`
- Test: `tests/test_ckg_rl_usim_v3_launcher.py`

- [ ] **Step 1: Run the full affected regression set.**

Run:

```powershell
.\py.bat -m pytest `
  tests\test_ckg_rl_usim_v3_core.py `
  tests\test_ckg_rl_usim_v3_launcher.py `
  tests\test_usim_v2_core_alignment.py `
  tests\test_coursefit_pseudocold_minimal_repair.py `
  tests\test_learner_guided_cold_refinement.py `
  tests\test_main_checkpoint_actor_inference_ab.py `
  tests\test_legacy_ppo_eval_probe.py -q `
  --basetemp .pytest_tmp\v31_full_regression
```

Expected: all tests pass.

- [ ] **Step 2: Run the V3.1 launcher dry run.**

Run: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v31_seed2025.ps1 -DryRun`

Expected: the contract reports V3.1, seed 2025, `quota=6/6/6/2`, no overwrite, frozen teacher, and CKG/prerequisite components enabled.

- [ ] **Step 3: Run a one-batch CPU trace using the existing real-CPU core test.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v3_core.py::test_v3_real_cpu_parent_forward_and_recppo_complete_one_pseudocold_batch -q --basetemp .pytest_tmp/v31_cpu_trace`

Expected: a finite loss, non-empty replay, and all V3 diagnostic fields are finite.

### Task 5: Seed-2025 acceptance run and decision gate

**Files:**
- Create at runtime: `outputs/ckg_rl_usim_v31/seed2025/strict_item_cold_balanced_thr1_seed_2025/`
- Create at runtime: `checkpoints/ckg_rl_usim_v31/seed2025/strict_item_cold_balanced_thr1_seed_2025/`

- [ ] **Step 1: Launch only the repaired seed 2025 run.**

Run: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_rl_usim_v31_seed2025.ps1`

- [ ] **Step 2: Inspect the live stdout and epoch CSV.**

Confirm the new `V3[...]` values are non-zero where expected, source shares are close to 0.30/0.30/0.30/0.10 before unavoidable deduplication, and CKG logit-bias magnitude is non-zero. Confirm inference remains target-free through the core contract tests.

- [ ] **Step 3: Apply the acceptance gate before any 2026/2027 run.**

Compare item-macro R@10/N@10 on the identical split against V3 seed 2025 (`0.2367106 / 0.1604255` cold; `0.1446015 / 0.0810917` hot; `0.1543425 / 0.0894816` overall). Proceed to multi-seed only if cold R@10 and N@10 both improve over V3 and hot/overall do not fall further. If candidate support is corrected but inference displacement remains materially above training displacement, stop before multiseed and design a separate V3.2 trust-gate experiment.
