# Cold-Consistent Knowledge Calibration (C3K) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, non-RL C3K training and full-ranking inference path with aligned cold evidence, deterministic pseudo-cold training, learner-conditioned calibration, timing traces, and a frozen three-seed campaign.

**Architecture:** New C3K modules import the existing FAST3 data, static split, course-artifact, and checkpoint helpers read-only. They do not modify or call legacy simulator/PPO/refinement paths. A single C3K score is used in sampled training and blocked full-catalog evaluation. A fixed train-popularity-only item plan masks the same ID branch as strict-cold inference and removes selected course columns from training history.

**Tech Stack:** Python, PyTorch, pandas, pytest, existing fast3_delta utilities, PowerShell.

---

## File structure

- Create: `fast3_delta/c3k_model.py` — model, structural score, paired loss, and training diagnostics.
- Create: `fast3_delta/c3k_eval.py` — item-bank builder, blocked full ranking, and inference timing.
- Create: `c3k_static.py` — independent C3K runner and validation selector.
- Create: `run_c3k_3seed.ps1` — frozen serial seeds 2025/2026/2027 launcher.
- Create: `tests/test_c3k_closed_loop.py` — behavioral, parity, and launcher tests.
- Create: this plan.

Existing legacy Python scripts, generic launchers, paper files, historical outputs, and checkpoints will not be modified.

### Task 1: Test the C3K evidence boundary before production code

**Files:**
- Create: `tests/test_c3k_closed_loop.py`
- Create: `fast3_delta/c3k_model.py`

- [ ] **Step 1: Write failing deterministic-plan and view tests.**

```python
def test_fixed_plan_is_reproducible_and_excludes_strict_cold():
    plan = build_pseudocold_plan(torch.tensor([0., 5., 7., 11.]), ratio=0.5,
                                 min_popularity=1, cold_threshold=1, seed=2025)
    assert not plan.selected_mask[0]
    assert int(plan.selected_mask.sum()) == 2

def test_paired_view_bypasses_random_id_dropout(model):
    model.cfg.dropout_prob = 1.0
    full, masked = model.paired_item_views(torch.tensor([0, 1]), llm(), torch.tensor([False, True]))
    assert torch.allclose(full[0], masked[0])
    assert not torch.allclose(full[1], masked[1])
```

- [ ] **Step 2: Run RED.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Expected: import failure because `fast3_delta.c3k_model` does not exist.

- [ ] **Step 3: Implement the exact view interface.**

```python
def item_view(self, item_ids, llm_scores, cold_style_mask):
    return self.get_item_vector(item_ids, llm_scores, force_cold=cold_style_mask,
                                disable_id_dropout=True)[0]

def paired_item_views(self, item_ids, llm_scores, pseudo_mask):
    full = self.item_view(item_ids, llm_scores, torch.zeros_like(pseudo_mask))
    masked = self.item_view(item_ids, llm_scores, pseudo_mask)
    return full, torch.where(pseudo_mask[:, None], masked, full)
```

`cold_style_mask` must be Boolean and item-level. The C3K implementation must not use `torch.rand`, `dropout_prob`, simulator functions, PPO loss, or course-global refinement.

- [ ] **Step 4: Run GREEN and commit.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Commit: `git add fast3_delta/c3k_model.py tests/test_c3k_closed_loop.py; git commit -m "feat: add C3K evidence boundary"`

### Task 2: Test and implement the one shared score

**Files:**
- Modify: `fast3_delta/c3k_model.py`
- Modify: `tests/test_c3k_closed_loop.py`

- [ ] **Step 1: Write failing target-removal, sign, and pair/catalog-parity tests.**

```python
def test_gate_signs(model):
    rho = model.knowledge_coefficients(torch.randn(3, 128), torch.randn(3, 128), torch.rand(3, 4))
    assert torch.all(rho[:, 0] >= 0)
    assert torch.all(rho[:, 1:] <= 0)

def test_pair_equals_catalog_for_one_candidate(model):
    bank = model.build_item_bank(torch.tensor([False, False, True, False]))
    pair = model.score_pairs(torch.tensor([0]), torch.tensor([2]), bank[2:3], {0: {1}})
    catalog = model.score_catalog(torch.tensor([0]), bank, {0: {1}}, item_block=2)
    assert torch.allclose(pair.view(-1), catalog[:, 2].view(-1), atol=1e-6)
```

- [ ] **Step 2: Run RED.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Expected: missing C3K score methods.

- [ ] **Step 3: Implement score and structural features.**

```python
s(u, c) = dot(normalize(user_proj(user_id)), normalize(item_view(c))) / temperature
          + residual_mlp(user_embedding, item_embedding)
          + sum(sign_constrained_gate(user_embedding, item_embedding, k(u, c)) * k(u, c))
```

`k` is exactly `[concept_continuity, prerequisite_gap, difficulty_gap, redundancy]` in [0, 1], obtained only from passed train-history and course metadata. Remove every row target before training pairs. The coefficients are `[a_c, -a_p, -a_d, -a_r]`, where `a = 0.20 * sigmoid(raw)`.

- [ ] **Step 4: Run GREEN, assert no RL surface, and commit.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Commit: `git add fast3_delta/c3k_model.py tests/test_c3k_closed_loop.py; git commit -m "feat: add C3K shared knowledge score"`

### Task 3: Implement fixed paired pseudo-cold ranking

**Files:**
- Modify: `fast3_delta/c3k_model.py`
- Modify: `tests/test_c3k_closed_loop.py`

- [ ] **Step 1: Write a failing backward-pass/diagnostic test.**

```python
def test_pseudo_cold_forward_has_rank_consistency_and_gate_losses(model):
    model.set_pseudo_cold_item_mask(torch.tensor([False, True, False, False]))
    loss, diag = model(batch(), pop(), llm(), user_seen_items=history())
    loss.backward()
    assert diag["pseudo_cold_count"] == 1
    assert diag["consistency_loss"] >= 0
    assert diag["gate_regularization"] >= 0
```

- [ ] **Step 2: Run RED.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Expected: C3K forward/diagnostics are missing.

- [ ] **Step 3: Implement sampled rank loss through the shared score.**

Selected pseudo-cold rows use the masked view in positive and sampled-negative scoring; other rows use the full view. Mask same-item and known-positive negatives. Add `0.10 * stop-gradient cosine consistency(masked, full) + 0.001 * gate regularization` to rank loss. The runner passes a globally pseudo-masked train history. No actor, critic, reward, simulator, PPO, reranker, or random ID mask is permitted.

- [ ] **Step 4: Run GREEN and commit.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Commit: `git add fast3_delta/c3k_model.py tests/test_c3k_closed_loop.py; git commit -m "feat: add paired pseudo-cold C3K loss"`

### Task 4: Implement blocked full-catalog scoring and timing

**Files:**
- Create: `fast3_delta/c3k_eval.py`
- Modify: `tests/test_c3k_closed_loop.py`

- [ ] **Step 1: Write failing strict-cold and score-routing tests.**

```python
def test_catalog_masks_strict_cold_only(model):
    bank = build_c3k_item_bank(model, torch.tensor([True, False, False, True]))
    assert bank.strict_cold_mask.tolist() == [True, False, False, True]

def test_evaluator_calls_shared_catalog_score(monkeypatch, model):
    calls = []
    monkeypatch.setattr(model, "score_catalog", lambda *a, **kw: calls.append(1) or torch.zeros((1, 4)))
    evaluate_c3k(model, loader(), torch.device("cpu"), k_list=(1,), full_ranking=True, user_seen_items={0: set()})
    assert calls
```

- [ ] **Step 2: Run RED.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Expected: `fast3_delta.c3k_eval` is missing.

- [ ] **Step 3: Implement blocked evaluation.**

Cache candidate vectors only. Create the bank with `strict_cold = item_train_popularity < cold_threshold`; hot candidates retain ID evidence. Score user/item blocks through `score_catalog`; apply seen-history masking after the shared score. Return cold/hot/all full-catalog item-macro metrics plus `item_bank_seconds`, `score_seconds`, `ranking_seconds`, `total_inference_seconds`, query/candidate counts, and CUDA peak memory when available.

- [ ] **Step 4: Run GREEN and commit.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Commit: `git add fast3_delta/c3k_eval.py tests/test_c3k_closed_loop.py; git commit -m "feat: add C3K full ranking and timing"`

### Task 5: Implement independent C3K runner and serial three-seed launcher

**Files:**
- Create: `c3k_static.py`
- Create: `run_c3k_3seed.ps1`
- Modify: `tests/test_c3k_closed_loop.py`

- [ ] **Step 1: Write failing launcher-isolation and manifest tests.**

```python
def test_launcher_has_no_legacy_rl_flags():
    text = Path("run_c3k_3seed.ps1").read_text(encoding="utf-8")
    assert "USIM_CKG_RL_V1" not in text
    assert "USIM_PPO_LOSS_WEIGHT" not in text
    assert "USIM_USE_REFINED_EVAL" not in text
```

- [ ] **Step 2: Run RED.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Expected: missing runner/launcher behavior.

- [ ] **Step 3: Implement C3K-only static flow.**

Read existing static split/course artifacts without editing existing code. Make a plan with `build_pseudocold_plan(item_train_popularity, ratio=0.10, min_popularity=1, cold_threshold=1, seed=seed)`, save its audit JSON, and call `mask_user_item_history(train_history, plan.selected_mask)` only for train batches. Validation/test restore declared train-only history. Select checkpoints only if calibration improves validation cold N@10 over the same encoder with calibration disabled and hot N@10 is within 0.003; ties are cold R@10 then overall N@10.

Record every epoch raw wall time. Write `stable_epoch_timing.json` that excludes epoch 1 and reports completed stable epochs' mean/std/min/max seconds, batches, samples/sec, seed, source hash, and selected epoch. Record checkpoint-restored full-ranking timing separately in `inference_timing.json`; do not mix setup, split creation, or initialization into stable training time.

- [ ] **Step 4: Implement fixed launcher.**

`run_c3k_3seed.ps1` runs 2025, 2026, and 2027 serially in separate `outputs/c3k/full_3seed/seed_<seed>` and checkpoint roots, fixed at 40 epochs and patience 6. It sets no legacy CKG-RL/PPO/refined-eval variable and has a dry-run mode.

- [ ] **Step 5: Run GREEN, dry run, and commit.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py -q`

Run: `powershell -ExecutionPolicy Bypass -File .\run_c3k_3seed.ps1 -DryRun`

Commit: `git add c3k_static.py run_c3k_3seed.ps1 tests/test_c3k_closed_loop.py; git commit -m "feat: add isolated C3K three-seed runner"`

### Task 6: Final audit and background launch

**Files:**
- Test: `tests/test_c3k_closed_loop.py`
- Test: `tests/test_pseudocold_v1.py`
- Test: `tests/test_usim_strict_cold_repair.py`

- [ ] **Step 1: Run regressions.**

Run: `./py.bat -m pytest tests/test_c3k_closed_loop.py tests/test_pseudocold_v1.py tests/test_usim_strict_cold_repair.py -q`

- [ ] **Step 2: Audit the source surface.**

Run: `rg -n "run_usim_episode|compute_ppo_loss|USIM_CKG_RL_V1|USIM_USE_REFINED_EVAL" fast3_delta/c3k_model.py fast3_delta/c3k_eval.py c3k_static.py run_c3k_3seed.ps1`

Expected: no match. Verify dropout bypass by direct C3K tests.

- [ ] **Step 3: Start a hidden serial launcher and confirm seed-2025.**

Use `Start-Process -WindowStyle Hidden` to run `run_c3k_3seed.ps1`, redirecting to `background_logs/c3k_full_3seed_YYYYMMDD/launcher.log`. Record PID and log path. Do not claim final metrics until all three frozen runs finish.
