# Clean Learner-Guided Full Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a minimal, leakage-controlled learner-guided cold-item refinement model that can replace the inconsistent PPO Full.

**Architecture:** A new isolated entry point imports the recovered recommender but replaces rollout with one deterministic refinement kernel shared by training and inference. A fixed item-level pseudo-cold mask blocks ID/history leakage; a non-negative learner-fit gate selects unique learners; bounded residual updates expose activation and displacement diagnostics.

**Tech Stack:** Python 3.12, PyTorch, pytest, PowerShell, existing `fast3_delta` data/evaluation/checkpoint utilities.

---

### Task 1: Freeze the experiment and isolate the new Full

**Files:**
- Create: `learner_guided_cold_refinement.py`
- Create: `tests/test_learner_guided_cold_refinement.py`
- Create: `run_learner_guided_full_seed2025.ps1`
- Reference: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`

- [ ] **Step 1: Write a failing isolation test**

```python
def test_new_full_has_an_isolated_entrypoint_and_output_roots():
    source = (ROOT / "learner_guided_cold_refinement.py").read_text("utf-8")
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text("utf-8")
    assert "usim_feedback_fast3_content_delta_recovered_51ea_candidate" in source
    assert "learner_guided_full" in runner
    assert "main_table_51ea12fc_candidate" not in runner
```

- [ ] **Step 2: Verify RED**

Run: `.\py.bat -m pytest tests\test_learner_guided_cold_refinement.py -q --basetemp=.pytest_tmp\learner_full_red`

Expected: FAIL because the new files do not exist.

- [ ] **Step 3: Create the isolated entry point**

```python
from usim_feedback_fast3_content_delta_recovered_51ea_candidate import *

METHOD_NAME = "learner_guided_cold_refinement"
```

The PowerShell script must use new output/checkpoint roots, record the Python SHA-256 and full configuration in `locked_config.json`, reject changed resumes, save optimizer/RNG state, and expose `Epochs`, `Patience`, `Seed`, `RunName`, and `ForceFresh` parameters.

- [ ] **Step 4: Verify GREEN**

Run the test from Step 2. Expected: PASS.

### Task 2: Implement the non-negative learner-fit gate

**Files:**
- Modify: `learner_guided_cold_refinement.py`
- Test: `tests/test_learner_guided_cold_refinement.py`

- [ ] **Step 1: Add failing numerical tests**

```python
def test_learner_fit_is_finite_nonnegative_and_penalty_monotone():
    similarity = torch.tensor([[-1.0, 0.0, 1.0]])
    concept = torch.tensor([[0.2, 0.2, 0.2]])
    low_gap = torch.zeros_like(similarity)
    high_gap = torch.ones_like(similarity)
    low = learner_fit(similarity, concept, low_gap, low_gap, low_gap)
    high = learner_fit(similarity, concept, high_gap, high_gap, high_gap)
    assert torch.isfinite(low).all() and (low >= 0).all() and (low <= 1).all()
    assert torch.all(high <= low)
```

- [ ] **Step 2: Verify RED**

Run the single test. Expected: ImportError for `learner_fit`.

- [ ] **Step 3: Implement the minimal gate**

```python
def learner_fit(similarity, concept, prereq_gap, difficulty_gap, redundancy,
                concept_weight=0.25, prereq_beta=1.0, difficulty_beta=1.0):
    base = (0.5 * (similarity.clamp(-1, 1) + 1.0) + concept_weight * concept).clamp(0, 1)
    return (
        base
        * torch.exp(-prereq_beta * prereq_gap.clamp(0, 1))
        * torch.exp(-difficulty_beta * difficulty_gap.clamp(0, 1))
        * (1.0 - redundancy.clamp(0, 1))
    ).clamp(0, 1)
```

- [ ] **Step 4: Run the focused test**

Expected: PASS with no warning or non-finite value.

### Task 3: Implement bounded deterministic refinement

**Files:**
- Modify: `learner_guided_cold_refinement.py`
- Test: `tests/test_learner_guided_cold_refinement.py`

- [ ] **Step 1: Add failing behavioral tests**

Test that: warm rows remain bit-identical; `fit <= min_fit` rows do not move; active rows move toward the selected learner; each step is at most `step_cap`; total displacement is at most `total_cap`; stopped rows remain stopped; selected user IDs do not repeat.

```python
assert torch.equal(refined[~effective_cold], initial[~effective_cold])
assert torch.linalg.vector_norm(refined - initial, dim=1).max() <= total_cap + 1e-6
assert diagnostics["repeated_user_rate"] == 0.0
```

- [ ] **Step 2: Verify RED**

Expected: FAIL because `bounded_learner_refinement` is missing.

- [ ] **Step 3: Implement one shared kernel**

The function signature must be:

```python
def bounded_learner_refinement(
    initial_h, candidate_vectors, candidate_user_ids, candidate_fit,
    effective_cold, steps=3, lr=0.10, min_fit=0.05,
    step_cap=0.05, total_cap=0.10,
):
    ...
    return refined_h, diagnostics
```

Use deterministic `argmax`, mask previously selected IDs, maintain a per-row active mask, normalize `selected_user-current_h`, scale by selected fit, clip step norm, project total displacement, and restore warm rows exactly.

- [ ] **Step 4: Verify all bounded-update tests**

Expected: PASS.

### Task 4: Enforce leakage-controlled pseudo-cold training

**Files:**
- Modify: `learner_guided_cold_refinement.py`
- Modify: `run_learner_guided_full_seed2025.ps1`
- Test: `tests/test_learner_guided_cold_refinement.py`

- [ ] **Step 1: Add failing leakage tests**

Assert that the fixed item-tail mask is stable across batches/epochs; all masked items are removed from every learner history; forced-cold item vectors do not use true ID embeddings; `AuxWeight=0`; prerequisite auxiliary, PPO, course reward, PAAC, SAGE, CGRC, and LLM score are disabled in the formal script.

- [ ] **Step 2: Verify RED**

Expected: FAIL on missing script locks.

- [ ] **Step 3: Lock the clean configuration**

The script must contain:

```powershell
-UsePseudoColdTrain $true -PseudoColdMode "item_tail" -PseudoColdRatio 0.3
-TrainForceCold $true -AuxWeight 0.0 -AuxHotOnly $true
-PpoLossWeight 0.0 -UseCourseReward $false -UsePrereqAux $false
-UsePaac $false -UseSageLite $false -UseSageAuxLoss $false
-UseCgrcRecon $false
```

Set LLM scoring to disabled through the existing environment/config control.

- [ ] **Step 4: Run leakage tests**

Expected: PASS.

### Task 5: Route training and inference through the same refinement kernel

**Files:**
- Modify: `learner_guided_cold_refinement.py`
- Test: `tests/test_learner_guided_cold_refinement.py`

- [ ] **Step 1: Add a failing parity test**

Construct a tiny deterministic model and assert that training-mode refinement and `infer_refined_item_vectors` return the same cold vectors for identical model parameters, candidate bank, fit tensors, and seed; warm vectors must match the static bank.

- [ ] **Step 2: Verify RED**

Expected: FAIL because the new model has no shared inference hook.

- [ ] **Step 3: Implement `LearnerGuidedColdModel`**

Subclass the recovered `Fast3FeedbackUSIM`. Override the episode/refinement path so Actor/Critic and PPO are never called. Override `infer_refined_item_vectors` to call the same `bounded_learner_refinement` function used in `forward`.

- [ ] **Step 4: Run parity and legacy checkpoint-read tests**

Expected: PASS; checkpoint loading must not write into protected checkpoint directories.

### Task 6: Add diagnostics and formal safety gates

**Files:**
- Modify: `learner_guided_cold_refinement.py`
- Test: `tests/test_learner_guided_cold_refinement.py`

- [ ] **Step 1: Add failing diagnostic tests**

Require finite keys: `fit_mean`, `fit_p25`, `fit_p50`, `fit_p75`, `update_active_ratio`, `stopped_ratio`, `step_displacement_mean`, `step_displacement_max`, `total_displacement_mean`, `total_displacement_max`, and `repeated_user_rate`.

- [ ] **Step 2: Verify RED**

Expected: FAIL on missing keys.

- [ ] **Step 3: Implement aggregation and rejection checks**

Raise `RuntimeError` on non-finite fit/displacement, nonzero repeated-user rate, warm-row movement, or displacement above tolerance. Print the diagnostics in each `[STATIC-TRAIN]` line and save them in the epoch metrics CSV.

- [ ] **Step 4: Run diagnostic tests**

Expected: PASS.

### Task 7: Run regression and a one-epoch smoke

**Files:**
- Test: `tests/test_learner_guided_cold_refinement.py`
- Verify: existing targeted regression tests

- [ ] **Step 1: Run tests**

```powershell
.\py.bat -m pytest tests\test_learner_guided_cold_refinement.py tests\test_coursefit_pseudocold_minimal_repair.py tests\test_main_checkpoint_actor_inference_ab.py -q --basetemp=.pytest_tmp\clean_full_regression
```

Expected: all tests pass.

- [ ] **Step 2: Compile and parse**

```powershell
.\py.bat -m py_compile learner_guided_cold_refinement.py
```

Parse the PowerShell script with `Management.Automation.Language.Parser`; expected zero errors.

- [ ] **Step 3: Run a fresh one-epoch smoke**

Use `RunName=learner_guided_full_smoke_seed2025`, `Epochs=1`, `Patience=1`, and `ForceFresh`. Expected: no stderr, finite diagnostics, fixed pseudo-cold ratio near 30%, PPO/prerequisite auxiliary loss zero, checkpoint and full-ranking exports present.

### Task 8: Execute the seed-2025 validation screen

**Files:**
- Create: `run_learner_guided_seed2025_screen.ps1`
- Create: `learner_guided_screen_report.py`
- Test: `tests/test_learner_guided_screen_report.py`

- [ ] **Step 1: Encode the five-arm matrix**

Run A–E from the design for 15 epochs in serial unless GPU memory measurement proves two jobs fit safely. Each arm gets separate output/checkpoint/log roots and a locked config.

- [ ] **Step 2: Implement validation-only aggregation**

The report must read epoch validation metrics only, select each arm's best epoch by cold item-macro N@10, and refuse any test metric input during screening.

- [ ] **Step 3: Apply continuation gates**

Continue to 35 epochs only when `best_n10 >= global_best_n10 - 0.02`. Continue to 60 only for A and the best refinement arm. Do not start seeds 2026/2027 unless the best refinement arm reaches `N@10 >= 0.3058` and beats A by `>= 0.003`.

### Task 9: Complete causal controls and three seeds

**Files:**
- Create: `run_learner_guided_confirmatory_3seed.ps1`
- Modify: `learner_guided_screen_report.py`

- [ ] **Step 1: Freeze configuration from validation**

Write a frozen JSON manifest containing source hash, all hyperparameters, chosen T, caps, min-fit, and validation-selected epoch rule. Configuration changes must create a new experiment name.

- [ ] **Step 2: Run seeds 2025, 2026, 2027**

Run clean T=0, proposed Full, and random-user control. Save checkpoints each epoch and optimizer/RNG state for resumability.

- [ ] **Step 3: Evaluate test once**

After all seeds finish, load the validation-selected checkpoint for each seed and export test cold/hot item-macro R/N@5/10/20. Report mean and standard deviation; never choose an epoch or arm from test.

### Task 10: Paper integration gate

**Files:**
- Modify only after evidence: `paper_aaai27/main.tex`
- Create: `outputs/recppo_research_repair/learner_guided_full/frozen_method_manifest.json`

- [ ] **Step 1: Decide the method status**

If Full fails to beat clean T=0 and random-user controls, do not insert it into the main table and remove the refinement claim. If it passes, name it learner-guided iterative representation adaptation and explicitly state that no PPO is used.

- [ ] **Step 2: Report required ablations**

Include T=0, T=1, T=3, random user, and weak-aux results on MOOCCube. Cross-dataset main-table runs use the same frozen Full configuration.

- [ ] **Step 3: Preserve provenance**

Every table row must cite its output root, source SHA-256, split manifest, seeds, validation selection rule, and whether it is an old main-table result or a clean-Full rerun.

