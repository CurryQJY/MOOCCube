# CBI-Constrained Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one isolated MOOCCube seed-2025 experiment that replaces the simulator's ID target with the initial CBI representation, projects every simulated state into the content cosine trust region, and refines both cold and hot items at evaluation.

**Architecture:** Add a focused module containing the cone projection, constrained simulator subclass, and all-refined evaluation adapter. A small Python entrypoint patches only its own process before calling the existing static experiment; a protected PowerShell launcher uses independent output/checkpoint/log roots. Shared main-table sources remain untouched.

**Tech Stack:** Python 3.12, PyTorch 2.8, pytest, PowerShell 5.1, existing FAST3/USIM protocol.

---

## File map

- Create `cbi_trust_sim.py`: projection, constrained subclass, evaluation adapter, trust diagnostics.
- Create `run_cbi_trust_sim_seed2025.py`: isolated process entrypoint.
- Create `run_cbi_trust_sim_seed2025.ps1`: reproducible launcher and protection manifest.
- Create `tests/test_cbi_trust_sim.py`: focused TDD suite.
- Create `summarize_cbi_trust_sim_seed2025.py`: result comparison and integrity checks.
- Do not modify `usim_feedback_fast3_content_delta.py`, `fast3_delta/eval.py`, main-table launchers, paper tables, or existing experiment outputs.

### Task 1: Content cone projection

**Files:**
- Create: `cbi_trust_sim.py`
- Test: `tests/test_cbi_trust_sim.py`

- [ ] **Step 1: Write failing tests**

```python
def test_projection_keeps_in_domain_vector():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.9, 0.4358899]])
    projected, stats = project_to_content_cone(state, anchor, 0.8)
    assert torch.allclose(projected, F.normalize(state, dim=1), atol=1e-6)
    assert stats["projected_count"] == 0


def test_projection_hits_cosine_boundary():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.0, 1.0]])
    projected, stats = project_to_content_cone(state, anchor, 0.8660254037844386)
    assert torch.allclose(projected.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.all((projected * anchor).sum(dim=1) >= 0.8660253)
    assert stats["projected_count"] == 1


def test_projection_handles_antiparallel_input():
    anchor = torch.tensor([[1.0, 0.0]])
    projected, _ = project_to_content_cone(-anchor, anchor, 0.8660254037844386)
    assert torch.isfinite(projected).all()
    assert torch.equal(projected, anchor)
```

- [ ] **Step 2: Verify RED**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_red tests\test_cbi_trust_sim.py -q
```

Expected: import failure because `cbi_trust_sim` does not exist.

- [ ] **Step 3: Implement minimal projection**

```python
def project_to_content_cone(state, content_anchor, cosine_floor, eps=1e-8):
    anchor = F.normalize(content_anchor, dim=1)
    unit_state = F.normalize(state, dim=1)
    cosine = (unit_state * anchor).sum(dim=1, keepdim=True)
    outside = cosine < float(cosine_floor)
    orthogonal = unit_state - cosine * anchor
    orth_norm = orthogonal.norm(dim=1, keepdim=True)
    orth_unit = orthogonal / orth_norm.clamp_min(eps)
    boundary = float(cosine_floor) * anchor + math.sqrt(
        max(0.0, 1.0 - float(cosine_floor) ** 2)
    ) * orth_unit
    degenerate = outside & (orth_norm <= eps)
    projected = torch.where(outside, boundary, unit_state)
    projected = torch.where(degenerate, anchor, projected)
    final_cosine = (projected * anchor).sum(dim=1)
    return projected, {
        "projected_count": int(outside.sum().item()),
        "projected_ratio": float(outside.float().mean().item()),
        "min_cosine": float(final_cosine.min().item()),
        "mean_cosine": float(final_cosine.mean().item()),
    }
```

- [ ] **Step 4: Verify GREEN and commit**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_green tests\test_cbi_trust_sim.py -q
git add cbi_trust_sim.py tests/test_cbi_trust_sim.py
git commit -m "feat: add CBI content-cone projection"
```

### Task 2: Constrained simulator subclass

**Files:**
- Modify: `cbi_trust_sim.py`
- Modify: `tests/test_cbi_trust_sim.py`

- [ ] **Step 1: Add failing anchor and trajectory tests**

Instantiate a tiny `CBITrustFast3FeedbackUSIM`, replace candidate selection with deterministic tensors, and assert:

```python
assert model.last_effective_target is not supplied_id_target
assert torch.equal(model.last_effective_target, F.normalize(initial_state, dim=1))
assert len(model.recorded_step_cosines) == cfg.usim_steps
assert min(model.recorded_step_cosines) >= cfg.cbi_trust_cosine_floor - 1e-6
```

- [ ] **Step 2: Verify RED**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_model_red tests\test_cbi_trust_sim.py -q
```

- [ ] **Step 3: Implement the subclass**

Subclass `Fast3FeedbackUSIM` and copy only `run_usim_episode`. At entry set:

```python
initial_cbi_anchor = F.normalize(init_item_emb.detach(), dim=1)
effective_target = initial_cbi_anchor
content_anchor = F.normalize(self._content_base_embedding(item_idx).detach(), dim=1)
```

Ignore the caller-supplied ID target. Replace all target uses with `effective_target`. Immediately after the existing update add:

```python
current_h = current_h + self.cfg.usim_lr * grad
current_h, trust_stats = project_to_content_cone(
    current_h,
    content_anchor,
    self.cfg.cbi_trust_cosine_floor,
)
```

Add candidate statistics `trust_projected_ratio`, `trust_min_cosine`, and `trust_mean_cosine`. Average ratios and means across steps; retain the minimum of `trust_min_cosine`.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_model_green tests\test_cbi_trust_sim.py -q
git add cbi_trust_sim.py tests/test_cbi_trust_sim.py
git commit -m "feat: constrain USIM states to CBI trust region"
```

### Task 3: All-refined evaluation adapter

**Files:**
- Modify: `cbi_trust_sim.py`
- Modify: `tests/test_cbi_trust_sim.py`

- [ ] **Step 1: Write failing cold/hot bank tests**

Use a four-item fake model and assert that cold items call constrained inference with `force_cold=True`, hot items use `False`, and both cold/hot positives index the same cached bank.

- [ ] **Step 2: Implement adapter**

Reuse `build_all_refined_item_bank` and `cached_bank_positive_vectors` from `evaluate_cbi_all_refined_seed2025.py`. Cache per-model banks in `weakref.WeakKeyDictionary`:

```python
def trust_build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    bank, stats = build_all_refined_item_bank(model, device, llm_scores, item_batch)
    _EVAL_BANKS[model] = bank
    model.last_trust_bank_stats = stats
    return {"cold": bank, "hot": bank, "all": bank}


def trust_build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    del llm_s, pop_sel, eval_type
    return cached_bank_positive_vectors(_EVAL_BANKS[model], item_idx)
```

Patch both `fast3_delta.eval` and the isolated protocol module.

- [ ] **Step 3: Verify and commit**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_eval tests\test_cbi_trust_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py -q
git add cbi_trust_sim.py tests/test_cbi_trust_sim.py
git commit -m "feat: refine all items under CBI trust constraint"
```

### Task 4: Isolated entrypoint and launcher

**Files:**
- Create: `run_cbi_trust_sim_seed2025.py`
- Create: `run_cbi_trust_sim_seed2025.ps1`
- Modify: `tests/test_cbi_trust_sim.py`

- [ ] **Step 1: Write failing launcher contract tests**

Assert independent roots, seed 2025, 60 epochs, delta max 0.5, paper-style content, replace-item mode, and cosine floor `sqrt(0.75)`.

- [ ] **Step 2: Implement Python entrypoint**

```python
import fast3_delta.eval as eval_mod
import usim_feedback_fast3_content_delta as protocol
from cbi_trust_sim import CBITrustFast3FeedbackUSIM, install_trust_eval_adapter


def main():
    protocol.Fast3FeedbackUSIM = CBITrustFast3FeedbackUSIM
    install_trust_eval_adapter(protocol, eval_mod)
    protocol.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement protected PowerShell launcher**

Follow `run_cbi_faithful_seed2025.ps1`, but point `ScriptPath` to the new entrypoint and use:

```text
outputs\cbi_trust_sim_single_seed2025
checkpoints\cbi_trust_sim_single_seed2025
background_logs\cbi_trust_sim_single_seed2025
```

Record source hashes and refuse overlap with main-table roots.

- [ ] **Step 4: Parse, dry-run, test, and commit**

```powershell
$tokens=$null; $errors=$null
[Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\run_cbi_trust_sim_seed2025.ps1),[ref]$tokens,[ref]$errors)|Out-Null
if($errors.Count){$errors|ForEach-Object Message;exit 1}
.\run_cbi_trust_sim_seed2025.ps1 -DryRun
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_launcher tests\test_cbi_trust_sim.py -q
git add run_cbi_trust_sim_seed2025.py run_cbi_trust_sim_seed2025.ps1 tests/test_cbi_trust_sim.py
git commit -m "feat: add isolated CBI trust-sim experiment"
```

### Task 5: Summarizer and full verification

**Files:**
- Create: `summarize_cbi_trust_sim_seed2025.py`
- Modify: `tests/test_cbi_trust_sim.py`

- [ ] **Step 1: Write failing summary tests**

Assert JSON/CSV/Markdown output with Cold/Hot R@10 and N@10, original CBI comparison, seed-2025 baseline comparison, best epoch, projection ratio, and minimum content cosine.

- [ ] **Step 2: Implement summary and integrity gates**

Reject missing results, cosine below the configured floor, non-finite trust diagnostics, or changed protected-file hashes. Write outputs under `outputs/cbi_trust_sim_single_seed2025/comparison/`.

- [ ] **Step 3: Run focused suite and commit**

```powershell
.\py.bat -m pytest --basetemp .pytest_tmp\cbi_trust_final tests\test_cbi_trust_sim.py tests\test_evaluate_cbi_all_refined_seed2025.py tests\test_cbi_faithful_single_run.py -q
git add summarize_cbi_trust_sim_seed2025.py tests/test_cbi_trust_sim.py
git commit -m "feat: summarize CBI trust-sim results"
```

### Task 6: Run the single experiment

- [ ] **Step 1: Confirm GPU and dry-run**

```powershell
nvidia-smi
.\run_cbi_trust_sim_seed2025.ps1 -DryRun
```

- [ ] **Step 2: Launch hidden background run**

```powershell
$p=Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Resolve-Path '.\run_cbi_trust_sim_seed2025.ps1')) -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru
"PID=$($p.Id)"
```

- [ ] **Step 3: Monitor trust diagnostics**

```powershell
Get-Content .\background_logs\cbi_trust_sim_single_seed2025\training.log -Wait -Tail 30
```

Every epoch must report finite trust statistics and `trust_min_cosine >= 0.866025`.

- [ ] **Step 4: Summarize after natural completion**

```powershell
.\py.bat .\summarize_cbi_trust_sim_seed2025.py
```

- [ ] **Step 5: Final verification**

Re-run the focused tests, verify source/checkpoint hashes, and compare the corrected Cold/Hot metrics with the original CBI and all-item unconstrained replay.

## Self-review

- Spec coverage: ID target removal, initial-CBI anchor, per-step cosine projection, all-item evaluation, isolated roots, trust diagnostics, one seed, and protected main-table files are covered.
- Placeholder scan: no unresolved placeholders or implicit implementation steps remain.
- Type consistency: projection returns `(Tensor, dict)`, the subclass consumes those fields, and the evaluation adapter returns the bank interface expected by `evaluate_usim`.
