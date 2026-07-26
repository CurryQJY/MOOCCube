# Strict Stage-B Component Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated validation-only campaign that screens one Stage-B component at a time and records reproducible accept/reject decisions.

**Architecture:** A new runner reuses the frozen-Hot contract, validation loader, pseudo-cold masking, evaluator, and selection rules from the strict replication path. A campaign wrapper invokes one candidate at a time and maintains an append-only ledger; the baseline runner and main-table code stay untouched.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest, PowerShell launch wrapper.

---

### Task 1: Define the component configuration and content-anchor objective

**Files:**
- Create: `ckg_stageb_component_screen.py`
- Create: `tests/test_ckg_stageb_component_campaign.py`

- [ ] **Step 1: Write a failing test for final-space soft-anchor scaling**

```python
def test_soft_anchor_loss_is_zero_for_content_and_scales_by_fixed_tau():
    from ckg_stageb_component_screen import soft_anchor_loss

    base = torch.tensor([[1.0, 0.0]])
    adapted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    selected = torch.tensor([False, True])

    loss = soft_anchor_loss(adapted, anchors, selected, trust_tau=0.5)

    assert loss.item() == pytest.approx(8.0)
```

- [ ] **Step 2: Run the test and verify it fails because the module is absent**

Run: `pytest tests/test_ckg_stageb_component_campaign.py::test_soft_anchor_loss_is_zero_for_content_and_scales_by_fixed_tau -q`

Expected: import failure for `ckg_stageb_component_screen`.

- [ ] **Step 3: Implement the minimal soft-anchor loss and immutable config**

```python
def soft_anchor_loss(adapted, content_anchor, selected_mask, trust_tau):
    delta = adapted[selected_mask] - content_anchor[selected_mask]
    return (delta.norm(dim=1) / float(trust_tau)).pow(2).mean()
```

The configuration must reject test evaluation, an unregistered seed, negative
weights, non-fixed tau, or an unknown component name.

- [ ] **Step 4: Run the focused test**

Run: `pytest tests/test_ckg_stageb_component_campaign.py::test_soft_anchor_loss_is_zero_for_content_and_scales_by_fixed_tau -q`

Expected: `1 passed`.

### Task 2: Implement the strict runner and artifact contract

**Files:**
- Modify: `ckg_stageb_component_screen.py`
- Modify: `tests/test_ckg_stageb_component_campaign.py`

- [ ] **Step 1: Write failing tests for validation-only paths and selected-row guards**

```python
def test_component_runner_source_does_not_reference_test_or_stream_data():
    source = Path("ckg_stageb_component_screen.py").read_text(encoding="utf-8")
    assert "static_test.pkl" not in source
    assert "stream_data.pkl" not in source

def test_candidate_acceptance_requires_immutable_guard_and_cold_gain():
    from ckg_stageb_component_screen import decide_single_seed_screen
    assert decide_single_seed_screen(candidate, incumbent, immutable_baseline) == "provisionally_accepted"
```

- [ ] **Step 2: Run the tests and verify they fail because the runner/decision helper is absent**

Run: `pytest tests/test_ckg_stageb_component_campaign.py -q`

Expected: failure naming the missing helper.

- [ ] **Step 3: Reuse the strict replication contract and add only the soft-anchor loss**

The runner must load the dynamic Hot contract for seeds 2026/2027, freeze all
Hot parameters, build deterministic pseudo-cold masks, and use the same
full-catalog item-macro evaluator. Its training loss is `ranking + .10 *
soft_anchor`; it writes per-epoch metrics plus `component_result.json`.

- [ ] **Step 4: Run the focused test module**

Run: `pytest tests/test_ckg_stageb_component_campaign.py -q`

Expected: all tests pass.

### Task 3: Implement the campaign ledger and serial launcher

**Files:**
- Create: `ckg_stageb_component_campaign.py`
- Create: `run_ckg_stageb_component_campaign.ps1`
- Modify: `tests/test_ckg_stageb_component_campaign.py`

- [ ] **Step 1: Write failing tests for append-only ledger entries and strict seed order**

```python
def test_campaign_queue_starts_on_2027_and_replication_requires_provisional_acceptance():
    from ckg_stageb_component_campaign import campaign_steps
    assert campaign_steps()[0].seed == 2027
    assert campaign_steps()[1].requires_previous_acceptance is True
```

- [ ] **Step 2: Run the test and verify it fails because the campaign module is absent**

Run: `pytest tests/test_ckg_stageb_component_campaign.py::test_campaign_queue_starts_on_2027_and_replication_requires_provisional_acceptance -q`

Expected: import failure for `ckg_stageb_component_campaign`.

- [ ] **Step 3: Implement a fresh-root serial campaign**

The wrapper must use a new timestamped campaign root, run seed 2027 first,
append all terminal decisions to JSONL and CSV, run seed 2026 only for a
provisionally accepted candidate, and return a nonzero exit only for an
execution/contract failure. A rejected scientific candidate is a completed
campaign outcome, not a launcher failure.

- [ ] **Step 4: Run tests and a dry run**

Run: `pytest tests/test_ckg_stageb_component_campaign.py -q`

Expected: all tests pass.

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_stageb_component_campaign.ps1 -DryRun`

Expected: JSON declaring validation-only strict seeds and no GPU process.

### Task 4: Launch and verify the first component screen

**Files:**
- Create: `outputs/ckg_stageb_component_campaign_*/`
- Create: `background_logs/ckg_stageb_component_campaign_*/`

- [ ] **Step 1: Start the fresh campaign root**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckg_stageb_component_campaign.ps1`

Expected: seed 2027 soft-anchor screen starts and logs its validation rows.

- [ ] **Step 2: Verify terminal ledger fields after completion**

Run: `Get-Content outputs\ckg_stageb_component_campaign_*\component_ledger.jsonl`

Expected: each component/seed row has metrics, deltas, guard state, hashes,
and one of `provisionally_accepted`, `accepted`, `rejected`, or `failed`.
