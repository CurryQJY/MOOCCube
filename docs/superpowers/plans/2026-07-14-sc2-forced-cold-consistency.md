# SC2Rec-Style Forced-Cold Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and smoke-test an isolated forced-cold score-consistency experiment without modifying main-table source or aggregation files.

**Architecture:** A new Python entry point subclasses the current CKG-RL config/model, adds one-way warm-teacher to forced-cold-student KL loss, installs the subclass in memory, and delegates training/evaluation to the existing pipeline. A new PowerShell runner owns isolated output and checkpoint roots.

**Tech Stack:** Python, PyTorch, pytest, PowerShell, existing strict course-cold training pipeline.

---

### Task 1: Loss Contract

**Files:**
- Create: `tests/test_sc2_forced_cold_consistency.py`
- Create: `usim_feedback_fast3_sc2_consistency.py`

- [ ] **Step 1: Write failing tests for zero, positive-gradient, inactive, and candidate-mask behavior.**

```python
import torch

from usim_feedback_fast3_sc2_consistency import (
    forced_cold_distribution_consistency_loss,
)


def test_identical_distributions_have_zero_consistency_loss():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss = forced_cold_distribution_consistency_loss(logits, logits.clone(), temperature=0.2)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_divergent_student_has_positive_loss_and_gradient():
    teacher = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    student = torch.zeros_like(teacher, requires_grad=True)
    loss = forced_cold_distribution_consistency_loss(teacher, student, temperature=0.5)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert student.grad is not None and student.grad.abs().sum().item() > 0.0


def test_inactive_rows_return_differentiable_zero():
    student = torch.randn(2, 2, requires_grad=True)
    loss = forced_cold_distribution_consistency_loss(
        torch.randn(2, 2), student, active_rows=torch.zeros(2, dtype=torch.bool)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad is not None
```
- [ ] **Step 2: Run `./py.bat -m pytest tests/test_sc2_forced_cold_consistency.py -q` and verify import failure.**
- [ ] **Step 3: Implement `forced_cold_distribution_consistency_loss` with detached teacher targets, temperature scaling, row filtering, and candidate masking.**

```python
def forced_cold_distribution_consistency_loss(
    teacher_logits,
    student_logits,
    *,
    temperature=0.2,
    active_rows=None,
    invalid_candidate_mask=None,
):
    """One-way KL from a detached warm teacher to a forced-cold student."""
    # Validate shape/temperature, apply the same invalid-candidate mask to both
    # views, compute row-wise KL, and average only active rows. Return
    # student_logits.sum() * 0 when no row is active.
```
- [ ] **Step 4: Re-run the focused tests and verify all loss-contract tests pass.**

### Task 2: Isolated Model Entry Point

**Files:**
- Modify: `tests/test_sc2_forced_cold_consistency.py`
- Modify: `usim_feedback_fast3_sc2_consistency.py`

- [ ] **Step 1: Add failing tests for environment parsing and weighted loss/stat injection.**

```python
def test_sc2_config_reads_isolated_environment(monkeypatch):
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_WEIGHT", "0.25")
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_TEMP", "0.4")
    cfg = SC2ConsistencyConfig(n_users=2, n_items=3, content_dim=5)
    assert cfg.sc2_consistency_weight == pytest.approx(0.25)
    assert cfg.sc2_consistency_temp == pytest.approx(0.4)


def test_forward_adds_weighted_consistency_and_stats(monkeypatch):
    # Stub only the inherited base forward and the isolated consistency method.
    # Assert total == base + weight * consistency and all four diagnostic keys
    # are present. This verifies integration without mocking the loss helper.
```
- [ ] **Step 2: Run focused tests and verify the new assertions fail for missing config/model behavior.**
- [ ] **Step 3: Implement `SC2ConsistencyConfig`, `SC2ConsistencyFast3FeedbackUSIM`, `install_sc2_bindings`, and `main`.**

```python
class SC2ConsistencyConfig(legacy.Fast3Config):
    def __init__(self, n_users, n_items, content_dim=768):
        super().__init__(n_users, n_items, content_dim)
        self.sc2_consistency_weight = float(os.environ.get("USIM_SC2_CONSISTENCY_WEIGHT", "0.10"))
        self.sc2_consistency_temp = float(os.environ.get("USIM_SC2_CONSISTENCY_TEMP", "0.20"))
        self.sc2_consistency_warm_only = os.environ.get("USIM_SC2_CONSISTENCY_WARM_ONLY", "1") == "1"


class SC2ConsistencyFast3FeedbackUSIM(legacy.Fast3FeedbackUSIM):
    def forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        base_loss, stats = super().forward(batch, pop, llm_s, user_bank_raw, user_seen_items)
        consistency_loss, diagnostics = self._sc2_consistency_loss(batch, pop, llm_s, user_seen_items)
        weighted = self.cfg.sc2_consistency_weight * consistency_loss
        stats.update(diagnostics)
        stats["sc2_consistency_weighted_loss"] = float(weighted.detach().item())
        return base_loss + weighted, stats
```
- [ ] **Step 4: Run focused tests and verify they pass.**
- [ ] **Step 5: Verify train-to-eval writes isolated epoch diagnostics to `sc2_consistency_epoch_metrics.csv`.**

### Task 3: Isolated Smoke Runner

**Files:**
- Modify: `tests/test_sc2_forced_cold_consistency.py`
- Create: `run_sc2_forced_cold_consistency_smoke.ps1`

- [ ] **Step 1: Add a failing runner-isolation test that requires the new entry point, unique outputs/checkpoints, seed 2025, and one epoch.**

```python
def test_smoke_runner_is_isolated():
    text = Path("run_sc2_forced_cold_consistency_smoke.ps1").read_text(encoding="utf-8")
    assert "usim_feedback_fast3_sc2_consistency.py" in text
    assert "outputs\\sc2_forced_cold_consistency_smoke" in text
    assert "checkpoints\\sc2_forced_cold_consistency_smoke" in text
    assert "-Seeds @(2025)" in text
    assert "-Epochs 1" in text
    assert "-SkipAggregate" in text
```
- [ ] **Step 2: Run the focused test and verify failure because the runner is absent.**
- [ ] **Step 3: Create the runner with an optional `-DryRun` switch and no main-table aggregation.**

```powershell
param([switch]$DryRun)
$runnerArgs = @{
    ScriptPath = ".\usim_feedback_fast3_sc2_consistency.py"
    OutputRoot = "outputs\sc2_forced_cold_consistency_smoke"
    CheckpointRoot = "checkpoints\sc2_forced_cold_consistency_smoke"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 1
    Patience = 1
    ForceFresh = $true
    UseContentDelta = $false
    SkipAggregate = $true
}
if ($DryRun) { $runnerArgs; exit 0 }
& .\run_usim_feedback_fast3_content_delta_static.ps1 @runnerArgs
exit $LASTEXITCODE
```
- [ ] **Step 4: Run focused tests and the runner dry-run.**

### Task 4: Verification and Smoke Experiment

**Files:**
- No protected-file modifications.
- Runtime outputs only under `outputs/sc2_forced_cold_consistency_smoke/` and `checkpoints/sc2_forced_cold_consistency_smoke/`.

- [ ] **Step 1: Run the focused tests plus relevant existing strict-cold tests.**
- [ ] **Step 2: Recompute protected SHA-256 hashes and compare with the recorded values.**
- [ ] **Step 3: Run the seed-2025 one-epoch smoke experiment.**
- [ ] **Step 4: Inspect logs and result JSON for finite active consistency loss and strict cold item-macro metrics.**
- [ ] **Step 5: Recompute protected hashes again and report exact evidence, including any failure or limitation.**

### Task 5: Main-Table-Aligned Seed-2025 Formal Gate

**Files:**
- Modify: `tests/test_sc2_forced_cold_consistency.py`
- Create: `run_sc2_forced_cold_consistency_main_table_gate.ps1`

- [ ] **Step 1: Write a failing static-contract test for the formal runner.**

```python
def test_formal_gate_runner_matches_minimal_main_table_configuration():
    text = Path("run_sc2_forced_cold_consistency_main_table_gate.ps1").read_text(
        encoding="utf-8"
    )
    required = (
        'ScriptPath = ".\\usim_feedback_fast3_sc2_consistency.py"',
        'OutputRoot = "outputs\\sc2_forced_cold_consistency_main_table_gate"',
        'CheckpointRoot = "checkpoints\\sc2_forced_cold_consistency_main_table_gate"',
        'Protocol = "strict_item_cold_balanced"',
        'Seeds = @(2025)',
        'Epochs = 60',
        'Patience = 60',
        'UseContentDelta = $false',
        'UsePseudoColdTrain = $false',
        'UsePaac = $false',
        'UseSageLite = $false',
        'UseSageAuxLoss = $false',
        'UseCgrcRecon = $false',
        'UseSgUrinit = $false',
        'SkipAggregate = $true',
    )
    assert all(token in text for token in required)
```

- [ ] **Step 2: Run the new test and verify it fails because the formal runner is absent.**

Run:

```powershell
.\py.bat -m pytest tests\test_sc2_forced_cold_consistency.py::test_formal_gate_runner_matches_minimal_main_table_configuration -q
```

Expected: `FileNotFoundError` for
`run_sc2_forced_cold_consistency_main_table_gate.ps1`.

- [ ] **Step 3: Create the isolated formal runner.**

```powershell
param(
    [switch]$DryRun,
    [switch]$SkipGpuWait
)

$runnerArgs = @{
    PythonRunner = ".\py.bat"
    ScriptPath = ".\usim_feedback_fast3_sc2_consistency.py"
    OutputRoot = "outputs\sc2_forced_cold_consistency_main_table_gate"
    CheckpointRoot = "checkpoints\sc2_forced_cold_consistency_main_table_gate"
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Seeds = @(2025)
    Epochs = 60
    Patience = 60
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    UseSageLite = $false
    UseSageAuxLoss = $false
    UseCgrcRecon = $false
    UseSgUrinit = $false
    UseCourseFeedback = $true
    UseCourseReward = $true
    UsePrereqAux = $true
    UseCourseSample = $true
    UseUsimRefinedEval = $true
    PpoLossWeight = 1.0
    RolloutPolicy = "ppo"
    RlResidualScale = 1.0
    SaveCkpt = $true
    AutoResume = $false
    ForceFresh = $true
    SkipAggregate = $true
}
```

Before delegating, set `USIM_SC2_CONSISTENCY_WEIGHT=0.10`,
`USIM_SC2_CONSISTENCY_TEMP=0.20`, and
`USIM_SC2_CONSISTENCY_WARM_ONLY=1`. Reuse the smoke runner's numeric GPU-free
memory query and wait until at least 9 GB is available unless `-SkipGpuWait` is
passed. `-DryRun` prints the locked configuration and exits without training.

- [ ] **Step 4: Run the focused test, full SC2 test file, PowerShell parser, and dry run.**

Run:

```powershell
.\py.bat -m pytest tests\test_sc2_forced_cold_consistency.py -q
[scriptblock]::Create((Get-Content -Raw .\run_sc2_forced_cold_consistency_main_table_gate.ps1)) | Out-Null
.\run_sc2_forced_cold_consistency_main_table_gate.ps1 -DryRun -SkipGpuWait
```

Expected: all tests pass, PowerShell parsing succeeds, and the dry run reports
seed 2025, 60 epochs, disabled optional extensions, and SC2 weight `0.10`.

- [ ] **Step 5: Verify protected hashes, launch the runner in a hidden process, and verify the child command line and log.**

Launch with `Start-Process -WindowStyle Hidden`, redirect stdout/stderr to fresh
files under `outputs/sc2_forced_cold_consistency_main_table_gate/_launcher/`,
and write the launcher PID to `formal_gate_launcher.pid`. Verify that either the
launcher or its Python descendant is alive and that the log contains the SC2
entry-point banner. Do not terminate unrelated GPU processes.
