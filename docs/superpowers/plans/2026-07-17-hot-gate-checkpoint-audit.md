# Hot-Gate Checkpoint Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute an isolated seed-2025 Hot-gate audit with scale parity, short training, per-epoch checkpoints, and Hot-aware validation diagnostics.

**Architecture:** Add an isolated model/runner variant that restores Hot gating without Hot-only normalization, plus a launcher that writes to a fresh output root. Reuse the existing static FAST3 evaluator and preserve all protected main-table files.

**Tech Stack:** Python, PowerShell, PyTorch, pytest, existing FAST3 static runner.

---

### Task 1: Add regression coverage

**Files:**
- Modify: `tests/test_cbi_hot_gate.py`

- [ ] Add assertions that the isolated model returns a fused Hot vector without an extra normalization-only behavior and that the launcher locks `Epochs=8`, `Patience=8`, `ForceFresh=true`, and a distinct output root.
- [ ] Run `.\py.bat -m pytest tests/test_cbi_hot_gate.py -q --basetemp .pytest_tmp/hot_gate_audit_red` and confirm the new launcher assertions fail before implementation.

### Task 2: Implement the isolated audit runner

**Files:**
- Create: `cbi_hot_gate_audit_seed2025.py`
- Create: `run_cbi_hot_gate_audit_seed2025.ps1`

- [ ] Reuse `CBIHotGateFast3FeedbackUSIM` behavior but remove the Hot-only `F.normalize` before simulation.
- [ ] Lock seed 2025, epochs 8, patience 8, fresh isolated output/checkpoint/log roots, and preserve the existing ContentDelta/USIM settings.
- [ ] Export validation rows and balanced score fields without changing the shared evaluator or main-table scripts.

### Task 3: Verify and launch

**Files:**
- Test: `tests/test_cbi_hot_gate.py`

- [ ] Run the focused test and confirm all assertions pass.
- [ ] Run PowerShell `-DryRun` and verify the locked configuration.
- [ ] Confirm the fresh output/checkpoint roots do not contain prior manifests.
- [ ] Start the launcher in the background using its internal log writer.

### Task 4: Post-run analysis

- [ ] Verify manifest status is `completed`, exit code is 0, and protected-file hashes match.
- [ ] Extract per-epoch Cold/Hot validation and final test metrics.
- [ ] Report best Cold, best Hot, and balanced checkpoints; do not promote to multi-seed unless Hot and Overall satisfy the screening criterion.
