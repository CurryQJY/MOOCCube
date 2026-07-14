# KGRec Relation Sampling Fix Implementation Plan

> **For agentic workers:** Execute inline in the current workspace; no subagent delegation is required for this one-function correction.

**Goal:** Ensure KGRec training samples every loader KG relation type from `1` through `n_relations - 1`.

**Architecture:** Preserve the existing relation-aware sampler and change only its relation-index bounds and first-iteration initialization. Add a deterministic unit regression using `samp_rate=1.0`, then verify the corrected highest relation receives a nonzero gradient in a real Junyi batch.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Add a failing relation-coverage regression

**Files:**
- Modify: `tests/test_kgrec_native_scatter.py`

- [ ] Add `test_relation_aware_sampling_includes_every_noninteraction_relation`, constructing relation types `1..6` and calling `_relation_aware_edge_sampling(..., n_relations=7, samp_rate=1.0)`.
- [ ] Run `D:\Anaconda3\envs\zw\python.exe -m pytest -q tests/test_kgrec_native_scatter.py::test_relation_aware_sampling_includes_every_noninteraction_relation`.
- [ ] Confirm the old implementation fails because type `6` is absent.

### Task 2: Correct the sampling bounds

**Files:**
- Modify: `paper_aaai27/baseline_sources/KGRec/modules/KGRec.py:42`

- [ ] Change the relation loop from `range(n_relations - 1)` to `range(1, n_relations)`.
- [ ] Change the accumulator initialization condition from `i == 0` to `i == 1`.
- [ ] Re-run the focused regression and confirm it passes.

### Task 3: Verify behavior and regression safety

**Files:**
- Test: `tests/test_kgrec_native_scatter.py`
- Test: `tests/test_kgrec_strict_runner.py`
- Test: `tests/test_kgrec_strict_adapter.py`

- [ ] Run all KGRec tests selected by `D:\Anaconda3\envs\zw\python.exe -m pytest -q tests/test_kgrec_native_scatter.py tests/test_kgrec_strict_runner.py tests/test_kgrec_strict_adapter.py`.
- [ ] Run a real Junyi one-batch backward pass and confirm loader relation type `6` has a finite, nonzero gradient.
- [ ] Inspect the final diff and confirm no training protocol or hyperparameter changed.
