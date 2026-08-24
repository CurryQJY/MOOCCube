# P1 Top-K Motivation Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export actual full-catalog Top-20 recommendations from CKG-RL and CGRC checkpoints and quantify four pedagogical risks under a paired three-seed course-macro protocol.

**Architecture:** A small shared JSONL exporter receives already-masked score tensors from both native evaluators. Read-only model entrypoints reuse the original model construction and split logic, and a separate analysis script computes model-neutral risk signals and paired statistics.

**Tech Stack:** Python 3.12, PyTorch, pandas, NumPy, SciPy, pytest.

---

### Task 1: Shared Top-K Exporter

**Files:**
- Create: `ranking_topk_export.py`
- Create: `tests/test_ranking_topk_export.py`

- [ ] Write tests for ranking order, metadata, invalid-score filtering, sequential sample IDs, and atomic replacement.
- [ ] Run `./py.bat -m pytest tests/test_ranking_topk_export.py -q` and verify the tests fail because the module does not exist.
- [ ] Implement `TopKJsonlExporter` with `write_batch`, context-manager cleanup, and atomic `os.replace`.
- [ ] Run the focused test and verify it passes.

### Task 2: Native Evaluator Integration

**Files:**
- Modify: `hin_eval_common.py`
- Modify: `fast3_delta/eval.py`
- Modify: `cgrc_paper_static_hin.py`
- Modify: `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`
- Create: `tests/test_native_topk_export_hooks.py`

- [ ] Add failing evaluator tests proving exports are taken after seen-item masking and model-specific score adjustment.
- [ ] Add optional `export_topk_path`, `export_topk_k`, and metadata arguments to both evaluators.
- [ ] Enable the hook only on the final cold item-macro evaluation call so every cold-test interaction is written exactly once.
- [ ] Run focused evaluator tests and existing strict-cold evaluator tests.

### Task 3: Read-Only Checkpoint Entrypoints

**Files:**
- Create: `export_p1_cgrc_topk.py`
- Create: `export_p1_ckgrl_topk.py`
- Create: `run_p1_topk_exports.ps1`
- Create: `tests/test_p1_checkpoint_export_entrypoints.py`

- [ ] Test that CGRC selects `best_state` and CKG-RL selects `es_best_state`.
- [ ] Test that checkpoint save calls are blocked while ordinary output writes remain enabled.
- [ ] Build seed-specific commands from the frozen checkpoint and manifest paths.
- [ ] Add output manifests containing checkpoint SHA256, script SHA256, split identity, model, seed, and Top-K.
- [ ] Run the focused tests and a one-batch smoke export.

### Task 4: Six Full Exports And Validation

**Files:**
- Generate: `outputs/p1_motivation_topk/<model>/strict_item_cold_balanced_thr1_seed_<seed>/top20_cold_test.jsonl`
- Generate: `outputs/p1_motivation_topk/<model>/strict_item_cold_balanced_thr1_seed_<seed>/export_manifest.json`

- [ ] Export CKG-RL seeds 2025, 2026, and 2027.
- [ ] Export CGRC seeds 2025, 2026, and 2027.
- [ ] Verify JSONL record counts equal native cold-test interaction counts.
- [ ] Recompute R@10/N@10 from exported rankings and compare with native evaluator outputs to numerical tolerance.
- [ ] Reject any export with duplicate sample IDs, seen-item leakage, fewer than 20 valid candidates, or mismatched checkpoint provenance.

### Task 5: Risk Analysis

**Files:**
- Create: `paper_aaai27/scripts/analyze_p1_topk_motivation.py`
- Create: `tests/test_p1_topk_motivation_analysis.py`

- [ ] Write synthetic tests for all four risk definitions, cold proportion, empty cold-only lists, and course-macro aggregation.
- [ ] Build course artifacts and concept counts without popularity-derived difficulty.
- [ ] Stream the six JSONL files and emit recommendation-level, list-level, and course-macro tables.
- [ ] Compute paired bootstrap intervals and paired permutation p-values with a fixed seed.
- [ ] Run focused tests and validate row counts and finite-value coverage.

### Task 6: Paper Artifacts And Audit

**Files:**
- Create: `paper_aaai27/scripts/draw_p1_topk_motivation.py`
- Generate: `paper_aaai27/figures/mooccube_p1_topk_motivation.{pdf,svg,png}`
- Generate: `paper_aaai27/figures/mooccube_p1_topk_motivation_summary.csv`
- Generate: `paper_aaai27/figures/mooccube_p1_topk_motivation_paired.csv`

- [ ] Produce a compact four-metric comparison with uncertainty and a separate cold-proportion annotation.
- [ ] Check that metric directions and labels match the analysis definitions.
- [ ] Run all new tests plus the relevant evaluator regression tests.
- [ ] Record whether each result supports, weakens, or falsifies the paper's motivation claim.
