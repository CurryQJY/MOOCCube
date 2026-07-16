# P1 Risk Robustness And Paper Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the six P1 exports to their checkpoints and splits, test the observed risk trade-offs under fixed sensitivity settings, and produce paper-ready artifacts and wording.

**Architecture:** Harden the existing P1 analyzer at its provenance and native-count boundaries, then add a separate chunked robustness analyzer that consumes the frozen recommendation-level artifact. A focused drawing script consumes only compact CSV summaries, and manuscript edits cite the raw metric directions rather than a composite score.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, PyTorch checkpoint hashing, matplotlib, pytest, LaTeX.

---

### Task 1: Export Provenance And Coverage Gates

**Files:**
- Create: `export_p1_cgrc_topk.py`
- Modify: `paper_aaai27/scripts/analyze_p1_topk_motivation.py`
- Modify: `tests/test_p1_checkpoint_export_entrypoints.py`
- Modify: `tests/test_p1_topk_motivation_analysis.py`

- [ ] Add failing tests that require a CGRC export manifest to bind seed, Top-K, JSONL path, native result path, split hashes, script hashes, record count, and a checkpoint hash measured before and after replay.
- [ ] Run the two focused test modules and confirm failures are caused by the missing CGRC entrypoint and missing provenance/count gates.
- [ ] Implement the read-only CGRC entrypoint and atomic manifest writer.
- [ ] Extend native metric loading to return `count_full_cold` and reject export/native count mismatches.
- [ ] Validate CKG-RL export-time manifests and CGRC replay manifests before streaming records.
- [ ] Re-run the focused tests and retain the existing replay-drift columns separately from acceptance deltas.

### Task 2: Risk Semantics And Hot-Loop Repair

**Files:**
- Modify: `paper_aaai27/scripts/analyze_p1_topk_motivation.py`
- Modify: `tests/test_p1_topk_motivation_analysis.py`
- Modify: `docs/superpowers/specs/2026-07-15-p1-topk-motivation-experiment-design.md`

- [ ] Add failing tests for histories shorter than five, histories longer than five, and reuse of an already-binary prerequisite matrix.
- [ ] Parameterize readiness depth with default five and define readiness over `min(k, history size)` advanced courses.
- [ ] Store prerequisites as boolean artifacts and avoid a full-matrix comparison inside each record.
- [ ] Add an optional precomputed readiness argument used by streaming analysis.
- [ ] Update the original design wording to say "up to five available historical courses."
- [ ] Run the focused analysis tests.

### Task 3: Chunked Robustness Analysis

**Files:**
- Create: `paper_aaai27/scripts/analyze_p1_risk_robustness.py`
- Create: `tests/test_p1_risk_robustness.py`

- [ ] Add failing synthetic tests for the 3-by-3 sensitivity grid, list-before-course aggregation, cold-only missingness, history bins, rank aggregation, and primary-setting reproduction.
- [ ] Implement per-seed dense readiness caches for `k={3,5,10}` and scales `{P90,P95,max}`.
- [ ] Stream `recommendation_level.csv.gz` in record-aligned chunks, aggregate Top-10 course rows, rank rows, and history-stratified rows, and avoid new recommendation-level output.
- [ ] Compute paired bootstrap intervals and permutation p-values with analysis seed 2027.
- [ ] Emit `difficulty_sensitivity_course_macro.csv`, `difficulty_sensitivity_paired.csv`, `rank_profile_course_macro.csv`, `rank_profile_paired.csv`, `history_strata_course_macro.csv`, `history_strata_paired.csv`, and `robustness_manifest.json`.
- [ ] Run synthetic tests, then the real analysis and verify P95/Top-5 matches the frozen primary result.

### Task 4: Publication Figures

**Files:**
- Create: `paper_aaai27/scripts/draw_p1_topk_motivation.py`
- Create: `tests/test_draw_p1_topk_motivation.py`
- Generate: `paper_aaai27/figures/mooccube_p1_topk_motivation.{pdf,svg,png}`
- Generate: `paper_aaai27/figures/mooccube_p1_risk_robustness.{pdf,svg,png}`

- [ ] Add failing tests for favorable-effect orientation, expected metric order, required robustness cells, and all output formats.
- [ ] Implement a compact main forest plot plus cold-exposure panel from the frozen paired/model summaries.
- [ ] Implement a supplementary sensitivity/rank figure from compact robustness CSVs.
- [ ] Render all formats and inspect PNG dimensions, clipping, labels, zero lines, and direction annotations.
- [ ] Run the drawing tests.

### Task 5: Manuscript Integration

**Files:**
- Modify: `paper_aaai27/main.tex`
- Modify: `paper_aaai27/figures/p1_topk_motivation_analysis/risk_analysis_report.md`

- [ ] Insert the P1 comparison figure in RQ2 and define all four model-neutral risk signals, aggregation, and cold-only missingness succinctly.
- [ ] Report Top-10 raw differences, intervals, and cold exposure without claiming uniform risk reduction.
- [ ] Limit Top-20 stability wording to prerequisite, continuity, redundancy, and exposure; describe difficulty as cutoff-dependent.
- [ ] Add CGRC replay/RNG, proxy-signal, three-fit, and no-composite-weight limitations.
- [ ] Compile the paper and inspect page count, missing references, overfull boxes, and figure legibility.

### Task 6: Final Audit

**Files:**
- Verify all P1 files above.

- [ ] Run the P1 analyzer, robustness analyzer, and drawing script from repository root.
- [ ] Run `pytest` for exporter, evaluator hook, provenance, analysis, robustness, and drawing modules with a workspace-local basetemp.
- [ ] Recompute output row counts and hashes from the generated manifests.
- [ ] Confirm every manuscript claim maps to a summary or paired-statistics row.
- [ ] Report the supported, inconclusive, and falsified parts of the motivation claim without merging unrelated worktree changes.
