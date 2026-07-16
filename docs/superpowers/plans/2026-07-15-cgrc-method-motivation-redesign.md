# CGRC Method Motivation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old dataset-level risk panel with a reproducible diagnosis of generic CGRC transfer to the course domain and align the manuscript's problem-to-method motivation.

**Architecture:** A chunked analyzer consumes the frozen Top-20 recommendation artifact and emits compact course-macro and paired summaries. The existing drawing entrypoint consumes those summaries plus the frozen CGRC exposure export, while the manuscript uses Figure 1 only for baseline diagnosis and keeps CKG-RL comparisons in RQ2.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, pytest, LaTeX.

---

### Task 1: Lock The Baseline-Only Aggregation Contract

**Files:**
- Create: `tests/test_method_motivation.py`
- Create: `paper_aaai27/scripts/analyze_method_motivation.py`

- [x] Add a synthetic test with unequal interaction counts that requires averaging recommendations by `(seed, target course, rank)` before forming Top-10 and bottom-10 buckets.
- [x] Add tests for CGRC-only filtering, complete ranks 1--20, lower/higher favorable-effect orientation, and paired bootstrap output.
- [x] Run `python -m pytest tests/test_method_motivation.py -q --basetemp .pytest_tmp/method_motivation_red` and confirm the missing analyzer causes failure.
- [x] Implement chunked rank accumulation, bucket aggregation, paired statistics, manifest hashing, and output validation.
- [x] Re-run the focused test and require all assertions to pass.

### Task 2: Replace Figure 1 Panel B

**Files:**
- Modify: `paper_aaai27/scripts/draw_method_motivation.py`
- Modify: `tests/test_method_motivation.py`
- Generate: `paper_aaai27/figures/mooccube_method_motivation.{pdf,svg,png}`

- [x] Add a drawing test that supplies compact synthetic exposure and paired inputs and requires PDF, SVG, and PNG outputs.
- [x] Replace the old test-pair risk boxplot with a favorable-alignment forest plot for prerequisite gap, concept continuity, difficulty gap, and structural redundancy.
- [x] Add the CGRC Top-10 cold-course share to the exposure panel without changing the NDCG tail CDF.
- [x] Generate all three formats and inspect the PNG at original resolution for clipping, readable labels, and an explicit zero line.

### Task 3: Rebuild The Manuscript Motivation Chain

**Files:**
- Modify: `paper_aaai27/main.tex`

- [x] Replace the Introduction's absolute test-pair risk claims with the CGRC-only exposure and rank-alignment diagnosis.
- [x] Map missing cold evidence to content anchoring/masking/simulation and map prerequisite/redundancy failures to supervision, sampling, and rewards.
- [x] Replace the old RQ2 paragraph containing prerequisite-gap mean, popularity difficulty, and gated concept-bonus zero rate with the Top-10-versus-bottom-10 paired effects.
- [x] Keep the CKG-RL-versus-CGRC Top-10 audit explicitly labeled as post-hoc validation and retain its negative concept/difficulty findings.

### Task 4: Real-Data And Paper Verification

**Files:**
- Verify all files above.

- [x] Run the real motivation analyzer and require 204 paired course units, 4,080 rank-course rows, and 408 bucket-course rows.
- [x] Confirm the new CGRC Top-10 means match the frozen P1 model summary within floating-point tolerance.
- [x] Run focused motivation and P1 regression tests.
- [x] Compile `paper_aaai27/main.tex` and check page count, undefined references, overfull boxes, and Figure 1 legibility.
- [x] Scan the primary manuscript for obsolete claims about popularity-based difficulty or gated concept-bonus sparsity.
