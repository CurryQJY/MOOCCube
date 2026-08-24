# TDInit Results Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible Cold, Hot, and Overall CKG-RL versus CKG-RL+TDInit result tables from the completed three-seed logs.

**Architecture:** A standalone Python script loads the two per-seed CSV files, computes item-macro panel summaries and signed relative improvements, writes machine-readable CSV/LaTeX, and renders reference-style PNG/PDF tables. Tests verify the Overall aggregation and generated artifacts.

**Tech Stack:** Python, csv, matplotlib, pytest

---

### Task 1: Define tested aggregation behavior

**Files:**
- Create: `tests/test_generate_tdinit_results_table.py`

- [ ] Write a test that imports the new generator, checks count-weighted Overall aggregation, locks the baseline to the latest main-table CKG-RL source, and verifies all expected output artifacts.
- [ ] Run the test and confirm it fails because the generator does not yet exist.

### Task 2: Implement the isolated generator

**Files:**
- Create: `paper_aaai27/scripts/generate_tdinit_results_table.py`

- [ ] Load and align the baseline and TDInit rows by seed.
- [ ] Compute Cold, Hot, and count-weighted Overall item-macro summaries.
- [ ] Export comparison CSV and booktabs LaTeX.
- [ ] Render the combined and three individual tables to PNG/PDF.

### Task 3: Verify artifacts

**Files:**
- Create: `paper_aaai27/figures/ckg_rl_tdinit_3seed_latest/*`

- [ ] Run the focused pytest test.
- [ ] Run the generator against the completed experiments.
- [ ] Check all exported files exist and inspect the combined PNG visually.
