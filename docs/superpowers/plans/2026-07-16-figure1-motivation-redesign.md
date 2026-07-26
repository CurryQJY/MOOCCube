# Figure 1 Motivation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Figure 1 as an existing-method diagnosis with a targeted CKG-RL response map, without duplicating Figure 3's numerical method comparison.

**Architecture:** The plotting script will select PCGNN course-level structural diagnostics and CGRC course-level NDCG@10 exposure diagnostics from existing held-out test artifacts. A two-panel diagnostic layout plus a full-width response strip will use direct labels and hatch patterns, so no standalone legend or color-only encoding is required.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, pytest, LaTeX/latexmk, Poppler.

---

### Task 1: Lock the evidence and grayscale contract in tests

**Files:**
- Modify: `tests/test_method_motivation.py`

- [ ] **Step 1: Replace the direct-comparison test with a diagnostic-selection test**

Create fixtures containing PCGNN course-level prerequisite/difficulty rows, CGRC NDCG@10 rows, and decoy CKG-RL rows. Assert that the new selector returns 204 PCGNN units, 204 CGRC units, a CGRC low-NDCG fraction of 94/204, and no CKG-RL entry.

- [ ] **Step 2: Add a grayscale-encoding assertion**

Expose `MODEL_HATCHES` and assert that PCGNN uses `///` while CGRC uses a distinct dotted or cross hatch.

- [ ] **Step 3: Update the export test**

Call the new diagnostic figure function with course-macro, CGRC exposure, and model-summary fixtures. Assert PDF, SVG, and PNG outputs are nonempty.

- [ ] **Step 4: Run the focused test and verify it fails**

Run:

    D:\anaconda3\envs\req_py312\python.exe -m pytest tests\test_method_motivation.py -q --basetemp=.pytest_tmp\figure1_diagnosis_red

Expected: failure because the diagnostic selector, hatch contract, and new drawing function do not yet exist.

### Task 2: Implement the existing-method diagnostic figure

**Files:**
- Modify: `paper_aaai27/scripts/draw_method_motivation.py`

- [ ] **Step 1: Remove Figure 1's direct CKG-RL comparison path**

Remove the Figure 1-only `MAIN_COLD_EFFECTIVENESS`, `SUPPORTED_RISK_ROWS`, `supported_motivation_rows`, and `draw_supported_motivation_figure` path. Leave the separate Figure 3 audit functions untouched.

- [ ] **Step 2: Add explicit visual encodings**

Define PCGNN diagonal hatching (`///`) and CGRC dotted hatching (`...`). Use a pale-blue response strip with a solid dark-blue border.

- [ ] **Step 3: Add diagnostic selection**

Implement a function that:

- selects PCGNN `cutoff == 10` course-macro rows;
- selects 204 CGRC course-level NDCG@10 rows;
- rejects missing or duplicate seed-course coverage;
- returns PCGNN prerequisite/difficulty means and 95% bootstrap intervals;
- returns the CGRC fraction with NDCG@10 no greater than 0.10;
- reads CGRC Top-10 cold-course share from `model_summary.csv`;
- never selects CKG-RL rows.

- [ ] **Step 4: Draw the new layout**

Build a single-column figure with:

1. Figure title: `Why existing methods fall short`.
2. Panel (a): `PCGNN: structural mismatch`.
   Two horizontal zero-based bars for prerequisite gap and difficulty gap, with bootstrap intervals, direct values, diagonal hatching, and `lower is better`.
3. Panel (b): `CGRC: weak cold-course ranking`.
   A dotted-hatch course-level NDCG@10 histogram with a shaded region and marker at 0.10, `46% <= 0.10`, and `Top-10 cold share: 24.6%`.
4. Full-width strip: `CKG-RL response`.
   Left: `Course-knowledge sampling, rewards, prerequisite supervision`.
   Right: `Content anchoring, cold-ID masking, learner simulation`.

Do not draw any CKG-RL score, improvement percentage, or CKG-RL-versus-baseline confidence interval.

- [ ] **Step 5: Update the script entry point**

Read `course_macro.csv`, the three CGRC per-course export files already used by `collect_exposure_data`, and `model_summary.csv`. Export stable PDF, SVG, PNG, and a compact diagnostic CSV recording the plotted evidence and provenance.

- [ ] **Step 6: Run the focused test**

Run:

    D:\anaconda3\envs\req_py312\python.exe -m pytest tests\test_method_motivation.py -q --basetemp=.pytest_tmp\figure1_diagnosis_green

Expected: all tests in the file pass.

### Task 3: Align the manuscript narrative

**Files:**
- Modify: `paper_aaai27/main.tex`
- Modify: `tests/test_method_motivation.py`

- [ ] **Step 1: Rewrite the Introduction motivation paragraph**

State that Figure 1 diagnoses complementary existing-method problems: PCGNN retains prerequisite and difficulty mismatch under zero interaction; CGRC leaves 46% of audited cold-course cases at NDCG@10 no greater than 0.10 and exposes only 24.6% cold courses in Top-10 lists; CKG-RL maps course-knowledge constraints to the first problem and cold-course representation learning to the second.

- [ ] **Step 2: Replace the Figure 1 caption**

Use the concise problem-to-response caption from the approved spec, followed by a short provenance sentence identifying held-out MOOCCube test diagnostics and directing effect verification to Figure 3.

- [ ] **Step 3: Align the RQ2 opening**

Make the first RQ2 paragraph explain the Figure 1 diagnosis. Keep the following Figure 3 paragraphs as the numerical comparison and trade-off audit.

- [ ] **Step 4: Update manuscript tests**

Assert that the manuscript contains `Why existing methods fall short`, the method caption contains `46\%` and a Figure 3 reference, and the method caption no longer contains the direct CKG-RL Recall@10/NDCG@10 comparison.

### Task 4: Regenerate and verify

**Files:**
- Regenerate: `paper_aaai27/figures/mooccube_method_motivation.{pdf,svg,png}`
- Regenerate: `paper_aaai27/figures/mooccube_method_motivation_existing_diagnostics.csv`
- Regenerate: `paper_aaai27/main.pdf`

- [ ] **Step 1: Generate Figure 1**

Run:

    D:\anaconda3\envs\req_py312\python.exe paper_aaai27\scripts\draw_method_motivation.py

- [ ] **Step 2: Inspect the color rendering**

Open the generated PNG and verify direct labels, hatch patterns, response mapping, and absence of overlaps.

- [ ] **Step 3: Inspect a grayscale rendering**

Convert the PNG to grayscale with the configured Python imaging runtime and inspect it. PCGNN diagonal hatching, CGRC dotted hatching, and the response strip border must remain distinct.

- [ ] **Step 4: Run the motivation regression suite**

Run:

    D:\anaconda3\envs\req_py312\python.exe -m pytest tests\test_method_motivation.py tests\test_p1_motivation_evidence_table.py tests\test_draw_p1_topk_motivation.py tests\test_p1_motivation_mechanisms.py tests\test_p1_topk_motivation_analysis.py tests\test_p1_checkpoint_export_entrypoints.py -q --basetemp=.pytest_tmp\figure1_diagnosis_final

Expected: all tests pass.

- [ ] **Step 5: Compile the main paper**

Run the LaTeX paper compilation wrapper with the latexmk recipe from `paper_aaai27`.

- [ ] **Step 6: Render and inspect the final PDF page**

Use `pdftoppm` to render the page containing Figure 1. Verify that the title, hatch patterns, direct labels, and caption remain legible at final single-column size.

- [ ] **Step 7: Check the LaTeX log**

Search `main.log` for undefined references, undefined citations, undefined control sequences, and overfull boxes. Expected: no matches.

## Execution Note

Execute inline in the existing workspace. Do not stage, commit, push, reset, or clean unrelated user changes.
