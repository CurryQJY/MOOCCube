# AAAI Efficiency Table Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the current efficiency analysis as an independent LaTeX table and a one-page PDF.

**Architecture:** Extract the efficiency table markup into pure render functions in the existing revision-table builder. Add a dedicated exporter that reuses the existing data loaders and compiles a generated standalone TeX wrapper with `latexmk`.

**Tech Stack:** Python, pandas, pytest, LaTeX, latexmk

---

### Task 1: Define The Renderer Contract

**Files:**
- Create: `tests/test_efficiency_table_export.py`
- Modify: `paper_aaai27/scripts/build_revision_tables.py`

- [ ] **Step 1: Write the failing fragment-renderer test**

Create an in-memory cost DataFrame, call `render_efficiency_table(cost)`, and assert that the result contains one `table*`, all three datasets, `--` for unavailable values, and no unrelated revision section.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_efficiency_table_export.py -q`

Expected: FAIL because `render_efficiency_table` does not exist.

- [ ] **Step 3: Implement the pure renderer**

Extract the current cost-table markup into `render_efficiency_table(cost)` and add `render_efficiency_standalone(cost)` for the PDF wrapper. Reuse the fragment renderer inside the existing combined LaTeX writer.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_efficiency_table_export.py -q`

Expected: PASS.

### Task 2: Add The Dedicated Export Command

**Files:**
- Create: `paper_aaai27/scripts/export_efficiency_table.py`
- Modify: `tests/test_efficiency_table_export.py`

- [ ] **Step 1: Write the failing export-path test**

Assert that `write_efficiency_tex_exports(cost, output_dir)` writes `efficiency_table_aaai.tex` and `efficiency_table_aaai_standalone.tex` with the expected table label.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/test_efficiency_table_export.py -q`

Expected: FAIL because the export writer does not exist.

- [ ] **Step 3: Implement the writer and command**

Add the writer to `build_revision_tables.py`. Add `export_efficiency_table.py` to load the current seed results, summarize cost, write both TeX outputs, and run `latexmk -pdf -interaction=nonstopmode -halt-on-error` on the standalone source.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_efficiency_table_export.py -q`

Expected: PASS.

### Task 3: Generate And Verify Artifacts

**Files:**
- Generate: `paper_aaai27/efficiency_table_aaai.tex`
- Generate: `paper_aaai27/efficiency_table_aaai_standalone.tex`
- Generate: `paper_aaai27/efficiency_table_aaai.pdf`

- [ ] **Step 1: Run the dedicated exporter**

Run: `python paper_aaai27/scripts/export_efficiency_table.py`

Expected: all three output paths are printed and the command exits with code 0.

- [ ] **Step 2: Verify tests and generated content**

Run: `python -m pytest tests/test_efficiency_table_export.py tests/test_significance_input_scope.py -q`

Expected: all tests pass.

- [ ] **Step 3: Verify the PDF**

Run: `pdfinfo paper_aaai27/efficiency_table_aaai.pdf`

Expected: `Pages: 1` and no LaTeX errors in `paper_aaai27/efficiency_table_aaai_standalone.log`.
