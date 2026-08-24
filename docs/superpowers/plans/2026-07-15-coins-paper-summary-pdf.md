# COINS Paper Summary PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a polished 6–8 page Chinese A4 PDF summarizing the COINS paper and verify its content and rendering.

**Architecture:** A focused ReportLab generator will define reusable styles, flowables, tables, a compact method diagram, and page decorations. The generated PDF will be validated with pypdf text/page checks and PyMuPDF full-page rendering.

**Tech Stack:** Python 3, ReportLab 4.4.5, pypdf, PyMuPDF, Microsoft YaHei/Noto Sans SC fonts.

---

### Task 1: Implement the PDF generator

**Files:**
- Create: `tools/generate_coins_summary_pdf.py`
- Create: `output/pdf/COINS_论文精读摘要.pdf`

- [ ] **Step 1:** Define A4 document margins, Chinese fonts, heading/body/table styles, page header/footer, and helper flowables.
- [ ] **Step 2:** Add the title page, one-page executive summary, research problem, COINS method, equations, experimental evidence, critical review, and course cold-start inspiration.
- [ ] **Step 3:** Add the simplified RQ-OPQ/Adaptive Gate diagram and the offline, ablation, and online-result tables.
- [ ] **Step 4:** Run `D:\anaconda3\envs\zw\python.exe tools\generate_coins_summary_pdf.py` and expect exit code 0 with the final PDF path and page count.

### Task 2: Verify content and visual rendering

**Files:**
- Create: `tmp/pdfs/coins_summary_page_*.png`

- [ ] **Step 1:** Use pypdf to confirm the PDF opens, has 6–8 pages, and contains expected section text and key metrics.
- [ ] **Step 2:** Use PyMuPDF to render every page to PNG at 1.5x scale.
- [ ] **Step 3:** Inspect the rendered pages for clipped Chinese text, table overflow, overlapping flowables, broken symbols, and inconsistent headers/footers.
- [ ] **Step 4:** If defects are found, patch the generator, regenerate the PDF, and repeat both checks.

### Task 3: Final artifact check

**Files:**
- Verify: `output/pdf/COINS_论文精读摘要.pdf`

- [ ] **Step 1:** Confirm the final file exists and report its byte size and SHA-256 hash.
- [ ] **Step 2:** Confirm the source DOI, Table 1–3 values, course recommendation section, and limitation section are present in extracted text.
- [ ] **Step 3:** Deliver the clickable PDF path to the user.
