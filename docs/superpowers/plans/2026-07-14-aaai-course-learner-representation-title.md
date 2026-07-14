# AAAI Course and Learner Representation Title Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename panel (a) and its upper subsection so the figure explicitly covers both course and learner representation while preserving the approved visual hierarchy.

**Architecture:** Make two text-only substitutions in the deterministic Matplotlib figure source. Preserve all coordinates and styling, regenerate every export format, then visually inspect the standalone figure and the compiled AAAI manuscript page.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Update the approved figure labels

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1236-1248`

- [ ] **Step 1: Run a source assertion that fails before the edit**

```powershell
@'
from pathlib import Path
text = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py").read_text(encoding="utf-8")
assert '"Course and Learner Representation"' in text
assert '"Course Encoder"' in text
assert '"Cold-course Evidence Encoder"' not in text
assert '"Content-anchored Course Encoder"' not in text
'@ | python -
```

Expected: `AssertionError` because the source still contains the previous labels.

- [ ] **Step 2: Apply the minimal text-only implementation**

Replace:

```python
txt(ax, 243, 121, "Cold-course Evidence Encoder", size=15.0, weight="bold")
txt(ax, 217, 152, "Content-anchored Course Encoder", size=12.8, weight="bold", color=COL["blue"])
```

with:

```python
txt(ax, 243, 121, "Course and Learner Representation", size=15.0, weight="bold")
txt(ax, 217, 152, "Course Encoder", size=12.8, weight="bold", color=COL["blue"])
```

- [ ] **Step 3: Re-run the source assertion**

Run the Step 1 command again.

Expected: exit code `0` with all four assertions satisfied.

### Task 2: Regenerate and verify the publication artifacts

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [ ] **Step 1: Regenerate all figure formats**

```powershell
python figures/plot_ckg_rl_framework_topconf.py
```

Run from `paper_aaai27`.

Expected: four `saved:` lines for SVG, PDF, PNG, and TIFF.

- [ ] **Step 2: Inspect the standalone PNG**

Open `paper_aaai27/figures/ckg_rl_framework_topconf.png` and confirm the panel title is not clipped, `Course Encoder` and `Learner Encoder` share the same size and weight, and no figure geometry moved.

- [ ] **Step 3: Compile the AAAI manuscript in an isolated directory**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_model_figure_title_verify main.tex
```

Run from `paper_aaai27`.

Expected: exit code `0` and `build_model_figure_title_verify/main.pdf`.

- [ ] **Step 4: Render and inspect the model-figure page**

```powershell
pdftoppm -f 4 -l 4 -png -r 180 build_model_figure_title_verify/main.pdf build_model_figure_title_verify/main_page4
```

Expected: the figure is fully visible on page 4 with no clipping, overlap, or unreadable title text.
