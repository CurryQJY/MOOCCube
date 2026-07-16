# AAAI Lower-Stack Whitespace Balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the final course representation and complete learner encoder upward together so panel-a bottom whitespace visually matches the whitespace above `Course Encoder`.

**Architecture:** Change only the two existing vertical anchors: `q_vec_y` from 540 to 510 and `learner_cy` from 614 to 584. All dependent arrows, vectors, headings, icons, and labels already derive from these anchors, so their relative geometry remains unchanged.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Balance the panel-a upper and lower whitespace

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1389-1404`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Run the failing rendered-position assertion**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = spec_from_file_location("framework", p)
m = module_from_spec(spec)
spec.loader.exec_module(m)
fig = m.draw_figure()
positions = {(t.get_text(), t.get_position()) for t in fig.axes[0].texts}
assert ("$\\mathbf{e}_c$", (268.0, 519)) in positions
assert ("Learner Encoder", (213, 554)) in positions
assert ("$\\mathbf{e}_u$", (161, 614)) in positions
assert ("$\\mathbf{z}_u$", (339, 614)) in positions
'@ | python -
```

Expected: FAIL because these elements are currently 30 units lower.

- [x] **Step 2: Apply the two anchor changes**

```python
    q_vec_y = 510
```

```python
    learner_cy = 584
```

Do not change any dependent expressions or horizontal coordinates.

- [x] **Step 3: Re-run the rendered-position assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm the lower stack moved as one unit, the course-output arrow remains attached, and top/bottom panel whitespace is visually balanced.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references. The rendered top and bottom whitespace difference should be below 4 pixels.

Keep the implementation in the current branch and workspace without creating an implementation commit.
