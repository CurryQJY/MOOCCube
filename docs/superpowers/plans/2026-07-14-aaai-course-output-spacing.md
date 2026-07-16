# AAAI Course Output Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the final course representation and its incoming arrow endpoint upward by 12 canvas units to increase clearance above `Learner Encoder`.

**Architecture:** Keep the existing Matplotlib drawing structure and change only the shared `q_vec_y` layout coordinate from 552 to 540. Because the arrow endpoint, vector, and label all derive from this coordinate, the three elements remain aligned while the arrow source stays attached to the fusion gate.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Raise the final course representation group

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1389-1392`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Run the failing layout-coordinate assertion**

```powershell
@'
import ast
from pathlib import Path
p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
draw_left = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "draw_left")
q_assign = next(
    n for n in ast.walk(draw_left)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "q_vec_y" for t in n.targets)
)
assert ast.literal_eval(q_assign.value) == 540
'@ | python -
```

Expected: FAIL because the current value is 552.

- [x] **Step 2: Apply the minimal coordinate change**

```python
    q_vec_y = 540
```

- [x] **Step 3: Re-run the layout-coordinate assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate all figure outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0.

- [x] **Step 5: Verify placement and compile the paper**

Visually confirm the course vector and label moved upward together, the vertical arrow remains connected, and the learner encoder is unchanged. Then run:

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the figure implementation in the current branch and workspace without creating an implementation commit.
