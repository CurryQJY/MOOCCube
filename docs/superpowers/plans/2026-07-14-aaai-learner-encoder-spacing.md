# AAAI Learner Encoder Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete learner-encoder group upward by 10 canvas units while preserving its internal alignment and the final course representation position.

**Architecture:** Use `learner_cy = 614` as the single vertical anchor for the learner row. Express the heading and lower labels as offsets from that anchor so all learner elements stay synchronized during this and future layout adjustments.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Raise the learner-encoder group

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1394-1404`
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
assert ("Learner Encoder", (213, 584)) in positions
assert ("$\\mathbf{e}_u$", (161, 644)) in positions
assert ("$\\mathbf{z}_u$", (339, 644)) in positions
assert ("$\\mathbf{e}_c$", (268.0, 549)) in positions
'@ | python -
```

Expected: FAIL because the learner heading and labels are currently at y=594 and y=654.

- [x] **Step 2: Apply the shared learner anchor**

```python
    learner_cy = 614
    txt(ax, 213, learner_cy - 30, "Learner Encoder", size=role_scale(12.8, "component"), weight="bold", color=COL["orange"])
```

Use `learner_cy + 30` for both lower labels and leave all other learner-element expressions unchanged.

- [x] **Step 3: Re-run the rendered-position assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all figure outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm the learner group moved upward together, has more bottom clearance, and does not collide with the final course representation.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the implementation in the current branch and workspace without creating an implementation commit.
