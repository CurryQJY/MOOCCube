# AAAI Learner Process and Title Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move only the learner representation-initialization process downward by 8 units while keeping the `Learner Encoder` heading and final course representation fixed.

**Architecture:** Decouple the heading from the learner process by introducing `learner_title_y = 554`. Change the process anchor to `learner_cy = 592`; all process graphics and lower labels remain derived from this shared anchor and therefore move together.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Separate the learner title and process anchors

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
assert ("Learner Encoder", (213, 554)) in positions
assert ("$\\mathbf{e}_u$", (161, 622)) in positions
assert ("$\\mathbf{z}_u$", (339, 622)) in positions
assert ("$\\mathbf{e}_c$", (268.0, 519)) in positions
'@ | python -
```

Expected: FAIL because the lower learner labels are currently at y=614.

- [x] **Step 2: Introduce the independent anchors**

```python
    learner_title_y = 554
    learner_cy = 592
    txt(ax, 213, learner_title_y, "Learner Encoder", size=role_scale(12.8, "component"), weight="bold", color=COL["orange"])
```

Leave every process element tied to `learner_cy` and keep `q_vec_y = 510` unchanged.

- [x] **Step 3: Re-run the rendered-position assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm the heading and course output remain fixed, the complete learner process moves downward together, and bottom clearance remains comfortable.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, zero undefined references, and more than 35 rendered pixels of panel-a bottom clearance.

Keep the implementation in the current branch and workspace without creating an implementation commit.
