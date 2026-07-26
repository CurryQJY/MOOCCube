# AAAI Update-Arrow Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align only the red state-update arrow with the black main-flow arrow at absolute y=272.

**Architecture:** Introduce a local `update_y = y + 82` coordinate inside `draw_state_transition_function` and use it only for the red `update` arrow. Leave every state representation and all other arrows tied to their existing coordinates.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Move only the update arrow

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1034-1043`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Run the failing alignment assertion**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = spec_from_file_location("framework", p)
m = module_from_spec(spec)
spec.loader.exec_module(m)

arrows = []
states = []
m.flow_arrow = lambda ax, pts, **kwargs: arrows.append((pts, kwargs))
m.tilted_vector = lambda ax, cx, cy, **kwargs: states.append((cx, cy, kwargs))
fig, ax = plt.subplots()
m.draw_state_transition_function(ax, 930, 190, 220, 164)
update = next(pts for pts, kwargs in arrows if kwargs.get("kind") == "update")
assert update[0][1] == update[1][1] == 272
assert len(states) == 2
assert all(cy == 264 for _, cy, _ in states)
'@ | python -
```

Expected: FAIL because the red update arrow is currently at y=264.

- [x] **Step 2: Apply the isolated y-coordinate change**

```python
    update_y = y + 82
```

Use `update_y` for both red-arrow endpoints and do not change other coordinates.

- [x] **Step 3: Re-run the alignment assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm the red and right black arrows share one horizontal centerline while both state vectors remain fixed.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the implementation in the current branch and workspace without creating an implementation commit.
