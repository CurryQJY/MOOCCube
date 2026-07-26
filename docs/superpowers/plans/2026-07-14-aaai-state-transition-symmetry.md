# AAAI State-Transition Upper-Flow Rebalancing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `u_j` and extend the right brown curved arrow so the upper action flow fills the available space evenly without moving the centered representations or red arrow.

**Architecture:** Treat the moved `h_t` head and right representation center as fixed outer anchors. Set `action_cx` to their midpoint, place `u_j` there, and use the existing 13-unit node clearance on both sides. This produces equal brown-arrow spans while preserving the independently centered lower red-arrow layout.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Recenter the upper action flow between its state anchors

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:975-1060`

- [ ] **Step 1: Run the geometry assertion before editing**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import math
import matplotlib.pyplot as plt

p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = spec_from_file_location("figmod", p)
m = module_from_spec(spec)
spec.loader.exec_module(m)

fig, ax = plt.subplots()
vectors, curves, updates = [], [], []
m.tilted_vector = lambda ax, cx, cy, **kwargs: vectors.append((cx, cy, kwargs))
m.curved_arrow = lambda ax, start, end, **kwargs: curves.append((start, end, kwargs))
m.flow_arrow = lambda ax, pts, **kwargs: updates.append((pts, kwargs))
x, y, w, h = 930, 190, 220, 164
m.draw_state_transition_function(ax, x, y, w, h)
center = x + w / 2
left_start, left_end, _ = curves[0]
right_start, right_end, _ = curves[1]
labels = {t.get_text(): t.get_position() for t in ax.texts}
action_x = labels[r"$u_j$"][0]
expected_action_x = (left_start[0] + vectors[1][0]) / 2

assert abs(action_x - expected_action_x) < 1e-9
assert right_end == (vectors[1][0], left_start[1])
assert left_end == (action_x - 13, y + 50)
assert right_start == (action_x + 13, y + 50)
assert abs((left_end[0] - left_start[0]) - (right_end[0] - right_start[0])) < 1e-9

theta = math.radians(-48)
x_local = 63 / 2
y_offset = (y + 52) - vectors[0][1]
y_local = (y_offset - x_local * math.sin(theta)) / math.cos(theta)
head_right_x = vectors[0][0] + x_local * math.cos(theta) - y_local * math.sin(theta)
assert abs(head_right_x - left_start[0]) < 1e-9

update_start, update_end = updates[0][0]
assert abs((vectors[0][0] + vectors[1][0]) / 2 - center) < 1e-9
assert abs((update_start[0] + update_end[0]) / 2 - center) < 1e-9
'@ | python -
```

Expected: `AssertionError` because the current `u_j` remains at x=1040 instead of the anchor midpoint x=1051.321.

- [ ] **Step 2: Apply the minimal implementation**

Derive the upper action-flow center and right endpoint from the fixed state anchors:

```python
right_arrow_end_x = right_c[0]
action_cx = (left_arrow_start_x + right_arrow_end_x) / 2
guide_c = (action_cx, y + 50)
```

Keep the left curve from `(left_arrow_start_x, arrow_y)` to `(guide_c.x - 13, y + 50)` and the right curve from `(guide_c.x + 13, y + 50)` to `(right_arrow_end_x, arrow_y)`.

- [ ] **Step 3: Re-run the geometry assertion**

Run the Step 1 command again.

Expected: exit code `0`; `u_j` lies at the anchor midpoint, the two brown arrows have equal spans, and the lower centered layout is unchanged.

### Task 2: Regenerate and verify publication outputs

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [ ] **Step 1: Run the figure generator**

```powershell
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines.

- [ ] **Step 2: Inspect the standalone PNG**

Confirm the upper brown flow fills the right-side whitespace, both brown arrows are equal, and the centered red-arrow/state layout is unchanged.

- [ ] **Step 3: Compile the AAAI paper**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: exit code `0`; record the current manuscript page count.

- [ ] **Step 4: Render and inspect page 4**

```powershell
pdftoppm -f 4 -l 4 -singlefile -png -r 180 build_state_transition_symmetry_verify/main.pdf ../tmp/aaai_state_transition_page4
```

Expected: the rebalanced upper action flow remains legible and unclipped in the manuscript.
