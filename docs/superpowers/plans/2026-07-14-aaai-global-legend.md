# AAAI Framework Global Legend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compact panel c's ranking computation and add a visually independent legend that defines conventions used across the complete framework figure.

**Architecture:** Keep all panel-c horizontal coordinates unchanged and apply one shared `rank_dy = -55` offset to every ranking-computation element. Shorten the dashed ranking region to end at y=502, then draw a neutral 390-by-126 legend box at y=516 with four recurring figure conventions.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Lock the panel-c geometry with a failing assertion

**Files:**
- Test: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py`

- [ ] **Step 1: Run the pre-change geometry assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib.pyplot as plt

p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = spec_from_file_location("figmod", p)
m = module_from_spec(spec)
spec.loader.exec_module(m)
fig, ax = plt.subplots()
m.draw_right(ax)
labels = {t.get_text(): t.get_position() for t in ax.texts}

assert labels["Ranking Logit"] == (1549, 206)
assert labels["User Embedding"][1] == 247
assert labels["Course Embedding"][1] == 485
assert labels["Legend"] == (1549, 535)
assert labels["Cold Course"][1] == 560
assert labels["Embedding"][1] == 560
assert labels["Main Flow"][1] == 603
assert labels["Feedback / Auxiliary Flow"][1] == 603
'@ | python -
```

Expected: failure because the ranking content has not moved and `Legend` does not exist.

### Task 2: Shift the ranking computation and add the global legend

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1450-1540`

- [ ] **Step 1: Apply a shared ranking offset and shorten the dashed region**

Add `rank_dy = -55` near the panel-c geometry constants. Apply it to `dot_cy`, both embedding groups, both input-arrow y coordinates, the Top-K box, its title, and every recommendation row. Keep `Ranking Logit` at y=206. Change the dashed region to:

```python
box(ax, 1340 + dx, 176, 390, 326, "#f8f8f2", ec=COL["line"], lw=1.7, r=0.5, ls=(0, (7, 5)))
```

- [ ] **Step 2: Replace the unused old legend with the approved global legend**

```python
def draw_global_legend(ax, x=1354, y=516, w=390, h=126, z=25):
    box(ax, x, y, w, h, COL["paper"], ec=COL["panel"], lw=1.3, r=4, z=z)
    txt(ax, x + w / 2, y + 19, "Legend", size=10.0, weight="bold", z=z + 2)

    snowflake_icon(ax, x + 28, y + 44, r=5.2, color=COL["ice_dark"], lw=1.0, z=z + 2)
    txt(ax, x + 43, y + 44, "Cold Course", size=8.2, ha="left", color=COL["muted"], z=z + 2)
    vector(ax, x + 214, y + 39, w=42, h=10, colors=COURSE_EMB_COLORS, z=z + 2)
    txt(ax, x + 264, y + 44, "Embedding", size=8.2, ha="left", color=COL["muted"], z=z + 2)

    flow_arrow(ax, [(x + 18, y + 87), (x + 58, y + 87)], kind="data", z=z + 2, lw=1.35, ms=8.0)
    txt(ax, x + 67, y + 87, "Main Flow", size=8.2, ha="left", color=COL["muted"], z=z + 2)
    flow_arrow(ax, [(x + 207, y + 87), (x + 247, y + 87)], kind="aux", z=z + 2, lw=1.15, ms=8.0)
    txt(ax, x + 256, y + 87, "Feedback / Auxiliary Flow", size=8.0, ha="left", color=COL["muted"], z=z + 2)
```

Call `draw_global_legend(ax)` once from `draw_right` after the Top-K rows.

- [ ] **Step 3: Re-run the geometry assertion**

Run Task 1 Step 1 again.

Expected: exit code 0; all labels occupy the approved y coordinates.

### Task 3: Regenerate and verify publication outputs

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [ ] **Step 1: Regenerate the figure**

Run from `paper_aaai27`:

```powershell
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0.

- [ ] **Step 2: Inspect the standalone PNG**

Confirm that the ranking computation remains connected, the dashed region ends above the legend, the legend is visually independent, and all four entries remain readable.

- [ ] **Step 3: Compile and inspect the manuscript**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
pdftoppm -f 4 -l 4 -singlefile -png -r 180 build_state_transition_symmetry_verify/main.pdf tmp_global_legend_page4
```

Expected: an 11-page PDF, zero fatal errors, zero overfull boxes, and a readable non-overlapping legend on page 4.
