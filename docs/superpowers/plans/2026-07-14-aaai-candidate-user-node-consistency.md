# AAAI Candidate-User Node Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make $u_1$, $u_j$, and $u_N$ identical in size and shape while retaining a color-only selection highlight for $u_j$.

**Architecture:** Modify only the `learner_action_set` rendering loop. Use one shared radius and line width for every circle, then choose fill and text colors from the existing `selected` flag.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Standardize candidate-user geometry and color semantics

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1130-1148`

- [ ] **Step 1: Run the failing geometry assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches

p = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = spec_from_file_location("figmod", p)
m = module_from_spec(spec)
spec.loader.exec_module(m)
fig, ax = plt.subplots()
m.learner_action_set(ax, 0, 0)
circles = [p for p in ax.patches if isinstance(p, patches.Circle)]
labels = {t.get_text(): t for t in ax.texts}

assert [c.get_radius() for c in circles] == [7.6, 7.6, 7.6]
assert [c.get_linewidth() for c in circles] == [1.0, 1.0, 1.0]
assert circles[0].get_facecolor() == circles[2].get_facecolor()
assert circles[1].get_facecolor() != circles[0].get_facecolor()
assert labels[r"$u_1$"].get_color() == m.COL["orange"]
assert labels[r"$u_j$"].get_color() == m.COL["paper"]
assert labels[r"$u_N$"].get_color() == m.COL["orange"]
'@ | python -
```

Expected: `AssertionError` because the current radii are `[7.2, 8.2, 7.2]` and the selected node is encoded by size and border width.

- [ ] **Step 2: Apply the minimal rendering change**

Replace the conditional radius and line width with shared geometry, and derive colors from `selected`:

```python
        node_fill = COL["orange"] if selected else COL["orange_soft"]
        node_text = COL["paper"] if selected else COL["orange"]
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                7.6,
                facecolor=node_fill,
                edgecolor=COL["orange"],
                linewidth=1.0,
                zorder=z + 3,
            )
        )
        ax.text(cx, cy + 0.2, label, fontsize=6.4, fontweight="bold", color=node_text, ha="center", va="center", zorder=z + 4)
```

- [ ] **Step 3: Re-run the geometry assertion**

Run Task 1 Step 1 again.

Expected: exit code 0; all geometry assertions and color assertions pass.

### Task 2: Regenerate and verify publication outputs

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [ ] **Step 1: Regenerate the figure**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0.

- [ ] **Step 2: Inspect the standalone PNG**

Confirm that all three candidate-user circles have the same diameter and outline, only $u_j$ uses the dark selected fill, and no label is clipped.

- [ ] **Step 3: Compile and inspect page 4**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
pdftoppm -f 4 -l 4 -singlefile -png -r 180 build_state_transition_symmetry_verify/main.pdf tmp_candidate_users_page4
```

Expected: an 11-page PDF, zero fatal errors, zero overfull boxes, and three equal candidate-user nodes visible in panel b.
