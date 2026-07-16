# AAAI Reward Icon Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense prerequisite and difficulty glyphs with a directed dependency chain and a sparse outline gauge that remain legible at the figure's production scale.

**Architecture:** Keep `reward_term_icon` as the single reusable icon renderer and modify only its `prereq` and `difficulty` branches. Preserve all chip geometry and labels, then verify artist structure, final-pixel readability, and AAAI paper compilation.

**Tech Stack:** Python, Matplotlib, PowerShell, LaTeX/latexmk

---

### Task 1: Lock the production-scale icon requirements

**Files:**
- Test: one-off Python assertion against `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py`

- [x] **Step 1: Run the failing prerequisite assertion**

Run from the repository root:

```powershell
@'
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

path = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = importlib.util.spec_from_file_location("framework_figure", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fig, ax = plt.subplots()
module.reward_term_icon(ax, 0, 0, "prereq", "#a9572b", "#f6dfd1", size=20)
circles = [artist for artist in ax.patches if isinstance(artist, patches.Circle)]
arrows = [artist for artist in ax.patches if isinstance(artist, patches.FancyArrowPatch)]
blocks = [artist for artist in ax.patches if isinstance(artist, (patches.Rectangle, patches.Polygon))]
assert len(circles) == 3, len(circles)
assert len(arrows) == 2, len(arrows)
assert len(blocks) == 0, len(blocks)
print("prereq dependency-chain structure: PASS")
'@ | python -
```

Expected: FAIL because the current prerequisite icon has no circular dependency nodes and uses a staircase polygon.

- [x] **Step 2: Run the failing gauge-sparsity assertion**

Run from the repository root:

```powershell
@'
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

path = Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec = importlib.util.spec_from_file_location("framework_figure", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fig, ax = plt.subplots()
module.reward_term_icon(ax, 0, 0, "difficulty", "#8c6a15", "#f7e8ad", size=20)
arcs = [artist for artist in ax.patches if isinstance(artist, patches.Arc)]
hubs = [artist for artist in ax.patches if isinstance(artist, patches.Circle)]
assert len(arcs) == 1
assert arcs[0].get_linewidth() <= 1.6, arcs[0].get_linewidth()
assert len(ax.lines) == 4, len(ax.lines)
assert len(hubs) == 1
assert hubs[0].get_radius() <= 1.3, hubs[0].get_radius()
print("difficulty sparse-gauge structure: PASS")
'@ | python -
```

Expected: FAIL because the current arc is 2.1 points wide and its production-scale hub radius is 2.0.

### Task 2: Draw the directed prerequisite chain

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:865-899`

- [x] **Step 1: Replace the staircase with three nodes and two arrows**

Use three positions from lower-left to upper-right. Draw the arrows first so their ends remain underneath the node outlines, then draw three pale-filled circles:

```python
nodes = [
    (cx - 10 * scale, cy + 7 * scale),
    (cx, cy),
    (cx + 10 * scale, cy - 7 * scale),
]
node_r = 2.7 * scale
for (x1, y1), (x2, y2) in zip(nodes[:-1], nodes[1:]):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    start = (x1 + ux * (node_r + 0.9 * scale), y1 + uy * (node_r + 0.9 * scale))
    end = (x2 - ux * (node_r + 1.1 * scale), y2 - uy * (node_r + 1.1 * scale))
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.0 * scale,
            facecolor=color,
            edgecolor=color,
            lw=1.3,
            shrinkA=0,
            shrinkB=0,
            zorder=z + 1,
        )
    )
for px, py in nodes:
    ax.add_patch(
        patches.Circle(
            (px, py),
            node_r,
            facecolor=accent,
            edgecolor=color,
            lw=1.15,
            zorder=z + 2,
        )
    )
```

- [x] **Step 2: Re-run the prerequisite assertion**

Run the Task 1 Step 1 command.

Expected: `prereq dependency-chain structure: PASS`.

### Task 3: Draw the sparse difficulty gauge

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:900-928`

- [x] **Step 1: Thin and separate the gauge components**

Use a thin open arc, three short ticks, a high-position needle, and a small hub:

```python
gauge_cy = cy + 4 * scale
ax.add_patch(
    patches.Arc(
        (cx, gauge_cy),
        27 * scale,
        21 * scale,
        theta1=205,
        theta2=335,
        color=color,
        lw=1.5,
        zorder=z,
    )
)
for angle in (220, 270, 320):
    radians = math.radians(angle)
    ax.plot(
        [cx + 9.2 * scale * math.cos(radians), cx + 11.8 * scale * math.cos(radians)],
        [gauge_cy + 7.2 * scale * math.sin(radians), gauge_cy + 9.6 * scale * math.sin(radians)],
        color=color,
        lw=1.05,
        solid_capstyle="round",
        zorder=z + 1,
    )
needle_angle = math.radians(315)
ax.plot(
    [cx, cx + 9.5 * scale * math.cos(needle_angle)],
    [gauge_cy, gauge_cy + 7.8 * scale * math.sin(needle_angle)],
    color=color,
    lw=1.45,
    solid_capstyle="round",
    zorder=z + 2,
)
ax.add_patch(
    patches.Circle(
        (cx, gauge_cy),
        1.8 * scale,
        facecolor=accent,
        edgecolor=color,
        lw=0.8,
        zorder=z + 3,
    )
)
```

- [x] **Step 2: Re-run the gauge assertion**

Run the Task 1 Step 2 command.

Expected: `difficulty sparse-gauge structure: PASS`.

### Task 4: Regenerate and verify the paper figure

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Regenerate all figure formats**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0.

- [x] **Step 2: Inspect the production output**

Crop the Course-info Reward region from the 4191-by-1761 PNG and inspect it at original resolution. Confirm that all three prerequisite nodes are separated, both arrowheads are visible, the gauge outline is not filled, and the needle does not merge with the ticks or hub.

- [x] **Step 3: Compile the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

- [x] **Step 4: Keep the implementation uncommitted**

Do not stage, commit, merge, or push the figure implementation. Preserve all unrelated working-tree changes.
