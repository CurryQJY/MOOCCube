# AAAI Exploration Pipeline Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the three exploration-set cards into a readable arrow-connected pipeline with non-overlapping, consistently sized labels.

**Architecture:** Replace the existing unequal card geometry with three explicitly positioned cards centered inside the exploration container. Use two standard micro-flow arrows between card borders and one shared font-size expression for every stage label.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Rebuild the exploration-stage row

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1425-1430`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Run the failing pipeline assertion**

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

calls = []
original = m.flow_arrow
m.flow_arrow = lambda ax, pts, **kwargs: calls.append((pts, kwargs))
fig, ax = plt.subplots()
m.draw_middle(ax)
assert ([(714, 269), (726, 269)], {"kind": "micro", "z": 12}) in calls
assert ([(804, 269), (816, 269)], {"kind": "micro", "z": 12}) in calls

m.flow_arrow = original
fig = m.draw_figure()
labels = {t.get_text(): t for t in fig.axes[0].texts}
assert labels["Retrieve $M$"].get_position() == (672, 269)
assert labels["Sample $N$"].get_position() == (765, 269)
assert labels["Fit Rerank"].get_position() == (858, 269)
assert len({labels[s].get_fontsize() for s in ("Retrieve $M$", "Sample $N$", "Fit Rerank")}) == 1
'@ | python -
```

Expected: FAIL because the new labels and inter-card arrows do not yet exist.

- [x] **Step 2: Apply the new card row**

Use x/width pairs `(632,80)`, `(728,74)`, and `(818,80)` at y=252 with height 34. Center the labels at x=672, 765, and 858, use `role_scale(9.0, "symbol")` for all three, and add:

```python
    flow_arrow(ax, [(714, 269), (726, 269)], kind="micro", z=12)
    flow_arrow(ax, [(804, 269), (816, 269)], kind="micro", z=12)
```

- [x] **Step 3: Re-run the pipeline assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm each label stays within its box, the arrows remain between borders, and the row is visually centered.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the implementation in the current branch and workspace without creating an implementation commit.
