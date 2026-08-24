# AAAI Course-Reward Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enlarge all icons and internal labels in the `Course-info Reward` panel while preserving the existing panel and chip layout.

**Architecture:** Change only size and line-width arguments inside `course_info_reward_panel`. Keep every anchor and box dimension fixed, then validate the rendered label and icon bounds against the existing geometry.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Increase course-reward visual scale

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1113-1137`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Run the failing icon-size assertion**

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

books = []
terms = []
m.book_open_icon = lambda ax, cx, cy, **kwargs: books.append(kwargs)
m.reward_term_icon = lambda ax, cx, cy, kind, color, accent, **kwargs: terms.append((kind, kwargs))
fig, ax = plt.subplots()
m.course_info_reward_panel(ax, 0, 0, 310, 120)
assert books[0]["size"] == m.role_scale(40, "icon")
assert books[0]["lw"] == 1.3
assert len(terms) == 4
assert all(kwargs["size"] == m.role_scale(20, "icon") for _, kwargs in terms)
'@ | python -
```

Expected: FAIL because the current book and reward-term sizes are 30 and 15 before role scaling.

- [x] **Step 2: Apply the new icon and text sizes**

Use `role_scale(40, "icon")` and line width 1.3 for the book, `role_scale(20, "icon")` for every reward-term icon, `role_scale(10.0, "component")` for the panel heading, `role_scale(9.0, "symbol")` for the course symbol, and `role_scale(8.0, "symbol")` for each chip label.

- [x] **Step 3: Re-run the icon-size assertion**

Expected: exit code 0.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines. Confirm every icon is visibly larger and all enlarged labels remain inside their boxes.

- [x] **Step 5: Compile and verify the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the implementation in the current branch and workspace without creating an implementation commit.
