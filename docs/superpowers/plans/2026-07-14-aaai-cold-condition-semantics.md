# AAAI Cold-Condition Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the course input visually neutral and attach the snowflake only to explicit ID-masked or cold-course conditions.

**Architecture:** Remove cold markers from shared course paths, add one marker beside the panel-a `ID mask`, and update the global legend wording. Keep the two cold Top-K markers and all existing model paths unchanged.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Move the cold marker from shared course paths to the ID mask

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:672-678,1112-1124,1287-1387`

- [x] **Step 1: Run the failing snowflake-placement assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib.pyplot as plt

p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
calls=[]
m.snowflake_icon=lambda ax,x,y,**kwargs: calls.append((x,y,kwargs))

fig,ax=plt.subplots(); m.course_icon(ax,0,0,84,80,framed=False)
assert calls == []

calls.clear(); fig,ax=plt.subplots(); m.course_info_reward_panel(ax,0,0,310,120)
assert calls == []

calls.clear(); fig,ax=plt.subplots(); m.draw_left(ax)
assert len(calls) == 1
x,y,kwargs=calls[0]
assert (x,y) == (349,367)
assert abs(kwargs["r"]-4.8) < 1e-9
'@ | python -
```

Expected: failure because the generic course input and course-info reward still contain snowflakes, while the ID mask has none.

- [x] **Step 2: Apply the semantic marker change**

Delete the `snowflake_icon` call from `course_icon` and from `course_info_reward_panel`. Add the following immediately after the `ID mask` text:

```python
    snowflake_icon(
        ax,
        right_tower_cx + 41,
        tower_y + 79,
        r=4.8,
        color=COL["ice_dark"],
        lw=0.8,
        z=13,
    )
```

- [x] **Step 3: Re-run the snowflake-placement assertion**

Expected: exit code 0.

### Task 2: Update the global legend and preserve explicit cold results

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1250-1280,1545-1560`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [x] **Step 1: Run the failing legend assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
fig=m.draw_figure(); labels=[t.get_text() for t in fig.axes[0].texts]
assert "Cold / ID-Masked" in labels
assert "Cold Course" not in labels
'@ | python -
```

Expected: failure because the legend still says `Cold Course`.

- [x] **Step 2: Rename the legend entry**

Replace `Cold Course` with `Cold / ID-Masked`; retain the current snowflake and typography.

- [x] **Step 3: Verify explicit cold-row markers**

Monkeypatch `snowflake_icon`, render two cold and two non-cold `recommendation_row` calls, and assert exactly two snowflake calls.

- [x] **Step 4: Regenerate and inspect all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0; the neutral course input has no snowflake, the ID mask has one, and the legend wording fits.

- [x] **Step 5: Compile and inspect page 4**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
pdftoppm -f 4 -l 4 -singlefile -png -r 180 build_state_transition_symmetry_verify/main.pdf tmp_cold_semantics_page4
```

Expected: an 11-page PDF with zero fatal errors, zero overfull boxes, and no undefined references.

Implementation remains uncommitted so the user's ongoing figure-editing workspace is preserved.
