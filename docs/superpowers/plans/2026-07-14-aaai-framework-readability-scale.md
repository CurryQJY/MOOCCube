# AAAI Framework Readability Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enlarge titles and semantic symbols throughout the detailed three-panel CKG-RL figure as far as the existing boxes allow without collisions or boundary overflow.

**Architecture:** Add explicit role-based scale constants and apply them only at active detailed-figure call sites. Keep `TEXT_SCALE`, the canvas, outer panels, explanatory prose, and unused overview renderers unchanged; cap the two tight titles (`Course and Learner Representation` and `Exploration Set Construction`) below the general role multiplier.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk, Poppler

---

### Task 1: Add role scales and enlarge panel-level hierarchy

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:30-90,1277-1284`

- [ ] **Step 1: Run a failing panel-title assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
fig=m.draw_figure(); texts={t.get_text():t for t in fig.axes[0].texts}
assert m.ROLE_SCALE == {"panel":1.08,"component":1.15,"symbol":1.12,"icon":1.10,"legend":1.20,"arrow":1.08}
assert abs(texts["Course-knowledge Guided Simulation"].get_fontsize()-15.552) < 1e-6
assert abs(texts["Strict Item-Cold Ranking"].get_fontsize()-14.58) < 1e-6
assert abs(texts["Course and Learner Representation"].get_fontsize()-11.25) < 1e-6
'@ | python -
```

Expected: failure because `ROLE_SCALE` does not exist and the panel titles still use their old sizes.

- [ ] **Step 2: Add the role constants**

```python
ROLE_SCALE = {
    "panel": 1.08,
    "component": 1.15,
    "symbol": 1.12,
    "icon": 1.10,
    "legend": 1.20,
    "arrow": 1.08,
}


def role_scale(value, role):
    return value * ROLE_SCALE[role]
```

- [ ] **Step 3: Apply panel-title scaling**

Scale a/b/c labels and the b/c titles with `role_scale(..., "panel")`. Keep the long panel-a title at `12.5`, because its measured bounding box already occupies x=82.3–393.7 inside the available x=82–394 range. Scale the panel-c subtitle by `1.06` rather than the title multiplier.

- [ ] **Step 4: Re-run the panel-title assertion**

Expected: exit code 0.

### Task 2: Enlarge panel-a component titles, symbols, and icons

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:620-725,1287-1387`

- [ ] **Step 1: Run a failing panel-a assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import matplotlib.pyplot as plt
p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
fig,ax=plt.subplots(); m.draw_left(ax); texts={t.get_text():t for t in ax.texts}
for label, expected in {
 "Course Encoder":12.8*1.15*.9,
 "Content Emb.":10.8*1.15*.9,
 "Behavior Emb.":10.8*1.15*.9,
 "Learner Encoder":12.8*1.15*.9,
 r"$\mathbf{e}_c$":12.0*1.12*.9,
 r"$\mathbf{e}_u$":11.0*1.12*.9,
 r"$\mathbf{z}_u$":11.5*1.12*.9,
}.items():
 assert abs(texts[label].get_fontsize()-expected) < 1e-6, (label,texts[label].get_fontsize(),expected)
'@ | python -
```

Expected: failure on `Course Encoder`.

- [ ] **Step 2: Scale panel-a titles and symbols**

Apply `component` scaling to `Course Encoder`, `Content Emb.`, `Behavior Emb.`, and `Learner Encoder`. Apply `symbol` scaling to the content/behavior variables, `ID mask`, MLP labels, gate symbols, and the final course/learner embeddings.

- [ ] **Step 3: Scale panel-a pictograms within their existing regions**

Use `icon` scaling for the course snowflake/book, learner icon, and learner embedding-bar heights. Re-center any height increase around the original bar center; do not change branch x coordinates or card dimensions.

- [ ] **Step 4: Re-run the panel-a assertion**

Expected: exit code 0.

### Task 3: Enlarge panel-b titles, state/action symbols, and reward icons

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:422-508,873-1180,1388-1452`

- [ ] **Step 1: Run a failing panel-b assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
fig=m.draw_figure(); texts={t.get_text():t for t in fig.axes[0].texts}
expected={
 "Exploration Set Construction":12.2*1.05*.9,
 "Actor-Critic Agent":12.4*1.15*.9,
 "State Transition Function":10.5*1.15*.9,
 "Reward Function":13.0*1.15*.9,
 "Emb. Alignment Reward":9.2*1.15,
 "Course-info Reward":9.2*1.15,
}
for label,size in expected.items(): assert abs(texts[label].get_fontsize()-size)<1e-6,(label,texts[label].get_fontsize(),size)
'@ | python -
```

Expected: failure on the exploration title.

- [ ] **Step 2: Scale panel-b titles with the measured caps**

Use `1.05` for `Exploration Set Construction` (its measured horizontal scale cap is 1.07). Use `component` for `Actor-Critic Agent`, `State Transition Function`, `Reward Function`, `Emb. Alignment Reward`, and `Course-info Reward`.

- [ ] **Step 3: Scale panel-b symbols and recurrent icons**

Apply `symbol` scaling to state labels, time labels, exploration variables, action/reward labels, transition $u_j$, transition states, candidate-user labels, course-info symbols, and reward-point variables. Increase all three candidate-user radii equally from `7.6` to `8.0`; keep their geometry identical and preserve color-only selection. Use `icon` scaling for clocks, brain icon, selected-action badge radius, transition action circle, course-info book/snowflake, and reward-term icons.

- [ ] **Step 4: Re-run the panel-b assertion and candidate-node equality assertion**

Expected: exit code 0; candidate-user radii remain equal.

### Task 4: Enlarge panel-c ranking content and the global legend

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1235-1276,1453-1547`

- [ ] **Step 1: Run a failing panel-c assertion**

```powershell
@'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
p=Path("paper_aaai27/figures/plot_ckg_rl_framework_topconf.py")
spec=spec_from_file_location("m",p); m=module_from_spec(spec); spec.loader.exec_module(m)
fig=m.draw_figure(); texts={t.get_text():t for t in fig.axes[0].texts}
expected={
 "Ranking Logit":13.2*1.15*.9,
 "User Embedding":10.8*1.15*.9,
 "Course Embedding":10.8*1.15*.9,
 "Legend":10.0*1.20*.9,
 "Cold Course":8.2*1.20*.9,
 "Embedding":8.2*1.20*.9,
 "Main Flow":8.2*1.20*.9,
 "Feedback /\nAuxiliary Flow":8.0*1.20*.9,
}
for label,size in expected.items(): assert abs(texts[label].get_fontsize()-size)<1e-6,(label,texts[label].get_fontsize(),size)
'@ | python -
```

Expected: failure on `Ranking Logit`.

- [ ] **Step 2: Scale ranking titles, variables, and icons**

Apply `component` scaling to `Ranking Logit`, `User Embedding`, `Course Embedding`, and `Top-K`. Apply `symbol` scaling to $z_u$, $z_{c,T}$, the dot-product glyph, ranks, course IDs, and row book/snowflake glyphs. Apply `icon` scaling to the user/book icons and dot-product circle, keeping arrow endpoints attached.

- [ ] **Step 3: Scale and re-space the legend**

Use `legend` scaling for the title and all four labels. Increase the snowflake radius from `5.2` to `6.0`, the embedding swatch from `42x10` to `48x12`, and both arrow examples from 40 to 48 units with mutation scale `9.0`. Shift label starts right by the extra glyph width and keep the two-line feedback label inside x=1744.

- [ ] **Step 4: Re-run the panel-c assertion and legend-bound assertion**

Expected: exit code 0; all legend text bounding boxes remain inside x=1354–1744 and y=516–642.

### Task 5: Increase arrow visibility and verify the complete publication figure

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:76-87`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`
- Verify: `paper_aaai27/main.tex`

- [ ] **Step 1: Increase standard arrowheads and strokes**

Multiply every `ARROW_STYLES` mutation scale by `ROLE_SCALE["arrow"]` and each standard line width by `1.04`. Do not enlarge the already prominent red inter-panel arrows.

- [ ] **Step 2: Regenerate all four figure formats**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines and exit code 0.

- [ ] **Step 3: Inspect the standalone figure and dense-region crops**

Inspect panel a's branch cards, panel b's candidate row, state transition and reward cards, panel c's ranking content, and the legend. Confirm that enlarged elements remain inside their semantic parents and do not cover arrows or neighboring labels.

- [ ] **Step 4: Compile and inspect page 4**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
pdftoppm -f 4 -l 4 -singlefile -png -r 180 build_state_transition_symmetry_verify/main.pdf tmp_readability_page4
```

Expected: an 11-page PDF with zero fatal errors, zero overfull boxes, and no undefined references.

Implementation remains uncommitted so the user's ongoing figure-editing workspace is preserved.
